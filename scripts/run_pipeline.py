from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from typing import List
from app.pipeline.run import run_pipeline as run_local_pipeline, RunConfig
try:
    import pretty_midi
except Exception:
    pretty_midi = None
from app.models.note_event import NoteEvent
from app.transcription.guitar_mapper import map_notes_to_guitar

def _run_klangio(audio_path: str, out_dir: Path, outputs: List[str], derive_notes: bool) -> int:
    from app.cloud.klangio_api import client_from_env
    client = client_from_env()
    job = client.create_transcription_job(audio_path, model='universal', outputs=outputs, title=Path(audio_path).stem)
    job_id = job['job_id']
    print('klangio job:', job)
    status = client.wait_for_completion(job_id, poll_s=3.0, max_wait_s=300.0)
    print('klangio status:', status)
    if (status.get('status') or '').upper() != 'COMPLETED':
        raise SystemExit(f'Klangio job failed: {status}')
    import time
    time.sleep(8)
    wrote = 0
    _FMT_MAP = {
        'mxml':       ('xml',        'musicxml'),
        'midi':       ('midi',       'mid'),
        'midi_quant': ('midi_quant', 'quant.mid'),
        'gp5':        ('gp5',        'gp5'),
        'pdf':        ('pdf',        'pdf'),
    }
    for fmt in outputs:
        dl_fmt, ext = _FMT_MAP.get(fmt, (fmt, fmt))
        try:
            client.download_result(job_id, dl_fmt, str(out_dir / f'out.{ext}'))
            wrote += 1
        except Exception as e:
            print(f'[KLANGIO] Warning: could not download {fmt}: {e}')
    if derive_notes:
        midi_path = out_dir / 'out.mid'
        if not midi_path.exists():
            print("[WARN] --derive-notes requested but out.mid not downloaded. Add 'midi' to --outputs.")
        elif pretty_midi is None:
            print('[WARN] pretty_midi not installed; cannot derive notes.json. Install: python -m pip install pretty_midi')
        else:
            notes = _noteevents_from_midi(str(midi_path))
            mapped = map_notes_to_guitar(notes, max_fret=24, chord_window_s=0.1)
            rows = []
            for n in mapped:
                if n.string is None or n.fret is None:
                    continue
                rows.append({'pitch': int(n.pitch), 'onset_s': float(n.onset), 'dur_s': float(n.duration), 'string': int(n.string), 'fret': int(n.fret)})
            (out_dir / 'notes.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
            wrote += 1
            print(f'[KLANGIO] derived notes.json: {len(rows)} notes')
    print(f'[KLANGIO] wrote {wrote} file(s) to {out_dir}')
    return wrote

def _noteevents_from_midi(midi_path: str) -> List[NoteEvent]:
    pm = pretty_midi.PrettyMIDI(midi_path)
    inst = None
    best_n = -1
    for i in pm.instruments:
        if i.is_drum:
            continue
        if len(i.notes) > best_n:
            best_n = len(i.notes)
            inst = i
    if inst is None:
        return []
    out: List[NoteEvent] = []
    for n in inst.notes:
        onset = float(n.start)
        dur = float(max(0.01, n.end - n.start))
        out.append(NoteEvent(pitch=int(n.pitch), onset=onset, duration=dur, velocity=int(max(1, min(127, n.velocity)))))
    out.sort(key=lambda x: (x.onset, x.pitch))
    return out

def main() -> None:
    parser = argparse.ArgumentParser(description='Notecryption transcription pipeline')
    parser.add_argument('--in', dest='input_path', required=True, help='Input audio (wav/mp3)')
    parser.add_argument('--out', dest='output_dir', required=True, help='Output directory')
    parser.add_argument('--backend', choices=('local', 'klangio'), default='local', help='Transcription backend: local (your pipeline) or klangio (cloud).')
    parser.add_argument('--use-llm', action='store_true', help='Enable LLM-based clean-up (requires OPENAI_API_KEY).')
    parser.add_argument('--outputs', default='mxml,midi', help='Klangio outputs: comma-separated: mxml,midi,midi_quant,gp5,pdf')
    parser.add_argument('--derive-notes', action='store_true', help='After Klangio returns MIDI, derive notes.json and map to guitar.')
    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.backend == 'local':
        cfg = RunConfig(use_llm=args.use_llm)
        notes = run_local_pipeline(args.input_path, str(out_dir), cfg)
        print(f'Transcription complete (local): {len(notes)} notes')
    else:
        if not os.environ.get('KLANGIO_API_KEY'):
            raise SystemExit('Missing KLANGIO_API_KEY env var.')
        outputs = [s.strip() for s in args.outputs.split(',') if s.strip()]
        _run_klangio(args.input_path, out_dir, outputs, args.derive_notes)
        print('Transcription complete (klangio).')
    for ext in ('notes.json', 'out.musicxml', 'out.mid', 'out.gp5', 'out.pdf', 'out_quant.mid'):
        p = out_dir / ext
        if p.exists():
            print(f'  -> {p}')
if __name__ == '__main__':
    main()