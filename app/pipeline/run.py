from __future__ import annotations
import collections
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import soundfile as sf
from app.audio.preprocess import PreprocessConfig, preprocess_audio
from app.audio.separation_demucs import DemucsConfig, pick_best_guitar_stem, separate_with_demucs
from app.llm.postprocess import LLMConfig, refine_notes_with_llm
from app.models.note_event import NoteEvent
from app.postprocess.cleanup import cleanup_notes, quantize_notes
from app.transcription.articulation import annotate_articulations
from app.transcription.basic_pitch_poly import BasicPitchConfig, transcribe_with_basic_pitch
from app.transcription.guitar_mapper import map_notes_to_guitar
try:
    from app.export.musicxml_export import export_notes_to_musicxml, MusicXmlConfig
except Exception:
    export_notes_to_musicxml = None
    MusicXmlConfig = None
try:
    from app.export.midi_export import export_midi_from_dicts
except Exception:
    export_midi_from_dicts = None


@dataclass
class RunConfig:
    use_demucs: bool = False
    demucs_model: str = 'htdemucs'
    demucs_device: str = 'cpu'

    # ── Audio preprocessing ──
    target_sr: int = 22050
    use_loudnorm: bool = True
    highpass_hz: int = 80
    lowpass_hz: int = 8000          # higher cutoff preserves harmonics for better pitch detection
    use_hpss: bool = True           # isolate harmonic content from mic noise/room sound
    use_compressor: bool = True     # even out dynamics so quiet notes aren't dropped

    # ── Basic Pitch thresholds ──
    bp_onset_threshold: float = 0.45    # lower = catches more real notes
    bp_frame_threshold: float = 0.30
    bp_min_amplitude: float = 0.12      # lower = picks up softer playing

    # ── Post-processing ──
    consolidate_gap_s: float = 0.06
    min_dur_s: float = 0.05             # allow slightly shorter picked notes
    dedupe_onset_tol_s: float = 0.03
    pitch_range: Tuple[int, int] = (40, 88)
    max_fret: int = 19                  # was 24 → stay in normal playing range
    chord_window_s: float = 0.06        # was 0.1 → tighter chord grouping
    max_poly: int = 6

    # ── Quantization (off by default, enable for cleaner output) ──
    enable_quantize: bool = True        # was False → Klang always quantizes
    quantize_bpm: int = 120
    quantize_grid: str = '1/16'

    # ── Articulation (less aggressive) ──
    enable_articulation: bool = True

    # ── Export ──
    export_musicxml: bool = True
    musicxml_title: str = 'Transcription'
    musicxml_bpm: int = 120
    musicxml_key_fifths: int = 0
    musicxml_filename: str = 'out.musicxml'

    # ── YIN verification ──
    use_yin_verify: bool = False
    yin_attack_skip_s: float = 0.04
    yin_analysis_s: float = 0.2
    max_yin_drift: int = 1              # was 2 → stricter pitch verification

    # ── LLM ──
    use_llm: bool = False
    llm_model: str = 'gpt-4o-mini'


def _hz_to_midi(f: float) -> float:
    f = float(f)
    if f <= 0.0:
        return 0.0
    return 69.0 + 12.0 * np.log2(f / 440.0)


def _stable_pitch(
    audio: np.ndarray, sr: int, *,
    fmin: float = 65.0, fmax: float = 2100.0,
    frame_length: int = 2048, hop_length: int = 256,
    trough_threshold: float = 0.1, min_valid_frames: int = 4,
    min_agreement: float = 0.4,
) -> Optional[int]:
    try:
        import librosa
    except Exception:
        return None
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    audio = np.asarray(audio, dtype=np.float32)
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    if not audio.size:
        return None
    max_abs = float(np.max(np.abs(audio)))
    if max_abs > 0.01:
        audio = audio / max_abs
    f0 = librosa.yin(audio, fmin=fmin, fmax=fmax, sr=sr,
                     frame_length=frame_length, hop_length=hop_length,
                     trough_threshold=trough_threshold, center=True)
    midi_votes: List[int] = []
    for freq in f0:
        if not (np.isfinite(freq) and freq > 0.0):
            continue
        midi = int(round(_hz_to_midi(float(freq))))
        if 40 <= midi <= 96:
            midi_votes.append(midi)
    if len(midi_votes) < min_valid_frames:
        return None
    counts = collections.Counter(midi_votes)
    best, best_count = counts.most_common(1)[0]
    if best_count < len(midi_votes) * min_agreement:
        return None
    return best


def _verify_pitches_with_yin(
    notes: List[NoteEvent], audio: np.ndarray, sr: int,
    *, attack_skip_s: float = 0.04, analysis_s: float = 0.2,
    max_drift: int = 1, chord_window_s: float = 0.06,
) -> List['NoteEvent']:
    from dataclasses import replace as dc_replace

    # Identify chord notes — YIN is monophonic and cannot reliably
    # pitch-detect individual voices in a simultaneous cluster.
    in_chord = [False] * len(notes)
    sorted_idx = sorted(range(len(notes)), key=lambda i: notes[i].onset)
    for a in range(len(sorted_idx)):
        ia = sorted_idx[a]
        for b in range(a + 1, len(sorted_idx)):
            ib = sorted_idx[b]
            if notes[ib].onset - notes[ia].onset > chord_window_s:
                break
            in_chord[ia] = True
            in_chord[ib] = True

    out = []
    for i, n in enumerate(notes):
        start = int((float(n.onset) + attack_skip_s) * sr)
        end = int((float(n.onset) + attack_skip_s + analysis_s) * sr)
        end = min(end, len(audio))
        if end <= start:
            out.append(n)
            continue
        yin_pitch = _stable_pitch(audio[start:end], sr)
        if yin_pitch is None:
            # YIN couldn't confirm — keep original
            out.append(n)
        elif abs(yin_pitch - int(n.pitch)) <= max_drift:
            # YIN confirms or minor correction
            out.append(dc_replace(n, pitch=yin_pitch))
        else:
            # YIN disagrees significantly
            if in_chord[i]:
                # YIN can't handle polyphony — trust Basic Pitch
                out.append(n)
            else:
                # Solo note — trust YIN
                out.append(dc_replace(n, pitch=yin_pitch))
    return out


def _detect_bpm(audio: np.ndarray, sr: int) -> Optional[float]:
    """Estimate tempo using librosa. Returns None if unavailable or out of range."""
    try:
        import librosa
        tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
        bpm = float(tempo[0]) if hasattr(tempo, '__len__') else float(tempo)
        # Fold into 60–200 BPM range
        while bpm < 60:
            bpm *= 2.0
        while bpm > 200:
            bpm /= 2.0
        if 60.0 <= bpm <= 200.0:
            return bpm
    except Exception:
        pass
    return None


def _load_mono_float32(path: str) -> Tuple[np.ndarray, int]:
    audio, sr = sf.read(path, dtype='float32')
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    audio = np.asarray(audio, dtype=np.float32)
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return (audio, int(sr))


def consolidate_fragments(
    notes: List['NoteEvent'], gap: float
) -> List['NoteEvent']:
    if not notes:
        return notes
    notes = sorted(notes, key=lambda n: (n.pitch, n.onset))
    out: List['NoteEvent'] = []
    cur = notes[0]
    for n in notes[1:]:
        if n.pitch == cur.pitch:
            gap_t = n.onset - (cur.onset + cur.duration)
            if gap_t <= gap:
                end = max(cur.onset + cur.duration, n.onset + n.duration)
                cur.duration = end - cur.onset
                cur.velocity = max(cur.velocity, n.velocity)
                continue
        out.append(cur)
        cur = n
    out.append(cur)
    out.sort(key=lambda n: (n.onset, n.pitch))
    return out


def _cap_polyphony(
    notes: List['NoteEvent'], max_poly: int = 6, window_s: float = 0.05
) -> List['NoteEvent']:
    if not notes or max_poly <= 0:
        return notes
    notes_s = sorted(notes, key=lambda n: (n.onset, n.pitch))
    groups: List[List['NoteEvent']] = []
    cur: List['NoteEvent'] = [notes_s[0]]
    t0 = notes_s[0].onset
    for n in notes_s[1:]:
        if n.onset - t0 <= window_s:
            cur.append(n)
        else:
            groups.append(cur)
            cur = [n]
            t0 = n.onset
    groups.append(cur)
    out: List['NoteEvent'] = []
    for g in groups:
        if len(g) <= max_poly:
            out.extend(g)
        else:
            g.sort(key=lambda n: -n.velocity)
            out.extend(g[:max_poly])
    out.sort(key=lambda n: (n.onset, n.pitch))
    return out


def _remove_overlapping_same_string(notes: List[Dict]) -> List[Dict]:
    """If two notes are assigned to the same string and overlap in time,
    truncate the earlier note. A real guitar string can only play one note."""
    if not notes:
        return notes
    notes = sorted(notes, key=lambda r: (int(r['string']), float(r['onset_s'])))
    out: List[Dict] = []
    for n in notes:
        if out and out[-1]['string'] == n['string']:
            prev = out[-1]
            prev_end = float(prev['onset_s']) + float(prev['dur_s'])
            if prev_end > float(n['onset_s']):
                prev['dur_s'] = max(0.01, float(n['onset_s']) - float(prev['onset_s']))
        out.append(n)
    out.sort(key=lambda r: (float(r['onset_s']), int(r['pitch'])))
    return out


def run_pipeline(
    audio_path: str,
    out_dir: Optional[str] = None,
    cfg: Optional[RunConfig] = None,
) -> List[Dict]:
    cfg = cfg or RunConfig()
    out_path: Optional[Path]
    if out_dir is not None:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
    else:
        out_path = None

    # ── Source separation ──
    input_for_preprocess = audio_path
    if cfg.use_demucs:
        track_dir = separate_with_demucs(
            audio_path,
            DemucsConfig(model=cfg.demucs_model, device=cfg.demucs_device)
        )
        stem_path = pick_best_guitar_stem(track_dir)
        input_for_preprocess = str(stem_path)
        print(f'[DEMUX] using stem: {stem_path}')

    # ── Preprocess audio ──
    wav_path = preprocess_audio(
        input_for_preprocess,
        PreprocessConfig(
            target_sr=cfg.target_sr,
            use_loudnorm=cfg.use_loudnorm,
            highpass_hz=cfg.highpass_hz,
            lowpass_hz=cfg.lowpass_hz,
            use_hpss=cfg.use_hpss,
            use_compressor=cfg.use_compressor,
        ),
    )
    audio, sr = _load_mono_float32(wav_path)

    # ── Transcribe with Basic Pitch ──
    bp_cfg = BasicPitchConfig(
        onset_threshold=cfg.bp_onset_threshold,
        frame_threshold=cfg.bp_frame_threshold,
        min_note_length=0.01,
        min_amplitude=cfg.bp_min_amplitude,
        pitch_min=cfg.pitch_range[0],
        pitch_max=cfg.pitch_range[1],
        suppress_octave_errors=True,
        prefer_cpu=True,
    )
    raw_notes = transcribe_with_basic_pitch(wav_path, cfg=bp_cfg)
    print(f'[PIPELINE] raw_notes: {len(raw_notes)}')

    # ── Consolidate fragments ──
    raw_notes = consolidate_fragments(raw_notes, gap=cfg.consolidate_gap_s)
    print(f'[PIPELINE] after consolidate: {len(raw_notes)}')

    # ── Cleanup ──
    raw_notes = cleanup_notes(
        raw_notes,
        min_dur=cfg.min_dur_s,
        dedupe_onset_tol=cfg.dedupe_onset_tol_s,
        pitch_range=cfg.pitch_range,
    )
    print(f'[PIPELINE] after cleanup: {len(raw_notes)}')

    # ── Cap polyphony ──
    raw_notes = _cap_polyphony(raw_notes, max_poly=cfg.max_poly)
    print(f'[PIPELINE] after polyphony cap: {len(raw_notes)}')

    # ── YIN pitch verification ──
    if cfg.use_yin_verify:
        raw_notes = _verify_pitches_with_yin(
            raw_notes, audio, sr,
            attack_skip_s=cfg.yin_attack_skip_s,
            analysis_s=cfg.yin_analysis_s,
            max_drift=cfg.max_yin_drift,
        )
        print(f'[PIPELINE] after YIN verify: {len(raw_notes)}')

    # ── Quantize ──
    if cfg.enable_quantize:
        detected_bpm = _detect_bpm(audio, sr)
        bpm = detected_bpm or cfg.quantize_bpm
        if detected_bpm:
            print(f'[PIPELINE] auto-detected BPM: {detected_bpm:.1f}')
        raw_notes = quantize_notes(raw_notes, bpm=bpm, grid=cfg.quantize_grid)
        print(f'[PIPELINE] after quantize: {len(raw_notes)}')
        # Dedup: snapping can land two same-pitch notes on the same grid slot
        raw_notes = cleanup_notes(
            raw_notes,
            min_dur=cfg.min_dur_s,
            dedupe_onset_tol=0.001,
            pitch_range=cfg.pitch_range,
        )
        print(f'[PIPELINE] after post-quantize dedup: {len(raw_notes)}')

    # ── Map to guitar fretboard ──
    mapped = map_notes_to_guitar(
        raw_notes, max_fret=cfg.max_fret, chord_window_s=cfg.chord_window_s
    )
    print(f'[PIPELINE] after mapping: {len(mapped)}')

    # ── LLM refinement ──
    if cfg.use_llm:
        llm_cfg = LLMConfig(enabled=True, model=cfg.llm_model)
        mapped = refine_notes_with_llm(mapped, llm_cfg)

    # ── Build output dicts ──
    out: List[Dict] = []
    skipped = 0
    for n in mapped:
        if n.string is None or n.fret is None:
            skipped += 1
            continue
        out.append({
            'pitch': int(n.pitch),
            'onset_s': float(n.onset),
            'dur_s': float(n.duration),
            'string': int(n.string),
            'fret': int(n.fret),
        })
    print(f'[PIPELINE] skipped due to missing string/fret: {skipped}')
    print(f'[PIPELINE] final notes: {len(out)}')

    out.sort(key=lambda r: (float(r['onset_s']), int(r['string']), int(r['pitch'])))

    # ── Remove overlapping notes on same string ──
    out = _remove_overlapping_same_string(out)

    # ── Annotate articulations ──
    if cfg.enable_articulation and out:
        out = annotate_articulations(audio=audio, sr=sr, notes=out)

    # ── Export ──
    if cfg.export_musicxml and out_path and export_notes_to_musicxml and MusicXmlConfig and out:
        xml_file = out_path / cfg.musicxml_filename
        try:
            export_notes_to_musicxml(
                out, str(xml_file),
                MusicXmlConfig(
                    title=cfg.musicxml_title,
                    bpm=cfg.musicxml_bpm,
                    key_fifths=cfg.musicxml_key_fifths,
                    time_sig=(4, 4),
                    divisions=960,
                    instrument_name='Guitar',
                ),
            )
        except Exception:
            pass

    if out_path and out and export_midi_from_dicts is not None:
        try:
            export_midi_from_dicts(out, str(out_path / 'out.mid'), bpm=cfg.musicxml_bpm)
        except Exception:
            pass

    if out_path and out:
        try:
            import json as _json
            (out_path / 'notes.json').write_text(
                _json.dumps(out, indent=2), encoding='utf-8'
            )
        except Exception:
            pass

    return out
