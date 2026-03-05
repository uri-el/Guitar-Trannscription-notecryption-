from __future__ import annotations
import argparse
from pathlib import Path
from app.cloud.klangio_api import client_from_env

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--audio', required=True, help='Path to audio file (mp3/wav)')
    ap.add_argument('--out', required=True, help='Output directory')
    ap.add_argument('--model', default='universal', help='Klangio model (default: universal)')
    ap.add_argument('--outputs', default='mxml', help='Comma-separated: mxml,gp5,midi,midi_quant,pdf')
    ap.add_argument('--wait', type=float, default=180.0, help='Max wait seconds')
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    client = client_from_env()
    outputs = [s.strip() for s in args.outputs.split(',') if s.strip()]
    job = client.create_transcription_job(args.audio, model=args.model, outputs=outputs, title=Path(args.audio).stem)
    job_id = job['job_id']
    print('job:', job)
    status = client.wait_for_completion(job_id, max_wait_s=args.wait)
    print('status:', status)
    if (status.get('status') or '').upper() != 'COMPLETED':
        raise SystemExit(f'Job failed: {status}')
    import time
    time.sleep(5)
    for fmt in outputs:
        ext = 'mxml' if fmt == 'mxml' else fmt
        out_path = out_dir / f'klangio_{job_id}.{ext}'
        dl_fmt = 'xml' if fmt == 'mxml' else fmt
        client.download_result(job_id, dl_fmt, str(out_path))
        print('saved:', out_path)
if __name__ == '__main__':
    main()