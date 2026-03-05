from __future__ import annotations
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import numpy as np
import soundfile as sf

@dataclass
class DemucsConfig:
    model: str = 'htdemucs'
    device: str = 'cpu'
    jobs: int = 0
    two_stems: Optional[str] = None

def separate_with_demucs(audio_path: str, cfg: Optional[DemucsConfig]=None) -> Path:
    cfg = cfg or DemucsConfig()
    out_root = Path(tempfile.mkdtemp(prefix='demucs_sep_')).resolve()
    cmd = ['python', '-m', 'demucs', '-n', cfg.model, '-o', str(out_root)]
    if cfg.device:
        cmd += ['-d', cfg.device]
    if cfg.jobs and cfg.jobs > 0:
        cmd += ['-j', str(int(cfg.jobs))]
    if cfg.two_stems:
        cmd += ['--two-stems', cfg.two_stems]
    cmd += [audio_path]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError('Demucs failed. Install with: python -m pip install demucs') from e
    sep_dir = out_root / 'separated' / cfg.model
    if not sep_dir.exists():
        sep_dir = out_root / cfg.model
    track_dirs = [p for p in sep_dir.glob('*') if p.is_dir()]
    if not track_dirs:
        raise RuntimeError(f'Could not find Demucs output under {sep_dir}')
    return track_dirs[0]

def pick_best_guitar_stem(track_dir: Path) -> Path:
    other = track_dir / 'other.wav'
    bass = track_dir / 'bass.wav'
    if other.exists() and (not bass.exists()):
        return other
    if bass.exists() and (not other.exists()):
        return bass
    if not other.exists() and (not bass.exists()):
        wavs = list(track_dir.glob('*.wav'))
        if not wavs:
            raise RuntimeError(f'No stem wavs found in {track_dir}')
        return wavs[0]
    e_other = _band_energy(other, f_lo=80, f_hi=1200)
    e_bass = _band_energy(bass, f_lo=80, f_hi=1200)
    return bass if e_bass > 1.15 * e_other else other

def _band_energy(wav_path: Path, f_lo: float, f_hi: float) -> float:
    y, sr = sf.read(str(wav_path), dtype='float32')
    if y.ndim > 1:
        y = np.mean(y, axis=1).astype(np.float32)
    n = int(min(len(y), sr * 8))
    y = y[:n]
    if n < 2048:
        return float(np.mean(y ** 2))
    Y = np.fft.rfft(y)
    freqs = np.fft.rfftfreq(len(y), d=1.0 / sr)
    mask = (freqs >= f_lo) & (freqs <= f_hi)
    if not np.any(mask):
        return float(np.mean(y ** 2))
    return float(np.mean(np.abs(Y[mask]) ** 2))