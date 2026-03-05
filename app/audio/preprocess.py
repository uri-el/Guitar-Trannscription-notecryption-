from __future__ import annotations
import os
import subprocess
import tempfile

def _ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return 'ffmpeg'
from dataclasses import dataclass
from typing import Optional
import numpy as np
import soundfile as sf


@dataclass
class PreprocessConfig:
    target_sr: int = 22050
    mono: bool = True
    use_loudnorm: bool = True
    highpass_hz: int = 80        # was 70 → filter more low rumble
    lowpass_hz: int = 6000       # was 8000 → less high-freq noise
    use_hpss: bool = False
    hpss_margin: float = 2.0
    # New: apply a gentle compressor to even out dynamics
    use_compressor: bool = False
    compressor_threshold: float = -20.0  # dB
    compressor_ratio: float = 3.0


def ensure_wav_mono(input_path: str, target_sr: int = 22050) -> str:
    out = _temp_wav_path()
    cmd = [
        'ffmpeg', '-y', '-i', input_path,
        '-ac', '1', '-ar', str(int(target_sr)), out
    ]
    _run_ffmpeg(cmd)
    return out


def preprocess_audio(
    input_path: str, cfg: Optional[PreprocessConfig] = None
) -> str:
    cfg = cfg or PreprocessConfig()
    out = _temp_wav_path()
    af = []

    if cfg.use_loudnorm:
        af.append('loudnorm=I=-16:TP=-1.5:LRA=11')

    if cfg.highpass_hz and cfg.highpass_hz > 0:
        # Use 2nd-order highpass for steeper rolloff
        af.append(f'highpass=f={int(cfg.highpass_hz)}:poles=2')

    if cfg.lowpass_hz and cfg.lowpass_hz > 0:
        af.append(f'lowpass=f={int(cfg.lowpass_hz)}:poles=2')

    if cfg.use_compressor:
        af.append(
            f'acompressor=threshold={cfg.compressor_threshold}dB:'
            f'ratio={cfg.compressor_ratio}:attack=5:release=50'
        )

    cmd = ['ffmpeg', '-y', '-i', input_path]
    if cfg.mono:
        cmd += ['-ac', '1']
    cmd += ['-ar', str(int(cfg.target_sr))]
    if af:
        cmd += ['-af', ','.join(af)]
    cmd += [out]
    _run_ffmpeg(cmd)

    if not cfg.use_hpss:
        return out

    try:
        import librosa
    except Exception:
        return out

    y, sr = sf.read(out, dtype='float32')
    if isinstance(y, np.ndarray) and y.ndim > 1:
        y = np.mean(y, axis=1).astype(np.float32)
    if y is None or len(y) < 512:
        return out

    S = librosa.stft(y)
    H, _P = librosa.decompose.hpss(S, margin=(cfg.hpss_margin, cfg.hpss_margin))
    y_h = librosa.istft(H).astype(np.float32)
    out2 = _temp_wav_path()
    sf.write(out2, y_h, sr)
    try:
        os.remove(out)
    except Exception:
        pass
    return out2


def _run_ffmpeg(cmd: list[str]) -> None:
    cmd[0] = _ffmpeg_exe()
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except FileNotFoundError as e:
        raise RuntimeError(
            "ffmpeg not found. Install ffmpeg and ensure it's on PATH."
        ) from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            'ffmpeg failed. Check input file and ffmpeg installation.'
        ) from e


def _temp_wav_path() -> str:
    fd, path = tempfile.mkstemp(suffix='.wav')
    os.close(fd)
    return path
