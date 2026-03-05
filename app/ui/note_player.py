from __future__ import annotations
import os
import tempfile
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple, List

import numpy as np
import soundfile as sf


@dataclass
class SynthNote:
    pitch: int
    onset_s: float
    dur_s: float


def midi_to_freq(midi: int) -> float:
    return float(440.0 * (2.0 ** ((midi - 69) / 12.0)))


def _karplus_strong(freq: float, n_samples: int, sr: int) -> np.ndarray:
    """
    Karplus-Strong plucked-string synthesis.

    Algorithm (period-at-a-time vectorised):
      1. Fill a delay line of length N = round(sr / freq) with lowpass-filtered
         noise, then apply a pluck-position comb filter for tonal shaping.
      2. Each period: output the current delay line, then advance it by running
         a one-pole lowpass (averaging) filter with a pitch-dependent decay factor.

    This runs in O(n_samples / N) numpy iterations — fast enough in pure Python.
    """
    if n_samples <= 0:
        return np.zeros(0, dtype=np.float32)

    N = max(2, round(sr / freq))

    # ── Decay factor ───────────────────────────────────────────────
    # Lower strings sustain longer (0.9993 @ MIDI 40, 0.9985 @ MIDI 76)
    midi = 69.0 + 12.0 * np.log2(max(freq, 1.0) / 440.0)
    decay = np.clip(0.9993 - (midi - 40.0) / 200.0, 0.9980, 0.9996)

    # ── Excitation: lowpass-filtered noise ─────────────────────────
    rng = np.random.default_rng(int(freq * 1000) % (2 ** 31))
    noise = rng.standard_normal(N)

    # Warm up with a boxcar lowpass (simulates pick/finger pluck timbre)
    warm = max(1, round(N * 0.05))
    kernel = np.ones(warm) / warm
    buf = np.convolve(noise, kernel, mode="same")

    # ── Pluck-position comb filter ──────────────────────────────────
    # Notch at integer multiples of 1/pluck_pos, like a finger touching
    # the string at ~12% from the bridge (close to bridge = brighter).
    pn = max(1, round(N * 0.12))
    if pn < N:
        tmp = buf.copy()
        tmp[:N - pn] = (buf[:N - pn] - buf[pn:]) * 0.5
        buf = tmp

    # Normalise excitation to ±1
    mx = np.max(np.abs(buf))
    if mx > 1e-9:
        buf /= mx

    # ── Synthesise via period-at-a-time KS ─────────────────────────
    out = np.empty(n_samples, dtype=np.float32)
    pos = 0
    while pos < n_samples:
        chunk = min(N, n_samples - pos)
        out[pos : pos + chunk] = buf[:chunk].astype(np.float32)
        pos += chunk
        # One-pole lowpass + decay (the KS "string filter")
        rolled = np.roll(buf, -1)
        rolled[-1] = buf[0]          # wrap: end feeds back to start
        buf = (buf + rolled) * (0.5 * decay)

    # Short fade-out at end of note to remove click on cut-off
    fade = min(int(0.008 * sr), n_samples)
    if fade > 0:
        out[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)

    return out


def _render(notes: list[SynthNote], sr: int, end_t: float) -> np.ndarray:
    n_samples = int(end_t * sr) + 1
    y = np.zeros(n_samples, dtype=np.float32)
    for n in notes:
        f = midi_to_freq(int(n.pitch))
        start = max(0, int(n.onset_s * sr))
        # Let the note ring for at least its full duration; cap at end of buffer
        dur = max(1, int(n.dur_s * sr))
        end = min(n_samples, start + dur)
        if end <= start:
            continue
        tone = _karplus_strong(f, end - start, sr)
        y[start:end] += tone
    mx = float(np.max(np.abs(y))) if y.size else 0.0
    if mx > 1e-6:
        y /= mx
    return y


def synth_notes_to_wav_path(
    notes: Iterable[SynthNote],
    *,
    sr: int = 44100,
    gain: float = 0.35,
    total_duration_s: Optional[float] = None,
) -> Optional[str]:
    notes = list(notes)
    if not notes and not total_duration_s:
        return None
    end_t = float(total_duration_s) if total_duration_s else (
        max(n.onset_s + n.dur_s for n in notes) if notes else 0.0)
    if end_t <= 0:
        return None
    y = _render(notes, sr, end_t) * float(gain)
    fd, path = tempfile.mkstemp(suffix=".wav", prefix="note_synth_")
    os.close(fd)
    sf.write(path, y, sr)
    return path


def synth_notes_to_array(
    notes: Iterable[SynthNote],
    *,
    sr: int = 44100,
    gain: float = 0.35,
    total_duration_s: Optional[float] = None,
) -> Tuple[np.ndarray, int]:
    notes = list(notes)
    if not notes and not total_duration_s:
        return np.zeros(0, dtype=np.float32), sr
    end_t = float(total_duration_s) if total_duration_s else (
        max(n.onset_s + n.dur_s for n in notes) if notes else 0.0)
    if end_t <= 0:
        return np.zeros(0, dtype=np.float32), sr
    y = _render(notes, sr, end_t) * float(gain)
    return y, sr


def notes_to_synthnotes(notes: List, default_vel: int = 80) -> List[SynthNote]:
    out: List[SynthNote] = []
    for n in notes or []:
        onset = float(getattr(n, "onset", getattr(n, "onset_s", 0.0)))
        dur = float(getattr(n, "duration", getattr(n, "dur_s", 0.1)))
        pitch = int(getattr(n, "pitch", 60))
        out.append(SynthNote(pitch=pitch, onset_s=onset, dur_s=dur))
    return out
