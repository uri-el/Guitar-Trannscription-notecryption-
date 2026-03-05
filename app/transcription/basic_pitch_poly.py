from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple
import warnings
import numpy as np
from app.models.note_event import NoteEvent

@dataclass
class BasicPitchConfig:
    # ── Much more conservative thresholds to reduce ghost notes ──
    onset_threshold: float = 0.6        # was 0.5  → higher = fewer false onsets
    frame_threshold: float = 0.4        # was 0.3  → higher = stricter note frames
    min_note_length: float = 0.058      # was 0.08 → allow short picked notes
    merge_same_pitch_gap: float = 0.04  # was 0.05 → slightly tighter merge
    min_amplitude: float = 0.20         # was 0.15 → reject more quiet ghosts
    suppress_octave_errors: bool = True
    pitch_min: int = 40                 # was 30   → guitar low E = MIDI 40
    pitch_max: int = 88                 # was 96   → guitar high E fret 24
    prefer_cpu: bool = False

    # ── New: additional filtering ──
    velocity_floor: int = 25            # reject notes below this velocity
    max_simultaneous: int = 6           # guitar has 6 strings max
    onset_cluster_window: float = 0.04  # notes within this window = chord


def transcribe_with_basic_pitch(
    audio_path: str, cfg: Optional[BasicPitchConfig] = None
) -> List[NoteEvent]:
    cfg = cfg or BasicPitchConfig()
    if cfg.prefer_cpu:
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    try:
        from basic_pitch.inference import predict
    except ImportError as e:
        raise RuntimeError(
            f'Basic Pitch is not installed. Please run: pip install basic-pitch[tf]\n'
            f'Original error: {e}'
        ) from e
    try:
        _, _, note_events = predict(
            audio_path,
            onset_threshold=cfg.onset_threshold,
            frame_threshold=cfg.frame_threshold,
            minimum_note_length=cfg.min_note_length,
        )
    except Exception as e:
        warnings.warn(f'Basic Pitch inference failed: {e}')
        return []
    if note_events is None or len(note_events) == 0:
        return []
    raw_notes = _convert_to_note_events(note_events, cfg)
    processed = _post_process_notes(raw_notes, cfg)
    processed.sort(key=lambda x: (x.onset, x.pitch))
    return processed


def _convert_to_note_events(
    note_events: List[Tuple[float, float, int, float]], cfg: BasicPitchConfig
) -> List[NoteEvent]:
    notes = []
    for row in note_events:
        try:
            if len(row) < 3:
                continue
            start = float(row[0])
            end = float(row[1])
            pitch = int(round(float(row[2])))
            amp = float(row[3]) if len(row) > 3 else 0.7
        except (ValueError, TypeError, IndexError):
            continue
        duration = end - start
        if duration <= 0 or duration < cfg.min_note_length:
            continue
        pitch = max(0, min(127, pitch))
        if not cfg.pitch_min <= pitch <= cfg.pitch_max:
            continue
        if amp < cfg.min_amplitude:
            continue
        velocity = int(max(1, min(127, round(amp * 127))))
        if velocity < cfg.velocity_floor:
            continue
        notes.append(NoteEvent(
            pitch=pitch, onset=start, duration=duration, velocity=velocity
        ))
    return notes


def _post_process_notes(
    notes: List[NoteEvent], cfg: BasicPitchConfig
) -> List[NoteEvent]:
    if not notes:
        return notes
    if cfg.merge_same_pitch_gap > 0:
        notes = _merge_same_pitch_notes(notes, cfg.merge_same_pitch_gap)
    if cfg.suppress_octave_errors:
        notes = _suppress_octave_errors(notes)
    # Cap polyphony per onset cluster
    notes = _cap_polyphony_per_cluster(
        notes, cfg.max_simultaneous, cfg.onset_cluster_window
    )
    # Remove very short notes that survived
    notes = [n for n in notes if n.duration >= cfg.min_note_length]
    return notes


def _merge_same_pitch_notes(
    notes: List[NoteEvent], max_gap: float
) -> List[NoteEvent]:
    if len(notes) < 2:
        return notes
    notes = sorted(notes, key=lambda x: (x.pitch, x.onset))
    merged = []
    current = notes[0]
    for n in notes[1:]:
        if n.pitch == current.pitch:
            gap = n.onset - (current.onset + current.duration)
            if 0 <= gap <= max_gap:
                new_end = max(
                    current.onset + current.duration,
                    n.onset + n.duration
                )
                current.duration = new_end - current.onset
                current.velocity = max(current.velocity, n.velocity)
                continue
        merged.append(current)
        current = n
    merged.append(current)
    return merged


def _suppress_octave_errors(notes: List[NoteEvent]) -> List[NoteEvent]:
    """More aggressive harmonic/octave suppression than before."""
    if len(notes) < 2:
        return notes
    notes = sorted(notes, key=lambda x: x.onset)
    keep = [True] * len(notes)
    # True harmonic intervals: octave (12), octave+P5 (19), 2 octaves (24),
    # 2 octaves+M3 (28). P5 (7) is a real guitar interval (power chords)
    # and must NOT be suppressed.
    HARMONIC_INTERVALS = frozenset((12, 19, 24, 28))
    VELOCITY_RATIO = 0.70   # was 0.6 → more aggressive
    MIN_OVERLAP_S = 0.02    # was 0.03 → catch more overlapping harmonics

    for i, n1 in enumerate(notes):
        if not keep[i]:
            continue
        for j, n2 in enumerate(notes):
            if i == j or not keep[j]:
                continue
            overlap_start = max(n1.onset, n2.onset)
            overlap_end = min(
                n1.onset + n1.duration, n2.onset + n2.duration
            )
            if overlap_end - overlap_start < MIN_OVERLAP_S:
                continue
            interval = n2.pitch - n1.pitch
            if interval in HARMONIC_INTERVALS:
                if n2.velocity < VELOCITY_RATIO * n1.velocity:
                    keep[j] = False
            elif -interval in HARMONIC_INTERVALS:
                if n1.velocity < VELOCITY_RATIO * n2.velocity:
                    keep[i] = False
                    break
    return [n for i, n in enumerate(notes) if keep[i]]


def _cap_polyphony_per_cluster(
    notes: List[NoteEvent], max_poly: int, window: float
) -> List[NoteEvent]:
    """Within each onset cluster, keep only the max_poly loudest notes."""
    if not notes or max_poly <= 0:
        return notes
    notes = sorted(notes, key=lambda n: (n.onset, n.pitch))
    groups: List[List[NoteEvent]] = []
    cur = [notes[0]]
    t0 = notes[0].onset
    for n in notes[1:]:
        if n.onset - t0 <= window:
            cur.append(n)
        else:
            groups.append(cur)
            cur = [n]
            t0 = n.onset
    groups.append(cur)

    out: List[NoteEvent] = []
    for g in groups:
        if len(g) <= max_poly:
            out.extend(g)
        else:
            g.sort(key=lambda n: -n.velocity)
            out.extend(g[:max_poly])
    out.sort(key=lambda n: (n.onset, n.pitch))
    return out


__all__ = [
    'BasicPitchConfig',
    'transcribe_with_basic_pitch',
]
