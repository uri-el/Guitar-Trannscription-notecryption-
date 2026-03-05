from __future__ import annotations
from dataclasses import replace
from typing import Dict, List, Optional, Tuple
from app.models.note_event import NoteEvent


def cleanup_notes(
    notes: List[NoteEvent],
    *,
    min_dur: float = 0.06,              # was 0.12 → allow fast picked notes
    dedupe_onset_tol: float = 0.03,     # was 0.04 → tighter dedup
    pitch_range: Tuple[int, int] = (40, 88),
    max_velocity_diff_for_ghost: int = 40,
) -> List[NoteEvent]:
    """Clean up notes with Klang-quality filtering.

    Key improvements:
    - Ghost note removal: if a quiet note overlaps a loud note of similar pitch,
      remove the quiet one
    - Tighter deduplication
    - Remove notes outside guitar range
    """
    # Basic filtering
    out = [
        n for n in notes
        if n.duration >= min_dur and pitch_range[0] <= n.pitch <= pitch_range[1]
    ]
    out.sort(key=lambda n: (n.onset, n.pitch))

    # Deduplicate same pitch within tolerance
    deduped: List[NoteEvent] = []
    last_by_pitch: Dict[int, NoteEvent] = {}
    for n in out:
        prev = last_by_pitch.get(n.pitch)
        if prev is not None and abs(n.onset - prev.onset) <= dedupe_onset_tol:
            keep = n if n.velocity >= prev.velocity else prev
            keep = replace(keep, onset=min(n.onset, prev.onset))
            last_by_pitch[n.pitch] = keep
            continue
        last_by_pitch[n.pitch] = n
        deduped.append(n)

    deduped.sort(key=lambda n: (n.onset, n.pitch))

    # Remove ghost notes: quiet notes that overlap with a much louder note
    # within 1-2 semitones (common Basic Pitch artifact)
    cleaned = _remove_ghost_notes(deduped, max_velocity_diff_for_ghost)

    return cleaned


def _remove_ghost_notes(
    notes: List[NoteEvent], vel_diff_threshold: int = 40
) -> List[NoteEvent]:
    """Remove quiet notes that are likely detection artifacts near louder notes."""
    if len(notes) < 2:
        return notes
    notes = sorted(notes, key=lambda n: (n.onset, n.pitch))
    keep = [True] * len(notes)
    for i, n1 in enumerate(notes):
        if not keep[i]:
            continue
        for j in range(i + 1, len(notes)):
            if not keep[j]:
                continue
            n2 = notes[j]
            # Only check notes close in time
            if n2.onset > n1.onset + n1.duration + 0.05:
                break
            # Check for overlap
            overlap_start = max(n1.onset, n2.onset)
            overlap_end = min(n1.onset + n1.duration, n2.onset + n2.duration)
            if overlap_end - overlap_start < 0.02:
                continue
            # If pitches are within 2 semitones and one is much quieter
            pitch_diff = abs(n1.pitch - n2.pitch)
            if pitch_diff <= 2 and pitch_diff > 0:
                if n1.velocity - n2.velocity > vel_diff_threshold:
                    keep[j] = False
                elif n2.velocity - n1.velocity > vel_diff_threshold:
                    keep[i] = False
                    break
    return [n for i, n in enumerate(notes) if keep[i]]


def quantize_notes(
    notes: List[NoteEvent], *, bpm: float, grid: str = '1/16'
) -> List[NoteEvent]:
    """Quantize note onsets and durations to a rhythmic grid.

    Improved to also snap durations to musical values and handle
    triplet grids.
    """
    if bpm <= 0:
        return notes
    beat = 60.0 / float(bpm)

    grid_map = {
        '1/4': 1.0,
        '1/8': 0.5,
        '1/8T': 1.0 / 3.0,
        '1/16': 0.25,
        '1/16T': 1.0 / 6.0,
        '1/32': 0.125,
    }
    step = beat * grid_map.get(grid, 0.25)

    out: List[NoteEvent] = []
    for n in notes:
        q_on = round(n.onset / step) * step
        q_off = round((n.onset + n.duration) / step) * step
        dur = max(step, q_off - q_on)
        out.append(replace(n, onset=float(q_on), duration=float(dur)))
    out.sort(key=lambda n: (n.onset, n.pitch))
    return out
