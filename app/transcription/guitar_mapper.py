from __future__ import annotations
from dataclasses import replace
from typing import Dict, List, Optional, Tuple
import itertools
from app.models.note_event import NoteEvent

STANDARD_TUNING: Dict[int, int] = {6: 40, 5: 45, 4: 50, 3: 55, 2: 59, 1: 64}


def _candidate_positions(
    pitch: int, max_fret: int = 24, tuning: Dict[int, int] = STANDARD_TUNING
) -> List[Tuple[int, int]]:
    cands: List[Tuple[int, int]] = []
    for s in (6, 5, 4, 3, 2, 1):
        fret = pitch - tuning[s]
        if 0 <= fret <= max_fret:
            cands.append((s, int(fret)))
    return cands


def _pos_cost(
    pos: Tuple[int, int],
    prev: Optional[Tuple[int, int]],
    *,
    ideal_fret: int = 0,          # was 5 → strongly prefer open position
) -> float:
    """Cost function tuned to match Klang-style open/low-position preference.

    Key changes from original:
    - ideal_fret = 0 (open position preferred)
    - Open strings get a BONUS instead of penalty
    - Fret cost is quadratic to strongly penalise high positions
    - String proximity matters less than fret proximity
    """
    s, f = pos
    # Quadratic cost that strongly penalises high frets
    cost = 0.05 * (f ** 1.5)

    # BONUS for open strings (was a penalty of +0.35!)
    if f == 0:
        cost -= 0.8

    # Bonus for frets 0-4 (first position)
    if 0 <= f <= 4:
        cost -= 0.3

    # Light penalty for going above 7th fret
    if f > 7:
        cost += 0.5 * (f - 7)

    # Prefer thicker strings for lower pitches (more natural voicing)
    # string 6=low E .. string 1=high E
    # For low pitches, prefer higher string numbers (lower strings)
    if prev is not None:
        ps, pf = prev
        cost += 0.6 * abs(f - pf)     # was 1.0 → fret jumps still matter
        cost += 0.2 * abs(s - ps)     # was 0.35 → string jumps matter less
    return float(cost)


def _group_into_chord_windows(
    notes: List[NoteEvent], window_s: float
) -> List[List[NoteEvent]]:
    if not notes:
        return []
    notes = sorted(notes, key=lambda n: (n.onset, n.pitch))
    groups: List[List[NoteEvent]] = []
    cur = [notes[0]]
    t0 = notes[0].onset
    for n in notes[1:]:
        if n.onset - t0 <= window_s:
            cur.append(n)
        else:
            groups.append(cur)
            cur = [n]
            t0 = n.onset
    groups.append(cur)
    return groups


def _viterbi_map_monophonic(
    notes: List[NoteEvent], max_fret: int
) -> List[NoteEvent]:
    if not notes:
        return []
    ns = sorted(notes, key=lambda n: (n.onset, n.pitch))
    cand_lists = [
        _candidate_positions(n.pitch, max_fret=max_fret) for n in ns
    ]
    out: List[NoteEvent] = []
    if any(len(c) == 0 for c in cand_lists):
        prev = None
        for n, cands in zip(ns, cand_lists):
            if not cands:
                out.append(n)
                prev = None
                continue
            best = min(cands, key=lambda p: _pos_cost(p, prev))
            out.append(replace(n, string=best[0], fret=best[1]))
            prev = best
        out.sort(key=lambda n: (n.onset, n.pitch))
        return out

    dp: List[Dict[Tuple[int, int], float]] = []
    back: List[Dict[Tuple[int, int], Tuple[int, int]]] = []
    dp0: Dict[Tuple[int, int], float] = {
        p: _pos_cost(p, None) for p in cand_lists[0]
    }
    dp.append(dp0)
    back.append({})

    for i in range(1, len(ns)):
        dpi: Dict[Tuple[int, int], float] = {}
        backi: Dict[Tuple[int, int], Tuple[int, int]] = {}
        for p in cand_lists[i]:
            best_prev = None
            best_cost = float('inf')
            for pp, prev_cost in dp[i - 1].items():
                c = prev_cost + _pos_cost(p, pp)
                if c < best_cost:
                    best_cost = c
                    best_prev = pp
            dpi[p] = best_cost
            backi[p] = best_prev
        dp.append(dpi)
        back.append(backi)

    last = min(dp[-1].items(), key=lambda kv: kv[1])[0]
    path = [last]
    for i in range(len(ns) - 1, 0, -1):
        last = back[i][last]
        path.append(last)
    path.reverse()

    out = [
        replace(n, string=s, fret=f) for n, (s, f) in zip(ns, path)
    ]
    out.sort(key=lambda n: (n.onset, n.pitch))
    return out


def _voicing_cost(combo: Tuple[Tuple[int, int], ...],
                  prev_anchor: Optional[Tuple[int, int]]) -> float:
    """Evaluate a chord voicing, heavily preferring open/low positions."""
    strings = [s for s, f in combo]
    if len(set(strings)) != len(strings):
        return float('inf')  # duplicate strings

    frets = [f for s, f in combo]
    non_open_frets = [f for f in frets if f > 0]

    spread = (max(non_open_frets) - min(non_open_frets)) if non_open_frets else 0

    # Fret height cost (quadratic)
    fret_cost = sum(0.05 * (f ** 1.5) for f in frets)

    # Open string bonus
    open_bonus = sum(0.8 for f in frets if f == 0)

    # First position bonus
    first_pos_bonus = sum(0.3 for f in frets if 0 <= f <= 4)

    # Spread penalty (playability)
    spread_cost = 1.0 * spread if spread > 4 else 0.3 * spread

    # Movement cost
    move = 0.0
    if prev_anchor is not None:
        avg_fret = int(round(sum(frets) / len(frets)))
        min_string = min(strings)
        move = abs(avg_fret - prev_anchor[1]) + 0.3 * abs(min_string - prev_anchor[0])

    return fret_cost - open_bonus - first_pos_bonus + spread_cost + 0.8 * move


def _solve_voicing_for_group(
    group: List[NoteEvent],
    prev_anchor: Optional[Tuple[int, int]],
    max_fret: int,
) -> List[NoteEvent]:
    if len(group) == 1:
        return _viterbi_map_monophonic(group, max_fret=max_fret)

    group = sorted(group, key=lambda n: -n.pitch)[:6]
    cand_lists = [
        _candidate_positions(n.pitch, max_fret=max_fret) for n in group
    ]
    if any(len(c) == 0 for c in cand_lists):
        return _viterbi_map_monophonic(group, max_fret=max_fret)

    best_combo = None
    best_cost = float('inf')
    for combo in itertools.product(*cand_lists):
        cost = _voicing_cost(combo, prev_anchor)
        if cost < best_cost:
            best_cost = cost
            best_combo = combo

    if best_combo is None:
        return _viterbi_map_monophonic(group, max_fret=max_fret)

    out = [
        replace(n, string=s, fret=f)
        for n, (s, f) in zip(group, best_combo)
    ]
    out.sort(key=lambda n: (n.onset, n.pitch))
    return out


def map_notes_to_guitar(
    notes: List[NoteEvent],
    max_fret: int = 24,
    *,
    chord_window_s: float = 0.06,   # was 0.1 → tighter chord grouping
) -> List[NoteEvent]:
    if not notes:
        return []
    groups = _group_into_chord_windows(notes, window_s=float(chord_window_s))
    out: List[NoteEvent] = []
    prev_anchor: Optional[Tuple[int, int]] = None
    for g in groups:
        mapped = _solve_voicing_for_group(g, prev_anchor, max_fret=max_fret)
        out.extend(mapped)
        mapped2 = [
            m for m in mapped if m.string is not None and m.fret is not None
        ]
        if mapped2:
            avg_fret = int(round(
                sum(m.fret for m in mapped2 if m.fret is not None) / len(mapped2)
            ))
            min_string = min(
                m.string for m in mapped2 if m.string is not None
            )
            prev_anchor = (int(min_string), int(avg_fret))
    out.sort(key=lambda n: (n.onset, n.pitch))
    return out


def map_to_guitar(notes: List[NoteEvent], max_fret: int = 24) -> List[NoteEvent]:
    return map_notes_to_guitar(notes, max_fret=max_fret)


__all__ = ['STANDARD_TUNING', 'map_notes_to_guitar', 'map_to_guitar']
