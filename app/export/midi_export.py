from __future__ import annotations
from typing import Any, Dict, List
import pretty_midi

def export_midi(notes: List[Any], out_path: str) -> None:
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=24)
    for note in notes:
        instrument.notes.append(pretty_midi.Note(velocity=int(getattr(note, 'velocity', 80)), pitch=int(note.pitch), start=float(note.onset), end=float(note.onset + note.duration)))
    pm.instruments.append(instrument)
    pm.write(out_path)

def export_midi_from_dicts(notes: List[Dict], out_path: str, bpm: int=120) -> None:
    pm = pretty_midi.PrettyMIDI(initial_tempo=float(bpm))
    instrument = pretty_midi.Instrument(program=24)
    for n in notes:
        onset = float(n['onset_s'])
        dur = float(n['dur_s'])
        pitch = int(n['pitch'])
        if dur <= 0 or pitch < 0 or pitch > 127:
            continue
        instrument.notes.append(pretty_midi.Note(velocity=80, pitch=pitch, start=onset, end=onset + dur))
    pm.instruments.append(instrument)
    pm.write(out_path)