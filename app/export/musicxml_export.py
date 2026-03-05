from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import math
import xml.etree.ElementTree as ET


@dataclass
class MusicXmlConfig:
    title: str = 'Transcription'
    bpm: int = 120
    time_sig: Tuple[int, int] = (4, 4)
    key_fifths: int = 0
    divisions: int = 960
    instrument_name: str = 'Guitar'
    add_chord_symbols: bool = False
    # Klang uses transposing clef for guitar (8va bassa)
    use_transposing_clef: bool = True


_NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


def midi_to_step_alter_oct(midi: int) -> Tuple[str, int, int]:
    pc = midi % 12
    octv = midi // 12 - 1
    name = _NOTE_NAMES[pc]
    if len(name) == 1:
        return (name, 0, octv)
    return (name[0], 1, octv)


def seconds_to_ticks(sec: float, bpm: int, divisions: int) -> int:
    q = 60.0 / float(bpm)
    return int(round(sec / q * divisions))


def _ticks_to_note_type(dur_ticks: int, divisions: int) -> Tuple[str, bool]:
    """Returns (note_type, is_dotted)."""
    quarter = divisions
    ratios = [
        (4.0, 'whole', False),
        (3.0, 'half', True),
        (2.0, 'half', False),
        (1.5, 'quarter', True),
        (1.0, 'quarter', False),
        (0.75, 'eighth', True),
        (0.5, 'eighth', False),
        (0.375, '16th', True),
        (0.25, '16th', False),
        (0.125, '32nd', False),
    ]
    ratio = dur_ticks / float(quarter) if quarter > 0 else 1.0
    best = 'quarter'
    best_dotted = False
    best_diff = 999.0
    for r, name, dotted in ratios:
        diff = abs(ratio - r)
        if diff < best_diff:
            best_diff = diff
            best = name
            best_dotted = dotted
    return best, best_dotted


def _sub(
    parent: ET.Element, tag: str, text: Optional[str] = None, **attrib
) -> ET.Element:
    el = ET.SubElement(parent, tag, attrib)
    if text is not None:
        el.text = str(text)
    return el


def _note_el(
    *, pitch_midi: int, dur_ticks: int, voice: int, staff: int,
    string: int, fret: int,
    tech: Optional[Dict] = None,
    is_chord: bool = False,
    tie_start: bool = False, tie_stop: bool = False,
) -> ET.Element:
    n = ET.Element('note')

    # Chord indicator (simultaneous note in same voice)
    if is_chord:
        _sub(n, 'chord')

    pitch = _sub(n, 'pitch')
    step, alter, octv = midi_to_step_alter_oct(int(pitch_midi))
    _sub(pitch, 'step', step)
    if alter != 0:
        _sub(pitch, 'alter', str(alter))
    _sub(pitch, 'octave', str(octv))

    _sub(n, 'duration', str(int(dur_ticks)))
    _sub(n, 'voice', str(int(voice)))

    note_type, is_dotted = _ticks_to_note_type(dur_ticks, 960)
    _sub(n, 'type', note_type)
    if is_dotted:
        _sub(n, 'dot')

    _sub(n, 'staff', str(int(staff)))

    if tie_start:
        _sub(n, 'tie', type='start')
    if tie_stop:
        _sub(n, 'tie', type='stop')

    notations = None
    if tech or tie_start or tie_stop:
        notations = _sub(n, 'notations')
    if tie_start and notations is not None:
        _sub(notations, 'tied', type='start')
    if tie_stop and notations is not None:
        _sub(notations, 'tied', type='stop')

    # Always add string/fret technical info
    technical = None
    if notations is None:
        notations = _sub(n, 'notations')
    technical = _sub(notations, 'technical')
    _sub(technical, 'string', str(int(string)))
    _sub(technical, 'fret', str(int(fret)))

    if tech and technical is not None:
        t = tech.get('type')
        if t == 'bend':
            bend = _sub(technical, 'bend')
            semis = float(tech.get('semitones', 1.0))
            # Quantize bend to nearest quarter tone for cleaner display
            semis_q = round(semis * 4) / 4
            if abs(semis_q) >= 0.25:
                _sub(bend, 'bend-alter', f'{semis_q:.2f}'.rstrip('0').rstrip('.'))
        elif t == 'slide':
            _sub(notations, 'slide', type='start')
        elif t == 'hammer':
            _sub(notations, 'hammer-on', type='start')
        elif t == 'pull':
            _sub(notations, 'pull-off', type='start')

    return n


def export_notes_to_musicxml(
    notes: List[Dict], out_path: str,
    cfg: MusicXmlConfig = MusicXmlConfig(),
) -> None:
    notes = [dict(n) for n in notes]
    notes.sort(key=lambda r: (float(r['onset_s']), int(r['pitch'])))

    score = ET.Element('score-partwise', version='3.1')
    work = _sub(score, 'work')
    _sub(work, 'work-title', cfg.title)

    part_list = _sub(score, 'part-list')
    score_part = _sub(part_list, 'score-part', id='P1')
    _sub(score_part, 'part-name', cfg.instrument_name)

    part = _sub(score, 'part', id='P1')

    beats, beat_type = cfg.time_sig
    ticks_per_measure = int(cfg.divisions * beats * (4 / beat_type))

    def measure_index(on_ticks: int) -> int:
        return int(on_ticks // ticks_per_measure) + 1

    # Pre-compute tick positions
    measures: Dict[int, List[Dict]] = {}
    for n in notes:
        on = seconds_to_ticks(float(n['onset_s']), cfg.bpm, cfg.divisions)
        dur = max(1, seconds_to_ticks(float(n['dur_s']), cfg.bpm, cfg.divisions))
        n['_on'] = on
        n['_dur'] = dur
        mi = measure_index(on)
        measures.setdefault(mi, []).append(n)

    max_m = max(measures.keys()) if measures else 1

    for mi in range(1, max_m + 1):
        meas = _sub(part, 'measure', number=str(mi))

        if mi == 1:
            attr = _sub(meas, 'attributes')
            _sub(attr, 'divisions', str(cfg.divisions))

            key = _sub(attr, 'key')
            _sub(key, 'fifths', str(cfg.key_fifths))

            ts = _sub(attr, 'time')
            _sub(ts, 'beats', str(beats))
            _sub(ts, 'beat-type', str(beat_type))

            _sub(attr, 'staves', '2')

            clef1 = _sub(attr, 'clef', number='1')
            _sub(clef1, 'sign', 'G')
            _sub(clef1, 'line', '2')
            if cfg.use_transposing_clef:
                _sub(clef1, 'clef-octave-change', '-1')

            clef2 = _sub(attr, 'clef', number='2')
            _sub(clef2, 'sign', 'TAB')
            _sub(clef2, 'line', '5')

            sd = _sub(attr, 'staff-details', number='2')
            _sub(sd, 'staff-lines', '6')
            # Add tuning info for TAB
            tuning_pitches = [
                ('E', 2), ('A', 2), ('D', 3), ('G', 3), ('B', 3), ('E', 4)
            ]
            for line_num, (step, octave) in enumerate(tuning_pitches, 1):
                st = _sub(sd, 'staff-tuning', line=str(line_num))
                _sub(st, 'tuning-step', step)
                _sub(st, 'tuning-octave', str(octave))

            direction = _sub(meas, 'direction', placement='above')
            dir_type = _sub(direction, 'direction-type')
            met = _sub(dir_type, 'metronome')
            _sub(met, 'beat-unit', 'quarter')
            _sub(met, 'per-minute', str(cfg.bpm))
            _sub(direction, 'sound', tempo=str(cfg.bpm))

        items = measures.get(mi, [])
        items.sort(key=lambda r: (int(r['_on']), int(r['pitch'])))

        cursor = (mi - 1) * ticks_per_measure

        def add_rest(dur_ticks: int, staff: int):
            if dur_ticks <= 0:
                return
            rn = _sub(meas, 'note')
            _sub(rn, 'rest')
            _sub(rn, 'duration', str(int(dur_ticks)))
            _sub(rn, 'voice', '1')
            note_type, is_dotted = _ticks_to_note_type(dur_ticks, cfg.divisions)
            _sub(rn, 'type', note_type)
            if is_dotted:
                _sub(rn, 'dot')
            _sub(rn, 'staff', str(int(staff)))

        # Group notes at the same onset tick into chords
        i = 0
        while i < len(items):
            n = items[i]
            on = int(n['_on'])
            dur = int(n['_dur'])

            if on > cursor:
                add_rest(on - cursor, staff=1)
                add_rest(on - cursor, staff=2)
                cursor = on

            # Collect all notes at this onset
            chord_notes = [n]
            j = i + 1
            while j < len(items) and int(items[j]['_on']) == on:
                chord_notes.append(items[j])
                j += 1

            for ci, cn in enumerate(chord_notes):
                tech = cn.get('tech')
                pitch = int(cn['pitch'])
                string = max(1, min(6, int(cn['string'])))
                fret = max(0, min(24, int(cn['fret'])))
                cn_dur = int(cn['_dur'])
                is_chord = ci > 0  # First note is not chord, subsequent are

                # Staff 1 (standard notation)
                meas.append(_note_el(
                    pitch_midi=pitch, dur_ticks=cn_dur,
                    voice=1, staff=1,
                    string=string, fret=fret, tech=tech,
                    is_chord=is_chord,
                ))
                # Staff 2 (tab)
                meas.append(_note_el(
                    pitch_midi=pitch, dur_ticks=cn_dur,
                    voice=1, staff=2,
                    string=string, fret=fret, tech=tech,
                    is_chord=is_chord,
                ))

            # Advance cursor by max duration in the chord
            max_dur = max(int(cn['_dur']) for cn in chord_notes)
            cursor = on + max_dur
            i = j

        end = mi * ticks_per_measure
        if cursor < end:
            add_rest(end - cursor, staff=1)
            add_rest(end - cursor, staff=2)

        if mi == max_m:
            bar = _sub(meas, 'barline', location='right')
            _sub(bar, 'bar-style', 'light-heavy')

    tree = ET.ElementTree(score)
    ET.indent(tree, space='  ')
    tree.write(out_path, encoding='utf-8', xml_declaration=True)
