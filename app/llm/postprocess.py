from __future__ import annotations
import json
import os
from dataclasses import dataclass
from typing import List, Optional
from app.models.note_event import NoteEvent

@dataclass
class LLMConfig:
    enabled: bool = False
    model: str = 'gpt-4.1-mini'
    max_output_tokens: int = 800
    temperature: float = 0.2
    api_key_env: str = 'OPENAI_API_KEY'

def _notes_to_text(notes: List[NoteEvent]) -> str:
    lines = []
    for n in notes:
        lines.append(f"time={n.onset:.3f}s dur={n.duration:.3f}s pitch={int(n.pitch)} vel={int(getattr(n, 'velocity', 80))} string={getattr(n, 'string', None)} fret={getattr(n, 'fret', None)}")
    return '\n'.join(lines)

def _build_prompt(notes: List[NoteEvent]) -> str:
    return f'You are an expert guitarist and music theorist.\nYou receive noisy guitar transcription from an audio model.\nEach line below is one note with onset (seconds), duration (seconds), MIDI pitch, velocity, string, and fret (if assigned).\n\nYour task:\n- Remove obvious noise notes (extremely short or out-of-context).\n- Fix impossible string/fret combinations and unplayable stretches.\n- Prefer realistic guitar voicings and small hand movements between chords.\n- Make only MINIMAL corrections; do not invent long runs of new notes.\n\nReturn ONLY JSON, no explanation, with this schema:\n[\n  {{"onset": float_seconds,\n   "duration": float_seconds,\n   "pitch": int_midi,\n   "velocity": int_0_127,\n   "string": int_or_null,\n   "fret": int_or_null,\n   "chord_id": int_or_null }}\n]\n\nHere is the current note sequence:\n\n{_notes_to_text(notes)}\n'

def refine_notes_with_llm(notes: List[NoteEvent], cfg: Optional[LLMConfig]=None) -> List[NoteEvent]:
    if not notes:
        return notes
    cfg = cfg or LLMConfig()
    if not cfg.enabled:
        return notes
    api_key = os.getenv(cfg.api_key_env, '')
    if not api_key:
        return notes
    try:
        from openai import OpenAI
    except Exception:
        return notes
    client = OpenAI(api_key=api_key)
    prompt = _build_prompt(notes)
    try:
        resp = client.chat.completions.create(model=cfg.model, messages=[{'role': 'system', 'content': 'You output ONLY JSON, no explanation.'}, {'role': 'user', 'content': prompt}], temperature=cfg.temperature, max_tokens=cfg.max_output_tokens)
        content = resp.choices[0].message.content or ''
    except Exception:
        return notes
    try:
        data = json.loads(content)
    except Exception:
        return notes
    refined: List[NoteEvent] = []
    for item in data:
        try:
            refined.append(NoteEvent(pitch=int(item['pitch']), onset=float(item['onset']), duration=float(item['duration']), velocity=int(item.get('velocity', 80)), string=item.get('string'), fret=item.get('fret'), chord_id=item.get('chord_id')))
        except Exception:
            continue
    return refined or notes