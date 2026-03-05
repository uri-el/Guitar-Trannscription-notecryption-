from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

@dataclass
class NoteEvent:
    pitch: int
    onset: float
    duration: float
    velocity: int = 80
    string: Optional[int] = None
    fret: Optional[int] = None
    chord_id: Optional[int] = None

    @property
    def offset(self) -> float:
        return float(self.onset + self.duration)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)