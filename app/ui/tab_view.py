from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from PyQt6.QtCore import Qt, QRectF
from PyQt6.QtGui import QFont, QPainter, QPen
from PyQt6.QtWidgets import QWidget

from app.models.note_event import NoteEvent


@dataclass
class TabRenderConfig:
    strings: Tuple[str, ...] = ("e", "B", "G", "D", "A", "E")  # display order top->bottom
    line_spacing: int = 18
    left_pad: int = 56
    top_pad: int = 24
    system_gap: int = 24
    beats_per_system: float = 4.0
    seconds_per_system: float = 2.5  # fallback if tempo unknown
    px_per_second: float = 260.0
    show_measure_numbers: bool = True


class TabView(QWidget):
    """
    Clean, wrapped guitar tab view similar to Songsterr "systems".

    Expects NoteEvent with optional .string (1..6) and .fret.
    If missing, will render pitch as fret number without string mapping.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self._notes: List[NoteEvent] = []
        self._duration_s: float = 0.0
        self._playhead_s: float = 0.0
        self.cfg = TabRenderConfig()
        self.setMinimumHeight(260)

    def set_notes(self, notes: List[NoteEvent], duration_s: float) -> None:
        self._notes = list(notes or [])
        self._duration_s = float(duration_s or 0.0)
        self.update()

    def set_playhead_seconds(self, t: float) -> None:
        self._playhead_s = max(0.0, float(t))
        self.update()

    def sizeHint(self):  # noqa: N802
        # height depends on systems; keep a reasonable default
        return super().sizeHint()

    def _note_x(self, onset_s: float, system_start_s: float) -> float:
        return self.cfg.left_pad + (onset_s - system_start_s) * self.cfg.px_per_second

    def _system_bounds(self, sys_len: float) -> List[Tuple[float, float]]:
        if self._duration_s <= 0:
            return [(0.0, sys_len)]
        n = max(1, int((self._duration_s / sys_len) + 0.999))
        out = []
        for i in range(n):
            a = i * sys_len
            b = min(self._duration_s, (i + 1) * sys_len)
            out.append((a, b))
        return out

    def paintEvent(self, ev):  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.fillRect(self.rect(), self.palette().window())

        cfg = self.cfg
        w = self.width()

        # fit one system exactly to the available row width
        usable_w = max(1, w - cfg.left_pad - 14)
        sys_len = max(0.5, usable_w / cfg.px_per_second)

        # fonts
        font_label = QFont()
        font_label.setPointSize(9)
        font_fret = QFont("Consolas")
        font_fret.setPointSize(10)

        pen_line = QPen(self.palette().mid().color())
        pen_line.setWidth(1)

        pen_note = QPen(self.palette().text().color())
        pen_note.setWidth(2)

        systems = self._system_bounds(sys_len)
        sys_height = (len(cfg.strings) - 1) * cfg.line_spacing + 28

        # expand widget height to fit all rows
        needed_h = cfg.top_pad + len(systems) * (sys_height + cfg.system_gap)
        if self.minimumHeight() != needed_h:
            self.setMinimumHeight(needed_h)

        y = cfg.top_pad
        system_idx = 0

        for (a_s, b_s) in systems:
            system_idx += 1

            # measure number (approx)
            if cfg.show_measure_numbers:
                p.setFont(font_label)
                p.setPen(self.palette().mid().color())
                p.drawText(8, y + 2, 40, 16, int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter), str(system_idx))

            # draw strings lines + labels
            p.setPen(pen_line)
            for i, sname in enumerate(cfg.strings):
                yy = y + i * cfg.line_spacing
                p.drawLine(cfg.left_pad, yy, w - 14, yy)
                p.setFont(font_label)
                p.setPen(self.palette().mid().color())
                p.drawText(0, yy - 8, cfg.left_pad - 8, 16, int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter), sname)
                p.setPen(pen_line)

            # filter notes for this system window
            notes = [n for n in self._notes if (n.onset >= a_s and n.onset < b_s + 1e-6)]
            # draw notes (fret text)
            for n in notes:
                x = self._note_x(n.onset, a_s)
                if x < cfg.left_pad or x > w - 18:
                    continue

                # string mapping
                string_idx = None
                fret_txt = None

                if getattr(n, "string", None) is not None and getattr(n, "fret", None) is not None:
                    # your mapper uses string 1..6 (1 high e)
                    s = int(getattr(n, "string"))
                    f = int(getattr(n, "fret"))
                    # display order is e,B,G,D,A,E => string 1 is high e => index 0
                    string_idx = max(0, min(5, s - 1))
                    fret_txt = str(f)
                else:
                    # fallback: render pitch number
                    fret_txt = str(int(n.pitch))

                if string_idx is None:
                    string_idx = 0

                yy = y + string_idx * cfg.line_spacing

                p.setFont(font_fret)
                p.fillRect(QRectF(x - 10, yy - 9, 24, 18), self.palette().window())
                p.setPen(self.palette().text().color())
                p.drawText(QRectF(x - 10, yy - 10, 24, 20), Qt.AlignmentFlag.AlignCenter, fret_txt)

            # playhead (only if inside this system)
            if self._duration_s > 0 and (self._playhead_s >= a_s and self._playhead_s <= b_s):
                x = self._note_x(self._playhead_s, a_s)
                pen_ph = QPen(Qt.GlobalColor.red)
                pen_ph.setWidth(2)
                p.setPen(pen_ph)
                p.drawLine(int(x), int(y - 8), int(x), int(y + (len(cfg.strings) - 1) * cfg.line_spacing + 8))

            # next system
            y += (len(cfg.strings) - 1) * cfg.line_spacing + cfg.system_gap