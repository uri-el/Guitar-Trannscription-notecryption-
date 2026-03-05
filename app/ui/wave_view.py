from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from PyQt6.QtCore import Qt, QRectF, QPointF
from PyQt6.QtGui import QPainter, QPen
from PyQt6.QtWidgets import QWidget


def _downsample_abs(x: np.ndarray, target: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).flatten()
    if x.size == 0:
        return np.zeros((0,), dtype=np.float32)
    if target <= 0:
        return np.zeros((0,), dtype=np.float32)
    if x.size <= target:
        return np.abs(x)
    # block reduce to peak abs per bin
    n = x.size
    step = n / float(target)
    out = np.zeros((target,), dtype=np.float32)
    for i in range(target):
        a = int(i * step)
        b = int((i + 1) * step)
        if b <= a:
            b = min(n, a + 1)
        out[i] = float(np.max(np.abs(x[a:b])))
    return out


class WaveView(QWidget):
    """
    Dual waveform view:
      - top: original audio waveform
      - bottom: synth-from-notes waveform
    Also draws a vertical playhead.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(220)

        self._orig: Optional[np.ndarray] = None
        self._synth: Optional[np.ndarray] = None
        self._sr: int = 44100
        self._duration_s: float = 0.0

        self._playhead_s: float = 0.0

        self._cache_w: int = 0
        self._cache_orig: Optional[np.ndarray] = None
        self._cache_synth: Optional[np.ndarray] = None

    def set_waveforms(
        self,
        *,
        orig: Optional[np.ndarray],
        synth: Optional[np.ndarray],
        sr: int,
        duration_s: float,
    ) -> None:
        self._orig = None if orig is None else np.asarray(orig, dtype=np.float32).flatten()
        self._synth = None if synth is None else np.asarray(synth, dtype=np.float32).flatten()
        self._sr = int(sr) if sr else 44100
        self._duration_s = float(duration_s) if duration_s else 0.0
        self._cache_w = 0
        self._cache_orig = None
        self._cache_synth = None
        self.update()

    def set_playhead_seconds(self, t: float) -> None:
        self._playhead_s = max(0.0, float(t))
        self.update()

    def _ensure_cache(self, w: int) -> None:
        if w <= 0:
            return
        if self._cache_w == w and self._cache_orig is not None:
            return
        self._cache_w = w
        self._cache_orig = _downsample_abs(self._orig if self._orig is not None else np.zeros((0,), dtype=np.float32), w)
        self._cache_synth = _downsample_abs(self._synth if self._synth is not None else np.zeros((0,), dtype=np.float32), w)

    def paintEvent(self, ev):  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, False)

        # background
        p.fillRect(self.rect(), self.palette().window())

        w = self.width()
        h = self.height()
        if w <= 2 or h <= 2:
            return

        gap = 18
        mid = h // 2
        top_rect = QRectF(0, 0, w, mid - gap / 2)
        bot_rect = QRectF(0, mid + gap / 2, w, h - (mid + gap / 2))

        self._ensure_cache(w)

        # draw center lines
        pen_grid = QPen(self.palette().mid().color())
        pen_grid.setWidth(1)
        p.setPen(pen_grid)
        p.drawLine(int(top_rect.left()), int(top_rect.center().y()), int(top_rect.right()), int(top_rect.center().y()))
        p.drawLine(int(bot_rect.left()), int(bot_rect.center().y()), int(bot_rect.right()), int(bot_rect.center().y()))

        # draw waveforms
        pen_wave = QPen(self.palette().text().color())
        pen_wave.setWidth(1)
        p.setPen(pen_wave)

        def draw_wave(rect: QRectF, yabs: np.ndarray):
            if yabs is None or yabs.size == 0:
                return
            cx = rect.left()
            cy = rect.center().y()
            amp = rect.height() * 0.45
            for i in range(min(w, yabs.size)):
                a = float(yabs[i])
                x = cx + i
                y1 = cy - a * amp
                y2 = cy + a * amp
                p.drawLine(int(x), int(y1), int(x), int(y2))

        draw_wave(top_rect, self._cache_orig)
        draw_wave(bot_rect, self._cache_synth)

        # labels
        p.setPen(self.palette().mid().color())
        p.drawText(QRectF(8, 6, w - 16, 16), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, "ORIGINAL")
        p.drawText(QRectF(8, mid + gap / 2 + 6, w - 16, 16), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, "SYNTH (from notes)")

        # playhead
        if self._duration_s > 0.0:
            x = (self._playhead_s / self._duration_s) * (w - 1)
            x = max(0.0, min(float(w - 1), x))
            pen_ph = QPen(Qt.GlobalColor.red)
            pen_ph.setWidth(2)
            p.setPen(pen_ph)
            p.drawLine(int(x), int(top_rect.top()), int(x), int(bot_rect.bottom()))