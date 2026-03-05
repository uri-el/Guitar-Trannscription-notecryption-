from __future__ import annotations

import base64
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import QUrl
from PyQt6.QtWebEngineWidgets import QWebEngineView


class ScoreView(QWebEngineView):
    """
    Score renderer + playhead bridge for OSMD (score.html).

    API used by MainWindow:
      - load_musicxml(path, title=...)
      - set_duration_seconds(d)
      - set_playhead_seconds(t)
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        web_dir = Path(__file__).resolve().parent / "web"
        self._html_path = (web_dir / "score.html").resolve()

        self._ready: bool = False
        self._pending_xml_b64: Optional[str] = None
        self._pending_duration: Optional[float] = None
        self._pending_playhead: Optional[float] = None

        self.loadFinished.connect(self._on_loaded)
        self.setUrl(QUrl.fromLocalFile(str(self._html_path)))

    def _on_loaded(self, ok: bool) -> None:
        self._ready = bool(ok)
        if not self._ready:
            return

        # Flush pending calls in order: xml -> duration -> playhead
        if self._pending_xml_b64 is not None:
            self._js_load_musicxml_b64(self._pending_xml_b64)
            self._pending_xml_b64 = None

        if self._pending_duration is not None:
            self.set_duration_seconds(self._pending_duration)
            self._pending_duration = None

        if self._pending_playhead is not None:
            self.set_playhead_seconds(self._pending_playhead)
            self._pending_playhead = None

    def load_musicxml(self, musicxml_path: str, *, title: str = "") -> None:
        xml = Path(musicxml_path).read_text(encoding="utf-8", errors="replace")
        b64 = base64.b64encode(xml.encode("utf-8")).decode("ascii")

        if not self._ready:
            self._pending_xml_b64 = b64
            self.setUrl(QUrl.fromLocalFile(str(self._html_path)))
            return

        self._js_load_musicxml_b64(b64)

    def _js_load_musicxml_b64(self, b64: str) -> None:
        # Decode base64 to string inside the page to avoid JS-string escaping issues
        js = (
            "(() => {"
            f" const b64 = {b64!r};"
            " const bytes = Uint8Array.from(atob(b64), c => c.charCodeAt(0));"
            " const xml = new TextDecoder('utf-8').decode(bytes);"
            " if (window.loadMusicXML) window.loadMusicXML(xml);"
            "})();"
        )
        self.page().runJavaScript(js)

    def set_duration_seconds(self, d: float) -> None:
        if not self._ready:
            self._pending_duration = float(d)
            return
        self.page().runJavaScript(
            f"if (window.setDurationSeconds) window.setDurationSeconds({float(d)});"
        )

    def set_playhead_seconds(self, t: float) -> None:
        if not self._ready:
            self._pending_playhead = float(t)
            return
        self.page().runJavaScript(
            f"if (window.setPlayheadSeconds) window.setPlayheadSeconds({float(t)});"
        )

    def save_pdf(self, file_path: str) -> None:
        from PyQt6.QtGui import QPageLayout, QPageSize
        from PyQt6.QtCore import QMarginsF
        layout = QPageLayout(
            QPageSize(QPageSize.PageSizeId.A4),
            QPageLayout.Orientation.Landscape,
            QMarginsF(10, 10, 10, 10),
        )
        self.page().printToPdf(file_path, layout)
