from __future__ import annotations

import os
import tempfile
import time
import traceback
from contextlib import suppress
from pathlib import Path
from typing import Optional

import numpy as np

from PyQt6.QtCore import QThread, pyqtSignal, Qt, QUrl, QSize
from PyQt6.QtGui import QColor, QFont, QPalette
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from PyQt6.QtWidgets import (
    QButtonGroup,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QComboBox,
    QHBoxLayout,
    QProgressBar,
    QSlider,
    QStackedWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QFrame,
    QStyle,
)

from app.audio.preprocess import ensure_wav_mono
from app.live.audio_stream import AudioStreamConfig, LiveAudioInput
from app.models.note_event import NoteEvent
from app.pipeline.run import run_pipeline

from .note_player import synth_notes_to_wav_path, synth_notes_to_array, notes_to_synthnotes
from .score_view import ScoreView
from .tab_view import TabView
from .wave_view import WaveView


def _fmt_s(t: float) -> str:
    t = max(0.0, float(t))
    s = int(t)
    m = s // 60
    s = s % 60
    return f"{m}:{s:02d}"


# ─── Worker: runs local pipeline on a file ───────────────────────────
class Worker(QThread):
    done = pyqtSignal(list)
    fail = pyqtSignal(str)

    def __init__(self, audio_path: str, out_dir: str, title: str = "Transcription",
                 model: str = "basic_pitch"):
        super().__init__()
        self.audio_path = audio_path
        self.out_dir = out_dir
        self.title = title
        self.model = model

    def run(self):
        try:
            from app.pipeline.run import RunConfig
            cfg = RunConfig(musicxml_title=self.title)
            notes = run_pipeline(self.audio_path, out_dir=self.out_dir, cfg=cfg)
            self.done.emit(notes)
        except Exception:
            self.fail.emit(traceback.format_exc())


# ─── RecordWorker: just records audio ────────────────────────────────
class RecordWorker(QThread):
    elapsed = pyqtSignal(float)
    level = pyqtSignal(float)
    fail = pyqtSignal(str)

    def __init__(self, device_index: int | None, sample_rate: int = 44100):
        super().__init__()
        self.device_index = device_index
        self.sample_rate = sample_rate
        self._stop = False
        self._audio: LiveAudioInput | None = None

    def request_stop(self):
        self._stop = True

    def get_audio(self) -> tuple:
        if self._audio is None:
            return None, 0
        audio = self._audio.ring.get_all_audio()
        sr = int(self._audio.cfg.sample_rate)
        return audio, sr

    def run(self):
        try:
            import sounddevice as sd

            cfg = AudioStreamConfig(
                sample_rate=self.sample_rate,
                channels=1,
                buffer_seconds=600.0,
            )
            self._audio = LiveAudioInput(cfg)
            dev = (int(sd.default.device[0]) if self.device_index is None else int(self.device_index))
            self._audio.start(device=dev)

            while not self._stop:
                time.sleep(0.05)
                t = self._audio.ring.stream_time()
                self.elapsed.emit(float(t))
                chunk = self._audio.ring.get_last_seconds(0.05)
                if chunk.size > 0:
                    rms = float(np.sqrt(np.mean(chunk ** 2)))
                    self.level.emit(min(1.0, rms * 15))
        except Exception:
            self.fail.emit(traceback.format_exc())
        finally:
            try:
                if self._audio:
                    self._audio.stop()
            except Exception:
                pass


APP_QSS = """
QMainWindow, QWidget#Central { background: #111111; }

/* ── Dark bars ────────────────────────────────────────────── */
QFrame#TopBar {
  background: #111111;
  border: none;
  border-bottom: 1px solid #2a2a2a;
}
QFrame#NavBar {
  background: #111111;
  border: none;
  border-bottom: 1px solid #2a2a2a;
}
QFrame#ActionBar {
  background: #111111;
  border: none;
  border-top: 1px solid #2a2a2a;
}

/* ── White content & transport ───────────────────────────── */
QFrame#Content   { background: #ffffff; border: none; }
QFrame#TransportBar { background: #ffffff; border: none; border-top: 1px solid #e0e0e0; }

/* ── Default label colours ───────────────────────────────── */
QLabel { color: #111111; font-size: 12px; background: transparent; }
QLabel#TitleLabel { color: #111111; }
QFrame#TopBar QLabel,
QFrame#NavBar QLabel,
QFrame#ActionBar QLabel { color: #d0d0d0; }

/* ── Mic combo ───────────────────────────────────────────── */
QComboBox {
  background: #1e1e1e;
  color: #d0d0d0;
  border: 1px solid #333333;
  border-radius: 3px;
  padding: 3px 8px;
  min-height: 22px;
}
QComboBox::drop-down { border: none; width: 18px; }
QComboBox QAbstractItemView { background: #1e1e1e; color: #d0d0d0; selection-background-color: #333; }

/* ── Nav view-toggle buttons ─────────────────────────────── */
QPushButton#NavBtn {
  background: transparent;
  color: #888888;
  border: none;
  border-radius: 0px;
  padding: 6px 20px;
  font-size: 12px;
}
QPushButton#NavBtn:checked {
  color: #ffffff;
  border-bottom: 2px solid #ffffff;
}
QPushButton#NavBtn:hover:!checked { color: #bbbbbb; }

/* ── Action bar buttons ──────────────────────────────────── */
QPushButton#ActionBtn {
  background: #1e1e1e;
  color: #d0d0d0;
  border: none;
  border-radius: 0px;
  padding: 8px 0px;
  font-size: 12px;
  min-height: 32px;
}
QPushButton#ActionBtn:hover   { background: #2a2a2a; }
QPushButton#ActionBtn:pressed { background: #333333; }
QPushButton#ActionBtn:disabled { color: #555555; background: #181818; }

/* ── Refresh Mics (small dark button in top bar) ─────────── */
QPushButton#SmallBtn {
  background: #1e1e1e;
  color: #d0d0d0;
  border: 1px solid #333333;
  border-radius: 3px;
  padding: 3px 10px;
  font-size: 12px;
}
QPushButton#SmallBtn:hover { background: #2a2a2a; }

/* ── Recording indicator ─────────────────────────────────── */
QFrame#ActionBar QLabel#RecLabel { color: #ff3333; font-size: 12px; font-weight: bold; background: transparent; }

/* ── Level bar ───────────────────────────────────────────── */
QProgressBar#LevelBar {
  background: #1e1e1e;
  border: 1px solid #333333;
  border-radius: 3px;
  max-width: 80px;
  min-width: 80px;
  max-height: 10px;
}
QProgressBar#LevelBar::chunk { background: #44bb44; border-radius: 2px; }

/* ── Seek slider ─────────────────────────────────────────── */
QSlider::groove:horizontal {
  height: 4px;
  background: #d0d0d0;
  border-radius: 2px;
}
QSlider::sub-page:horizontal {
  background: #444444;
  border-radius: 2px;
}
QSlider::handle:horizontal {
  width: 14px;
  height: 14px;
  margin: -5px 0;
  background: #111111;
  border-radius: 7px;
}

/* ── Play button ─────────────────────────────────────────── */
QToolButton { background: transparent; border: none; color: #111111; }
QToolButton:hover { background: #f0f0f0; border-radius: 18px; }
"""


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Notecryption")
        self.resize(1280, 760)
        self.setStyleSheet(APP_QSS)

        # ── State ──
        self.audio_path: Optional[str] = None
        self._out_dir = str(Path("data/output").resolve())
        self._last_notes: list[NoteEvent] = []
        self.worker: Optional[Worker] = None
        self._record_worker: Optional[RecordWorker] = None
        self._recording_path: Optional[str] = None

        self._orig_wav_tmp: Optional[str] = None
        self._orig_wave: Optional[np.ndarray] = None
        self._orig_sr: int = 44100
        self._audio_duration_s: float = 0.0

        self._notes_wav_path: Optional[str] = None
        self._synth_wave: Optional[np.ndarray] = None

        # ── Views ──
        self.score = ScoreView()
        self.tab = TabView()
        self.waves = WaveView()

        _dark = QPalette()
        _dark.setColor(QPalette.ColorRole.Window, QColor("#1e1e1e"))
        _dark.setColor(QPalette.ColorRole.Text, QColor("#ffffff"))
        _dark.setColor(QPalette.ColorRole.Mid, QColor("#888888"))
        for v in (self.tab, self.waves):
            v.setPalette(_dark)

        self.stack = QStackedWidget()
        self.stack.addWidget(self.score)  # 0
        self.stack.addWidget(self.tab)    # 1
        self.stack.addWidget(self.waves)  # 2

        # ── Top bar: mic + refresh + level + model ──
        self.mic_combo = QComboBox()
        self.btn_refresh_mics = QPushButton("Refresh Mics")
        self.btn_refresh_mics.clicked.connect(self.refresh_mics)
        self.refresh_mics()
        self.btn_refresh_mics.setObjectName("SmallBtn")

        self.level_bar = QProgressBar()
        self.level_bar.setObjectName("LevelBar")
        self.level_bar.setRange(0, 100)
        self.level_bar.setValue(0)
        self.level_bar.setTextVisible(False)
        self.level_bar.setFixedHeight(10)
        self.level_bar.setVisible(False)

        self.model_combo = QComboBox()
        self.model_combo.addItem("Basic Pitch", "basic_pitch")
        self.model_combo.addItem("Klang.io", "klang")
        # Klang.io not yet implemented — disable it
        _item = self.model_combo.model().item(1)
        if _item:
            from PyQt6.QtCore import Qt as _Qt
            _item.setFlags(_item.flags() & ~_Qt.ItemFlag.ItemIsEnabled)

        top_bar = QFrame()
        top_bar.setObjectName("TopBar")
        top_l = QHBoxLayout(top_bar)
        top_l.setContentsMargins(12, 5, 12, 5)
        top_l.setSpacing(8)
        lbl_dev = QLabel("Input device:")
        top_l.addWidget(lbl_dev)
        top_l.addWidget(self.mic_combo, 1)
        top_l.addWidget(self.level_bar)
        top_l.addWidget(self.btn_refresh_mics)
        lbl_model = QLabel("Model:")
        top_l.addWidget(lbl_model)
        top_l.addWidget(self.model_combo)

        # ── Nav bar: Show Score / Tab / Waves / Save PDF ──
        self.btn_show_score = QPushButton("Show Score")
        self.btn_show_tab = QPushButton("Show Tab")
        self.btn_show_waves = QPushButton("Show waves")
        self.btn_save_pdf = QPushButton("Save PDF")
        for b in (self.btn_show_score, self.btn_show_tab, self.btn_show_waves):
            b.setCheckable(True)
            b.setObjectName("NavBtn")
        self.btn_save_pdf.setObjectName("NavBtn")

        group = QButtonGroup(self)
        group.setExclusive(True)
        group.addButton(self.btn_show_score)
        group.addButton(self.btn_show_tab)
        group.addButton(self.btn_show_waves)
        self.btn_show_score.setChecked(True)

        self.btn_show_score.clicked.connect(lambda: self._set_view(0))
        self.btn_show_tab.clicked.connect(lambda: self._set_view(1))
        self.btn_show_waves.clicked.connect(lambda: self._set_view(2))
        self.btn_save_pdf.clicked.connect(self.save_pdf)

        nav_bar = QFrame()
        nav_bar.setObjectName("NavBar")
        nav_l = QHBoxLayout(nav_bar)
        nav_l.setContentsMargins(0, 0, 0, 0)
        nav_l.setSpacing(0)
        nav_l.addWidget(self.btn_show_score)
        nav_l.addStretch(1)
        nav_l.addWidget(self.btn_show_tab)
        nav_l.addStretch(1)
        nav_l.addWidget(self.btn_show_waves)
        nav_l.addStretch(1)
        nav_l.addWidget(self.btn_save_pdf)

        # ── Content frame: Title + view ──
        self.title_label = QLabel("Transcription")
        self.title_label.setObjectName("TitleLabel")
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter)
        self.title_label.setFont(QFont("Times New Roman", 28))
        self.title_label.setContentsMargins(0, 14, 0, 10)
        self.title_label.setFixedHeight(60)

        content = QFrame()
        content.setObjectName("Content")
        content_l = QVBoxLayout(content)
        content_l.setContentsMargins(0, 0, 0, 0)
        content_l.setSpacing(0)
        content_l.addWidget(self.stack, 1)

        # ── Action bar (dark): Choose/Transcribe/Record/Stop ──
        self.btn_choose = QPushButton("Choose Audio")
        self.btn_transcribe = QPushButton("Transcribe")
        self.btn_record = QPushButton("Record")
        self.btn_stop_record = QPushButton("Stop Recording")
        self.btn_stop_record.setEnabled(False)
        for b in (self.btn_choose, self.btn_transcribe, self.btn_record, self.btn_stop_record):
            b.setObjectName("ActionBtn")

        self.lbl_rec = QLabel("● REC")
        self.lbl_rec.setObjectName("RecLabel")
        self.lbl_rec.setVisible(False)

        self.btn_choose.clicked.connect(self.choose_audio)
        self.btn_transcribe.clicked.connect(self.transcribe)
        self.btn_record.clicked.connect(self.start_recording)
        self.btn_stop_record.clicked.connect(self.stop_recording)

        action_bar = QFrame()
        action_bar.setObjectName("ActionBar")
        action_l = QHBoxLayout(action_bar)
        action_l.setContentsMargins(0, 0, 0, 0)
        action_l.setSpacing(1)
        action_l.addWidget(self.btn_choose, 1)
        action_l.addWidget(self.btn_transcribe, 1)
        action_l.addWidget(self.btn_record, 1)
        action_l.addWidget(self.lbl_rec)
        action_l.addWidget(self.btn_stop_record, 1)

        # ── Transport bar (white): seek + play/pause ──
        self.lbl_time_left = QLabel("0:00")
        self.lbl_time_right = QLabel("-0:00")
        self.lbl_time_left.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.lbl_time_right.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        self.seek = QSlider(Qt.Orientation.Horizontal)
        self.seek.setMinimum(0)
        self.seek.setMaximum(0)
        self.seek.sliderMoved.connect(self._on_seek_moved)

        self.btn_play = QToolButton()
        self.btn_play.setAutoRaise(True)
        self.btn_play.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        self.btn_play.setIconSize(QSize(36, 36))
        self.btn_play.setStyleSheet("color: #111111;")
        self.btn_play.clicked.connect(self.toggle_play)

        transport_bar = QFrame()
        transport_bar.setObjectName("TransportBar")
        transport_l = QVBoxLayout(transport_bar)
        transport_l.setContentsMargins(16, 8, 16, 10)
        transport_l.setSpacing(4)

        seek_row = QHBoxLayout()
        seek_row.setSpacing(10)
        seek_row.addWidget(self.lbl_time_left)
        seek_row.addWidget(self.seek, 1)
        seek_row.addWidget(self.lbl_time_right)
        transport_l.addLayout(seek_row)

        play_row = QHBoxLayout()
        play_row.addStretch(1)
        play_row.addWidget(self.btn_play)
        play_row.addStretch(1)
        transport_l.addLayout(play_row)

        # ── Media player (plays synth wav) ──
        self.player = QMediaPlayer(self)
        self.audio_out = QAudioOutput(self)
        self.player.setAudioOutput(self.audio_out)
        self.audio_out.setVolume(0.9)

        self.player.durationChanged.connect(self._on_duration_changed)
        self.player.positionChanged.connect(self._on_position_changed)
        self.player.playbackStateChanged.connect(self._on_playback_state)

        # ── Root layout ──
        root = QVBoxLayout()
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(top_bar)
        root.addWidget(nav_bar)
        root.addWidget(content, 1)
        root.addWidget(action_bar)
        root.addWidget(transport_bar)

        central = QWidget()
        central.setObjectName("Central")
        central.setLayout(root)
        self.setCentralWidget(central)
        self.statusBar().showMessage("Ready")

        self._set_view(0)
    # ────────────────────────────────────────────────────────────────
    # Mic handling

    def refresh_mics(self) -> None:
        self.mic_combo.clear()
        try:
            import sounddevice as sd
            devices = sd.query_devices()
            hostapis = sd.query_hostapis()

            def hostapi_name(d: dict) -> str:
                try:
                    return str(hostapis[int(d.get("hostapi", 0))]["name"])
                except Exception:
                    return ""

            def hostapi_rank(h: str) -> int:
                hl = (h or "").lower()
                if "wasapi" in hl:
                    return 0
                if "wdm-ks" in hl or "wdm" in hl:
                    return 1
                if "directsound" in hl:
                    return 2
                if "mme" in hl:
                    return 3
                return 4

            def is_useful_input(d: dict) -> bool:
                if int(d.get("max_input_channels", 0)) <= 0:
                    return False
                name = str(d.get("name", "")).strip().lower()
                bad = ["microsoft sound mapper", "primary sound", "mapper", "stereo mix", "loopback"]
                return not any(b in name for b in bad)

            default_in = sd.default.device[0]
            self.mic_combo.addItem(f"Default input (index {default_in})", None)

            candidates = []
            for i, d in enumerate(devices):
                if is_useful_input(d):
                    candidates.append((i, d, hostapi_name(d)))

            best_by_name = {}
            for i, d, h in candidates:
                name = str(d.get("name", f"Device {i}")).strip()
                key = name.lower()
                cur = best_by_name.get(key)
                if cur is None:
                    best_by_name[key] = (i, d, h)
                else:
                    _, _, h2 = cur
                    if hostapi_rank(h) < hostapi_rank(h2):
                        best_by_name[key] = (i, d, h)

            final = list(best_by_name.values())
            final.sort(key=lambda t: (hostapi_rank(t[2]), str(t[1].get("name", "")).lower()))
            for i, d, h in final:
                name = str(d.get("name", f"Device {i}")).strip()
                sr = int(float(d.get("default_samplerate", 0) or 0))
                ch = int(d.get("max_input_channels", 0))
                self.mic_combo.addItem(f"{name}  |  {h}  |  ch={ch} sr={sr}  (idx {i})", i)

            self.mic_combo.setCurrentIndex(0)
        except Exception as e:
            self.mic_combo.addItem(f"(sounddevice error: {e})", None)

    def _selected_device_index(self) -> int | None:
        return self.mic_combo.currentData()

    # ────────────────────────────────────────────────────────────────
    # View switching

    def _set_view(self, idx: int) -> None:
        self.stack.setCurrentIndex(idx)
        buttons = {0: self.btn_show_score, 1: self.btn_show_tab, 2: self.btn_show_waves}
        for i, b in buttons.items():
            b.setChecked(i == idx)

    # ────────────────────────────────────────────────────────────────
    # Audio file selection

    def choose_audio(self) -> None:
        fp, _ = QFileDialog.getOpenFileName(
            self, "Select audio file", "",
            "Audio Files (*.wav *.mp3 *.flac *.ogg *.m4a);;All (*)",
        )
        if fp:
            self.audio_path = fp
            self.statusBar().showMessage(f"Loaded: {fp}")

    # ────────────────────────────────────────────────────────────────
    # Recording

    def start_recording(self) -> None:
        if self._record_worker and self._record_worker.isRunning():
            QMessageBox.information(self, "Record", "Already recording.")
            return

        dev = self._selected_device_index()
        try:
            if dev is not None:
                import sounddevice as sd
                sd.query_devices(int(dev))
        except Exception:
            dev = None

        # Clean up any previous temp recording
        if self._recording_path:
            with suppress(Exception):
                os.remove(self._recording_path)
            self._recording_path = None

        self._record_worker = RecordWorker(dev)
        self._record_worker.elapsed.connect(self._on_record_elapsed)
        self._record_worker.level.connect(self._on_record_level)
        self._record_worker.fail.connect(self.on_fail)
        self._record_worker.start()

        self.btn_record.setEnabled(False)
        self.btn_stop_record.setEnabled(True)
        self.btn_transcribe.setEnabled(False)
        self.lbl_rec.setVisible(True)
        self.level_bar.setVisible(True)
        self.level_bar.setValue(0)
        self.statusBar().showMessage("Recording… play your guitar!")

    def _on_record_elapsed(self, seconds: float) -> None:
        m = int(seconds) // 60
        s = int(seconds) % 60
        self.statusBar().showMessage(f"Recording: {m}:{s:02d}")

    def _on_record_level(self, level: float) -> None:
        self.level_bar.setValue(int(min(1.0, level) * 100))

    def stop_recording(self) -> None:
        if not self._record_worker or not self._record_worker.isRunning():
            QMessageBox.information(self, "Record", "Not currently recording.")
            return

        audio, sr = self._record_worker.get_audio()
        self._record_worker.request_stop()
        self._record_worker.wait(3000)

        self.btn_record.setEnabled(True)
        self.btn_stop_record.setEnabled(False)
        self.btn_transcribe.setEnabled(True)
        self.lbl_rec.setVisible(False)
        self.level_bar.setVisible(False)

        if audio is None or audio.size < sr:
            self.statusBar().showMessage("Recording too short — try again.")
            return

        # Check for silence
        peak = float(np.max(np.abs(audio))) if audio.size > 0 else 0.0
        dur = audio.size / sr
        print(f"[RECORD] {dur:.1f}s peak={peak:.4f}")
        if peak < 0.001:
            QMessageBox.warning(
                self, "Mic Check",
                f"Recording was silent (peak={peak:.5f}).\n\n"
                "Possible causes:\n"
                "• Wrong input device selected — try a different one from the dropdown\n"
                "• Microphone access blocked — check Windows Settings → Privacy → Microphone\n"
                "• Mic muted in Windows sound settings",
            )
            self.statusBar().showMessage("Silent recording — check mic and try again")
            return

        # Save to temp WAV
        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        import soundfile as sf_mod
        sf_mod.write(path, audio, sr)
        self._recording_path = path
        self.audio_path = path
        print(f"[RECORD] saved {dur:.1f}s to {path}")
        self.statusBar().showMessage(f"Recording saved ({dur:.1f}s). Press Transcribe.")

    # ────────────────────────────────────────────────────────────────
    # Transcription

    def transcribe(self) -> None:
        if not self.audio_path:
            QMessageBox.warning(self, "Error", "Select or record audio first.")
            return
        self._run_transcription(self.audio_path)

    def _run_transcription(self, audio_path: str) -> None:
        Path(self._out_dir).mkdir(parents=True, exist_ok=True)
        title = Path(audio_path).stem
        model = self.model_combo.currentData() or "basic_pitch"
        self.worker = Worker(audio_path, self._out_dir, title=title, model=model)
        self.worker.done.connect(self.on_transcribe_done)
        self.worker.fail.connect(self.on_fail)
        self.statusBar().showMessage("Transcribing…")
        self.worker.start()

    def on_transcribe_done(self, raw: list) -> None:
        notes: list[NoteEvent] = []
        for r in raw:
            notes.append(NoteEvent(
                pitch=int(r.get("pitch", 60)),
                onset=float(r["onset_s"]),
                duration=float(r["dur_s"]),
                string=int(r["string"]),
                fret=int(r["fret"]),
                velocity=80,
                chord_id=None,
            ))
        self._last_notes = notes
        self.statusBar().showMessage(f"Transcription done: {len(notes)} notes")

        duration_s = max((n.onset + n.duration for n in notes), default=0.0)
        self.tab.set_notes(notes, duration_s)

        # populate wave view
        try:
            import soundfile as sf
            orig, orig_sr = sf.read(self.audio_path, dtype="float32", always_2d=False)
            if orig.ndim > 1:
                orig = orig.mean(axis=1)
            synth, _ = synth_notes_to_array(notes_to_synthnotes(notes), sr=orig_sr, total_duration_s=duration_s)
            self.waves.set_waveforms(orig=orig, synth=synth, sr=orig_sr, duration_s=duration_s)
        except Exception:
            pass

        if self.render_score_view():
            self.score.set_duration_seconds(duration_s)
            self._set_view(0)
        else:
            self._set_view(2)

        # Clean up temp recording file after transcription
        if self._recording_path:
            with suppress(Exception):
                os.remove(self._recording_path)
            self._recording_path = None

    def render_score_view(self) -> bool:
        xml_path = Path(self._out_dir) / "out.musicxml"
        if not xml_path.exists():
            print(f"MusicXML not found: {xml_path}")
            return False
        try:
            title = Path(self.audio_path).stem if self.audio_path else ""
            self.score.load_musicxml(str(xml_path), title=title)
            return True
        except Exception as e:
            print(f"Error loading MusicXML: {e}")
            return False

    # ────────────────────────────────────────────────────────────────
    # Playback

    def _current_notes(self) -> list[NoteEvent]:
        return self._last_notes

    def play_notes(self) -> None:
        notes = self._current_notes()
        if not notes:
            QMessageBox.information(self, "Notes", "No transcribed notes to play.")
            return

        # Stop any current playback and clean up old synth file
        self.player.stop()
        if self._notes_wav_path:
            with suppress(Exception):
                os.remove(self._notes_wav_path)
            self._notes_wav_path = None

        path = synth_notes_to_wav_path(notes_to_synthnotes(notes), sr=44100, gain=0.25)
        if not path:
            QMessageBox.information(self, "Notes", "Could not synthesize notes.")
            return

        self._notes_wav_path = path
        url = QUrl.fromLocalFile(str(Path(path).resolve()))
        self.player.mediaStatusChanged.connect(self._on_media_loaded)
        self.player.setSource(url)

    def toggle_play(self) -> None:
        state = self.player.playbackState()
        if state == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
        elif state == QMediaPlayer.PlaybackState.PausedState:
            self.player.play()
        else:
            # Stopped — synthesise and play
            self.play_notes()

    def _on_media_loaded(self, status: QMediaPlayer.MediaStatus) -> None:
        if status == QMediaPlayer.MediaStatus.LoadedMedia:
            with suppress(Exception):
                self.player.mediaStatusChanged.disconnect(self._on_media_loaded)
            self.player.play()

    def _on_position_changed(self, pos: int) -> None:
        dur = self.player.duration()
        if not self.seek.isSliderDown():
            self.seek.setValue(pos)
        self.lbl_time_left.setText(_fmt_s(pos / 1000.0))
        remaining = max(0.0, (dur - pos) / 1000.0)
        self.lbl_time_right.setText(f"-{_fmt_s(remaining)}")
        t = pos / 1000.0
        with suppress(Exception):
            self.tab.set_playhead_seconds(t)
        with suppress(Exception):
            self.score.set_playhead_seconds(t)
        with suppress(Exception):
            self.waves.set_playhead_seconds(t)

    def _on_duration_changed(self, dur: int) -> None:
        self.seek.setMaximum(max(0, dur))
        self.lbl_time_right.setText(f"-{_fmt_s(dur / 1000.0)}")

    def _on_playback_state(self, state: QMediaPlayer.PlaybackState) -> None:
        playing = state == QMediaPlayer.PlaybackState.PlayingState
        icon = (
            QStyle.StandardPixmap.SP_MediaPause
            if playing
            else QStyle.StandardPixmap.SP_MediaPlay
        )
        self.btn_play.setIcon(self.style().standardIcon(icon))

    def _on_seek_moved(self, pos: int) -> None:
        self.player.setPosition(pos)

    # ────────────────────────────────────────────────────────────────
    # PDF export

    def save_pdf(self) -> None:
        msg = QMessageBox(self)
        msg.setWindowTitle("Save as PDF")
        msg.setText("Which format would you like to save?")
        btn_score = msg.addButton("Score (Notation)", QMessageBox.ButtonRole.AcceptRole)
        btn_tab   = msg.addButton("Tab View",         QMessageBox.ButtonRole.AcceptRole)
        msg.addButton(QMessageBox.StandardButton.Cancel)
        msg.exec()
        clicked = msg.clickedButton()

        if clicked == btn_score:
            fp, _ = QFileDialog.getSaveFileName(
                self, "Save Score as PDF", "", "PDF Files (*.pdf)"
            )
            if fp:
                self.score.save_pdf(fp)
                self.statusBar().showMessage(f"Saving score PDF: {fp}")

        elif clicked == btn_tab:
            fp, _ = QFileDialog.getSaveFileName(
                self, "Save Tab as PDF", "", "PDF Files (*.pdf)"
            )
            if fp:
                self._save_tab_pdf(fp)

    def _save_tab_pdf(self, file_path: str) -> None:
        try:
            from PyQt6.QtPrintSupport import QPrinter
            from PyQt6.QtGui import QPainter
            printer = QPrinter(QPrinter.PrinterMode.HighResolution)
            printer.setOutputFormat(QPrinter.OutputFormat.PdfFormat)
            printer.setOutputFileName(file_path)
            pixmap = self.tab.grab()
            painter = QPainter(printer)
            target = painter.viewport()
            size = pixmap.size()
            size.scale(target.size(), Qt.AspectRatioMode.KeepAspectRatio)
            painter.setViewport(target.x(), target.y(), size.width(), size.height())
            painter.setWindow(pixmap.rect())
            painter.drawPixmap(0, 0, pixmap)
            painter.end()
            self.statusBar().showMessage(f"Tab PDF saved: {file_path}")
        except Exception as e:
            QMessageBox.warning(self, "PDF Error", f"Could not save tab PDF:\n{e}")

    # ────────────────────────────────────────────────────────────────
    # Error handling

    def on_fail(self, msg: str) -> None:
        QMessageBox.critical(self, "Error", msg)

    # ────────────────────────────────────────────────────────────────
    # Close event

    def closeEvent(self, event) -> None:
        if self._record_worker and self._record_worker.isRunning():
            self._record_worker.request_stop()
            self._record_worker.wait(1500)
        self.player.stop()
        if self._notes_wav_path:
            with suppress(Exception):
                os.remove(self._notes_wav_path)
        super().closeEvent(event)
