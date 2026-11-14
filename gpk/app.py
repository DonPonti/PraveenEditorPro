#!/usr/bin/env python3
"""
Workout Vlog Editor: desktop app (manual + automatic editing)

    python app.py

Workflow
    1. Media tab: add your clips/photos (files or a whole folder), pick your music (Settings tab).
    2. Click "Auto-edit" to let the engine build a beat-synced edit... or build one by hand by
       adding clips from the bin to the timeline.
    3. Click any clip on the timeline to fine-tune it in the inspector (trim, duration, speed,
       camera move, flash, glitch, whip, speed ramp, dips). Drag clips to reorder.
    4. "Preview" renders a fast half-res version and plays it. "Export" renders the final video.
    5. Save the project (.json) any time and reopen it later.
"""

import copy
import json
import os
import shutil
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

# A windowed (no-console) exe has no stdout/stderr; some libraries write to them and would crash.
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w")
os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(__import__("tempfile").gettempdir()) / "praveeneditor_numba"))

import numpy as np
from PySide6.QtCore import Qt, QThread, Signal, QUrl, QTimer, QRect
from PySide6.QtGui import QPainter, QColor, QPixmap, QImage, QKeySequence, QAction, QPen, QIcon
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QListWidget,
    QListWidgetItem, QPushButton, QLabel, QFileDialog, QMessageBox, QDoubleSpinBox, QSpinBox,
    QComboBox, QCheckBox, QLineEdit, QFormLayout, QGroupBox, QTabWidget, QScrollArea,
    QProgressBar, QStackedWidget, QToolBar, QSizePolicy, QAbstractItemView, QSlider,
    QPlainTextEdit,
)

import vlog_core as core

# Set VLOG_NO_MM=1 to disable the built-in video player (thumbnail previews only)
# if your system's Qt media backend ever misbehaves.
HAS_MM = False
if os.environ.get("VLOG_NO_MM") != "1":
    try:
        from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
        from PySide6.QtMultimediaWidgets import QVideoWidget
        HAS_MM = True
    except Exception:                   # pragma: no cover
        HAS_MM = False

DARK = """
QWidget { background:#1b1d23; color:#e6e8ee; font-size:13px; }
QMainWindow, QDialog { background:#15171c; }
QGroupBox { border:1px solid #2c3039; border-radius:6px; margin-top:10px; padding-top:8px; }
QGroupBox::title { subcontrol-origin: margin; left:10px; color:#8e95a5; }
QPushButton { background:#2a2e38; border:1px solid #3a3f4c; border-radius:5px; padding:6px 12px; }
QPushButton:hover { background:#343947; }
QPushButton:disabled { color:#666b78; background:#22252d; }
QPushButton#primary { background:#3b6df0; border-color:#5b87ff; font-weight:bold; }
QPushButton#primary:hover { background:#4d7bff; }
QPushButton#danger { background:#5a2a2f; border-color:#7a3a41; }
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QListWidget {
    background:#22252d; border:1px solid #343947; border-radius:4px; padding:3px; }
QListWidget::item { padding:4px; }
QListWidget::item:hover { background:#2a2e38; }
QListWidget::item:selected { background:#3b6df0; color:#ffffff; }
QScrollArea { border:none; }
QTabWidget::pane { border:1px solid #2c3039; }
QTabBar::tab { background:#22252d; padding:6px 14px; border:1px solid #2c3039; }
QTabBar::tab:selected { background:#2f3440; }
QToolBar { background:#15171c; border-bottom:1px solid #2c3039; spacing:6px; padding:4px; }
QProgressBar { border:1px solid #343947; border-radius:4px; text-align:center; }
QProgressBar::chunk { background:#3b6df0; }
"""


REFRAME_CHOICES = [
    ("Smart: follow the subject (recommended)", "smart"),
    ("Fit whole frame (blurred background)", "fit"),
    ("Fill: crop to vertical", "fill"),
]


def to_pixmap(bgr):
    if bgr is None:
        return QPixmap()
    bgr = np.ascontiguousarray(bgr)
    h, w = bgr.shape[:2]
    img = QImage(bgr.data, w, h, 3 * w, QImage.Format.Format_BGR888).copy()
    return QPixmap.fromImage(img)


class ThumbCache:
    def __init__(self):
        self.cache = {}

    def get(self, source, start=0.0, width=240):
        key = (source, round(start, 1), width)
        if key not in self.cache:
            self.cache[key] = to_pixmap(core.thumbnail(source, start, width))
        return self.cache[key]


class Worker(QThread):
    progress = Signal(str, float)
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def run(self):
        try:
            res = self.fn(lambda msg, frac=None: self.progress.emit(msg, -1.0 if frac is None else frac))
            self.done.emit(res)
        except core.EditorError as e:
            self.failed.emit(str(e))
        except Exception as e:                       # unexpected: still show it
            self.failed.emit(f"{type(e).__name__}: {e}")


# ----------------------------------------------------------------------------
# Timeline widget
# ----------------------------------------------------------------------------
class Timeline(QWidget):
    selected = Signal(int)
    moved = Signal(int, int)
    COLORS = ["#4a90e2", "#e2a04a", "#e2504a", "#5ac48a"]

    def __init__(self, thumbs):
        super().__init__()
        self.thumbs = thumbs
        self.items = []
        self.sel = -1
        self.pxps = 70
        self.press = None
        self.setMinimumHeight(150)
        self.setMouseTracking(True)

    def set_items(self, items, sel):
        self.items, self.sel = items, sel
        total = sum(max(34, int(i.duration * self.pxps)) for i in items)
        self.setMinimumWidth(max(500, total + 40))
        self.update()

    def rects(self):
        x, out = 10, []
        for it in self.items:
            w = max(34, int(it.duration * self.pxps))
            out.append(QRect(x, 28, w - 3, 90))
            x += w
        return out

    def hit(self, x):
        rs = self.rects()
        for i, r in enumerate(rs):
            if r.left() <= x <= r.right() + 3:
                return i
        if rs and x > rs[-1].right():
            return len(rs) - 1
        return -1 if not rs else 0

    def paintEvent(self, _):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#12141a"))
        if not self.items:
            p.setPen(QColor("#6a7080"))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                       "Timeline is empty: click Auto-edit, or add clips from the Media tab")
            return
        t = 0.0
        for i, (it, r) in enumerate(zip(self.items, self.rects())):
            p.fillRect(QRect(r.x(), 14, r.width(), 8), QColor(self.COLORS[it.section % 4]))
            pm = self.thumbs.get(it.source, it.src_start)
            p.save()
            p.setClipRect(r)
            if pm.isNull():
                p.fillRect(r, QColor("#2a2d35"))
            else:
                s = pm.scaled(r.size(), Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                              Qt.TransformationMode.SmoothTransformation)
                p.drawPixmap(r.x() + (r.width() - s.width()) // 2,
                             r.y() + (r.height() - s.height()) // 2, s)
            p.restore()
            p.fillRect(QRect(r.x(), r.bottom() - 21, r.width(), 22), QColor(0, 0, 0, 165))
            badges = ("★" if it.hero else "") + ("F" if it.flash else "") + ("G" if it.glitch else "") \
                + ("W" if it.whip else "") + ("R" if it.ramp else "") \
                + ("·" + it.move[:2] if it.move else "")
            p.setPen(QColor("#ffffff"))
            p.drawText(QRect(r.x() + 4, r.bottom() - 21, r.width() - 6, 22),
                       Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                       f"{it.duration:.1f}s {badges}")
            p.setPen(QColor("#7a8090"))
            p.drawText(r.x(), r.bottom() + 16, f"{t:.1f}")
            if i == self.sel:
                p.setPen(QPen(QColor("#ffffff"), 3))
                p.drawRect(r.adjusted(1, 1, -1, -1))
            t += it.duration
        if self.press and self.press[2]:
            p.setPen(QPen(QColor("#3b6df0"), 3))
            tgt = self.hit(self.press[1])
            rs = self.rects()
            if 0 <= tgt < len(rs):
                x = rs[tgt].left() if tgt <= self.press[0] else rs[tgt].right() + 3
                p.drawLine(x, 10, x, 128)

    def mousePressEvent(self, e):
        i = self.hit(int(e.position().x()))
        if i >= 0 and self.items:
            self.selected.emit(i)
            self.press = [i, int(e.position().x()), False]

    def mouseMoveEvent(self, e):
        if self.press:
            self.press[1] = int(e.position().x())
            self.press[2] = abs(self.press[1] - self.rects()[self.press[0]].center().x()) > 18
            self.update()

    def mouseReleaseEvent(self, e):
        if self.press and self.press[2]:
            src, tgt = self.press[0], self.hit(int(e.position().x()))
            if tgt >= 0 and tgt != src:
                self.moved.emit(src, tgt)
        self.press = None
        self.update()


# ----------------------------------------------------------------------------
# Preview panel
# ----------------------------------------------------------------------------
class PreviewPanel(QWidget):
    def __init__(self):
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.stack = QStackedWidget()
        self.label = QLabel("Select a clip to preview it")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setMinimumSize(200, 240)
        self.label.setStyleSheet("background:#0c0d10;")
        self.stack.addWidget(self.label)
        self.player = None
        self.seg = None
        self.pending = None
        if HAS_MM:
            try:
                self.video = QVideoWidget()
                self.player = QMediaPlayer()
                self.audio = QAudioOutput()
                self.player.setAudioOutput(self.audio)
                self.player.setVideoOutput(self.video)
                self.stack.addWidget(self.video)
                self.player.mediaStatusChanged.connect(self._status)
                self.timer = QTimer(self)
                self.timer.setInterval(40)
                self.timer.timeout.connect(self._tick)
                self.timer.start()
            except Exception:
                self.player = None
        lay.addWidget(self.stack, 1)
        row = QHBoxLayout()
        self.btn = QPushButton("Play / Pause")
        self.btn.clicked.connect(self.toggle)
        self.info = QLabel("")
        self.info.setStyleSheet("color:#8e95a5;")
        row.addWidget(self.btn)
        row.addWidget(self.info, 1)
        lay.addLayout(row)

    def _status(self, st):
        if self.pending is not None and st in (QMediaPlayer.MediaStatus.LoadedMedia,
                                               QMediaPlayer.MediaStatus.BufferedMedia):
            self.player.setPosition(self.pending)
            self.pending = None
            self.player.play()

    def _tick(self):
        if self.seg and self.player and \
                self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            if self.player.position() >= self.seg[1]:
                self.player.setPosition(self.seg[0])

    def toggle(self):
        if not self.player:
            return
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
        else:
            self.player.play()

    def show_pixmap(self, pm, text=""):
        if self.player:
            self.player.pause()
        self.seg = None
        if pm and not pm.isNull():
            self.label.setPixmap(pm.scaled(self.label.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                           Qt.TransformationMode.SmoothTransformation))
        else:
            self.label.setText(text or "No preview")
        self.stack.setCurrentIndex(0)

    def _load(self, path, start_ms, muted):
        self.stack.setCurrentIndex(1)
        self.audio.setMuted(muted)
        url = QUrl.fromLocalFile(str(path))
        if self.player.source() == url:
            self.player.setPosition(start_ms)
            self.player.play()
        else:
            self.pending = start_ms
            self.player.setSource(url)

    def play_segment(self, path, start, end, fallback_pm=None):
        if not self.player:
            self.show_pixmap(fallback_pm, "Video preview unavailable")
            return
        self.seg = (int(start * 1000), int(end * 1000))
        self._load(path, int(start * 1000), True)

    def play_file(self, path):
        if not self.player:
            self.show_pixmap(None, f"Saved: {path}")
            return
        self.seg = None
        self._load(path, 0, False)


# ----------------------------------------------------------------------------
# Main window
# ----------------------------------------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Praveen Editor v1.4 Audio Studio")
        self.resize(1200, 720)
        self.project = core.Project()
        self.media = []
        self.sel = -1
        self.thumbs = ThumbCache()
        self.history, self.redo = [], []
        self._last_snap = 0.0
        self._loading = False
        self.worker = None
        self.project_path = None
        self._build_ui()
        self.apply_settings()
        self.refresh()

    # ---------------- UI construction
    def _build_ui(self):
        tb = QToolBar()
        tb.setMovable(False)
        self.addToolBar(tb)

        def button(text, fn, shortcut=None, primary=False, tip=""):
            b = QPushButton(text)
            if primary:
                b.setObjectName("primary")
            if shortcut:
                b.setShortcut(QKeySequence(shortcut))
            if tip:
                b.setToolTip(tip)
            b.clicked.connect(fn)
            tb.addWidget(b)
            return b

        button("New", self.new_project, "Ctrl+N")
        button("Open", self.open_project, "Ctrl+O")
        button("Save", self.save_project, "Ctrl+S")
        tb.addSeparator()
        self.btn_undo = button("Undo", self.undo, "Ctrl+Z")
        self.btn_redo = button("Redo", self.do_redo, "Ctrl+Y")
        tb.addSeparator()
        self.btn_auto = button("Auto-edit", self.auto_edit, primary=True,
                               tip="Build a beat-synced edit from your clips and music")
        self.btn_prev = button("Preview", lambda: self.render(True),
                               tip="Fast half-resolution render, plays in the app")
        self.btn_export = button("Export", lambda: self.render(False), primary=True,
                                 tip="Render the final video")

        # --- left: tabs
        self.tabs = QTabWidget()
        self.tabs.addTab(self._scroll(self._media_tab()), "Media")
        self.tabs.addTab(self._scroll(self._settings_tab()), "Settings")
        self.tabs.addTab(self._scroll(self._ai_tab()), "AI Creator")
        self.tabs.addTab(self._scroll(self._captions_tab()), "Captions")
        self.tabs.addTab(self._scroll(self._audio_tab()), "Audio Mix")

        # --- centre: preview
        self.preview = PreviewPanel()

        # --- right: inspector
        right = self._scroll(self._inspector())

        top = QSplitter(Qt.Orientation.Horizontal)
        top.addWidget(self.tabs)
        top.addWidget(self.preview)
        top.addWidget(right)
        top.setSizes([340, 560, 340])

        # --- bottom: timeline
        self.timeline = Timeline(self.thumbs)
        self.timeline.selected.connect(self.select)
        self.timeline.moved.connect(self.move_item)
        self.scroll = QScrollArea()
        self.scroll.setWidget(self.timeline)
        self.scroll.setWidgetResizable(True)
        self.scroll.setMinimumHeight(175)
        self.total_label = QLabel("")
        zoom_row = QHBoxLayout()
        zoom_row.addWidget(self.total_label, 1)
        zoom_row.addWidget(QLabel("Zoom"))
        self.zoom = QSpinBox()
        self.zoom.setRange(20, 300)
        self.zoom.setValue(70)
        self.zoom.setSuffix(" px/s")
        self.zoom.valueChanged.connect(self._zoom)
        zoom_row.addWidget(self.zoom)
        bottom = QWidget()
        bl = QVBoxLayout(bottom)
        bl.setContentsMargins(6, 4, 6, 4)
        bl.addLayout(zoom_row)
        bl.addWidget(self.scroll)

        split = QSplitter(Qt.Orientation.Vertical)
        split.addWidget(top)
        split.addWidget(bottom)
        split.setSizes([600, 240])
        self.setCentralWidget(split)

        self.status = QLabel("Ready")
        self.bar = QProgressBar()
        self.bar.setMaximumWidth(260)
        self.bar.setVisible(False)
        self.statusBar().addWidget(self.status, 1)
        self.statusBar().addPermanentWidget(self.bar)

        self.pv_timer = QTimer(self)
        self.pv_timer.setSingleShot(True)
        self.pv_timer.timeout.connect(self.preview_item)

    @staticmethod
    def _scroll(widget):
        sa = QScrollArea()
        sa.setWidgetResizable(True)
        sa.setFrameShape(QScrollArea.Shape.NoFrame)
        sa.setWidget(widget)
        return sa

    def _media_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        self.checklist = QLabel("")
        self.checklist.setWordWrap(True)
        self.checklist.setStyleSheet("color:#9fb4ff; padding:2px;")
        lay.addWidget(self.checklist)
        hint = QLabel("Click a file to select it (Ctrl/Shift for several). "
                      "Double-click to add it to the timeline.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#8e95a5;")
        lay.addWidget(hint)
        self.bin = QListWidget()
        self.bin.setIconSize(QPixmap(64, 64).size())
        self.bin.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.bin.setMinimumHeight(160)
        self.bin.itemDoubleClicked.connect(lambda _: self.add_to_timeline())
        self.bin.itemClicked.connect(self._bin_clicked)
        self.bin.itemSelectionChanged.connect(self.update_checklist)
        lay.addWidget(self.bin, 1)
        row = QHBoxLayout()
        b1 = QPushButton("Add files…")
        b1.clicked.connect(self.pick_files)
        b2 = QPushButton("Add folder…")
        b2.clicked.connect(self.pick_folder)
        row.addWidget(b1)
        row.addWidget(b2)
        lay.addLayout(row)
        b3 = QPushButton("Add selected to timeline")
        b3.clicked.connect(self.add_to_timeline)
        lay.addWidget(b3)
        b4 = QPushButton("Remove from bin")
        b4.clicked.connect(self.remove_from_bin)
        lay.addWidget(b4)
        b5 = QPushButton("Auto-edit now")
        b5.setObjectName("primary")
        b5.clicked.connect(self.auto_edit)
        lay.addWidget(b5)
        return w

    def _bin_clicked(self, item):
        path = item.data(Qt.ItemDataRole.UserRole)
        self.preview.show_pixmap(self.thumbs.get(path, 0.0, 480))
        self.preview.info.setText(Path(path).name)

    def update_checklist(self):
        sel = len(self.bin.selectedItems()) if hasattr(self, "bin") else 0
        music = Path(self.project.music).name if self.project.music else "NOT CHOSEN (Settings tab)"
        fmt = self.cb_format.currentData() if hasattr(self, "cb_format") else self.project.format
        out = "VERTICAL 1080×1920 (Instagram)" if fmt == "vertical" else "Horizontal 1920×1080"
        self.checklist.setText(f"Output: {out}\n"
                               f"Clips in bin: {len(self.media)}  ·  selected: {sel}\n"
                               f"Music: {music}\nTimeline: {len(self.project.items)} clips")

    def _settings_tab(self):
        w = QWidget()
        outer = QVBoxLayout(w)

        g1 = QGroupBox("Music")
        f1 = QFormLayout(g1)
        self.music_label = QLabel("(none)")
        self.music_label.setWordWrap(True)
        bm = QPushButton("Choose music…")
        bm.clicked.connect(self.pick_music)
        self.sp_music_start = QDoubleSpinBox()
        self.sp_music_start.setRange(0, 3600)
        self.sp_music_start.setSuffix(" s")
        self.sp_music_start.setDecimals(2)
        f1.addRow(self.music_label)
        f1.addRow(bm)
        f1.addRow("Start song at", self.sp_music_start)
        outer.addWidget(g1)

        g2 = QGroupBox("Look")
        f2 = QFormLayout(g2)
        self.cb_style = QComboBox()
        self.cb_style.addItems(list(core.STYLES))
        self.cb_effects = QComboBox()
        self.cb_effects.addItems(["hype", "subtle", "off"])
        self.cb_format = QComboBox()
        self.cb_format.addItem("Vertical 9:16 (Instagram Reels)", "vertical")
        self.cb_format.addItem("Horizontal 16:9", "horizontal")
        self.cb_reframe = QComboBox()
        for text, key in REFRAME_CHOICES:
            self.cb_reframe.addItem(text, key)
        self.sp_move = QSpinBox()
        self.sp_move.setRange(0, 150)
        self.sp_move.setValue(100)
        self.sp_move.setSuffix(" %")
        self.sp_move.setToolTip("How much the camera moves/zooms. 0 = no zoom at all.")
        self.ck_sfx = QCheckBox("Sound effects (riser / impact / whoosh)")
        self.ck_sfx.setChecked(True)
        self.ed_lut = QLineEdit()
        self.ed_lut.setPlaceholderText("optional .cube LUT file")
        bl = QPushButton("Browse LUT…")
        bl.clicked.connect(self.pick_lut)
        f2.addRow("Style", self.cb_style)
        f2.addRow("Effects", self.cb_effects)
        f2.addRow("Output", self.cb_format)
        f2.addRow("Landscape clips", self.cb_reframe)
        f2.addRow("Camera zoom strength", self.sp_move)
        f2.addRow(self.ck_sfx)
        f2.addRow(self.ed_lut)
        f2.addRow(bl)
        outer.addWidget(g2)

        g3 = QGroupBox("Text")
        f3 = QFormLayout(g3)
        self.ed_title = QLineEdit()
        self.ed_sub = QLineEdit()
        self.ed_outro = QLineEdit()
        f3.addRow("Title", self.ed_title)
        f3.addRow("Subtitle", self.ed_sub)
        f3.addRow("Outro", self.ed_outro)
        outer.addWidget(g3)

        g4 = QGroupBox("Auto-edit")
        f4 = QFormLayout(g4)
        self.sp_length = QSpinBox()
        self.sp_length.setRange(5, 180)
        self.sp_length.setValue(30)
        self.sp_length.setSuffix(" s")
        self.sp_seed = QSpinBox()
        self.sp_seed.setRange(0, 99999)
        self.sp_seed.setSpecialValueText("random")
        ba = QPushButton("Auto-edit now")
        ba.setObjectName("primary")
        ba.clicked.connect(self.auto_edit)
        f4.addRow("Target length", self.sp_length)
        f4.addRow("Seed", self.sp_seed)
        f4.addRow(ba)
        outer.addWidget(g4)
        outer.addStretch(1)
        return w

    def _ai_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        intro = QLabel(
            "Gemini Creator Studio\nGenerate hooks, titles, captions, hashtags, a filming shot list "
            "and editing notes. Your footage stays on your PC."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("color:#aabaff; padding:6px;")
        lay.addWidget(intro)
        form = QFormLayout()
        self.ai_topic = QLineEdit()
        self.ai_topic.setPlaceholderText("e.g. Push-day workout, morning routine in Warangal…")
        self.ai_type = QComboBox()
        self.ai_type.addItems(["Workout vlog", "Gym transformation", "Daily vlog",
                               "Travel vlog", "Fitness tutorial", "Lifestyle video",
                               "YouTube video", "Productivity / study vlog"])
        self.ai_tone = QComboBox()
        self.ai_tone.addItems(["Energetic", "Cinematic", "Calm and aesthetic",
                               "Funny and casual", "Documentary", "Professional"])
        self.ai_key = QLineEdit()
        self.ai_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.ai_key.setPlaceholderText("Optional: paste Gemini API key (not saved)")
        form.addRow("Video idea", self.ai_topic)
        form.addRow("Content type", self.ai_type)
        form.addRow("Tone", self.ai_tone)
        form.addRow("Gemini API key", self.ai_key)
        lay.addLayout(form)
        row = QHBoxLayout()
        self.btn_ai_generate = QPushButton("Generate creator pack")
        self.btn_ai_generate.setObjectName("primary")
        self.btn_ai_generate.clicked.connect(self.generate_ai_pack)
        self.btn_ai_apply = QPushButton("Apply title / hook")
        self.btn_ai_apply.clicked.connect(self.apply_ai_pack)
        self.btn_ai_copy = QPushButton("Copy all")
        self.btn_ai_copy.clicked.connect(self.copy_ai_pack)
        row.addWidget(self.btn_ai_generate)
        row.addWidget(self.btn_ai_apply)
        row.addWidget(self.btn_ai_copy)
        lay.addLayout(row)
        self.ai_output = QPlainTextEdit()
        self.ai_output.setReadOnly(True)
        self.ai_output.setPlaceholderText("Your AI creator pack will appear here…")
        self.ai_output.setMinimumHeight(320)
        lay.addWidget(self.ai_output, 1)
        note = QLabel(
            "Setup: pip install -U google-genai. Create an API key at "
            "https://aistudio.google.com/app/apikey. The key is used for this session only. "
            "Gemini app subscriptions do not automatically provide Gemini API quota."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color:#8e95a5;")
        lay.addWidget(note)
        self.ai_pack = None
        return w

    def generate_ai_pack(self):
        topic = self.ai_topic.text().strip()
        if not topic:
            self.error("Describe the video first.")
            return
        self.collect_settings()
        fmt = self.project.format
        content_type = self.ai_type.currentText()
        tone = self.ai_tone.currentText()
        key = self.ai_key.text().strip() or None
        self.ai_output.setPlainText("Gemini is preparing your creator pack…")
        def job(cb):
            cb("Generating creator pack with Gemini", 0.25)
            result = core.gemini_creator_pack(topic, fmt, content_type, tone, key)
            cb("Creator pack ready", 1.0)
            return result
        self._start(job, self._ai_done)

    def _ai_done(self, data):
        self.ai_pack = data
        self.ai_output.setPlainText(json.dumps(data, ensure_ascii=False, indent=2))
        self.say("Gemini creator pack ready. Review it before applying.")

    def apply_ai_pack(self):
        if not self.ai_pack:
            self.say("Generate a creator pack first.")
            return
        self.ed_title.setText(str(self.ai_pack.get("title", "")))
        self.ed_sub.setText(str(self.ai_pack.get("hook", self.ai_pack.get("subtitle", ""))))
        self.ed_outro.setText(str(self.ai_pack.get("outro", "")))
        self.collect_settings()
        self.say("AI title, hook and outro applied to project settings.")

    def copy_ai_pack(self):
        value = self.ai_output.toPlainText().strip()
        if value:
            QApplication.clipboard().setText(value)
            self.say("Creator pack copied to clipboard.")

    def _audio_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        intro = QLabel(
            "Audio Mix Studio\nKeep camera sound, add a separate narration track, "
            "balance music, and automatically lower the music under speech."
        )
        intro.setWordWrap(True)
        lay.addWidget(intro)

        music = QGroupBox("Background music")
        mf = QFormLayout(music)
        self.audio_music_gain = QDoubleSpinBox()
        self.audio_music_gain.setRange(0.0, 2.0)
        self.audio_music_gain.setSingleStep(0.05)
        self.audio_music_gain.setDecimals(2)
        self.audio_music_gain.setSuffix(" ×")
        mf.addRow("Music volume", self.audio_music_gain)
        lay.addWidget(music)

        source = QGroupBox("Original clip audio")
        sf = QFormLayout(source)
        self.ck_clip_audio = QCheckBox("Include audio recorded by the video clips")
        self.audio_clip_gain = QDoubleSpinBox()
        self.audio_clip_gain.setRange(0.0, 2.0)
        self.audio_clip_gain.setSingleStep(0.05)
        self.audio_clip_gain.setDecimals(2)
        self.audio_clip_gain.setSuffix(" ×")
        sf.addRow(self.ck_clip_audio)
        sf.addRow("Clip audio volume", self.audio_clip_gain)
        lay.addWidget(source)

        voice = QGroupBox("Voice-over / narration")
        vf = QFormLayout(voice)
        self.voiceover_label = QLabel("No voice-over selected")
        self.voiceover_label.setWordWrap(True)
        choose = QPushButton("Choose voice-over audio…")
        choose.clicked.connect(self.pick_voiceover)
        clear = QPushButton("Remove voice-over")
        clear.clicked.connect(self.clear_voiceover)
        buttons = QHBoxLayout()
        buttons.addWidget(choose)
        buttons.addWidget(clear)
        self.audio_voice_offset = QDoubleSpinBox()
        self.audio_voice_offset.setRange(0.0, 3600.0)
        self.audio_voice_offset.setDecimals(2)
        self.audio_voice_offset.setSuffix(" s")
        self.audio_voice_gain = QDoubleSpinBox()
        self.audio_voice_gain.setRange(0.0, 2.0)
        self.audio_voice_gain.setSingleStep(0.05)
        self.audio_voice_gain.setDecimals(2)
        self.audio_voice_gain.setSuffix(" ×")
        vf.addRow(self.voiceover_label)
        vf.addRow(buttons)
        vf.addRow("Start at timeline time", self.audio_voice_offset)
        vf.addRow("Voice-over volume", self.audio_voice_gain)
        lay.addWidget(voice)

        mix = QGroupBox("Automatic mixing")
        xf = QFormLayout(mix)
        self.ck_duck_music = QCheckBox("Lower background music while speech is playing")
        self.ck_duck_music.setChecked(True)
        self.ck_duck_music.setToolTip("Uses dialogue and voice-over audio to drive automatic music ducking.")
        xf.addRow(self.ck_duck_music)
        lay.addWidget(mix)

        note = QLabel(
            "Tip: add a voice-over file (WAV, MP3, M4A, AAC, FLAC or OGG), "
            "set its timeline start time, then export. Audio is mixed locally by FFmpeg. "
            "The current voice-over track is one continuous file; multiple independent "
            "voice-over clips and keyframe volume automation are planned for a later version."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color:#9aa2b2; padding:6px;")
        lay.addWidget(note)
        lay.addStretch(1)
        return w

    def pick_voiceover(self):
        f, _ = QFileDialog.getOpenFileName(
            self, "Choose voice-over", "",
            "Audio (*.wav *.mp3 *.m4a *.aac *.flac *.ogg *.mp4 *.mov *.mkv)"
        )
        if f:
            self.project.voiceover = f
            self.voiceover_label.setText(Path(f).name)
            self.say("Voice-over selected")

    def clear_voiceover(self):
        self.project.voiceover = ""
        self.voiceover_label.setText("No voice-over selected")
        self.say("Voice-over removed")

    def _captions_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        intro = QLabel(
            "Automatic captions\nTranscribe dialogue from your timeline locally with Whisper. "
            "Your video and audio are processed on this computer, not uploaded to Gemini."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("color:#aabaff; padding:6px;")
        lay.addWidget(intro)

        form = QFormLayout()
        self.caption_model = QComboBox()
        self.caption_model.addItem("Tiny — fastest / lowest resource use", "tiny")
        self.caption_model.addItem("Base — better accuracy", "base")
        self.caption_model.addItem("Small — higher accuracy, slower", "small")
        self.caption_style = QComboBox()
        self.caption_style.addItem("Bold creator (yellow text + dark box)", "bold")
        self.caption_style.addItem("Clean (white text + subtle box)", "clean")
        self.caption_style.addItem("Minimal (white text + outline)", "minimal")
        form.addRow("Whisper model", self.caption_model)
        form.addRow("Caption appearance", self.caption_style)
        lay.addLayout(form)

        row = QHBoxLayout()
        self.btn_caption_generate = QPushButton("Generate captions")
        self.btn_caption_generate.setObjectName("primary")
        self.btn_caption_generate.clicked.connect(self.generate_captions)
        self.btn_caption_clear = QPushButton("Clear captions")
        self.btn_caption_clear.clicked.connect(self.clear_captions)
        row.addWidget(self.btn_caption_generate)
        row.addWidget(self.btn_caption_clear)
        lay.addLayout(row)

        self.caption_output = QPlainTextEdit()
        self.caption_output.setReadOnly(True)
        self.caption_output.setPlaceholderText("Generated timed captions will appear here…")
        lay.addWidget(self.caption_output, 1)
        note = QLabel(
            "One-time setup: python -m pip install faster-whisper\n"
            "The first generation downloads the selected Whisper model. Tiny is recommended "
            "for older or lower-memory PCs. Captions are saved inside the project JSON and "
            "burned into exported videos."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color:#8e95a5;")
        lay.addWidget(note)
        return w

    def generate_captions(self):
        if self.worker and self.worker.isRunning():
            return
        if not self.project.items:
            self.error("Add video clips to the timeline first.")
            return
        self.snapshot()
        self.collect_settings()
        self.project.caption_style = self.caption_style.currentData()
        proj = copy.deepcopy(self.project)
        model_size = self.caption_model.currentData()
        self.caption_output.setPlainText(
            "Preparing local Whisper transcription…\n"
            "The first run may take a few minutes while the model downloads."
        )
        self._start(
            lambda cb: core.transcribe_project(proj, model_size=model_size, cb=cb),
            self._captions_done,
        )

    def _captions_done(self, cues):
        self.project.captions = cues
        self.project.caption_style = self.caption_style.currentData()
        lines = [f"[{c['start']:06.2f}s – {c['end']:06.2f}s]  {c['text']}" for c in cues]
        self.caption_output.setPlainText("\n".join(lines))
        self.say(f"Generated {len(cues)} timed captions. Preview or export to see them.")

    def clear_captions(self):
        self.snapshot()
        self.project.captions = []
        if hasattr(self, "caption_output"):
            self.caption_output.clear()
        self.say("Captions cleared from this project.")

    def _inspector(self):
        box = QGroupBox("Clip")
        lay = QVBoxLayout(box)
        self.lb_src = QLabel("No clip selected")
        self.lb_src.setWordWrap(True)
        lay.addWidget(self.lb_src)
        form = QFormLayout()
        self.sp_start = QDoubleSpinBox()
        self.sp_start.setRange(0, 36000)
        self.sp_start.setSingleStep(0.1)
        self.sp_start.setSuffix(" s")
        self.sp_dur = QDoubleSpinBox()
        self.sp_dur.setRange(0.2, 120)
        self.sp_dur.setSingleStep(0.1)
        self.sp_dur.setSuffix(" s")
        self.sp_speed = QDoubleSpinBox()
        self.sp_speed.setRange(0.2, 3.0)
        self.sp_speed.setSingleStep(0.05)
        self.sp_speed.setSuffix("×")
        self.cb_move = QComboBox()
        self.cb_move.addItems(core.MOVE_CHOICES)
        self.cb_section = QComboBox()
        self.cb_section.addItems(core.SECTION_NAMES)
        form.addRow("Start in source", self.sp_start)
        form.addRow("Duration", self.sp_dur)
        form.addRow("Speed", self.sp_speed)
        form.addRow("Camera move", self.cb_move)
        form.addRow("Section tag", self.cb_section)
        self.cb_item_reframe = QComboBox()
        self.cb_item_reframe.addItem("Project default", "")
        for text, key in REFRAME_CHOICES:
            self.cb_item_reframe.addItem(text.split(":")[0].split(" (")[0], key)
        self.sl_focus = QSlider(Qt.Orientation.Horizontal)
        self.sl_focus.setRange(0, 100)
        self.sl_focus.setValue(50)
        self.lb_focus = QLabel("50%")
        frow = QHBoxLayout()
        frow.addWidget(self.sl_focus, 1)
        frow.addWidget(self.lb_focus)
        form.addRow("Framing", self.cb_item_reframe)
        form.addRow("Subject position", frow)
        lay.addLayout(form)
        self.btn_clip_prev = QPushButton("▶ Preview this clip (with effects)")
        self.btn_clip_prev.clicked.connect(self.preview_clip)
        lay.addWidget(self.btn_clip_prev)
        self.checks = {}
        for key, text in [("hero", "★ Drop shot (impact + riser sound)"), ("flash", "Flash in"),
                          ("glitch", "Glitch hit"), ("whip", "Whip blur"),
                          ("ramp", "Speed ramp (slow → fast)"), ("dip_in", "Dip in from black"),
                          ("dip_out", "Dip out to black")]:
            c = QCheckBox(text)
            self.checks[key] = c
            lay.addWidget(c)
            c.toggled.connect(self._inspector_changed)
        for wdg in (self.sp_start, self.sp_dur, self.sp_speed):
            wdg.valueChanged.connect(self._inspector_changed)
        self.cb_move.currentIndexChanged.connect(self._inspector_changed)
        self.cb_section.currentIndexChanged.connect(self._inspector_changed)
        self.cb_item_reframe.currentIndexChanged.connect(self._inspector_changed)
        self.sl_focus.valueChanged.connect(self._inspector_changed)

        r1 = QHBoxLayout()
        for text, fn in [("◀ Move", lambda: self.shift_item(-1)), ("Move ▶", lambda: self.shift_item(1))]:
            b = QPushButton(text)
            b.clicked.connect(fn)
            r1.addWidget(b)
        lay.addLayout(r1)
        r2 = QHBoxLayout()
        for text, fn in [("Split", self.split_item), ("Duplicate", self.duplicate_item)]:
            b = QPushButton(text)
            b.clicked.connect(fn)
            r2.addWidget(b)
        lay.addLayout(r2)
        bd = QPushButton("Delete clip")
        bd.setObjectName("danger")
        bd.clicked.connect(self.delete_item)
        lay.addWidget(bd)
        lay.addStretch(1)
        self.inspector_box = box
        box.setEnabled(False)
        return box

    # ---------------- settings <-> project
    def collect_settings(self):
        p = self.project
        p.style = self.cb_style.currentText()
        p.effects = self.cb_effects.currentText()
        p.format = self.cb_format.currentData()
        p.reframe = self.cb_reframe.currentData()
        p.move_strength = self.sp_move.value() / 100.0
        p.sfx = self.ck_sfx.isChecked()
        p.lut = self.ed_lut.text().strip()
        p.title, p.subtitle, p.outro = (self.ed_title.text(), self.ed_sub.text(), self.ed_outro.text())
        p.music_offset = self.sp_music_start.value()
        if hasattr(self, "caption_style"):
            p.caption_style = self.caption_style.currentData() or p.caption_style
        if hasattr(self, "audio_music_gain"):
            p.music_gain = self.audio_music_gain.value()
            p.use_clip_audio = self.ck_clip_audio.isChecked()
            p.clip_audio_gain = self.audio_clip_gain.value()
            p.voiceover = getattr(p, "voiceover", "")
            p.voiceover_offset = self.audio_voice_offset.value()
            p.voiceover_gain = self.audio_voice_gain.value()
            p.duck_music = self.ck_duck_music.isChecked()

    def apply_settings(self):
        p = self.project
        self.cb_style.setCurrentText(p.style)
        self.cb_effects.setCurrentText(p.effects)
        self.cb_format.setCurrentIndex(max(0, self.cb_format.findData(p.format)))
        self.cb_reframe.setCurrentIndex(max(0, self.cb_reframe.findData(p.reframe)))
        self.sp_move.setValue(int(round(p.move_strength * 100)))
        self.ck_sfx.setChecked(p.sfx)
        self.ed_lut.setText(p.lut)
        self.ed_title.setText(p.title)
        self.ed_sub.setText(p.subtitle)
        self.ed_outro.setText(p.outro)
        self.sp_music_start.setValue(p.music_offset)
        if hasattr(self, "caption_style"):
            idx = self.caption_style.findData(getattr(p, "caption_style", "bold"))
            if idx >= 0:
                self.caption_style.setCurrentIndex(idx)
        if hasattr(self, "caption_output"):
            cues = getattr(p, "captions", []) or []
            self.caption_output.setPlainText("\n".join(
                f"[{c.get('start', 0):06.2f}s – {c.get('end', 0):06.2f}s]  {c.get('text', '')}"
                for c in cues
            ))
        self.music_label.setText(Path(p.music).name if p.music else "(none)")
        if hasattr(self, "audio_music_gain"):
            self.audio_music_gain.setValue(float(getattr(p, "music_gain", 0.90)))
            self.ck_clip_audio.setChecked(bool(getattr(p, "use_clip_audio", True)))
            self.audio_clip_gain.setValue(float(getattr(p, "clip_audio_gain", 1.0)))
            self.audio_voice_offset.setValue(float(getattr(p, "voiceover_offset", 0.0)))
            self.audio_voice_gain.setValue(float(getattr(p, "voiceover_gain", 1.0)))
            self.ck_duck_music.setChecked(bool(getattr(p, "duck_music", True)))
            vo = getattr(p, "voiceover", "") or ""
            self.voiceover_label.setText(Path(vo).name if vo else "No voice-over selected")
        if hasattr(self, "checklist"):
            self.update_checklist()

    # ---------------- helpers
    def say(self, text):
        self.status.setText(text)

    def set_busy(self, busy):
        for b in (self.btn_auto, self.btn_prev, self.btn_export, self.btn_clip_prev):
            b.setEnabled(not busy)
        self.bar.setVisible(busy)
        if not busy:
            self.bar.setValue(0)

    def error(self, text):
        QMessageBox.warning(self, "Workout Vlog Editor", text)

    # ---------------- undo / redo
    def _snap(self):
        return json.dumps([asdict(i) for i in self.project.items])

    def snapshot(self, coalesce=False):
        now = time.time()
        if coalesce and now - self._last_snap < 1.0:
            return
        self._last_snap = now
        self.history.append(self._snap())
        self.history = self.history[-80:]
        self.redo.clear()

    def _restore(self, s):
        self.project.items = [core.Item(**d) for d in json.loads(s)]
        self.sel = min(self.sel, len(self.project.items) - 1)
        self.refresh()

    def undo(self):
        if self.history:
            self.redo.append(self._snap())
            self._restore(self.history.pop())

    def do_redo(self):
        if self.redo:
            self.history.append(self._snap())
            self._restore(self.redo.pop())

    # ---------------- media bin
    def pick_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Add clips and photos", "",
            "Media (*.mp4 *.mov *.mkv *.m4v *.avi *.webm *.jpg *.jpeg *.png *.webp)")
        self.add_media(files)

    def pick_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Add a folder of footage")
        if d:
            self.add_media([d])

    def add_media(self, paths):
        new = [p for p in core.scan_paths(paths) if p not in self.media]
        for p in new:
            self.media.append(p)
            it = QListWidgetItem(QIcon(self.thumbs.get(p, 0.0, 120)), Path(p).name)
            it.setData(Qt.ItemDataRole.UserRole, p)
            it.setToolTip(p)
            self.bin.addItem(it)
        self.update_checklist()
        self.say(f"Added {len(new)} file(s) to the bin")

    def remove_from_bin(self):
        for it in self.bin.selectedItems():
            p = it.data(Qt.ItemDataRole.UserRole)
            self.media = [m for m in self.media if m != p]
            self.bin.takeItem(self.bin.row(it))
        self.update_checklist()

    def add_to_timeline(self):
        its = self.bin.selectedItems()
        if not its:
            self.say("Select a clip in the bin first")
            return
        self.snapshot()
        pos = self.sel + 1 if self.sel >= 0 else len(self.project.items)
        for it in its:
            path = it.data(Qt.ItemDataRole.UserRole)
            kind = core.media_kind(path)
            self.project.items.insert(pos, core.Item(source=path, kind=kind, duration=2.0,
                                                     section=1, move="push" if kind == "photo" else None))
            pos += 1
        self.sel = pos - 1
        self.refresh()

    def pick_music(self):
        f, _ = QFileDialog.getOpenFileName(self, "Choose music", "",
                                           "Audio (*.mp3 *.wav *.m4a *.aac *.flac *.ogg)")
        if f:
            self.project.music = f
            self.music_label.setText(Path(f).name)
            self.update_checklist()

    def pick_lut(self):
        f, _ = QFileDialog.getOpenFileName(self, "Choose LUT", "", "LUT (*.cube)")
        if f:
            self.ed_lut.setText(f)

    # ---------------- project files
    def new_project(self):
        if self.project.items and QMessageBox.question(
                self, "New project", "Discard the current timeline?") != QMessageBox.StandardButton.Yes:
            return
        self.project = core.Project()
        self.history.clear()
        self.redo.clear()
        self.sel = -1
        self.project_path = None
        self.apply_settings()
        self.refresh()

    def save_project(self):
        self.collect_settings()
        path = self.project_path
        if not path:
            path, _ = QFileDialog.getSaveFileName(self, "Save project", "my_edit.json", "Project (*.json)")
        if path:
            self.project.save(path)
            self.project_path = path
            self.say(f"Saved {path}")

    def open_project(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open project", "", "Project (*.json)")
        if not path:
            return
        try:
            self.project = core.Project.load(path)
        except Exception as e:
            self.error(f"Could not open the project:\n{e}")
            return
        self.project_path = path
        self.history.clear()
        self.redo.clear()
        self.sel = 0 if self.project.items else -1
        self.add_media([i.source for i in self.project.items if Path(i.source).exists()])
        self.apply_settings()
        self.refresh()

    # ---------------- automatic editing
    def auto_edit(self):
        if self.worker and self.worker.isRunning():
            return
        self.collect_settings()
        p = self.project
        if not self.media:
            self.error("Add some clips first (Media tab).")
            return
        if not p.music:
            self.pick_music()
            p = self.project
            if not p.music:
                self.error("Auto-edit needs a music file. Choose one in the Settings tab.")
                return
        if p.items and QMessageBox.question(
                self, "Auto-edit", "This replaces the current timeline. Continue?") != QMessageBox.StandardButton.Yes:
            return
        media, music = list(self.media), p.music
        length, seed = self.sp_length.value(), (self.sp_seed.value() or None)
        keep = copy.deepcopy(p)
        mstart = "auto" if self.sp_music_start.value() == 0 else self.sp_music_start.value()

        def job(cb):
            return core.auto_edit(media, music, length=length, style=keep.style, effects=keep.effects,
                                  seed=seed, music_start=mstart, title=keep.title,
                                  subtitle=keep.subtitle, outro=keep.outro, fmt=keep.format,
                                  fps=keep.fps, sfx=keep.sfx, lut=keep.lut, reframe=keep.reframe,
                                  move_strength=keep.move_strength, cb=cb)

        self._start(job, self._auto_done)

    def _auto_done(self, proj):
        self.snapshot()
        self.project = proj
        self.sel = 0 if proj.items else -1
        self.apply_settings()
        self.refresh()
        self.say(f"Auto-edit ready: {len(proj.items)} clips, {proj.total_length():.1f}s. "
                 f"Click any clip to fine-tune it.")

    # ---------------- rendering
    def render(self, preview):
        if self.worker and self.worker.isRunning():
            return
        self.collect_settings()
        if not self.project.items:
            self.error("The timeline is empty.")
            return
        if not self.project.music:
            self.pick_music()
            if not self.project.music:
                self.error("Rendering needs a music file. Choose one in the Settings tab.")
                return
        if preview:
            out = str(Path(tempfile.gettempdir()) / f"vlog_preview_{int(time.time())}.mp4")
        else:
            out, _ = QFileDialog.getSaveFileName(self, "Export video", "my_vlog.mp4", "MP4 video (*.mp4)")
            if not out:
                return
        proj = copy.deepcopy(self.project)
        self._start(lambda cb: core.render_project(proj, out, preview=preview, cb=cb),
                    lambda path: self._render_done(path, preview))

    def _render_done(self, path, preview):
        if preview:
            self.preview.play_file(path)
            self.say("Preview ready (half resolution, no grain/bloom)")
        else:
            self.say(f"Exported {path}")
            QMessageBox.information(self, "Export finished", f"Saved:\n{path}")

    # ---------------- worker plumbing
    def _start(self, job, on_done):
        self.set_busy(True)
        self.worker = Worker(job)
        self.worker.progress.connect(self._progress)
        self.worker.done.connect(lambda res: (self.set_busy(False), on_done(res)))
        self.worker.failed.connect(lambda msg: (self.set_busy(False), self.error(msg), self.say("Failed")))
        self.worker.start()

    def _progress(self, msg, frac):
        self.say(msg)
        if frac >= 0:
            self.bar.setValue(int(frac * 100))

    # ---------------- timeline + inspector
    def _zoom(self, v):
        self.timeline.pxps = v
        self.refresh()

    def refresh(self):
        items = self.project.items
        if self.sel >= len(items):
            self.sel = len(items) - 1
        self.timeline.set_items(items, self.sel)
        self.update_checklist()
        total = self.project.total_length()
        self.total_label.setText(f"{len(items)} clips · {total:.1f}s total")
        self.load_inspector()

    def select(self, i):
        self.sel = i
        self.timeline.set_items(self.project.items, i)
        self.load_inspector()
        self.pv_timer.start(120)

    def load_inspector(self):
        self._loading = True
        ok = 0 <= self.sel < len(self.project.items)
        self.inspector_box.setEnabled(ok)
        if ok:
            it = self.project.items[self.sel]
            self.lb_src.setText(f"{Path(it.source).name}  ({it.kind})")
            video = it.kind == "video"
            self.sp_start.setEnabled(video)
            self.sp_speed.setEnabled(video)
            self.checks["ramp"].setEnabled(video)
            self.sp_start.setValue(it.src_start)
            self.sp_dur.setValue(it.duration)
            self.sp_speed.setValue(it.speed)
            self.cb_move.setCurrentText(it.move or "none")
            self.cb_section.setCurrentIndex(it.section % 4)
            self.cb_item_reframe.setCurrentIndex(max(0, self.cb_item_reframe.findData(it.reframe or "")))
            self.sl_focus.setValue(int(round(it.focus_x * 100)))
            self.lb_focus.setText(f"{int(round(it.focus_x * 100))}%")
            for k, c in self.checks.items():
                c.setChecked(bool(getattr(it, k)))
        else:
            self.lb_src.setText("No clip selected")
        self._loading = False

    def _inspector_changed(self, *_):
        if self._loading or not (0 <= self.sel < len(self.project.items)):
            return
        self.snapshot(coalesce=True)
        it = self.project.items[self.sel]
        it.src_start = self.sp_start.value()
        it.duration = self.sp_dur.value()
        it.speed = self.sp_speed.value()
        mv = self.cb_move.currentText()
        it.move = None if mv == "none" else mv
        it.section = self.cb_section.currentIndex()
        it.reframe = self.cb_item_reframe.currentData() or ""
        it.focus_x = self.sl_focus.value() / 100.0
        self.lb_focus.setText(f"{self.sl_focus.value()}%")
        for k, c in self.checks.items():
            setattr(it, k, c.isChecked())
        self.timeline.set_items(self.project.items, self.sel)
        self.total_label.setText(f"{len(self.project.items)} clips · {self.project.total_length():.1f}s total")
        self.pv_timer.start(350)

    def preview_item(self):
        """Show the clip exactly as it will be framed in the final video."""
        if not (0 <= self.sel < len(self.project.items)):
            return
        it = self.project.items[self.sel]
        self.collect_settings()
        img = core.thumbnail(it.source, it.src_start, 960)
        if img is None:
            self.preview.show_pixmap(None, "Could not read this file")
            return
        vertical = self.project.format == "vertical"
        W, H = (540, 960) if vertical else (960, 540)
        h, w = img.shape[:2]
        mode = core.effective_reframe(it.reframe or self.project.reframe, w, h, W, H)
        frame = core.compose_frame(img, it.reframe or self.project.reframe, it.focus_x, W, H)
        self.preview.show_pixmap(to_pixmap(frame))
        label = {"fill": "Fill (crop)", "smart": "Smart", "fit": "Fit"}[mode]
        self.preview.info.setText(f"Final framing: {'9:16 vertical' if vertical else '16:9'}"
                                  f" · {label}.  Use ▶ Preview this clip to see effects.")

    def preview_clip(self):
        if self.worker and self.worker.isRunning():
            return
        if not (0 <= self.sel < len(self.project.items)):
            return
        self.collect_settings()
        proj = copy.deepcopy(self.project)
        idx = self.sel
        out = str(Path(tempfile.gettempdir()) / f"vlog_clip_{int(time.time() * 1000)}.mp4")
        self._start(lambda cb: core.render_clip_preview(proj, idx, out, cb),
                    lambda path: (self.preview.play_file(path), self.say("Clip preview (half resolution)")))

    # ---------------- timeline edits
    def move_item(self, src, tgt):
        self.snapshot()
        items = self.project.items
        it = items.pop(src)
        items.insert(tgt, it)
        self.sel = tgt
        self.refresh()

    def shift_item(self, d):
        i = self.sel
        j = i + d
        if 0 <= i < len(self.project.items) and 0 <= j < len(self.project.items):
            self.move_item(i, j)

    def delete_item(self):
        if 0 <= self.sel < len(self.project.items):
            self.snapshot()
            del self.project.items[self.sel]
            self.refresh()

    def duplicate_item(self):
        if 0 <= self.sel < len(self.project.items):
            self.snapshot()
            self.project.items.insert(self.sel + 1, copy.deepcopy(self.project.items[self.sel]))
            self.project.items[self.sel + 1].hero = False
            self.sel += 1
            self.refresh()

    def split_item(self):
        if not (0 <= self.sel < len(self.project.items)):
            return
        it = self.project.items[self.sel]
        if it.duration < 0.8:
            self.say("Clip is too short to split")
            return
        self.snapshot()
        half = it.duration / 2
        b = copy.deepcopy(it)
        b.hero = False
        b.flash = b.glitch = b.whip = b.dip_in = False
        b.duration = round(it.duration - half, 4)
        b.src_start = round(it.src_start + half * (it.speed if it.kind == "video" else 1), 3)
        it.duration = round(half, 4)
        it.dip_out = False
        self.project.items.insert(self.sel + 1, b)
        self.refresh()

    def keyPressEvent(self, e):
        if e.key() == Qt.Key.Key_Delete and isinstance(self.focusWidget(), (Timeline, QScrollArea)):
            self.delete_item()
        else:
            super().keyPressEvent(e)


def selftest():
    """Run `praveeneditor.exe --selftest` to verify the packaged app. Writes a report to
    praveeneditor_selftest.txt in your home folder (there is no console in the windowed exe)."""
    lines = []
    try:
        core.check_ffmpeg()
        lines.append(f"ffmpeg  OK  {shutil.which('ffmpeg')}")
    except Exception as e:
        lines.append(f"ffmpeg  FAIL  {e}")
    try:
        import librosa
        sr = 22050
        y = np.zeros(sr * 6)
        y[::sr // 2] = 1.0
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        lines.append(f"librosa OK  tempo={float(np.atleast_1d(tempo)[0]):.0f} beats={len(beats)}")
    except Exception as e:
        lines.append(f"librosa FAIL  {type(e).__name__}: {e}")
    try:
        import cv2
        lines.append(f"opencv  OK  {cv2.__version__}")
    except Exception as e:
        lines.append(f"opencv  FAIL  {e}")
    report = "\n".join(lines)
    (Path.home() / "praveeneditor_selftest.txt").write_text(report)
    print(report)


def install_error_hook():
    """Show unexpected errors in a dialog and log them (a windowed exe has no console)."""
    import traceback
    log = Path.home() / "praveeneditor_log.txt"

    def hook(etype, value, tb):
        text = "".join(traceback.format_exception(etype, value, tb))
        try:
            with open(log, "a", encoding="utf-8") as f:
                f.write(time.strftime("%Y-%m-%d %H:%M:%S") + "\n" + text + "\n")
        except Exception:
            pass
        try:
            QMessageBox.critical(None, "Praveen Editor: unexpected error",
                                 f"{value}\n\nDetails were saved to:\n{log}")
        except Exception:
            pass

    sys.excepthook = hook


def main():
    if "--selftest" in sys.argv:
        selftest()
        return
    app = QApplication(sys.argv)
    app.setStyleSheet(DARK)
    install_error_hook()
    win = MainWindow()
    win.showMaximized()
    try:
        core.check_ffmpeg()
    except core.EditorError as e:
        QMessageBox.warning(win, "ffmpeg missing", str(e))
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
