"""
Study Tracker — desktop launcher. Run from the project root:
  python launcher.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from embedded_session import CollectDataThread, StudySessionThread
from PyQt6.QtCore import QProcess, QProcessEnvironment, QSize, Qt
from PyQt6.QtGui import QFont, QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


def env_with_pythonpath() -> QProcessEnvironment:
    env = QProcessEnvironment.systemEnvironment()
    existing = env.value("PYTHONPATH", "")
    sep = os.pathsep
    src_str = str(SRC)
    if existing:
        env.insert("PYTHONPATH", src_str + sep + existing)
    else:
        env.insert("PYTHONPATH", src_str)
    return env


class VideoFrame(QLabel):
    """Scales camera frames; parent controls outer size."""

    def __init__(self) -> None:
        super().__init__()
        self._qimg: QImage | None = None
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(0, 0)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setStyleSheet(
            "background-color: #000000; border-radius: 12px; border: 1px solid #2a3040;"
        )

    def set_frame(self, qimg: QImage) -> None:
        self._qimg = qimg.copy()
        self._refresh()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._refresh()

    def _refresh(self) -> None:
        if self._qimg is None or self._qimg.isNull():
            return
        pm = QPixmap.fromImage(self._qimg)
        self.setPixmap(
            pm.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def clear_frame(self) -> None:
        self._qimg = None
        self.clear()


class VideoAspectContainer(QWidget):
    """Gives the preview a stable 16:9 height for width so layouts behave when resized."""

    def __init__(self) -> None:
        super().__init__()
        self.video = VideoFrame()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.video)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

    def heightForWidth(self, w: int) -> int:
        return max(90, int(w * 9 / 16))

    def hasHeightForWidth(self) -> bool:
        return True

    def sizeHint(self) -> QSize:
        return QSize(480, 270)


class LauncherWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Study Tracker")
        self.setMinimumSize(320, 360)
        self.resize(880, 900)

        self._processes: list[QProcess] = []
        self._study_thread: StudySessionThread | None = None
        self._collect_thread: CollectDataThread | None = None

        central = QWidget()
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)
        outer.setSpacing(0)
        outer.setContentsMargins(12, 12, 12, 12)

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(6)

        self._video_wrap = VideoAspectContainer()
        splitter.addWidget(self._video_wrap)
        self.video = self._video_wrap.video

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        scroll_inner = QWidget()
        scroll_inner.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        layout = QVBoxLayout(scroll_inner)
        layout.setSpacing(12)
        layout.setContentsMargins(16, 8, 16, 16)

        title = QLabel("Study Tracker")
        title.setObjectName("title")
        title.setFont(QFont("Segoe UI", 22, QFont.Weight.DemiBold))
        title.setWordWrap(True)
        subtitle = QLabel(
            "Camera preview above. Start a study session or data collection; "
            "train the model and open reports from the buttons below."
        )
        subtitle.setObjectName("subtitle")
        subtitle.setWordWrap(True)

        layout.addWidget(title)
        layout.addWidget(subtitle)

        cam_row = QHBoxLayout()
        cam_row.addWidget(QLabel("Camera index:"))
        self.camera_index = QComboBox()
        self.camera_index.addItems(["0", "1", "2"])
        cam_row.addWidget(self.camera_index)
        cam_row.addStretch()
        layout.addLayout(cam_row)

        # Study session — title/desc full width, buttons on next row
        study_card = self._card_frame()
        sv = QVBoxLayout()
        sv.setSpacing(8)
        st = QLabel("Study session")
        st.setObjectName("cardTitle")
        sd = QLabel(
            "Live focus model + Spotify playlists. Stopping saves session_results.csv."
        )
        sd.setObjectName("cardDesc")
        sd.setWordWrap(True)
        sd.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        sv.addWidget(st)
        sv.addWidget(sd)
        study_btns = QHBoxLayout()
        study_btns.setSpacing(8)
        self.btn_study_start = QPushButton("Start")
        self.btn_study_start.setObjectName("runBtn")
        self.btn_study_start.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_study_start.setMinimumHeight(36)
        self.btn_study_start.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self.btn_study_start.clicked.connect(self._start_study)
        self.btn_study_stop = QPushButton("Stop")
        self.btn_study_stop.setObjectName("stopBtn")
        self.btn_study_stop.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_study_stop.setMinimumHeight(36)
        self.btn_study_stop.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self.btn_study_stop.setEnabled(False)
        self.btn_study_stop.clicked.connect(self._stop_study)
        study_btns.addWidget(self.btn_study_start, stretch=1)
        study_btns.addWidget(self.btn_study_stop, stretch=1)
        sv.addLayout(study_btns)
        study_card.setLayout(sv)
        layout.addWidget(study_card)

        # Collect data
        coll_card = self._card_frame()
        cv = QVBoxLayout()
        cv.setSpacing(8)
        ct = QLabel("Collect training data")
        ct.setObjectName("cardTitle")
        cd = QLabel(
            "Record face blendshapes. Focused writes data_focused.csv; "
            "distracted writes data_distracted.csv."
        )
        cd.setObjectName("cardDesc")
        cd.setWordWrap(True)
        cd.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        cv.addWidget(ct)
        cv.addWidget(cd)
        row = QHBoxLayout()
        row.setSpacing(8)
        self.btn_collect_focused = QPushButton("Collect focused")
        self.btn_collect_focused.setObjectName("runBtn")
        self.btn_collect_focused.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_collect_focused.setMinimumHeight(36)
        self.btn_collect_focused.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self.btn_collect_focused.clicked.connect(lambda: self._start_collect("focused"))
        self.btn_collect_distracted = QPushButton("Collect distracted")
        self.btn_collect_distracted.setObjectName("runBtn")
        self.btn_collect_distracted.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_collect_distracted.setMinimumHeight(36)
        self.btn_collect_distracted.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self.btn_collect_distracted.clicked.connect(lambda: self._start_collect("distracted"))
        row.addWidget(self.btn_collect_focused, stretch=1)
        row.addWidget(self.btn_collect_distracted, stretch=1)
        cv.addLayout(row)
        self.btn_collect_stop = QPushButton("Stop collection")
        self.btn_collect_stop.setObjectName("stopBtn")
        self.btn_collect_stop.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_collect_stop.setMinimumHeight(36)
        self.btn_collect_stop.setEnabled(False)
        self.btn_collect_stop.clicked.connect(self._stop_collect)
        cv.addWidget(self.btn_collect_stop)
        coll_card.setLayout(cv)
        layout.addWidget(coll_card)

        for name, desc, script in [
            (
                "Train model",
                "Train Random Forest on data_focused.csv and data_distracted.csv; writes focus_model.pkl.",
                SRC / "train_model.py",
            ),
            (
                "Session report",
                "Plot focus over time from session_results.csv (opens chart window).",
                SRC / "report.py",
            ),
        ]:
            layout.addWidget(self._action_card(name, desc, script))

        log_label = QLabel("Output")
        log_label.setObjectName("sectionLabel")
        layout.addWidget(log_label)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setPlaceholderText("Logs from sessions and scripts appear here…")
        self.log.setMinimumHeight(72)
        self.log.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.MinimumExpanding)
        layout.addWidget(self.log)

        scroll.setWidget(scroll_inner)
        splitter.addWidget(scroll)

        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)
        splitter.setSizes([320, 520])

        outer.addWidget(splitter)

        self._apply_style()

    def _card_frame(self) -> QFrame:
        f = QFrame()
        f.setObjectName("card")
        f.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        return f

    def _action_card(self, title: str, description: str, script: Path) -> QFrame:
        card = self._card_frame()
        outer = QVBoxLayout(card)
        outer.setSpacing(8)
        t = QLabel(title)
        t.setObjectName("cardTitle")
        d = QLabel(description)
        d.setObjectName("cardDesc")
        d.setWordWrap(True)
        d.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        outer.addWidget(t)
        outer.addWidget(d)
        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        btn = QPushButton("Run")
        btn.setObjectName("runBtn")
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setMinimumHeight(36)
        btn.setMinimumWidth(96)
        btn.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
        btn.clicked.connect(lambda checked=False, s=script: self._run_script(s))
        btn_row.addWidget(btn)
        outer.addLayout(btn_row)
        return card

    def _append_log(self, text: str) -> None:
        self.log.append(text.rstrip())

    def _camera_ix(self) -> int:
        return int(self.camera_index.currentText())

    def _any_camera_active(self) -> bool:
        return (self._study_thread is not None and self._study_thread.isRunning()) or (
            self._collect_thread is not None and self._collect_thread.isRunning()
        )

    def _start_study(self) -> None:
        if self._any_camera_active():
            QMessageBox.information(
                self,
                "Camera busy",
                "Stop the current session or data collection before starting another.",
            )
            return
        self.video.clear_frame()
        th = StudySessionThread(ROOT, camera_index=self._camera_ix(), parent=self)
        th.frame_ready.connect(self.video.set_frame)
        th.log_message.connect(self._append_log)
        th.session_saved.connect(self._append_log)
        th.failed.connect(self._on_study_failed)
        th.finished.connect(self._on_study_finished)
        self._study_thread = th
        self.btn_study_start.setEnabled(False)
        self.btn_study_stop.setEnabled(True)
        self.btn_collect_focused.setEnabled(False)
        self.btn_collect_distracted.setEnabled(False)
        th.start()

    def _on_study_failed(self, msg: str) -> None:
        self._append_log(f"Study session error: {msg}")
        QMessageBox.critical(self, "Study session", msg)
        self._reset_study_ui()

    def _on_study_finished(self) -> None:
        self._reset_study_ui()

    def _reset_study_ui(self) -> None:
        self.btn_study_start.setEnabled(True)
        self.btn_study_stop.setEnabled(False)
        self.btn_collect_focused.setEnabled(True)
        self.btn_collect_distracted.setEnabled(True)
        self._study_thread = None

    def _stop_study(self) -> None:
        if self._study_thread is not None and self._study_thread.isRunning():
            self._study_thread.request_stop()
            self._study_thread.wait(8000)

    def _start_collect(self, label: str) -> None:
        if self._any_camera_active():
            QMessageBox.information(
                self,
                "Camera busy",
                "Stop the current session or data collection before starting another.",
            )
            return
        self.video.clear_frame()
        th = CollectDataThread(ROOT, label, camera_index=self._camera_ix(), parent=self)
        th.frame_ready.connect(self.video.set_frame)
        th.log_message.connect(self._append_log)
        th.saved.connect(self._append_log)
        th.failed.connect(self._on_collect_failed)
        th.finished.connect(self._on_collect_finished)
        self._collect_thread = th
        self.btn_collect_focused.setEnabled(False)
        self.btn_collect_distracted.setEnabled(False)
        self.btn_collect_stop.setEnabled(True)
        self.btn_study_start.setEnabled(False)
        th.start()

    def _on_collect_failed(self, msg: str) -> None:
        self._append_log(f"Collect error: {msg}")
        QMessageBox.critical(self, "Collect data", msg)
        self._reset_collect_ui()

    def _on_collect_finished(self) -> None:
        self._reset_collect_ui()

    def _reset_collect_ui(self) -> None:
        self.btn_collect_focused.setEnabled(True)
        self.btn_collect_distracted.setEnabled(True)
        self.btn_collect_stop.setEnabled(False)
        self.btn_study_start.setEnabled(True)
        self._collect_thread = None

    def _stop_collect(self) -> None:
        if self._collect_thread is not None and self._collect_thread.isRunning():
            self._collect_thread.request_stop()
            self._collect_thread.wait(8000)

    def _run_script(self, script: Path) -> None:
        if not script.is_file():
            QMessageBox.warning(
                self,
                "Missing file",
                f"Script not found:\n{script}",
            )
            return

        proc = QProcess(self)
        proc.setProgram(sys.executable)
        proc.setArguments([str(script)])
        proc.setWorkingDirectory(str(ROOT))
        proc.setProcessEnvironment(env_with_pythonpath())

        proc.started.connect(lambda: self._append_log(f"Started: {script.name}"))
        proc.readyReadStandardOutput.connect(
            lambda: self._append_log(bytes(proc.readAllStandardOutput()).decode(errors="replace"))
        )
        proc.readyReadStandardError.connect(
            lambda: self._append_log(bytes(proc.readAllStandardError()).decode(errors="replace"))
        )

        def finished(code: int, status: QProcess.ExitStatus) -> None:
            if status == QProcess.ExitStatus.NormalExit:
                self._append_log(f"Finished {script.name} (exit {code}).")
            else:
                self._append_log(f"{script.name} crashed or was terminated.")

        proc.finished.connect(finished)

        proc.start()
        if not proc.waitForStarted(3000):
            QMessageBox.critical(
                self,
                "Could not start",
                f"Failed to start:\n{script}",
            )
            return

        self._processes.append(proc)

    def closeEvent(self, event) -> None:
        if self._study_thread is not None and self._study_thread.isRunning():
            self._study_thread.request_stop()
            self._study_thread.wait(5000)
        if self._collect_thread is not None and self._collect_thread.isRunning():
            self._collect_thread.request_stop()
            self._collect_thread.wait(5000)
        super().closeEvent(event)

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow { background-color: #0f1218; }
            QScrollArea { background-color: transparent; border: none; }
            QLabel { color: #c5cad6; }
            QLabel#title { color: #f0f2f7; }
            QLabel#subtitle { color: #9aa3b2; font-size: 13px; }
            QLabel#sectionLabel {
                color: #c5cad6;
                font-size: 12px;
                font-weight: 600;
                letter-spacing: 0.5px;
                margin-top: 8px;
            }
            QFrame#card {
                background-color: #181c24;
                border: 1px solid #2a3040;
                border-radius: 14px;
                padding: 14px 16px;
            }
            QLabel#cardTitle { color: #eef1f6; font-size: 15px; font-weight: 600; }
            QLabel#cardDesc { color: #8b95a8; font-size: 12px; margin-top: 4px; }
            QPushButton#runBtn {
                background-color: #3b5bdb;
                color: #ffffff;
                border: none;
                border-radius: 10px;
                font-size: 13px;
                font-weight: 600;
                padding: 10px 16px;
            }
            QPushButton#runBtn:hover { background-color: #4c6ef5; }
            QPushButton#runBtn:pressed { background-color: #364fc7; }
            QPushButton#runBtn:disabled { background-color: #2a3040; color: #6b7280; }
            QPushButton#stopBtn {
                background-color: #2f3644;
                color: #e8eaed;
                border: 1px solid #4a5568;
                border-radius: 10px;
                font-size: 13px;
                font-weight: 600;
                padding: 10px 16px;
            }
            QPushButton#stopBtn:hover { background-color: #3d4654; }
            QPushButton#stopBtn:pressed { background-color: #252b36; }
            QPushButton#stopBtn:disabled { background-color: #1a1f28; color: #555; border-color: #2a3040; }
            QComboBox {
                background-color: #12151c;
                color: #e8eaed;
                border: 1px solid #2a3040;
                border-radius: 8px;
                padding: 6px 12px;
                min-width: 80px;
            }
            QTextEdit {
                background-color: #12151c;
                color: #d0d6e0;
                border: 1px solid #2a3040;
                border-radius: 10px;
                padding: 10px;
                font-family: Consolas, "Cascadia Mono", monospace;
                font-size: 11px;
            }
            QSplitter::handle {
                background-color: #2a3040;
                border-radius: 2px;
            }
            QSplitter::handle:hover {
                background-color: #3d4450;
            }
            """
        )


def main() -> None:
    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 10))
    w = LauncherWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
