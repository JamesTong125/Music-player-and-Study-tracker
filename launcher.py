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
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
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
        self.setObjectName("videoFrame")
        self._qimg: QImage | None = None
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(0, 0)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        # Styling is handled by the app stylesheet; keep fallback in sync with theme.
        self.setStyleSheet("background-color: #DCC4B8; border-radius: 12px;")

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
    """Fixed-size preview container so it doesn't steal layout space."""

    def __init__(self) -> None:
        super().__init__()
        self.video = VideoFrame()
        self._aspect_w: int = 16
        self._aspect_h: int = 9
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.video)
        # Keep preview size stable; video scales inside.
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMinimumHeight(280)
        self.setMaximumHeight(340)

    def heightForWidth(self, w: int) -> int:
        if self._aspect_w <= 0 or self._aspect_h <= 0:
            return max(90, int(w * 9 / 16))
        return max(90, int(w * self._aspect_h / self._aspect_w))

    def hasHeightForWidth(self) -> bool:
        # Disable height-for-width so aspect changes won't resize the layout.
        return False

    def sizeHint(self) -> QSize:
        return QSize(480, 300)

    def set_frame(self, qimg: QImage) -> None:
        # Update aspect ratio from actual frames (webcams often deliver 4:3).
        if qimg is not None and not qimg.isNull() and qimg.width() > 0 and qimg.height() > 0:
            self._aspect_w = qimg.width()
            self._aspect_h = qimg.height()
        self.video.set_frame(qimg)

    def clear_frame(self) -> None:
        self.video.clear_frame()


class LauncherWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Study Tracker")
        self.setMinimumSize(320, 360)
        self.resize(920, 920)

        self._processes: list[QProcess] = []
        self._study_thread: StudySessionThread | None = None
        self._collect_thread: CollectDataThread | None = None

        central = QWidget()
        central.setObjectName("centralRoot")
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)
        outer.setSpacing(0)
        outer.setContentsMargins(28, 28, 28, 28)

        # Single scroll area for the whole UI (no split-pane scrolling).
        scroll = QScrollArea()
        scroll.setObjectName("mainScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        scroll_inner = QWidget()
        scroll_inner.setObjectName("scrollContents")
        scroll_inner.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        layout = QVBoxLayout(scroll_inner)
        layout.setSpacing(16)
        layout.setContentsMargins(0, 0, 0, 0)

        hero = self._card_frame()
        hero.setObjectName("heroCard")
        hero_l = QVBoxLayout(hero)
        hero_l.setSpacing(8)
        hero_l.setContentsMargins(22, 22, 22, 22)

        title_row = QHBoxLayout()
        title_row.setSpacing(10)
        title = QLabel("Study Tracker")
        title.setObjectName("title")
        title.setWordWrap(True)
        badge = QLabel("Focus • Music • Flow")
        badge.setObjectName("badge")
        badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        badge.setMinimumHeight(26)
        badge.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
        title_row.addWidget(title, stretch=1)
        title_row.addWidget(badge)

        subtitle = QLabel(
            "Start a session, collect training data, then train your model and view reports. "
            "Preview and controls live in one layout."
        )
        subtitle.setObjectName("subtitle")
        subtitle.setWordWrap(True)

        hero_l.addLayout(title_row)
        hero_l.addWidget(subtitle)
        layout.addWidget(hero)

        # Two balanced columns: camera + output | action cards (square overall feel).
        body = QWidget()
        body.setObjectName("bodyRow")
        body_l = QHBoxLayout(body)
        body_l.setSpacing(16)
        body_l.setContentsMargins(0, 0, 0, 0)

        left = QWidget()
        left.setObjectName("leftPane")
        left_l = QVBoxLayout(left)
        left_l.setSpacing(12)
        left_l.setContentsMargins(0, 0, 0, 0)

        self._video_wrap = VideoAspectContainer()
        left_l.addWidget(self._video_wrap)
        self.video = self._video_wrap.video

        cam_card = self._card_frame()
        cam_card.setObjectName("camCard")
        cam_l = QHBoxLayout(cam_card)
        cam_l.setSpacing(10)
        cam_l.setContentsMargins(16, 14, 16, 14)
        cam_label = QLabel("Camera")
        cam_label.setObjectName("fieldLabel")
        self.camera_index = QComboBox()
        self.camera_index.addItems(["0", "1", "2"])
        cam_l.addWidget(cam_label)
        cam_l.addWidget(self.camera_index)
        cam_l.addStretch(1)
        left_l.addWidget(cam_card)

        log_label_left = QLabel("Output")
        log_label_left.setObjectName("sectionLabel")
        left_l.addWidget(log_label_left)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setPlaceholderText("Logs from sessions and scripts appear here…")
        self.log.setObjectName("outputLog")
        self.log.setMinimumHeight(64)
        self.log.setMaximumHeight(88)
        self.log.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        left_l.addWidget(self.log)
        left_l.addStretch(1)

        right = QWidget()
        right.setObjectName("rightPane")
        grid = QGridLayout(right)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(12)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        grid.setRowStretch(0, 1)
        grid.setRowStretch(1, 1)

        # Study session
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
        grid.addWidget(study_card, 0, 0)

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
        grid.addWidget(coll_card, 0, 1)

        grid.addWidget(
            self._action_card(
                "Train model",
                "Train Random Forest on data_focused.csv and data_distracted.csv; writes focus_model.pkl.",
                SRC / "train_model.py",
            ),
            1,
            0,
        )
        grid.addWidget(
            self._action_card(
                "Session report",
                "Plot focus over time from session_results.csv (opens chart window).",
                SRC / "report.py",
            ),
            1,
            1,
        )

        body_l.addWidget(left, 1)
        body_l.addWidget(right, 1)
        layout.addWidget(body)

        scroll.setWidget(scroll_inner)
        scroll.viewport().setStyleSheet("background-color: #F6E5DA;")
        outer.addWidget(scroll)

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
        self._video_wrap.clear_frame()
        th = StudySessionThread(ROOT, camera_index=self._camera_ix(), parent=self)
        th.frame_ready.connect(self._video_wrap.set_frame)
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
        self._video_wrap.clear_frame()
        th = CollectDataThread(ROOT, label, camera_index=self._camera_ix(), parent=self)
        th.frame_ready.connect(self._video_wrap.set_frame)
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
            QWidget {
                font-family: "Comic Sans MS", "Comic Sans", "Chalkboard SE", "Segoe Print", cursive;
                font-size: 11px;
            }
            QMainWindow { background-color: #F6E5DA; }
            QWidget#centralRoot,
            QWidget#leftPane,
            QWidget#rightPane,
            QWidget#bodyRow,
            QWidget#scrollContents {
                background-color: #F6E5DA;
            }
            QScrollArea#mainScroll {
                background-color: #F6E5DA;
                border: none;
            }
            QScrollArea { background-color: #F6E5DA; border: none; }
            QLabel { color: #2E2A2A; font-weight: normal; }
            QLabel#title { color: #2A2524; font-size: 22px; font-weight: bold; }
            QLabel#subtitle { color: #5C524C; font-size: 13px; }
            QLabel#fieldLabel { color: #4A403A; font-size: 12px; font-weight: bold; }
            QLabel#sectionLabel {
                color: #4A403A;
                font-size: 12px;
                font-weight: bold;
                letter-spacing: 0.3px;
                margin-top: 8px;
            }
            QFrame#card {
                background-color: #DCC4B8;
                border: 1px solid #C4A994;
                border-radius: 22px;
                padding: 18px 20px;
                min-height: 152px;
            }
            QFrame#heroCard {
                background-color: #DCC4B8;
                border: 1px solid #C4A994;
                border-radius: 26px;
                min-height: 120px;
            }
            QFrame#camCard {
                background-color: #DCC4B8;
                border: 1px solid #C4A994;
                border-radius: 22px;
                padding: 18px 20px;
            }
            QLabel#badge {
                background-color: #E8B8A8;
                color: #5C2A22;
                border: 1px solid #D49A88;
                border-radius: 13px;
                padding: 4px 10px;
                font-size: 11px;
                font-weight: bold;
            }
            QLabel#cardTitle { color: #2A2524; font-size: 15px; font-weight: bold; }
            QLabel#cardDesc { color: #5C524C; font-size: 12px; margin-top: 2px; }
            QPushButton#runBtn {
                background-color: #E85A5A;
                color: #ffffff;
                border: 1px solid #D94E4E;
                border-radius: 14px;
                font-size: 13px;
                font-weight: bold;
                padding: 10px 16px;
            }
            QPushButton#runBtn:hover { background-color: #F06666; border-color: #E85A5A; }
            QPushButton#runBtn:pressed { background-color: #D94E4E; border-color: #C94A4A; }
            QPushButton#runBtn:disabled { background-color: #E8B0B0; color: #7A6A6A; border-color: #D8A0A0; }
            QPushButton#stopBtn {
                background-color: #D8BFB4;
                color: #3A332F;
                border: 1px solid #C9AEA2;
                border-radius: 14px;
                font-size: 13px;
                font-weight: bold;
                padding: 10px 16px;
            }
            QPushButton#stopBtn:hover { background-color: #C9AEA2; border-color: #B89E92; }
            QPushButton#stopBtn:pressed { background-color: #B89E92; border-color: #A88E82; }
            QPushButton#stopBtn:disabled { background-color: #DDD5CF; color: #8A807A; border-color: #CFC4BC; }
            QComboBox {
                background-color: #DCC4B8;
                color: #2E2A2A;
                border: 1px solid #C4A994;
                border-radius: 14px;
                padding: 8px 12px;
                min-width: 80px;
            }
            QComboBox::drop-down { border: 0px; width: 28px; }
            QComboBox QAbstractItemView {
                background-color: #DCC4B8;
                selection-background-color: #C4A994;
                selection-color: #2E2A2A;
                border: 1px solid #C4A994;
                border-radius: 12px;
                padding: 6px;
            }
            QTextEdit {
                background-color: #DCC4B8;
                color: #3A3432;
                border: 1px solid #C4A994;
                border-radius: 18px;
                padding: 10px;
                font-family: "Comic Sans MS", "Comic Sans", "Chalkboard SE", "Segoe Print", cursive;
                font-size: 11px;
            }
            QTextEdit#outputLog {
                min-height: 0px;
                padding: 8px 10px;
                font-size: 10px;
            }
            QLabel#videoFrame {
                background-color: #DCC4B8;
                border: 1px solid #C4A994;
                border-radius: 12px;
            }
            """
        )


def main() -> None:
    app = QApplication(sys.argv)
    ui_font = QFont()
    ui_font.setFamilies(
        ["Comic Sans MS", "Comic Sans", "Chalkboard SE", "Segoe Print", "Segoe UI"]
    )
    ui_font.setPointSize(10)
    app.setFont(ui_font)
    w = LauncherWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
