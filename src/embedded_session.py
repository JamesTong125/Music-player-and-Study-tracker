"""
Shared webcam + MediaPipe logic for CLI (OpenCV window) and PyQt (embedded preview).
"""
from __future__ import annotations

import sys
import time
from collections import deque
from pathlib import Path

import cv2
import joblib
import mediapipe as mp
import numpy as np
import pandas as pd
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtGui import QImage

from spotify_helper import get_spotify_client

# Must match train_model.py: df['label'].map({'focused': 1, 'distracted': 0})
FOCUSED_CLASS = 1


def blendshapes_to_feature_row(model, blendshape_list) -> list[float]:
    """Build one row in the same column order as training (DataFrame → sklearn feature_names_in_)."""
    names = getattr(model, "feature_names_in_", None)
    if names is None:
        return [b.score for b in blendshape_list]
    d = {b.category_name: float(b.score) for b in blendshape_list}
    return [d.get(str(n), 0.0) for n in names]


def bgr_to_qimage(bgr: np.ndarray) -> QImage:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
    return qimg.copy()


def draw_hud(
    frame: np.ndarray,
    focus_ratio: float,
    elapsed_sec: int,
) -> None:
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (280, 120), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    color = (0, 255, 0) if focus_ratio > 0.8 else (0, 0, 255)
    cv2.putText(
        frame,
        f"Focus: {focus_ratio:.2f}",
        (20, 40),
        cv2.FONT_HERSHEY_DUPLEX,
        0.7,
        color,
        1,
    )
    cv2.putText(
        frame,
        f"TIME: {elapsed_sec // 60:02d}:{elapsed_sec % 60:02d}",
        (20, 70),
        cv2.FONT_HERSHEY_DUPLEX,
        0.7,
        (255, 255, 255),
        1,
    )
    cv2.rectangle(frame, (20, 90), (220, 105), (50, 50, 50), -1)
    bar_width = int(200 * focus_ratio)
    cv2.rectangle(frame, (20, 90), (20 + bar_width, 105), (0, 255, 255), -1)


def run_study_session_cli(root: Path, camera_index: int = 0) -> None:
    """Original behavior: OpenCV window + keyboard quit."""
    try:
        model = joblib.load(root / "focus_model.pkl")
        sp = get_spotify_client()
        print("Models and Spotify linked successfully.")
    except Exception as e:
        print(f"Initialization Error: {e}")
        sys.exit(1)
    task_path = root / "face_landmarker.task"
    buffer_size = 30
    prediction_buffer: deque[int] = deque(maxlen=buffer_size)
    current_music_state = "playing"
    session_log: list[dict] = []
    session_start_time = time.time()
    start_time = time.time()

    def live_callback(result, output_image, timestamp_ms):
        nonlocal current_music_state
        if not result.face_blendshapes:
            return
        features = [blendshapes_to_feature_row(model, result.face_blendshapes[0])]
        pred = int(model.predict(features)[0])
        prediction_buffer.append(pred)
        focus_ratio = prediction_buffer.count(FOCUSED_CLASS) / len(prediction_buffer)
        if focus_ratio < 0.2 and current_music_state == "playing":
            try:
                sp.start_playback(context_uri="spotify:playlist:3QxVnk4pmbhtVCIw2VKews")
                current_music_state = "playing"
            except Exception:
                pass
        elif focus_ratio > 0.8 and current_music_state == "playing":
            try:
                sp.start_playback(context_uri="spotify:playlist:5F724GJnhZOPgokLtmIJBR")
                current_music_state = "playing"
            except Exception:
                pass
        focus_value = 1 if pred == FOCUSED_CLASS else 0
        session_log.append(
            {"timestamp": time.time() - session_start_time, "focus_score": focus_value}
        )

    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(task_path)),
        running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
        result_callback=live_callback,
        output_face_blendshapes=True,
    )

    cap = cv2.VideoCapture(camera_index)
    try:
        with mp.tasks.vision.FaceLandmarker.create_from_options(options) as landmarker:
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break
                frame = cv2.flip(frame, 1)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
                landmarker.detect_async(mp_image, int(time.time() * 1000))
                fr = (
                    prediction_buffer.count(FOCUSED_CLASS) / len(prediction_buffer)
                    if prediction_buffer
                    else 0.5
                )
                draw_hud(frame, fr, int(time.time() - start_time))
                cv2.imshow("Study tracker with music", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    if session_log:
        df_session = pd.DataFrame(session_log)
        df_session.to_csv(root / "session_results.csv", index=False)
        print(f"\n[SUCCESS] Session saved with {len(df_session)} data points.")
        print("Run 'python report.py' to view your focus graph.")


class StudySessionThread(QThread):
    frame_ready = pyqtSignal(QImage)
    log_message = pyqtSignal(str)
    session_saved = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, root: Path, camera_index: int = 0, parent=None) -> None:
        super().__init__(parent)
        self._root = root
        self._camera_index = camera_index
        self._stop = False

    def request_stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        try:
            model = joblib.load(self._root / "focus_model.pkl")
            sp = get_spotify_client()
            self.log_message.emit("Models and Spotify linked successfully.")
        except Exception as e:
            self.failed.emit(str(e))
            return

        task_path = self._root / "face_landmarker.task"
        buffer_size = 30
        prediction_buffer: deque[int] = deque(maxlen=buffer_size)
        current_music_state = "playing"
        session_log: list[dict] = []
        session_start_time = time.time()
        start_time = time.time()

        def live_callback(result, output_image, timestamp_ms):
            nonlocal current_music_state
            if not result.face_blendshapes:
                return
            features = [blendshapes_to_feature_row(model, result.face_blendshapes[0])]
            pred = int(model.predict(features)[0])
            prediction_buffer.append(pred)
            focus_ratio = prediction_buffer.count(FOCUSED_CLASS) / len(prediction_buffer)
            if focus_ratio < 0.2 and current_music_state == "playing":
                try:
                    sp.start_playback(context_uri="spotify:playlist:3QxVnk4pmbhtVCIw2VKews")
                    current_music_state = "playing"
                except Exception:
                    pass
            elif focus_ratio > 0.8 and current_music_state == "playing":
                try:
                    sp.start_playback(context_uri="spotify:playlist:5F724GJnhZOPgokLtmIJBR")
                    current_music_state = "playing"
                except Exception:
                    pass
            focus_value = 1 if pred == FOCUSED_CLASS else 0
            session_log.append(
                {"timestamp": time.time() - session_start_time, "focus_score": focus_value}
            )

        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(task_path)),
            running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
            result_callback=live_callback,
            output_face_blendshapes=True,
        )

        cap = cv2.VideoCapture(self._camera_index)
        try:
            with mp.tasks.vision.FaceLandmarker.create_from_options(options) as landmarker:
                while cap.isOpened() and not self._stop:
                    success, frame = cap.read()
                    if not success:
                        break
                    frame = cv2.flip(frame, 1)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
                    landmarker.detect_async(mp_image, int(time.time() * 1000))
                    fr = (
                        prediction_buffer.count(FOCUSED_CLASS) / len(prediction_buffer)
                        if prediction_buffer
                        else 0.5
                    )
                    draw_hud(frame, fr, int(time.time() - start_time))
                    self.frame_ready.emit(bgr_to_qimage(frame))
        finally:
            cap.release()

        if session_log:
            out = self._root / "session_results.csv"
            pd.DataFrame(session_log).to_csv(out, index=False)
            self.session_saved.emit(f"Saved {len(session_log)} samples to {out.name}.")
        else:
            self.log_message.emit("No session data captured.")


def run_collect_data_cli(root: Path, label: str, camera_index: int = 0) -> None:
    task_path = root / "face_landmarker.task"
    captured_data: list[dict] = []

    def process_result(result, output_image, timestamp_ms):
        if not result.face_blendshapes:
            return
        scores = {b.category_name: b.score for b in result.face_blendshapes[0]}
        scores["label"] = label
        captured_data.append(scores)

    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(task_path)),
        running_mode=VisionRunningMode.LIVE_STREAM,
        result_callback=process_result,
        output_face_blendshapes=True,
    )

    cap = cv2.VideoCapture(camera_index)
    try:
        with FaceLandmarker.create_from_options(options) as landmarker:
            print(f"Recording state: {label.upper()}. Press 'q' to stop and save.")
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break
                frame = cv2.flip(frame, 1)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
                landmarker.detect_async(mp_image, int(time.time() * 1000))
                cv2.putText(
                    frame,
                    f"State: {label}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )
                cv2.imshow("Phase 1: Feature Extraction", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    if captured_data:
        out = root / f"data_{label}.csv"
        pd.DataFrame(captured_data).to_csv(out, index=False)
        print(f"Saved {len(captured_data)} frames to {out.name}")


class CollectDataThread(QThread):
    frame_ready = pyqtSignal(QImage)
    log_message = pyqtSignal(str)
    saved = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(
        self,
        root: Path,
        label: str,
        camera_index: int = 0,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._root = root
        self._label = label
        self._camera_index = camera_index
        self._stop = False

    def request_stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        task_path = self._root / "face_landmarker.task"
        captured_data: list[dict] = []

        def process_result(result, output_image, timestamp_ms):
            if not result.face_blendshapes:
                return
            scores = {b.category_name: b.score for b in result.face_blendshapes[0]}
            scores["label"] = self._label
            captured_data.append(scores)

        BaseOptions = mp.tasks.BaseOptions
        FaceLandmarker = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(task_path)),
            running_mode=VisionRunningMode.LIVE_STREAM,
            result_callback=process_result,
            output_face_blendshapes=True,
        )

        cap = cv2.VideoCapture(self._camera_index)
        try:
            with FaceLandmarker.create_from_options(options) as landmarker:
                while cap.isOpened() and not self._stop:
                    success, frame = cap.read()
                    if not success:
                        break
                    frame = cv2.flip(frame, 1)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
                    landmarker.detect_async(mp_image, int(time.time() * 1000))
                    cv2.putText(
                        frame,
                        f"State: {self._label}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2,
                    )
                    self.frame_ready.emit(bgr_to_qimage(frame))
        finally:
            cap.release()

        if captured_data:
            name = f"data_{self._label}.csv"
            out = self._root / name
            pd.DataFrame(captured_data).to_csv(out, index=False)
            self.saved.emit(f"Saved {len(captured_data)} frames to {name}.")
        else:
            self.log_message.emit("No frames captured.")
