import cv2
import mediapipe as mp
import joblib
import time
import numpy as np
from collections import deque
#from spotify_helper import get_spotify_client


MODEL_PATH = 'focus_model.pkl'
TASK_PATH = 'face_landmarker.task'
BUFFER_SIZE = 30  # ~1 second of video
session_log = []
session_start_time = time.time()


try:

    model = joblib.load(MODEL_PATH)
    #sp = get_spotify_client()
    print("Models and Spotify linked successfully.")

except Exception as e:

    print(f"Initialization Error: {e}")
    exit()

prediction_buffer = deque(maxlen=BUFFER_SIZE)
current_music_state = "playing"
start_time = time.time()


def live_callback(result, output_image, timestamp_ms):

    global current_music_state

    if result.face_blendshapes:

        
        features = [[b.score for b in result.face_blendshapes[0]]]
        
       
        pred = model.predict(features)[0]
        prediction_buffer.append(pred)
        
        
        focus_ratio = prediction_buffer.count(0) / len(prediction_buffer)
        
        
        if focus_ratio < 0.2 and current_music_state == "playing":
            try:
                #sp.pause_playback()
                current_music_state = "paused"
            except: pass
        elif focus_ratio > 0.8 and current_music_state == "paused":
            try:
                #sp.start_playback()
                current_music_state = "playing"
            except: pass

        
        focus_value = 1 if pred == 0 else 0 
        session_log.append({

            "timestamp": time.time() - session_start_time, # Seconds since start
            "focus_score": focus_value

        })


def draw_hud(frame):
    h, w, _ = frame.shape
    
    
    focus_ratio = prediction_buffer.count(0) / len(prediction_buffer) if prediction_buffer else 0
    
    
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (280, 120), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    
    color = (0, 255, 0) if current_music_state == "playing" else (0, 0, 255)
    cv2.putText(frame, f"STATE: {current_music_state.upper()}", (20, 40), 
                cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 1)
    
    
    elapsed = int(time.time() - start_time)
    cv2.putText(frame, f"TIME: {elapsed//60:02d}:{elapsed%60:02d}", (20, 70), 
                cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1)

    
    cv2.rectangle(frame, (20, 90), (220, 105), (50, 50, 50), -1) # Background bar
    bar_width = int(200 * focus_ratio)
    cv2.rectangle(frame, (20, 90), (20 + bar_width, 105), (0, 255, 255), -1) # Progress bar


options = mp.tasks.vision.FaceLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path=TASK_PATH),
    running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
    result_callback=live_callback,
    output_face_blendshapes=True
)


cap = cv2.VideoCapture(0)
with mp.tasks.vision.FaceLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        success, frame = cap.read()
        if not success: break
        
        frame = cv2.flip(frame, 1)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        landmarker.detect_async(mp_image, int(time.time() * 1000))
        
        draw_hud(frame)
        
        cv2.imshow('Neural Focus DJ Pro', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()


if session_log:
    import pandas as pd
    df_session = pd.DataFrame(session_log)
    df_session.to_csv("session_results.csv", index=False)
    print(f"\n[SUCCESS] Session saved with {len(df_session)} data points.")
    print("Run 'python report.py' to view your focus graph.")