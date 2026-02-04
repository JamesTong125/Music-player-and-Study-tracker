import mediapipe as mp
import cv2
import time
import pandas as pd

# --- Configuration ---
MODEL_PATH = 'face_landmarker.task'
RECORDING_MODE = True  # Set to True to save data to CSV
label = "distracted"      # Change this to "distracted" when recording the second set

# --- MediaPipe Setup ---
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

captured_data = []

def process_result(result, output_image, timestamp_ms):
    if result.face_blendshapes:
        # Extract the 52 blendshape scores
        scores = {b.category_name: b.score for b in result.face_blendshapes[0]}
        scores['label'] = label # Tag this row (e.g., "focused")
        captured_data.append(scores)

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=process_result,
    output_face_blendshapes=True
)

# --- Video Loop ---
cap = cv2.VideoCapture(0)
with FaceLandmarker.create_from_options(options) as landmarker:
    print(f"Recording state: {label.upper()}. Press 'q' to stop and save.")
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success: break

        # VS Code Tip: If the window is too small, resize it here
        frame = cv2.flip(frame, 1) # Mirror view
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        
        # detect_async is used for LIVE_STREAM mode
        landmarker.detect_async(mp_image, int(time.time() * 1000))

        cv2.putText(frame, f"State: {label}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Phase 1: Feature Extraction', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# --- Save to VS Code Workspace ---
if RECORDING_MODE:
    df = pd.DataFrame(captured_data)
    df.to_csv(f'data_{label}.csv', index=False)
    print(f"Saved {len(df)} frames to data_{label}.csv")

cap.release()
cv2.destroyAllWindows()