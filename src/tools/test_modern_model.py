import cv2
import torch
import json
import os
from src.pipelines.fatigue_modern import modern_fatigue_score
from src.tools.preprocess_videos import get_rotation_code # Reuse your rotation logic

# 1. Setup
VIDEO_PATH = "data/raw/IMG_8221.MOV" 
CONFIG = {'modern': {'threshold': 0.6}} 

def test_inference():
    cap = cv2.VideoCapture(VIDEO_PATH)
    v_name = os.path.basename(VIDEO_PATH)
    rot_code = get_rotation_code(v_name) # Get the correct rotation from your JSON
    
    print(f"Testing Modern AI on: {v_name} (Rotation: {rot_code})")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # Fix 1: Rotate the frame so it's upright
        if rot_code is not None:
            frame = cv2.rotate(frame, rot_code)

        # Fix 2: The modern_fatigue_score needs to be called on a FACE CROP
        # For now, let's just see the score on the full frame, 
        # but in the real app, we will pass only the face.
        score = modern_fatigue_score(frame, CONFIG)
        
        # UI Overlay
        color = (0, 0, 255) if score > 0.6 else (0, 255, 0)
        status = "FATIGUE" if score > 0.6 else "ALERT"
        
        cv2.putText(frame, f"{status}: {score:.2f}", (50, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
        
        # Resize for display so it fits your screen
        display_frame = cv2.resize(frame, (960, 540))
        cv2.imshow("Modern AI Corrected Test", display_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_inference()