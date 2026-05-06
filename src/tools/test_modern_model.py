"""
Visual sanity check: run the modern fatigue detector on a video file with the
correct rotation and the proper face-crop pipeline.

Usage:
    python -m src.tools.test_modern_model --video data/raw/IMG_8221.MOV
"""
from __future__ import annotations

import argparse
import os

import cv2

from src.pipelines.fatigue_classical import ClassicalFatigueDetector
from src.pipelines.fatigue_modern import ModernFatigueDetector
from src.pipelines.hybrid import HybridFatigueDetector
from src.tools.preprocess_videos import get_rotation_code

DEFAULT_CONFIG = {
    'classical': {
        'ear_threshold': 0.25,
        'ear_consec_frames': 10,
        'mar_threshold': 0.55,
        'yawn_consec_frames': 10,
        'pitch_nod_threshold_deg': 18,
    },
    'modern': {'model_path': 'models/fatigue_model.pt', 'threshold': 0.6, 'smoothing_window': 6},
    'hybrid': {'weight_classical': 0.55, 'threshold': 0.45},
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--video", default="data/raw/IMG_8221.MOV")
    p.add_argument("--threshold", type=float, default=0.6)
    return p.parse_args()


def test_inference(video_path: str, threshold: float) -> None:
    if not os.path.exists(video_path):
        raise FileNotFoundError(video_path)

    import json
    rotation_map = {}
    if os.path.exists('configs/rotation_map.json'):
        with open('configs/rotation_map.json') as f:
            rotation_map = json.load(f)

    cap = cv2.VideoCapture(video_path)
    v_name = os.path.basename(video_path)
    rot_code = get_rotation_code(v_name, rotation_map)
    print(f"Testing on: {v_name} (Rotation: {rot_code})")

    classical = ClassicalFatigueDetector(DEFAULT_CONFIG)
    modern = ModernFatigueDetector(DEFAULT_CONFIG)
    hybrid = HybridFatigueDetector(classical, modern, DEFAULT_CONFIG)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if rot_code is not None:
            frame = cv2.rotate(frame, rot_code)

        result = hybrid.analyze(frame)
        score = result['combined_score']
        m_score = result['modern_score']
        c_score = result['classical_score']
        face_present = result.get('face_present', False)

        if not face_present:
            color = (0, 200, 255)
            status = "NO FACE"
        else:
            color = (0, 0, 255) if score > threshold else (0, 255, 0)
            status = "FATIGUE" if score > threshold else "ALERT"

        cv2.putText(frame, f"{status} hybrid={score:.2f} c={c_score:.2f} m={m_score:.2f}",
                    (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
        bbox = result.get('face_bbox')
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)

        display_frame = cv2.resize(frame, (960, 540))
        cv2.imshow("Modern AI Test (face-crop)", display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    args = parse_args()
    test_inference(args.video, args.threshold)
