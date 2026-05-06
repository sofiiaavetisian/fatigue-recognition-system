"""
Auto-label frames using the classical detector as a teacher and write:
  data/processed/{alert,fatigue}/*.jpg          - face crops for the modern CNN
  data/processed/features.csv                    - per-frame features for the
                                                   sklearn classical classifier
  data/splits/{train,val,test}/{alert,fatigue}/  - 70/15/15 split copies

Run:
    python -m src.tools.preprocess_videos
"""
from __future__ import annotations

import csv
import json
import os
import random
import shutil

import cv2
import yaml

from src.pipelines.fatigue_classical import ClassicalFatigueDetector
from src.pipelines.fatigue_modern import crop_face

RAW_DIR = "data/raw"
PROC_DIR = "data/processed"
SPLIT_DIR = "data/splits"
FEATURES_CSV = os.path.join(PROC_DIR, "features.csv")

POS_LABEL_THRESHOLD = 0.85   # frame is "fatigue" if classical score >= this
NEG_LABEL_THRESHOLD = 0.05   # frame is "alert" if classical score <= this
SAVE_EVERY_N = 10            # subsample to keep dataset manageable
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15  # test = remainder


def _load_configs():
    with open('configs/base.yaml', 'r') as f:
        full_config = yaml.safe_load(f)
    rotation_path = 'configs/rotation_map.json'
    rotation_map = {}
    if os.path.exists(rotation_path):
        with open(rotation_path, 'r') as f:
            rotation_map = json.load(f)
    detector_config = full_config.get('fatigue', full_config)
    return detector_config, rotation_map


def get_rotation_code(v_name: str, rotation_map: dict):
    angle = rotation_map.get(v_name, 0)
    if angle == 90:
        return cv2.ROTATE_90_CLOCKWISE
    if angle == 180:
        return cv2.ROTATE_180
    if angle == 270:
        return cv2.ROTATE_90_COUNTERCLOCKWISE
    return None


def _ensure_clean_dirs():
    for label in ['alert', 'fatigue']:
        path = os.path.join(PROC_DIR, label)
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path, exist_ok=True)


def run_extraction() -> None:
    detector_config, rotation_map = _load_configs()
    detector = ClassicalFatigueDetector(detector_config)
    _ensure_clean_dirs()

    videos = [f for f in os.listdir(RAW_DIR) if f.lower().endswith((".mov", ".mp4"))]
    print(f"Starting strict extraction from {len(videos)} videos...")

    feature_rows: list[dict] = []

    for v_name in videos:
        cap = cv2.VideoCapture(os.path.join(RAW_DIR, v_name))
        rot_code = get_rotation_code(v_name, rotation_map)
        frame_idx = 0
        # Per-video reset so EAR/yawn counters don't bleed across videos.
        detector.reset_counters()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if rot_code is not None:
                frame = cv2.rotate(frame, rot_code)

            results = detector.analyze(frame)
            score = results.get('fatigue_score', 0.0)
            face_bbox = results.get('face_bbox')

            if frame_idx % SAVE_EVERY_N == 0 and face_bbox is not None:
                label = None
                if score >= POS_LABEL_THRESHOLD:
                    label = "fatigue"
                elif score <= NEG_LABEL_THRESHOLD:
                    label = "alert"

                if label is not None:
                    face = crop_face(frame, face_bbox)
                    if face is not None and face.size > 0:
                        img_name = f"{os.path.splitext(v_name)[0]}_f{frame_idx}.jpg"
                        out_path = os.path.join(PROC_DIR, label, img_name)
                        cv2.imwrite(out_path, face)
                        feature_rows.append({
                            "image": f"{label}/{img_name}",
                            "label": 1 if label == "fatigue" else 0,
                            "ear": results.get('ear', 0.0),
                            "mar": results.get('mar', 0.0),
                            "pitch_deg": results.get('pitch_deg', 0.0),
                            "eye_closed_progress": results.get('eye_closed_progress', 0.0),
                            "yawn_progress": results.get('yawn_progress', 0.0),
                        })

            frame_idx += 1
        cap.release()
        print(f"Done processing: {v_name}")

    # Write features.csv
    if feature_rows:
        os.makedirs(PROC_DIR, exist_ok=True)
        with open(FEATURES_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(feature_rows[0].keys()))
            writer.writeheader()
            writer.writerows(feature_rows)
        print(f"Wrote {len(feature_rows)} feature rows to {FEATURES_CSV}")
    else:
        print("WARNING: No features extracted (no faces or no labels passed thresholds).")

    print("Strict extraction complete!")


def create_splits() -> None:
    for s in ['train', 'val', 'test']:
        for l in ['alert', 'fatigue']:
            os.makedirs(os.path.join(SPLIT_DIR, s, l), exist_ok=True)

    for label in ['alert', 'fatigue']:
        src_dir = os.path.join(PROC_DIR, label)
        if not os.path.isdir(src_dir):
            continue
        images = os.listdir(src_dir)
        random.shuffle(images)

        n = len(images)
        train_idx = int(n * TRAIN_FRAC)
        val_idx = int(n * (TRAIN_FRAC + VAL_FRAC))
        splits = {
            'train': images[:train_idx],
            'val': images[train_idx:val_idx],
            'test': images[val_idx:],
        }
        for split_name, split_imgs in splits.items():
            for img in split_imgs:
                src = os.path.join(src_dir, img)
                dst = os.path.join(SPLIT_DIR, split_name, label, img)
                shutil.copy(src, dst)
    print(f"Splits created in {SPLIT_DIR}")


if __name__ == "__main__":
    run_extraction()
    create_splits()
