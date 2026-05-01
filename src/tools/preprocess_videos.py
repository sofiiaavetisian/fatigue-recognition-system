import cv2
import os
import json
import random
import shutil
from src.pipelines.fatigue_classical import ClassicalFatigueDetector
import yaml

# 1. Load Configs
with open('configs/base.yaml', 'r') as f:
    full_config = yaml.safe_load(f)
with open('configs/rotation_map.json', 'r') as f:
    rotation_map = json.load(f)

# 2. Initialize the "Teacher" (Classical Detector)
if 'fatigue' in full_config:
    detector_config = full_config['fatigue']
else:
    detector_config = full_config

detector = ClassicalFatigueDetector(detector_config)

def get_rotation_code(v_name):
    angle = rotation_map.get(v_name, 0)
    if angle == 90: return cv2.ROTATE_90_CLOCKWISE
    if angle == 180: return cv2.ROTATE_180
    if angle == 270: return cv2.ROTATE_90_COUNTERCLOCKWISE
    return None

def run_extraction():
    raw_dir = "data/raw"
    proc_dir = "data/processed"
    
    for label in ['alert', 'fatigue']:
        path = os.path.join(proc_dir, label)
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path, exist_ok=True)

    videos = [f for f in os.listdir(raw_dir) if f.endswith((".MOV", ".mp4"))]
    print(f"Starting strict extraction from {len(videos)} videos...")

    for v_name in videos:
        cap = cv2.VideoCapture(os.path.join(raw_dir, v_name))
        rot_code = get_rotation_code(v_name)
        frame_idx = 0
        
        detector.eye_counter = 0
        detector.yawn_counter = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            if rot_code is not None:
                frame = cv2.rotate(frame, rot_code)

            # Analyze every frame so the counters work properly
            score = detector.analyze(frame)
            # We only save every 10th frame to keep data manageable
            if frame_idx % 10 == 0:
                label = None
                if score > 0.85: 
                    label = "fatigue"
                elif score < 0.05: 
                    label = "alert"
                
                if label:
                    img_name = f"{v_name.split('.')[0]}_f{frame_idx}.jpg"
                    cv2.imwrite(os.path.join(proc_dir, label, img_name), frame)
            
            frame_idx += 1
        cap.release()
        print(f"Done processing: {v_name}")

    print("Strict extraction complete!")

def create_splits():
    proc_dir = "data/processed"
    split_dir = "data/splits"
    
    for s in ['train', 'val', 'test']:
        for l in ['alert', 'fatigue']:
            os.makedirs(os.path.join(split_dir, s, l), exist_ok=True)

    for label in ['alert', 'fatigue']:
        images = os.listdir(os.path.join(proc_dir, label))
        random.shuffle(images)
        
        # 70% Train, 15% Val, 15% Test
        train_idx = int(len(images) * 0.7)
        val_idx = int(len(images) * 0.85)
        
        splits = {
            'train': images[:train_idx],
            'val': images[train_idx:val_idx],
            'test': images[val_idx:]
        }
        
        for split_name, split_imgs in splits.items():
            for img in split_imgs:
                src = os.path.join(proc_dir, label, img)
                dst = os.path.join(split_dir, split_name, label, img)
                shutil.copy(src, dst)
    print(f"Splits created in {split_dir}")

if __name__ == "__main__":
    run_extraction()
    create_splits()