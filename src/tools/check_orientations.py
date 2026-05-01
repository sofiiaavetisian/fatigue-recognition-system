import cv2
import os
import json

# Load existing map or create empty one
MAP_PATH = 'configs/rotation_map.json'
if os.path.exists(MAP_PATH):
    with open(MAP_PATH, 'r') as f:
        rotation_map = json.load(f)
else:
    rotation_map = {"DEFAULT": 0}

raw_dir = "data/raw"
videos = [f for f in os.listdir(raw_dir) if f.endswith((".MOV", ".mp4"))]

print("--- Orientation Checker ---")
print("Press 'r' to rotate 90 deg, 's' to save and next, 'q' to quit.")

for v in videos:
    cap = cv2.VideoCapture(os.path.join(raw_dir, v))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 60)
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()
        
    cap.release()
    if not ret: continue

    current_angle = rotation_map.get(v, 0)
    
    while True:
        # Apply current rotation for preview
        preview = frame.copy()
        if current_angle == 90: preview = cv2.rotate(preview, cv2.ROTATE_90_CLOCKWISE)
        elif current_angle == 180: preview = cv2.rotate(preview, cv2.ROTATE_180)
        elif current_angle == 270: preview = cv2.rotate(preview, cv2.ROTATE_90_COUNTERCLOCKWISE)

        cv2.imshow(f"Checking: {v}", cv2.resize(preview, (640, 480)))
        key = cv2.waitKey(0) & 0xFF
        
        if key == ord('r'): # Cycle through rotations
            current_angle = (current_angle + 90) % 360
        elif key == ord('s'): # Save this angle to map
            rotation_map[v] = current_angle
            break
        elif key == ord('q'):
            exit()

    cv2.destroyAllWindows()

with open(MAP_PATH, 'w') as f:
    json.dump(rotation_map, f, indent=2)
print(f"Map saved to {MAP_PATH}")