from __future__ import annotations
import cv2
import numpy as np
from scipy.spatial import distance as dist

class ClassicalFatigueDetector:
    # MediaPipe Face Mesh Indices for Eyes, Mouth, and Head Pose
    LEFT_EYE = [362, 385, 387, 263, 373, 380]
    RIGHT_EYE = [33, 160, 158, 133, 153, 144]
    MOUTH = [13, 14, 78, 308] # Upper lip, Lower lip, Left corner, Right corner
    NOSE_TIP = 1
    CHIN = 152
    FOREHEAD = 10

    def __init__(self, config):
        self.cfg = config['classical'] 
        self.eye_counter = 0
        self.yawn_counter = 0
        
        import mediapipe as mp
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True, 
            min_detection_confidence=0.5
        )

    def _calculate_ear(self, eye_pts):
        # Vertical distances: ||p2-p6|| and ||p3-p5||
        v1 = dist.euclidean(eye_pts[1], eye_pts[5])
        v2 = dist.euclidean(eye_pts[2], eye_pts[4])
        # Horizontal distance: ||p1-p4||
        h = dist.euclidean(eye_pts[0], eye_pts[3])
        return (v1 + v2) / (2.0 * h)

    def _calculate_mar(self, mouth_pts):
        v = dist.euclidean(mouth_pts[0], mouth_pts[1])
        h = dist.euclidean(mouth_pts[2], mouth_pts[3])
        return v / h

    def _calculate_head_pitch_ratio(self, pts):
        # Measures vertical face compression. As head nods down, nose-to-chin distance shrinks.
        face_height = dist.euclidean(pts[self.FOREHEAD], pts[self.CHIN])
        nose_to_chin = pts[self.CHIN][1] - pts[self.NOSE_TIP][1]
        return nose_to_chin / face_height

    def analyze(self, frame) -> float:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            return 0.0

        landmarks = results.multi_face_landmarks[0].landmark
        h, w, _ = frame.shape
        pts = np.array([(int(l.x * w), int(l.y * h)) for l in landmarks])

        # 1. EAR and MAR calculations[cite: 1]
        avg_ear = (self._calculate_ear(pts[self.LEFT_EYE]) + self._calculate_ear(pts[self.RIGHT_EYE])) / 2.0
        mar = self._calculate_mar(pts[self.MOUTH])
        pitch_ratio = self._calculate_head_pitch_ratio(pts)

        fatigue_score = 0.0

        # 2. Progressive Eye Score (Climbs as eyes stay closed)
        if avg_ear < self.cfg['ear_threshold']:
            self.eye_counter += 1
            # Score grows linearly toward 0.70 until frame limit is hit[cite: 14]
            progress = min(self.eye_counter / self.cfg['ear_consec_frames'], 1.0)
            fatigue_score += (0.7 * progress)
        else:
            self.eye_counter = 0

        # 3. Yawn Score (Adds 0.3 if threshold met)
        if mar > self.cfg['mar_threshold']:
            self.yawn_counter += 1
            if self.yawn_counter >= self.cfg['yawn_consec_frames']:
                fatigue_score += 0.3
        else:
            self.yawn_counter = 0

        # 4. Head Nod Score (Detection for drooping head)
        # Threshold 0.33 represents a ~20% vertical compression of the face
        if pitch_ratio < 0.33:
            fatigue_score += 0.5

        return min(fatigue_score, 1.0)

_detector = None
def classical_fatigue_score(frame, settings=None) -> float:
    global _detector
    if _detector is None and settings is not None:
        _detector = ClassicalFatigueDetector(settings)
    
    if _detector:
        return _detector.analyze(frame)
    return 0.0