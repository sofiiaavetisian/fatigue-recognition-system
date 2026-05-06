from __future__ import annotations
import os
import pickle

import cv2
import numpy as np


class ClassicalFatigueDetector:
    # MediaPipe Face Mesh Indices for Eyes, Mouth, and Head Pose
    LEFT_EYE = [362, 385, 387, 263, 373, 380]
    RIGHT_EYE = [33, 160, 158, 133, 153, 144]
    MOUTH = [13, 14, 78, 308] # Upper lip, Lower lip, Left corner, Right corner
    NOSE_TIP = 1
    CHIN = 152
    LEFT_EYE_OUTER = 263   # subject's left eye outer corner
    RIGHT_EYE_OUTER = 33   # subject's right eye outer corner
    LEFT_MOUTH = 308
    RIGHT_MOUTH = 78

    # Generic 3D face model (mm), origin at nose tip, +Y up. Standard reference.
    _MODEL_POINTS = np.array([
        (0.0, 0.0, 0.0),          # nose tip
        (0.0, -330.0, -65.0),     # chin
        (-225.0, 170.0, -135.0),  # left eye outer (subject's left)
        (225.0, 170.0, -135.0),   # right eye outer
        (-150.0, -150.0, -125.0), # left mouth corner
        (150.0, -150.0, -125.0),  # right mouth corner
    ], dtype=np.float64)

    def __init__(self, config: dict):
        # Handle cases where config is either the full dict or the 'fatigue' sub-dict
        self.cfg = config.get('classical', config)
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

        # Optional sklearn classifier trained from features.csv. If absent, we
        # fall back to the hand-tuned rule-based score.
        self.classifier_bundle: dict | None = None
        classifier_path = self.cfg.get('classifier_path', 'models/classical_classifier.pkl')
        if classifier_path and os.path.exists(classifier_path):
            try:
                with open(classifier_path, "rb") as f:
                    self.classifier_bundle = pickle.load(f)
            except Exception as e:
                print(f"Failed to load classical classifier {classifier_path}: {e}")
                self.classifier_bundle = None

    def reset_counters(self) -> None:
        self.eye_counter = 0
        self.yawn_counter = 0

    @staticmethod
    def _euclidean(a, b) -> float:
        return float(np.linalg.norm(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)))

    def _calculate_ear(self, eye_pts):
        v1 = self._euclidean(eye_pts[1], eye_pts[5])
        v2 = self._euclidean(eye_pts[2], eye_pts[4])
        h = self._euclidean(eye_pts[0], eye_pts[3])
        return (v1 + v2) / (2.0 * h)

    def _calculate_mar(self, mouth_pts):
        v = self._euclidean(mouth_pts[0], mouth_pts[1])
        h = self._euclidean(mouth_pts[2], mouth_pts[3])
        return v / h

    def _calculate_head_pitch_deg(self, pts: np.ndarray, frame_w: int, frame_h: int) -> float:
        """Pitch angle in degrees via solvePnP. Negative ~= chin toward chest (nodding forward)."""
        image_points = np.array([
            pts[self.NOSE_TIP],
            pts[self.CHIN],
            pts[self.LEFT_EYE_OUTER],
            pts[self.RIGHT_EYE_OUTER],
            pts[self.LEFT_MOUTH],
            pts[self.RIGHT_MOUTH],
        ], dtype=np.float64)

        focal_length = float(frame_w)
        camera_matrix = np.array([
            [focal_length, 0.0, frame_w / 2.0],
            [0.0, focal_length, frame_h / 2.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        success, rvec, tvec = cv2.solvePnP(
            self._MODEL_POINTS, image_points, camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            return 0.0

        rmat, _ = cv2.Rodrigues(rvec)
        proj = np.hstack((rmat, tvec))
        _, _, _, _, _, _, euler = cv2.decomposeProjectionMatrix(proj)
        # euler is shape (3, 1) in degrees: [pitch, yaw, roll]
        return float(np.asarray(euler).flatten()[0])

    def extract_features(self, frame) -> dict | None:
        """
        Extracts geometric features and a face bounding box from a frame.
        Returns None if no face is detected.
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        if not results.multi_face_landmarks:
            return None

        landmarks = results.multi_face_landmarks[0].landmark
        h, w, _ = frame.shape
        pts = np.array([(int(l.x * w), int(l.y * h)) for l in landmarks])

        avg_ear = (self._calculate_ear(pts[self.LEFT_EYE])
                   + self._calculate_ear(pts[self.RIGHT_EYE])) / 2.0
        mar = self._calculate_mar(pts[self.MOUTH])
        pitch_deg = self._calculate_head_pitch_deg(pts, w, h)

        xs = pts[:, 0]
        ys = pts[:, 1]
        bbox = (
            max(int(xs.min()), 0),
            max(int(ys.min()), 0),
            min(int(xs.max()), w - 1),
            min(int(ys.max()), h - 1),
        )

        return {
            "ear": float(avg_ear),
            "mar": float(mar),
            "pitch_deg": float(pitch_deg),
            "face_bbox": bbox,
            "frame_size": (w, h),
        }

    def analyze(self, frame) -> dict:
        """
        Calculates metrics and returns a dictionary for the Hybrid Pipeline.
        """
        feats = self.extract_features(frame)
        if feats is None:
            return {
                "fatigue_score": 0.0, "ear": 0.0, "mar": 0.0, "pitch_deg": 0.0,
                "eye_closed_progress": 0.0, "yawn_progress": 0.0,
                "face_bbox": None, "face_present": False,
            }

        avg_ear = feats["ear"]
        mar = feats["mar"]
        pitch_deg = feats["pitch_deg"]

        ear_thresh = self.cfg.get('ear_threshold', 0.25)
        consec_frames = max(int(self.cfg.get('ear_consec_frames', 15)), 1)
        if avg_ear < ear_thresh:
            self.eye_counter += 1
        else:
            self.eye_counter = 0
        eye_progress = min(self.eye_counter / consec_frames, 1.0)

        mar_thresh = self.cfg.get('mar_threshold', 0.5)
        yawn_frames = max(int(self.cfg.get('yawn_consec_frames', 20)), 1)
        if mar > mar_thresh:
            self.yawn_counter += 1
        else:
            self.yawn_counter = 0
        yawn_progress = min(self.yawn_counter / yawn_frames, 1.0)

        pitch_thresh_deg = float(self.cfg.get('pitch_nod_threshold_deg', 18.0))
        # Negative pitch ≈ chin to chest (nodding forward); positive ≈ head tilted back.
        # We treat magnitude beyond threshold as a fatigue/distraction cue.
        pitch_excess = max(abs(pitch_deg) - pitch_thresh_deg, 0.0) / pitch_thresh_deg

        rule_score = 0.7 * eye_progress
        if self.yawn_counter >= yawn_frames:
            rule_score += 0.3
        if pitch_excess > 0:
            rule_score += min(0.5, 0.5 * pitch_excess)
        rule_score = float(min(rule_score, 1.0))

        if self.classifier_bundle is not None:
            ml_score = self._classifier_score(
                avg_ear, mar, pitch_deg, eye_progress, yawn_progress
            )
            fatigue_score = ml_score
            score_source = self.classifier_bundle.get("model_kind", "ml")
        else:
            fatigue_score = rule_score
            score_source = "rule"

        return {
            "fatigue_score": float(fatigue_score),
            "rule_score": rule_score,
            "score_source": score_source,
            "ear": avg_ear,
            "mar": mar,
            "pitch_deg": pitch_deg,
            "eye_closed_progress": float(eye_progress),
            "yawn_progress": float(yawn_progress),
            "face_bbox": feats["face_bbox"],
            "face_present": True,
        }

    def _classifier_score(self, ear: float, mar: float, pitch_deg: float,
                          eye_progress: float, yawn_progress: float) -> float:
        bundle = self.classifier_bundle
        order = bundle.get("feature_order", ["ear", "mar", "pitch_deg",
                                             "eye_closed_progress", "yawn_progress"])
        feature_map = {
            "ear": ear, "mar": mar, "pitch_deg": pitch_deg,
            "eye_closed_progress": eye_progress, "yawn_progress": yawn_progress,
        }
        x = np.array([[feature_map[k] for k in order]], dtype=np.float64)
        scaler = bundle.get("scaler")
        if scaler is not None:
            x = scaler.transform(x)
        model = bundle["model"]
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(x)[0]
            classes = list(getattr(model, "classes_", [0, 1]))
            try:
                pos_idx = classes.index(1)
            except ValueError:
                pos_idx = -1
            return float(probs[pos_idx])
        # Hard fallback: classifier without predict_proba.
        return float(model.predict(x)[0])


# For backward compatibility with older scripts
_detector = None
def classical_fatigue_score(frame, settings=None) -> float:
    global _detector
    if _detector is None and settings is not None:
        _detector = ClassicalFatigueDetector(settings)

    if _detector:
        res = _detector.analyze(frame)
        return res["fatigue_score"]
    return 0.0