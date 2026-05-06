from __future__ import annotations
import numpy as np


def fuse_scores(classical_score: float, modern_score: float, classical_weight: float = 0.55) -> float:
    modern_weight = 1.0 - classical_weight
    return (classical_weight * classical_score) + (modern_weight * modern_score)


class HybridFatigueDetector:
    """
    Coordinates classical + modern detectors. The classical detector runs Face Mesh
    once and produces both geometric features and a face bbox; the modern detector
    consumes the bbox to inference on a face crop only.
    """

    def __init__(self, classical_detector, modern_detector, config: dict):
        self.classical = classical_detector
        self.modern = modern_detector
        hybrid_cfg = config.get('hybrid', {}) if isinstance(config, dict) else {}
        self.c_weight = float(hybrid_cfg.get('weight_classical', 0.55))
        self.threshold = float(hybrid_cfg.get('threshold', 0.5))
        # Require this many consecutive above-threshold frames before raising
        # the alarm. Filters out 1-2 frame spikes from normal blinks.
        self.alarm_consec_frames = max(int(hybrid_cfg.get('alarm_consec_frames', 1)), 1)
        self._above_threshold_count = 0

    def reset(self) -> None:
        if hasattr(self.classical, 'reset_counters'):
            self.classical.reset_counters()
        if hasattr(self.modern, 'reset'):
            self.modern.reset()
        self._above_threshold_count = 0

    def analyze(self, frame: np.ndarray) -> dict:
        try:
            c_results = self.classical.analyze(frame)
            if not isinstance(c_results, dict):
                c_results = {}
            c_score = float(c_results.get('fatigue_score', 0.0))
            face_bbox = c_results.get('face_bbox')
            face_present = bool(c_results.get('face_present', face_bbox is not None))

            m_score = self.modern.analyze(frame, face_bbox=face_bbox) if face_present else 0.0

            combined_score = fuse_scores(c_score, m_score, self.c_weight)
            instantaneous_above = face_present and combined_score >= self.threshold
            if instantaneous_above:
                self._above_threshold_count += 1
            else:
                self._above_threshold_count = 0
            is_fatigued = self._above_threshold_count >= self.alarm_consec_frames

            return {
                "combined_score": float(combined_score),
                "classical_score": c_score,
                "modern_score": float(m_score),
                "is_fatigued": is_fatigued,
                "above_threshold_streak": int(self._above_threshold_count),
                "face_present": face_present,
                "face_bbox": face_bbox,
                "ear": c_results.get('ear', 0.0),
                "mar": c_results.get('mar', 0.0),
                "pitch_deg": c_results.get('pitch_deg', 0.0),
            }
        except Exception as e:
            print(f"HYBRID ERROR: {e}")
            self._above_threshold_count = 0
            return {
                "combined_score": 0.0, "classical_score": 0.0,
                "modern_score": 0.0, "is_fatigued": False,
                "above_threshold_streak": 0,
                "face_present": False, "face_bbox": None,
                "ear": 0.0, "mar": 0.0, "pitch_deg": 0.0,
            }
