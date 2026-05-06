from __future__ import annotations
import numpy as np

def fuse_scores(classical_score: float, modern_score: float, classical_weight: float = 0.45) -> float:
    modern_weight = 1.0 - classical_weight
    return (classical_weight * classical_score) + (modern_weight * modern_score)

class HybridFatigueDetector:
    def __init__(self, classical_detector, modern_detector, config: dict):
        self.classical = classical_detector
        self.modern = modern_detector
        self.c_weight = config.get('hybrid', {}).get('weight_classical', 0.45)
        self.threshold = config.get('hybrid', {}).get('threshold', 0.5)

    def analyze(self, frame: np.ndarray) -> dict: # Removed landmarks from here
        """
        Coordinates both detectors and fuses their results.
        """
        try:
            # 1. Get the Classical score
            # We call it with just the frame now
            c_results = self.classical.analyze(frame) 
            c_score = c_results.get('fatigue_score', 0.0) if isinstance(c_results, dict) else 0.0
            
            # 2. Get the Modern (AI) score
            m_score = self.modern.analyze(frame)
            
            # 3. Fuse them
            combined_score = fuse_scores(c_score, m_score, self.c_weight)
            is_fatigued = combined_score >= self.threshold
            
            return {
                "combined_score": combined_score,
                "classical_score": c_score,
                "modern_score": m_score,
                "is_fatigued": is_fatigued,
                "face_box": c_results.get("face_box") if isinstance(c_results, dict) else None,
            }
        except Exception as e:
            print(f"HYBRID ERROR: {e}")
            # Return zeros so the app doesn't crash
            return {
                "combined_score": 0.0, "classical_score": 0.0, 
                "modern_score": 0.0, "is_fatigued": False
            }