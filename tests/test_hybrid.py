"""
Tests for HybridFatigueDetector. The classical and modern detectors are mocked
so the hybrid logic is tested in isolation.
"""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock

import numpy as np

from src.pipelines.hybrid import HybridFatigueDetector, fuse_scores


class TestHybridFatigue(unittest.TestCase):
    def setUp(self):
        self.mock_classical = MagicMock()
        self.mock_modern = MagicMock()
        self.config = {
            'hybrid': {
                'weight_classical': 0.55,
                'weight_modern': 0.45,
                'threshold': 0.45,
                'alarm_consec_frames': 1,
            }
        }
        self.hybrid = HybridFatigueDetector(self.mock_classical, self.mock_modern, self.config)

    def test_fusion_math_logic(self):
        """0.55/0.45 split puts a pure-classical 1.0 at exactly 0.55."""
        score = fuse_scores(1.0, 0.0, classical_weight=0.55)
        self.assertAlmostEqual(score, 0.55)
        score2 = fuse_scores(0.0, 1.0, classical_weight=0.55)
        self.assertAlmostEqual(score2, 0.45)

    def test_threshold_trigger_when_face_present(self):
        self.mock_classical.analyze.return_value = {
            "fatigue_score": 0.5, "face_bbox": (10, 10, 100, 100), "face_present": True,
        }
        self.mock_modern.analyze.return_value = 0.5

        result = self.hybrid.analyze(np.zeros((100, 100, 3), dtype=np.uint8))
        self.assertTrue(result["is_fatigued"])
        self.assertEqual(result["combined_score"], 0.5)
        self.mock_modern.analyze.assert_called_once()
        # The bbox from classical should be forwarded to modern.
        _, kwargs = self.mock_modern.analyze.call_args
        self.assertEqual(kwargs.get("face_bbox"), (10, 10, 100, 100))

    def test_below_threshold(self):
        self.mock_classical.analyze.return_value = {
            "fatigue_score": 0.4, "face_bbox": (10, 10, 100, 100), "face_present": True,
        }
        self.mock_modern.analyze.return_value = 0.4

        result = self.hybrid.analyze(np.zeros((100, 100, 3), dtype=np.uint8))
        self.assertFalse(result["is_fatigued"])

    def test_no_face_skips_modern_and_returns_zero(self):
        self.mock_classical.analyze.return_value = {
            "fatigue_score": 0.0, "face_bbox": None, "face_present": False,
        }

        result = self.hybrid.analyze(np.zeros((100, 100, 3), dtype=np.uint8))
        self.assertFalse(result["is_fatigued"])
        self.assertFalse(result["face_present"])
        self.mock_modern.analyze.assert_not_called()

    def test_reset_calls_through_to_children(self):
        self.mock_classical.reset_counters = MagicMock()
        self.mock_modern.reset = MagicMock()
        self.hybrid.reset()
        self.mock_classical.reset_counters.assert_called_once()
        self.mock_modern.reset.assert_called_once()

    def test_alarm_requires_consecutive_above_threshold_frames(self):
        cfg = {'hybrid': {
            'weight_classical': 0.55, 'weight_modern': 0.45,
            'threshold': 0.45, 'alarm_consec_frames': 3,
        }}
        hybrid = HybridFatigueDetector(self.mock_classical, self.mock_modern, cfg)
        self.mock_classical.analyze.return_value = {
            "fatigue_score": 0.6, "face_bbox": (10, 10, 100, 100), "face_present": True,
        }
        self.mock_modern.analyze.return_value = 0.6
        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        # First two frames above threshold but streak too short → no alarm
        self.assertFalse(hybrid.analyze(frame)["is_fatigued"])
        self.assertFalse(hybrid.analyze(frame)["is_fatigued"])
        # Third frame finally trips the alarm
        self.assertTrue(hybrid.analyze(frame)["is_fatigued"])

    def test_below_threshold_resets_alarm_streak(self):
        cfg = {'hybrid': {
            'weight_classical': 0.55, 'weight_modern': 0.45,
            'threshold': 0.45, 'alarm_consec_frames': 3,
        }}
        hybrid = HybridFatigueDetector(self.mock_classical, self.mock_modern, cfg)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        # Two above-threshold frames, then a below-threshold frame, then two more above
        self.mock_classical.analyze.return_value = {
            "fatigue_score": 0.6, "face_bbox": (10, 10, 100, 100), "face_present": True,
        }
        self.mock_modern.analyze.return_value = 0.6
        hybrid.analyze(frame)
        hybrid.analyze(frame)
        self.mock_classical.analyze.return_value = {
            "fatigue_score": 0.1, "face_bbox": (10, 10, 100, 100), "face_present": True,
        }
        self.mock_modern.analyze.return_value = 0.1
        hybrid.analyze(frame)  # streak reset
        self.mock_classical.analyze.return_value = {
            "fatigue_score": 0.6, "face_bbox": (10, 10, 100, 100), "face_present": True,
        }
        self.mock_modern.analyze.return_value = 0.6
        self.assertFalse(hybrid.analyze(frame)["is_fatigued"])  # only 1 since reset
        self.assertFalse(hybrid.analyze(frame)["is_fatigued"])  # 2
        self.assertTrue(hybrid.analyze(frame)["is_fatigued"])   # 3


if __name__ == "__main__":
    unittest.main()
