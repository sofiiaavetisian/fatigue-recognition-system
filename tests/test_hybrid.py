import unittest
from unittest.mock import MagicMock
import numpy as np
# Ensure your imports match your project structure
from src.pipelines.hybrid import HybridFatigueDetector, fuse_scores

class TestHybridFatigue(unittest.TestCase):

    def setUp(self):
        """Set up mock detectors using your base.yaml values."""
        self.mock_classical = MagicMock()
        self.mock_modern = MagicMock()
        
        # Exact values from your base.yaml
        self.config = {
            'hybrid': {
                'weight_classical': 0.55,
                'weight_modern': 0.45,
                'threshold': 0.45
            }
        }
        
        self.hybrid = HybridFatigueDetector(
            self.mock_classical, 
            self.mock_modern, 
            self.config
        )

    def test_fusion_math_logic(self):
        """Verify the 0.55/0.45 weight split logic."""
        score = fuse_scores(1.0, 0.0, classical_weight=0.55)
        self.assertAlmostEqual(score, 0.55)

    def test_threshold_trigger(self):
        """Test if the 0.45 threshold triggers correctly."""
        
        self.mock_classical.analyze.return_value = {"fatigue_score": 0.5}
        self.mock_modern.analyze.return_value = 0.5
        
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result = self.hybrid.analyze(frame)
        
        self.assertTrue(result["is_fatigued"], "Should trigger alarm at 0.50 combined score")
        self.assertEqual(result["combined_score"], 0.5)

    def test_below_threshold(self):
        """Ensure it DOES NOT trigger when just below 0.45."""
        
        self.mock_classical.analyze.return_value = {"fatigue_score": 0.4}
        self.mock_modern.analyze.return_value = 0.4
        
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result = self.hybrid.analyze(frame)
        
        self.assertFalse(result["is_fatigued"], "Should NOT trigger alarm at 0.40 combined score")

if __name__ == "__main__":
    unittest.main()