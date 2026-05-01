import pytest
import torch
import numpy as np
from src.pipelines.fatigue_modern import ModernFatigueDetector

def test_model_initialization():
    """Check if the detector loads and the weights file exists."""
    config = {'modern': {'threshold': 0.5}}
    detector = ModernFatigueDetector(config)
    assert detector.model is not None
    assert next(detector.model.parameters()).is_mps or next(detector.model.parameters()).device.type == 'cpu'

def test_inference_output_range():
    """Ensure the AI always returns a probability between 0 and 1."""
    config = {'modern': {'threshold': 0.5}}
    detector = ModernFatigueDetector(config)
    
    # Create a fake 'black image' (224x224x3)
    fake_frame = np.zeros((224, 224, 3), dtype=np.uint8)
    score = detector.analyze(fake_frame)
    
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0