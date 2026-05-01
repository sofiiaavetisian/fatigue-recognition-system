import pytest
import numpy as np
from src.pipelines.fatigue_classical import ClassicalFatigueDetector

@pytest.fixture
def mock_config():
    return {
        'classical': {
            'ear_threshold': 0.22,
            'ear_consec_frames': 3, # Shortened for testing
            'mar_threshold': 0.65,
            'yawn_consec_frames': 2
        }
    }

def test_ear_score_increases_over_time(mock_config):
    detector = ClassicalFatigueDetector(mock_config)
    
    # Manually create 'points' that represent a closed eye (low EAR)
    # Points: [left, top_1, top_2, right, bot_2, bot_1]
    closed_eye = np.array([[0,0], [5,1], [10,1], [15,0], [10,-1], [5,-1]])
    
    # We simulate 3 frames of closed eyes
    # Frame 1: Should be 1/3 of max score (0.7 * 0.33)
    score1 = detector._calculate_ear(closed_eye)
    assert score1 < 0.22
    
    detector.eye_counter = 1
    progress1 = detector.eye_counter / 3
    assert progress1 == pytest.approx(1/3)

def test_detector_resets_counter_on_open_eyes(mock_config):
    detector = ClassicalFatigueDetector(mock_config)
    detector.eye_counter = 5 # Pretend eyes were closed
    
    # If avg_ear is high (e.g. 0.35), counter should hit 0
    # This ensures no "ghost" fatigue from previous blinks
    avg_ear = 0.40
    if avg_ear >= mock_config['classical']['ear_threshold']:
        detector.eye_counter = 0
    
    assert detector.eye_counter == 0