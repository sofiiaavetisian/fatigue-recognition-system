"""
Tests for ClassicalFatigueDetector. Face Mesh is mocked so these run without
a real camera or face image, and without depending on a specific mediapipe
version (the `mp.solutions` API was removed in newer mediapipe builds for
Python 3.13, but the project itself runs on 3.11 where it still exists).
"""
from __future__ import annotations

import sys
from unittest.mock import MagicMock

import numpy as np
import pytest

# Stub mediapipe before the SUT imports it so the tests run regardless of
# whether mediapipe is installed or which API it exposes.
_mp_stub = MagicMock()
_mp_stub.solutions.face_mesh.FaceMesh.return_value = MagicMock()
sys.modules["mediapipe"] = _mp_stub

from src.pipelines.fatigue_classical import ClassicalFatigueDetector  # noqa: E402


@pytest.fixture
def cfg():
    return {
        'classical': {
            'ear_threshold': 0.22,
            'ear_consec_frames': 3,
            'mar_threshold': 0.65,
            'yawn_consec_frames': 2,
            'pitch_nod_threshold_deg': 18,
            'classifier_path': '',  # disable classifier for these tests
        }
    }


def _make_detector(cfg):
    """Construct a detector. mediapipe is already stubbed at module import."""
    det = ClassicalFatigueDetector(cfg)
    det.face_mesh = MagicMock()
    return det


def _fake_face_mesh_results(landmarks_xy: np.ndarray):
    """Build the nested object Face Mesh returns."""
    landmarks = []
    for x, y in landmarks_xy:
        lm = MagicMock()
        lm.x = float(x)
        lm.y = float(y)
        lm.z = 0.0
        landmarks.append(lm)
    container = MagicMock()
    container.landmark = landmarks
    results = MagicMock()
    results.multi_face_landmarks = [container]
    return results


def _build_landmark_array(closed_eyes: bool = False, mouth_open: bool = False) -> np.ndarray:
    """
    Build 478 normalized landmarks for a synthetic upright face.
    We only set the indices the detector reads; the rest stay near (0.5, 0.5).
    """
    pts = np.full((478, 2), 0.5, dtype=np.float32)

    # Head pose anchors (rough, normalized 0..1)
    pts[1] = (0.50, 0.50)    # nose tip
    pts[152] = (0.50, 0.85)  # chin
    pts[263] = (0.62, 0.42)  # left eye outer
    pts[33] = (0.38, 0.42)   # right eye outer
    pts[308] = (0.58, 0.65)  # left mouth corner
    pts[78] = (0.42, 0.65)   # right mouth corner
    pts[10] = (0.50, 0.20)   # forehead

    # Right eye (RIGHT_EYE = [33, 160, 158, 133, 153, 144])
    eye_y = 0.42
    eye_h = 0.005 if closed_eyes else 0.04
    pts[33] = (0.34, eye_y)            # outer left point of right eye
    pts[160] = (0.36, eye_y - eye_h)
    pts[158] = (0.40, eye_y - eye_h)
    pts[133] = (0.42, eye_y)            # inner
    pts[153] = (0.40, eye_y + eye_h)
    pts[144] = (0.36, eye_y + eye_h)

    # Left eye (LEFT_EYE = [362, 385, 387, 263, 373, 380])
    pts[362] = (0.58, eye_y)
    pts[385] = (0.60, eye_y - eye_h)
    pts[387] = (0.64, eye_y - eye_h)
    pts[263] = (0.66, eye_y)
    pts[373] = (0.64, eye_y + eye_h)
    pts[380] = (0.60, eye_y + eye_h)

    # Mouth (MOUTH = [13, 14, 78, 308])
    mouth_h = 0.10 if mouth_open else 0.01
    pts[13] = (0.50, 0.65 - mouth_h / 2)  # upper lip
    pts[14] = (0.50, 0.65 + mouth_h / 2)  # lower lip
    pts[78] = (0.42, 0.65)                 # right corner
    pts[308] = (0.58, 0.65)                # left corner
    return pts


def test_no_face_returns_zero_score(cfg):
    det = _make_detector(cfg)
    empty_results = MagicMock()
    empty_results.multi_face_landmarks = None
    det.face_mesh.process.return_value = empty_results

    out = det.analyze(np.zeros((480, 640, 3), dtype=np.uint8))
    assert out["fatigue_score"] == 0.0
    assert out["face_present"] is False
    assert out["face_bbox"] is None


def test_open_eyes_keep_score_low(cfg):
    det = _make_detector(cfg)
    pts = _build_landmark_array(closed_eyes=False, mouth_open=False)
    det.face_mesh.process.return_value = _fake_face_mesh_results(pts)

    out = det.analyze(np.zeros((480, 640, 3), dtype=np.uint8))
    assert out["face_present"] is True
    assert out["fatigue_score"] < 0.3
    assert out["ear"] > cfg['classical']['ear_threshold']


def test_closed_eyes_accumulate_eye_counter(cfg):
    det = _make_detector(cfg)
    pts = _build_landmark_array(closed_eyes=True, mouth_open=False)
    det.face_mesh.process.return_value = _fake_face_mesh_results(pts)

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    for _ in range(cfg['classical']['ear_consec_frames']):
        out = det.analyze(frame)

    # After enough closed-eye frames, eye_progress saturates and the score
    # surpasses the open-eye baseline by a meaningful margin.
    assert det.eye_counter >= cfg['classical']['ear_consec_frames']
    assert out["eye_closed_progress"] == pytest.approx(1.0)
    assert out["fatigue_score"] >= 0.6


def test_open_eyes_resets_eye_counter(cfg):
    det = _make_detector(cfg)
    det.eye_counter = 5
    pts = _build_landmark_array(closed_eyes=False)
    det.face_mesh.process.return_value = _fake_face_mesh_results(pts)
    det.analyze(np.zeros((480, 640, 3), dtype=np.uint8))
    assert det.eye_counter == 0


def test_reset_counters_zeroes_state(cfg):
    det = _make_detector(cfg)
    det.eye_counter = 7
    det.yawn_counter = 4
    det.reset_counters()
    assert det.eye_counter == 0
    assert det.yawn_counter == 0


def test_face_bbox_is_within_frame(cfg):
    det = _make_detector(cfg)
    pts = _build_landmark_array()
    det.face_mesh.process.return_value = _fake_face_mesh_results(pts)
    out = det.analyze(np.zeros((480, 640, 3), dtype=np.uint8))
    x1, y1, x2, y2 = out["face_bbox"]
    assert 0 <= x1 < x2 <= 640 - 1
    assert 0 <= y1 < y2 <= 480 - 1
