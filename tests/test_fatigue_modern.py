"""
Tests for ModernFatigueDetector. The CNN is replaced with a stub so these
tests run quickly and don't need a real .pt file.
"""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
import torch

from src.pipelines.fatigue_modern import ModernFatigueDetector, crop_face


@pytest.fixture
def cfg():
    return {'modern': {'threshold': 0.5, 'smoothing_window': 3, 'model_path': ''}}


def _make_detector(cfg, fixed_logit: float = 0.0) -> ModernFatigueDetector:
    """Construct a detector and replace its model with a stub returning a fixed logit."""
    det = ModernFatigueDetector(cfg)
    det.model_loaded = True

    class StubModel:
        def __call__(self, _):
            return torch.tensor([[fixed_logit]], dtype=torch.float32)
        def eval(self):
            return self
        def to(self, _):
            return self

    det.model = StubModel()
    return det


def test_crop_face_returns_subregion():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[40:60, 40:60] = 255
    crop = crop_face(frame, (40, 40, 60, 60), padding_frac=0.0)
    assert crop is not None
    assert crop.shape == (20, 20, 3)
    assert crop.mean() > 200


def test_crop_face_handles_invalid_bbox():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    assert crop_face(frame, None) is None
    assert crop_face(frame, (0, 0, 0, 0)) is None  # collapses to empty


def test_no_model_loaded_returns_zero(cfg):
    det = ModernFatigueDetector(cfg)  # model_path='' so model_loaded stays False
    assert det.model_loaded is False
    score = det.analyze(np.zeros((224, 224, 3), dtype=np.uint8), face_bbox=(0, 0, 100, 100))
    assert score == 0.0


def test_analyze_with_face_crop_returns_probability(cfg):
    det = _make_detector(cfg, fixed_logit=2.0)  # sigmoid(2) ~= 0.88
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    score = det.analyze(frame, face_bbox=(100, 100, 300, 300))
    assert 0.0 <= score <= 1.0
    # Should reflect the stubbed logit, not be zero
    assert score == pytest.approx(0.8807970, abs=1e-3)


def test_smoothing_window_averages_recent_scores(cfg):
    det = _make_detector(cfg, fixed_logit=10.0)  # sigmoid ~= 1.0
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    bbox = (50, 50, 200, 200)
    s1 = det.analyze(frame, face_bbox=bbox)
    s2 = det.analyze(frame, face_bbox=bbox)
    s3 = det.analyze(frame, face_bbox=bbox)
    assert all(0.99 < s < 1.0 + 1e-9 for s in (s1, s2, s3))
    assert len(det.score_buffer) == 3


def test_reset_clears_smoothing_buffer(cfg):
    det = _make_detector(cfg, fixed_logit=10.0)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    det.analyze(frame, face_bbox=(50, 50, 200, 200))
    assert len(det.score_buffer) > 0
    det.reset()
    assert len(det.score_buffer) == 0


def test_no_face_bbox_falls_back_to_full_frame(cfg):
    det = _make_detector(cfg, fixed_logit=0.0)  # sigmoid = 0.5
    frame = np.zeros((224, 224, 3), dtype=np.uint8)
    score = det.analyze(frame, face_bbox=None)
    assert score == pytest.approx(0.5, abs=1e-3)


def test_invalid_face_bbox_returns_buffered_or_zero(cfg):
    det = _make_detector(cfg, fixed_logit=0.0)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    # Bbox outside frame → crop_face returns None → no inference run.
    score = det.analyze(frame, face_bbox=(200, 200, 300, 300))
    assert score == 0.0
