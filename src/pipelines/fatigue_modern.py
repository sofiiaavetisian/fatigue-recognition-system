from __future__ import annotations
import os
import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
import numpy as np
from collections import deque

FACE_PADDING_FRAC = 0.12


def crop_face(frame: np.ndarray, bbox: tuple[int, int, int, int] | None,
              padding_frac: float = FACE_PADDING_FRAC) -> np.ndarray | None:
    """Crop a padded face region from a BGR frame. Returns None if invalid."""
    if frame is None or bbox is None:
        return None
    x1, y1, x2, y2 = bbox
    h, w = frame.shape[:2]
    bw = max(x2 - x1, 1)
    bh = max(y2 - y1, 1)
    pad_x = int(padding_frac * bw)
    pad_y = int(padding_frac * bh)
    x1 = max(x1 - pad_x, 0)
    y1 = max(y1 - pad_y, 0)
    x2 = min(x2 + pad_x, w)
    y2 = min(y2 + pad_y, h)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = frame[y1:y2, x1:x2]
    return crop if crop.size > 0 else None


class ModernFatigueDetector:
    def __init__(self, config: dict):
        # 1. Configuration & Smoothing Setup
        self.cfg = config.get('modern', config)
        self.threshold = self.cfg.get('threshold', 0.5)

        # Buffer for stability (averages the last N frames)
        window_size = self.cfg.get('smoothing_window', 5)
        self.score_buffer = deque(maxlen=window_size)

        # 2. Device Selection (CPU keeps us portable across CI and laptops)
        self.device = torch.device("cpu")

        # 3. Image preprocessing for the face crop
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # 4. Initialize Architecture
        self.model = models.mobilenet_v3_small(weights=None)
        num_features = self.model.classifier[3].in_features
        self.model.classifier[3] = nn.Linear(num_features, 1)

        # 5. Load the Trained Weights
        model_path = self.cfg.get('model_path', 'models/fatigue_model.pt')
        self.model_loaded = False
        if os.path.exists(model_path):
            try:
                state_dict = torch.load(model_path, map_location=self.device)
                if isinstance(state_dict, torch.nn.Module):
                    state_dict = state_dict.state_dict()
                self.model.load_state_dict(state_dict)
                self.model_loaded = True
            except Exception as e:
                print(f"Error loading model weights from {model_path}: {e}")
        else:
            print(f"WARNING: No model found at {model_path}; modern detector will return 0.0")

        self.model.to(self.device)
        self.model.eval()

    def reset(self) -> None:
        self.score_buffer.clear()

    def analyze(self, frame: np.ndarray, face_bbox: tuple[int, int, int, int] | None = None) -> float:
        """
        Run inference on the face crop. If no face_bbox is provided, falls back
        to the whole frame (for backwards compat / debug only).
        """
        if frame is None or not self.model_loaded:
            return 0.0
        try:
            face = crop_face(frame, face_bbox) if face_bbox is not None else frame
            if face is None or face.size == 0:
                # No usable face crop → don't add a score; let smoothing decay.
                if self.score_buffer:
                    return sum(self.score_buffer) / len(self.score_buffer)
                return 0.0

            rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            img_t = self.transform(rgb).unsqueeze(0).to(self.device)

            with torch.no_grad():
                logits = self.model(img_t)
                probability = torch.sigmoid(logits).item()

            self.score_buffer.append(probability)
            return sum(self.score_buffer) / len(self.score_buffer)
        except Exception as e:
            print(f"ModernFatigueDetector.analyze error: {e}")
            return 0.0

# Singleton pattern for the app (legacy callers; whole-frame fallback only)
_detector = None

def modern_fatigue_score(frame: np.ndarray, settings: dict = None,
                         face_bbox: tuple[int, int, int, int] | None = None) -> float:
    global _detector
    if _detector is None and settings is not None:
        _detector = ModernFatigueDetector(settings)
    if _detector:
        return _detector.analyze(frame, face_bbox=face_bbox)
    return 0.0