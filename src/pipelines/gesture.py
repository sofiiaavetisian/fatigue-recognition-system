from __future__ import annotations

import json
import os
from dataclasses import dataclass
from math import hypot
from pathlib import Path

import cv2


@dataclass
class GestureDetection:
    label: str | None
    confidence: float


@dataclass
class ThumbsUpCalibration:
    thumb_tip_to_index_mcp_ratio_min: float = 1.12
    thumb_tip_wrist_to_ip_wrist_ratio_min: float = 1.03
    thumb_up_dy_max: float = -0.03
    thumb_verticality_min: float = 0.35
    thumb_tip_above_knuckle_offset: float = 0.015
    thumbs_up_score_threshold: float = 0.65

    @classmethod
    def from_json(cls, path: str | Path | None) -> "ThumbsUpCalibration":
        calib = cls()
        if not path:
            return calib
        p = Path(path)
        if not p.exists():
            return calib
        with p.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        for key, value in raw.items():
            if hasattr(calib, key):
                setattr(calib, key, float(value))
        return calib


class StableGestureEmitter:
    """Emit a gesture only once after N stable frames."""

    def __init__(self, min_stable_frames: int = 4) -> None:
        if min_stable_frames < 1:
            raise ValueError("min_stable_frames must be >= 1")
        self.min_stable_frames = min_stable_frames
        self._current_label: str | None = None
        self._count = 0
        self._emitted = False

    def update(self, label: str | None) -> str | None:
        if label is None:
            self._current_label = None
            self._count = 0
            self._emitted = False
            return None

        if label != self._current_label:
            self._current_label = label
            self._count = 1
            self._emitted = False
            return None

        self._count += 1
        if not self._emitted and self._count >= self.min_stable_frames:
            self._emitted = True
            return label

        return None


class HandGestureDetector:
    """MediaPipe-based rule classifier for thumbs_up, peace_sign, ok_sign."""

    THUMB_TIP = 4
    THUMB_IP = 3
    THUMB_MCP = 2
    INDEX_MCP = 5
    INDEX_PIP = 6
    INDEX_TIP = 8
    MIDDLE_MCP = 9
    MIDDLE_PIP = 10
    MIDDLE_TIP = 12
    RING_MCP = 13
    RING_PIP = 14
    RING_TIP = 16
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_TIP = 20
    WRIST = 0

    def __init__(
        self,
        max_num_hands: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        preferred_hand: str = "right",
        calibration_path: str | None = "models/thumbs_up_calibration.json",
    ) -> None:
        if preferred_hand not in {"right", "left", "any"}:
            raise ValueError("preferred_hand must be one of: right, left, any")
        self.preferred_hand = preferred_hand
        self.calibration = ThumbsUpCalibration.from_json(calibration_path)

        # Force CPU graph to reduce OpenGL-context issues in constrained runtimes.
        os.environ.setdefault("MEDIAPIPE_DISABLE_GPU", "1")
        import mediapipe as mp

        self._mp_hands = mp.solutions.hands
        try:
            self._hands = self._mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=max_num_hands,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
        except RuntimeError as exc:
            raise RuntimeError(
                "Failed to initialize MediaPipe Hands. This often happens in headless "
                "environments without an OpenGL context."
            ) from exc

    def close(self) -> None:
        self._hands.close()

    @staticmethod
    def _dist(a, b) -> float:
        return hypot(a.x - b.x, a.y - b.y)

    def _finger_extended(self, lm, mcp: int, pip: int, tip: int) -> bool:
        # A finger is considered extended if tip is clearly farther from MCP and wrist than PIP.
        tip_mcp = self._dist(lm[tip], lm[mcp])
        pip_mcp = self._dist(lm[pip], lm[mcp])
        tip_wrist = self._dist(lm[tip], lm[self.WRIST])
        pip_wrist = self._dist(lm[pip], lm[self.WRIST])
        return tip_mcp > pip_mcp * 1.15 and tip_wrist > pip_wrist * 1.05

    def _thumb_extended(self, lm) -> bool:
        tip = self._dist(lm[self.THUMB_TIP], lm[self.INDEX_MCP])
        ip = self._dist(lm[self.THUMB_IP], lm[self.INDEX_MCP])
        tip_wrist = self._dist(lm[self.THUMB_TIP], lm[self.WRIST])
        ip_wrist = self._dist(lm[self.THUMB_IP], lm[self.WRIST])
        return (
            tip > ip * self.calibration.thumb_tip_to_index_mcp_ratio_min
            and tip_wrist > ip_wrist * self.calibration.thumb_tip_wrist_to_ip_wrist_ratio_min
        )

    def _thumb_metrics(self, lm) -> dict[str, float]:
        dx = lm[self.THUMB_TIP].x - lm[self.THUMB_MCP].x
        dy = lm[self.THUMB_TIP].y - lm[self.THUMB_MCP].y
        verticality = abs(dy) / (abs(dx) + 1e-6)
        return {
            "dx": dx,
            "dy": dy,
            "verticality": verticality,
            "tip_y": lm[self.THUMB_TIP].y,
            "ip_y": lm[self.THUMB_IP].y,
            "index_mcp_y": lm[self.INDEX_MCP].y,
        }

    def _is_thumbs_up(self, lm, index_ext: bool, middle_ext: bool, ring_ext: bool, pinky_ext: bool) -> tuple[bool, float]:
        # Keep non-thumb fingers mostly folded.
        others_folded = (not index_ext) and (not middle_ext) and (not ring_ext) and (not pinky_ext)
        if not others_folded:
            return False, 0.0

        # Thumb direction should be generally upward in image coordinates.
        metrics = self._thumb_metrics(lm)
        upward = metrics["dy"] < self.calibration.thumb_up_dy_max
        mostly_vertical = metrics["verticality"] > self.calibration.thumb_verticality_min

        # Tip should be above the knuckle line, with a small tolerance.
        above_knuckles = metrics["tip_y"] < min(metrics["ip_y"], metrics["index_mcp_y"]) + self.calibration.thumb_tip_above_knuckle_offset

        # Confidence increases when thumb is clearly vertical/up.
        confidence = 0.0
        if upward:
            confidence += 0.35
        if mostly_vertical:
            confidence += 0.25
        if above_knuckles:
            confidence += 0.25
        if self._thumb_extended(lm):
            confidence += 0.15

        return confidence >= self.calibration.thumbs_up_score_threshold, min(confidence, 1.0)

    def _classify(self, lm) -> GestureDetection:
        scale = max(self._dist(lm[self.WRIST], lm[self.MIDDLE_MCP]), 1e-6)

        index_ext = self._finger_extended(lm, self.INDEX_MCP, self.INDEX_PIP, self.INDEX_TIP)
        middle_ext = self._finger_extended(lm, self.MIDDLE_MCP, self.MIDDLE_PIP, self.MIDDLE_TIP)
        ring_ext = self._finger_extended(lm, self.RING_MCP, self.RING_PIP, self.RING_TIP)
        pinky_ext = self._finger_extended(lm, self.PINKY_MCP, self.PINKY_PIP, self.PINKY_TIP)
        thumb_ext = self._thumb_extended(lm)

        thumb_index_dist = self._dist(lm[self.THUMB_TIP], lm[self.INDEX_TIP])
        thumb_index_touch = thumb_index_dist < 0.30 * scale

        # OK sign: thumb-index touch + three fingers extended.
        if thumb_index_touch and middle_ext and ring_ext and pinky_ext:
            conf = max(0.0, min(1.0, 1.0 - (thumb_index_dist / (0.30 * scale))))
            return GestureDetection("ok_sign", conf)

        # Peace sign: index and middle extended, ring and pinky folded.
        if index_ext and middle_ext and (not ring_ext) and (not pinky_ext):
            conf = 0.85
            if thumb_index_touch:
                conf = 0.65
            return GestureDetection("peace_sign", conf)

        # Thumbs-up: allow moderate hand tilt instead of strict vertical-only pose.
        is_thumb, thumb_conf = self._is_thumbs_up(lm, index_ext, middle_ext, ring_ext, pinky_ext)
        if thumb_ext and is_thumb:
            return GestureDetection("thumbs_up", thumb_conf)

        return GestureDetection(None, 0.0)

    def _select_landmarks(self, results):
        if not results.multi_hand_landmarks:
            return None

        selected_idx = 0
        if self.preferred_hand != "any" and results.multi_handedness:
            wanted = self.preferred_hand.capitalize()
            for idx, handedness in enumerate(results.multi_handedness):
                label = handedness.classification[0].label
                if label == wanted:
                    selected_idx = idx
                    break

        return results.multi_hand_landmarks[selected_idx].landmark

    def detect(self, frame_bgr) -> GestureDetection:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self._hands.process(frame_rgb)
        lm = self._select_landmarks(results)
        if lm is None:
            return GestureDetection(None, 0.0)
        return self._classify(lm)

    def extract_thumb_metrics(self, frame_bgr) -> dict[str, float] | None:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self._hands.process(frame_rgb)
        lm = self._select_landmarks(results)
        if lm is None:
            return None
        return self._thumb_metrics(lm)
