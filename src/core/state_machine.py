from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from src.core.types import ActivationConfig


class ActivationState(str, Enum):
    INACTIVE = "inactive"
    LISTENING = "listening"
    ACTIVE = "active"
    COOLDOWN = "cooldown"


@dataclass
class ActivationSnapshot:
    state: ActivationState
    next_expected: str | None
    matched_count: int


class GestureActivationFSM:
    """Finite state machine for ordered gesture activation."""

    def __init__(self, config: ActivationConfig) -> None:
        if len(config.sequence) < 2:
            raise ValueError("Gesture sequence must have at least two gestures.")
        self.cfg = config
        self.reset()

    def reset(self) -> None:
        self.state = ActivationState.LISTENING
        self.sequence_index = 0
        self.sequence_start_ts: float | None = None
        self.last_match_ts: float | None = None

    def deactivate(self) -> None:
        self.reset()

    def snapshot(self) -> ActivationSnapshot:
        next_expected = None
        if self.state in (ActivationState.INACTIVE, ActivationState.LISTENING):
            next_expected = self.cfg.sequence[self.sequence_index]
        return ActivationSnapshot(
            state=self.state,
            next_expected=next_expected,
            matched_count=self.sequence_index,
        )

    def consume(self, gesture: str | None, ts: float) -> ActivationSnapshot:
        if self.state == ActivationState.ACTIVE:
            return self.snapshot()

        if self.sequence_start_ts is not None and ts - self.sequence_start_ts > self.cfg.sequence_timeout_sec:
            self.reset()

        if self.last_match_ts is not None and ts - self.last_match_ts > self.cfg.max_gap_between_gestures_sec:
            self.reset()

        if gesture is None:
            return self.snapshot()

        expected = self.cfg.sequence[self.sequence_index]
        if gesture == expected:
            if self.sequence_index == 0:
                self.sequence_start_ts = ts
            self.sequence_index += 1
            self.last_match_ts = ts

            if self.sequence_index == len(self.cfg.sequence):
                self.state = ActivationState.ACTIVE
            return self.snapshot()

        # Ignore duplicate emissions of the previously matched gesture.
        # This avoids accidental resets caused by detector jitter/frame flicker.
        if self.sequence_index > 0:
            prev = self.cfg.sequence[self.sequence_index - 1]
            if gesture == prev:
                self.last_match_ts = ts
                return self.snapshot()

        self.reset()
        return self.snapshot()
