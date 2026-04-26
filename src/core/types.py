from dataclasses import dataclass


@dataclass(frozen=True)
class ActivationConfig:
    sequence: list[str]
    sequence_timeout_sec: float
    max_gap_between_gestures_sec: float
    cooldown_sec: float
