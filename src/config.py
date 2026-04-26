from __future__ import annotations

from pathlib import Path

import yaml

from src.core.types import ActivationConfig


class Settings:
    def __init__(self, raw: dict) -> None:
        self.raw = raw

    @property
    def activation(self) -> ActivationConfig:
        a = self.raw["activation"]
        return ActivationConfig(
            sequence=list(a["sequence"]),
            sequence_timeout_sec=float(a["sequence_timeout_sec"]),
            max_gap_between_gestures_sec=float(a["max_gap_between_gestures_sec"]),
            cooldown_sec=float(a["cooldown_sec"]),
        )


def load_settings(path: str | Path) -> Settings:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Settings(raw)
