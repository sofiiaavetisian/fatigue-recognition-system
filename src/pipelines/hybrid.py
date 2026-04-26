from __future__ import annotations


def fuse_scores(classical_score: float, modern_score: float, classical_weight: float = 0.45) -> float:
    modern_weight = 1.0 - classical_weight
    return classical_weight * classical_score + modern_weight * modern_score
