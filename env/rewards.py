from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from env.pipeline import PipelineState

EPS = 1e-4


def compute_reward(
    step_count:    int,
    max_steps:     int,
    done:          bool,
    progress:      int,
    target_length: int,
    prev_progress: int,
    pipeline:      "PipelineState",
    custom_tools:  dict,
) -> tuple[float, dict]:
    """
    Episode return is provably in (0, 1) for any possible episode.

    Per-step base = 1 / (2 * max_steps)  → sums to at most 0.5
    Terminal bonus (only on completion)   → at most 0.50
    Total return                          → always in (0, 1)
    """
    per_step = 1.0 / (2.0 * max_steps)

    efficiency_component  = 0.0
    quality_component     = 0.0
    compression_component = 0.0

    if done and progress >= target_length:
        efficiency = max(0.0, 1.0 - step_count / max_steps)
        efficiency_component = 0.35 * efficiency

        q = getattr(pipeline, "quality_score", 0.0)
        if q >= 0.90:
            quality_component = 0.10
        elif q >= 0.75:
            quality_component = 0.05

        if custom_tools and step_count < progress:
            compression_component = 0.05

    terminal = efficiency_component + quality_component + compression_component
    raw      = per_step + terminal
    reward   = float(f"{max(EPS, min(1.0 - EPS, raw)):.6f}")

    breakdown = {
        "per_step":              round(per_step, 6),
        "efficiency_component":  round(efficiency_component, 6),
        "quality_component":     round(quality_component, 6),
        "compression_component": round(compression_component, 6),
        "terminal":              round(terminal, 6),
        "raw":                   round(raw, 6),
        "total":                 reward,
    }
    return reward, breakdown
