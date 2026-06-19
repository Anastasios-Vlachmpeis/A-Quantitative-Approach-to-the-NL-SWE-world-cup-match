"""Proper scoring rules for 1X2 predictions (Plan 05)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from nlvswe.models.base import OUTCOMES_1X2

Outcome = str  # "home" | "draw" | "away"


def _as_prob_vector(probs_1x2: dict[str, float]) -> np.ndarray:
    return np.array([float(probs_1x2[k]) for k in OUTCOMES_1X2], dtype=float)


def _outcome_index(outcome: Outcome) -> int:
    try:
        return OUTCOMES_1X2.index(outcome)
    except ValueError as exc:
        raise ValueError(f"invalid outcome {outcome!r}; expected one of {OUTCOMES_1X2}") from exc


def ranked_probability_score(probs_1x2: dict[str, float], outcome: Outcome) -> float:
    """RPS for ordered 1X2 (home < draw < away). Lower is better."""
    p = _as_prob_vector(probs_1x2)
    idx = _outcome_index(outcome)
    cdf_pred = np.cumsum(p)[:-1]
    cdf_obs = np.array([1.0 if idx <= k else 0.0 for k in range(len(p) - 1)], dtype=float)
    return float(np.mean((cdf_pred - cdf_obs) ** 2))


def log_loss_1x2(probs_1x2: dict[str, float], outcome: Outcome) -> float:
    """Multiclass log loss for a single match. Lower is better."""
    p = max(float(probs_1x2[outcome]), 1e-15)
    return float(-np.log(p))


def brier_1x2(probs_1x2: dict[str, float], outcome: Outcome) -> float:
    """Multiclass Brier score for a single match. Lower is better."""
    p = _as_prob_vector(probs_1x2)
    y = np.zeros(len(OUTCOMES_1X2), dtype=float)
    y[_outcome_index(outcome)] = 1.0
    return float(np.sum((p - y) ** 2))


SCORERS = {
    "rps": ranked_probability_score,
    "log_loss": log_loss_1x2,
    "brier": brier_1x2,
}


@dataclass(frozen=True)
class ScoreSummary:
    metric: str
    mean: float
    ci_low: float
    ci_high: float
    n: int


def _per_match_scores(
    preds: pd.DataFrame,
    metric: str,
) -> np.ndarray:
    scorer = SCORERS[metric]
    values: list[float] = []
    for _, row in preds.iterrows():
        probs = {"home": row["p_home"], "draw": row["p_draw"], "away": row["p_away"]}
        values.append(scorer(probs, str(row["outcome"])))
    return np.asarray(values, dtype=float)


def bootstrap_ci(
    values: np.ndarray,
    *,
    n_samples: int,
    seed: int,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Return mean and (1-alpha) percentile CI via bootstrap."""
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.empty(n_samples, dtype=float)
    n = len(values)
    for i in range(n_samples):
        sample = values[rng.integers(0, n, size=n)]
        means[i] = sample.mean()
    point = float(values.mean())
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return point, lo, hi


def score_predictions(
    preds: pd.DataFrame,
    *,
    metrics: tuple[str, ...] = ("rps", "log_loss", "brier"),
    bootstrap_samples: int = 2000,
    seed: int = 0,
) -> pd.DataFrame:
    """Aggregate scores with bootstrap CIs, grouped by model."""
    if preds.empty:
        return pd.DataFrame(columns=["model", "metric", "mean", "ci_low", "ci_high", "n"])

    rows: list[dict] = []
    for model, group in preds.groupby("model", sort=True):
        for metric in metrics:
            values = _per_match_scores(group, metric)
            mean, lo, hi = bootstrap_ci(values, n_samples=bootstrap_samples, seed=seed)
            rows.append(
                {
                    "model": model,
                    "metric": metric,
                    "mean": mean,
                    "ci_low": lo,
                    "ci_high": hi,
                    "n": int(len(values)),
                }
            )
    return pd.DataFrame(rows)
