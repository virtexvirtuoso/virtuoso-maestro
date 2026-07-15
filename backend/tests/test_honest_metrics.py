"""
Tests for the selection-bias / overfitting statistics added in
feature/honest-metrics-dsr-pbo.

Covers: probabilistic_sharpe_ratio, expected_max_sharpe, deflated_sharpe_ratio,
effective_trials, cscv_pbo, evaluate_batch_overfitting.

Known-answer anchors are used throughout (no mocks) so a passing test proves the
math, per the repo anti-cheating rule.
"""

import numpy as np
import pandas as pd
import pytest

from engine_v2.statistical_tests import (
    EULER_MASCHERONI,
    probabilistic_sharpe_ratio,
    expected_max_sharpe,
    deflated_sharpe_ratio,
    effective_trials,
    cscv_pbo,
    evaluate_batch_overfitting,
    OverfittingVerdict,
)


# ─── PSR ─────────────────────────────────────────────────────────────────────


def test_psr_equals_half_when_sr_equals_benchmark():
    # SR == benchmark, normal returns -> P(true SR > benchmark) == 0.5
    psr = probabilistic_sharpe_ratio(sr=0.10, n=250, skew=0.0, kurt=3.0, sr_benchmark=0.10)
    assert psr == pytest.approx(0.5, abs=1e-9)


def test_psr_normal_denominator_matches_closed_form():
    # skew=0, kurt=3 -> denom == sqrt(1 + 0.5*sr^2)
    from scipy.stats import norm

    sr, n = 0.12, 500
    denom = np.sqrt(1.0 + 0.5 * sr**2)
    expected = float(norm.cdf((sr * np.sqrt(n - 1)) / denom))
    got = probabilistic_sharpe_ratio(sr=sr, n=n, skew=0.0, kurt=3.0, sr_benchmark=0.0)
    assert got == pytest.approx(expected, abs=1e-12)


def test_psr_rejects_excess_kurtosis():
    # pandas .kurtosis() returns EXCESS (Normal==0); passing it must be caught
    with pytest.raises(ValueError, match="EXCESS"):
        probabilistic_sharpe_ratio(sr=0.1, n=250, skew=0.0, kurt=0.0)


def test_psr_requires_two_observations():
    with pytest.raises(ValueError):
        probabilistic_sharpe_ratio(sr=0.1, n=1, skew=0.0, kurt=3.0)


def test_psr_monotonic_increasing_in_sr():
    lo = probabilistic_sharpe_ratio(sr=0.05, n=250, skew=0.0, kurt=3.0)
    hi = probabilistic_sharpe_ratio(sr=0.20, n=250, skew=0.0, kurt=3.0)
    assert hi > lo


# ─── expected_max_sharpe ─────────────────────────────────────────────────────


def test_expected_max_sharpe_n2_coefficient():
    # For N=2: z1=ppf(0.5)=0, coeff = gamma * ppf(1 - 1/(2e)) ~= 0.5192
    var = 0.04  # sd of SR across trials = 0.2
    got = expected_max_sharpe(sr_variance=var, n_trials=2)
    assert got == pytest.approx(0.5192 * np.sqrt(var), abs=2e-3)


def test_expected_max_sharpe_grows_with_trials():
    var = 0.04
    a = expected_max_sharpe(var, 2)
    b = expected_max_sharpe(var, 20)
    c = expected_max_sharpe(var, 200)
    assert a < b < c


def test_expected_max_sharpe_requires_two_trials():
    with pytest.raises(ValueError):
        expected_max_sharpe(0.04, 1)


def test_euler_constant_sane():
    assert EULER_MASCHERONI == pytest.approx(0.5772, abs=1e-4)


# ─── DSR ─────────────────────────────────────────────────────────────────────


def test_dsr_never_exceeds_psr_zero():
    sr, n = 0.15, 500
    psr0 = probabilistic_sharpe_ratio(sr, n, 0.0, 3.0, sr_benchmark=0.0)
    dsr = deflated_sharpe_ratio(sr, n, 0.0, 3.0, sr_variance=0.02, n_trials=20)
    assert dsr <= psr0 + 1e-12


def test_dsr_decreases_as_more_trials_tested():
    sr, n = 0.15, 500
    few = deflated_sharpe_ratio(sr, n, 0.0, 3.0, sr_variance=0.02, n_trials=5)
    many = deflated_sharpe_ratio(sr, n, 0.0, 3.0, sr_variance=0.02, n_trials=100)
    assert many < few  # more selection -> harder to clear


# ─── effective_trials ────────────────────────────────────────────────────────


def test_effective_trials_identical_columns_collapse_to_one():
    rng = np.random.default_rng(0)
    base = rng.standard_normal(400)
    mat = np.column_stack([base, base, base, base])  # 4 identical -> 1 cluster
    assert effective_trials(mat) == 1


def test_effective_trials_independent_columns_stay_separate():
    rng = np.random.default_rng(1)
    mat = rng.standard_normal((800, 6))  # ~independent -> ~6 clusters
    n_eff = effective_trials(mat, corr_threshold=0.5)
    assert n_eff >= 5  # allow a rare spurious merge


def test_effective_trials_three_correlated_groups():
    rng = np.random.default_rng(2)
    cols = []
    for _ in range(3):
        g = rng.standard_normal(600)
        for _ in range(4):
            cols.append(g + 0.01 * rng.standard_normal(600))  # near-identical within group
    mat = np.column_stack(cols)  # 12 cols, 3 groups
    assert effective_trials(mat, corr_threshold=0.5) == 3


# ─── cscv_pbo ────────────────────────────────────────────────────────────────


def test_pbo_noise_centers_on_half():
    # Single-seed PBO has high variance (~+/-0.3); the invariant is that pure
    # noise centers on 0.5. Average across seeds to test that robustly.
    pbos = []
    for seed in range(12):
        rng = np.random.default_rng(seed)
        mat = rng.standard_normal((600, 20))  # pure noise, no persistent skill
        pbo, lam = cscv_pbo(mat, n_blocks=10)
        assert lam.size > 0
        pbos.append(pbo)
    assert 0.35 <= float(np.mean(pbos)) <= 0.65


def test_pbo_dominant_strategy_is_near_zero():
    rng = np.random.default_rng(4)
    noise = rng.standard_normal((600, 9)) * 0.01
    # one strategy with a strong, persistent positive drift
    dominant = 0.02 + rng.standard_normal((600, 1)) * 0.001
    mat = np.hstack([noise, dominant])
    pbo, _ = cscv_pbo(mat, n_blocks=10)
    assert pbo <= 0.05


def test_pbo_rejects_odd_blocks():
    mat = np.random.default_rng(5).standard_normal((600, 5))
    with pytest.raises(ValueError, match="even"):
        cscv_pbo(mat, n_blocks=9)


def test_pbo_rejects_thin_blocks():
    mat = np.random.default_rng(6).standard_normal((100, 5))  # 10 bars/block < 50
    with pytest.raises(ValueError, match="thin"):
        cscv_pbo(mat, n_blocks=10)


def test_pbo_requires_two_strategies():
    mat = np.random.default_rng(7).standard_normal((600, 1))
    with pytest.raises(ValueError):
        cscv_pbo(mat, n_blocks=10)


# ─── evaluate_batch_overfitting ──────────────────────────────────────────────


def _series(arr, idx):
    return pd.Series(arr, index=idx)


def test_batch_overfitting_end_to_end():
    rng = np.random.default_rng(8)
    idx = pd.date_range("2024-01-01", periods=600, freq="D")
    returns = {f"noise_{i}": _series(rng.standard_normal(600) * 0.01, idx) for i in range(8)}
    # one genuinely strong strategy
    returns["winner"] = _series(0.02 + rng.standard_normal(600) * 0.001, idx)

    verdict = evaluate_batch_overfitting(returns, batch_id="test-batch", n_blocks=10)
    assert isinstance(verdict, OverfittingVerdict)
    assert verdict.n_strategies == 9
    assert verdict.n_effective_trials >= 2
    # winner should have the highest DSR and low PBO for the batch
    assert verdict.dsr["winner"] == max(verdict.dsr.values())
    assert verdict.pbo <= 0.10
    assert "Overfitting Report" in verdict.report


def test_batch_overfitting_requires_two_strategies():
    idx = pd.date_range("2024-01-01", periods=100, freq="D")
    with pytest.raises(ValueError):
        evaluate_batch_overfitting({"only": _series(np.zeros(100), idx)})
