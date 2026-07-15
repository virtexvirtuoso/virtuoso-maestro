"""
Statistical Tests for Walk-Forward Validation

Provides:
1. walk_forward_ttest — Tests whether OOS fold returns are significantly > 0
2. batch_fdr_correction — Benjamini-Hochberg FDR correction across a batch of strategies
3. evaluate_alpha_scout_batch — End-to-end: WalkForwardResults in, FDR verdicts out

Integration:
- Single strategy: walk_forward_ttest on its fold results
- Batch (Alpha Scout): evaluate_alpha_scout_batch after all WF runs complete
- The batch function is the one Maestro calls in synthesis

No external dependencies beyond scipy (already in venv).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
from itertools import combinations
from scipy.stats import ttest_1samp, norm

if TYPE_CHECKING:
    import pandas as pd
    from .walk_forward_optuna import WalkForwardResult

logger = logging.getLogger(__name__)


@dataclass
class WFTestResult:
    """Result of a walk-forward t-test on OOS fold returns."""

    strategy_name: str
    n_folds: int
    fold_sharpes: List[float]
    mean_sharpe: float
    std_sharpe: float
    t_statistic: float
    p_value: float  # one-sided: P(mean > 0)
    significant: bool  # at the given alpha before FDR
    alpha: float

    # Set after batch FDR correction
    p_value_adjusted: Optional[float] = None
    significant_after_fdr: Optional[bool] = None


def walk_forward_ttest(
    strategy_name: str,
    fold_sharpes: List[float],
    alpha: float = 0.05,
) -> WFTestResult:
    """
    Test whether a strategy's OOS fold Sharpe ratios are significantly > 0.

    Uses a one-sample t-test (H0: mean Sharpe = 0, H1: mean Sharpe > 0).
    Requires at least 3 folds for a meaningful test.

    Args:
        strategy_name: Identifier for the strategy
        fold_sharpes: List of OOS Sharpe ratios, one per walk-forward fold
        alpha: Significance level (default 0.05)

    Returns:
        WFTestResult with t-statistic, p-value, and significance verdict
    """
    n = len(fold_sharpes)
    if n < 3:
        return WFTestResult(
            strategy_name=strategy_name,
            n_folds=n,
            fold_sharpes=fold_sharpes,
            mean_sharpe=np.mean(fold_sharpes) if fold_sharpes else 0.0,
            std_sharpe=np.std(fold_sharpes, ddof=1) if n > 1 else 0.0,
            t_statistic=0.0,
            p_value=1.0,  # insufficient data = no significance
            significant=False,
            alpha=alpha,
        )

    arr = np.array(fold_sharpes)
    t_stat, p_two_sided = ttest_1samp(arr, 0.0)

    # One-sided p-value: we only care if mean > 0
    if t_stat > 0:
        p_one_sided = p_two_sided / 2.0
    else:
        p_one_sided = 1.0 - (p_two_sided / 2.0)

    return WFTestResult(
        strategy_name=strategy_name,
        n_folds=n,
        fold_sharpes=fold_sharpes,
        mean_sharpe=float(np.mean(arr)),
        std_sharpe=float(np.std(arr, ddof=1)),
        t_statistic=float(t_stat),
        p_value=float(p_one_sided),
        significant=p_one_sided < alpha,
        alpha=alpha,
    )


def batch_fdr_correction(
    results: List[WFTestResult],
    q: float = 0.05,
) -> List[WFTestResult]:
    """
    Apply Benjamini-Hochberg FDR correction across a batch of strategy test results.

    When Alpha Scout generates N hypotheses and each is walk-forward tested,
    this controls the expected proportion of false discoveries at level q.

    BH procedure:
    1. Sort p-values ascending: p_(1) <= p_(2) <= ... <= p_(m)
    2. Find largest k where p_(k) <= k/m * q
    3. Reject all H0 for i = 1..k

    Mutates results in-place (sets p_value_adjusted and significant_after_fdr)
    and returns them sorted by adjusted p-value.

    Args:
        results: List of WFTestResult from walk_forward_ttest
        q: FDR level (default 0.05 = expect ≤5% false discoveries)

    Returns:
        Same list, with p_value_adjusted and significant_after_fdr populated,
        sorted by adjusted p-value ascending
    """
    m = len(results)

    if m == 0:
        return results

    # Single hypothesis — no correction needed
    if m == 1:
        results[0].p_value_adjusted = results[0].p_value
        results[0].significant_after_fdr = results[0].significant
        return results

    # Sort by raw p-value
    sorted_results = sorted(results, key=lambda r: r.p_value)

    # BH adjusted p-values (step-up)
    # adjusted_p[i] = min(p[i] * m / (i+1), 1.0)
    # Then enforce monotonicity: adjusted_p[i] = min(adjusted_p[i], adjusted_p[i+1])
    raw_adjusted = [min(r.p_value * m / (i + 1), 1.0) for i, r in enumerate(sorted_results)]

    # Enforce monotonicity from right to left
    for i in range(m - 2, -1, -1):
        raw_adjusted[i] = min(raw_adjusted[i], raw_adjusted[i + 1])

    # Apply to results
    for i, r in enumerate(sorted_results):
        r.p_value_adjusted = raw_adjusted[i]
        r.significant_after_fdr = raw_adjusted[i] < q

    return sorted_results


def format_batch_report(results: List[WFTestResult], q: float = 0.05) -> str:
    """
    Format a batch of FDR-corrected results as a readable report.

    Args:
        results: List of WFTestResult (after batch_fdr_correction)
        q: FDR level used

    Returns:
        Formatted string report
    """
    lines = [
        f"=== Batch FDR Report (BH q={q}, m={len(results)}) ===",
        "",
        f"{'Strategy':<30} {'Folds':>5} {'Mean SR':>8} {'t-stat':>7} {'p-raw':>8} {'p-adj':>8} {'Verdict':>10}",
        "-" * 88,
    ]

    passed = 0
    for r in results:
        adj = r.p_value_adjusted if r.p_value_adjusted is not None else r.p_value
        verdict = "PASS" if r.significant_after_fdr else "FAIL"
        if r.significant_after_fdr:
            passed += 1
        lines.append(
            f"{r.strategy_name:<30} {r.n_folds:>5} {r.mean_sharpe:>8.3f} "
            f"{r.t_statistic:>7.2f} {r.p_value:>8.4f} {adj:>8.4f} {verdict:>10}"
        )

    lines.append("-" * 88)
    lines.append(f"Passed: {passed}/{len(results)} (FDR controlled at q={q})")

    return "\n".join(lines)


# ─── Synthesis Integration ───────────────────────────────────────────────────


def extract_fold_sharpes(wf_result: "WalkForwardResult") -> List[float]:
    """
    Extract per-fold OOS Sharpe ratios from a WalkForwardResult.

    This bridges the walk-forward engine output to the statistical test input.

    Args:
        wf_result: WalkForwardResult from WalkForwardOptuna.run()

    Returns:
        List of OOS Sharpe ratios, one per fold
    """
    return [r.sharpe_ratio for r in wf_result.fold_results if r.sharpe_ratio is not None]


@dataclass
class BatchVerdict:
    """Final verdict for an Alpha Scout batch after FDR correction."""

    batch_id: str
    batch_size: int
    n_validator_pass: int
    n_fdr_pass: int
    n_fdr_fail: int
    results: List[WFTestResult]
    report: str
    timestamp: str


def evaluate_alpha_scout_batch(
    strategies: Dict[str, List[float]],
    batch_id: str = "",
    q: float = 0.05,
    trace_dir: Optional[str] = None,
) -> BatchVerdict:
    """
    End-to-end FDR evaluation for an Alpha Scout batch.

    This is the function Maestro calls in synthesis after all Pattern B chains
    complete for a batch. It:
    1. Runs walk_forward_ttest on each strategy's fold Sharpes
    2. Applies batch_fdr_correction across all results
    3. Produces a verdict and optionally writes a trace log

    Args:
        strategies: Dict of strategy_name -> list of OOS fold Sharpe ratios.
            Only include strategies that passed the Validator gate individually.
            Validator FAILs should be excluded (already rejected).
        batch_id: Identifier for this batch (e.g., "AS-2026-05-22-daily")
        q: FDR level (default 0.05)
        trace_dir: If set, write JSONL trace to this directory

    Returns:
        BatchVerdict with full results and human-readable report

    Example:
        # After collecting Validator PASS results from a batch:
        strategies = {
            "MomentumTrendConfirm": [1.2, 0.8, 1.5, 0.9, 1.1, ...],  # fold Sharpes
            "FundingDiv": [0.6, 0.4, 0.8, ...],
            "LuckyNoise": [0.3, -0.1, 0.4, ...],  # barely passed Validator
        }
        verdict = evaluate_alpha_scout_batch(strategies, batch_id="AS-2026-05-22-daily")
        # verdict.n_fdr_pass might be 2 (LuckyNoise rejected by FDR)
    """
    if not batch_id:
        batch_id = f"batch-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"

    # Step 1: t-test each strategy
    test_results = []
    for name, fold_sharpes in strategies.items():
        result = walk_forward_ttest(name, fold_sharpes)
        test_results.append(result)

    # Step 2: FDR correction
    corrected = batch_fdr_correction(test_results, q=q)

    # Step 3: Build report
    report = format_batch_report(corrected, q=q)

    n_fdr_pass = sum(1 for r in corrected if r.significant_after_fdr)
    n_fdr_fail = len(corrected) - n_fdr_pass

    verdict = BatchVerdict(
        batch_id=batch_id,
        batch_size=len(strategies),
        n_validator_pass=len(strategies),  # all inputs already passed Validator
        n_fdr_pass=n_fdr_pass,
        n_fdr_fail=n_fdr_fail,
        results=corrected,
        report=report,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

    # Step 4: Trace log
    if trace_dir:
        _write_trace(verdict, trace_dir)

    logger.info(f"Batch {batch_id}: {n_fdr_pass}/{len(strategies)} survived FDR (q={q})")

    return verdict


def get_fdr_survivors(verdict: BatchVerdict) -> List[str]:
    """Get strategy names that survived FDR correction."""
    return [r.strategy_name for r in verdict.results if r.significant_after_fdr]


def get_fdr_rejects(verdict: BatchVerdict) -> List[Tuple[str, float]]:
    """Get strategy names rejected by FDR with their adjusted p-values."""
    return [(r.strategy_name, r.p_value_adjusted) for r in verdict.results if not r.significant_after_fdr]


def _write_trace(verdict: BatchVerdict, trace_dir: str):
    """Write FDR trace to JSONL for the quant team trace log."""
    try:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        trace_path = Path(trace_dir) / today
        trace_path.mkdir(parents=True, exist_ok=True)

        trace_file = trace_path / f"{verdict.batch_id}-fdr.jsonl"

        entries = []

        # Batch summary entry
        entries.append(
            {
                "type": "fdr_batch_summary",
                "batch_id": verdict.batch_id,
                "timestamp": verdict.timestamp,
                "batch_size": verdict.batch_size,
                "n_validator_pass": verdict.n_validator_pass,
                "n_fdr_pass": verdict.n_fdr_pass,
                "n_fdr_fail": verdict.n_fdr_fail,
                "q_level": 0.05,
            }
        )

        # Per-strategy entries
        for r in verdict.results:
            entries.append(
                {
                    "type": "fdr_strategy_result",
                    "batch_id": verdict.batch_id,
                    "strategy_name": r.strategy_name,
                    "n_folds": r.n_folds,
                    "mean_sharpe": round(r.mean_sharpe, 4),
                    "t_statistic": round(r.t_statistic, 4),
                    "p_value_raw": round(r.p_value, 6),
                    "p_value_adjusted": round(r.p_value_adjusted, 6) if r.p_value_adjusted is not None else None,
                    "significant_raw": bool(r.significant),
                    "significant_fdr": bool(r.significant_after_fdr) if r.significant_after_fdr is not None else None,
                    "verdict": "PASS" if r.significant_after_fdr else "FAIL",
                }
            )

        with open(trace_file, "a") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")

        logger.info(f"FDR trace written to {trace_file}")

    except Exception as e:
        logger.warning(f"Failed to write FDR trace: {e}")


# ─── Selection-Bias & Overfitting Statistics ─────────────────────────────────
#
# In-house implementations of the Bailey & Lopez de Prado overfitting toolkit.
# We deliberately do NOT import an external package for these (see the vault
# lit-review Walk-Forward-Methodology-Lit-Review-2026-06-30): the math is small,
# and gating capital deployment on an unvetted dependency is a supply-chain risk.
#
# References:
#   - Bailey & Lopez de Prado (2014), "The Deflated Sharpe Ratio",
#     Journal of Portfolio Management. doi:10.3905/jpm.2014.40.5.094
#   - Bailey, Borwein, Lopez de Prado & Zhu (2016),
#     "The Probability of Backtest Overfitting", J. Computational Finance.
#     doi:10.21314/jcf.2016.322
#
# IMPORTANT conventions (two common bugs these guard against):
#   1. `sr` is the PER-PERIOD (NON-annualized) Sharpe ratio. Do NOT pass the
#      annualized Sharpe that vectorbt reports — compute mean/std of periodic
#      returns instead. `evaluate_batch_overfitting` does this correctly.
#   2. `kurt` is NON-EXCESS kurtosis (Normal == 3). pandas `.kurtosis()` returns
#      EXCESS kurtosis (Normal == 0), so callers must pass `series.kurtosis() + 3`.
#      `probabilistic_sharpe_ratio` sanity-checks this.

EULER_MASCHERONI = 0.5772156649015329


def probabilistic_sharpe_ratio(
    sr: float,
    n: int,
    skew: float,
    kurt: float,
    sr_benchmark: float = 0.0,
) -> float:
    """
    Probabilistic Sharpe Ratio: PSR(sr*) = P(true SR > sr_benchmark).

    Bailey & Lopez de Prado (2014).

    Args:
        sr: Observed PER-PERIOD (non-annualized) Sharpe ratio.
        n: Number of return observations.
        skew: Skewness of the returns (Fisher; Normal == 0).
        kurt: NON-EXCESS kurtosis of the returns (Normal == 3).
        sr_benchmark: Benchmark Sharpe to beat (0 == "is it > 0").

    Returns:
        Probability in [0, 1] that the true Sharpe exceeds sr_benchmark.
    """
    if n < 2:
        raise ValueError("probabilistic_sharpe_ratio requires n >= 2 observations")
    if kurt < 1.0:
        # Non-excess kurtosis is >= 1 for any real distribution. A value < 1
        # almost certainly means excess kurtosis was passed (Normal == 0).
        raise ValueError(
            f"kurt={kurt} looks like EXCESS kurtosis; pass NON-excess "
            f"(Normal == 3, i.e. series.kurtosis() + 3)"
        )
    denom = np.sqrt(1.0 - skew * sr + ((kurt - 1.0) / 4.0) * sr ** 2)
    if denom <= 0 or not np.isfinite(denom):
        return float("nan")
    return float(norm.cdf(((sr - sr_benchmark) * np.sqrt(n - 1)) / denom))


def expected_max_sharpe(sr_variance: float, n_trials: int) -> float:
    """
    Expected maximum Sharpe ratio across `n_trials` INDEPENDENT strategies whose
    true Sharpe is zero (the deflation benchmark for the DSR).

    Bailey & Lopez de Prado (2014), expected-maximum-SR term.

    Args:
        sr_variance: Variance of the Sharpe ratios ACROSS the trial universe
            (i.e. across strategies), NOT across the folds of one strategy.
        n_trials: Effective number of independent trials (see effective_trials).

    Returns:
        Expected max Sharpe under the null. Grows with n_trials.
    """
    if n_trials < 2:
        raise ValueError("expected_max_sharpe requires n_trials >= 2")
    if sr_variance < 0:
        raise ValueError("sr_variance must be non-negative")
    g = EULER_MASCHERONI
    z1 = norm.ppf(1.0 - 1.0 / n_trials)
    z2 = norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    return float(np.sqrt(sr_variance) * ((1.0 - g) * z1 + g * z2))


def deflated_sharpe_ratio(
    sr: float,
    n: int,
    skew: float,
    kurt: float,
    sr_variance: float,
    n_trials: int,
) -> float:
    """
    Deflated Sharpe Ratio: PSR evaluated against the expected-max-Sharpe null.

    DSR > 0.95 is the usual bar for "survives selection bias across the trials
    that were run". See module references.

    Args:
        sr: Observed PER-PERIOD (non-annualized) Sharpe of the strategy.
        n: Number of return observations.
        skew: Skewness of returns (Normal == 0).
        kurt: NON-EXCESS kurtosis of returns (Normal == 3).
        sr_variance: Variance of Sharpe across the trial universe.
        n_trials: Effective number of independent trials.

    Returns:
        DSR in [0, 1]. Always <= PSR(sr_benchmark=0).
    """
    sr0 = expected_max_sharpe(sr_variance, n_trials)
    return probabilistic_sharpe_ratio(sr, n, skew, kurt, sr_benchmark=sr0)


def effective_trials(returns_matrix: np.ndarray, corr_threshold: float = 0.5) -> int:
    """
    Effective number of INDEPENDENT trials among correlated strategies.

    Passing the raw count of strategies tested (e.g. 66) to the DSR overstates
    independence when many strategies are variants on the same asset/signal.
    This clusters strategies by absolute return correlation (single-linkage via
    union-find at `corr_threshold`) and returns the number of clusters.

    Args:
        returns_matrix: (T, N) array — periodic returns, one column per strategy.
        corr_threshold: |corr| above which two strategies are deemed the same
            cluster (default 0.5).

    Returns:
        Number of correlation clusters (>= 1). This is a floor on the true
        breadth; it does not attempt to add Optuna internal-trial inflation.
    """
    R = np.asarray(returns_matrix, dtype=float)
    if R.ndim != 2:
        raise ValueError("returns_matrix must be 2-D (T, N)")
    _, n = R.shape
    if n <= 1:
        return int(n)

    C = np.corrcoef(R, rowvar=False)
    C = np.nan_to_num(C, nan=0.0)  # constant columns -> nan corr -> treat as uncorrelated

    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        for j in range(i + 1, n):
            if abs(C[i, j]) > corr_threshold:
                union(i, j)

    return len({find(i) for i in range(n)})


def cscv_pbo(returns_matrix: np.ndarray, n_blocks: int = 10) -> Tuple[float, np.ndarray]:
    """
    Probability of Backtest Overfitting via Combinatorially-Symmetric CV (CSCV).

    Bailey, Borwein, Lopez de Prado & Zhu (2016). Operates on a strategy x time
    return matrix — NO labels required, so it is valid for rule-based signals.

    Algorithm: split T rows into `n_blocks` disjoint equal blocks; for every
    combination of n_blocks/2 blocks as in-sample (IS), the complement is
    out-of-sample (OOS). Pick the best-IS strategy, find its OOS relative rank
    omega, take the logit lambda = ln(omega / (1 - omega)). PBO is the fraction
    of combinations where lambda <= 0 (best-IS lands in the bottom half OOS).

    Args:
        returns_matrix: (T, N) array — periodic returns, one column per strategy.
        n_blocks: Even number of blocks (default 10). Requires >= 50 bars/block.

    Returns:
        (pbo, lambdas) where pbo in [0, 1] and lambdas is the logit distribution.
    """
    R = np.asarray(returns_matrix, dtype=float)
    if R.ndim != 2:
        raise ValueError("returns_matrix must be 2-D (T, N)")
    T, N = R.shape
    if N < 2:
        raise ValueError("cscv_pbo requires N >= 2 strategies")
    if n_blocks % 2 != 0:
        raise ValueError("n_blocks must be even")
    if T // n_blocks < 50:
        raise ValueError(
            f"{T // n_blocks} bars/block is too thin for PBO (need >= 50); "
            f"reduce n_blocks or use more data"
        )

    block_size = T // n_blocks
    R = R[: block_size * n_blocks]  # trim ragged tail
    blocks = np.stack(np.split(R, n_blocks, axis=0))  # (n_blocks, block_size, N)
    half = n_blocks // 2
    all_idx = set(range(n_blocks))

    def col_sharpe(x: np.ndarray) -> np.ndarray:
        mu = x.mean(axis=0)
        sd = x.std(axis=0, ddof=1)
        sd = np.where(sd == 0, np.nan, sd)
        return mu / sd

    lambdas: List[float] = []
    for is_idx in combinations(range(n_blocks), half):
        oos_idx = tuple(sorted(all_idx - set(is_idx)))
        is_ret = np.concatenate([blocks[k] for k in is_idx], axis=0)
        oos_ret = np.concatenate([blocks[k] for k in oos_idx], axis=0)

        is_sr = col_sharpe(is_ret)
        oos_sr = col_sharpe(oos_ret)
        if np.all(np.isnan(is_sr)):
            continue
        n_star = int(np.nanargmax(is_sr))

        # OOS rank of the IS-best strategy: 1 (worst) .. N (best)
        ranks = np.argsort(np.argsort(np.nan_to_num(oos_sr, nan=-np.inf)))
        rank = int(ranks[n_star]) + 1
        omega = rank / (N + 1)
        omega = min(max(omega, 1e-9), 1.0 - 1e-9)
        lambdas.append(float(np.log(omega / (1.0 - omega))))

    lam = np.array(lambdas)
    pbo = float(np.mean(lam <= 0.0)) if lam.size else float("nan")
    return pbo, lam


@dataclass
class OverfittingVerdict:
    """DSR + PBO verdict for a batch of strategies evaluated together."""
    batch_id: str
    n_strategies: int
    n_effective_trials: int
    pbo: float
    dsr: Dict[str, float]                 # strategy_name -> DSR
    per_period_sharpe: Dict[str, float]   # strategy_name -> non-annualized SR
    n_obs: Dict[str, int]                 # strategy_name -> return count
    report: str
    timestamp: str


def evaluate_batch_overfitting(
    returns_by_strategy: "Dict[str, pd.Series]",
    batch_id: str = "",
    n_blocks: int = 10,
    corr_threshold: float = 0.5,
    dsr_threshold: float = 0.95,
) -> OverfittingVerdict:
    """
    Batch-level selection-bias evaluation: Deflated Sharpe per strategy + a
    single PBO for the batch.

    This is the correct entry point for the "is our best strategy real, or did
    we just test our way into it" question. It computes PER-PERIOD Sharpe from
    the return series (NOT the annualized figure), and uses the effective
    (correlation-clustered) trial count for the deflation — both of which the
    original upgrade spec got wrong.

    Args:
        returns_by_strategy: strategy_name -> periodic return Series (aligned on
            a shared DatetimeIndex; the intersection is used for PBO).
        batch_id: identifier for the batch.
        n_blocks: PBO block count (even; >= 50 bars/block).
        corr_threshold: correlation clustering threshold for effective_trials.
        dsr_threshold: DSR bar for the report's PASS/FAIL column.

    Returns:
        OverfittingVerdict with per-strategy DSR, batch PBO, and a report.
    """
    import pandas as pd  # local import: keeps module import light

    if not batch_id:
        batch_id = f"batch-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"

    names = list(returns_by_strategy.keys())
    if len(names) < 2:
        raise ValueError("evaluate_batch_overfitting requires >= 2 strategies")

    # Per-strategy per-period Sharpe / skew / kurtosis / n (on each full series).
    per_sr: Dict[str, float] = {}
    per_skew: Dict[str, float] = {}
    per_kurt: Dict[str, float] = {}  # NON-excess (Normal == 3)
    per_n: Dict[str, int] = {}
    for name in names:
        r = pd.Series(returns_by_strategy[name]).dropna()
        per_n[name] = int(len(r))
        sd = r.std(ddof=1)
        per_sr[name] = float(r.mean() / sd) if sd and sd > 0 else 0.0
        per_skew[name] = float(r.skew()) if len(r) > 2 else 0.0
        per_kurt[name] = float(r.kurtosis() + 3.0) if len(r) > 3 else 3.0  # excess -> non-excess

    # Variance of Sharpe ACROSS the trial universe (across strategies).
    sr_variance = float(np.var(list(per_sr.values()), ddof=1)) if len(per_sr) > 1 else 0.0

    # Aligned (T, N) matrix on the common index for effective_trials + PBO.
    aligned = pd.DataFrame(returns_by_strategy).dropna(how="any")
    matrix = aligned.to_numpy()
    n_eff = effective_trials(matrix, corr_threshold=corr_threshold) if matrix.shape[1] > 1 else 1
    n_eff = max(n_eff, 2)  # deflation needs >= 2

    try:
        pbo, _ = cscv_pbo(matrix, n_blocks=n_blocks)
    except ValueError as e:
        logger.warning(f"PBO skipped for {batch_id}: {e}")
        pbo = float("nan")

    dsr: Dict[str, float] = {}
    for name in names:
        try:
            dsr[name] = deflated_sharpe_ratio(
                sr=per_sr[name],
                n=per_n[name],
                skew=per_skew[name],
                kurt=per_kurt[name],
                sr_variance=sr_variance,
                n_trials=n_eff,
            )
        except ValueError as e:
            logger.warning(f"DSR skipped for {name}: {e}")
            dsr[name] = float("nan")

    # Report
    lines = [
        f"=== Overfitting Report ({batch_id}) ===",
        f"strategies={len(names)}  effective_trials={n_eff}  "
        f"PBO={pbo:.3f}" + ("" if not np.isnan(pbo) else " (insufficient data)"),
        "",
        f"{'Strategy':<30} {'SR/period':>10} {'n':>5} {'DSR':>7} {'Verdict':>10}",
        "-" * 66,
    ]
    for name in sorted(names, key=lambda x: (np.nan_to_num(dsr[x], nan=-1)), reverse=True):
        d = dsr[name]
        verdict = "PASS" if (not np.isnan(d) and d >= dsr_threshold) else "FAIL"
        lines.append(
            f"{name:<30} {per_sr[name]:>10.4f} {per_n[name]:>5} "
            f"{d:>7.3f} {verdict:>10}"
        )
    lines.append("-" * 66)
    lines.append(f"DSR bar: {dsr_threshold}   PBO acceptable if < 0.5")
    report = "\n".join(lines)

    return OverfittingVerdict(
        batch_id=batch_id,
        n_strategies=len(names),
        n_effective_trials=n_eff,
        pbo=pbo,
        dsr=dsr,
        per_period_sharpe=per_sr,
        n_obs=per_n,
        report=report,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
