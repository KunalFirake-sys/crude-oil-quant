"""
analysis.py
-----------
Critical analysis layer for the WTI crude oil futures backtest.

Addresses three core concerns in quantitative finance:
  1. Overfitting      — IS/OOS split
  2. Data-snooping    — Parameter sensitivity heatmap
  3. Regime dependence — Performance across market regimes

Typical usage:
    from src.analysis import run_is_oos_split, parameter_sensitivity
                             regime_analysis, cost_sensitivity
"""

import numpy as np
import pandas as pd
from scipy import stats
from itertools import product

from src.signals import generate_signals
from src.backtesting import run_backtest
from src.metrics import calculate_metrics


# ── 1. In-Sample / Out-of-Sample Split ───────────────────────────────────────

def run_is_oos_split(
    df: pd.DataFrame,
    is_end_date: str = "2021-12-31",
) -> tuple[dict, dict, dict]:
    """
    Split data into in-sample (IS) and out-of-sample (OOS) periods and
    run the backtest on each with IDENTICAL parameters.

    WHY THIS MATTERS
    ----------------
    Most strategies are optimised (intentionally or not) on the full dataset.
    If you only report full-period metrics, you cannot tell whether the strategy
    genuinely works or just fits historical noise. An OOS period the strategy
    has never 'seen' is the closest proxy for live performance. A strategy that
    degrades severely OOS is likely overfit.

    The t-test checks whether the difference in mean daily returns between IS
    and OOS is statistically significant. A p-value < 0.05 suggests the two
    periods behave differently — a red flag for regime dependence or overfitting.

    Parameters
    ----------
    df : pd.DataFrame
        Full signals DataFrame (output of generate_signals).
    is_end_date : str
        Last date of the in-sample period. Default '2021-12-31'.

    Returns
    -------
    is_metrics : dict
        Performance metrics for the in-sample period.
    oos_metrics : dict
        Performance metrics for the out-of-sample period.
    ttest : dict
        Keys: t_stat, p_value, significant (bool at 5% level).
    """

    is_df  = df[df.index <= is_end_date].copy()
    oos_df = df[df.index >  is_end_date].copy()

    is_results,  is_trades  = run_backtest(is_df)
    oos_results, oos_trades = run_backtest(oos_df)

    is_metrics  = calculate_metrics(is_results,  is_trades)
    oos_metrics = calculate_metrics(oos_results, oos_trades)

    # t-test on daily returns
    is_rets  = is_results["daily_returns"].dropna()
    oos_rets = oos_results["daily_returns"].dropna()
    t_stat, p_value = stats.ttest_ind(is_rets, oos_rets, equal_var=False)

    ttest = {
        "t_stat"     : round(float(t_stat), 4),
        "p_value"    : round(float(p_value), 4),
        "significant": bool(p_value < 0.05),
    }

    print("\n" + "=" * 50)
    print("  IS / OOS Analysis")
    print("=" * 50)
    print(f"  In-Sample  (≤ {is_end_date})")
    print(f"    Return   : {is_metrics['total_return']:>+.2f}%")
    print(f"    Sharpe   : {is_metrics['sharpe_ratio']:>.3f}")
    print(f"  Out-of-Sample (> {is_end_date})")
    print(f"    Return   : {oos_metrics['total_return']:>+.2f}%")
    print(f"    Sharpe   : {oos_metrics['sharpe_ratio']:>.3f}")
    print(f"  t-stat: {ttest['t_stat']}  p-value: {ttest['p_value']}  "
          f"Significant: {ttest['significant']}")
    print("=" * 50 + "\n")

    return is_metrics, oos_metrics, ttest


# ── 2. Parameter Sensitivity ─────────────────────────────────────────────────

def parameter_sensitivity(
    df: pd.DataFrame,
    fast_range: list[int] = [10, 15, 20, 25, 30],
    slow_range:  list[int] = [40, 50, 60, 70, 80],
) -> tuple[np.ndarray, list[int], list[int]]:
    """
    Sweep all combinations of fast and slow MA windows and record Sharpe ratio.

    WHY THIS MATTERS
    ----------------
    If a strategy only works for a narrow band of parameters (e.g. fast=20,
    slow=50 is great but fast=19 or fast=21 collapses), it is almost certainly
    overfit. A robust strategy should show a smooth, stable Sharpe surface
    across a wide parameter range. This is called 'parameter robustness' and
    is a basic sanity check before deploying any systematic strategy.

    Parameters
    ----------
    df : pd.DataFrame
        Raw OHLCV DataFrame (output of get_data, NOT pre-signalled).
    fast_range : list[int]
        Fast MA windows to test.
    slow_range : list[int]
        Slow MA windows to test.

    Returns
    -------
    sharpe_matrix : np.ndarray
        2D array of shape (len(fast_range), len(slow_range)).
        sharpe_matrix[i][j] = Sharpe for fast_range[i], slow_range[j].
    fast_range : list[int]
        Row labels (fast windows).
    slow_range : list[int]
        Column labels (slow windows).
    """

    sharpe_matrix = np.full((len(fast_range), len(slow_range)), np.nan)

    total = len(fast_range) * len(slow_range)
    done  = 0

    for i, fast in enumerate(fast_range):
        for j, slow in enumerate(slow_range):
            if fast >= slow:
                # Invalid combo: fast must be shorter than slow
                sharpe_matrix[i][j] = np.nan
                done += 1
                continue

            try:
                sig     = generate_signals(df.copy(), fast_window=fast, slow_window=slow)
                results, trades = run_backtest(sig)
                metrics = calculate_metrics(results, trades)
                sharpe_matrix[i][j] = metrics["sharpe_ratio"]
            except Exception:
                sharpe_matrix[i][j] = np.nan

            done += 1
            print(f"[sensitivity] {done}/{total} — fast={fast}, slow={slow}, "
                  f"Sharpe={sharpe_matrix[i][j]:.3f}")

    print("\n[sensitivity] Sharpe matrix (rows=fast, cols=slow):")
    df_matrix = pd.DataFrame(
        sharpe_matrix, index=[f"f{f}" for f in fast_range],
        columns=[f"s{s}" for s in slow_range]
    )
    print(df_matrix.round(3).to_string())

    return sharpe_matrix, fast_range, slow_range


# ── 3. Regime Analysis ───────────────────────────────────────────────────────

def regime_analysis(df: pd.DataFrame) -> dict:
    """
    Measure strategy performance broken down by market regime.

    WHY THIS MATTERS
    ----------------
    A strategy may look great on average but only work in one specific regime
    (e.g. trending bull markets). If crude oil enters a different regime in
    live trading, the strategy may fail. Knowing which regimes drive performance
    tells you when to trust the strategy and when to be cautious.

    Regimes defined:
      bull     : Close > 200-day MA (uptrend)
      bear     : Close < 200-day MA (downtrend)
      high_vol : 30-day rolling std > median of that series
      low_vol  : 30-day rolling std ≤ median

    Parameters
    ----------
    df : pd.DataFrame
        Output of run_backtest() — must contain 'Close', 'daily_returns',
        'position'.

    Returns
    -------
    dict
        Keys: 'bull', 'bear', 'high_vol', 'low_vol'.
        Each value is a dict with: annualised_return, sharpe, n_days.
    """

    df = df.copy()

    # Regime labels
    ma200         = df["Close"].rolling(200, min_periods=200).mean()
    rolling_std   = df["Close"].rolling(30, min_periods=30).std()
    median_std    = rolling_std.median()

    regimes = {
        "bull"    : df.index[df["Close"] > ma200],
        "bear"    : df.index[df["Close"] < ma200],
        "high_vol": df.index[rolling_std > median_std],
        "low_vol" : df.index[rolling_std <= median_std],
    }

    results = {}
    rets    = df["daily_returns"].fillna(0)

    for name, idx in regimes.items():
        r = rets.loc[rets.index.intersection(idx)]
        if len(r) < 20:
            results[name] = {"annualised_return": np.nan, "sharpe": np.nan, "n_days": len(r)}
            continue

        ann_return = r.mean() * 252 * 100
        sharpe     = (r.mean() / r.std() * np.sqrt(252)) if r.std() != 0 else 0.0

        results[name] = {
            "annualised_return": round(ann_return, 2),
            "sharpe"           : round(sharpe, 3),
            "n_days"           : len(r),
        }

    print("\n" + "=" * 50)
    print("  Regime Analysis")
    print("=" * 50)
    for regime, m in results.items():
        print(f"  {regime:<10} | days={m['n_days']:>4} | "
              f"ann_ret={m['annualised_return']:>+.2f}% | "
              f"sharpe={m['sharpe']:>.3f}")
    print("=" * 50 + "\n")

    return results


# ── 4. Cost Sensitivity ──────────────────────────────────────────────────────

def cost_sensitivity(
    df: pd.DataFrame,
    cost_range: list[float] = [0, 0.0001, 0.0005, 0.001, 0.002, 0.005],
) -> list[tuple[float, float, float]]:
    """
    Re-run the backtest across a range of transaction cost assumptions.

    WHY THIS MATTERS
    ----------------
    Many backtests look profitable before costs but fail live because real
    costs (slippage, spread, commission) are higher than assumed. Cost
    sensitivity shows the breakeven cost level — the point at which the
    strategy stops being profitable. If your real costs are close to that
    breakeven, the strategy is not viable in production.

    Parameters
    ----------
    df : pd.DataFrame
        Output of generate_signals().
    cost_range : list[float]
        Transaction costs to test (fraction of trade value).

    Returns
    -------
    list[tuple[float, float, float]]
        Each tuple: (cost, sharpe_ratio, total_return_pct)
    """

    output = []

    print("\n" + "=" * 50)
    print("  Cost Sensitivity")
    print("=" * 50)
    print(f"  {'Cost':>8}  {'Sharpe':>8}  {'Return':>8}")

    for cost in cost_range:
        results, trades = run_backtest(df.copy(), cost_per_trade=cost)
        metrics         = calculate_metrics(results, trades)
        sharpe          = metrics["sharpe_ratio"]
        ret             = metrics["total_return"]
        output.append((cost, sharpe, ret))
        print(f"  {cost:>8.4f}  {sharpe:>8.3f}  {ret:>+8.2f}%")

    print("=" * 50 + "\n")

    return output


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from src.dataloader import get_data

    raw = get_data()
    df  = generate_signals(raw.copy())
    results, trade_log = run_backtest(df.copy())

    print("\n--- IS/OOS ---")
    run_is_oos_split(df.copy())

    print("\n--- Regime ---")
    regime_analysis(results.copy())

    print("\n--- Cost Sensitivity ---")
    cost_sensitivity(df.copy())

    print("\n--- Parameter Sensitivity (small grid for speed) ---")
    parameter_sensitivity(raw.copy(), fast_range=[20, 30], slow_range=[50, 80])