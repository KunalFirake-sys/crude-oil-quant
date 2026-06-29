"""
visualise.py
------------
Generates and saves all backtest charts to results/.

Typical usage:
    from src.visualise import *
    plot_equity_curve(results)
    plot_drawdown(results)
    ...
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Style defaults ────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor" : "white",
    "axes.facecolor"   : "white",
    "axes.grid"        : True,
    "grid.alpha"       : 0.3,
    "grid.linestyle"   : "--",
    "axes.spines.top"  : False,
    "axes.spines.right": False,
    "font.size"        : 11,
})

IS_OOS_DATE = "2021-12-31"   # vertical split line used across charts


def _save(fig: plt.Figure, name: str) -> None:
    path = os.path.join(RESULTS_DIR, f"{name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[visualise] Saved → {path}")


# ── 1. Equity Curve ───────────────────────────────────────────────────────────

def plot_equity_curve(portfolio_df: pd.DataFrame) -> None:
    """
    Plot strategy equity curve vs buy-and-hold benchmark.

    A vertical dashed line marks the IS/OOS split so you can visually
    assess whether out-of-sample performance holds up.

    Parameters
    ----------
    portfolio_df : pd.DataFrame
        Output of run_backtest(). Must contain 'portfolio_value' and
        'buy_and_hold' columns with a DatetimeIndex.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(portfolio_df.index, portfolio_df["portfolio_value"],
            color="#1f77b4", linewidth=1.8, label="Strategy")
    ax.plot(portfolio_df.index, portfolio_df["buy_and_hold"],
            color="#ff7f0e", linewidth=1.8, linestyle="--", label="Buy & Hold")

    # IS/OOS split line
    ax.axvline(pd.Timestamp(IS_OOS_DATE), color="grey", linestyle=":",
               linewidth=1.5, label=f"IS/OOS split ({IS_OOS_DATE})")
    ax.text(pd.Timestamp(IS_OOS_DATE), ax.get_ylim()[0],
            " OOS →", color="grey", fontsize=9, va="bottom")

    ax.set_title("Equity Curve — Strategy vs Buy & Hold", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value (USD)")
    ax.yaxis.set_major_formatter(mtick.StrMethodFormatter("${x:,.0f}"))
    ax.legend()
    fig.tight_layout()
    _save(fig, "equity_curve")


# ── 2. Drawdown ───────────────────────────────────────────────────────────────

def plot_drawdown(portfolio_df: pd.DataFrame) -> None:
    """
    Plot the rolling drawdown of the strategy as a filled area chart.

    Drawdown measures the percentage decline from the most recent equity peak.
    Persistent or deep drawdowns signal periods of strategy stress.

    Parameters
    ----------
    portfolio_df : pd.DataFrame
        Must contain 'portfolio_value' with a DatetimeIndex.
    """
    equity      = portfolio_df["portfolio_value"]
    rolling_max = equity.cummax()
    drawdown    = (equity - rolling_max) / rolling_max * 100

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.fill_between(drawdown.index, drawdown, 0,
                    color="#d62728", alpha=0.5, label="Drawdown")
    ax.plot(drawdown.index, drawdown, color="#d62728", linewidth=0.8)
    ax.axhline(0, color="black", linewidth=0.8)

    ax.set_title("Strategy Drawdown", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown (%)")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.legend()
    fig.tight_layout()
    _save(fig, "drawdown")


# ── 3. Signals Chart ──────────────────────────────────────────────────────────

def plot_signals(df: pd.DataFrame) -> None:
    """
    Plot price with MA overlays and buy/sell signal markers.

    Only the most recent 2 years are shown to keep the chart readable.
    Green triangles = long entry, red triangles = short entry.

    Parameters
    ----------
    df : pd.DataFrame
        Output of generate_signals(). Must contain 'Close', 'fast_ma',
        'slow_ma', and 'position'.
    """
    # Last 2 years
    cutoff = df.index.max() - pd.DateOffset(years=2)
    d      = df[df.index >= cutoff].copy()

    # Detect entries: position changes TO +1 or -1
    pos_change = d["position"].diff().fillna(0)
    buys  = d[pos_change == 1]
    sells = d[pos_change == -1]

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(d.index, d["Close"],    color="#333333", linewidth=1.2, label="Close", zorder=2)
    ax.plot(d.index, d["fast_ma"], color="#1f77b4", linewidth=1.2,
            linestyle="--", label=f"Fast MA", zorder=3)
    ax.plot(d.index, d["slow_ma"], color="#ff7f0e", linewidth=1.2,
            linestyle="--", label=f"Slow MA", zorder=3)

    ax.scatter(buys.index,  buys["Close"],  marker="^", color="#2ca02c",
               s=80, zorder=5, label="Long entry")
    ax.scatter(sells.index, sells["Close"], marker="v", color="#d62728",
               s=80, zorder=5, label="Short entry")

    ax.set_title("Price, Moving Averages & Signals (Last 2 Years)",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    fig.tight_layout()
    _save(fig, "signals")


# ── 4. Parameter Heatmap ─────────────────────────────────────────────────────

def plot_parameter_heatmap(
    sharpe_matrix: np.ndarray,
    fast_labels: list,
    slow_labels: list,
) -> None:
    """
    Seaborn heatmap of Sharpe ratios across MA parameter combinations.

    A smooth, stable surface indicates parameter robustness. Isolated
    bright spots surrounded by poor performance are a sign of overfitting.

    Parameters
    ----------
    sharpe_matrix : np.ndarray
        2D array (rows=fast MA, cols=slow MA) from parameter_sensitivity().
    fast_labels : list
        Fast MA window values (row labels).
    slow_labels : list
        Slow MA window values (column labels).
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    sns.heatmap(
        sharpe_matrix,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        center=0,
        xticklabels=slow_labels,
        yticklabels=fast_labels,
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "Sharpe Ratio"},
    )

    ax.set_title("Parameter Sensitivity — Sharpe Ratio Grid",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Slow MA Window")
    ax.set_ylabel("Fast MA Window")
    fig.tight_layout()
    _save(fig, "parameter_heatmap")


# ── 5. IS vs OOS Comparison ──────────────────────────────────────────────────

def plot_is_oos_comparison(is_metrics: dict, oos_metrics: dict) -> None:
    """
    Side-by-side bar chart comparing IS and OOS performance.

    If OOS bars are substantially lower than IS bars, the strategy is
    likely overfit to the training period.

    Parameters
    ----------
    is_metrics : dict
        Output of calculate_metrics() for the in-sample period.
    oos_metrics : dict
        Output of calculate_metrics() for the out-of-sample period.
    """
    metrics_to_plot = {
        "Sharpe Ratio": ("sharpe_ratio", ""),
        "CAGR (%)":     ("cagr",         "%"),
        "Max Drawdown": ("max_drawdown",  "%"),
    }

    labels  = list(metrics_to_plot.keys())
    is_vals  = [is_metrics[v[0]]  for v in metrics_to_plot.values()]
    oos_vals = [oos_metrics[v[0]] for v in metrics_to_plot.values()]

    x     = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.bar(x - width/2, is_vals,  width, color="#1f77b4", label="In-Sample",     alpha=0.85)
    ax.bar(x + width/2, oos_vals, width, color="#ff7f0e", label="Out-of-Sample", alpha=0.85)

    ax.set_title("In-Sample vs Out-of-Sample Performance",
                 fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.legend()
    fig.tight_layout()
    _save(fig, "is_oos_comparison")


# ── 6. Regime Performance ────────────────────────────────────────────────────

def plot_regime_performance(regime_metrics: dict) -> None:
    """
    Grouped bar chart of strategy performance across market regimes.

    Identifies which market conditions the strategy relies on. A strategy
    that only works in bull/low-vol regimes is exposed to regime shifts.

    Parameters
    ----------
    regime_metrics : dict
        Output of regime_analysis(). Keys: bull, bear, high_vol, low_vol.
    """
    regimes = list(regime_metrics.keys())
    sharpes = [regime_metrics[r]["sharpe"]            for r in regimes]
    returns = [regime_metrics[r]["annualised_return"]  for r in regimes]

    x     = np.arange(len(regimes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.bar(x - width/2, sharpes, width, color="#1f77b4", label="Sharpe Ratio", alpha=0.85)
    ax.bar(x + width/2, returns, width, color="#2ca02c", label="Ann. Return (%)", alpha=0.85)

    ax.set_title("Strategy Performance by Market Regime",
                 fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([r.replace("_", " ").title() for r in regimes])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.legend()
    fig.tight_layout()
    _save(fig, "regime_performance")


# ── 7. Cost Sensitivity ──────────────────────────────────────────────────────

def plot_cost_sensitivity(cost_results: list[tuple]) -> None:
    """
    Line chart of Sharpe ratio vs transaction cost level.

    Shows the breakeven cost — above which the strategy is no longer
    viable. If your real trading costs are near the breakeven, do not
    deploy the strategy live.

    Parameters
    ----------
    cost_results : list[tuple]
        Output of cost_sensitivity(). Each tuple: (cost, sharpe, total_return).
    """
    costs   = [r[0] * 100 for r in cost_results]   # convert to %
    sharpes = [r[1]        for r in cost_results]

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(costs, sharpes, color="#1f77b4", linewidth=2, marker="o", label="Sharpe Ratio")
    ax.axhline(0.5, color="#d62728", linestyle="--", linewidth=1.2,
               label="Breakeven threshold (Sharpe = 0.5)")
    ax.axhline(0,   color="black",   linewidth=0.8)

    # Mark first point where Sharpe drops below 0.5
    for cost, sharpe in zip(costs, sharpes):
        if sharpe < 0.5:
            ax.axvline(cost, color="#ff7f0e", linestyle=":", linewidth=1.5,
                       label=f"Breakeven ≈ {cost:.3f}%")
            break

    ax.set_title("Cost Sensitivity — Sharpe vs Transaction Cost",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Transaction Cost (%)")
    ax.set_ylabel("Sharpe Ratio")
    ax.xaxis.set_major_formatter(mtick.PercentFormatter())
    ax.legend()
    fig.tight_layout()
    _save(fig, "cost_sensitivity")


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from src.dataloader import get_data
    from src.signals import generate_signals
    from src.backtesting import run_backtest
    from src.analysis import (run_is_oos_split, parameter_sensitivity,
                               regime_analysis, cost_sensitivity)

    raw     = get_data()
    df      = generate_signals(raw.copy())
    results, trade_log = run_backtest(df.copy())

    plot_equity_curve(results)
    plot_drawdown(results)
    plot_signals(df)

    is_m, oos_m, _ = run_is_oos_split(df.copy())
    plot_is_oos_comparison(is_m, oos_m)

    regime_m = regime_analysis(results.copy())
    plot_regime_performance(regime_m)

    cost_r = cost_sensitivity(df.copy())
    plot_cost_sensitivity(cost_r)

    sharpe_mat, fl, sl = parameter_sensitivity(
        raw.copy(), fast_range=[20, 30], slow_range=[50, 80]
    )
    plot_parameter_heatmap(sharpe_mat, fl, sl)

    print("\nAll charts saved to results/")


def plot_equity_curve_all(
    all_results: dict[str, pd.DataFrame],
) -> None:
    """
    Plot equity curves for all strategies + buy-and-hold on one chart.

    Parameters
    ----------
    all_results : dict
        Keys are strategy names (e.g. 'V1', 'V2', 'V3', 'V4').
        Values are DataFrames with 'portfolio_value' and 'buy_and_hold'.
    """
    colors = {"V1": "#1f77b4", "V2": "#ff7f0e", "V3": "#2ca02c", "V4": "#9467bd"}
    fig, ax = plt.subplots(figsize=(12, 6))

    bah_plotted = False
    for name, df in all_results.items():
        ax.plot(df.index, df["portfolio_value"],
                color=colors.get(name, "grey"), linewidth=1.8, label=name)
        if not bah_plotted:
            ax.plot(df.index, df["buy_and_hold"],
                    color="black", linewidth=1.4, linestyle="--", label="Buy & Hold")
            bah_plotted = True

    ax.axvline(pd.Timestamp(IS_OOS_DATE), color="grey", linestyle=":",
               linewidth=1.5, label=f"IS/OOS ({IS_OOS_DATE})")

    ax.set_title("All Strategies — Equity Curve Comparison", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value (USD)")
    ax.yaxis.set_major_formatter(mtick.StrMethodFormatter("${x:,.0f}"))
    ax.legend()
    fig.tight_layout()
    _save(fig, "equity_curve_all_strategies")


def plot_v3_signals(df: pd.DataFrame) -> None:
    """
    Plot V3 Donchian channel with price and entry signals (last 2 years).

    Parameters
    ----------
    df : pd.DataFrame
        Output of generate_signals_v3(). Must contain 'Close',
        'upper_band', 'lower_band', 'position'.
    """
    cutoff = df.index.max() - pd.DateOffset(years=2)
    d      = df[df.index >= cutoff].copy()

    pos_change = d["position"].diff().fillna(0)
    buys  = d[pos_change == 1]
    sells = d[pos_change == -1]

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(d.index, d["Close"],       color="#333333", linewidth=1.2, label="Close",       zorder=2)
    ax.plot(d.index, d["upper_band"], color="#2ca02c", linewidth=1.2,
            linestyle="--", label="Upper Band (20d High)", zorder=3)
    ax.plot(d.index, d["lower_band"], color="#d62728", linewidth=1.2,
            linestyle="--", label="Lower Band (20d Low)",  zorder=3)

    ax.fill_between(d.index, d["upper_band"], d["lower_band"],
                    alpha=0.05, color="grey")

    ax.scatter(buys.index,  buys["Close"],  marker="^", color="#2ca02c", s=80, zorder=5, label="Long entry")
    ax.scatter(sells.index, sells["Close"], marker="v", color="#d62728", s=80, zorder=5, label="Short entry")

    ax.set_title("V3 — Donchian Channel Breakout Signals (Last 2 Years)",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    fig.tight_layout()
    _save(fig, "v3_signals")