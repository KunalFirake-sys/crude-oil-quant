"""
main.py
-------
Full WTI crude oil futures backtest pipeline — all 4 strategies.

To run a single strategy, change STRATEGY_VERSION:
    "v1" → Dual MA Crossover
    "v2" → Dual MA + 200d Regime Filter
    "v3" → Donchian Channel Breakout
    "v4" → Regime-Switching Combined
    "all" → Run all four and print comparison table
"""

from src.dataloader import get_data
from src.signals     import (generate_signals, generate_signals_v2,
                              generate_signals_v3, generate_signals_v4)
from src.backtesting    import run_backtest
from src.metrics     import calculate_metrics, print_metrics
from src.analysis    import (run_is_oos_split, parameter_sensitivity,
                              regime_analysis, cost_sensitivity)
from src.visualize   import (plot_equity_curve, plot_drawdown, plot_signals,
                              plot_parameter_heatmap, plot_is_oos_comparison,
                              plot_regime_performance, plot_cost_sensitivity,
                              plot_equity_curve_all, plot_v3_signals)


# ── Config ────────────────────────────────────────────────────────────────────

STRATEGY_VERSION = "all"      # "v1" | "v2" | "v3" | "v4" | "all"

FAST_WINDOW      = 30
SLOW_WINDOW      = 80
MOMENTUM_WINDOW  = 10
MIN_HOLD         = 3
REGIME_WINDOW    = 200
BREAKOUT_WINDOW  = 20

INITIAL_CAPITAL  = 100_000
COST_PER_TRADE   = 0.0002
RISK_PER_TRADE   = 0.02

RUN_ANALYSIS     = True       # set False to skip slow parameter sweep
PARAM_FAST_RANGE = [10, 15, 20, 25, 30]
PARAM_SLOW_RANGE = [40, 50, 60, 70, 80]


# ── Strategy builder ──────────────────────────────────────────────────────────

def build_signals(raw, version: str):
    v = version.lower()
    if v == "v1":
        return generate_signals(raw, FAST_WINDOW, SLOW_WINDOW, MOMENTUM_WINDOW, MIN_HOLD)
    elif v == "v2":
        return generate_signals_v2(raw, FAST_WINDOW, SLOW_WINDOW, MOMENTUM_WINDOW, MIN_HOLD, REGIME_WINDOW)
    elif v == "v3":
        return generate_signals_v3(raw, BREAKOUT_WINDOW, REGIME_WINDOW)
    elif v == "v4":
        return generate_signals_v4(raw, FAST_WINDOW, SLOW_WINDOW, MOMENTUM_WINDOW, BREAKOUT_WINDOW, REGIME_WINDOW)
    else:
        raise ValueError(f"Unknown version: '{version}'. Use v1/v2/v3/v4/all.")


def run_strategy(raw, version: str):
    """Run one strategy end-to-end, return (df, results, trade_log, metrics)."""
    print(f"\n{'='*55}")
    print(f"  Strategy {version.upper()}")
    print(f"{'='*55}")
    df                 = build_signals(raw.copy(), version)
    results, trade_log = run_backtest(df.copy(), INITIAL_CAPITAL, RISK_PER_TRADE, COST_PER_TRADE)
    metrics            = calculate_metrics(results, trade_log)
    print_metrics(metrics)
    return df, results, trade_log, metrics


# ── Comparison table ──────────────────────────────────────────────────────────

def print_comparison(all_metrics: dict):
    """Print a side-by-side summary table for all strategies."""
    keys = {
        "Total Return (%)": "total_return",
        "CAGR (%)":         "cagr",
        "Sharpe Ratio":     "sharpe_ratio",
        "Max Drawdown (%)": "max_drawdown",
        "Volatility (%)":   "volatility",
        "Win Rate (%)":     "win_rate",
        "Profit Factor":    "profit_factor",
        "Total Trades":     "total_trades",
        "Avg Trade Dur.":   "avg_trade_duration",
    }

    versions = list(all_metrics.keys())
    col_w    = 12

    print("\n" + "=" * (22 + col_w * len(versions)))
    header = f"  {'Metric':<20}" + "".join(f"{v:>{col_w}}" for v in versions)
    print(header)
    print("=" * (22 + col_w * len(versions)))

    for label, key in keys.items():
        row = f"  {label:<20}"
        for v in versions:
            val = all_metrics[v][key]
            if isinstance(val, float):
                row += f"{val:>{col_w}.2f}"
            else:
                row += f"{val:>{col_w}}"
        print(row)

    print("=" * (22 + col_w * len(versions)) + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():

    raw = get_data()

    if STRATEGY_VERSION.lower() == "all":

        # Run all 4
        all_dfs     = {}
        all_results = {}
        all_metrics = {}

        for v in ["v1", "v2", "v3", "v4"]:
            df, results, trade_log, metrics = run_strategy(raw, v)
            all_dfs[v.upper()]     = df
            all_results[v.upper()] = results
            all_metrics[v.upper()] = metrics

        # Comparison table
        print_comparison(all_metrics)

        # Charts — all strategies
        plot_equity_curve_all(all_results)
        plot_v3_signals(all_dfs["V3"])

        # Standard charts using V4 (most advanced) as default
        plot_equity_curve(all_results["V4"])
        plot_drawdown(all_results["V4"])
        plot_signals(all_dfs["V2"])   # V2 has fast/slow MA columns needed

        if RUN_ANALYSIS:
            is_m, oos_m, _ = run_is_oos_split(all_dfs["V4"].copy())
            regime_m        = regime_analysis(all_results["V4"].copy())
            cost_r          = cost_sensitivity(all_dfs["V4"].copy())
            sm, fl, sl      = parameter_sensitivity(raw.copy(), PARAM_FAST_RANGE, PARAM_SLOW_RANGE)
            plot_is_oos_comparison(is_m, oos_m)
            plot_regime_performance(regime_m)
            plot_cost_sensitivity(cost_r)
            plot_parameter_heatmap(sm, fl, sl)

    else:
        # Single strategy
        df, results, trade_log, metrics = run_strategy(raw, STRATEGY_VERSION)

        plot_equity_curve(results)
        plot_drawdown(results)

        if STRATEGY_VERSION.lower() == "v3":
            plot_v3_signals(df)
        else:
            plot_signals(df)

        if RUN_ANALYSIS:
            is_m, oos_m, _ = run_is_oos_split(df.copy())
            regime_m        = regime_analysis(results.copy())
            cost_r          = cost_sensitivity(df.copy())
            sm, fl, sl      = parameter_sensitivity(raw.copy(), PARAM_FAST_RANGE, PARAM_SLOW_RANGE)
            plot_is_oos_comparison(is_m, oos_m)
            plot_regime_performance(regime_m)
            plot_cost_sensitivity(cost_r)
            plot_parameter_heatmap(sm, fl, sl)

    print("[main] Done. Charts saved to results/")


if __name__ == "__main__":
    main()