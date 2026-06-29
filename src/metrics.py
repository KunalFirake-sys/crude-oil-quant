"""
metrics.py
----------
Calculates and displays performance metrics for the WTI crude oil backtest.

Typical usage:
    from src.metrics import calculate_metrics, print_metrics
    metrics = calculate_metrics(results, trade_log)
    print_metrics(metrics)
"""

import numpy as np
import pandas as pd


def calculate_metrics(
    portfolio_df: pd.DataFrame,
    trade_log: list[dict],
    risk_free_rate: float = 0.05,
) -> dict:
    """
    Compute strategy performance metrics from backtest output.

    Parameters
    ----------
    portfolio_df : pd.DataFrame
        Output of run_backtest(). Must contain 'daily_returns' and 'portfolio_value'.
    trade_log : list[dict]
        Trade log from run_backtest(). Each entry has 'date', 'equity_before',
        'equity_after', and 'action'.
    risk_free_rate : float
        Annualised risk-free rate for Sharpe calculation. Default 0.05 (5%).

    Returns
    -------
    dict
        Keys: total_return, cagr, sharpe_ratio, max_drawdown, win_rate,
              profit_factor, calmar_ratio, total_trades, avg_trade_duration,
              volatility
    """

    rets   = portfolio_df["daily_returns"].fillna(0)
    equity = portfolio_df["portfolio_value"]
    n_days = len(rets)
    n_years = n_days / 252  # trading days per year

    # ── 1. Total Return ───────────────────────────────────────────────────────
    # (Final equity − Initial equity) / Initial equity × 100
    total_return = (equity.iloc[-1] - equity.iloc[0]) / equity.iloc[0] * 100

    # ── 2. CAGR ───────────────────────────────────────────────────────────────
    # (Final / Initial) ^ (1 / years) − 1
    cagr = ((equity.iloc[-1] / equity.iloc[0]) ** (1 / n_years) - 1) * 100

    # ── 3. Sharpe Ratio ───────────────────────────────────────────────────────
    # (Mean daily return − daily risk-free rate) / Std dev of daily returns × √252
    daily_rf = risk_free_rate / 252
    excess   = rets - daily_rf
    sharpe   = (excess.mean() / excess.std()) * np.sqrt(252) if excess.std() != 0 else 0.0

    # ── 4. Max Drawdown ───────────────────────────────────────────────────────
    # Worst peak-to-trough decline in the equity curve.
    # rolling_max tracks the running peak; drawdown is current / peak − 1.
    rolling_max  = equity.cummax()
    drawdown     = (equity - rolling_max) / rolling_max
    max_drawdown = drawdown.min() * 100  # expressed as negative %

    # ── 5. Volatility ─────────────────────────────────────────────────────────
    # Annualised std dev of daily returns × √252
    volatility = rets.std() * np.sqrt(252) * 100

    # ── 6. Calmar Ratio ───────────────────────────────────────────────────────
    # CAGR / |Max Drawdown| — reward per unit of worst drawdown risk
    calmar = cagr / abs(max_drawdown) if max_drawdown != 0 else 0.0

    # ── 7. Trade-level metrics ────────────────────────────────────────────────
    total_trades = len(trade_log)

    win_rate      = 0.0
    profit_factor = 0.0
    avg_duration  = 0.0

    if total_trades > 0:
        # Pair trades: entry is BUY or SELL SHORT, exit is the next trade
        # P&L = equity_after of exit − equity_after of entry
        pnls = []
        for i in range(len(trade_log) - 1):
            entry = trade_log[i]
            exit_ = trade_log[i + 1]
            # Only count closes: FLAT or direction flip
            if entry["action"] in ("BUY", "SELL SHORT") and exit_["action"] in ("FLAT", "BUY", "SELL SHORT"):
                pnl = exit_["equity_after"] - entry["equity_after"]
                pnls.append(pnl)

        wins         = [p for p in pnls if p > 0]
        losses       = [p for p in pnls if p < 0]
        gross_profit = sum(wins)
        gross_loss   = abs(sum(losses))

        win_rate      = len(wins) / len(pnls) * 100 if pnls else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss != 0 else float("inf")

        dates = [pd.Timestamp(t["date"]) for t in trade_log]
        if len(dates) > 1:
            durations    = [(dates[i+1] - dates[i]).days for i in range(len(dates)-1)]
            avg_duration = sum(durations) / len(durations)

    return {
        "total_return"      : round(total_return, 2),
        "cagr"              : round(cagr, 2),
        "sharpe_ratio"      : round(sharpe, 3),
        "max_drawdown"      : round(max_drawdown, 2),
        "win_rate"          : round(win_rate, 2),
        "profit_factor"     : round(profit_factor, 3),
        "calmar_ratio"      : round(calmar, 3),
        "total_trades"      : total_trades,
        "avg_trade_duration": round(avg_duration, 1),
        "volatility"        : round(volatility, 2),
    }


def print_metrics(metrics: dict) -> None:
    """
    Print a formatted table of strategy performance metrics.

    Parameters
    ----------
    metrics : dict
        Output of calculate_metrics().
    """

    labels = {
        "total_return"      : ("Total Return",          "%"),
        "cagr"              : ("CAGR",                  "%"),
        "sharpe_ratio"      : ("Sharpe Ratio",          ""),
        "max_drawdown"      : ("Max Drawdown",          "%"),
        "volatility"        : ("Annualised Volatility", "%"),
        "calmar_ratio"      : ("Calmar Ratio",          ""),
        "win_rate"          : ("Win Rate",              "%"),
        "profit_factor"     : ("Profit Factor",         ""),
        "total_trades"      : ("Total Trades",          ""),
        "avg_trade_duration": ("Avg Trade Duration",    "days"),
    }

    print("\n" + "=" * 45)
    print("  Strategy Performance Metrics")
    print("=" * 45)
    for key, (label, unit) in labels.items():
        val = metrics[key]
        if unit == "%":
            print(f"  {label:<26} {val:>+.2f}%")
        elif unit == "days":
            print(f"  {label:<26} {val:>6.1f} days")
        elif isinstance(val, int):
            print(f"  {label:<26} {val:>6}")
        else:
            print(f"  {label:<26} {val:>8.3f}")
    print("=" * 45 + "\n")


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from src.dataloader import get_data
    from src.signals import generate_signals

    df = get_data()
    df = generate_signals(df, fast_window=30, slow_window=80)

    # Debug: confirm windows and MA values
    print(df[["Close", "fast_ma", "slow_ma", "signal", "position"]].head(10).to_string())
    print(f"\nUnique signals: {df['signal'].value_counts().to_dict()}")
    print(f"Unique positions: {df['position'].value_counts().to_dict()}")