"""
backtest.py
-----------
Event-driven backtest for the dual MA crossover strategy on WTI crude oil futures.

Typical usage:
    from src.backtest import run_backtest
    results, trade_log = run_backtest(df)
"""

import pandas as pd
import numpy as np


def run_backtest(
    df: pd.DataFrame,
    initial_capital: float = 100_000.0,
    risk_per_trade: float = 0.02,
    cost_per_trade: float = 0.001,
) -> tuple[pd.DataFrame, list[dict]]:
    """
    Simulate the strategy on historical data and return portfolio metrics.

    Parameters
    ----------
    df : pd.DataFrame
        Output of generate_signals(). Must contain 'Close' and 'position'.
    initial_capital : float
        Starting cash in USD. Default 100,000.
    risk_per_trade : float
        Fraction of current equity risked per trade (fixed fractional sizing).
        Default 0.02 → 2%.
    cost_per_trade : float
        One-way transaction cost as a fraction of trade value.
        Default 0.001 → 0.1%.

    Returns
    -------
    results : pd.DataFrame
        Original DataFrame plus:
        - portfolio_value  : equity curve of the strategy
        - daily_returns    : daily P&L as a fraction of prior equity
        - buy_and_hold     : equity curve of simply holding crude from day 1
    trade_log : list[dict]
        One entry per position change with keys:
        date, action, price, shares, cost, equity_before, equity_after
    """

    df = df.copy()

    # ── 1. Daily close-to-close price change (%) ─────────────────────────────
    # pct_change() on Close gives the return for each bar.
    # Multiplied by position: long (+1) profits when price rises,
    # short (-1) profits when price falls.
    close_returns = df["Close"].pct_change().fillna(0)

    # ── 2. Initialise tracking variables ─────────────────────────────────────
    equity        = initial_capital
    portfolio     = []   # equity value at end of each bar
    daily_rets    = []   # fractional daily return of the strategy
    trade_log     = []   # one dict per trade entry/exit

    prev_position = 0    # position held coming into the current bar
    shares_held   = 0.0  # number of (fractional) contracts held

    for i, (date, row) in enumerate(df.iterrows()):
        cur_position = row["position"]
        price        = row["Close"]
        bar_return   = close_returns.iloc[i]

        # ── 2a. P&L on existing holding ──────────────────────────────────────
        # Mark the current holding to market before any trade today.
        pnl = shares_held * prev_position * bar_return * price
        equity += pnl

        # ── 2b. Position change → trade ──────────────────────────────────────
        if cur_position != prev_position:
            equity_before = equity

            # Fixed fractional sizing: risk risk_per_trade of current equity.
            # Dollar amount at risk determines how many units we buy/short.
            dollars_at_risk = equity * risk_per_trade
            shares_held     = dollars_at_risk / price if price > 0 else 0.0

            # Transaction cost applied on the notional value of the new position.
            trade_value = shares_held * price
            cost        = trade_value * cost_per_trade
            equity     -= cost

            # Determine human-readable action label
            if cur_position == 1:
                action = "BUY"
            elif cur_position == -1:
                action = "SELL SHORT"
            else:
                action = "FLAT"
                shares_held = 0.0   # no position, no holding

            trade_log.append({
                "date"         : date,
                "action"       : action,
                "price"        : round(price, 4),
                "shares"       : round(shares_held, 4),
                "cost"         : round(cost, 4),
                "equity_before": round(equity_before, 2),
                "equity_after" : round(equity, 2),
            })

        # ── 2c. Daily return as fraction of equity ────────────────────────────
        # Guard against divide-by-zero on the first bar.
        if i == 0 or portfolio[-1] == 0:
            daily_ret = 0.0
        else:
            daily_ret = (equity - portfolio[-1]) / portfolio[-1]

        portfolio.append(equity)
        daily_rets.append(daily_ret)

        prev_position = cur_position

    # ── 3. Attach results to DataFrame ───────────────────────────────────────
    df["portfolio_value"] = portfolio
    df["daily_returns"]   = daily_rets

    # ── 4. Buy-and-hold benchmark ─────────────────────────────────────────────
    # Invest all initial capital in crude at the first valid close price,
    # hold until the end. No costs applied (passive benchmark).
    first_valid_price      = df["Close"].iloc[0]
    bah_shares             = initial_capital / first_valid_price
    df["buy_and_hold"]     = bah_shares * df["Close"]

    # ── 5. Summary print ─────────────────────────────────────────────────────
    final_equity  = portfolio[-1]
    total_return  = (final_equity - initial_capital) / initial_capital * 100
    bah_return    = (df["buy_and_hold"].iloc[-1] - initial_capital) / initial_capital * 100
    n_trades      = len(trade_log)

    print("\n" + "=" * 50)
    print("  Backtest Summary")
    print("=" * 50)
    print(f"  Initial capital : ${initial_capital:>12,.2f}")
    print(f"  Final equity    : ${final_equity:>12,.2f}")
    print(f"  Strategy return : {total_return:>+.2f}%")
    print(f"  Buy & Hold      : {bah_return:>+.2f}%")
    print(f"  Total trades    : {n_trades}")
    print("=" * 50 + "\n")

    return df, trade_log


if __name__ == "__main__":
    from src.dataloader import get_data
    from src.signals import generate_signals
    df = generate_signals(get_data())
    results, trade_log = run_backtest(df)
    print(results[["Close", "position", "portfolio_value", "buy_and_hold"]].tail(10))
    print("\nLast 3 trades:", trade_log[-3:])