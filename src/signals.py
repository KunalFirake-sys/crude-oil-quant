"""
signals.py
----------
Generates trading signals for the WTI crude oil futures backtest.

Strategy: Dual Moving Average Crossover with Momentum Filter + Persistence
  - Long  (+1) : fast MA > slow MA  AND  ROC > 0  for min_hold consecutive bars
  - Short (-1) : fast MA < slow MA  AND  ROC < 0  for min_hold consecutive bars
  - Flat  ( 0) : otherwise

Typical usage:
    from src.signals import generate_signals
    df = generate_signals(df)
"""

import pandas as pd


def generate_signals(
    df: pd.DataFrame,
    fast_window: int = 30,
    slow_window: int = 80,
    momentum_window: int = 10,
    min_hold: int = 3,
) -> pd.DataFrame:
    """
    Add signal columns to an OHLCV DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned OHLCV DataFrame from data_loader.get_data().
        Must contain a 'Close' column.
    fast_window : int
        Rolling window (bars) for the fast moving average. Default 30.
    slow_window : int
        Rolling window (bars) for the slow moving average. Default 80.
    momentum_window : int
        Look-back period (bars) for Rate of Change. Default 10.
    min_hold : int
        Number of consecutive bars a raw signal must persist before
        being confirmed. Reduces noise and over-trading. Default 3.

    Returns
    -------
    pd.DataFrame
        Original DataFrame with five new columns appended:
        - fast_ma   : fast simple moving average of Close
        - slow_ma   : slow simple moving average of Close
        - roc       : Rate of Change (%) over momentum_window bars
        - signal    : confirmed signal (+1 / -1 / 0) after persistence filter
        - position  : look-ahead-bias-free signal (shifted forward 1 bar)

    Notes
    -----
    Look-ahead bias prevention
    ~~~~~~~~~~~~~~~~~~~~~~~~~~
    All indicators are calculated on data available up to and including
    the current bar. The confirmed signal is then shifted by 1 bar into
    `position`, meaning we only act on a signal after the bar that
    generated it has fully closed. The backtest must use `position`.

    Persistence filter
    ~~~~~~~~~~~~~~~~~~
    A raw signal of +1 or -1 is only confirmed if it held for min_hold
    consecutive bars. This prevents rapid flipping around a crossover
    and reduces total trade count to a realistic range.
    """

    df = df.copy()

    # ── 1. Moving Averages ───────────────────────────────────────────────────
    df["fast_ma"] = df["Close"].rolling(window=fast_window, min_periods=fast_window).mean()
    df["slow_ma"] = df["Close"].rolling(window=slow_window, min_periods=slow_window).mean()

    # ── 2. Rate of Change ────────────────────────────────────────────────────
    # ROC = ((Close - Close_n_bars_ago) / Close_n_bars_ago) * 100
    df["roc"] = (
        (df["Close"] - df["Close"].shift(momentum_window))
        / df["Close"].shift(momentum_window)
        * 100
    )

    # ── 3. Raw Signal ────────────────────────────────────────────────────────
    long_condition  = (df["fast_ma"] > df["slow_ma"]) & (df["roc"] > 0)
    short_condition = (df["fast_ma"] < df["slow_ma"]) & (df["roc"] < 0)

    raw = pd.Series(0, index=df.index)
    raw[long_condition]  =  1
    raw[short_condition] = -1

    # ── 4. Persistence Filter ────────────────────────────────────────────────
    # Signal confirmed only if raw held for min_hold consecutive bars.
    # rolling sum == min_hold  → all bars were +1
    # rolling sum == -min_hold → all bars were -1
    roll = raw.rolling(min_hold)

    long_confirmed  = roll.sum() ==  min_hold
    short_confirmed = roll.sum() == -min_hold

    df["signal"] = 0
    df.loc[long_confirmed,  "signal"] =  1
    df.loc[short_confirmed, "signal"] = -1

    # ── 5. Position (look-ahead-bias-free) ───────────────────────────────────
    # Shift by 1: today's confirmed signal becomes tomorrow's position.
    df["position"] = df["signal"].shift(1).fillna(0).astype(int)

    print(
        f"[signals] Windows — fast={fast_window}, slow={slow_window}, "
        f"momentum={momentum_window}, min_hold={min_hold}\n"
        f"[signals] Signal counts:\n"
        f"  Long  (+1): {(df['position'] ==  1).sum():>6,}\n"
        f"  Short (-1): {(df['position'] == -1).sum():>6,}\n"
        f"  Flat  ( 0): {(df['position'] ==  0).sum():>6,}"
    )

    return df


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from src.dataloader import get_data

    df = get_data()
    df = generate_signals(df)
    print(df[["Close", "fast_ma", "slow_ma", "roc", "signal", "position"]].head(10).to_string())
    print(f"\nUnique signals:   {df['signal'].value_counts().to_dict()}")
    print(f"Unique positions: {df['position'].value_counts().to_dict()}")


def generate_signals_v2(
    df: pd.DataFrame,
    fast_window: int = 30,
    slow_window: int = 80,
    momentum_window: int = 10,
    min_hold: int = 3,
    regime_window: int = 200,
) -> pd.DataFrame:
    """
    V2: Dual MA crossover with 200-day regime filter.
    Long only when Close > 200MA. Short only when Close < 200MA.
    """
    df = generate_signals(df, fast_window, slow_window, momentum_window, min_hold)

    ma200       = df["Close"].rolling(window=regime_window, min_periods=regime_window).mean()
    bull_regime = df["Close"] > ma200
    bear_regime = df["Close"] < ma200

    df.loc[(df["signal"] ==  1) & ~bull_regime, "signal"] = 0
    df.loc[(df["signal"] == -1) & ~bear_regime, "signal"] = 0

    df["position"] = df["signal"].shift(1).fillna(0).astype(int)

    print(
        f"[signals_v2] Regime-filtered counts:\n"
        f"  Long  (+1): {(df['position'] ==  1).sum():>6,}\n"
        f"  Short (-1): {(df['position'] == -1).sum():>6,}\n"
        f"  Flat  ( 0): {(df['position'] ==  0).sum():>6,}"
    )
    return df


def generate_signals_v3(
    df: pd.DataFrame,
    breakout_window: int = 20,
    regime_window: int = 200,
) -> pd.DataFrame:
    """
    V3: Donchian Channel Breakout with 200-day regime filter.

    Logic
    -----
    - upper_band : rolling 20-day highest High, shifted 1 bar (no look-ahead)
    - lower_band : rolling 20-day lowest Low,  shifted 1 bar (no look-ahead)
    - ma200      : 200-day MA of Close,         shifted 1 bar (no look-ahead)

    Signal rules:
      +1 when Close > upper_band AND Close > ma200  (bullish breakout)
      -1 when Close < lower_band AND Close < ma200  (bearish breakout)
       0 otherwise
    """
    df = df.copy()

    # All indicators shifted 1 bar — today's signal uses yesterday's values
    df["upper_band"] = df["High"].rolling(window=breakout_window, min_periods=breakout_window).max().shift(1)
    df["lower_band"] = df["Low"].rolling(window=breakout_window,  min_periods=breakout_window).min().shift(1)
    ma200            = df["Close"].rolling(window=regime_window,  min_periods=regime_window).mean().shift(1)

    long_cond  = (df["Close"] > df["upper_band"]) & (df["Close"] > ma200)
    short_cond = (df["Close"] < df["lower_band"]) & (df["Close"] < ma200)

    df["signal"] = 0
    df.loc[long_cond,  "signal"] =  1
    df.loc[short_cond, "signal"] = -1

    # Shift for look-ahead bias prevention
    df["position"] = df["signal"].shift(1).fillna(0).astype(int)

    print(
        f"[signals_v3] Donchian breakout_window={breakout_window}, regime={regime_window}\n"
        f"  Long  (+1): {(df['position'] ==  1).sum():>6,}\n"
        f"  Short (-1): {(df['position'] == -1).sum():>6,}\n"
        f"  Flat  ( 0): {(df['position'] ==  0).sum():>6,}"
    )
    return df


def generate_signals_v4(
    df: pd.DataFrame,
    fast_window: int = 20,
    slow_window: int = 50,
    momentum_window: int = 10,
    breakout_window: int = 20,
    regime_window: int = 200,
) -> pd.DataFrame:
    """
    V4: Regime-Switching Combined strategy.

    BULL regime (Close > 200MA):
      +1 when fast MA > slow MA AND ROC > 0
       0 otherwise — no shorting in bull

    BEAR regime (Close < 200MA):
      -1 when Close breaks below 20-day low
       0 otherwise — no longing in bear
    """
    df = df.copy()

    # All shifted 1 bar — no look-ahead bias
    fast_ma      = df["Close"].rolling(window=fast_window,     min_periods=fast_window).mean().shift(1)
    slow_ma      = df["Close"].rolling(window=slow_window,     min_periods=slow_window).mean().shift(1)
    ma200        = df["Close"].rolling(window=regime_window,   min_periods=regime_window).mean().shift(1)
    lower_band   = df["Low"].rolling(window=breakout_window,   min_periods=breakout_window).min().shift(1)

    roc = (
        (df["Close"] - df["Close"].shift(momentum_window))
        / df["Close"].shift(momentum_window)
        * 100
    ).shift(1)

    bull_regime = df["Close"] > ma200
    bear_regime = df["Close"] < ma200

    df["signal"] = 0

    # Bull: MA crossover + momentum confirmation
    long_cond  = bull_regime & (fast_ma > slow_ma) & (roc > 0)
    # Bear: price breaks below Donchian lower band
    short_cond = bear_regime & (df["Close"] < lower_band)

    df.loc[long_cond,  "signal"] =  1
    df.loc[short_cond, "signal"] = -1

    # Shift for look-ahead bias prevention
    df["position"] = df["signal"].shift(1).fillna(0).astype(int)

    print(
        f"[signals_v4] Regime-switching combined\n"
        f"  Long  (+1): {(df['position'] ==  1).sum():>6,}\n"
        f"  Short (-1): {(df['position'] == -1).sum():>6,}\n"
        f"  Flat  ( 0): {(df['position'] ==  0).sum():>6,}"
    )
    return df