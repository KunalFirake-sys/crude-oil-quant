"""
data_loader.py
--------------
Downloads, cleans, and serves WTI crude oil futures (CL=F) OHLCV data.

Typical usage:
    from src.data_loader import get_data
    df = get_data()
"""

import os
import pandas as pd
import yfinance as yf

# ── Constants ────────────────────────────────────────────────────────────────
TICKER      = "CL=F"
START_DATE  = "2015-01-01"
END_DATE    = "2024-12-31"
DATA_DIR    = "data"
CSV_PATH    = os.path.join(DATA_DIR, "cl_futures_data.csv")
REQUIRED_COLS = ["Open", "High", "Low", "Close", "Volume"]


# ── Private helpers ──────────────────────────────────────────────────────────

def _download_raw() -> pd.DataFrame:
    """
    Download raw OHLCV data from Yahoo Finance for the configured ticker
    and date range.

    Returns
    -------
    pd.DataFrame
        Raw DataFrame as returned by yfinance (MultiIndex columns possible).

    Raises
    ------
    ValueError
        If yfinance returns an empty DataFrame.
    """
    print(f"[data_loader] Downloading {TICKER} from Yahoo Finance "
          f"({START_DATE} → {END_DATE}) ...")

    raw = yf.download(
        TICKER,
        start=START_DATE,
        end=END_DATE,
        auto_adjust=True,   # adjusts OHLC for splits/dividends
        progress=False,
    )

    if raw.empty:
        raise ValueError(
            f"yfinance returned no data for {TICKER}. "
            "Check your internet connection or ticker symbol."
        )

    return raw


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean a raw OHLCV DataFrame.

    Steps
    -----
    1. Flatten MultiIndex columns if present (yfinance ≥0.2 quirk).
    2. Keep only OHLCV columns.
    3. Ensure the index is a proper DatetimeIndex named 'Date'.
    4. Forward-fill missing OHLCV values (carries last known price forward).
    5. Drop any remaining rows where Close is still NaN.
    6. Sort chronologically and remove duplicate dates.

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame from yfinance.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with columns [Open, High, Low, Close, Volume]
        and a DatetimeIndex named 'Date'.
    """
    # 1. Flatten MultiIndex columns (e.g. ('Close', 'CL=F') → 'Close')
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    # 2. Keep only the columns we need
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise KeyError(f"Expected columns not found after download: {missing}")

    df = df[REQUIRED_COLS].copy()

    # 3. Normalise index
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"

    # 4. Forward-fill gaps in all columns
    df.ffill(inplace=True)

    # 5. Drop rows where Close is still NaN (e.g. very first row if no prior data)
    before = len(df)
    df.dropna(subset=["Close"], inplace=True)
    dropped = before - len(df)
    if dropped:
        print(f"[data_loader] Dropped {dropped} row(s) with no Close price.")

    # 6. Sort and deduplicate
    df.sort_index(inplace=True)
    df = df[~df.index.duplicated(keep="first")]

    return df


def _save(df: pd.DataFrame) -> None:
    """
    Persist the cleaned DataFrame to CSV.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned OHLCV DataFrame to save.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(CSV_PATH)
    print(f"[data_loader] Saved {len(df)} rows → {CSV_PATH}")


def _load_csv() -> pd.DataFrame:
    """
    Load and lightly validate the cached CSV from disk.

    Returns
    -------
    pd.DataFrame
        Cleaned OHLCV DataFrame with DatetimeIndex named 'Date'.

    Raises
    ------
    FileNotFoundError
        If the CSV does not exist at CSV_PATH.
    """
    print(f"[data_loader] Loading cached data from {CSV_PATH} ...")
    df = pd.read_csv(CSV_PATH, index_col="Date", parse_dates=True)
    return df


def _print_summary(df: pd.DataFrame) -> None:
    """
    Print a concise summary of the loaded dataset, including date range,
    row count, and any calendar-day gaps longer than 5 days.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned OHLCV DataFrame with DatetimeIndex.
    """
    print("\n" + "=" * 50)
    print(f"  WTI Crude Oil Futures — Data Summary")
    print("=" * 50)
    print(f"  Date range : {df.index.min().date()} → {df.index.max().date()}")
    print(f"  Rows       : {len(df):,}")
    print(f"  Columns    : {list(df.columns)}")
    print(f"  Close NaNs : {df['Close'].isna().sum()}")

    # Detect gaps > 5 calendar days between consecutive trading dates
    gaps = df.index.to_series().diff().dt.days
    large_gaps = gaps[gaps > 5].dropna()

    if large_gaps.empty:
        print("  Gaps > 5d  : None found")
    else:
        print(f"  Gaps > 5d  : {len(large_gaps)} found")
        for date, gap in large_gaps.items():
            print(f"               {date.date()}  ({int(gap)} days since prior row)")

    print("=" * 50 + "\n")


# ── Public API ───────────────────────────────────────────────────────────────

def get_data(force_download: bool = False) -> pd.DataFrame:
    """
    Primary entry point. Returns a clean OHLCV DataFrame.

    Behaviour
    ---------
    - If ``data/cl_futures_data.csv`` exists (and ``force_download`` is False),
      loads from disk (fast, no network needed).
    - Otherwise, downloads from Yahoo Finance, cleans, saves, then returns.

    Parameters
    ----------
    force_download : bool, optional
        Set True to ignore any cached CSV and re-download. Default False.

    Returns
    -------
    pd.DataFrame
        Columns : Open, High, Low, Close, Volume
        Index   : DatetimeIndex named 'Date', sorted ascending, no duplicates.

    Examples
    --------
    >>> from src.data_loader import get_data
    >>> df = get_data()
    >>> df.head()
    """
    if not force_download and os.path.exists(CSV_PATH):
        df = _load_csv()
    else:
        raw = _download_raw()
        df  = _clean(raw)
        _save(df)

    _print_summary(df)
    return df

if __name__ == "__main__":
    df = get_data()