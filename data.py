from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
import yfinance as yf

CACHE_DIR = Path("data_cache")
CACHE_DIR.mkdir(exist_ok=True)
META_DIR = CACHE_DIR / "_meta"
META_DIR.mkdir(exist_ok=True)


def _cache_file(ticker, interval):
    safe = ticker.replace("/", "_").replace(":", "_")
    return CACHE_DIR / f"{safe}_{interval}.csv"


def _meta_file(ticker, interval):
    safe = ticker.replace("/", "_").replace(":", "_")
    return META_DIR / f"{safe}_{interval}.txt"


def _save_metadata(ticker, interval, df):
    try:
        last_bar = df.index[-1].isoformat() if not df.empty else "N/A"
        now = datetime.now(timezone.utc).isoformat()
        _meta_file(ticker, interval).write_text(
            f"downloaded_at_utc={now}\n"
            f"last_bar={last_bar}\n"
            f"rows={len(df)}\n"
            f"interval={interval}\n",
            encoding="utf-8",
        )
    except Exception:
        pass


def download_history(ticker, period="2y", interval="1d", refresh=False):
    """Download OHLCV history, optionally bypassing the local cache."""
    cache = _cache_file(ticker, interval)

    if cache.exists() and not refresh:
        try:
            df = pd.read_csv(cache, index_col=0, parse_dates=True)
            if not df.empty:
                return df
        except Exception:
            pass

    try:
        df = yf.download(
            ticker,
            period=period,
            interval=interval,
            auto_adjust=True,
            progress=False,
            threads=False,
        )
    except Exception:
        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    needed = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        return pd.DataFrame()

    df = df[needed].copy()
    df.index = pd.to_datetime(df.index)
    df = df.apply(pd.to_numeric, errors="coerce").dropna()
    if df.empty:
        return df

    try:
        df.to_csv(cache)
        _save_metadata(ticker, interval, df)
    except Exception:
        pass

    return df
