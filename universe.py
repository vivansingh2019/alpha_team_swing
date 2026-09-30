"""Dynamic NSE equity universe loader.

The scanner is intentionally not hard-coded to NIFTY 500. It can use the
current NSE equity master when internet access is available, or a local
universe.csv cache/fallback.
"""
from pathlib import Path
import io
import pandas as pd
import requests

NSE_EQUITY_URL = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
CACHE = Path("universe.csv")


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    cols = {c.strip().upper(): c for c in df.columns}
    symbol_col = cols.get("SYMBOL")
    if not symbol_col:
        raise ValueError("Universe file must contain SYMBOL column")
    out = pd.DataFrame()
    out["SYMBOL"] = df[symbol_col].astype(str).str.strip().str.upper()
    if "NAME OF COMPANY" in cols:
        out["COMPANY"] = df[cols["NAME OF COMPANY"]].astype(str).str.strip()
    else:
        out["COMPANY"] = out["SYMBOL"]
    if "SERIES" in cols:
        out["SERIES"] = df[cols["SERIES"]].astype(str).str.strip().str.upper()
    else:
        out["SERIES"] = "EQ"
    out = out[(out["SYMBOL"] != "") & (out["SERIES"] == "EQ")]
    out = out.drop_duplicates("SYMBOL").sort_values("SYMBOL").reset_index(drop=True)
    out["YF_SYMBOL"] = out["SYMBOL"] + ".NS"
    return out


def fetch_nse_universe(timeout=20) -> pd.DataFrame:
    headers = {"User-Agent": "Mozilla/5.0", "Accept": "text/csv,*/*"}
    r = requests.get(NSE_EQUITY_URL, headers=headers, timeout=timeout)
    r.raise_for_status()
    df = pd.read_csv(io.BytesIO(r.content))
    out = _clean(df)
    if len(out) < 500:
        raise RuntimeError(f"Unexpectedly small NSE universe: {len(out)}")
    out.to_csv(CACHE, index=False)
    return out


def load_universe(refresh=False) -> pd.DataFrame:
    if refresh or not CACHE.exists():
        try:
            return fetch_nse_universe()
        except Exception as exc:
            if not CACHE.exists():
                raise RuntimeError(
                    "Could not fetch NSE universe and no universe.csv cache exists. "
                    f"Reason: {exc}"
                ) from exc
    return _clean(pd.read_csv(CACHE))


if __name__ == "__main__":
    u = load_universe(refresh=True)
    print(f"Loaded {len(u)} NSE EQ symbols")
    print(u.head(10).to_string(index=False))
