"""Lightweight fundamental enrichment using Yahoo Finance data with local caching.

Fundamentals are a ranking input, not a hard gate.
Yahoo Finance returns some ratio fields as decimals (e.g. 0.291 = 29.1%)
while debtToEquity is commonly returned as a percentage (e.g. 122.8 = 1.228x).
This module normalizes those fields before scoring/displaying them.
"""
from pathlib import Path
import json
import math
import time
import yfinance as yf

FUND_CACHE = Path("fundamental_cache")
FUND_CACHE.mkdir(exist_ok=True)
FUND_VERSION = 2


def _clean(v):
    try:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return None
        value = float(v)
        if not math.isfinite(value):
            return None
        return value
    except Exception:
        return None


def _ratio(v):
    """Normalize Yahoo ratio/percentage fields to decimal form.

    Examples:
      0.391  -> 0.391 (39.1%)
      39.1   -> 0.391 (39.1%)
      1.589  -> 1.589 (158.9%)
      158.9  -> 1.589 (158.9%)
    """
    value = _clean(v)
    if value is None:
        return None
    if abs(value) > 2:
        return value / 100.0
    return value


def _debt_to_equity(v):
    """Normalize Yahoo debtToEquity to an x multiple.

    Yahoo commonly reports debtToEquity as a percentage, so 122.8 means
    122.8%, i.e. 1.228x debt/equity.
    """
    value = _clean(v)
    if value is None:
        return None
    # Yahoo/yfinance has appeared in both ratio-style and percentage-style
    # representations across endpoints. Treat values above 2 as percentages.
    return value / 100.0 if abs(value) > 2 else value


def _pct(v):
    value = _clean(v)
    return None if value is None else round(value * 100, 2)


def _score(f):
    earned = 0.0
    possible = 0.0

    eg = f.get("earnings_growth")
    if eg is not None:
        earned += 25 if eg >= 0.25 else 18 if eg >= 0.12 else 10 if eg >= 0 else 0
        possible += 25

    rg = f.get("revenue_growth")
    if rg is not None:
        earned += 15 if rg >= 0.20 else 11 if rg >= 0.10 else 6 if rg >= 0 else 0
        possible += 15

    roe = f.get("roe")
    if roe is not None:
        earned += 15 if roe >= 0.20 else 11 if roe >= 0.12 else 6 if roe >= 0 else 0
        possible += 15

    pe = f.get("pe")
    eg_pct = (eg * 100) if eg is not None else None
    if pe is not None and pe > 0:
        if eg_pct and eg_pct > 0:
            peg = pe / eg_pct
            earned += 20 if peg <= 1.2 else 15 if peg <= 1.8 else 9 if peg <= 2.5 else 3
        else:
            earned += 14 if pe <= 20 else 9 if pe <= 35 else 4
        possible += 20

    de = f.get("debt_to_equity")
    if de is not None:
        earned += 15 if de <= 0.4 else 11 if de <= 1.0 else 6 if de <= 2 else 1
        possible += 15

    margin = f.get("profit_margin")
    if margin is not None:
        earned += 10 if margin >= 0.15 else 7 if margin >= 0.08 else 4 if margin >= 0 else 0
        possible += 10

    return int(round(earned / possible * 100)) if possible else 50


def _normalize_cached(f):
    """Normalize an older cache record created by pre-v2 code."""
    if not isinstance(f, dict):
        return None
    if f.get("fundamental_version") == FUND_VERSION:
        return f

    # Old cache values may have been stored in mixed units. Rebuild the
    # normalized fields from the displayed percentage fields when possible.
    def old_ratio(raw, pct_key):
        raw_v = _clean(raw)
        if raw_v is not None:
            # Old code stored the raw Yahoo value. Normalize using the same
            # rules as the new fetch path.
            return _ratio(raw_v)
        pct_v = _clean(f.get(pct_key))
        return None if pct_v is None else pct_v / 100.0

    f["earnings_growth"] = old_ratio(f.get("earnings_growth"), "earnings_growth_pct")
    f["revenue_growth"] = old_ratio(f.get("revenue_growth"), "revenue_growth_pct")
    f["roe"] = old_ratio(f.get("roe"), "roe_pct")
    f["profit_margin"] = old_ratio(f.get("profit_margin"), "profit_margin_pct")

    # Old debtToEquity was stored directly from Yahoo (percentage-style).
    old_de = _clean(f.get("debt_to_equity"))
    f["debt_to_equity"] = None if old_de is None else (old_de / 100.0 if abs(old_de) > 2 else old_de)

    pe = _clean(f.get("pe"))
    eg = f.get("earnings_growth")
    f["peg"] = round(pe / (eg * 100), 2) if pe and eg and eg > 0 else None
    f["fundamental_score"] = _score(f)
    f["earnings_growth_pct"] = _pct(f.get("earnings_growth"))
    f["revenue_growth_pct"] = _pct(f.get("revenue_growth"))
    f["roe_pct"] = _pct(f.get("roe"))
    f["profit_margin_pct"] = _pct(f.get("profit_margin"))
    f["fundamental_version"] = FUND_VERSION
    return f


def get_fundamentals(symbol, refresh=False, pause=0.0):
    safe = symbol.replace("/", "_")
    path = FUND_CACHE / f"{safe}.json"

    if path.exists() and not refresh:
        try:
            cached = _normalize_cached(json.loads(path.read_text(encoding="utf-8")))
            if cached is not None:
                path.write_text(json.dumps(cached, indent=2), encoding="utf-8")
                return cached
        except Exception:
            pass

    if pause:
        time.sleep(pause)

    ticker = yf.Ticker(symbol)
    try:
        info = ticker.get_info()
    except Exception as exc:
        return {"available": False, "error": str(exc), "fundamental_score": 50}

    earnings_growth = _ratio(info.get("earningsGrowth"))
    revenue_growth = _ratio(info.get("revenueGrowth"))
    roe = _ratio(info.get("returnOnEquity"))
    roa = _ratio(info.get("returnOnAssets"))
    profit_margin = _ratio(info.get("profitMargins"))
    operating_margin = _ratio(info.get("operatingMargins"))
    debt_to_equity = _debt_to_equity(info.get("debtToEquity"))
    pe = _clean(info.get("trailingPE"))

    f = {
        "available": True,
        "fundamental_version": FUND_VERSION,
        "market_cap": _clean(info.get("marketCap")),
        "pe": pe,
        "forward_pe": _clean(info.get("forwardPE")),
        "eps": _clean(info.get("trailingEps")),
        "earnings_growth": earnings_growth,
        "revenue_growth": revenue_growth,
        "roe": roe,
        "roa": roa,
        "debt_to_equity": debt_to_equity,
        "profit_margin": profit_margin,
        "operating_margin": operating_margin,
        "sector": info.get("sector") or "Unknown",
        "industry": info.get("industry") or "Unknown",
        "company": info.get("longName") or info.get("shortName") or symbol,
    }

    f["peg"] = round(pe / (earnings_growth * 100), 2) if pe and earnings_growth and earnings_growth > 0 else None
    f["fundamental_score"] = _score(f)
    f["earnings_growth_pct"] = _pct(f.get("earnings_growth"))
    f["revenue_growth_pct"] = _pct(f.get("revenue_growth"))
    f["roe_pct"] = _pct(f.get("roe"))
    f["profit_margin_pct"] = _pct(f.get("profit_margin"))

    path.write_text(json.dumps(f, indent=2), encoding="utf-8")
    return f
