import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from brain import setup_analysis, market_regime
from config import BENCHMARK, MAX_WORKERS, MAX_STOCKS, MIN_HISTORY, FUNDAMENTAL_TOP_N
from data import download_history
from fundamental import get_fundamentals
from universe import load_universe

IST = ZoneInfo("Asia/Kolkata")


def _scan_one(row, bench, market, fresh_data=True):
    ticker = row["YF_SYMBOL"]
    try:
        df = download_history(ticker, period="2y", refresh=fresh_data)
        if len(df) < MIN_HISTORY:
            return None
        result = setup_analysis(row["SYMBOL"], df, bench, market)
        if result:
            result["company"] = row.get("COMPANY", row["SYMBOL"])
            result["series"] = row.get("SERIES", "EQ")
            result["yf_symbol"] = ticker
            result["data_last_bar"] = df.index[-1].isoformat()
            result["_history"] = df
        return result
    except Exception as exc:
        return {"ticker": row["SYMBOL"], "error": str(exc)}


def _enrich_fundamentals(results, bench, market):
    ranked = sorted(results, key=lambda r: r.get("technical_score", r.get("score", 0)), reverse=True)
    selected = ranked[:FUNDAMENTAL_TOP_N]
    selected_symbols = {r["ticker"] for r in selected}

    for r in results:
        if r["ticker"] not in selected_symbols:
            r["fundamental_available"] = False
            r["fundamental_score"] = 50
            continue

        f = get_fundamentals(r.get("yf_symbol", r["ticker"] + ".NS"))
        history = r.get("_history")

        if history is not None and f.get("available"):
            reranked = setup_analysis(r["ticker"], history, bench, market, fundamental=f)
            if reranked:
                company = r.get("company", r["ticker"])
                series = r.get("series", "EQ")
                yf_symbol = r.get("yf_symbol", r["ticker"] + ".NS")
                data_last_bar = r.get("data_last_bar")
                r.clear()
                r.update(reranked)
                r["company"] = company
                r["series"] = series
                r["yf_symbol"] = yf_symbol
                r["data_last_bar"] = data_last_bar

        r["fundamental_available"] = bool(f.get("available"))
        r["fundamental_score"] = int(f.get("fundamental_score", 50))
        r["fundamentals"] = f
        r["sector"] = f.get("sector", "Unknown")
        r["industry"] = f.get("industry", "Unknown")


def _remove_internal_fields(results):
    for r in results:
        r.pop("_history", None)


def run(refresh_universe=False, max_stocks=None, fresh_data=True):
    scan_started = datetime.now(IST)
    universe = load_universe(refresh=refresh_universe)
    limit = max_stocks if max_stocks is not None else MAX_STOCKS
    if limit > 0:
        universe = universe.head(limit)

    # Benchmark is also refreshed so market regime is not based on an old cache.
    bench = download_history(BENCHMARK, period="2y", refresh=fresh_data)
    if bench.empty:
        raise RuntimeError("Benchmark data unavailable")
    market = market_regime(bench)

    results, errors = [], []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = [pool.submit(_scan_one, row, bench, market, fresh_data) for _, row in universe.iterrows()]
        for future in as_completed(futures):
            result = future.result()
            if not result:
                continue
            if "error" in result:
                errors.append(result)
            else:
                results.append(result)

    _enrich_fundamentals(results, bench, market)
    _remove_internal_fields(results)

    # Radar order is driven by the new V2 priority, not legacy status/score.
    # Status remains a label; priority is the ranking value.
    results.sort(key=lambda x: (x.get("priority_score", 0), x.get("stock_quality_score", 0), x.get("entry_quality", 0)), reverse=True)

    lifecycle = {
        "breakout_candidates": sum(bool(x.get("breakout")) for x in results),
        "strong_breakouts": sum(x.get("breakout_state") == "STRONG BREAKOUT" for x in results),
        "confirmed_retests": sum(x.get("breakout_state") == "CONFIRMED / RETEST" for x in results),
        "failed_breakouts": sum(x.get("breakout_state") == "FAILED BREAKOUT" for x in results),
        "technical_entry_candidates": sum(bool(x.get("technical_entry_candidate")) for x in results),
        "market_blocked_entries": sum(bool(x.get("market_gate_blocked")) for x in results),
    }

    generated = datetime.now(IST)
    payload = {
        "generated_at": generated.isoformat(timespec="seconds"),
        "scan_started_at": scan_started.isoformat(timespec="seconds"),
        "scan_finished_at": generated.isoformat(timespec="seconds"),
        "timezone": "Asia/Kolkata",
        "data_mode": "FRESH_REQUEST" if fresh_data else "CACHE_ALLOWED",
        "data_source": "Yahoo Finance via yfinance",
        "benchmark": BENCHMARK,
        "universe_source": "NSE EQ",
        "universe_count": len(universe),
        "stocks_scanned": len(results),
        "errors": len(errors),
        "market": market,
        "lifecycle": lifecycle,
        "results": results,
    }
    Path("state.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({k: payload[k] for k in payload if k != "results"}, indent=2))
    print(f"Results: {len(results)} | Errors: {len(errors)}")


if __name__ == "__main__":
    refresh = os.getenv("REFRESH_UNIVERSE", "0") == "1"
    limit = int(os.getenv("MAX_STOCKS", "0"))
    fresh = os.getenv("FRESH_DATA", "1") != "0"
    run(refresh_universe=refresh, max_stocks=limit or None, fresh_data=fresh)
