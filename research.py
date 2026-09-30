from brain import market_regime, setup_analysis
from data import download_history
from fundamental import get_fundamentals
from config import BENCHMARK


def analyze_symbol(yf_symbol, display_symbol=None):
    bench = download_history(BENCHMARK, period="2y")
    df = download_history(yf_symbol, period="2y")
    if bench.empty or df.empty:
        raise RuntimeError("Price data unavailable")
    market = market_regime(bench)
    f = get_fundamentals(yf_symbol)
    result = setup_analysis(display_symbol or yf_symbol.replace(".NS", ""), df, bench, market, f)
    if result is None:
        raise RuntimeError("Not enough history for analysis")
    result["fundamentals"] = f
    result["company"] = f.get("company", display_symbol or yf_symbol)
    result["sector"] = f.get("sector", "Unknown")
    result["industry"] = f.get("industry", "Unknown")
    result["yf_symbol"] = yf_symbol
    result["history"] = df
    return result
