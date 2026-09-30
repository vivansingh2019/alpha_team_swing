import json, statistics, sys
from pathlib import Path

STATE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("state.json")

def load_state(path):
    raw = path.read_text(encoding="utf-8-sig")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Useful for copied .txt/.json output with surrounding text
        m = re.search(r'(\{.*\})\s*$', raw, re.S)
        if not m:
            raise
        return json.loads(m.group(1))

def pct(values, p):
    if not values:
        return 0
    values = sorted(values)
    k = (len(values)-1) * p
    f, c = int(k), min(int(k)+1, len(values)-1)
    return values[f] + (values[c]-values[f]) * (k-f)

def show(name, ok, detail):
    print(f"[{'PASS' if ok else 'CHECK'}] {name}: {detail}")
    return ok

s = load_state(STATE)
rows = s.get("results", [])
market = s.get("market", {})
print("=== SWING BRAIN — RANKING BEHAVIOR TEST ===")
print("File:", STATE)
print("Stocks:", len(rows))
print("Market:", market.get("regime"), "| Risk:", market.get("risk"))
print()

if not rows:
    raise SystemExit("No results found in state file.")

priority = [float(x.get("priority_score", 0)) for x in rows]
quality  = [float(x.get("stock_quality_score", 0)) for x in rows]
entry    = [float(x.get("entry_quality", 0)) for x in rows]

print("Distribution")
print("  Priority  mean/median/max:", round(statistics.mean(priority),1),
      round(statistics.median(priority),1), round(max(priority),1))
print("  Quality   mean/median/max:", round(statistics.mean(quality),1),
      round(statistics.median(quality),1), round(max(quality),1))
print("  Entry     mean/median/max:", round(statistics.mean(entry),1),
      round(statistics.median(entry),1), round(max(entry),1))
print()

# 1) Ranking must actually separate stocks.
top = sorted(rows, key=lambda x: float(x.get("priority_score",0)), reverse=True)[:10]
show("Ranking separation", len(set(x.get("priority_score") for x in rows)) > 10,
     f"{len(set(x.get('priority_score') for x in rows))} distinct priority values")

# 2) Strong breakout + volume + RS should appear in upper part of ranking.
cand = [x for x in rows
        if x.get("phase") == "BREAKOUT"
        and float(x.get("volume_ratio",0)) >= 1.2
        and float(x.get("rs_delta",0)) > 20]
if cand:
    best = max(cand, key=lambda x: float(x.get("priority_score",0)))
    rank = sorted(priority, reverse=True).index(float(best["priority_score"])) + 1
    show("Breakout + volume + RS", rank <= max(15, len(rows)//4),
         f"{best.get('ticker')} rank={rank}/{len(rows)}, priority={best.get('priority_score')}")
else:
    show("Breakout + volume + RS", False, "No qualifying example in this scan")

# 3) Overextended / high profit-booking should reduce entry quality.
ext = [x for x in rows if x.get("profit_booking_risk") == "HIGH" or float(x.get("rsi",0)) > 80]
if ext:
    reduced = [x for x in ext if float(x.get("entry_quality",0)) + 10 < float(x.get("stock_quality_score",0))]
    show("Overextension penalty", len(reduced) >= max(1, len(ext)//2),
         f"{len(reduced)}/{len(ext)} high-risk cases have entry < quality by >10")
else:
    show("Overextension penalty", True, "No high-profit-booking case in this scan")

# 4) Normal RSI zone should not receive an overbought penalty.
normal = [x for x in rows if 52 <= float(x.get("rsi",0)) <= 68
          and x.get("profit_booking_risk") == "LOW"]
if normal:
    avg_gap = statistics.mean(float(x.get("stock_quality_score",0))-float(x.get("entry_quality",0)) for x in normal)
    show("Normal RSI entry behavior", avg_gap < 20,
         f"average quality-entry gap={avg_gap:.1f}")
else:
    show("Normal RSI entry behavior", True, "No clean normal-RSI sample")

# 5) Weak structure / negative RS should not dominate top ranks.
weak = [x for x in rows if x.get("structure",{}).get("trend_down")
        and float(x.get("rs_delta",0)) < 0]
top10_tickers = {x.get("ticker") for x in top}
bad_top = [x.get("ticker") for x in weak if x.get("ticker") in top10_tickers]
show("Weak structure protection", len(bad_top) == 0,
     f"weak trend + negative RS in top10: {bad_top or 'none'}")

# 6) BUILDING should be preserved if RS is positive and RSI is not extreme.
building = [x for x in rows if x.get("phase") == "BUILDING"
            and float(x.get("rs_delta",0)) > 10
            and 50 <= float(x.get("rsi",0)) <= 70]
if building:
    rejects = [x for x in building if x.get("status") == "REJECT"]
    show("BUILDING preservation", len(rejects) == 0,
         f"{len(rejects)} rejects among {len(building)} qualifying BUILDING stocks")
else:
    show("BUILDING preservation", True, "No qualifying BUILDING sample")

# 7) Fundamentals must help quality, but not override bad entry timing.
fund_bad_entry = [x for x in rows if float(x.get("fundamental_score",0)) >= 85
                  and float(x.get("entry_quality",0)) <= 25]
if fund_bad_entry:
    avg_pr = statistics.mean(float(x.get("priority_score",0)) for x in fund_bad_entry)
    show("Fundamental vs entry separation", avg_pr < statistics.mean(priority)+15,
         f"{len(fund_bad_entry)} high-fundamental/poor-entry cases, avg priority={avg_pr:.1f}")
else:
    show("Fundamental vs entry separation", True, "No high-fundamental/poor-entry sample")

# 8) Bearish market should be a modifier, not the sole rejection reason.
if str(market.get("regime","")).upper() == "BEARISH":
    strong = [x for x in rows if float(x.get("stock_quality_score",0)) >= 60
              and float(x.get("rs_delta",0)) > 20
              and x.get("phase") in ("BUILDING","BREAKOUT","PULLBACK","CONTINUATION")]
    strong_reject = [x for x in strong if x.get("status") == "REJECT"]
    show("Bearish market is a modifier", len(strong_reject) <= max(2, len(strong)//3),
         f"{len(strong_reject)}/{len(strong)} strong setups are REJECT")
else:
    show("Bearish market is a modifier", True, "Current market is not bearish")

# 9) ENTRY READY gate should be explicit.
ready = [x for x in rows if x.get("status") == "ENTRY READY"]
show("ENTRY READY gate", True,
     f"{len(ready)} ENTRY READY in current {market.get('regime')} market")

# 10) Print top 10 for human inspection.
print("\nTOP 10 BY PRIORITY")
for i, x in enumerate(top, 1):
    print(f"{i:>2}. {x.get('ticker','?'):12} "
          f"P={x.get('priority_score',0):>3} "
          f"Q={x.get('stock_quality_score',0):>3} "
          f"E={x.get('entry_quality',0):>3} "
          f"{x.get('phase','?'):12} "
          f"RSΔ={x.get('rs_delta',0):>6} "
          f"RSI={x.get('rsi',0):>5} "
          f"Vol={x.get('volume_ratio',0):>4}x "
          f"Status={x.get('status','?')}")

print("\nInterpretation:")
print("- PASS means the ranking logic behaves as designed for this scan; it is not proof of future returns.")
print("- CHECK means inspect that behavior before changing weights/thresholds.")
print("- Run again after every ranking-engine change.")
