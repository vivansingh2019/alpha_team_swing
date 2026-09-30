from typing import List, Dict
import numpy as np
import pandas as pd

from config import ATR_PERIOD, RS_PERIOD, VOLUME_LOOKBACK, MIN_RR

# Master Pine alignment
BREAKOUT_BUFFER_PCT = 0.15
RES_ZONE_PCT = 1.0
FOLLOW_BARS = 3
FRESH_ATR = 0.75
HEALTHY_ATR = 1.5
CHASE_ATR = 2.5
PIVOT_LEN = 3


def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def rsi(s, n=14):
    d = s.diff()
    up = d.clip(lower=0)
    down = -d.clip(upper=0)
    au = up.ewm(alpha=1 / n, adjust=False).mean()
    ad = down.ewm(alpha=1 / n, adjust=False).mean()
    rs = au / ad.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def atr(df, n=14):
    pc = df.Close.shift(1)
    tr = pd.concat([(df.High - df.Low), (df.High - pc).abs(), (df.Low - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def enrich(df):
    d = df.copy().dropna().sort_index()
    d["EMA20"] = ema(d.Close, 20)
    d["EMA50"] = ema(d.Close, 50)
    d["EMA200"] = ema(d.Close, 200)
    d["RSI"] = rsi(d.Close)
    d["ATR"] = atr(d, ATR_PERIOD)
    d["VOL20"] = d.Volume.rolling(VOLUME_LOOKBACK).mean()
    d["VOL_RATIO"] = d.Volume / d.VOL20.replace(0, np.nan)
    d["HIGH20"] = d.High.shift(1).rolling(20).max()
    d["HIGH50"] = d.High.shift(1).rolling(50).max()
    d["LOW20"] = d.Low.shift(1).rolling(20).min()
    d["LOW50"] = d.Low.shift(1).rolling(50).min()
    d["VWAP20"] = (d.Close * d.Volume).rolling(20).sum() / d.Volume.rolling(20).sum()
    d["RANGE20"] = (d.High.rolling(20).max() - d.Low.rolling(20).min()) / d.Close
    d["ATR_PCT"] = d.ATR / d.Close
    d["RS63"] = d.Close / d.Close.shift(RS_PERIOD) - 1
    return d


def market_regime(bench):
    d = enrich(bench)
    if len(d) < 210:
        return {"regime": "UNKNOWN", "score": 5, "risk": "UNKNOWN"}
    x = d.iloc[-1]
    ema50_slope = d.EMA50.iloc[-1] - d.EMA50.iloc[-21]
    if x.Close > x.EMA50 > x.EMA200 and ema50_slope > 0:
        return {"regime": "BULLISH", "score": 10, "risk": "LOW"}
    if x.Close < x.EMA50 < x.EMA200 and ema50_slope < 0:
        return {"regime": "BEARISH", "score": 3, "risk": "HIGH"}
    if x.Close >= x.EMA50 and ema50_slope >= 0:
        return {"regime": "BULLISH_BUT_MIXED", "score": 7, "risk": "MODERATE"}
    return {"regime": "MIXED", "score": 5, "risk": "MODERATE"}


def confirmed_pivots(d, pivot_len=PIVOT_LEN):
    """Return last confirmed pivot high/low using the Master Pine pivot logic.

    A pivot at bar i is only available at i+pivot_len, so the current bar never
    gets to use a future-confirmed pivot. This mirrors ta.pivothigh/ta.pivotlow.
    """
    n = len(d)
    if n < 2 * pivot_len + 1:
        return np.nan, np.nan, np.nan, np.nan
    highs = d.High.to_numpy(dtype=float)
    lows = d.Low.to_numpy(dtype=float)
    ph_idx = []
    pl_idx = []
    for i in range(pivot_len, n - pivot_len):
        if highs[i] == np.max(highs[i-pivot_len:i+pivot_len+1]):
            ph_idx.append(i)
        if lows[i] == np.min(lows[i-pivot_len:i+pivot_len+1]):
            pl_idx.append(i)
    confirmed_cutoff = n - 1 - pivot_len
    ph = [i for i in ph_idx if i <= confirmed_cutoff]
    pl = [i for i in pl_idx if i <= confirmed_cutoff]
    last_ph = highs[ph[-1]] if ph else np.nan
    prev_ph = highs[ph[-2]] if len(ph) >= 2 else np.nan
    last_pl = lows[pl[-1]] if pl else np.nan
    prev_pl = lows[pl[-2]] if len(pl) >= 2 else np.nan
    return last_ph, prev_ph, last_pl, prev_pl


def detect_structure(d):
    x = d.iloc[-1]
    recent_high = d.High.iloc[-21:-1].max()
    recent_low = d.Low.iloc[-21:-1].min()
    prev_high = d.High.iloc[-42:-21].max()
    prev_low = d.Low.iloc[-42:-21].min()
    bos_up = x.Close > recent_high
    bos_down = x.Close < recent_low
    trend_up = recent_high > prev_high and recent_low > prev_low
    trend_down = recent_high < prev_high and recent_low < prev_low
    last_ph, prev_ph, last_pl, prev_pl = confirmed_pivots(d)
    return {
        "bos_up": bool(bos_up), "bos_down": bool(bos_down),
        "trend_up": bool(trend_up), "trend_down": bool(trend_down),
        "recent_high": float(recent_high), "recent_low": float(recent_low),
        "last_ph": None if pd.isna(last_ph) else float(last_ph),
        "prev_ph": None if pd.isna(prev_ph) else float(prev_ph),
        "last_pl": None if pd.isna(last_pl) else float(last_pl),
        "prev_pl": None if pd.isna(prev_pl) else float(prev_pl),
    }


def detect_smc(d):
    x = d.iloc[-1]
    a = float(x.ATR) if pd.notna(x.ATR) and x.ATR > 0 else max(float(x.Close) * 0.01, 1)
    prev20h = d.High.iloc[-21:-1].max()
    prev20l = d.Low.iloc[-21:-1].min()
    sweep_low = x.Low < prev20l and x.Close > prev20l
    sweep_high = x.High > prev20h and x.Close < prev20h
    disp = abs(x.Close - d.Close.iloc[-2]) > 0.8 * a
    ob_bull = bool(d.Close.iloc[-2] < d.Open.iloc[-2] and disp and x.Close > d.High.iloc[-2])
    ob_bear = bool(d.Close.iloc[-2] > d.Open.iloc[-2] and disp and x.Close < d.Low.iloc[-2])
    fvg_bull = bool(d.Low.iloc[-1] > d.High.iloc[-3])
    fvg_bear = bool(d.High.iloc[-1] < d.Low.iloc[-3])
    return {
        "sweep_low": bool(sweep_low), "sweep_high": bool(sweep_high),
        "displacement": bool(disp), "ob_bull": ob_bull, "ob_bear": ob_bear,
        "fvg_bull": fvg_bull, "fvg_bear": fvg_bear,
    }


def relative_strength(stock, bench):
    n = min(RS_PERIOD, len(stock) - 1, len(bench) - 1)
    if n <= 0:
        return 0.0, 0.0, 0.0
    sr = stock.Close.iloc[-1] / stock.Close.iloc[-n - 1] - 1
    br = bench.Close.iloc[-1] / bench.Close.iloc[-n - 1] - 1
    return float(sr), float(br), float(sr - br)


def _breakout_engine(d):
    """Port of Master Pine V5.7 breakout quality + entry/risk engine."""
    x = d.iloc[-1]
    resistance_series = d.High.rolling(60).max()
    breakout_res_series = resistance_series.shift(1)
    buffer_series = breakout_res_series * BREAKOUT_BUFFER_PCT / 100.0
    breakout_series = d.Close > (breakout_res_series + buffer_series)
    attempt_series = d.High > (breakout_res_series + buffer_series)

    body = (d.Close - d.Open).abs()
    rng = (d.High - d.Low).replace(0, np.nan)
    close_location = (d.Close - d.Low) / rng
    body_pct = body / rng
    strong_close = close_location >= 0.70
    large_body = body_pct >= 0.55
    strong_candle = breakout_series & strong_close & large_body
    volume_confirm = breakout_series & (d.VOL_RATIO >= 1.2)
    acceptance = breakout_series & (d.Close > breakout_res_series) & (d.Low >= breakout_res_series * (1.0 - RES_ZONE_PCT / 100.0))

    prev_break = breakout_series.shift(1).eq(True)
    events = breakout_series & ~prev_break
    event_indices = np.flatnonzero(events.to_numpy())
    event_idx = int(event_indices[-1]) if len(event_indices) else None

    bars_since = None if event_idx is None else len(d) - 1 - event_idx
    current_res = breakout_res_series.iloc[-1]
    failed = False
    retest = False
    if event_idx is not None and bars_since is not None:
        failed = bars_since <= FOLLOW_BARS and float(x.Close) < float(current_res) * (1.0 - RES_ZONE_PCT / 100.0)
        retest = bars_since > 0 and float(x.Low) <= float(current_res) * (1.0 + RES_ZONE_PCT / 100.0) and float(x.Close) > float(current_res)

    if failed:
        breakout_state, breakout_score = "FAILED BREAKOUT", -20
    elif retest:
        breakout_state, breakout_score = "CONFIRMED / RETEST", 20
    elif bool(strong_candle.iloc[-1] and volume_confirm.iloc[-1] and acceptance.iloc[-1]):
        breakout_state, breakout_score = "STRONG BREAKOUT", 18
    elif bool(breakout_series.iloc[-1]):
        breakout_state, breakout_score = "EARLY BREAKOUT", 12
    elif bool(attempt_series.iloc[-1]):
        breakout_state, breakout_score = "BREAKOUT TEST", 5
    else:
        breakout_state, breakout_score = "PENDING", 3

    return {
        "breakout_resistance": None if pd.isna(current_res) else float(current_res),
        "breakout": bool(breakout_series.iloc[-1]),
        "breakout_attempt": bool(attempt_series.iloc[-1]),
        "breakout_event_idx": event_idx,
        "breakout_date": None if event_idx is None else str(d.index[event_idx].date()),
        "bars_since_breakout": bars_since,
        "successful_retest": bool(retest),
        "failed_breakout": bool(failed),
        "breakout_state": breakout_state,
        "breakout_score": breakout_score,
        "breakout_volume_confirm": bool(volume_confirm.iloc[-1]),
        "breakout_acceptance": bool(acceptance.iloc[-1]),
        "breakout_strong_candle": bool(strong_candle.iloc[-1]),
    }


def _phase(d, structure, bx):
    x = d.iloc[-1]
    atrv = max(float(x.ATR), float(x.Close) * 0.005)
    resistance = max(structure["recent_high"], float(x.HIGH50))
    dist_to_res = (resistance - float(x.Close)) / float(x.Close)
    near_res = -0.015 <= dist_to_res <= 0.06
    compression = len(d) >= 20 and float(d.RANGE20.iloc[-1]) < float(d.RANGE20.iloc[-20:].median()) * 0.85
    higher_lows = structure["trend_up"]
    extension = float(x.Close) > float(x.EMA20) + 2.0 * atrv
    pullback = float(x.Close) > float(x.EMA50) and float(x.Low) <= float(x.EMA20) and float(x.Close) > float(x.Open)

    # Master breakout state takes precedence over the generic 20-bar BOS test.
    if bx["breakout_state"] in ("STRONG BREAKOUT", "EARLY BREAKOUT"):
        return "BREAKOUT"
    if bx["breakout_state"] == "CONFIRMED / RETEST":
        return "PULLBACK"
    if bx["breakout_state"] == "FAILED BREAKOUT":
        return "NEUTRAL"
    if pullback:
        return "PULLBACK"
    if near_res and (higher_lows or compression):
        return "BUILDING"
    if extension:
        return "EXTENDED"
    return "NEUTRAL"


def classify_status(result):
    """Classify stock quality + entry readiness using the V2 ranking signals."""
    score = int(result.get("score", 0))
    priority = int(result.get("priority_score", score))
    phase = result.get("phase", "NEUTRAL")
    rs_delta = float(result.get("rs_delta", 0.0))
    cmp = float(result.get("cmp", 0.0))
    ema200 = float(result.get("ema200", 0.0))
    entry_quality = int(result.get("entry_quality", 0))
    rr = float(result.get("rr", 0.0))
    profit_risk = result.get("profit_booking_risk", "LOW")
    structure = result.get("structure", {}) or {}
    market_regime = result.get("market_regime", "UNKNOWN")

    above_ema200 = cmp > ema200 if ema200 > 0 else False
    trend_up = bool(structure.get("trend_up"))
    bos_up = bool(structure.get("bos_up"))
    actionable_phase = phase in ("BREAKOUT", "CONTINUATION", "PULLBACK")
    building_phase = phase == "BUILDING"

    # ENTRY READY remains strict. Bearish market does not reject a stock,
    # but it prevents the highest-conviction label while market risk is HIGH.
    if (
        market_regime != "BEARISH"
        and priority >= 78
        and entry_quality >= 72
        and rr >= MIN_RR
        and actionable_phase
        and (bos_up or trend_up)
        and profit_risk != "HIGH"
    ):
        return "ENTRY READY"

    # EXTENDED / high profit-booking setups stay visible but are not entries.
    if phase == "EXTENDED" or profit_risk == "HIGH":
        return "WATCH" if priority >= 55 else "NEUTRAL"

    # Strong actionable/building setups.
    if (priority >= 65 and (actionable_phase or building_phase)) or (
        priority >= 55
        and (building_phase or phase in ("PULLBACK", "CONTINUATION"))
        and rs_delta > 0
    ):
        return "WATCH"

    # Developing stocks.
    if priority >= 45 and (
        actionable_phase
        or building_phase
        or (above_ema200 and rs_delta > 0)
        or trend_up
    ):
        return "DEVELOPING"

    if priority >= 35 or (above_ema200 and rs_delta > 0):
        return "NEUTRAL"

    return "REJECT"



def valuation_risk_from_fundamentals(fundamental):
    """Classify valuation risk without making valuation a hard reject.

    Uses P/E and PEG when available. Growth remains visible separately;
    valuation risk only modifies ranking priority.
    """
    f = fundamental or {}
    pe = f.get("pe")
    peg = f.get("peg")

    try:
        pe = float(pe) if pe is not None else None
    except (TypeError, ValueError):
        pe = None
    try:
        peg = float(peg) if peg is not None else None
    except (TypeError, ValueError):
        peg = None

    if pe is None and peg is None:
        return "UNKNOWN"

    # Very high absolute valuation or very high growth-adjusted valuation.
    if (pe is not None and pe >= 80) or (peg is not None and peg >= 3.5):
        return "HIGH"

    # Expensive, but not extreme.
    if (pe is not None and pe >= 40) or (peg is not None and peg >= 2.0):
        return "MODERATE"

    return "LOW"


def ranking_engine_v2(tech_score, fundamental_score, entry_quality, rs_delta,
                      phase, rr, rsi, volume_ratio, market_regime,
                      continuation, profit_booking_risk, valuation_risk="UNKNOWN"):
    """Return separate stock-quality, entry-quality and final priority scores.

    The purpose is to avoid treating a strong stock and a good immediate entry
    as the same thing. Scores are bounded to 0-100.
    """
    tech = float(np.clip(tech_score, 0, 100))
    fund = float(np.clip(fundamental_score, 0, 100))
    entry = float(np.clip(entry_quality, 0, 100))

    # RS is supplied in percentage points (e.g. +10 means +10% relative edge).
    rs = float(np.clip(50 + rs_delta * 2.5, 0, 100))

    setup_base = {
        "BREAKOUT": 90,
        "CONTINUATION": 84,
        "PULLBACK": 82,
        "BUILDING": 76,
        "NEUTRAL": 50,
        "EXTENDED": 38,
    }.get(phase, 50)

    # Fresh volume expansion improves breakout/continuation quality.
    if phase in ("BREAKOUT", "CONTINUATION"):
        if volume_ratio >= 2.0:
            setup_base += 6
        elif volume_ratio >= 1.2:
            setup_base += 3

    # RSI is treated as timing, not as a simple bullish/bearish signal.
    entry_timing = entry
    if rsi > 80:
        entry_timing -= 25
    elif rsi > 75:
        entry_timing -= 14
    elif rsi > 72:
        entry_timing -= 7
    elif 52 <= rsi <= 68:
        entry_timing += 4
    elif rsi < 40:
        entry_timing -= 8

    if rr < MIN_RR:
        entry_timing -= 12
    elif rr >= 3:
        entry_timing += 5
    elif rr >= 2:
        entry_timing += 3

    if profit_booking_risk == "HIGH":
        entry_timing -= 22
    elif profit_booking_risk == "MODERATE":
        entry_timing -= 8

    entry_timing = float(np.clip(entry_timing, 0, 100))

    # Valuation is a risk modifier, not a hard rejection.
    # Strong growth/setup can still rank well, but expensive stocks are flagged.
    valuation_modifier = {
        "HIGH": -10,
        "MODERATE": -5,
        "LOW": 0,
        "UNKNOWN": 0,
    }.get(valuation_risk, 0)

    # Market is a risk modifier, not a hard rejection.
    market_modifier = {
        "BULLISH": 5,
        "BULLISH_BUT_MIXED": 2,
        "MIXED": 0,
        "UNKNOWN": 0,
        "BEARISH": -5,
    }.get(market_regime, 0)

    stock_quality = (
        tech * 0.40
        + fund * 0.20
        + rs * 0.20
        + setup_base * 0.20
    )
    priority = (
        stock_quality * 0.45
        + entry_timing * 0.35
        + setup_base * 0.10
        + rs * 0.10
        + market_modifier
        + valuation_modifier
    )

    stock_quality = int(round(np.clip(stock_quality, 0, 100)))
    entry_timing = int(round(np.clip(entry_timing, 0, 100)))
    priority = int(round(np.clip(priority, 0, 100)))

    return {
        "stock_quality_score": stock_quality,
        "entry_quality_v2": entry_timing,
        "priority_score": priority,
        "valuation_risk": valuation_risk,
        "valuation_modifier": valuation_modifier,
    }


def setup_analysis(ticker, df, bench, market, fundamental=None):
    d = enrich(df)
    if len(d) < 210:
        return None
    x = d.iloc[-1]
    structure = detect_structure(d)
    smc = detect_smc(d)
    bx = _breakout_engine(d)
    sr, br, rs_delta = relative_strength(d, bench)

    # Technical score: ranking, not a pass/fail checklist.
    tech = 0
    reasons: List[str] = []
    risks: List[str] = []

    if market["regime"] == "BULLISH":
        tech += 5
    elif market["regime"] in ("BULLISH_BUT_MIXED", "MIXED"):
        tech += 3
    else:
        risks.append("Market risk elevated")

    if x.Close > x.EMA20 > x.EMA50 > x.EMA200:
        tech += 15; reasons.append("Strong EMA trend alignment")
    elif x.Close > x.EMA50 > x.EMA200:
        tech += 11; reasons.append("Primary trend bullish")
    elif x.Close > x.EMA200:
        tech += 6
    else:
        risks.append("Price below long-term trend")

    if rs_delta > 0.08:
        tech += 15; reasons.append("Relative strength leading NIFTY")
    elif rs_delta > 0.03:
        tech += 11; reasons.append("Relative strength positive")
    elif rs_delta > 0:
        tech += 6
    else:
        risks.append("Relative strength not leading")

    if x.VOL_RATIO >= 1.8:
        tech += 10; reasons.append(f"Volume expansion {x.VOL_RATIO:.1f}x")
    elif x.VOL_RATIO >= 1.2:
        tech += 6; reasons.append(f"Volume supportive {x.VOL_RATIO:.1f}x")

    if 52 <= x.RSI <= 72:
        tech += 7
    elif 72 < x.RSI <= 78:
        tech += 4; risks.append("Momentum getting stretched")
    elif x.RSI > 78:
        risks.append("Momentum stretched")

    if structure["bos_up"]:
        tech += 14; reasons.append("Bullish breakout/BOS")
    elif structure["trend_up"]:
        tech += 8; reasons.append("Higher-high/higher-low structure")
    elif structure["trend_down"]:
        tech -= 8; risks.append("Lower-high/lower-low structure")

    if smc["sweep_low"]: tech += 4; reasons.append("Liquidity sweep/reclaim")
    if smc["ob_bull"]: tech += 3; reasons.append("Bullish order-block proxy")
    if smc["fvg_bull"]: tech += 2; reasons.append("Bullish FVG proxy")
    if smc["displacement"]: tech += 4; reasons.append("Price displacement")

    phase = _phase(d, structure, bx)
    if phase == "BUILDING":
        tech += 5; reasons.append("Constructive building/compression")
    elif phase == "BREAKOUT":
        reasons.append("Fresh breakout phase")
    elif phase == "CONTINUATION":
        tech += 5; reasons.append("Healthy post-breakout continuation")
    elif phase == "PULLBACK":
        tech += 4; reasons.append("Pullback in primary trend")
    elif phase == "EXTENDED":
        risks.append("Price extended from short-term mean")

    # Master breakout quality is explicit evidence, not a generic BOS proxy.
    tech += int(bx["breakout_score"] if bx["breakout_score"] > 0 else 0)
    if bx["breakout_volume_confirm"]:
        reasons.append("Breakout volume confirmed")
    if bx["breakout_acceptance"]:
        reasons.append("Breakout acceptance above resistance")
    if bx["failed_breakout"]:
        risks.append("Master breakout failure condition")

    # Master Pine Entry / Risk engine.
    resistance = max(structure["recent_high"], float(x.HIGH50))
    support = min(structure["recent_low"], float(x.LOW50))
    atrv = max(float(x.ATR), float(x.Close) * 0.005)
    trigger_price = bx["breakout_resistance"] * (1.0 + BREAKOUT_BUFFER_PCT / 100.0) if bx["breakout_resistance"] is not None else np.nan
    last_pl = structure.get("last_pl")
    invalidation = float(last_pl) if last_pl is not None else float(x.Low)
    extension_atr = abs(float(x.Close) - trigger_price) / atrv if pd.notna(trigger_price) and atrv > 0 else np.nan
    trigger_active = pd.notna(trigger_price) and float(x.Close) >= float(trigger_price)
    fresh_entry = trigger_active and pd.notna(extension_atr) and extension_atr <= FRESH_ATR
    healthy_entry = trigger_active and pd.notna(extension_atr) and FRESH_ATR < extension_atr <= HEALTHY_ATR
    wait_pullback = trigger_active and pd.notna(extension_atr) and HEALTHY_ATR < extension_atr < CHASE_ATR
    do_not_chase = trigger_active and pd.notna(extension_atr) and extension_atr >= CHASE_ATR

    if bx["failed_breakout"] or structure["bos_down"]:
        entry_state, entry_score = "ENTRY INVALID", -20
    elif do_not_chase:
        entry_state, entry_score = "EXTENDED / NO CHASE", -15
    elif bx["successful_retest"] and pd.notna(extension_atr) and extension_atr <= HEALTHY_ATR:
        entry_state, entry_score = "RETEST ENTRY", 18
    elif fresh_entry:
        entry_state, entry_score = "FRESH ENTRY", 20
    elif healthy_entry:
        entry_state, entry_score = "HEALTHY ENTRY", 15
    elif wait_pullback:
        entry_state, entry_score = "WAIT PULLBACK", 3
    else:
        entry_state, entry_score = "WAIT", 5

    entry_for_rr = trigger_price if pd.notna(trigger_price) else float(x.Close)
    risk_per_share = max(entry_for_rr - invalidation, 0.01)
    risk_pct = risk_per_share / entry_for_rr * 100.0 if entry_for_rr > 0 else 0.0
    prev_ph = structure.get("prev_ph")
    target = max(float(prev_ph), entry_for_rr + atrv * 3.0) if prev_ph is not None and float(prev_ph) > entry_for_rr else entry_for_rr + atrv * 3.0
    reward_per_share = max(target - entry_for_rr, 0.0)
    rr = reward_per_share / risk_per_share if risk_per_share > 0 else 0.0
    risk_ok = rr >= MIN_RR

    reentry_low = bx["breakout_resistance"] * (1.0 - RES_ZONE_PCT / 100.0) if bx["breakout_resistance"] is not None else np.nan
    reentry_high = bx["breakout_resistance"] * (1.0 + RES_ZONE_PCT / 100.0) if bx["breakout_resistance"] is not None else np.nan

    # Keep scanner-friendly levels while making them derive from Master trigger/invalidation.
    entry_low = reentry_low if bx["successful_retest"] else entry_for_rr
    entry_high = reentry_high if bx["successful_retest"] else entry_for_rr
    sl = invalidation
    target1 = target
    target2 = entry_for_rr + atrv * 4.0
    room = max(resistance - entry_high, 0.0) if pd.notna(entry_high) else 0.0

    # Fundamental contribution is intentionally modest: technical timing stays primary.
    fund_score = int((fundamental or {}).get("fundamental_score", 50))
    fundamental_available = bool((fundamental or {}).get("available", False))
    valuation_risk = valuation_risk_from_fundamentals(fundamental) if fundamental_available else "UNKNOWN"
    total = round(tech * 0.72 + fund_score * 0.28) if fundamental_available else min(100, tech)
    total = max(0, min(100, int(total)))

    # Entry quality follows the Master entry state, not the stock's underlying quality.
    entry_quality = {20: 100, 18: 90, 15: 80, 5: 45, 3: 25, -15: 10, -20: 0}.get(entry_score, 25)
    if not risk_ok:
        entry_quality -= 20
    if market["regime"] == "BEARISH":
        entry_quality -= 3
    entry_quality = max(0, min(100, int(entry_quality)))

    continuation = (
        "STRONG" if total >= 82 and phase in ("BREAKOUT", "CONTINUATION")
        else "MODERATE" if total >= 68 else "LOW"
    )
    profit_risk = (
        "HIGH" if phase == "EXTENDED" or x.RSI > 80
        else "MODERATE" if x.RSI > 72 or x.Close > x.EMA20 + atrv
        else "LOW"
    )

    ranking = ranking_engine_v2(
        tech_score=tech,
        fundamental_score=fund_score,
        entry_quality=entry_quality,
        rs_delta=float(rs_delta * 100),
        phase=phase,
        rr=rr,
        rsi=float(x.RSI),
        volume_ratio=float(x.VOL_RATIO),
        market_regime=market["regime"],
        continuation=continuation,
        profit_booking_risk=profit_risk,
        valuation_risk=valuation_risk,
    )

    # Keep the legacy score for backward compatibility, but use priority_score
    # for the new status classification and dashboard ranking.
    provisional = {
        "score": total,
        "priority_score": ranking["priority_score"],
        "phase": phase,
        "cmp": float(x.Close),
        "ema200": float(x.EMA200),
        "rs_delta": float(rs_delta * 100),
        "entry_quality": ranking["entry_quality_v2"],
        "rr": rr,
        "profit_booking_risk": profit_risk,
        "market_regime": market["regime"],
        "structure": structure,
        "breakout_resistance": None if pd.isna(bx["breakout_resistance"]) else round(bx["breakout_resistance"], 2),
        "breakout_trigger": None if pd.isna(trigger_price) else round(trigger_price, 2),
        "breakout": bx["breakout"],
        "breakout_attempt": bx["breakout_attempt"],
        "breakout_date": bx["breakout_date"],
        "breakout_age": bx["bars_since_breakout"],
        "breakout_state": bx["breakout_state"],
        "breakout_score": bx["breakout_score"],
        "breakout_volume_confirm": bx["breakout_volume_confirm"],
        "breakout_acceptance": bx["breakout_acceptance"],
        "successful_retest": bx["successful_retest"],
        "failed_breakout": bx["failed_breakout"],
        "entry_state": entry_state,
        "entry_trigger": None if pd.isna(trigger_price) else round(trigger_price, 2),
        "invalidation": round(invalidation, 2),
        "risk_pct": round(risk_pct, 2),
        "extension_atr": None if pd.isna(extension_atr) else round(extension_atr, 2),
        "reentry_zone_low": None if pd.isna(reentry_low) else round(reentry_low, 2),
        "reentry_zone_high": None if pd.isna(reentry_high) else round(reentry_high, 2),
        "risk_ok": bool(risk_ok),
        "technical_entry_candidate": bool(entry_state in ("FRESH ENTRY", "HEALTHY ENTRY", "RETEST ENTRY")),
        "market_gate_blocked": bool(market["regime"] == "BEARISH" and entry_state in ("FRESH ENTRY", "HEALTHY ENTRY", "RETEST ENTRY")),
    }
    status = classify_status(provisional)

    return {
        "ticker": ticker,
        "date": str(d.index[-1].date()),
        "cmp": round(float(x.Close), 2),
        "status": status,
        "score": total,
        "priority_score": ranking["priority_score"],
        "stock_quality_score": ranking["stock_quality_score"],
        "technical_score": int(max(0, min(100, tech))),
        "fundamental_score": fund_score,
        "valuation_risk": ranking["valuation_risk"],
        "valuation_modifier": ranking["valuation_modifier"],
        "entry_quality": ranking["entry_quality_v2"],
        "phase": phase,
        "setup": phase,
        "market_regime": market["regime"],
        "market_risk": market.get("risk", "UNKNOWN"),
        "rs_stock": round(sr * 100, 2),
        "rs_bench": round(br * 100, 2),
        "rs_delta": round(rs_delta * 100, 2),
        "rsi": round(float(x.RSI), 1),
        "volume_ratio": round(float(x.VOL_RATIO), 2),
        "ema20": round(float(x.EMA20), 2),
        "ema50": round(float(x.EMA50), 2),
        "ema200": round(float(x.EMA200), 2),
        "resistance": round(resistance, 2),
        "support": round(support, 2),
        "entry_low": round(entry_low, 2),
        "entry_high": round(entry_high, 2),
        "stop_loss": round(sl, 2),
        "target1": round(target1, 2),
        "target2": round(target2, 2),
        "rr": round(rr, 1),
        "continuation": continuation,
        "profit_booking_risk": profit_risk,
        "reasons": reasons,
        "risks": risks,
        "fundamental_available": fundamental_available,
        "smc": smc,
        "structure": structure,
        "breakout_resistance": None if pd.isna(bx["breakout_resistance"]) else round(bx["breakout_resistance"], 2),
        "breakout_trigger": None if pd.isna(trigger_price) else round(trigger_price, 2),
        "breakout": bx["breakout"],
        "breakout_attempt": bx["breakout_attempt"],
        "breakout_date": bx["breakout_date"],
        "breakout_age": bx["bars_since_breakout"],
        "breakout_state": bx["breakout_state"],
        "breakout_score": bx["breakout_score"],
        "breakout_volume_confirm": bx["breakout_volume_confirm"],
        "breakout_acceptance": bx["breakout_acceptance"],
        "breakout_strong_candle": bx["breakout_strong_candle"],
        "successful_retest": bx["successful_retest"],
        "failed_breakout": bx["failed_breakout"],
        "entry_state": entry_state,
        "entry_trigger": None if pd.isna(trigger_price) else round(trigger_price, 2),
        "invalidation": round(invalidation, 2),
        "risk_pct": round(risk_pct, 2),
        "extension_atr": None if pd.isna(extension_atr) else round(extension_atr, 2),
        "reentry_zone_low": None if pd.isna(reentry_low) else round(reentry_low, 2),
        "reentry_zone_high": None if pd.isna(reentry_high) else round(reentry_high, 2),
        "risk_ok": bool(risk_ok),
        "technical_entry_candidate": bool(entry_state in ("FRESH ENTRY", "HEALTHY ENTRY", "RETEST ENTRY")),
        "market_gate_blocked": bool(market["regime"] == "BEARISH" and entry_state in ("FRESH ENTRY", "HEALTHY ENTRY", "RETEST ENTRY")),
    }
