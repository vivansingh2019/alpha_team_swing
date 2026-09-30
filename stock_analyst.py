"""Swing Brain Stock Analyst module.

Uses the existing Swing Brain brain.py logic as the calculation source and adds
an analyst-oriented presentation layer: breakout lifecycle, entry gate, risk,
RS/volume context, fundamentals and chart levels.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from brain import enrich, market_regime, setup_analysis
from config import BENCHMARK, MIN_RR
from data import download_history
from fundamental import get_fundamentals


@dataclass
class AnalystResult:
    result: Dict[str, Any]
    history: pd.DataFrame
    market: Dict[str, Any]
    fundamentals: Dict[str, Any]
    lifecycle: Dict[str, Any]


def _safe_float(value, default=0.0):
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _round(value, digits=2):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
        return round(float(value), digits)
    except (TypeError, ValueError):
        return value


def _breakout_lifecycle(d: pd.DataFrame) -> Dict[str, Any]:
    """Trace the latest 20-day breakout using the same prior-bar resistance idea.

    This is a presentation/diagnostic layer. The stock's master score/status still
    comes directly from brain.setup_analysis().
    """
    if len(d) < 30:
        return {
            "state": "NO DATA",
            "age_days": None,
            "breakout_date": None,
            "level": None,
            "retest": False,
            "failed": False,
            "sustain": False,
            "note": "Insufficient history",
        }

    work = d.copy()
    work["BO_LEVEL"] = work["HIGH20"]
    work["BO"] = work["Close"] > work["BO_LEVEL"]

    bo_rows = work.index[work["BO"].fillna(False)]
    if len(bo_rows) == 0:
        x = work.iloc[-1]
        level = _safe_float(x.get("HIGH20"), _safe_float(x.get("HIGH50"), _safe_float(x.Close)))
        distance = (_safe_float(x.Close) - level) / level * 100 if level else 0
        state = "BREAKOUT TEST" if _safe_float(x.High) > level else "BUILDING"
        return {
            "state": state,
            "age_days": None,
            "breakout_date": None,
            "level": _round(level),
            "retest": False,
            "failed": False,
            "sustain": False,
            "distance_pct": _round(distance, 2),
            "note": "No confirmed close above the prior 20-day resistance in the available window.",
        }

    bo_idx = bo_rows[-1]
    pos = work.index.get_loc(bo_idx)
    level = _safe_float(work.loc[bo_idx, "BO_LEVEL"])
    if not level:
        level = _safe_float(work.loc[bo_idx, "Close"])

    after = work.iloc[pos:]
    retest = False
    failed = False
    for _, row in after.iloc[1:].iterrows():
        close = _safe_float(row.Close)
        low = _safe_float(row.Low)
        if low <= level * 1.01 and close >= level:
            retest = True
        if close < level * 0.99:
            failed = True

    last = work.iloc[-1]
    current_close = _safe_float(last.Close)
    age = max(0, len(work) - 1 - pos)
    sustain = current_close >= level * 0.99 and not failed

    if failed:
        state = "FAILED"
    elif retest:
        state = "RETEST / CONFIRMED"
    elif age == 0:
        state = "NEW BREAKOUT"
    elif sustain:
        state = "SUSTAINING"
    else:
        state = "BREAKOUT TEST"

    return {
        "state": state,
        "age_days": int(age),
        "breakout_date": str(pd.Timestamp(bo_idx).date()),
        "level": _round(level),
        "retest": bool(retest),
        "failed": bool(failed),
        "sustain": bool(sustain),
        "distance_pct": _round((current_close - level) / level * 100 if level else 0, 2),
        "note": "Latest breakout lifecycle derived from prior-bar 20-day resistance.",
    }



def _pivot_state(d: pd.DataFrame, pivot_len: int = 3) -> Dict[str, Any]:
    """Completed-pivot swing structure, following the Master indicator's logic."""
    if len(d) < pivot_len * 2 + 10:
        return {"state": "MIXED", "score": 8, "higher_high": False, "higher_low": False,
                "lower_high": False, "lower_low": False, "current_break": False}
    h = d["High"].astype(float)
    l = d["Low"].astype(float)
    # centered pivots: a pivot is only known after pivot_len bars on the right.
    ph = h.eq(h.rolling(2 * pivot_len + 1, center=True).max())
    pl = l.eq(l.rolling(2 * pivot_len + 1, center=True).min())
    ph_vals = h[ph].dropna().iloc[:-pivot_len] if len(h) > pivot_len else h[ph].dropna()
    pl_vals = l[pl].dropna().iloc[:-pivot_len] if len(l) > pivot_len else l[pl].dropna()
    last_ph = float(ph_vals.iloc[-1]) if len(ph_vals) else None
    prev_ph = float(ph_vals.iloc[-2]) if len(ph_vals) >= 2 else None
    last_pl = float(pl_vals.iloc[-1]) if len(pl_vals) else None
    prev_pl = float(pl_vals.iloc[-2]) if len(pl_vals) >= 2 else None
    hh = last_ph is not None and prev_ph is not None and last_ph > prev_ph
    hl = last_pl is not None and prev_pl is not None and last_pl > prev_pl
    lh = last_ph is not None and prev_ph is not None and last_ph < prev_ph
    ll = last_pl is not None and prev_pl is not None and last_pl < prev_pl
    up = ((last_ph - last_pl) / last_pl * 100.0) if last_ph and last_pl else None
    prev_up = ((prev_ph - last_pl) / last_pl * 100.0) if prev_ph and last_pl else None
    expanding = up is not None and prev_up is not None and up > prev_up * 1.05
    decel = up is not None and prev_up is not None and up < prev_up * 0.95
    pb = ((last_ph - last_pl) / last_ph * 100.0) if last_ph else None
    prev_pb = ((prev_ph - prev_pl) / prev_ph * 100.0) if prev_ph and prev_pl else None
    pb_compress = pb is not None and prev_pb is not None and pb < prev_pb * 0.95
    pb_expand = pb is not None and prev_pb is not None and pb > prev_pb * 1.05
    if ll or (lh and not hl):
        state, score = "BREAKING", -15
    elif hh and hl and expanding and pb_compress:
        state, score = "ACCELERATING", 20
    elif hh and hl:
        state, score = "BUILDING", 15
    elif decel or pb_expand:
        state, score = "DECELERATING", 2
    else:
        state, score = "MIXED", 8
    current_break = last_pl is not None and float(d.Close.iloc[-1]) < last_pl
    return {"state": state, "score": score, "higher_high": hh, "higher_low": hl,
            "lower_high": lh, "lower_low": ll, "current_break": current_break,
            "last_ph": last_ph, "last_pl": last_pl}


def _phase_c_evidence(d: pd.DataFrame, benchmark: Optional[pd.DataFrame], result: Dict[str, Any], lifecycle: Dict[str, Any]) -> Dict[str, Any]:
    """Phase C evidence layer. Mirrors the Master indicator's evidence/context concepts.

    This layer does not change the existing Swing Brain score or entry decision.
    It only exposes the evidence that explains the state.
    """
    x = d.iloc[-1]
    prev = d.iloc[-2] if len(d) >= 2 else x
    rvol = _safe_float(x.get("VOL_RATIO"), 0.0)
    price_change = (_safe_float(x.Close) / max(_safe_float(prev.Close), 1e-9) - 1.0) * 100.0
    rng = max(_safe_float(x.High) - _safe_float(x.Low), 1e-9)
    close_loc = (_safe_float(x.Close) - _safe_float(x.Low)) / rng
    body_pct = abs(_safe_float(x.Close) - _safe_float(x.Open)) / max(rng, 1e-9) * 100.0
    abs_move = d.Close.pct_change().abs() * 100.0
    avg_abs_move = _safe_float(abs_move.rolling(10).mean().iloc[-1], 0.0)
    response_ratio = abs(price_change) / avg_abs_move if avg_abs_move > 0 else None
    price_response = "STRONG" if response_ratio is not None and response_ratio >= 1.25 else "NORMAL" if abs(price_change) >= 0.75 else "WEAK"
    strong_close = close_loc >= 0.70
    weak_close = close_loc <= 0.40
    large_body = body_pct >= 55.0
    if rvol >= 3.0 and price_change > 0 and strong_close and large_body:
        volume_state, volume_score = "DEMAND EXPANSION", 20
    elif rvol >= 1.5 and price_change > 0 and strong_close:
        volume_state, volume_score = "DEMAND", 18
    elif rvol <= 0.85 and price_change > 0 and strong_close and (response_ratio or 0) >= 1.0:
        volume_state, volume_score = "LOW-SUPPLY ADVANCE", 15
    elif rvol <= 0.85 and price_change > 0 and strong_close:
        volume_state, volume_score = "LOW-VOLUME DRIFT", 8
    elif rvol >= 1.5 and abs(price_change) < 0.75:
        volume_state, volume_score = "ABSORPTION", 2
    elif rvol >= 1.5 and price_change > 0 and weak_close:
        volume_state, volume_score = "SUPPLY WARNING", -8
    elif rvol >= 1.5 and price_change < 0 and weak_close:
        volume_state, volume_score = "DISTRIBUTION", -15
    elif rvol < 1.0 and price_change < 0:
        volume_state, volume_score = "LOW-PARTICIPATION DROP", -3
    elif rvol >= 3.0 and abs(price_change) > 2.0 and weak_close:
        volume_state, volume_score = "EXHAUSTION", -12
    else:
        volume_state, volume_score = "NEUTRAL", 5
    pv_score = int(np.clip(volume_score + (3 if price_response == "STRONG" else 1 if price_response == "NORMAL" else 0), -15, 20))

    resistance = _safe_float(d.High.iloc[-61:-1].max(), _safe_float(x.Close)) if len(d) >= 62 else _safe_float(d.High.iloc[:-1].max(), _safe_float(x.Close))
    zone_bottom = resistance * 0.99
    entered = (d.High >= zone_bottom) & (d.Low <= resistance) & (d.Close < resistance)
    # cooldown 5 bars: count a test only when separated from the prior accepted test.
    tests = 0; last_i = -999
    flags = entered.fillna(False).to_numpy()
    for i, flag in enumerate(flags):
        if flag and i - last_i >= 5:
            tests += 1; last_i = i
    tests = min(tests, 10)
    distance_res = (resistance - _safe_float(x.Close)) / resistance * 100.0 if resistance else 0.0
    piv = _pivot_state(d)
    proximity = 5 if distance_res <= 1 else 4 if distance_res <= 3 else 2 if distance_res <= 5 else 0
    test_score = 5 if tests >= 4 else 4 if tests == 3 else 3 if tests == 2 else 2 if tests == 1 else 0
    hl_score = 4 if piv["higher_low"] else 0
    # pivot-derived pullback compression is represented by the pivot state.
    compression_score = 4 if piv["state"] in ("BUILDING", "ACCELERATING") else 0
    pressure_score = int(np.clip(proximity + test_score + hl_score + compression_score, 0, 20))
    pressure_state = "HIGH PRESSURE" if pressure_score >= 16 else "PRESSURE" if pressure_score >= 11 else "BUILDING" if pressure_score >= 7 else "FAR / BUILDING" if distance_res > 8 else "APPROACHING"

    # Completed daily/weekly context (Master uses previous completed bars).
    dd = d.iloc[:-1].copy() if len(d) > 1 else d.copy()
    daily_close = _safe_float(dd.Close.iloc[-1]) if len(dd) else 0.0
    daily_ema20 = _safe_float(dd.Close.ewm(span=20, adjust=False).mean().iloc[-1]) if len(dd) else 0.0
    daily_ema50 = _safe_float(dd.Close.ewm(span=50, adjust=False).mean().iloc[-1]) if len(dd) else 0.0
    daily_trend = "BULLISH" if daily_close > daily_ema20 > daily_ema50 else "BEARISH" if daily_close < daily_ema20 < daily_ema50 else "NEUTRAL"
    weekly = dd.resample("W-FRI").agg({"Close":"last", "Volume":"sum", "High":"max", "Low":"min"}).dropna()
    if len(weekly) >= 21:
        wc = weekly.Close.ewm(span=20, adjust=False).mean(); wf = weekly.Close.ewm(span=5, adjust=False).mean()
        weekly_trend = "BULLISH" if weekly.Close.iloc[-1] > wf.iloc[-1] > wc.iloc[-1] else "BEARISH" if weekly.Close.iloc[-1] < wf.iloc[-1] < wc.iloc[-1] else "NEUTRAL"
    else:
        weekly_trend = "INSUFFICIENT DATA"
    atr_pct = (d.ATR / d.Close * 100.0).replace([np.inf, -np.inf], np.nan)
    atr_prev = _safe_float(atr_pct.iloc[-2], 0.0) if len(atr_pct) >= 2 else 0.0
    atr_base = _safe_float(atr_pct.iloc[:-1].rolling(60).mean().iloc[-1], 0.0) if len(atr_pct) > 60 else atr_prev
    compression = "STRONG" if atr_base and atr_prev <= atr_base * 0.85 else "NORMAL" if atr_base and atr_prev <= atr_base else "LOOSE"

    # Completed-bar RS versus NIFTY.
    rs_spread = _safe_float(result.get("rs_delta"), 0.0)
    if benchmark is not None and len(benchmark) > 64 and len(dd) > 64:
        bn = enrich(benchmark)
        n = min(63, len(dd)-1, len(bn)-1)
        sr = dd.Close.iloc[-1] / dd.Close.iloc[-n-1] - 1.0
        br = bn.Close.iloc[-1] / bn.Close.iloc[-n-1] - 1.0
        rs_spread = (sr - br) * 100.0
    rs_state = "STRONGER" if rs_spread >= 5 else "SUPPORTIVE" if rs_spread >= 0 else "NEUTRAL" if rs_spread >= -5 else "WEAKER"
    weekly_pts = 30 if weekly_trend == "BULLISH" else 15 if weekly_trend == "NEUTRAL" else 0 if weekly_trend == "BEARISH" else 10
    daily_pts = 20 if daily_trend == "BULLISH" else 10 if daily_trend == "NEUTRAL" else 0 if daily_trend == "BEARISH" else 10
    comp_pts = 15 if compression == "STRONG" else 10 if compression == "NORMAL" else 4
    rs_pts = 20 if rs_state == "STRONGER" else 15 if rs_state == "SUPPORTIVE" else 10 if rs_state == "NEUTRAL" else 2
    daily_res = _safe_float(dd.High.iloc[-61:-1].max(), daily_close) if len(dd) >= 62 else _safe_float(dd.High.iloc[:-1].max(), daily_close)
    daily_dist = (daily_res - daily_close) / daily_res * 100.0 if daily_res else 0.0
    room_pts = 15 if daily_dist >= 8 else 12 if daily_dist >= 5 else 9 if daily_dist >= 3 else 5 if daily_dist >= 1 else 2
    swing_context_score = int(np.clip(weekly_pts + daily_pts + comp_pts + rs_pts + room_pts, 0, 100))
    swing_context = "FAVORABLE" if swing_context_score >= 70 else "SUPPORTIVE" if swing_context_score >= 60 else "CAUTION" if swing_context_score >= 45 else "UNFAVOURABLE"
    context_ok = swing_context_score >= 60 and weekly_trend != "BEARISH"

    daily_res20 = _safe_float(dd.High.iloc[-21:-1].max(), daily_close) if len(dd) >= 22 else daily_close
    daily_bos_bull = daily_close > daily_res20
    daily_bos_bear = daily_close < (_safe_float(dd.Low.iloc[-21:-1].min(), daily_close) if len(dd) >= 22 else daily_close)
    daily_breakout = daily_close > daily_res * 1.0015
    near_res = 0 <= daily_dist <= 3
    daily_swing = "DAMAGED" if daily_trend == "BEARISH" and daily_bos_bear else "BREAKOUT" if daily_breakout else "PRE-BREAKOUT" if daily_trend == "BULLISH" and near_res else "BASE BUILDING" if daily_trend == "BULLISH" and compression == "STRONG" else "CONSTRUCTIVE" if daily_trend == "BULLISH" else "RECOVERY" if weekly_trend == "BULLISH" else "DAMAGED"
    vol5 = _safe_float(dd.Volume.tail(5).mean(), 0.0); vol20 = _safe_float(dd.Volume.tail(20).mean(), 0.0)
    participation_ratio = vol5 / vol20 if vol20 > 0 else None
    participation = "EXPANDING" if participation_ratio is not None and participation_ratio >= 1.15 else "CONSTRUCTIVE" if participation_ratio is not None and participation_ratio >= 0.90 else "QUIET" if participation_ratio is not None and participation_ratio >= 0.70 else "WEAK" if participation_ratio is not None else "NO DATA"

    # Breakout confirmation evidence from current bar.
    breakout_res = _safe_float(d.High.iloc[-61:-1].max(), _safe_float(x.Close)) if len(d) >= 62 else resistance
    strong_candle = _safe_float(x.Close) > _safe_float(x.Low) + 0.70 * rng and body_pct >= 55
    bo = _safe_float(x.Close) > breakout_res * 1.0015
    bo_volume = bo and rvol >= 1.2
    acceptance = bo and _safe_float(x.Low) >= breakout_res * 0.99
    confirmation_flags = {"breakout": bo, "strong_candle": strong_candle and bo, "volume_confirmed": bo_volume, "acceptance": acceptance,
                          "retest": bool(lifecycle.get("retest")), "smc_supportive": bool(result.get("smc", {}).get("displacement") or result.get("smc", {}).get("ob_bull") or result.get("smc", {}).get("fvg_bull"))}
    confirmation_score = sum(1 for v in confirmation_flags.values() if v)
    confirmation_state = "CONFIRMED" if confirmation_flags["retest"] or confirmation_score >= 4 else "STRONG" if confirmation_score >= 3 else "PENDING" if bo else "BUILDING"

    # Historical thesis memory: replay a lightweight completed-day state machine.
    thesis_raw = int(np.clip((max(-15, min(20, piv["score"])) + 15) / 35 * 45 + swing_context_score * .45 + (0 if result.get("market_regime") == "BEARISH" else 6), 0, 100))
    persistent = "WATCH"; setup_age = 0; weak_days = 0; quality_memory = thesis_raw; memory_peak = thesis_raw
    if len(dd) >= 120:
        # Use recent completed days to approximate the Master persistent state without inventing catalyst inputs.
        for i in range(max(20, len(dd)-120), len(dd)):
            sub = dd.iloc[:i+1]
            c = _safe_float(sub.Close.iloc[-1]); e20 = _safe_float(sub.Close.ewm(span=20, adjust=False).mean().iloc[-1]); e50 = _safe_float(sub.Close.ewm(span=50, adjust=False).mean().iloc[-1])
            bull = c > e20 > e50
            q = int(np.clip((25 if bull else 5) + (15 if rs_state in ("STRONGER","SUPPORTIVE") else 5) + (10 if compression != "LOOSE" else 4) + (20 if c >= _safe_float(sub.High.tail(21).iloc[:-1].max(), c) else 5), 0, 100))
            candidate = q >= 78 and weekly_trend != "BEARISH" and result.get("market_regime") != "BEARISH"
            if persistent == "WATCH" and candidate: persistent = "CANDIDATE"; setup_age = 0; weak_days = 0
            elif persistent == "CANDIDATE":
                setup_age += 1
                if c >= _safe_float(sub.High.tail(21).iloc[:-1].max(), c) and candidate: persistent = "CONFIRMED"; weak_days = 0
                elif q < 68: persistent = "WATCH"; setup_age = 0
            elif persistent == "CONFIRMED":
                setup_age += 1
                weak_days = weak_days + 1 if not bull else 0
                if weak_days >= 2 or (not bull and c < _safe_float(sub.Low.tail(21).iloc[:-1].min(), c)): persistent = "INVALIDATED"
                elif q < 68: persistent = "REVIEW"
            elif persistent == "REVIEW":
                setup_age += 1
                weak_days = weak_days + 1 if not bull else 0
                if weak_days >= 2: persistent = "INVALIDATED"
                elif candidate: persistent = "CONFIRMED"; weak_days = 0
            elif persistent == "INVALIDATED" and candidate and c >= _safe_float(sub.High.tail(21).iloc[:-1].max(), c):
                persistent = "CANDIDATE"; setup_age = 0; weak_days = 0
            memory_peak = max(memory_peak, q); quality_memory = q if persistent in ("WATCH","CANDIDATE") else max(q, quality_memory - 4)

    hold_context_review = not context_ok or result.get("market_regime") == "BEARISH"
    hold_invalid = bool(lifecycle.get("failed")) or piv["current_break"]
    hold_technical = "INVALIDATED" if hold_invalid else "INTACT"
    hold_thesis = "INVALIDATED" if hold_invalid else "REVIEW" if hold_context_review else "SUPPORTIVE"
    hold_action = "EXIT / INVALIDATION" if hold_invalid else "HOLD / REVIEW" if hold_context_review else "HOLD / THESIS INTACT"
    pace_state = "FAST" if avg_abs_move >= (_safe_float(x.ATR) / max(_safe_float(x.Close), 1e-9) * 100.0) * 1.10 else "SLOW" if avg_abs_move <= (_safe_float(x.ATR) / max(_safe_float(x.Close), 1e-9) * 100.0) * 0.65 else "NORMAL"
    return {
        "swing_state": piv["state"], "swing_score": piv["score"], "higher_high": piv["higher_high"], "higher_low": piv["higher_low"],
        "volume_state": volume_state, "pv_score": pv_score, "price_response_state": price_response, "pressure_state": pressure_state,
        "pressure_score": pressure_score, "resistance_tests": tests, "weekly_state": weekly_trend, "daily_state": daily_trend,
        "compression_state": compression, "rs_context_state": rs_state, "swing_context": swing_context, "swing_context_score": swing_context_score,
        "daily_swing_state": daily_swing, "participation_state": participation, "participation_ratio": round(participation_ratio,2) if participation_ratio is not None else None,
        "confirmation_state": confirmation_state, "confirmation_score": confirmation_score, "confirmation": confirmation_flags,
        "persistent_state": persistent, "setup_age_days": setup_age, "weak_days": weak_days, "quality_memory": int(round(quality_memory)), "memory_peak": int(round(memory_peak)),
        "hold_thesis_state": hold_thesis, "hold_price_state": hold_technical, "hold_action": hold_action,
        "evidence_health": {"structure": True, "price_volume": True, "swing_context": weekly_trend != "INSUFFICIENT DATA", "risk": bool(result.get("rr") is not None), "hold": True, "master": True},
        "evidence_score": int(round(np.clip(np.mean([piv["score"] + 15, pv_score + 15, swing_context_score, confirmation_score / 6 * 100, 100 if context_ok else 40]), 0, 100))),
        "phase_c_version": "C1",
    }

def _analyst_overlay(result: Dict[str, Any], d: pd.DataFrame, fundamentals: Dict[str, Any], lifecycle: Optional[Dict[str, Any]] = None, benchmark: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    x = d.iloc[-1]
    structure = result.get("structure", {}) or {}
    smc = result.get("smc", {}) or {}

    trend_score = 100 if _safe_float(x.Close) > _safe_float(x.EMA20) > _safe_float(x.EMA50) > _safe_float(x.EMA200) else 80 if _safe_float(x.Close) > _safe_float(x.EMA50) > _safe_float(x.EMA200) else 55 if _safe_float(x.Close) > _safe_float(x.EMA200) else 25
    structure_score = 90 if structure.get("bos_up") else 78 if structure.get("trend_up") else 60 if not structure.get("trend_down") else 35
    volume_score = min(100, 45 + _safe_float(x.VOL_RATIO) * 25)
    rs_score = min(100, max(0, 50 + _safe_float(result.get("rs_delta")) * 2.5))
    momentum_score = min(100, max(0, 100 - abs(_safe_float(x.RSI) - 62) * 2.2))
    smc_score = 50
    smc_score += 15 if smc.get("sweep_low") else 0
    smc_score += 12 if smc.get("ob_bull") else 0
    smc_score += 8 if smc.get("fvg_bull") else 0
    smc_score += 10 if smc.get("displacement") else 0
    smc_score = min(100, smc_score)

    structure_total = int(round(
        trend_score * 0.30 + structure_score * 0.25 + volume_score * 0.15 + rs_score * 0.15 + momentum_score * 0.10 + smc_score * 0.05
    ))

    rr = _safe_float(result.get("rr"))
    risk_gate = "PASS" if rr >= MIN_RR and result.get("profit_booking_risk") != "HIGH" else "FAIL"
    market_gate = "BLOCKED" if result.get("market_regime") == "BEARISH" else "OPEN"

    entry_quality = _safe_float(result.get("entry_quality"))
    lifecycle = lifecycle or {}
    evidence = _phase_c_evidence(d, benchmark, result, lifecycle)
    lifecycle_failed = bool(lifecycle.get("failed")) or lifecycle.get("state") == "FAILED"
    if lifecycle_failed:
        entry_state = "WAIT REBUILD"
    elif result.get("phase") in ("BREAKOUT", "CONTINUATION") and entry_quality >= 65:
        entry_state = "ACTIVE / WATCH"
    elif result.get("phase") == "PULLBACK":
        entry_state = "PULLBACK WATCH"
    elif result.get("phase") == "BUILDING":
        entry_state = "WAIT BREAKOUT"
    elif result.get("phase") == "EXTENDED":
        entry_state = "NO CHASE"
    else:
        entry_state = "WAIT"

    pace = "FAST" if evidence.get("pace_state") == "FAST" else "SLOW" if evidence.get("pace_state") == "SLOW" else "NORMAL"
    if _safe_float(x.RSI) > 75:
        pace = "FAST / STRETCHED"
    elif evidence.get("volume_state") in ("DEMAND EXPANSION", "DEMAND"):
        pace = "FAST / VOLUME LED"

    valuation = result.get("valuation_risk", "UNKNOWN")
    growth = "STRONG" if _safe_float(fundamentals.get("revenue_growth_pct")) >= 15 and _safe_float(fundamentals.get("earnings_growth_pct")) >= 15 else "MIXED" if fundamentals.get("available") else "UNKNOWN"

    blockers = []
    if rr < MIN_RR:
        blockers.append(f"R:R < {MIN_RR}")
    if risk_gate == "FAIL":
        blockers.append("Risk gate fail")
    if market_gate == "BLOCKED":
        blockers.append("Market gate blocked")
    if entry_quality < 60:
        blockers.append("Entry quality < 60")
    if lifecycle_failed:
        blockers.append("Breakout failed — wait for a fresh rebuild")
    elif result.get("phase") in ("BUILDING", "NEUTRAL"):
        blockers.append("Breakout confirmation pending")

    return {
        **evidence,
        "pace_state": evidence.get("pace_state", "NORMAL"),
        "trend": "BULLISH" if trend_score >= 75 else "MIXED" if trend_score >= 50 else "BEARISH",
        "structure_label": "GOOD" if structure_total >= 70 else "NEUTRAL" if structure_total >= 50 else "WEAK",
        "structure_score": structure_total,
        "price_volume": "STRONG" if _safe_float(x.VOL_RATIO) >= 1.5 else "NEUTRAL" if _safe_float(x.VOL_RATIO) >= 0.85 else "WEAK",
        "swing_context": "FAVORABLE" if result.get("rs_delta", 0) > 0 and trend_score >= 55 else "NEUTRAL",
        "distance_to_resistance_pct": _round((_safe_float(result.get("resistance")) - _safe_float(result.get("cmp"))) / max(_safe_float(result.get("cmp")), 1) * 100, 2),
        "pace": pace,
        "entry_state": entry_state,
        "risk_gate": risk_gate,
        "market_gate": market_gate,
        "timing_score": int(min(100, max(0, entry_quality))),
        "blockers": blockers,
        "rs_state": "LEADING" if _safe_float(result.get("rs_delta")) >= 3 else "POSITIVE" if _safe_float(result.get("rs_delta")) > 0 else "FADING/WEAK",
        "volume_state": "ACCUMULATION" if _safe_float(x.VOL_RATIO) >= 1.2 and _safe_float(x.Close) >= _safe_float(x.Open) else "NEUTRAL",
        "valuation": valuation,
        "growth": growth,
        "fundamental_data": "SUFFICIENT" if fundamentals.get("available") else "LIMITED",
        "smc_state": "SUPPORTIVE" if smc_score >= 65 else "NEUTRAL",
        "weekly_state": "BULLISH" if _safe_float(result.get("cmp")) > _safe_float(result.get("ema50")) else "MIXED",
        "thesis": "REBUILD REQUIRED" if lifecycle_failed or evidence.get("persistent_state") == "INVALIDATED" else "REVIEW" if evidence.get("persistent_state") == "REVIEW" else "CONFIRMED" if evidence.get("persistent_state") == "CONFIRMED" else "CANDIDATE" if evidence.get("persistent_state") == "CANDIDATE" else "WATCHLIST",
    }


def analyze_stock(yf_symbol: str, ticker: str, company: str = "", fresh_data: bool = True) -> Optional[AnalystResult]:
    """Freshly analyse one stock through the existing Swing Brain engine."""
    history = download_history(yf_symbol, period="2y", refresh=fresh_data)
    if history is None or history.empty or len(history) < 210:
        return None

    benchmark = download_history(BENCHMARK, period="2y", refresh=fresh_data)
    if benchmark is None or benchmark.empty:
        return None

    market = market_regime(benchmark)
    fundamentals = get_fundamentals(yf_symbol)
    result = setup_analysis(ticker, history, benchmark, market, fundamental=fundamentals if fundamentals.get("available") else None)
    if result is None:
        return None

    result["company"] = company or ticker
    result["yf_symbol"] = yf_symbol
    result["data_last_bar"] = history.index[-1].isoformat()
    result["fundamentals"] = fundamentals
    result["sector"] = fundamentals.get("sector", "Unknown")
    result["industry"] = fundamentals.get("industry", "Unknown")

    d = enrich(history)
    lifecycle = _breakout_lifecycle(d)
    overlay = _analyst_overlay(result, d, fundamentals, lifecycle, benchmark)
    result["analyst"] = overlay
    result["lifecycle"] = lifecycle
    result["breakout_level"] = lifecycle.get("level")
    result["breakout_age"] = lifecycle.get("age_days")
    result["entry_state"] = overlay["entry_state"]
    result["market_gate"] = overlay["market_gate"]
    result["risk_gate"] = overlay["risk_gate"]

    return AnalystResult(result=result, history=d, market=market, fundamentals=fundamentals, lifecycle=lifecycle)
