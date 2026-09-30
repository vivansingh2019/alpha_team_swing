"""Momentum Lab - Module 4.

Multi-timeframe momentum cockpit based on the user's Momentum Lab v2 logic.
It evaluates the same core ideas (move, VWAP, RVOL, structure, momentum,
pullback, trap/exhaustion, breakout/reclaim, pressure and health) independently
for each timeframe. It is a research/diagnostic module; it does not rewrite
Radar, Stock Analyst or Swing Glance scores.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple
import numpy as np
import pandas as pd
import streamlit as st

from data import download_history


def _f(v, default=0.0):
    try:
        if v is None or pd.isna(v):
            return default
        return float(v)
    except Exception:
        return default


def _tone(v: Any) -> str:
    s = str(v or "").upper()
    if any(k in s for k in ("FAIL", "FAILED", "NO BUY", "DO NOT CHASE", "RISK", "DOWNSIDE", "HIGH SUPPLY", "BEARISH")):
        return "bad"
    if any(k in s for k in ("HEALTHY", "STRONG", "BUY CANDIDATE", "READY", "CONFIRMED", "BUYING", "ACCUMULATION", "BULLISH", "POSITIVE")):
        return "good"
    if any(k in s for k in ("WATCH", "WAIT", "PENDING", "BUILDING", "NEUTRAL", "MEDIUM", "CONSOLIDATION", "PULLBACK", "EXTENDED", "CAUTION", "MIXED")):
        return "warn"
    return "neutral"


def _money(v):
    try:
        return f"₹{float(v):,.2f}"
    except Exception:
        return "—"


def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    pc = df.Close.shift(1)
    tr = pd.concat([(df.High-df.Low), (df.High-pc).abs(), (df.Low-pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()


def _resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    x = df.copy().sort_index()
    if not isinstance(x.index, pd.DatetimeIndex):
        x.index = pd.to_datetime(x.index)
    # Only completed/meaningful bars; drop incomplete NaN aggregates.
    out = x.resample(rule, origin="start_day").agg({
        "Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"
    }).dropna()
    return out


def _core(df: pd.DataFrame) -> Dict[str, Any]:
    if df is None or len(df) < 40:
        return {"state": "WAIT DATA", "action": "WAIT", "health": "DATA N/A", "score": 0}

    d = df.copy().dropna().sort_index()
    c = d.Close.astype(float); o = d.Open.astype(float); h = d.High.astype(float); l = d.Low.astype(float); v = d.Volume.astype(float)
    ema = c.ewm(span=20, adjust=False).mean()
    atr = _atr(d, 14)
    avg_vol = v.rolling(20).mean()
    rvol = (v / avg_vol.replace(0, np.nan)).fillna(0)
    prev_close = c.shift(1)
    move = (c / prev_close - 1.0) * 100.0
    # For a timeframe-neutral cockpit, "move" is bar-to-bar; on 1D this is the daily move.
    atr_pct = (atr / c * 100.0).replace([np.inf, -np.inf], np.nan).fillna(0)
    bar_range = (h-l).clip(lower=1e-9)
    upper_wick = (h - np.maximum(o,c)) / bar_range * 100
    lower_wick = (np.minimum(o,c) - l) / bar_range * 100
    upper_reject = upper_wick >= 45
    lower_reject = lower_wick >= 45
    above_vwap = c > ((h+l+c)/3).rolling(20).mean()
    vwap = ((h+l+c)/3).rolling(20).mean()
    vwap_dist = (c / vwap - 1) * 100
    rh = h.rolling(5).max()
    rl = l.rolling(5).min()
    bull_struct = c > rh.shift(1)
    bear_struct = c < rl.shift(1)
    higher_low = (l > l.shift(1)) & (c > c.shift(1))
    lower_high = (h < h.shift(1)) & (c < c.shift(1))
    vol_strong = rvol >= 1.5
    vol_very = rvol >= 2.0
    vol_weak = rvol < .9
    poor_progress = vol_strong & (move.abs() < np.maximum(atr_pct * .20, .05))
    bull_response = vol_strong & (move > 0) & (c >= o)
    bear_response = vol_strong & (move < 0) & (c <= o)

    buy = pd.Series(50.0, index=d.index)
    buy += np.where(((c-l)/bar_range) >= .65, 15, np.where(((c-l)/bar_range) >= .50, 5, -5))
    buy += np.where(c > o, 10, -10)
    buy += np.where(lower_reject, 8, 0)
    buy += np.where(above_vwap, 10, -10)
    buy += np.where(bull_response, 10, 0)
    buy += np.where(higher_low, 7, 0)
    buy -= np.where(upper_reject, 10, 0)
    buy -= np.where(poor_progress & (c <= o), 10, 0)
    buy = buy.clip(0,100)

    sell = pd.Series(50.0, index=d.index)
    sell += np.where(((c-l)/bar_range) <= .35, 15, np.where(((c-l)/bar_range) <= .50, 5, -5))
    sell += np.where(c < o, 10, -10)
    sell += np.where(upper_reject, 10, 0)
    sell += np.where(~above_vwap, 10, -10)
    sell += np.where(bear_response, 10, 0)
    sell += np.where(lower_high, 7, 0)
    sell -= np.where(lower_reject, 8, 0)
    sell -= np.where(poor_progress & (c >= o), 10, 0)
    sell = sell.clip(0,100)

    bp = float(buy.iloc[-5:].mean()); sp = float(sell.iloc[-5:].mean())
    ratio = bp/sp if sp > 0 else np.nan
    pressure_type = "CAPITULATION" if bool((vol_very & (move.abs() >= np.maximum(atr_pct*1.25,1.0))).iloc[-1]) else "SELLING" if sp >= 65 and sp > bp + 5 else "BUYING" if bp >= 65 and bp > sp + 5 else "BALANCED"

    open_now = float(o.iloc[-1])
    current = float(c.iloc[-1])
    # Momentum state uses the same thresholds from the source indicator, adapted to the current bar/timeframe.
    move_from_open = (current / open_now - 1) * 100 if open_now else 0
    early = move_from_open >= 1.5 and move_from_open < 3 and bool(above_vwap.iloc[-1]) and (bool(vol_strong.iloc[-1]) or bool((move > atr_pct*.5).iloc[-1]))
    healthy = move_from_open >= 3 and move_from_open < 7 and bool(above_vwap.iloc[-1]) and bool(vol_strong.iloc[-1]) and not bool(upper_reject.iloc[-1])
    extended = 7 <= move_from_open < 10
    extreme = move_from_open >= 10
    dd_high = (h.rolling(10).max().iloc[-1] - current) / h.rolling(10).max().iloc[-1] * 100
    controlled_pull = dd_high > .5 and dd_high < 3 and not bool(bear_response.iloc[-1])
    deep_pull = dd_high >= 3
    healthy_pull = controlled_pull and bool(above_vwap.iloc[-1]) and (bool(vol_weak.iloc[-1]) or float(rvol.iloc[-1]) < 1.2)
    range_pct = (h.rolling(10).max().iloc[-1]-l.rolling(10).min().iloc[-1])/current*100 if current else 0
    tight_range = range_pct <= 2.5
    range_state = tight_range and abs(move_from_open) >= 2 and not bool((move > atr_pct*.5).iloc[-1]) and not bool((move < -atr_pct*.5).iloc[-1])
    up_spike = bool(((move > atr_pct*.5) & vol_very & (move_from_open > 3)).iloc[-1])
    down_spike = bool(((move < -atr_pct*.5) & vol_very & (move_from_open < -3)).iloc[-1])
    reversal = down_spike and bool(lower_reject.iloc[-1]) and bool(above_vwap.iloc[-1])
    prior_res = h.rolling(20).max().shift(1)
    break_up = bool((c > prior_res).iloc[-1]) and bool(vol_strong.iloc[-1])
    break_fail = bool((h > prior_res).iloc[-1] and (c < prior_res).iloc[-1] and vol_strong.iloc[-1])
    reclaim = bool((c > prior_res*(1+.0025)).iloc[-1])
    reclaim_fail = bool((h > prior_res).iloc[-1] and (c < prior_res).iloc[-1] and (c < c.shift(1)).iloc[-1])

    trap = 0
    trap += 20 if bool(upper_reject.iloc[-1]) else 0
    trap += 20 if bool(poor_progress.iloc[-1]) else 0
    trap += 25 if reclaim_fail else 0
    trap += 15 if bool((~above_vwap & (move_from_open > 3)).iloc[-1]) else 0
    trap += 10 if bool(lower_high.iloc[-1]) else 0
    trap += 10 if deep_pull and bool(bear_response.iloc[-1]) else 0
    trap_risk = "HIGH" if trap >= 60 else "MEDIUM" if trap >= 35 else "LOW"
    ex = (20 if extreme else 0) + (20 if abs(float(vwap_dist.iloc[-1])) >= 5 else 0) + (20 if bool(upper_reject.iloc[-1]) else 0) + (20 if bool(poor_progress.iloc[-1]) else 0) + (20 if dd_high >= 3 else 0)
    exhaustion = "HIGH" if ex >= 60 else "MEDIUM" if ex >= 35 else "LOW"

    if break_fail or (reclaim_fail and not bool(above_vwap.iloc[-1])):
        state, action = "FAILED MOVE", "NO BUY"
    elif reversal:
        state, action = "REVERSAL WATCH", "WAIT CONFIRM"
    elif break_up and reclaim:
        state, action = "SHAKEOUT / RECLAIM", "WATCH RECLAIM"
    elif ex >= 60:
        state, action = "EXHAUSTION", "DO NOT CHASE"
    elif range_state:
        state, action = "CONSOLIDATION", "WAIT RANGE BREAK"
    elif healthy_pull:
        state, action = "HEALTHY PULLBACK", "WATCH CONTINUATION"
    elif healthy:
        state, action = "HEALTHY MOMENTUM", "BUY CANDIDATE"
    elif extended:
        state, action = "EXTENDED MOMENTUM", "DO NOT CHASE"
    elif early:
        state, action = "EARLY MOMENTUM", "WATCH"
    elif up_spike:
        state, action = "MOMENTUM SPIKE", "WAIT FOR HOLD"
    elif down_spike:
        state, action = "DOWNSIDE SPIKE", "WAIT"
    else:
        state, action = "NEUTRAL", "OBSERVE"

    score = 50 + (10 if above_vwap.iloc[-1] else -10) + (10 if vol_strong.iloc[-1] else -5) + (10 if bull_struct.iloc[-1] else 0) + (10 if higher_low.iloc[-1] else 0) - (15 if upper_reject.iloc[-1] else 0) - (15 if poor_progress.iloc[-1] else 0) - (10 if deep_pull else 0) - (10 if not above_vwap.iloc[-1] else 0)
    score = int(max(0,min(100,score)))
    health = score + (10 if bool((c.iloc[-1] > ema.iloc[-1])) else -10) + (10 if bool(bull_struct.iloc[-1]) else 0)
    health = int(max(0,min(100,health)))
    health_state = "HEALTHY" if health >= 70 else "RISK" if health <= 45 else "WATCH"

    return {
        "state": state, "action": action, "move": move_from_open, "rvol": float(rvol.iloc[-1]),
        "vwap_dist": float(vwap_dist.iloc[-1]), "structure": "BULLISH" if bull_struct.iloc[-1] else "BEARISH" if bear_struct.iloc[-1] else "MIXED",
        "mom_score": score, "pull_score": int(max(0,min(100,50 + (15 if above_vwap.iloc[-1] else -15) + (15 if (vol_weak.iloc[-1] or rvol.iloc[-1]<1.2) else -10) + (15 if higher_low.iloc[-1] else -15) - (10 if (move < -atr_pct*.5).iloc[-1] else 0) - (15 if deep_pull else 0)))),
        "trap": trap_risk, "exhaust": exhaustion, "range": "TIGHT" if tight_range else "WIDE", "breakout": "UP" if break_up else "FAILED" if break_fail else "NONE",
        "reclaim": "CONFIRMED" if reclaim else "FAILED" if reclaim_fail else "PENDING", "from_high": float(dd_high), "health": health_state, "health_score": health,
        "close": current, "bar_time": d.index[-1], "pressure": pressure_type, "buy_pressure": round(bp), "sell_pressure": round(sp),
        "dominant": "BUYERS" if bp-sp>=15 else "SELLERS" if sp-bp>=15 else "BATTLE", "volatility": "HIGH" if atr_pct.iloc[-1]>=2 else "NORMAL" if atr_pct.iloc[-1]>=1 else "LOW",
    }


def _prepare(yf_symbol: str, fresh: bool) -> Dict[str, pd.DataFrame]:
    out: Dict[str,pd.DataFrame] = {}
    # yfinance intraday limits: 5/15/30m are short-history feeds; 60m covers the 4H composite.
    # Intraday feeds have provider-specific history limits. Prefer native 60m
    # for 1H, but fall back to 30m -> 1H when the 60m request is unavailable.
    for tf, period, interval in [("5m","60d","5m"),("15m","60d","15m"),("30m","60d","30m"),("1H","730d","60m"),("1D","2y","1d")]:
        d = download_history(yf_symbol, period=period, interval=interval, refresh=fresh)
        if d is not None and not d.empty:
            out[tf] = d

    if "1H" not in out and "30m" in out:
        out["1H"] = _resample_ohlcv(out["30m"], "1h")

    if "1H" in out:
        out["4H"] = _resample_ohlcv(out["1H"], "4h")
    elif "30m" in out:
        out["4H"] = _resample_ohlcv(out["30m"], "4h")

    if "1D" in out:
        out["1W"] = _resample_ohlcv(out["1D"], "W-FRI")
    return out


def render(yf_symbol: str, ticker: str, company: str = "", fresh: bool = True):
    st.markdown("""
    <style>
    .ml-hero{border:1px solid rgba(60,160,230,.3);border-radius:18px;padding:15px 18px;background:linear-gradient(135deg,rgba(15,55,95,.28),rgba(80,25,105,.14),rgba(15,100,80,.10));margin-bottom:12px}
    .ml-grid{display:grid;grid-template-columns:repeat(7,minmax(0,1fr));gap:8px;margin:10px 0 14px}.ml-tf{border:1px solid rgba(90,150,220,.24);border-radius:13px;padding:9px 10px;background:rgba(8,25,45,.22);min-width:0}.ml-tf .tf{font-size:.72rem;opacity:.6;font-weight:800}.ml-tf .state{font-size:.82rem;font-weight:850;margin-top:5px;overflow-wrap:anywhere}.ml-tf .score{font-size:.76rem;margin-top:3px;opacity:.72}.ml-section{font-size:1.05rem;font-weight:850;margin:12px 0 7px}.ml-card{border:1px solid rgba(80,140,210,.22);border-radius:15px;padding:12px 14px;background:rgba(8,25,45,.14);min-width:0}.ml-card.flow{border-color:rgba(49,214,149,.32)}.ml-card.risk{border-color:rgba(255,102,116,.38)}.ml-row{display:flex;justify-content:space-between;gap:10px;padding:6px 0;border-bottom:1px solid rgba(120,120,120,.11);font-size:.82rem}.ml-row:last-child{border-bottom:0}.ml-row span{opacity:.65}.ml-row b{text-align:right;overflow-wrap:anywhere}.ml-good{color:#31d695}.ml-warn{color:#f3c65d}.ml-bad{color:#ff6674}.ml-blue{color:#4b9fff}.ml-purple{color:#9b69dc}.ml-note{font-size:.76rem;opacity:.62;line-height:1.4;margin-top:8px}@media(max-width:1100px){.ml-grid{grid-template-columns:repeat(4,minmax(0,1fr))}}@media(max-width:700px){.ml-grid{grid-template-columns:repeat(2,minmax(0,1fr))}}
    </style>""", unsafe_allow_html=True)

    st.markdown(f"<div class='ml-hero'><div style='font-size:.72rem;letter-spacing:.14em;opacity:.6;font-weight:800'>MOMENTUM LAB • MODULE 4 • MTF</div><div style='font-size:1.8rem;font-weight:850'>{ticker} <span style='font-size:.9rem;opacity:.58'>{company}</span></div><div style='opacity:.68;font-size:.84rem'>Multi-timeframe momentum health • price + VWAP + RVOL + structure + pressure • OHLCV research proxy</div></div>", unsafe_allow_html=True)

    with st.spinner(f"Loading multi-timeframe momentum for {ticker}..."):
        frames = _prepare(yf_symbol, fresh)
    if not frames:
        st.error("No intraday/daily data returned. Check internet/data source and try Fresh data again.")
        return

    results = {tf: _core(df) for tf, df in frames.items() if not df.empty}
    order = ["5m","15m","30m","1H","4H","1D","1W"]
    st.markdown("### ⚡ MTF Momentum Map")
    cards = []
    for tf in order:
        x = results.get(tf)
        if x is None:
            cards.append(f"<div class='ml-tf ml-missing'><div class='tf'>{tf}</div><div class='state ml-warn'>DATA N/A</div><div class='score'>Momentum —<br>Health —</div></div>")
            continue
        tone = _tone(x["state"])
        cls = "bad" if tone == "bad" else "good" if tone == "good" else "warn" if tone == "warn" else "blue"
        cards.append(f"<div class='ml-tf'><div class='tf'>{tf}</div><div class='state ml-{cls}'>{x['state']}</div><div class='score'>Momentum {x['mom_score']}/100<br>Health {x['health_score']}/100</div></div>")
    st.markdown("<div class='ml-grid'>" + "".join(cards) + "</div>", unsafe_allow_html=True)

    available = [results[tf] for tf in order if tf in results]
    short = [results[tf] for tf in ["5m","15m","30m"] if tf in results]
    swing = [results[tf] for tf in ["1H","4H"] if tf in results]
    higher = [results[tf] for tf in ["1D","1W"] if tf in results]
    def _avg(rows, key):
        return float(np.mean([r[key] for r in rows])) if rows else None
    short_m = _avg(short, "mom_score"); swing_m = _avg(swing, "mom_score"); higher_m = _avg(higher, "mom_score")
    all_m = _avg(available, "mom_score"); all_h = _avg(available, "health_score")
    if all_m is None:
        alignment = "DATA N/A"
    elif all_m >= 65 and (higher_m is None or higher_m >= 55):
        alignment = "BULLISH"
    elif all_m <= 40 and (higher_m is None or higher_m <= 45):
        alignment = "BEARISH"
    else:
        alignment = "MIXED"
    st.markdown("### 🧭 MTF Alignment")
    ac1, ac2, ac3, ac4 = st.columns(4)
    ac1.metric("Short-term", f"{short_m:.0f}/100" if short_m is not None else "—")
    ac2.metric("Swing-term", f"{swing_m:.0f}/100" if swing_m is not None else "—")
    ac3.metric("Higher-TF", f"{higher_m:.0f}/100" if higher_m is not None else "—")
    ac4.metric("Alignment", alignment)
    st.caption("Alignment is a descriptive MTF summary of the module's timeframe scores; it does not rewrite Radar, Stock Analyst or Swing Glance decisions.")

    inspect_order = [tf for tf in order if tf in results]
    selected_tf = st.selectbox("Inspect timeframe", inspect_order, index=inspect_order.index("1D") if "1D" in inspect_order else 0, key="momentum_tf")
    x = results[selected_tf]

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("State", x["state"])
    c2.metric("Action", x["action"])
    c3.metric("Momentum", f"{x['mom_score']}/100")
    c4.metric("Health", f"{x['health_score']}/100 • {x['health']}")

    a,b,c = st.columns(3)
    with a:
        rows=[("Move",f"{x['move']:+.2f}%"),("RVOL",f"{x['rvol']:.2f}x"),("VWAP distance",f"{x['vwap_dist']:+.2f}%"),("Structure",x["structure"]),("Range",x["range"])]
        st.markdown("<div class='ml-card'><div class='ml-section'>1. Momentum Core</div>"+"".join(f"<div class='ml-row'><span>{k}</span><b class='ml-{('bad' if _tone(v)=='bad' else 'good' if _tone(v)=='good' else 'warn' if _tone(v)=='warn' else 'blue')}'>{v}</b></div>" for k,v in rows)+"</div>",unsafe_allow_html=True)
    with b:
        rows=[("Pullback score",f"{x['pull_score']}/100"),("Breakout",x["breakout"]),("Reclaim",x["reclaim"]),("From high",f"-{x['from_high']:.2f}%"),("Pressure type",x["pressure"])]
        st.markdown("<div class='ml-card'><div class='ml-section'>2. Location & Setup</div>"+"".join(f"<div class='ml-row'><span>{k}</span><b class='ml-{('bad' if _tone(v)=='bad' else 'good' if _tone(v)=='good' else 'warn' if _tone(v)=='warn' else 'purple')}'>{v}</b></div>" for k,v in rows)+"</div>",unsafe_allow_html=True)
    with c:
        rows=[("Buy pressure",f"{x['buy_pressure']}/100"),("Sell pressure",f"{x['sell_pressure']}/100"),("Dominant force",x["dominant"]),("Trap risk",x["trap"]),("Exhaustion",x["exhaust"])]
        st.markdown("<div class='ml-card risk'><div class='ml-section'>3. Pressure & Risk</div>"+"".join(f"<div class='ml-row'><span>{k}</span><b class='ml-{('bad' if _tone(v)=='bad' else 'good' if _tone(v)=='good' else 'warn' if _tone(v)=='warn' else 'purple')}'>{v}</b></div>" for k,v in rows)+"</div>",unsafe_allow_html=True)

    st.markdown("### 📊 Momentum alignment across timeframes")
    align_rows=[]
    for tf in order:
        z=results.get(tf)
        if z is None:
            align_rows.append({"Timeframe":tf,"State":"DATA N/A","Action":"WAIT","Momentum":"—","Health":"—","RVOL":"—","Structure":"—","Pressure":"—"})
        else:
            align_rows.append({"Timeframe":tf,"State":z["state"],"Action":z["action"],"Momentum":z["mom_score"],"Health":z["health_score"],"RVOL":f"{z['rvol']:.2f}x","Structure":z["structure"],"Pressure":z["pressure"]})
    st.dataframe(pd.DataFrame(align_rows), use_container_width=True, hide_index=True)

    latest = frames[selected_tf]
    if len(latest) >= 80:
        st.markdown(f"### 📈 {selected_tf} price / momentum view")
        chart = latest[[c for c in ["Close"] if c in latest.columns]].tail(160)
        st.line_chart(chart, height=300, use_container_width=True)

    bar = x.get("bar_time")
    st.caption(f"Data timeframe: {selected_tf} • Last bar: {bar} • Fresh request: {'ON' if fresh else 'OFF'} • Module 4 does not rewrite Radar, Stock Analyst or Swing Glance scores.")
