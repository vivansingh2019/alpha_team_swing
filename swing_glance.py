"""Swing Glance - Module 3.

A compact, one-stock institutional-style cockpit built on Swing Brain's existing
analyst data. This module is deliberately diagnostic/context-first: it does not
rewrite Radar or Stock Analyst scoring.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
import streamlit as st


def _f(v, default=0.0):
    try:
        if v is None or pd.isna(v):
            return default
        return float(v)
    except Exception:
        return default


def _money(v):
    try:
        return f"₹{float(v):,.2f}"
    except Exception:
        return "—"


def _tone(v: Any) -> str:
    s = str(v or "").upper()
    if any(k in s for k in ("FAIL", "FAILED", "BLOCKED", "INVALID", "WEAK", "NO CHASE", "DAMAGED", "RISK-OFF", "DISTRIBUTION", "REDUCE")):
        return "bad"
    if any(k in s for k in ("STRONG", "CONFIRMED", "PASS", "LEADING", "FAVORABLE", "BULLISH", "BUY", "ACCUMULATION", "INTACT", "OPEN")):
        return "good"
    if any(k in s for k in ("WATCH", "WAIT", "PENDING", "BUILDING", "REVIEW", "MIXED", "NEUTRAL", "PULLBACK", "PRESSURE", "STRETCHED", "CAUTION", "CANDIDATE")):
        return "warn"
    return "neutral"


def _spi(history: pd.DataFrame, length: int = 10) -> Dict[str, Any]:
    """Transparent SPI-style proxy: candle close location weighted by volume."""
    if history is None or len(history) < length + 2:
        return {"buy": None, "sell": None, "ratio": None, "bias": "DATA N/A", "score": 0, "volatility": "N/A", "vol_pct": None}
    h = history.copy()
    rng = (h.High - h.Low).clip(lower=1e-9)
    loc = ((h.Close - h.Low) / rng).clip(0, 1)
    buy_raw = h.Volume * loc
    sell_raw = h.Volume * (1 - loc)
    buy = float(buy_raw.tail(length).sum())
    sell = float(sell_raw.tail(length).sum())
    ratio = buy / sell if sell > 0 else None
    if ratio is None:
        bias = "DATA N/A"; score = 0
    elif ratio >= 1.75:
        bias = "STRONG BUY"; score = 10
    elif ratio >= 1.25:
        bias = "BUY BIAS"; score = 7
    elif ratio <= 0.80:
        bias = "SELL BIAS"; score = 0
    else:
        bias = "NEUTRAL"; score = 4
    atr = h.High.combine(h.Low, lambda a, b: a-b).abs().rolling(14).mean()
    vol_pct = float((atr.iloc[-1] / h.Close.iloc[-1]) * 100) if h.Close.iloc[-1] else None
    vol_text = "N/A" if vol_pct is None else "LOW" if vol_pct < 1 else "NORMAL" if vol_pct < 2 else "HIGH"
    return {"buy": buy, "sell": sell, "ratio": ratio, "bias": bias, "score": score, "volatility": vol_text, "vol_pct": vol_pct}


def _market_values(market: Dict[str, Any]) -> Dict[str, str]:
    regime = str(market.get("regime", "UNKNOWN"))
    return {
        "regime": regime,
        "score": str(market.get("score", "—")),
        "risk": str(market.get("risk", "—")),
    }


def _render_card(title: str, rows: list[tuple[str, str]], cls: str = "gl-card"):
    body = "".join(
        f"<div class='gl-row'><span>{label}</span><b class='tone-{_tone(value)}'>{value}</b></div>"
        for label, value in rows
    )
    st.markdown(f"<div class='{cls}'><div class='gl-title'>{title}</div>{body}</div>", unsafe_allow_html=True)


def render(ar, market: Dict[str, Any]):
    r = ar.result
    a = r.get("analyst", {}) or {}
    f = ar.fundamentals or {}
    life = ar.lifecycle or {}
    d = ar.history
    spi = _spi(d)
    mv = _market_values(market)

    st.markdown(
        """
        <style>
        .gl-hero{border:1px solid rgba(70,150,220,.28);border-radius:18px;padding:15px 18px;background:linear-gradient(135deg,rgba(20,55,95,.32),rgba(70,25,105,.16),rgba(20,105,90,.10));margin:8px 0 12px;}
        .gl-kicker{font-size:.72rem;letter-spacing:.14em;opacity:.62;font-weight:800;text-transform:uppercase}.gl-name{font-size:1.75rem;font-weight:850}.gl-sub{opacity:.68;font-size:.86rem;margin-top:3px;overflow-wrap:anywhere}
        .gl-decision{border:1px solid rgba(243,198,93,.4);border-radius:14px;padding:10px 14px;text-align:center;background:rgba(105,80,20,.13)}.gl-decision .v{font-size:1.05rem;font-weight:850}.gl-decision .s{font-size:.72rem;opacity:.62}
        .gl-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px;margin-bottom:10px}.gl-card{border:1px solid rgba(80,140,210,.22);border-radius:15px;padding:12px 14px;background:rgba(8,25,45,.15);min-width:0}.gl-card.flow{border-color:rgba(49,214,149,.30)}.gl-card.risk{border-color:rgba(255,102,116,.38)}.gl-card.level{border-color:rgba(70,150,220,.28)}.gl-card.quality{border-color:rgba(120,100,220,.34)}.gl-title{font-size:.92rem;font-weight:800;letter-spacing:.04em;margin-bottom:7px}.gl-row{display:flex;justify-content:space-between;gap:12px;padding:6px 0;border-bottom:1px solid rgba(120,120,120,.11);font-size:.80rem;min-width:0}.gl-row:last-child{border-bottom:0}.gl-row span{opacity:.66}.gl-row b{text-align:right;overflow-wrap:anywhere;word-break:break-word}.tone-good{color:#31d695}.tone-warn{color:#f3c65d}.tone-bad{color:#ff6674}.tone-neutral{color:inherit}
        .gl-strip{display:grid;grid-template-columns:repeat(7,minmax(0,1fr));gap:8px;margin-bottom:10px}.gl-kpi{border:1px solid rgba(90,150,220,.22);border-radius:12px;padding:9px 10px;background:rgba(10,25,45,.28);min-width:0}.gl-kpi .l{font-size:.67rem;opacity:.60;text-transform:uppercase}.gl-kpi .v{font-size:.88rem;font-weight:800;margin-top:3px;overflow-wrap:anywhere}.gl-note{font-size:.76rem;opacity:.60;line-height:1.4;margin-top:5px}.gl-map{display:flex;gap:7px;flex-wrap:wrap;margin:8px 0 12px}.gl-step{padding:7px 10px;border-radius:999px;border:1px solid rgba(120,120,120,.25);font-size:.76rem;font-weight:750}.gl-step.active{border-color:rgba(49,214,149,.65);background:rgba(49,214,149,.10)}.gl-step.warn{border-color:rgba(243,198,93,.65);background:rgba(243,198,93,.09)}.gl-step.bad{border-color:rgba(255,102,116,.65);background:rgba(255,102,116,.09)}
        @media(max-width:900px){.gl-strip{grid-template-columns:repeat(3,minmax(0,1fr))}.gl-grid{grid-template-columns:1fr}} @media(max-width:600px){.gl-strip{grid-template-columns:repeat(2,minmax(0,1fr))}}
        </style>
        """,
        unsafe_allow_html=True,
    )

    entry = a.get("entry_state", r.get("entry_state", "WAIT"))
    decision = "DO NOT CHASE" if "NO CHASE" in str(entry).upper() else "WAIT" if "WAIT" in str(entry).upper() or "WATCH" in str(entry).upper() else str(entry)
    quality = r.get("score", a.get("structure_score", "—"))
    setup = r.get("phase", "—")
    lifecycle_state = life.get("state", r.get("breakout_state", "—"))

    st.markdown(
        f"<div class='gl-hero'><div style='display:flex;justify-content:space-between;gap:16px;align-items:center;flex-wrap:wrap'><div><div class='gl-kicker'>Swing Glance • Module 3</div><div class='gl-name'>{r.get('ticker','—')} <span style='font-size:.9rem;opacity:.58'>{r.get('company','')}</span></div><div class='gl-sub'>One-stock institutional quick cockpit • transparent flow proxy • same Swing Brain evidence chain</div></div><div class='gl-decision'><div class='s'>DECISION</div><div class='v tone-{_tone(decision)}'>{decision}</div><div class='s'>Quality {quality}/100</div></div></div></div>",
        unsafe_allow_html=True,
    )

    kpis = [
        ("CMP", _money(r.get("cmp"))), ("Market", mv["regime"]), ("Leadership", a.get("rs_state", "—")),
        ("Setup", setup), ("Breakout", lifecycle_state), ("SPI", spi["bias"]), ("Risk", a.get("risk_gate", "—")),
    ]
    st.markdown("<div class='gl-strip'>" + "".join(f"<div class='gl-kpi'><div class='l'>{l}</div><div class='v tone-{_tone(v)}'>{v}</div></div>" for l,v in kpis) + "</div>", unsafe_allow_html=True)

    steps = ["MARKET", "FLOW", "STRUCTURE", "SETUP", "ENTRY", "RISK", "DECISION"]
    active_index = 6 if decision not in ("WAIT", "DO NOT CHASE") else 5
    st.markdown("<div class='gl-map'>" + "".join(f"<span class='gl-step {'active' if i <= active_index else 'warn'}'>{s}</span>" for i,s in enumerate(steps)) + "</div>", unsafe_allow_html=True)

    left1, right1 = st.columns(2)
    with left1:
        _render_card("1. MARKET CONTEXT", [
            ("Regime", mv["regime"]),
            ("Market score", mv["score"]),
            ("Market risk", mv["risk"]),
            ("Weekly", a.get("weekly_state", "—")),
            ("Swing context", a.get("swing_context", "—")),
        ])
    with right1:
        _render_card("2. SPI FLOW • TRANSPARENT PROXY", [
            ("SPI bias", spi["bias"]),
            ("SPI ratio", f"{spi['ratio']:.2f}" if spi["ratio"] is not None else "—"),
            ("Flow score", f"{spi['score']}/10"),
            ("Volatility", spi["volatility"] + (f" ({spi['vol_pct']:.2f}%)" if spi["vol_pct"] is not None else "")),
            ("Participation", a.get("participation_state", "—")),
        ], "gl-card flow")

    left2, right2 = st.columns(2)
    with left2:
        _render_card("3. TREND & STRUCTURE", [
            ("Daily trend", a.get("daily_state", "—")),
            ("Swing structure", a.get("swing_state", "—")),
            ("HH / HL", "YES" if a.get("higher_high") and a.get("higher_low") else "NO"),
            ("Compression", a.get("compression_state", "—")),
            ("RS context", a.get("rs_context_state", "—")),
        ])
    with right2:
        _render_card("4. KEY LEVELS", [
            ("Resistance", _money(r.get("resistance"))),
            ("Support", _money(r.get("support"))),
            ("Breakout level", _money(life.get("level"))),
            ("Planned entry", _money(r.get("entry_low"))),
            ("Target", _money(r.get("target1"))),
        ], "gl-card level")

    left3, right3 = st.columns(2)
    with left3:
        _render_card("5. SETUP & TRIGGER", [
            ("Setup", setup),
            ("Lifecycle", lifecycle_state),
            ("Breakout age", f"{life.get('age_days')} bars" if life.get("age_days") is not None else "—"),
            ("Breakout score", str(r.get("breakout_score", "—"))),
            ("Volume", a.get("price_volume", "—")),
            ("Pace", a.get("pace", "—")),
        ])
    with right3:
        _render_card("6. ENTRY & RISK", [
            ("Entry state", entry),
            ("Entry zone", f"{_money(r.get('entry_low'))} – {_money(r.get('entry_high'))}"),
            ("Stop / invalid", _money(r.get("stop_loss"))),
            ("Risk", f"{_f(r.get('risk_pct')):.2f}%" if r.get("risk_pct") is not None else "—"),
            ("Target", _money(r.get("target1"))),
            ("R:R", f"{_f(r.get('rr')):.2f}:1" if r.get("rr") is not None else "—"),
        ], "gl-card risk")

    st.markdown("### 7. QUALITY → DECISION")
    q1, q2, q3 = st.columns(3)
    with q1:
        _render_card("QUALITY", [("Overall", f"{quality}/100"), ("Technical", str(r.get("technical_score", "—"))), ("Fundamental", str(r.get("fundamental_score", "—"))), ("Evidence", f"{a.get('evidence_score','—')}/100")], "gl-card quality")
    with q2:
        _render_card("WHY", [("Primary reason", a.get("blockers", ["No major blocker"])[0] if a.get("blockers") else "No major blocker"), ("Thesis", a.get("thesis", "—")), ("Hold", a.get("hold_action", "—"))])
    with q3:
        _render_card("DECISION", [("Action", decision), ("Market gate", a.get("market_gate", "—")), ("Risk gate", a.get("risk_gate", "—")), ("Timing", f"{a.get('timing_score','—')}/100")], "gl-card risk" if _tone(decision)=="bad" else "gl-card")

    if len(d) >= 80:
        chart = d[[c for c in ["Close", "EMA20", "EMA50", "EMA200"] if c in d.columns]].tail(120).copy()
        chart.columns = [c.replace("Close", "Close").replace("EMA20", "EMA20").replace("EMA50", "EMA50").replace("EMA200", "EMA200") for c in chart.columns]
        st.markdown("### 8. PRICE / STRUCTURE GLANCE")
        st.line_chart(chart, height=300, use_container_width=True)

    st.caption("Module 3 is a quick research cockpit. SPI is a transparent proxy based on volume and candle close location; it is not proprietary institutional transaction data. Module 3 does not rewrite Radar or Stock Analyst scores.")
