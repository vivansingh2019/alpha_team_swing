import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

from universe import load_universe
from stock_analyst import analyze_stock
from swing_glance import render as render_swing_glance
from momentum_lab import render as render_momentum_lab
from config import BENCHMARK

IST = ZoneInfo("Asia/Kolkata")

st.set_page_config(page_title="Swing Brain", page_icon="🧠", layout="wide", initial_sidebar_state="expanded")

# ---------- UI helpers ----------
def _money(v):
    try:
        return f"₹{float(v):,.2f}"
    except (TypeError, ValueError):
        return "—"


def _num(v, digits=1):
    try:
        return f"{float(v):.{digits}f}"
    except (TypeError, ValueError):
        return "—"


def _value_tone(value):
    v = str(value or "").upper()
    if any(k in v for k in ("FAIL", "FAILED", "BLOCKED", "INVALID", "NO CHASE", "DAMAGED", "WEAK")):
        return "bad"
    if any(k in v for k in ("STRONG", "CONFIRMED", "PASS", "LEADING", "FAVORABLE", "BULLISH", "ACCUMULATION", "INTACT")):
        return "good"
    if any(k in v for k in ("WATCH", "WAIT", "PENDING", "BUILDING", "REVIEW", "MIXED", "NEUTRAL", "PULLBACK", "PRESSURE", "STRETCHED")):
        return "warn"
    return "neutral"


def _pill(label, value):
    tone = _value_tone(value)
    st.markdown(
        f"<div class='pill'><span class='pill-label'>{label}</span><span class='pill-value tone-{tone}'>{value}</span></div>",
        unsafe_allow_html=True,
    )


def _open_stock(ticker):
    st.session_state["analyst_ticker"] = ticker
    st.session_state["module"] = "🔎 Stock Analyst"
    st.rerun()


st.markdown(
    """
    <style>
    .block-container {padding-top: 1.1rem; padding-bottom: 2rem;}
    /* Medium, readable typography with wrapping instead of clipping. */
    .block-container, .block-container p, .block-container label, .block-container div {
        font-size: 0.96rem;
    }
    .hero {padding: 1rem 1.2rem; border: 1px solid rgba(120,120,120,.22); border-radius: 18px; background: linear-gradient(135deg, rgba(30,40,65,.14), rgba(15,90,90,.08)); margin-bottom: 1rem;}
    .hero h1 {margin:0; font-size:2rem;}
    .hero p {margin:.25rem 0 0; opacity:.72; overflow-wrap:anywhere; word-break:break-word;}
    .card {border:1px solid rgba(120,120,120,.22); border-radius:16px; padding:14px 16px; margin-bottom:10px; background:rgba(127,127,127,.045); min-width:0;}
    .card-title {font-size:1.08rem; font-weight:650; margin-bottom:3px; overflow-wrap:anywhere; word-break:break-word; white-space:normal;}
    .muted {opacity:.68; font-size:.92rem; overflow-wrap:anywhere; word-break:break-word; white-space:normal;}
    .big {font-size:1.55rem; font-weight:750; overflow-wrap:anywhere; word-break:break-word; white-space:normal;}
    .pill {display:flex; justify-content:space-between; align-items:flex-start; gap:10px; padding:7px 10px; border-bottom:1px solid rgba(120,120,120,.14); min-width:0;}
    .pill-label {opacity:.7; overflow-wrap:anywhere; word-break:break-word; white-space:normal;}
    .pill-value {font-weight:700; text-align:right; overflow-wrap:anywhere; word-break:break-word; white-space:normal;}
    .tone-good {color:#31d695 !important;}
    .tone-warn {color:#f3c65d !important;}
    .tone-bad {color:#ff6674 !important;}
    .tone-neutral {color:inherit !important;}
    .section-kicker {font-size:.74rem; letter-spacing:.13em; text-transform:uppercase; opacity:.55; font-weight:800; margin-bottom:3px;}
    .section-title {font-size:1.25rem; font-weight:800; margin:0 0 .65rem 0;}
    .kpi-card {border:1px solid rgba(90,150,220,.24); border-radius:12px; padding:10px 12px; min-height:72px; background:linear-gradient(145deg,rgba(20,55,95,.18),rgba(20,25,45,.16));}
    .kpi-label {font-size:.72rem; opacity:.62; text-transform:uppercase; letter-spacing:.05em; margin-bottom:4px;}
    .kpi-value {font-size:1.02rem; font-weight:750; line-height:1.15; overflow-wrap:anywhere; word-break:break-word;}
    .kpi-card.good {border-color:rgba(49,214,149,.35); background:linear-gradient(145deg,rgba(20,100,75,.20),rgba(20,35,40,.16));}
    .kpi-card.warn {border-color:rgba(243,198,93,.35); background:linear-gradient(145deg,rgba(105,80,20,.18),rgba(35,30,20,.15));}
    .kpi-card.bad {border-color:rgba(255,102,116,.35); background:linear-gradient(145deg,rgba(110,30,40,.18),rgba(40,20,25,.15));}
    .mini-note {font-size:.78rem; opacity:.62; line-height:1.35; overflow-wrap:anywhere;}
    .chart-card {border:1px solid rgba(70,150,220,.22); border-radius:16px; padding:10px 12px 5px; background:rgba(8,25,45,.16); min-width:0;}
    .chart-title {font-size:.82rem; font-weight:750; letter-spacing:.05em; opacity:.78; text-transform:uppercase; margin-bottom:2px;}
    .life {display:flex; gap:5px; align-items:center; flex-wrap:wrap; margin:.5rem 0 1rem;}
    .life span {padding:6px 10px; border-radius:999px; border:1px solid rgba(120,120,120,.25); font-size:.86rem; font-weight:550; white-space:normal; overflow-wrap:anywhere; word-break:break-word;}
    .life .active {font-weight:700; border-color: rgba(40,160,120,.7); background: rgba(40,160,120,.12);}
    .life .bad-life {border-color: rgba(220,70,70,.75); background: rgba(220,70,70,.10);}
    .analyst-hero {display:flex; justify-content:space-between; align-items:center; gap:18px; padding:18px 20px 12px; border:1px solid rgba(120,120,120,.22); border-radius:18px; background:linear-gradient(135deg,rgba(28,55,95,.34),rgba(70,25,105,.18),rgba(20,105,90,.12)); margin-bottom:14px; min-width:0;}
    .analyst-hero-wrap {padding:0;}
    .hero-kpis {display:grid; grid-template-columns:repeat(6,minmax(0,1fr)); gap:9px; margin-top:14px;}
    .hero-kpi {border:1px solid rgba(90,150,220,.22); border-radius:11px; padding:9px 10px; min-height:58px; background:rgba(10,25,45,.30); min-width:0;}
    .hero-kpi.good {border-color:rgba(49,214,149,.34); background:rgba(20,100,75,.16);}
    .hero-kpi.warn {border-color:rgba(243,198,93,.34); background:rgba(105,80,20,.15);}
    .hero-kpi.bad {border-color:rgba(255,102,116,.34); background:rgba(110,30,40,.15);}
    .hero-kpi-label {font-size:.68rem; opacity:.62; margin-bottom:4px;}
    .hero-kpi-value {font-size:.94rem; font-weight:750; line-height:1.15; overflow-wrap:anywhere; word-break:break-word;}
    .setup-card {border:1px solid rgba(70,150,220,.20); border-radius:16px; padding:14px; background:rgba(8,25,45,.14); min-width:0;}
    .setup-grid {display:grid; grid-template-columns:repeat(5,minmax(0,1fr)); gap:8px;}
    .setup-item {border:1px solid rgba(120,120,120,.17); border-radius:10px; padding:9px 10px; min-width:0; background:rgba(127,127,127,.025);}
    .setup-item-label {font-size:.68rem; opacity:.60; margin-bottom:4px;}
    .setup-item-value {font-size:.82rem; font-weight:750; line-height:1.15; overflow-wrap:anywhere; word-break:break-word;}
    .quick-card {border:1px solid rgba(70,150,220,.20); border-radius:16px; padding:14px; background:rgba(8,25,45,.14); min-width:0;}
    .quick-row {display:flex; justify-content:space-between; gap:8px; padding:7px 0; border-bottom:1px solid rgba(120,120,120,.10); min-width:0;}
    .quick-row:last-child {border-bottom:0;}
    .quick-label {opacity:.66; font-size:.80rem;}
    .quick-value {font-weight:750; text-align:right; overflow-wrap:anywhere; word-break:break-word;}
    .metric-strip {display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:10px; margin:4px 0 12px;}
    .metric-panel {border:1px solid rgba(80,140,210,.20); border-radius:14px; padding:12px; background:rgba(20,45,75,.10); min-width:0;}
    .metric-panel.stop {border-color:rgba(255,102,116,.42); background:rgba(110,30,40,.12);}
    .metric-panel.levels {border-color:rgba(49,214,149,.28);}
    .metric-panel.sector {border-color:rgba(120,100,220,.30);}
    .metric-panel.timing {border-color:rgba(243,198,93,.34);}
    .metric-panel-title {font-weight:750; font-size:.92rem; margin-bottom:8px;}
    .metric-line {display:flex; justify-content:space-between; gap:8px; padding:5px 0; border-bottom:1px solid rgba(120,120,120,.10); font-size:.80rem;}
    .metric-line:last-child {border-bottom:0;}
    .metric-line b {text-align:right; overflow-wrap:anywhere; word-break:break-word;}
    @media (max-width: 1100px) { .hero-kpis {grid-template-columns:repeat(3,minmax(0,1fr));} .setup-grid {grid-template-columns:repeat(3,minmax(0,1fr));} .metric-strip {grid-template-columns:repeat(2,minmax(0,1fr));} }
    @media (max-width: 700px) { .hero-kpis {grid-template-columns:repeat(2,minmax(0,1fr));} .setup-grid {grid-template-columns:repeat(2,minmax(0,1fr));} .metric-strip {grid-template-columns:1fr;} }
    .analyst-kicker {font-size:.78rem; letter-spacing:.12em; opacity:.62; font-weight:700;}
    .analyst-title {font-size:1.65rem; font-weight:800; overflow-wrap:anywhere;}
    .analyst-title span {font-size:.92rem; opacity:.62; font-weight:500; margin-left:6px;}
    .analyst-sub {font-size:.88rem; opacity:.68; margin-top:4px; overflow-wrap:anywhere;}
    .hero-decision {min-width:125px; text-align:center; padding:8px 12px; border-left:1px solid rgba(120,120,120,.2);}
    .hero-score {font-size:1.65rem; font-weight:800; margin-top:5px;}
    .hero-label {font-size:.75rem; opacity:.6;}
    .state-badge {display:inline-block; padding:5px 9px; border-radius:999px; border:1px solid rgba(120,120,120,.28); font-size:.76rem; font-weight:700; overflow-wrap:anywhere;}
    .state-badge.good {border-color:rgba(50,180,130,.65); background:rgba(50,180,130,.10);}
    .state-badge.warn {border-color:rgba(220,170,50,.65); background:rgba(220,170,50,.10);}
    .state-badge.bad {border-color:rgba(220,70,70,.65); background:rgba(220,70,70,.10);}
    .decision-box,.thesis-box,.conv-card {border:1px solid rgba(120,120,120,.20); border-radius:12px; padding:11px 12px; min-height:62px; background:rgba(127,127,127,.035); min-width:0;}
    .decision-box div,.thesis-box div,.conv-card div {font-size:.73rem; opacity:.62; margin-bottom:5px; text-transform:uppercase; letter-spacing:.03em; overflow-wrap:anywhere;}
    .decision-box b,.thesis-box b,.conv-card b {font-size:.92rem; line-height:1.2; overflow-wrap:anywhere; word-break:break-word; white-space:normal;}
    .e-card {border:1px solid rgba(120,120,120,.20); border-radius:14px; padding:12px 13px; margin-bottom:12px; background:rgba(127,127,127,.035); min-width:0;}
    .e-head {display:flex; justify-content:space-between; gap:8px; align-items:flex-start; margin-bottom:7px;}
    .e-title {font-size:.98rem; font-weight:750; overflow-wrap:anywhere;}
    .e-score {padding:4px 7px; border-radius:999px; border:1px solid rgba(120,120,120,.22); font-size:.72rem; font-weight:700; white-space:nowrap;}
    .e-score.good {border-color:rgba(50,180,130,.55); background:rgba(50,180,130,.08);}
    .e-score.warn {border-color:rgba(220,170,50,.55); background:rgba(220,170,50,.08);}
    .e-score.bad {border-color:rgba(220,70,70,.55); background:rgba(220,70,70,.08);}
    .e-row {display:flex; justify-content:space-between; gap:12px; padding:6px 0; border-bottom:1px solid rgba(120,120,120,.10); min-width:0;}
    .e-row:last-child {border-bottom:0;}
    .e-row span {opacity:.68; overflow-wrap:anywhere;}
    .e-row b {text-align:right; overflow-wrap:anywhere; word-break:break-word; white-space:normal;}
    @media (max-width: 900px) { .analyst-hero {align-items:flex-start; flex-direction:column;} .hero-decision {border-left:0; border-top:1px solid rgba(120,120,120,.2); width:100%;} }

    /* Streamlit widgets: allow medium-sized values and long labels to wrap. */
    [data-testid="stMetricLabel"], [data-testid="stMetricValue"], [data-testid="stMetricDelta"] {
        white-space:normal !important;
        overflow-wrap:anywhere !important;
        word-break:break-word !important;
    }
    [data-testid="stMetricLabel"] {font-size:.78rem !important; font-weight:650 !important; opacity:.68 !important;}
    [data-testid="stMetricValue"] {font-size:1.22rem !important; font-weight:750 !important; line-height:1.15 !important;}
    [data-testid="stMetric"] {padding:.55rem .7rem !important; border:1px solid rgba(80,140,210,.18) !important; border-radius:11px !important; background:rgba(20,45,75,.12) !important; min-height:64px !important;}
    [data-testid="stButton"] button {
        white-space:normal !important;
        overflow-wrap:anywhere !important;
        word-break:break-word !important;
        min-height:2.4rem;
        line-height:1.2 !important;
        font-size:.90rem !important;
        font-weight:550 !important;
    }
    [data-testid="stSelectbox"] label, [data-testid="stCheckbox"] label {
        white-space:normal !important;
        overflow-wrap:anywhere !important;
    }
    [data-testid="stDataFrame"] {font-size:.90rem !important;}
    </style>
    """,
    unsafe_allow_html=True,
)

state_path = Path("state.json")
state = {}
if state_path.exists():
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:
        state = {}

try:
    universe = load_universe(refresh=False)
except Exception:
    universe = pd.DataFrame(columns=["SYMBOL", "COMPANY", "YF_SYMBOL"])

results = state.get("results", [])
market = state.get("market", {})
now = datetime.now(IST)

if "module" not in st.session_state:
    st.session_state["module"] = "🔥 Radar"

with st.sidebar:
    st.markdown("## 🧠 Swing Brain")
    st.caption("Research cockpit • no broker orders")
    module = st.radio("Module", ["🔥 Radar", "🔎 Stock Analyst", "⚡ Swing Glance", "⚡ Momentum Lab"], key="module")
    st.divider()
    st.caption(f"Market time\n{now.strftime('%d-%b-%Y %H:%M:%S IST')}")
    st.caption(f"Market regime: **{market.get('regime', 'UNKNOWN')}**")
    st.caption(f"Last scan: **{state.get('scan_finished_at') or state.get('generated_at') or 'N/A'}**")

st.markdown(
    "<div class='hero'><h1>🧠 Swing Brain</h1><p>Price + Volume + Relative Strength + Structure + Fundamentals — one evidence chain.</p></div>",
    unsafe_allow_html=True,
)


# ========================= RADAR =========================
if module == "🔥 Radar":
    st.subheader("📡 Live Radar")
    latest_dates = [x.get("data_last_bar", "")[:10] for x in results[:100] if x.get("data_last_bar")]
    data_status = "🟢 FRESH / TODAY" if state.get("data_mode") == "FRESH_REQUEST" and now.date().isoformat() in latest_dates else "🟠 FRESH / OLD BAR" if state.get("data_mode") == "FRESH_REQUEST" else "🟡 CACHE MODE"

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Market", market.get("regime", "UNKNOWN"))
    m2.metric("Universe", state.get("universe_count", 0))
    m3.metric("Scanned", state.get("stocks_scanned", 0))
    m4.metric("Breakouts", sum(x.get("phase") == "BREAKOUT" for x in results))
    m5.metric("Entry Ready", sum(x.get("status") == "ENTRY READY" for x in results))
    m6.metric("Data", data_status)

    st.divider()
    st.subheader("🔥 What just happened?")
    fresh = [x for x in results if x.get("phase") == "BREAKOUT"]
    fresh = sorted(fresh, key=lambda x: (x.get("priority_score", x.get("score", 0)), x.get("volume_ratio", 0)), reverse=True)[:8]
    if fresh:
        cols = st.columns(min(4, len(fresh)))
        for i, x in enumerate(fresh):
            with cols[i % len(cols)]:
                st.markdown(
                    f"<div class='card'><div class='card-title'>🚀 {x.get('ticker')}</div><div class='muted'>{x.get('company','')}</div><div class='big'>{_money(x.get('cmp'))}</div><div class='muted'>Priority {x.get('priority_score', x.get('score'))} • RVOL {_num(x.get('volume_ratio'),1)}x • RS Δ {_num(x.get('rs_delta'),1)}%</div></div>",
                    unsafe_allow_html=True,
                )
                st.button(
                    "Open Analyst",
                    key=f"fresh_{x.get('ticker')}",
                    use_container_width=True,
                    on_click=_open_stock,
                    args=(x.get("ticker"),),
                )
    else:
        st.info("No fresh breakout phase in the current scan.")

    st.divider()
    st.subheader("📊 Radar buckets")
    # Mutually exclusive buckets prevent the old duplicate-table problem.
    used = set()
    buckets = [
        ("🚀 Actionable", lambda x: x.get("status") == "ENTRY READY", 8),
        ("🟢 Fresh Breakout", lambda x: x.get("phase") == "BREAKOUT", 10),
        ("🔄 Continuation / Pullback", lambda x: x.get("phase") in ("CONTINUATION", "PULLBACK"), 10),
        ("⭐ Building", lambda x: x.get("phase") == "BUILDING", 10),
        ("⚠️ Extended", lambda x: x.get("phase") == "EXTENDED", 8),
    ]
    for title, predicate, limit in buckets:
        bucket = [x for x in results if x.get("ticker") not in used and predicate(x)]
        bucket.sort(key=lambda x: (x.get("priority_score", x.get("score", 0)), x.get("score", 0)), reverse=True)
        bucket = bucket[:limit]
        if not bucket:
            continue
        used.update(x.get("ticker") for x in bucket)
        st.markdown(f"### {title}")
        for start in range(0, len(bucket), 2):
            pair = bucket[start:start+2]
            rowcols = st.columns(2)
            for col, x in zip(rowcols, pair):
                with col:
                    c1, c2, c3, c4 = st.columns([1.8, 1, 1, 1])
                    c1.markdown(f"**{x.get('ticker')}**  \\n{_money(x.get('cmp'))} • {x.get('phase','—')}")
                    c2.metric("Score", x.get("priority_score", x.get("score")))
                    c3.metric("RS Δ", f"{_num(x.get('rs_delta'),1)}%")
                    c4.metric("RVOL", f"{_num(x.get('volume_ratio'),1)}x")
                    st.button(
                        f"🔎 Open {x.get('ticker')} Analyst",
                        key=f"radar_{title}_{x.get('ticker')}",
                        use_container_width=True,
                        on_click=_open_stock,
                        args=(x.get("ticker"),),
                    )

    st.caption("Technical setup and market gating are kept separate. A bearish market can block an entry without hiding a technically interesting setup.")


# ========================= MOMENTUM LAB =========================
elif module == "⚡ Momentum Lab":
    st.subheader("⚡ Momentum Lab")
    st.caption("Multi-timeframe momentum cockpit — see what momentum is doing on 5m, 15m, 30m, 1H, 4H, 1D and 1W.")
    if universe.empty:
        st.warning("NSE universe not available. Run the universe refresh first.")
        st.stop()
    options = [f"{r.SYMBOL} — {r.COMPANY}" for r in universe.itertuples()]
    current = st.session_state.get("momentum_ticker", st.session_state.get("glance_ticker", st.session_state.get("analyst_ticker", "")))
    default_index = next((i for i, o in enumerate(options) if o.startswith(current + " —")), 0)
    selected = st.selectbox("Search any NSE stock", options, index=default_index, key="momentum_select")
    selected_symbol = selected.split(" — ", 1)[0] if selected else ""
    selected_row = universe[universe.SYMBOL == selected_symbol].iloc[0]
    m1, m2 = st.columns([1,4])
    with m1:
        fresh = st.checkbox("Fresh data", value=True, key="momentum_fresh")
    with m2:
        if st.button(f"Analyse {selected_symbol} in Momentum Lab", type="primary", use_container_width=True):
            st.session_state["momentum_ticker"] = selected_symbol
            st.session_state["momentum_run"] = {"yf_symbol": selected_row.YF_SYMBOL, "ticker": selected_symbol, "company": selected_row.COMPANY, "fresh": fresh}
            st.rerun()
    run = st.session_state.get("momentum_run")
    if not run or run.get("ticker") != selected_symbol:
        st.info("Select a stock and Analyse it to open the multi-timeframe Momentum Lab cockpit.")
        st.stop()
    render_momentum_lab(run["yf_symbol"], run["ticker"], run.get("company", ""), fresh=run.get("fresh", True))

# ========================= SWING GLANCE =========================
elif module == "⚡ Swing Glance":
    st.subheader("⚡ Swing Glance")
    st.caption("Institutional-style one-stock cockpit — fast context, flow, levels, risk and decision.")
    if universe.empty:
        st.warning("NSE universe not available. Run the universe refresh first.")
        st.stop()
    options = [f"{r.SYMBOL} — {r.COMPANY}" for r in universe.itertuples()]
    current = st.session_state.get("glance_ticker", st.session_state.get("analyst_ticker", ""))
    default_index = next((i for i, o in enumerate(options) if o.startswith(current + " —")), 0)
    selected = st.selectbox("Search any NSE stock", options, index=default_index, key="glance_select")
    selected_symbol = selected.split(" — ", 1)[0] if selected else ""
    selected_row = universe[universe.SYMBOL == selected_symbol].iloc[0]
    g1, g2 = st.columns([1, 4])
    with g1:
        fresh = st.checkbox("Fresh data", value=True, key="glance_fresh")
    with g2:
        if st.button(f"Analyse {selected_symbol} in Glance", type="primary", use_container_width=True):
            with st.spinner(f"Building Swing Glance for {selected_symbol}..."):
                try:
                    ar = analyze_stock(selected_row.YF_SYMBOL, selected_symbol, selected_row.COMPANY, fresh_data=fresh)
                    if ar is None:
                        st.error("Not enough price history to analyse this stock.")
                    else:
                        st.session_state["glance_result"] = ar
                        st.session_state["glance_ticker"] = selected_symbol
                        st.rerun()
                except Exception as exc:
                    st.error(f"Swing Glance error: {exc}")
    recent = [x for x in results if x.get("phase") == "BREAKOUT"]
    recent = sorted(recent, key=lambda x: x.get("priority_score", x.get("score", 0)), reverse=True)[:8]
    if recent:
        st.markdown("**Recent breakout quick-open**")
        qcols = st.columns(min(4, len(recent)))
        for i, x in enumerate(recent):
            with qcols[i % len(qcols)]:
                if st.button(f"⚡ {x.get('ticker')}", key=f"gl_quick_{x.get('ticker')}", use_container_width=True):
                    st.session_state["glance_ticker"] = x.get("ticker")
                    st.session_state["glance_result"] = None
                    st.rerun()
    gar = st.session_state.get("glance_result")
    if gar and gar.result.get("ticker") != selected_symbol:
        gar = None
    if not gar:
        st.info("Select a stock and Analyse it to open the one-stock Swing Glance cockpit.")
        st.stop()
    render_swing_glance(gar, gar.market)

# ========================= STOCK ANALYST =========================
else:
    st.subheader("🔎 Stock Analyst")
    st.caption("TradingView-style information hierarchy, but every value is calculated from Swing Brain's own engine.")

    if universe.empty:
        st.warning("NSE universe not available. Run the universe refresh first.")
        st.stop()

    options = [f"{r.SYMBOL} — {r.COMPANY}" for r in universe.itertuples()]
    current = st.session_state.get("analyst_ticker", "")
    default_index = next((i for i, o in enumerate(options) if o.startswith(current + " —")), 0)
    selected = st.selectbox("Search any NSE stock", options, index=default_index, placeholder="Search RELIANCE, HAL, AZAD, Tata...")
    selected_symbol = selected.split(" — ", 1)[0] if selected else ""
    selected_row = universe[universe.SYMBOL == selected_symbol].iloc[0]

    recent = [x for x in results if x.get("phase") == "BREAKOUT"]
    recent = sorted(recent, key=lambda x: x.get("priority_score", x.get("score", 0)), reverse=True)[:8]
    if recent:
        st.markdown("**Recent breakout quick-open**")
        qcols = st.columns(min(4, len(recent)))
        for i, x in enumerate(recent):
            with qcols[i % len(qcols)]:
                st.button(
                    f"🚀 {x.get('ticker')}",
                    key=f"quick_{x.get('ticker')}",
                    use_container_width=True,
                    on_click=_open_stock,
                    args=(x.get("ticker"),),
                )

    a1, a2 = st.columns([1, 4])
    with a1:
        fresh = st.checkbox("Fresh data", value=True)
    with a2:
        if st.button(f"Analyse {selected_symbol}", type="primary", use_container_width=True):
            with st.spinner(f"Running Swing Brain analyst for {selected_symbol}..."):
                try:
                    ar = analyze_stock(selected_row.YF_SYMBOL, selected_symbol, selected_row.COMPANY, fresh_data=fresh)
                    if ar is None:
                        st.error("Not enough price history to analyse this stock.")
                    else:
                        st.session_state["analyst_result"] = ar
                        st.session_state["analyst_ticker"] = selected_symbol
                        st.rerun()
                except Exception as exc:
                    st.error(f"Analyst error: {exc}")

    ar = st.session_state.get("analyst_result")
    if ar and ar.result.get("ticker") != selected_symbol:
        ar = None

    if not ar:
        st.info("Select a stock or click a breakout above, then Analyse. The breakout card and radar rows both open this same module.")
        st.stop()

    r = ar.result
    f = ar.fundamentals
    a = r.get("analyst", {})
    life = ar.lifecycle

    # ---------- Phase D analyst cockpit ----------
    def _state_badge(value, tone="neutral"):
        return f"<span class='state-badge {tone}'>{value}</span>"

    def _tone(value):
        v = str(value or "").upper()
        if any(k in v for k in ("FAIL", "FAILED", "BLOCKED", "INVALID", "WEAK", "NO CHASE", "DAMAGED", "DISTRIBUTION")):
            return "bad"
        if any(k in v for k in ("STRONG", "CONFIRMED", "PASS", "LEADING", "FAVORABLE", "DEMAND", "ACCUMULATION", "INTACT")):
            return "good"
        if any(k in v for k in ("WATCH", "PENDING", "BUILDING", "REVIEW", "MIXED", "NEUTRAL", "CANDIDATE", "SUPPORTIVE", "APPROACHING", "PRESSURE")):
            return "warn"
        return "neutral"

    def _evidence_card(title, score, rows, tone="neutral"):
        score_txt = f"{score}/100" if score is not None else "—"
        body = "".join(
            f"<div class='e-row'><span>{label}</span><b>{value}</b></div>"
            for label, value in rows
        )
        st.markdown(
            f"<div class='e-card'><div class='e-head'><span class='e-title'>{title}</span><span class='e-score {tone}'>{score_txt}</span></div>{body}</div>",
            unsafe_allow_html=True,
        )

    # Hero decision strip + integrated KPIs (matches the reference cockpit hierarchy)
    def _hero_kpi(label, value):
        tone = _tone(value) if 'tone' in globals() else _value_tone(value)
        return f"<div class='hero-kpi {tone}'><div class='hero-kpi-label'>{label}</div><div class='hero-kpi-value'>{value}</div></div>"

    hero_kpis = ''.join([
        _hero_kpi('CMP', _money(r.get('cmp'))),
        _hero_kpi('Overall', r.get('score','—')),
        _hero_kpi('Technical', r.get('technical_score','—')),
        _hero_kpi('Fundamental', r.get('fundamental_score','—')),
        _hero_kpi('Setup Phase', r.get('phase','—')),
        _hero_kpi('Radar Status', r.get('status','—')),
    ])
    st.markdown(
        f"""<div class='analyst-hero-wrap'><div class='analyst-hero'>
        <div><div class='analyst-kicker'>SWING TRADE ANALYST</div>
        <div class='analyst-title'>🧠 {r.get('ticker','—')} <span>{r.get('company','')}</span></div>
        <div class='analyst-sub'>Price + Volume + Relative Strength + Structure + Fundamentals — one evidence chain.</div></div>
        <div class='hero-decision'>{_state_badge(a.get('entry_state','WAIT'), _tone(a.get('entry_state')))}<div class='hero-score'>{r.get('entry_quality',0)}/100</div><div class='hero-label'>Entry Score</div></div>
        </div><div class='hero-kpis'>{hero_kpis}</div></div>""",
        unsafe_allow_html=True,
    )

    # Breakout lifecycle
    st.markdown("### 🧬 Breakout lifecycle")
    steps = ["BUILDING", "BREAKOUT TEST", "NEW BREAKOUT", "SUSTAINING", "RETEST / CONFIRMED", "ENTRY"]
    current_state = life.get("state", "BUILDING")
    html = "<div class='life'>"
    for step in steps:
        active = "active" if step == current_state else ""
        html += f"<span class='{active}'>{step}</span>"
    html += "<span>→ TARGET</span>"
    if life.get("failed"):
        html += "<span class='active bad-life'>FAILED</span>"
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)

    l1, l2, l3, l4, l5 = st.columns(5)
    l1.metric("Breakout", life.get("state", "—"))
    l2.metric("Breakout Date", life.get("breakout_date", "—"))
    l3.metric("Age", f"{life.get('age_days')} bars" if life.get("age_days") is not None else "—")
    l4.metric("Breakout Level", _money(life.get("level")))
    l5.metric("Distance", f"{_num(life.get('distance_pct'),1)}%")

    # Current setup + chart cockpit
    st.markdown("### 🎯 Current setup")
    setup_col, analyst_col = st.columns([1.65, 1.0])
    with setup_col:
        st.markdown("<div class='setup-card'><div class='card-title'>🎯 Current setup</div><div class='setup-grid'>" +
            ''.join([
                f"<div class='setup-item'><div class='setup-item-label'>{label}</div><div class='setup-item-value tone-{_tone(value)}'>{value}</div></div>"
                for label, value in [
                    ('Trend', a.get('trend','—')),
                    ('Setup Phase', r.get('phase','—')),
                    ('Breakout', life.get('state','—')),
                    ('Volume', a.get('price_volume','—')),
                    ('Entry State', a.get('entry_state','—')),
                ]
            ]) + "</div>" +
            f"<div class='metric-line' style='margin-top:8px'><span>Planned Entry</span><b>{_money(r.get('entry_high'))}</b></div></div>",
            unsafe_allow_html=True,
        )
    with analyst_col:
        st.markdown("<div class='quick-card'><div class='card-title'>📊 Quick stats</div>" +
            ''.join([
                f"<div class='quick-row'><span class='quick-label'>{label}</span><span class='quick-value tone-{_tone(value)}'>{value}</span></div>"
                for label, value in [
                    ('Structure', f"{a.get('structure_label','—')} {a.get('structure_score',0)}/100"),
                    ('Price / Vol', a.get('price_volume','—')),
                    ('Swing Context', a.get('swing_context','—')),
                    ('Resistance', _money(r.get('resistance'))),
                    ('Distance', f"{_num(a.get('distance_to_resistance_pct'),1)}%"),
                    ('Pace', a.get('pace','—')),
                    ('R/R Response', 'STRONG' if float(r.get('rr',0) or 0) >= 2 else 'WEAK'),
                    ('Market Gate', a.get('market_gate','—')),
                ]
            ]) + "</div>",
            unsafe_allow_html=True,
        )

    chart_col, quick2 = st.columns([3.2, 1.0])
    with chart_col:
        st.markdown("<div class='chart-card'><div class='card-title'>📈 PRICE / VOLUME / STRUCTURE</div>", unsafe_allow_html=True)
        d = ar.history.tail(180).copy()
        chart = pd.DataFrame({"Close": d.Close, "EMA20": d.EMA20, "EMA50": d.EMA50, "EMA200": d.EMA200}).dropna(how="all")
        st.line_chart(chart, height=300, use_container_width=True)
        st.caption(f"Resistance {_money(r.get('resistance'))} • Support {_money(r.get('support'))} • RSI {_num(r.get('rsi'),1)} • RVOL {_num(r.get('volume_ratio'),1)}x")
        st.markdown("</div>", unsafe_allow_html=True)
    with quick2:
        st.markdown("<div class='quick-card'><div class='card-title'>⚡ Key levels</div>" +
            ''.join([
                f"<div class='quick-row'><span class='quick-label'>{label}</span><span class='quick-value tone-{_tone(value)}'>{value}</span></div>"
                for label, value in [
                    ('Resistance', _money(r.get('resistance'))),
                    ('Support', _money(r.get('support'))),
                    ('RSI', _num(r.get('rsi'),1)),
                    ('RVOL', f"{_num(r.get('volume_ratio'),1)}x"),
                    ('Leadership', a.get('rs_state','—')),
                    ('Weekly', a.get('weekly_state','—')),
                    ('Thesis', a.get('thesis','—')),
                ]
            ]) + "</div>", unsafe_allow_html=True)

    # Compact bottom strip from the reference UI
    st.markdown("<div class='metric-strip'>", unsafe_allow_html=True)
    panels = [
        ('stop', '🛡️ Stop / Target', [
            ('Stop / Invalid', _money(r.get('stop_loss'))), ('Target', _money(r.get('target1'))),
            ('R:R', f"{_num(r.get('rr'),1)}:1"), ('Risk', a.get('risk_gate','—'))]),
        ('levels', '📊 Key levels & metrics', [
            ('Resistance', _money(r.get('resistance'))), ('Support', _money(r.get('support'))),
            ('RSI', _num(r.get('rsi'),1)), ('RVOL', f"{_num(r.get('volume_ratio'),1)}x")]),
        ('sector', '🏛️ Sector & leadership', [
            ('Sector', r.get('sector','Unknown')), ('Leadership', a.get('rs_state','—')),
            ('Weekly', a.get('weekly_state','—')), ('Thesis', a.get('thesis','—'))]),
        ('timing', '🕐 Timing', [('Timing Score', f"{a.get('timing_score',0)}/100"), ('Entry State', a.get('entry_state','—'))]),
    ]
    for cls, title, rows in panels:
        body = ''.join(f"<div class='metric-line'><span>{k}</span><b class='tone-{_tone(v)}'>{v}</b></div>" for k,v in rows)
        st.markdown(f"<div class='metric-panel {cls}'><div class='metric-panel-title'>{title}</div>{body}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Decision / evidence hierarchy
    st.markdown("### 🧭 Decision map")
    d1, d2, d3, d4, d5 = st.columns(5)
    decision_items = [
        ("SETUP", r.get("phase", "—")),
        ("BREAKOUT", life.get("state", "—")),
        ("EVIDENCE", f"{a.get('evidence_score',0)}/100"),
        ("ENTRY", a.get("entry_state", "—")),
        ("GATE", a.get("market_gate", "—")),
    ]
    for col, (label, value) in zip((d1,d2,d3,d4,d5), decision_items):
        with col:
            st.markdown(f"<div class='decision-box'><div>{label}</div><b>{value}</b></div>", unsafe_allow_html=True)

    st.markdown("### 🔬 Evidence engine")
    st.caption("Quality scores describe the underlying setup; evidence scores describe how strongly that evidence contributes to the current decision.")
    ev1, ev2, ev3 = st.columns(3)
    with ev1:
        _evidence_card("Structure", a.get("swing_score"), [
            ("Swing state", a.get("swing_state", "—")),
            ("Higher High", "YES" if a.get("higher_high") else "NO"),
            ("Higher Low", "YES" if a.get("higher_low") else "NO"),
            ("Structure quality", f"{a.get('structure_label','—')} {a.get('structure_score',0)}/100"),
        ], _tone(a.get("swing_state")))
        _evidence_card("Price + Volume", None, [
            ("Price response", a.get("price_response_state", "—")),
            ("Volume state", a.get("volume_state", "—")),
            ("PV score", f"{a.get('pv_score','—')}/20"),
            ("Participation", a.get("participation_state", "—")),
        ], _tone(a.get("volume_state")))
    with ev2:
        _evidence_card("Resistance Pressure", None, [
            ("Pressure", a.get("pressure_state", "—")),
            ("Pressure score", f"{a.get('pressure_score','—')}/20"),
            ("Resistance tests", a.get("resistance_tests", "—")),
            ("Distance", f"{_num(a.get('distance_to_resistance_pct'),1)}%"),
        ], _tone(a.get("pressure_state")))
        _evidence_card("Swing Context", a.get("swing_context_score"), [
            ("Weekly", a.get("weekly_state", "—")),
            ("Daily", a.get("daily_state", "—")),
            ("Compression", a.get("compression_state", "—")),
            ("RS context", a.get("rs_context_state", "—")),
        ], _tone(a.get("swing_context")))
    with ev3:
        conf = a.get("confirmation", {}) or {}
        _evidence_card("Breakout Confirmation", None, [
            ("State", a.get("confirmation_state", "—")),
            ("Score", f"{a.get('confirmation_score','—')}/6"),
            ("Strong candle", "YES" if conf.get("strong_candle") else "NO"),
            ("Volume confirmed", "YES" if conf.get("volume_confirmed") else "NO"),
            ("Acceptance", "YES" if conf.get("acceptance") else "NO"),
            ("Retest", "YES" if conf.get("retest") else "NO"),
        ], _tone(a.get("confirmation_state")))
        _evidence_card("Daily Swing", None, [
            ("Swing state", a.get("daily_swing_state", "—")),
            ("Participation", a.get("participation_state", "—")),
            ("Pace", a.get("pace", "—")),
            ("RS", a.get("rs_context_state", "—")),
        ], _tone(a.get("daily_swing_state")))

    # Thesis / hold management
    st.markdown("### 🧠 Thesis & hold management")
    t1, t2, t3, t4, t5 = st.columns(5)
    thesis_items = [
        ("Thesis", a.get("persistent_state", a.get("thesis", "—"))),
        ("Setup Age", f"{a.get('setup_age_days',0)} bars"),
        ("Weak Days", a.get("weak_days", 0)),
        ("Quality Memory", a.get("quality_memory", "—")),
        ("Hold Action", a.get("hold_action", "—")),
    ]
    for col, (label, value) in zip((t1,t2,t3,t4,t5), thesis_items):
        with col:
            st.markdown(f"<div class='thesis-box'><div>{label}</div><b>{value}</b></div>", unsafe_allow_html=True)

    # Entry readiness gate
    st.markdown("### 🧱 Entry readiness gate")
    g1, g2, g3, g4 = st.columns(4)
    g1.metric("Entry", a.get("entry_state", "WAIT"))
    g2.metric("R:R Gate", a.get("risk_gate", "FAIL"))
    g3.metric("Market Gate", a.get("market_gate", "OPEN"))
    g4.metric("Timing", f"{a.get('timing_score',0)}/100")
    if a.get("blockers"):
        st.warning("Blockers: " + " • ".join(a["blockers"]))
    else:
        st.success("No major analyst blocker detected by the current rule set.")

    # Conviction strip
    st.markdown("### 🧠 Stock conviction")
    conviction_items = [
        ("Stock RS", f"{_num(r.get('rs_stock'),1)}%"),
        ("RS Δ", f"{_num(r.get('rs_delta'),1)}%"),
        ("RS State", a.get("rs_state", "—")),
        ("BO", life.get("state", "—")),
        ("RVOL", f"{_num(r.get('volume_ratio'),1)}x"),
        ("Volume", a.get("volume_state", "—")),
        ("Tech", f"{r.get('technical_score',0)}/100"),
        ("Fund", a.get("fundamental_data", "—")),
        ("Growth", a.get("growth", "—")),
        ("Valuation", a.get("valuation", "—")),
    ]
    conv = st.columns(5)
    for i, (label, value) in enumerate(conviction_items):
        with conv[i % 5]:
            st.markdown(f"<div class='conv-card'><div>{label}</div><b>{value}</b></div>", unsafe_allow_html=True)

    # Fundamentals and evidence summary
    st.markdown("### 📚 Fundamentals")
    frows = {
        "Revenue growth %": f.get("revenue_growth_pct"),
        "Earnings growth %": f.get("earnings_growth_pct"),
        "EPS": f.get("eps"), "P/E": f.get("pe"), "PEG": f.get("peg"),
        "ROE %": f.get("roe_pct"), "Debt / Equity": f.get("debt_to_equity"),
        "Profit margin %": f.get("profit_margin_pct"), "Sector": f.get("sector"), "Industry": f.get("industry"),
    }
    st.dataframe(pd.DataFrame([frows]), use_container_width=True, hide_index=True)

    with st.expander("Why / Risks / Evidence", expanded=False):
        e1, e2 = st.columns(2)
        with e1:
            st.markdown("**Why**")
            for item in r.get("reasons", []) or ["No major positive trigger yet."]:
                st.write("• " + item)
        with e2:
            st.markdown("**Risks**")
            for item in r.get("risks", []) or ["No major risk flag."]:
                st.write("• " + item)
        st.caption(f"Evidence health: {a.get('evidence_score',0)}/100 • Data bar: {r.get('data_last_bar','—')} • Market: {ar.market.get('regime','UNKNOWN')} • Benchmark: {BENCHMARK}")
