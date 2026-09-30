# Swing Brain v3

A retail-focused NSE swing research terminal for 1–3 week setups.

## Design principle
Broad discovery, simple ranking, limited actionable ideas. The engine does **not** require every indicator to pass. Market weakness is a risk modifier, not a blanket rejection.

## What it does
- Scans the NSE EQ universe.
- Finds BUILDING, BREAKOUT, PULLBACK, CONTINUATION and EXTENDED phases.
- Separates stock quality from current entry quality.
- Uses technical evidence first: trend, relative strength, volume, structure, support/resistance and simple SMC-style confluence.
- Adds a lightweight fundamental score using earnings growth, revenue growth, EPS, P/E/PEG, ROE, debt/equity and margin where Yahoo Finance data is available.
- Ranks a small actionable list rather than forcing all stocks through hard gates.
- Provides manual stock search with symbol/company autocomplete and automatic fundamental + technical analysis.
- Shows entry zone, stop, targets, R:R, continuation and profit-booking risk.

## Run on Windows
```bat
py -m venv .venv
.venv\Scripts\activate
py -m pip install -r requirements.txt
py scanner.py
streamlit run dashboard.py
```

For a first test:
```bat
set MAX_STOCKS=50
py scanner.py
streamlit run dashboard.py
```

For the full universe:
```bat
set MAX_STOCKS=0
py scanner.py
```

## Fundamental data
Fundamentals are cached in `fundamental_cache`. The scanner enriches the top technical candidates first so a full 2,000+ stock scan does not make hundreds of slow fundamental requests every run. Manual search fetches fundamentals for the selected stock.

## Important
Research/paper-trading only. No broker orders are placed. Backtest and paper-trade before risking capital. Data availability and Yahoo Finance fields can vary.
