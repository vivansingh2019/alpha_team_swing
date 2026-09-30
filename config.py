import os
from dotenv import load_dotenv

load_dotenv()

BENCHMARK = os.getenv("BENCHMARK", "^NSEI")
MIN_HISTORY = int(os.getenv("MIN_HISTORY", "220"))
ATR_PERIOD = int(os.getenv("ATR_PERIOD", "14"))
RS_PERIOD = int(os.getenv("RS_PERIOD", "63"))
VOLUME_LOOKBACK = int(os.getenv("VOLUME_LOOKBACK", "20"))
RISK_PER_TRADE = float(os.getenv("RISK_PER_TRADE", "0.01"))
MIN_RR = float(os.getenv("MIN_RR", "2.0"))
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "8"))
MAX_STOCKS = int(os.getenv("MAX_STOCKS", "0"))
FUNDAMENTAL_TOP_N = int(os.getenv("FUNDAMENTAL_TOP_N", "150"))
