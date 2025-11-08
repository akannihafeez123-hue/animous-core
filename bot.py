#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
COMPLETE Institutional AI Trading Bot
All features integrated: SMC, Volume Profile, Wyckoff, Order Flow, Market Microstructure
Full ML pipeline with RandomForest + optional Keras LSTM
Live Bitget trading with TP/SL
Webhook mode for cloud deployment
"""

import os
import time
import math
import json
import csv
import hmac
import hashlib
import base64
import logging
import threading
from datetime import datetime
from functools import wraps, lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from collections import deque

# Load environment
from dotenv import load_dotenv
load_dotenv()

# Telegram & Web
import telebot
from flask import Flask, request, jsonify

# Data / Math
import numpy as np
import pandas as pd
from scipy import stats

# ML
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Optional Keras
try:
    from tensorflow.keras.models import Sequential, load_model as keras_load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

# Trading
try:
    import ccxt
    CCXT_AVAILABLE = True
except Exception:
    CCXT_AVAILABLE = False
    print("WARNING: ccxt not installed. Install with: pip install ccxt")

# ============================================
# CONFIGURATION
# ============================================

# Telegram
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_ADMIN_ID = int(os.getenv("TELEGRAM_ADMIN_ID", "0"))
ADMIN_SECRET = os.getenv("ADMIN_SECRET", "")

# Bitget
BITGET_API_KEY = os.getenv("BITGET_API_KEY", "")
BITGET_SECRET = os.getenv("BITGET_SECRET", "")
BITGET_PASSWORD = os.getenv("BITGET_PASSWORD", "")
BITGET_TESTNET = os.getenv("BITGET_TESTNET", "true").lower() in ["1", "true", "yes"]
BITGET_REST_BASE = os.getenv("BITGET_REST_BASE", "https://api.bitget.com")

# Trading
DEFAULT_SYMBOL = os.getenv("DEFAULT_SYMBOL", "BTC/USDT")
ENABLE_LIVE_TRADES = os.getenv("ENABLE_LIVE_TRADES", "false").lower() in ["1", "true", "yes"]
DRY_RUN = os.getenv("DRY_RUN", "true").lower() != "false"
TP_MULTIPLIER = float(os.getenv("TP_MULTIPLIER", "2.0"))
SL_MULTIPLIER = float(os.getenv("SL_MULTIPLIER", "1.0"))
POSITION_SIZE_USD = float(os.getenv("POSITION_SIZE_USD", "50"))
MAX_POSITION_PERCENT = float(os.getenv("MAX_POSITION_PERCENT", "0.02"))
ORDER_SIZE = float(os.getenv("ORDER_SIZE", "0.001"))

# Model paths
MODEL_DIR = Path(os.getenv("MODEL_DIR", "/tmp/po_models"))
MODEL_DIR.mkdir(parents=True, exist_ok=True)
RF_MODEL_FILE = MODEL_DIR / "rf_model.joblib"
KERAS_MODEL_FILE = MODEL_DIR / "keras_model.h5"
DATA_FILE = MODEL_DIR / "training_data.csv"
TRADE_LOG_CSV = os.getenv("TRADE_LOG_CSV", "trade_log.csv")

# Symbols & Strategy
PO_SYMBOLS = os.getenv("PO_SYMBOLS", "BTCUSDT,ETHUSDT").split(",")
RANDOM_SEED = int(os.getenv("RANDOM_SEED", 42))

# Alignment thresholds
STRATEGY_ALIGN_THRESHOLD = float(os.getenv("STRATEGY_ALIGN_THRESHOLD", "0.8"))
INDICATOR_ALIGN_THRESHOLD = float(os.getenv("INDICATOR_ALIGN_THRESHOLD", "0.8"))
TIMEFRAME_AGREE_THRESHOLD = float(os.getenv("TIMEFRAME_AGREE_THRESHOLD", "0.8"))
REQUIRED_STRATEGY_AGREE = int(os.getenv("REQUIRED_STRATEGY_AGREE", "3"))
REQUIRED_INDICATOR_AGREE = int(os.getenv("REQUIRED_INDICATOR_AGREE", "3"))
REQUIRED_TIMEFRAME_AGREE = int(os.getenv("REQUIRED_TIMEFRAME_AGREE", "3"))
MIN_CONFIDENCE_TO_SIGNAL = float(os.getenv("MIN_CONFIDENCE_TO_SIGNAL", "0.8"))

# Monitors
AUTO_MONITOR_POLL = float(os.getenv("AUTO_MONITOR_POLL", "5.0"))
MAX_CONCURRENT_MONITORS = int(os.getenv("MAX_CONCURRENT_MONITORS", "8"))

# Institutional settings
INST_LIQUIDITY_LOOKBACK = int(os.getenv("INST_LIQUIDITY_LOOKBACK", "50"))
INST_SWING_PERIOD = int(os.getenv("INST_SWING_PERIOD", "20"))
INST_VOLUME_PROFILE_BINS = int(os.getenv("INST_VOLUME_PROFILE_BINS", "50"))
INST_MIN_SCORE = float(os.getenv("INST_MIN_SCORE", "0.75"))

# Keras
KERAS_WINDOW = int(os.getenv("KERAS_WINDOW", 32))
KERAS_EPOCHS = int(os.getenv("KERAS_EPOCHS", 6))
KERAS_BATCH = int(os.getenv("KERAS_BATCH", 32))

# Webhook
WEBHOOK_URL = os.getenv("WEBHOOK_URL", "").strip()
PORT = int(os.getenv("PORT", "8080"))

# Timeframes
SCALP_TFS = ["1m", "5m", "15m", "1h"]
SWING_TFS = ["4h", "1d", "1w", "1M"]
TIMEFRAMES_TOP_DOWN = ["1y", "6mo", "1mo", "1w", "1d", "4h", "1h", "15m", "5m", "1m"]

_INTERVAL_TO_GRAN = {
    "1m": "60", "5m": "300", "15m": "900", "30m": "1800",
    "1h": "3600", "4h": "14400", "1d": "86400"
}

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger("institutional-bot")

# ============================================
# GLOBALS
# ============================================

bot = telebot.TeleBot(TELEGRAM_TOKEN, threaded=False) if TELEGRAM_TOKEN else None
app = Flask(__name__)

_rf_model = None
_keras_model = None
_model_lock = threading.Lock()
_current_strategy = "quantum"
_STRATEGIES = ["quantum", "momentum", "breakout", "meanreversion", "smart_money", "volume_profile", "wyckoff", "market_maker", "institutional_flow"]
_active_monitors: Dict[str, Dict[str, Any]] = {}

# Bitget client
bitget_client = None
if CCXT_AVAILABLE and BITGET_API_KEY:
    try:
        params = {
            'apiKey': BITGET_API_KEY,
            'secret': BITGET_SECRET,
            'password': BITGET_PASSWORD,
            'enableRateLimit': True,
        }
        if BITGET_TESTNET:
            params['urls'] = {
                'api': {
                    'public': 'https://api-testnet.bitget.com',
                    'private': 'https://api-testnet.bitget.com'
                }
            }
        bitget_client = ccxt.bitget(params)
        logger.info("Bitget client initialized")
    except Exception as e:
        logger.error(f"Bitget init error: {e}")

logger.info(f"INIT: TF={TF_AVAILABLE} TELEGRAM={'✓' if TELEGRAM_TOKEN else '✗'} BITGET={'✓' if bitget_client else '✗'} DRY_RUN={DRY_RUN}")

# ============================================
# BITGET HELPERS (Original Implementation)
# ============================================

def _bitget_sign(method: str, request_path: str, body: str = "", timestamp: str = None) -> Dict[str, str]:
    if timestamp is None:
        timestamp = str(int(time.time() * 1000))
    prehash = timestamp + method.upper() + request_path + (body or "")
    h = hmac.new(BITGET_SECRET.encode() if BITGET_SECRET else b"", prehash.encode(), hashlib.sha256)
    sign = base64.b64encode(h.digest()).decode() if BITGET_SECRET else ""
    return {
        "ACCESS-KEY": BITGET_API_KEY,
        "ACCESS-SIGN": sign,
        "ACCESS-TIMESTAMP": timestamp,
        "ACCESS-PASSPHRASE": BITGET_PASSWORD,
        "Content-Type": "application/json"
    }

def bitget_public_get(path: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
    import requests
    url = BITGET_REST_BASE + path
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.error(f"bitget_public_get error: {e}")
        return {}

def bitget_ticker(symbol: str) -> float:
    sym = symbol.replace("-", "").replace("/", "").upper()
    resp = bitget_public_get("/api/spot/v1/market/ticker", params={"symbol": sym})
    try:
        data = resp.get("data") or resp.get("data", {})
        if isinstance(data, dict) and "last" in data:
            return float(data["last"])
        if isinstance(data, list) and len(data) and "last" in data[0]:
            return float(data[0]["last"])
        if "last" in resp:
            return float(resp["last"])
    except Exception:
        pass
    return 0.0

def bitget_klines(symbol: str, granularity: str = "300", limit: int = 500) -> pd.DataFrame:
    sym = symbol.replace("-", "").replace("/", "").upper()
    params = {"symbol": sym, "granularity": str(granularity), "limit": limit}
    resp = bitget_public_get("/api/spot/v1/market/history/kline", params=params)
    try:
        data = resp.get("data") or []
        rows = []
        for item in data:
            if isinstance(item, dict):
                ts = item.get("timestamp") or item.get("id") or item.get("time")
                o, h, l, c, v = float(item.get("open", 0)), float(item.get("high", 0)), float(item.get("low", 0)), float(item.get("close", 0)), float(item.get("volume", 0))
            elif isinstance(item, (list, tuple)) and len(item) >= 6:
                ts = int(item[0])
                o, h, l, c, v = float(item[1]), float(item[2]), float(item[3]), float(item[4]), float(item[5])
            else:
                continue
            if ts and ts < 1e12:
                ts = int(ts) * 1000
            rows.append({"datetime": pd.to_datetime(ts, unit="ms"), "Open": o, "High": h, "Low": l, "Close": c, "Volume": v})
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows).set_index("datetime").sort_index()
        return df
    except Exception as e:
        logger.error(f"bitget_klines parse error: {e}")
        return pd.DataFrame()

@lru_cache(maxsize=512)
def cached_bitget_candles(symbol: str, interval: str = "1m", limit: int = 500) -> pd.DataFrame:
    gran = _INTERVAL_TO_GRAN.get(interval, "300")
    return bitget_klines(symbol, granularity=gran, limit=limit)

def fetch_market_candles(symbol: str, interval: str = "1m", limit: int = 500) -> pd.DataFrame:
    """Unified candle fetcher - tries ccxt first, falls back to Bitget API"""
    if CCXT_AVAILABLE and bitget_client:
        try:
            ohlcv = bitget_client.fetch_ohlcv(symbol, timeframe=interval, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
            df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')
            df.set_index('Timestamp', inplace=True)
            return df
        except Exception as e:
            logger.warning(f"ccxt fetch failed, trying direct API: {e}")
    return cached_bitget_candles(symbol, interval, limit)

# ============================================
# UTILITY FUNCTIONS
# ============================================

def is_admin_chat(chat_id, username=None) -> bool:
    try:
        cid = str(chat_id)
        candidates = {str(TELEGRAM_ADMIN_ID)}
        if cid in candidates:
            return True
        if username and username in candidates:
            return True
    except Exception:
        pass
    return False

def admin_only(handler):
    @wraps(handler)
    def wrapper(message, *args, **kwargs):
        try:
            user_id = message.from_user.id
        except Exception:
            user_id = None
        if user_id != TELEGRAM_ADMIN_ID:
            if bot:
                bot.send_message(message.chat.id, "⛔ Admin-only command")
            return
        return handler(message, *args, **kwargs)
    return wrapper

def safe_send(chat_id, text):
    """Fixed safe_send - no Markdown to avoid parse errors"""
    try:
        if bot:
            text = str(text)[:4096]  # Telegram limit
            bot.send_message(chat_id, text, parse_mode=None)  # No formatting
            return True
    except Exception as e:
        logger.error(f"safe_send error: {e}")
        # Fallback: try chunking
        try:
            for i in range(0, min(len(text), 20000), 4000):
                bot.send_message(chat_id, text[i:i+4000], parse_mode=None)
            return True
        except Exception as e2:
            logger.error(f"safe_send chunk error: {e2}")
    return False

def write_trade_log(row: Dict):
    header = ["timestamp", "symbol", "side", "amount", "entry_price", "tp_price", "sl_price", "order_id", "status", "notes"]
    exists = os.path.exists(TRADE_LOG_CSV)
    with open(TRADE_LOG_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not exists:
            writer.writeheader()
        writer.writerow(row)

# ============================================
# INDICATORS (Complete Original Set)
# ============================================

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    roll_up = up.ewm(com=length - 1, adjust=False).mean()
    roll_down = down.ewm(com=length - 1, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-9)
    return 100 - (100 / (1 + rs))

def macd(series: pd.Series, fast=12, slow=26, signal=9):
    fast_ema = ema(series, fast)
    slow_ema = ema(series, slow)
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def bollinger_bands(series: pd.Series, window=20, n_std=2):
    ma = series.rolling(window).mean()
    std = series.rolling(window).std()
    upper = ma + n_std * std
    lower = ma - n_std * std
    return upper, lower, ma

def atr(df: pd.DataFrame, window=14):
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window).mean()

def detect_order_blocks(df: pd.DataFrame, vol_multiplier=2.0) -> pd.Series:
    vol_ma = df["Volume"].rolling(20).mean().fillna(method="bfill")
    vol_spike = df["Volume"] > vol_ma * vol_multiplier
    bullish_engulf = (df["Close"] > df["Open"]) & ((df["Close"] - df["Open"]) > (df["Open"].shift() - df["Close"].shift()) * 0.8)
    bearish_engulf = (df["Close"] < df["Open"]) & ((df["Open"] - df["Close"]) > (df["Close"].shift() - df["Open"].shift()) * 0.8)
    ob = vol_spike & (bullish_engulf | bearish_engulf)
    return ob.astype(int)

def detect_fvg(df: pd.DataFrame) -> pd.Series:
    fvg = pd.Series(0, index=df.index)
    for i in range(2, len(df)):
        b1, b3 = df.iloc[i - 2], df.iloc[i]
        if (b1["High"] < b3["Low"]) or (b1["Low"] > b3["High"]):
            fvg.iloc[i] = 1
    return fvg

def detect_bos(df: pd.DataFrame, lookback=20) -> pd.Series:
    bos = pd.Series(0, index=df.index)
    highs, lows = df["High"], df["Low"]
    for i in range(lookback, len(df)):
        prior_high = highs[i - lookback:i].max()
        prior_low = lows[i - lookback:i].min()
        if df["Close"].iloc[i] > prior_high:
            bos.iloc[i] = 1
        elif df["Close"].iloc[i] < prior_low:
            bos.iloc[i] = -1
    return bos

def supertrend(df: pd.DataFrame, period=10, multiplier=3.0) -> pd.Series:
    atr_series = atr(df, window=period)
    hl2 = (df["High"] + df["Low"]) / 2
    upperband = hl2 + (multiplier * atr_series)
    lowerband = hl2 - (multiplier * atr_series)
    st = pd.Series(index=df.index)
    for i in range(len(df)):
        if i == 0:
            st.iloc[i] = upperband.iloc[i]
        elif df["Close"].iloc[i] > st.iloc[i - 1]:
            st.iloc[i] = lowerband.iloc[i]
        else:
            st.iloc[i] = upperband.iloc[i]
    return st

def detect_volume_spikes(df: pd.DataFrame, mult=2.0) -> pd.Series:
    vol_ma = df["Volume"].rolling(20).mean().fillna(method="bfill")
    return (df["Volume"] > vol_ma * mult).astype(int)

def detect_support_resistance(df: pd.DataFrame, window=20):
    sr_high = df["High"].rolling(window).max()
    sr_low = df["Low"].rolling(window).min()
    return sr_high, sr_low

def ema_crossover(df: pd.DataFrame, fast=8, slow=21) -> pd.Series:
    e_fast = ema(df["Close"], fast)
    e_slow = ema(df["Close"], slow)
    cross = ((e_fast > e_slow) & (e_fast.shift() <= e_slow.shift())).astype(int) - ((e_fast < e_slow) & (e_fast.shift() >= e_slow.shift())).astype(int)
    return cross

def detect_bollinger_breakout(df: pd.DataFrame) -> pd.Series:
    upper, lower, ma = bollinger_bands(df["Close"], window=20, n_std=2)
    breakout = (df["Close"] > upper).astype(int) - (df["Close"] < lower).astype(int)
    return breakout

def detect_rsi_signals(series: pd.Series, overbought=70, oversold=30) -> pd.Series:
    r = rsi(series, length=14)
    return ((r < oversold).astype(int) - (r > overbought).astype(int))

def vwap(df: pd.DataFrame, window=None):
    pv = df["Close"] * df["Volume"]
    if window:
        vp = pv.rolling(window).sum()
        vv = df["Volume"].rolling(window).sum()
        return (vp / (vv + 1e-9)).fillna(method="bfill")
    return (pv.cumsum() / (df["Volume"].cumsum() + 1e-9)).fillna(method="bfill")

def vwma(df: pd.DataFrame, length=20):
    pv = df["Close"] * df["Volume"]
    return pv.rolling(length).sum() / (df["Volume"].rolling(length).sum() + 1e-9)

def session_volume_profile(df: pd.DataFrame, bins=20):
    try:
        hist_vals, edges = np.histogram(df["Close"], bins=bins, weights=df["Volume"])
        max_idx = int(np.argmax(hist_vals))
        poc = 0.5 * (edges[max_idx] + edges[max_idx + 1])
        vah, val = edges[max_idx + 1], edges[max_idx]
        return {"poc": poc, "vah": vah, "val": val}
    except Exception:
        return {"poc": None, "vah": None, "val": None}

def order_flow_delta_proxy(df: pd.DataFrame, window=10):
    delta = (df["Close"] - df["Open"]) * df["Volume"]
    return delta.rolling(window).sum().fillna(0)

def block_trade_proxy(df: pd.DataFrame, threshold_mult=5.0, window=50):
    med = df["Volume"].rolling(window).median().fillna(method="bfill")
    return (df["Volume"] > med * threshold_mult).astype(int)

def imbalance_sweep(df: pd.DataFrame, wick_factor=1.5):
    upper_wick = df["High"] - df[["Close", "Open"]].max(axis=1)
    lower_wick = df[["Close", "Open"]].min(axis=1) - df["Low"]
    body = (df["Close"] - df["Open"]).abs()
    return ((upper_wick > body * wick_factor) | (lower_wick > body * wick_factor)).astype(int)

# ============================================
# INSTITUTIONAL INDICATORS (All Functions)
# ============================================

def detect_liquidity_pools(df: pd.DataFrame, lookback: int = INST_LIQUIDITY_LOOKBACK) -> pd.DataFrame:
    result = pd.DataFrame(index=df.index)
    highs = df['High'].rolling(lookback).max()
    lows = df['Low'].rolling(lookback).min()
    high_tests = (df['High'] >= highs * 0.998).rolling(5).sum()
    low_tests = (df['Low'] <= lows * 1.002).rolling(5).sum()
    result['liquidity_above'] = (high_tests >= 2).astype(int)
    result['liquidity_below'] = (low_tests >= 2).astype(int)
    result['liquidity_level_high'] = highs
    result['liquidity_level_low'] = lows
    return result

def detect_market_structure_shift(df: pd.DataFrame, swing_period: int = INST_SWING_PERIOD) -> pd.Series:
    mss = pd.Series(0, index=df.index)
    swing_highs = df['High'].rolling(swing_period, center=True).max()
    swing_lows = df['Low'].rolling(swing_period, center=True).min()
    for i in range(swing_period * 2, len(df)):
        recent_high = swing_highs[i-swing_period:i].max()
        if df['Close'].iloc[i] > recent_high:
            mss.iloc[i] = 1
        recent_low = swing_lows[i-swing_period:i].min()
        if df['Close'].iloc[i] < recent_low:
            mss.iloc[i] = -1
    return mss

def detect_order_flow_imbalance(df: pd.DataFrame) -> pd.Series:
    buy_volume = np.where(df['Close'] > df['Open'], df['Volume'], 0)
    sell_volume = np.where(df['Close'] < df['Open'], df['Volume'], 0)
    cum_delta = pd.Series(buy_volume - sell_volume, index=df.index).cumsum()
    ofi = pd.Series(np.tanh(cum_delta / (df['Volume'].rolling(50).sum() + 1e-9)), index=df.index)
    return ofi

def detect_institutional_candles(df: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame(index=df.index)
    body = (df['Close'] - df['Open']).abs()
    upper_wick = df['High'] - df[['Close', 'Open']].max(axis=1)
    lower_wick = df[['Close', 'Open']].min(axis=1) - df['Low']
    
    bullish_engulf = (
        (df['Close'] > df['Open']) &
        (df['Close'].shift() < df['Open'].shift()) &
        (df['Close'] > df['Open'].shift()) &
        (df['Open'] < df['Close'].shift()) &
        (df['Volume'] > df['Volume'].rolling(20).mean() * 1.5)
    )
    bearish_engulf = (
        (df['Close'] < df['Open']) &
        (df['Close'].shift() > df['Open'].shift()) &
        (df['Close'] < df['Open'].shift()) &
        (df['Open'] > df['Close'].shift()) &
        (df['Volume'] > df['Volume'].rolling(20).mean() * 1.5)
    )
    bullish_rejection = (lower_wick > body * 2) & (df['Close'] > df['Open'])
    bearish_rejection = (upper_wick > body * 2) & (df['Close'] < df['Open'])
    
    result['bullish_engulf'] = bullish_engulf.astype(int)
    result['bearish_engulf'] = bearish_engulf.astype(int)
    result['bullish_rejection'] = bullish_rejection.astype(int)
    result['bearish_rejection'] = bearish_rejection.astype(int)
    return result

def calculate_volume_profile(df: pd.DataFrame, bins: int = INST_VOLUME_PROFILE_BINS) -> Dict:
    try:
        price_range = df['High'].max() - df['Low'].min()
        if price_range == 0 or np.isnan(price_range):
            return {'poc': None, 'vah': None, 'val': None}
        bin_size = price_range / bins
        volume_profile = {}
        for i in range(bins):
            price_level = df['Low'].min() + (i * bin_size)
            volume_at_level = df[(df['Low'] <= price_level) & (df['High'] >= price_level)]['Volume'].sum()
            volume_profile[price_level] = volume_at_level
        poc_price = max(volume_profile, key=volume_profile.get)
        total_volume = sum(volume_profile.values())
        target_volume = total_volume * 0.70
        sorted_levels = sorted(volume_profile.items(), key=lambda x: x[1], reverse=True)
        value_area_volume = 0
        value_area_prices = []
        for price, vol in sorted_levels:
            value_area_volume += vol
            value_area_prices.append(price)
            if value_area_volume >= target_volume:
                break
        vah = max(value_area_prices) if value_area_prices else None
        val = min(value_area_prices) if value_area_prices else None
        return {'poc': poc_price, 'vah': vah, 'val': val}
    except Exception:
        return {'poc': None, 'vah': None, 'val': None}

def calculate_vwap_bands(df: pd.DataFrame, std_mult: float = 2.0) -> pd.DataFrame:
    result = pd.DataFrame(index=df.index)
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    vwap_val = (typical_price * df['Volume']).cumsum() / (df['Volume'].cumsum() + 1e-9)
    squared_diff = (typical_price - vwap_val) ** 2
    variance = (squared_diff * df['Volume']).cumsum() / (df['Volume'].cumsum() + 1e-9)
    std = np.sqrt(variance)
    result['vwap'] = vwap_val
    result['vwap_upper_1'] = vwap_val + std * std_mult
    result['vwap_lower_1'] = vwap_val - std * std_mult
    result['vwap_upper_2'] = vwap_val + std * (std_mult * 2)
    result['vwap_lower_2'] = vwap_val - std * (std_mult * 2)
    return result

def detect_vwap_rejection(df: pd.DataFrame) -> pd.Series:
    vwap_bands = calculate_vwap_bands(df)
    rejection = pd.Series(0, index=df.index)
    bullish = (
        (df['Low'] <= vwap_bands['vwap_lower_1']) &
        (df['Close'] > vwap_bands['vwap_lower_1']) &
        (df['Close'] > df['Open'])
    )
    bearish = (
        (df['High'] >= vwap_bands['vwap_upper_1']) &
        (df['Close'] < vwap_bands['vwap_upper_1']) &
        (df['Close'] < df['Open'])
    )
    rejection[bullish] = 1
    rejection[bearish] = -1
    return rejection

def simulate_orderbook_pressure(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    result = pd.DataFrame(index=df.index)
    buy_pressure = np.where(df['Close'] > df['Open'], df['Volume'], 0)
    sell_pressure = np.where(df['Close'] < df['Open'], df['Volume'], 0)
    result['buy_pressure'] = pd.Series(buy_pressure, index=df.index).rolling(window).sum()
    result['sell_pressure'] = pd.Series(sell_pressure, index=df.index).rolling(window).sum()
    result['pressure_ratio'] = result['buy_pressure'] / (result['sell_pressure'] + 1e-9)
    result['spread_proxy'] = (df['High'] - df['Low']) / df['Close']
    result['spread_ma'] = result['spread_proxy'].rolling(window).mean()
    result['spread_expansion'] = (result['spread_proxy'] > result['spread_ma'] * 1.5).astype(int)
    return result

def detect_iceberg_orders(df: pd.DataFrame, window: int = 50) -> pd.Series:
    vol_momentum = df['Volume'].pct_change(window)
    price_momentum = df['Close'].pct_change(window)
    iceberg_score = vol_momentum.abs() / (price_momentum.abs() + 1e-9)
    return pd.Series(np.tanh(iceberg_score / 10), index=df.index)

def calculate_amihud_illiquidity(df: pd.DataFrame, window: int = 20) -> pd.Series:
    returns = df['Close'].pct_change().abs()
    dollar_volume = df['Close'] * df['Volume']
    illiquidity = returns / (dollar_volume + 1e-9)
    return illiquidity.rolling(window).mean()

def calculate_market_efficiency_ratio(df: pd.DataFrame, period: int = 20) -> pd.Series:
    net_change = (df['Close'] - df['Close'].shift(period)).abs()
    sum_changes = df['Close'].diff().abs().rolling(period).sum()
    return net_change / (sum_changes + 1e-9)

def detect_price_manipulation(df: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame(index=df.index)
    vol_spike = df['Volume'] > df['Volume'].rolling(50).mean() * 3
    price_reversal = (df['Close'].shift(-1) - df['Close']) / df['Close']
    pump_dump = vol_spike & (price_reversal.abs() > 0.02)
    price_range = (df['High'] - df['Low']) / df['Close']
    vol_normalized = df['Volume'] / (df['Volume'].rolling(50).mean() + 1e-9)
    wash_trade = (vol_normalized > 2) & (price_range < 0.005)
    result['pump_dump'] = pump_dump.astype(int)
    result['wash_trade'] = wash_trade.astype(int)
    return result

def detect_accumulation_distribution(df: pd.DataFrame, window: int = 20) -> pd.Series:
    clv = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'] + 1e-9)
    ad_line = (clv * df['Volume']).cumsum()
    return ad_line.rolling(window).mean()

def detect_wyckoff_phases(df: pd.DataFrame) -> Dict[str, pd.Series]:
    result = {}
    vol_spike = df['Volume'] > df['Volume'].rolling(50).mean() * 2
    range_bound = (df['High'] - df['Low']) / df['Close'] < 0.01
    phase_a = vol_spike & range_bound
    low_vol = df['Volume'] < df['Volume'].rolling(50).mean()
    tight_range = (df['High'] - df['Low']) / df['Close'] < 0.005
    phase_b = low_vol & tight_range
    swing_low = df['Low'].rolling(20).min()
    spring = (df['Low'] < swing_low.shift(1)) & (df['Close'] > df['Open'])
    vol_increase = df['Volume'] > df['Volume'].rolling(20).mean() * 1.3
    price_rise = df['Close'] > df['Close'].rolling(10).mean()
    phase_d = vol_increase & price_rise
    result['phase_a_stopping'] = phase_a.astype(int)
    result['phase_b_accumulation'] = phase_b.astype(int)
    result['phase_c_spring'] = spring.astype(int)
    result['phase_d_markup'] = phase_d.astype(int)
    return result

def calculate_composite_index(df: pd.DataFrame) -> pd.Series:
    ad_line = detect_accumulation_distribution(df)
    ofi = detect_order_flow_imbalance(df)
    pressure = simulate_orderbook_pressure(df)
    composite = (
        0.4 * (ad_line / (ad_line.abs().max() + 1e-9)) +
        0.3 * ofi +
        0.3 * np.tanh(pressure['pressure_ratio'] - 1)
    )
    return composite

def detect_market_regime(df: pd.DataFrame) -> Dict[str, float]:
    returns = df['Close'].pct_change()
    volatility = returns.rolling(20).std() * np.sqrt(252)
    mer = calculate_market_efficiency_ratio(df, period=20).iloc[-1]
    vol_ratio = df['Volume'].iloc[-20:].mean() / (df['Volume'].iloc[-100:].mean() + 1e-9)
    return {
        'trending': float(mer > 0.5),
        'mean_reverting': float(mer < 0.3),
        'volatile': float(volatility.iloc[-1] > volatility.quantile(0.75)) if not volatility.isnull().all() else 0.0,
        'quiet': float(vol_ratio < 0.7),
        'regime_score': float(mer)
    }

def calculate_hurst_exponent(series: pd.Series, max_lag: int = 100) -> float:
    try:
        lags = range(2, min(max_lag, len(series) // 2))
        tau = []
        for lag in lags:
            std = np.std(np.subtract(series[lag:], series[:-lag]))
            tau.append(std)
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return float(np.clip(poly[0], 0, 1))
    except Exception:
        return 0.5

# ============================================
# FEATURE ENGINEERING (Complete)
# ============================================

_base_features = [
    "close", "open", "high", "low", "volume", "ema8", "ema21", "rsi14", 
    "momentum5", "range", "vol_trend", "confluence_score", "ob", "fvg", 
    "vol_spike", "ema_cross", "vwap", "order_flow_delta", "block_trade_flag", 
    "imbalance_sweep", "participation_rate"
]

def indicator_features(df: pd.DataFrame) -> pd.DataFrame:
    """Complete feature engineering with all indicators"""
    d = df.copy()
    
    # Basic features
    d["close"] = d["Close"]
    d["open"] = d["Open"]
    d["high"] = d["High"]
    d["low"] = d["Low"]
    d["volume"] = d["Volume"].fillna(0)
    d["ema8"] = ema(d["close"], 8)
    d["ema21"] = ema(d["close"], 21)
    d["rsi14"] = rsi(d["close"], 14)
    d["momentum5"] = d["close"].pct_change(5).fillna(0)
    d["range"] = (d["high"] - d["low"]) / (d["open"] + 1e-9)
    
    # Volume trend
    try:
        v = d["volume"].dropna()
        if len(v) >= 10:
            recent = v[-10:].values
            coef = np.polyfit(np.arange(len(recent)), recent, 1)[0]
            d["vol_trend"] = coef
        else:
            d["vol_trend"] = 0.0
    except Exception:
        d["vol_trend"] = 0.0
    
    # Original indicators
    d["ob"] = detect_order_blocks(df)
    d["fvg"] = detect_fvg(df)
    d["bos"] = detect_bos(df)
    
    macd_line, macd_signal, macd_hist = macd(d["close"])
    d["macd_line"] = macd_line
    d["macd_signal"] = macd_signal
    d["macd_hist"] = macd_hist
    
    d["supertrend"] = supertrend(df)
    d["boll_break"] = detect_bollinger_breakout(df)
    d["ema_cross"] = ema_crossover(df)
    d["vol_spike"] = detect_volume_spikes(df)
    d["rsi_sig"] = detect_rsi_signals(d["close"])
    
    d["vwap"] = vwap(df)
    d["vwma20"] = vwma(df, length=20)
    vp = session_volume_profile(df)
    d["poc"] = vp["poc"]
    d["vah"] = vp["vah"]
    d["val"] = vp["val"]
    d["order_flow_delta"] = order_flow_delta_proxy(df, window=10)
    d["block_trade_flag"] = block_trade_proxy(df)
    d["imbalance_sweep"] = imbalance_sweep(df)
    d["participation_rate"] = df["Volume"] / (df["Volume"].rolling(144).mean() + 1e-9)
    d["dark_pool_proxy"] = ((d["Volume"] > d["Volume"].rolling(50).mean() * 1.8) & (d["momentum5"].abs() > 0.002)).astype(int)
    
    # Institutional features
    liq = detect_liquidity_pools(df)
    d['liquidity_above'] = liq['liquidity_above']
    d['liquidity_below'] = liq['liquidity_below']
    d['market_structure_shift'] = detect_market_structure_shift(df)
    d['order_flow_imbalance'] = detect_order_flow_imbalance(df)
    
    inst = detect_institutional_candles(df)
    d['inst_bullish_engulf'] = inst['bullish_engulf']
    d['inst_bearish_engulf'] = inst['bearish_engulf']
    d['inst_bullish_rejection'] = inst['bullish_rejection']
    d['inst_bearish_rejection'] = inst['bearish_rejection']
    
    vwap_bands = calculate_vwap_bands(df)
    d['vwap_enhanced'] = vwap_bands['vwap']
    d['vwap_upper'] = vwap_bands['vwap_upper_1']
    d['vwap_lower'] = vwap_bands['vwap_lower_1']
    d['vwap_rejection'] = detect_vwap_rejection(df)
    
    pressure = simulate_orderbook_pressure(df)
    d['buy_pressure'] = pressure['buy_pressure']
    d['sell_pressure'] = pressure['sell_pressure']
    d['pressure_ratio'] = pressure['pressure_ratio']
    d['iceberg_score'] = detect_iceberg_orders(df)
    
    d['amihud_illiquidity'] = calculate_amihud_illiquidity(df)
    d['market_efficiency'] = calculate_market_efficiency_ratio(df)
    d['accumulation_distribution'] = detect_accumulation_distribution(df)
    d['composite_index'] = calculate_composite_index(df)
    
    wyckoff = detect_wyckoff_phases(df)
    for phase, series in wyckoff.items():
        d[phase] = series
    
    regime = detect_market_regime(df)
    for key, val in regime.items():
        d[key] = val
    
    d['hurst_exponent'] = calculate_hurst_exponent(d['close'])
    
    manip = detect_price_manipulation(df)
    d['pump_dump_flag'] = manip['pump_dump']
    d['wash_trade_flag'] = manip['wash_trade']
    
    vp_enhanced = calculate_volume_profile(df)
    if vp_enhanced['poc'] is not None:
        current_price = d['close'].iloc[-1]
        d['distance_to_poc'] = abs(current_price - vp_enhanced['poc']) / current_price
        d['above_poc'] = (current_price > vp_enhanced['poc']).astype(int)
        d['in_value_area'] = ((current_price >= vp_enhanced['val']) & (current_price <= vp_enhanced['vah'])).astype(int)
    else:
        d['distance_to_poc'] = 0.0
        d['above_poc'] = 0
        d['in_value_area'] = 0
    
    # Confluence score
    conf = (
        d["ob"].fillna(0)*1.2 + 
        d["fvg"].fillna(0)*1.1 + 
        d["bos"].abs().fillna(0)*1.0 + 
        d["vol_spike"].fillna(0)*0.9 + 
        np.where(d["ema_cross"]>0,1.0,0.0) + 
        np.where(d["boll_break"]>0,1.0,0.0)
    )
    d["confluence_score"] = (conf - conf.min()) / (conf.max() - conf.min() + 1e-9)
    
    return d.fillna(0)

# ============================================
# STRATEGY SCORING (All Strategies)
# ============================================

def strategy_quantum_score(df: pd.DataFrame) -> float:
    d = indicator_features(df)
    latest = d.iloc[-1]
    score = 0.0
    score += 0.25 * float(latest["ob"])
    score += 0.20 * float(latest["fvg"])
    score += 0.20 * (1.0 if abs(latest["bos"])==1 else 0.0)
    macd_sig = 1.0 if latest["macd_hist"]>0 else 0.0
    score += 0.15 * macd_sig
    st_trend = 1.0 if latest["close"] > latest["supertrend"] else 0.0
    score += 0.1 * st_trend
    score += 0.1 * float(latest["vol_spike"])
    return float(np.clip(score, 0.0, 1.0))

def strategy_momentum_score(df: pd.DataFrame) -> float:
    d = indicator_features(df)
    latest = d.iloc[-1]
    score = 0.0
    score += 0.35 * float(np.tanh(abs(latest["momentum5"]) * 100))
    score += 0.25 * float(latest["vol_spike"])
    score += 0.20 * (1.0 if latest["rsi_sig"] > 0 else 0.0)
    score += 0.20 * (1.0 if latest["ema_cross"] > 0 else 0.0)
    return float(np.clip(score, 0.0, 1.0))

def strategy_breakout_score(df: pd.DataFrame) -> float:
    d = indicator_features(df)
    latest = d.iloc[-1]
    sr_high, sr_low = detect_support_resistance(df)
    res_broken = latest["close"] > sr_high.iloc[-1]
    vol_ok = latest["vol_spike"]>0
    boll_ok = latest["boll_break"]>0
    score = 0.5*(1.0 if res_broken else 0.0) + 0.3*float(vol_ok) + 0.2*float(boll_ok)
    return float(np.clip(score, 0.0, 1.0))

def strategy_meanreversion_score(df: pd.DataFrame) -> float:
    d = indicator_features(df)
    latest = d.iloc[-1]
    score = 0.0
    score += 0.4*(1.0 if latest["rsi_sig"]!=0 else 0.0)
    upper, lower, ma = bollinger_bands(d["close"], window=20, n_std=2)
    near_upper = abs(latest["close"]-upper.iloc[-1]) < (0.002*latest["close"])
    near_lower = abs(latest["close"]-lower.iloc[-1]) < (0.002*latest["close"])
    score += 0.3*float(near_upper or near_lower)
    score += 0.3*(1.0 if latest["vol_spike"]==0 and abs(latest["momentum5"])<0.001 else 0.0)
    return float(np.clip(score, 0.0, 1.0))

def strategy_smart_money_score(df: pd.DataFrame) -> float:
    d = indicator_features(df)
    latest = d.iloc[-1]
    score = 0.0
    score += 0.2 * float(latest['liquidity_above'] or latest['liquidity_below'])
    score += 0.25 * float(abs(latest['market_structure_shift']))
    score += 0.25 * float(abs(latest['order_flow_imbalance']))
    candle_signal = latest['inst_bullish_engulf'] + latest['inst_bullish_rejection']
    score += 0.3 * float(candle_signal > 0)
    return float(np.clip(score, 0.0, 1.0))

def strategy_volume_profile_score(df: pd.DataFrame) -> float:
    d = indicator_features(df)
    latest = d.iloc[-1]
    score = 0.0
    if latest['distance_to_poc'] > 0:
        score += 0.4 * (1 - min(latest['distance_to_poc'] * 100, 1.0))
    score += 0.3 * float(abs(latest['vwap_rejection']))
    recent_vol = df['Volume'].iloc[-5:].mean()
    avg_vol = df['Volume'].iloc[-100:].mean()
    score += 0.3 * float(recent_vol > avg_vol * 1.2)
    return float(np.clip(score, 0.0, 1.0))

def strategy_wyckoff_score(df: pd.DataFrame) -> float:
    d = indicator_features(df)
    latest = d.iloc[-1]
    score = 0.0
    score += 0.4 * float(latest['phase_d_markup'])
    score += 0.3 * float(d['phase_c_spring'].iloc[-3:].sum() > 0)
    score += 0.3 * float(np.tanh(latest['composite_index']))
    return float(np.clip(score, 0.0, 1.0))

def strategy_market_maker_score(df: pd.DataFrame) -> float:
    d = indicator_features(df)
    latest = d.iloc[-1]
    score = 0.0
    avoid_manip = 1.0 - float(latest['pump_dump_flag'])
    score += 0.2 * avoid_manip
    balanced = abs(latest['pressure_ratio'] - 1.0)
    score += 0.3 * (1 - min(balanced, 1.0))
    liquid = 1 - min(latest['amihud_illiquidity'] * 1000, 1.0)
    score += 0.3 * float(liquid)
    score += 0.2 * (1 - latest['market_efficiency'])
    return float(np.clip(score, 0.0, 1.0))

def strategy_institutional_flow_score(df: pd.DataFrame) -> float:
    d = indicator_features(df)
    latest = d.iloc[-1]
    score = 0.0
    score += 0.3 * float(abs(latest['iceberg_score']))
    ad = d['accumulation_distribution']
    ad_slope = (ad.iloc[-1] - ad.iloc[-20]) / 20 if len(ad) > 20 else 0.0
    score += 0.4 * float(np.tanh(ad_slope / 1000))
    score += 0.3 * float(abs(latest['order_flow_imbalance']))
    return float(np.clip(score, 0.0, 1.0))

_STRATEGY_FN = {
    "quantum": strategy_quantum_score,
    "momentum": strategy_momentum_score,
    "breakout": strategy_breakout_score,
    "meanreversion": strategy_meanreversion_score,
    "smart_money": strategy_smart_money_score,
    "volume_profile": strategy_volume_profile_score,
    "wyckoff": strategy_wyckoff_score,
    "market_maker": strategy_market_maker_score,
    "institutional_flow": strategy_institutional_flow_score
}

# ============================================
# ALIGNMENT & DECISION LOGIC
# ============================================

def evaluate_strategy_scores(symbol: str, interval="5m", top_n=4):
    scores = {}
    for name, fn in _STRATEGY_FN.items():
        try:
            df = fetch_market_candles(symbol, interval=interval, limit=500)
            scores[name] = float(fn(df)) if not df.empty else 0.0
        except Exception:
            scores[name] = 0.0
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]

def evaluate_indicator_alignment(df: pd.DataFrame, indicator_names: List[str], threshold=INDICATOR_ALIGN_THRESHOLD, top_k=4):
    scores = {}
    if df.empty:
        return 0, []
    latest = df.iloc[-1]
    for ind in indicator_names:
        try:
            val = float(latest[ind]) if ind in latest else 0.0
            s = val if val in (0.0, 1.0) else 1.0 / (1.0 + np.exp(-val))
            scores[ind] = float(np.clip(s, 0.0, 1.0))
        except Exception:
            scores[ind] = 0.0
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    aligned_count = sum(1 for _, v in ranked if v >= threshold)
    return aligned_count, ranked

def evaluate_timeframe_agreement(symbol: str, tfs: List[str] = None, threshold=TIMEFRAME_AGREE_THRESHOLD):
    tfs = tfs or ["1d", "4h", "1h", "15m"]
    agree = 0
    details = []
    for tf in tfs:
        try:
            df = fetch_market_candles(symbol, interval=tf, limit=200)
            if df.empty:
                details.append((tf, None))
                continue
            d = indicator_features(df)
            latest = d.iloc[-1]
            p = 0.5 * float(np.tanh(latest["momentum5"] * 100)) + 0.5 * float(latest["confluence_score"])
            p = float(np.clip(p, 0.0, 1.0))
            if p >= threshold:
                agree += 1
            details.append((tf, p))
        except Exception:
            details.append((tf, None))
    return agree, details

def fuse_predictions(rf_prob: float = None, keras_prob: float = None, tv_score: float = None):
    parts = []
    if rf_prob is not None:
        parts.append(("rf", rf_prob, 0.4))
    if keras_prob is not None:
        parts.append(("keras", keras_prob, 0.4))
    if tv_score is not None:
        parts.append(("tv", tv_score, 0.2))
    if not parts:
        return None
    total_weight = sum(w for _, _, w in parts)
    score = sum(p * w for _, p, w in parts) / total_weight
    return float(np.clip(score, 0.0, 1.0))

def decision_requirements_pass(symbol: str, rf_p: float = None, keras_p: float = None, tv_score: float = None):
    reasons: Dict[str, Any] = {}
    top_strats = evaluate_strategy_scores(symbol)
    strat_ok = sum(1 for _, s in top_strats if s >= STRATEGY_ALIGN_THRESHOLD) >= REQUIRED_STRATEGY_AGREE
    reasons["strategies"] = {"top": top_strats, "ok": strat_ok}
    
    df_recent = fetch_market_candles(symbol, interval="5m", limit=500)
    if df_recent.empty:
        reasons["indicators"] = {"ok": False, "reason": "no_data"}
        return False, reasons
    
    d = indicator_features(df_recent)
    important_inds = ["confluence_score", "momentum5", "ob", "fvg", "vol_spike", "ema_cross"]
    ind_count, top_inds = evaluate_indicator_alignment(d, important_inds, threshold=INDICATOR_ALIGN_THRESHOLD, top_k=4)
    ind_ok = ind_count >= REQUIRED_INDICATOR_AGREE
    reasons["indicators"] = {"top": top_inds, "aligned_count": ind_count, "ok": ind_ok}
    
    tf_agree, tf_details = evaluate_timeframe_agreement(symbol, tfs=["1d", "4h", "1h"], threshold=TIMEFRAME_AGREE_THRESHOLD)
    tf_ok = tf_agree >= REQUIRED_TIMEFRAME_AGREE
    reasons["timeframes"] = {"agree": tf_agree, "details": tf_details, "ok": tf_ok}
    
    fused = fuse_predictions(rf_prob=rf_p, keras_prob=keras_p, tv_score=tv_score)
    fusion_ok = fused is not None and fused >= STRATEGY_ALIGN_THRESHOLD
    reasons["fusion"] = {"fused": fused, "ok": fusion_ok}
    
    ok = strat_ok and ind_ok and tf_ok and fusion_ok
    reasons["final_ok"] = ok
    return ok, reasons

# ============================================
# ML MODEL MANAGEMENT
# ============================================

def save_rf_model(model):
    try:
        joblib.dump(model, RF_MODEL_FILE)
        logger.info("RF model saved")
    except Exception as e:
        logger.error(f"save_rf_model error: {e}")

def load_rf_model():
    global _rf_model
    with _model_lock:
        if RF_MODEL_FILE.exists():
            try:
                _rf_model = joblib.load(RF_MODEL_FILE)
                logger.info("RF model loaded")
            except Exception as e:
                logger.error(f"load_rf_model error: {e}")
                _rf_model = None
        else:
            _rf_model = None
    return _rf_model

def save_keras_model(model):
    try:
        if TF_AVAILABLE:
            model.save(KERAS_MODEL_FILE)
            logger.info("Keras model saved")
    except Exception as e:
        logger.error(f"save_keras_model error: {e}")

def load_keras_model():
    global _keras_model
    with _model_lock:
        if TF_AVAILABLE and KERAS_MODEL_FILE.exists():
            try:
                _keras_model = keras_load_model(KERAS_MODEL_FILE)
                logger.info("Keras model loaded")
            except Exception as e:
                logger.error(f"load_keras_model error: {e}")
                _keras_model = None
        else:
            _keras_model = None
    return _keras_model

def build_features_and_labels(df: pd.DataFrame, future_horizon=3, thr=0.0005):
    d = indicator_features(df)
    d["future_close"] = d["close"].shift(-future_horizon)
    d["label"] = (d["future_close"] > (d["close"] * (1 + thr))).astype(int)
    d = d.dropna()
    feature_cols = [c for c in d.columns if c not in ['label', 'future_close', 'datetime']]
    X = d[feature_cols]
    y = d["label"].astype(int)
    return X, y

def train_rf_model_from_symbols(symbols=None):
    symbols = symbols or PO_SYMBOLS
    rows = []
    for s in symbols:
        try:
            df = fetch_market_candles(s, interval="1h", limit=2000)
            if df.empty:
                continue
            X, y = build_features_and_labels(df, future_horizon=3)
            if X.empty:
                continue
            tmp = X.copy()
            tmp["label"] = y.values
            tmp["symbol"] = s
            rows.append(tmp)
            logger.info(f"Train RF: fetched {s} rows={len(X)}")
        except Exception as e:
            logger.error(f"Train fetch error {s}: {e}")
    if not rows:
        logger.warning("No training data")
        return None
    data = pd.concat(rows, ignore_index=True)
    data.to_csv(DATA_FILE, index=False)
    feature_cols = [c for c in data.columns if c not in ("label", "symbol")]
    X = data[feature_cols]
    y = data["label"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y)
    clf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_SEED, n_jobs=-1)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
