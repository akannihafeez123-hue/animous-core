#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bot.py — Pocket Master AI (single-file)
- Telegram bot + Flask API + TradingView webhook
- Deterministic institutional indicators (VWAP, POC, OB, FVG, BOS, SuperTrend, etc.)
- RandomForest + optional Keras LSTM; daily retrain hooks
- Bitget public market fetch (ticker + klines) and minimal signed order helper
- Strict alignment rules: require strategy, indicator, and timeframe agreement
- Autonomous monitors: /scalp and /swing to auto-execute when thresholds met
- Dry-run safe by default
DISCLAIMER: Backtest and validate before live use. Use DRY_RUN until production-ready.
"""

import os
import time
import json
import hmac
import hashlib
import base64
import threading
import traceback
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import requests
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Optional Keras
try:
    from tensorflow.keras.models import Sequential, load_model as keras_load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

# Telegram and Flask
import telebot
from flask import Flask, request, jsonify

# Dotenv
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ---------------- Config ----------------
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
ADMIN_CHAT = os.getenv("ADMIN_CHAT", "")
ADMIN_ID_LEGACY = os.getenv("ADMIN_ID", "")
ADMIN_SECRET = os.getenv("ADMIN_SECRET", "")

BITGET_API_KEY = os.getenv("BITGET_API_KEY", "")
BITGET_API_SECRET = os.getenv("BITGET_API_SECRET", "")
BITGET_API_PASSPHRASE = os.getenv("BITGET_API_PASSPHRASE", "")
BITGET_REST_BASE = os.getenv("BITGET_REST_BASE", "https://api.bitget.com")

TV_WEBHOOK_SECRET = os.getenv("TV_WEBHOOK_SECRET", "")

PORT = int(os.getenv("PORT", 8080))
MODEL_DIR = Path(os.getenv("MODEL_DIR", "/tmp/po_models"))
MODEL_DIR.mkdir(parents=True, exist_ok=True)
RF_MODEL_FILE = MODEL_DIR / "rf_model.joblib"
KERAS_MODEL_FILE = MODEL_DIR / "keras_model.h5"
DATA_FILE = MODEL_DIR / "training_data.csv"
TV_ALERTS_FILE = MODEL_DIR / "tv_alerts.jsonl"

PO_SYMBOLS = os.getenv("PO_SYMBOLS", "BTCUSDT,ETHUSDT").split(",")
RANDOM_SEED = int(os.getenv("RANDOM_SEED", 42))

STRATEGY_ALIGN_THRESHOLD = float(os.getenv("STRATEGY_ALIGN_THRESHOLD", 0.8))
INDICATOR_ALIGN_THRESHOLD = float(os.getenv("INDICATOR_ALIGN_THRESHOLD", 0.8))
TIMEFRAME_AGREE_THRESHOLD = float(os.getenv("TIMEFRAME_AGREE_THRESHOLD", 0.8))
REQUIRED_STRATEGY_AGREE = int(os.getenv("REQUIRED_STRATEGY_AGREE", 3))
REQUIRED_INDICATOR_AGREE = int(os.getenv("REQUIRED_INDICATOR_AGREE", 3))
REQUIRED_TIMEFRAME_AGREE = int(os.getenv("REQUIRED_TIMEFRAME_AGREE", 3))
MIN_CONFIDENCE_TO_SIGNAL = float(os.getenv("MIN_CONFIDENCE_TO_SIGNAL", 0.8))

DRY_RUN = os.getenv("DRY_RUN", "true").lower() != "false"
AUTO_MONITOR_POLL = float(os.getenv("AUTO_MONITOR_POLL", "5.0"))
MAX_CONCURRENT_MONITORS = int(os.getenv("MAX_CONCURRENT_MONITORS", "8"))
ORDER_SIZE = float(os.getenv("ORDER_SIZE", "0.001"))

KERAS_WINDOW = int(os.getenv("KERAS_WINDOW", 32))
KERAS_EPOCHS = int(os.getenv("KERAS_EPOCHS", 6))
KERAS_BATCH = int(os.getenv("KERAS_BATCH", 32))

_INTERVAL_TO_GRAN = {
    "1m": "60",
    "5m": "300",
    "15m": "900",
    "30m": "1800",
    "1h": "3600",
    "4h": "14400",
    "1d": "86400"
}

TIMEFRAMES_TOP_DOWN = ["1y", "6mo", "1mo", "1w", "1d", "4h", "1h", "15m", "5m", "1m"]

# ---------------- Globals ----------------
bot = telebot.TeleBot(TELEGRAM_TOKEN) if TELEGRAM_TOKEN else None
app = Flask(__name__)
_rf_model = None
_keras_model = None
_model_lock = threading.Lock()
_current_strategy = "quantum"
_STRATEGIES = ["quantum", "momentum", "breakout", "meanreversion"]
_active_monitors: Dict[str, Dict[str, Any]] = {}

print(f"[INIT] TF_AVAILABLE={TF_AVAILABLE} TELEGRAM={'SET' if TELEGRAM_TOKEN else 'MISSING'} BITGET={'SET' if BITGET_API_KEY else 'MISSING'} DRY_RUN={DRY_RUN}")

# ---------------- Bitget helpers ----------------
def _bitget_sign(method: str, request_path: str, body: str = "", timestamp: str = None) -> Dict[str, str]:
    if timestamp is None:
        timestamp = str(int(time.time() * 1000))
    prehash = timestamp + method.upper() + request_path + (body or "")
    h = hmac.new(BITGET_API_SECRET.encode() if BITGET_API_SECRET else b"", prehash.encode(), hashlib.sha256)
    sign = base64.b64encode(h.digest()).decode() if BITGET_API_SECRET else ""
    headers = {
        "ACCESS-KEY": BITGET_API_KEY,
        "ACCESS-SIGN": sign,
        "ACCESS-TIMESTAMP": timestamp,
        "ACCESS-PASSPHRASE": BITGET_API_PASSPHRASE,
        "Content-Type": "application/json"
    }
    return headers

def bitget_public_get(path: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
    url = BITGET_REST_BASE + path
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print("bitget_public_get error:", e)
        return {}

def bitget_signed_request(method: str, path: str, body: Dict[str, Any] = None, params: Dict[str, Any] = None) -> Dict[str, Any]:
    body_json = ""
    if body:
        body_json = json.dumps(body, separators=(",", ":"), ensure_ascii=False)
    timestamp = str(int(time.time() * 1000))
    headers = _bitget_sign(method, path, body_json, timestamp=timestamp)
    url = BITGET_REST_BASE + path
    try:
        if method.upper() == "GET":
            r = requests.get(url, params=params, headers=headers, timeout=10)
        else:
            r = requests.post(url, data=body_json.encode("utf-8"), headers=headers, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print("bitget_signed_request error:", e, getattr(e, "response", None))
        return {}

def bitget_ticker(symbol: str) -> float:
    sym = symbol.replace("-", "").replace("/", "").upper()
    resp = bitget_public_get("/api/spot/v1/market/ticker", params={"symbol": sym})
    try:
        data = resp.get("data") or resp.get("data", {})
        if isinstance(data, dict) and "last" in data:
            return float(data["last"])
        if isinstance(data, list) and len(data) and isinstance(data[0], dict) and "last" in data[0]:
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
        data = resp.get("data") or resp.get("data", [])
        rows = []
        for item in data:
            if isinstance(item, dict):
                ts = item.get("timestamp") or item.get("id") or item.get("time")
                o = float(item.get("open", 0)); h = float(item.get("high", 0)); l = float(item.get("low", 0)); c = float(item.get("close", 0)); v = float(item.get("volume", 0))
            elif isinstance(item, (list, tuple)) and len(item) >= 6:
                ts = int(item[0]); o, h, l, c, v = float(item[1]), float(item[2]), float(item[3]), float(item[4]), float(item[5])
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
        print("bitget_klines parse error:", e)
        return pd.DataFrame()

@lru_cache(maxsize=512)
def cached_bitget_candles(symbol: str, interval: str = "1m", limit: int = 500) -> pd.DataFrame:
    gran = _INTERVAL_TO_GRAN.get(interval, "300")
    return bitget_klines(symbol, granularity=gran, limit=limit)

def fetch_market_candles(symbol: str, interval: str = "1m", limit: int = 500) -> pd.DataFrame:
    return cached_bitget_candles(symbol, interval, limit)

# ---------------- Utilities ----------------
def is_admin_chat(chat_id, username=None) -> bool:
    try:
        cid = str(chat_id)
        candidates = {ADMIN_CHAT, ADMIN_ID_LEGACY}
        candidates = {c for c in candidates if c}
        norm = set()
        for a in candidates:
            a = a.strip()
            norm.add(a)
            if a.startswith("@"):
                norm.add(a.lstrip("@"))
            else:
                norm.add("@" + a)
        if cid in norm:
            return True
        if username and username in norm:
            return True
    except Exception:
        pass
    return False

def safe_send(chat_id, text):
    try:
        if bot:
            bot.send_message(chat_id, text, parse_mode="Markdown")
            return True
    except Exception as e:
        print("safe_send error:", e)
    return False

# ---------------- Indicators ----------------
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
        b1 = df.iloc[i - 2]
        b3 = df.iloc[i]
        if (b1["High"] < b3["Low"]) or (b1["Low"] > b3["High"]):
            fvg.iloc[i] = 1
    return fvg

def detect_bos(df: pd.DataFrame, lookback=20) -> pd.Series:
    bos = pd.Series(0, index=df.index)
    highs = df["High"]
    lows = df["Low"]
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
            continue
        if df["Close"].iloc[i] > st.iloc[i - 1]:
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
    else:
        return (pv.cumsum() / (df["Volume"].cumsum() + 1e-9)).fillna(method="bfill")

def vwma(df: pd.DataFrame, length=20):
    pv = df["Close"] * df["Volume"]
    return pv.rolling(length).sum() / (df["Volume"].rolling(length).sum() + 1e-9)

def session_volume_profile(df: pd.DataFrame, bins=20):
    try:
        hist_vals, edges = np.histogram(df["Close"], bins=bins, weights=df["Volume"])
        max_idx = int(np.argmax(hist_vals))
        poc = 0.5 * (edges[max_idx] + edges[max_idx + 1])
        vah = edges[max_idx + 1]
        val = edges[max_idx]
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

# ---------------- Feature assembly ----------------
_base_features = ["close", "open", "high", "low", "volume", "ema8", "ema21", "rsi14", "momentum5", "range", "vol_trend"]

def indicator_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["close"] = d["Close"]; d["open"] = d["Open"]; d["high"] = d["High"]; d["low"] = d["Low"]; d["volume"] = d["Volume"].fillna(0)
    d["ema8"] = ema(d["close"], 8); d["ema21"] = ema(d["close"], 21)
    d["rsi14"] = rsi(d["close"], 14); d["momentum5"] = d["close"].pct_change(5).fillna(0)
    d["range"] = (d["high"] - d["low"]) / (d["open"] + 1e-9)
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

    d["ob"] = detect_order_blocks(df)
    d["fvg"] = detect_fvg(df)
    d["bos"] = detect_bos(df)
    macd_line, macd_signal, macd_hist = macd(d["close"]); d["macd_line"] = macd_line; d["macd_signal"] = macd_signal; d["macd_hist"] = macd_hist
    d["supertrend"] = supertrend(df)
    d["boll_break"] = detect_bollinger_breakout(df)
    d["ema_cross"] = ema_crossover(df)
    d["vol_spike"] = detect_volume_spikes(df)
    d["rsi_sig"] = detect_rsi_signals(d["close"])

    d["vwap"] = vwap(df)
    d["vwma20"] = vwma(df, length=20)
    vp = session_volume_profile(df)
    d["poc"] = vp["poc"]; d["vah"] = vp["vah"]; d["val"] = vp["val"]
    d["order_flow_delta"] = order_flow_delta_proxy(df, window=10)
    d["block_trade_flag"] = block_trade_proxy(df)
    d["imbalance_sweep"] = imbalance_sweep(df)
    d["participation_rate"] = df["Volume"] / (df["Volume"].rolling(144).mean() + 1e-9)
    d["dark_pool_proxy"] = ((d["Volume"] > d["Volume"].rolling(50).mean() * 1.8) & (d["momentum5"].abs() > 0.002)).astype(int)

    conf = (d["ob"].fillna(0)*1.2 + d["fvg"].fillna(0)*1.1 + d["bos"].abs().fillna(0)*1.0 + d["vol_spike"].fillna(0)*0.9 + np.where(d["ema_cross"]>0,1.0,0.0) + np.where(d["boll_break"]>0,1.0,0.0))
    d["confluence_score"] = (conf - conf.min()) / (conf.max() - conf.min() + 1e-9)
    return d.fillna(0)

# ---------------- Strategy scoring ----------------
def strategy_quantum_score(df: pd.DataFrame) -> float:
    d = indicator_features(df); latest = d.iloc[-1]; score = 0.0
    score += 0.25 * float(latest["ob"]); score += 0.20 * float(latest["fvg"]); score += 0.20 * (1.0 if abs(latest["bos"])==1 else 0.0)
    macd_sig = 1.0 if latest["macd_hist"]>0 else 0.0; score += 0.15 * macd_sig
    st_trend = 1.0 if latest["close"] > latest["supertrend"] else 0.0; score += 0.1 * st_trend
    score += 0.1 * float(latest["vol_spike"]); return float(np.clip(score,0.0,1.0))

def strategy_momentum_score(df: pd.DataFrame) -> float:
    d = indicator_features(df); latest = d.iloc[-1]; score = 0.0
    score += 0.35 * float(np.tanh(abs(latest["momentum5"]) * 100))
    score += 0.25 * float(latest["vol_spike"])
    score += 0.20 * (1.0 if latest["rsi_sig"] > 0 else 0.0)
    score += 0.20 * (1.0 if latest["ema_cross"] > 0 else 0.0)
    return float(np.clip(score,0.0,1.0))

def strategy_breakout_score(df: pd.DataFrame) -> float:
    d = indicator_features(df); latest = d.iloc[-1]; sr_high, sr_low = detect_support_resistance(df)
    res_broken = latest["close"] > sr_high.iloc[-1]; vol_ok = latest["vol_spike"]>0; boll_ok = latest["boll_break"]>0
    score = 0.5*(1.0 if res_broken else 0.0) + 0.3*float(vol_ok) + 0.2*float(boll_ok); return float(np.clip(score,0.0,1.0))

def strategy_meanreversion_score(df: pd.DataFrame) -> float:
    d = indicator_features(df); latest = d.iloc[-1]; score = 0.0
    score += 0.4*(1.0 if latest["rsi_sig"]!=0 else 0.0)
    upper, lower, ma = bollinger_bands(d["close"], window=20, n_std=2)
    near_upper = abs(latest["close"]-upper.iloc[-1]) < (0.002*latest["close"])
    near_lower = abs(latest["close"]-lower.iloc[-1]) < (0.002*latest["close"])
    score += 0.3*float(near_upper or near_lower)
    score += 0.3*(1.0 if latest["vol_spike"]==0 and abs(latest["momentum5"])<0.001 else 0.0)
    return float(np.clip(score,0.0,1.0))

_STRATEGY_FN = {"quantum": strategy_quantum_score, "momentum": strategy_momentum_score, "breakout": strategy_breakout_score, "meanreversion": strategy_meanreversion_score}

# ---------------- Alignment helpers & decision ----------------
def evaluate_strategy_scores(symbol: str, lookback_period="2d", interval="5m", top_n=4):
    scores = {}
    for name, fn in _STRATEGY_FN.items():
        try:
            df = fetch_market_candles(symbol, interval=interval, limit=500)
            scores[name] = float(fn(df)) if not df.empty else 0.0
        except Exception:
            scores[name] = 0.0
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_n]

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

def sample_timeframe_period(tf: str) -> Tuple[str, str]:
    mapping = {
        "1y": ("1y", "1d"),
        "6mo": ("6mo", "1d"),
        "1mo": ("1mo", "4h"),
        "1w": ("7d", "1h"),
        "1d": ("30d", "15m"),
        "4h": ("14d", "15m"),
        "1h": ("7d", "5m"),
        "15m": ("2d", "1m"),
        "5m": ("1d", "1m"),
        "1m": ("7d", "1m")
    }
    return mapping.get(tf, ("7d", "5m"))

def evaluate_timeframe_agreement(symbol: str, tfs: List[str] = None, threshold=TIMEFRAME_AGREE_THRESHOLD):
    tfs = tfs or ["1y", "1mo", "1w", "1d", "4h", "1h"]
    agree = 0
    details = []
    for tf in tfs:
        period, interval = sample_timeframe_period(tf)
        try:
            df = fetch_market_candles(symbol, interval=interval, limit=500)
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
    important_inds = ["confluence_score", "momentum5", "ob", "fvg", "vol_spike", "ema_cross"]
    ind_count, top_inds = evaluate_indicator_alignment(df_recent, important_inds, threshold=INDICATOR_ALIGN_THRESHOLD, top_k=4)
    ind_ok = ind_count >= REQUIRED_INDICATOR_AGREE
    reasons["indicators"] = {"top": top_inds, "aligned_count": ind_count, "ok": ind_ok}

    tf_agree, tf_details = evaluate_timeframe_agreement(symbol, tfs=["1y", "1mo", "1w", "1d", "4h", "1h"], threshold=TIMEFRAME_AGREE_THRESHOLD)
    tf_ok = tf_agree >= REQUIRED_TIMEFRAME_AGREE
    reasons["timeframes"] = {"agree": tf_agree, "details": tf_details, "ok": tf_ok}

    fused = fuse_predictions(rf_prob=rf_p, keras_prob=keras_p, tv_score=tv_score)
    fusion_ok = fused is not None and fused >= STRATEGY_ALIGN_THRESHOLD
    reasons["fusion"] = {"fused": fused, "ok": fusion_ok}

    ok = strat_ok and ind_ok and tf_ok and fusion_ok
    reasons["final_ok"] = ok
    return ok, reasons

# ---------------- Model save/load ----------------
def save_rf_model(model):
    try:
        joblib.dump(model, RF_MODEL_FILE)
        print("[MODEL] RF saved")
    except Exception as e:
        print("save_rf_model error:", e)

def load_rf_model():
    global _rf_model
    with _model_lock:
        if RF_MODEL_FILE.exists():
            try:
                _rf_model = joblib.load(RF_MODEL_FILE)
                print("[MODEL] RF loaded")
            except Exception as e:
                print("load_rf_model error:", e)
                _rf_model = None
        else:
            _rf_model = None
    return _rf_model

def save_keras_model(model):
    try:
        if TF_AVAILABLE:
            model.save(KERAS_MODEL_FILE)
            print("[MODEL] Keras saved")
    except Exception as e:
        print("save_keras_model error:", e)

def load_keras_model():
    global _keras_model
    with _model_lock:
        if TF_AVAILABLE and KERAS_MODEL_FILE.exists():
            try:
                _keras_model = keras_load_model(KERAS_MODEL_FILE)
                print("[MODEL] Keras loaded")
            except Exception as e:
                print("load_keras_model error:", e)
                _keras_model = None
        else:
            _keras_model = None
    return _keras_model

# ---------------- Training utilities ----------------
def build_features_and_labels(df: pd.DataFrame, future_horizon=3, thr=0.0005):
    d = indicator_features(df)
    d["future_close"] = d["close"].shift(-future_horizon)
    d["label"] = (d["future_close"] > (d["close"] * (1 + thr))).astype(int)
    d = d.dropna()
    feature_cols = [*_base_features, "confluence_score", "momentum5", "ob", "fvg", "vol_spike", "ema_cross", "vwap", "order_flow_delta", "block_trade_flag", "imbalance_sweep", "participation_rate"]
    X = d[feature_cols]
    y = d["label"].astype(int)
    return X, y

def train_rf_model_from_yf(symbols=None):
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
            tmp = X.copy(); tmp["label"] = y.values; tmp["symbol"] = s
            rows.append(tmp)
            print(f"[TRAIN RF] fetched {s} rows={len(X)}")
        except Exception as e:
            print("train fetch error:", s, e)
    if not rows:
        print("[TRAIN RF] no training data")
        return None
    data = pd.concat(rows, ignore_index=True)
    data.to_csv(DATA_FILE, index=False)
    feature_cols = [c for c in data.columns if c not in ("label", "symbol")]
    X = data[feature_cols]; y = data["label"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y)
    clf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_SEED, n_jobs=-1)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test); acc = accuracy_score(y_test, preds)
    save_rf_model(clf); global _rf_model; _rf_model = clf
    print(f"[TRAIN RF] done acc={acc:.4f} rows={len(data)}")
    return {"accuracy": float(acc), "trained_rows": len(data)}

# ---------------- Prediction ----------------
def predict_with_rf_row(df: pd.DataFrame):
    model = load_rf_model() or _rf_model
    if model is None:
        return None
    try:
        feature_cols = [*_base_features, "confluence_score", "momentum5", "ob", "fvg", "vol_spike", "ema_cross", "vwap", "order_flow_delta", "block_trade_flag", "imbalance_sweep", "participation_rate"]
        X = df[feature_cols]
        probs = model.predict_proba(X)[:, 1]
        return probs
    except Exception as e:
        print("predict_with_rf_row error:", e)
        return None

def predict_signals_for_symbol(symbol: str, interval="5m"):
    df = fetch_market_candles(symbol, interval=interval, limit=500)
    if df.empty:
        return []
    d = indicator_features(df)
    rf_probs = predict_with_rf_row(d) or np.zeros(len(d))
    rf_p = float(rf_probs[-1])
    keras_p = None
    fused = fuse_predictions(rf_prob=rf_p, keras_prob=keras_p, tv_score=None)
    ok, reasons = decision_requirements_pass(symbol, rf_p=rf_p, keras_p=keras_p, tv_score=None)
    signal = "BUY" if ok else "HOLD"
    return [{"symbol": symbol, "time": str(d.index[-1]), "rf": rf_p, "fused": fused, "signal": signal, "reasons": reasons}]

# ---------------- Order helpers ----------------
def bitget_place_order(symbol: str, side: str, size: float, price: float = None, order_type: str = "limit", client_oid: str = None, dry_run: bool = True):
    if dry_run:
        return {"ok": True, "dry_run": True, "symbol": symbol, "side": side, "size": size, "price": price, "type": order_type}
    if not BITGET_API_KEY or not BITGET_API_SECRET:
        return {"ok": False, "error": "missing_credentials"}
    path = "/api/spot/v1/order"
    body = {
        "symbol": symbol.replace("-", "").upper(),
        "side": side,
        "type": order_type,
        "size": str(size)
    }
    if price is not None and order_type == "limit":
        body["price"] = str(price)
    if client_oid:
        body["clientOid"] = client_oid
    resp = bitget_signed_request("POST", path, body=body)
    return resp

def place_order_if_allowed(symbol: str, side: str, size: float, price: float = None, order_type: str = "limit", client_oid: str = None):
    print(f"[EXEC] Requesting order: {symbol} {side} size={size} price={price} type={order_type} dry_run={DRY_RUN}")
    if DRY_RUN:
        return {"ok": True, "dry_run": True, "symbol": symbol, "side": side, "size": size, "price": price}
    resp = bitget_place_order(symbol=symbol, side=side, size=size, price=price, order_type=order_type, client_oid=client_oid, dry_run=False)
    return resp

# ---------------- Monitors ----------------
def monitor_symbol_for_trade(symbol: str, timeframe: str, strategy_mode: str = "scalp", stop_event: threading.Event = None):
    print(f"[MONITOR] Starting monitor: {symbol} {timeframe} mode={strategy_mode}")
    attempts = 0
    while True:
        if stop_event and stop_event.is_set():
            print(f"[MONITOR] Stopped: {symbol} {timeframe}")
            return {"ok": False, "reason": "stopped"}
        try:
            df = fetch_market_candles(symbol, interval=timeframe, limit=500)
            if df.empty:
                attempts += 1
                time.sleep(max(1.0, AUTO_MONITOR_POLL))
                continue
            d = indicator_features(df)
            rf_probs = predict_with_rf_row(d)
            rf_p = float(rf_probs[-1]) if rf_probs is not None else None
            keras_p = None
            ok, reasons = decision_requirements_pass(symbol, rf_p=rf_p, keras_p=keras_p, tv_score=None)
            print(f"[MONITOR] {symbol} {timeframe} attempt={attempts} rf_p={rf_p} ok={ok}")
            if ok:
                side = "buy"
                size = ORDER_SIZE
                if strategy_mode == "scalp":
                    order_type = "market"
                    price = None
                else:
                    order_type = "limit"
                    price = None
                resp = place_order_if_allowed(symbol=symbol, side=side, size=size, price=price, order_type=order_type)
                if ADMIN_CHAT:
                    try:
                        safe_send(ADMIN_CHAT, f"AUTOTRADE {symbol} {strategy_mode} -> {resp}")
                    except Exception:
                        pass
                return {"ok": True, "order": resp, "reasons": reasons}
            time.sleep(max(1.0, AUTO_MONITOR_POLL))
            attempts += 1
        except Exception as e:
            print("[MONITOR] error:", e)
            time.sleep(2.0)
            attempts += 1
            continue

def start_monitor_thread(key: str, target_fn, *args, **kwargs):
    if key in _active_monitors:
        return {"ok": False, "error": "monitor_already_running"}
    if len(_active_monitors) >= MAX_CONCURRENT_MONITORS:
        return {"ok": False, "error": "max_monitors_reached"}
    stop_event = threading.Event()
    th = threading.Thread(target=lambda: target_fn(*args, stop_event=stop_event, **kwargs), daemon=True)
    _active_monitors[key] = {"thread": th, "stop_event": stop_event}
    th.start()
    return {"ok": True, "key": key}

def stop_monitor(key: str):
    info = _active_monitors.get(key)
    if not info:
        return {"ok": False, "error": "not_found"}
    info["stop_event"].set()
    del _active_monitors[key]
    return {"ok": True}

# ---------------- TradingView webhook ----------------
@app.route("/tv_alert", methods=["POST"])
def tradingview_alert():
    secret = request.headers.get("X-TV-SECRET", "")
    if TV_WEBHOOK_SECRET and secret != TV_WEBHOOK_SECRET:
        return jsonify({"ok": False, "error": "unauthorized"}), 401
    data = request.get_json(force=True, silent=True) or {}
    symbol = data.get("symbol") or data.get("ticker")
    tv_score = float(data.get("confidence", data.get("score", 0.0)))
    tv_type = data.get("type", "CUSTOM")
    try:
        with open(TV_ALERTS_FILE, "a") as f:
            f.write(json.dumps({"received_at": datetime.utcnow().isoformat(), "symbol": symbol, "payload": data}) + "\n")
        df = fetch_market_candles(symbol, interval="5m", limit=500)
        if df.empty:
            return jsonify({"ok": False, "reason": "no_data"}), 400
        d = indicator_features(df)
        rf_prob = None
        model = load_rf_model()
        if model is not None:
            try:
                feature_cols = [*_base_features, "confluence_score", "momentum5", "ob", "fvg", "vol_spike", "ema_cross", "vwap", "order_flow_delta", "block_trade_flag", "imbalance_sweep", "participation_rate"]
                X = d[feature_cols]
                rf_prob = float(model.predict_proba(X.tail(1))[0][1])
            except Exception:
                rf_prob = None
        keras_p = None
        fused = fuse_predictions(rf_prob=rf_prob, keras_prob=keras_p, tv_score=tv_score)
        ok, reasons = decision_requirements_pass(symbol, rf_p=rf_prob, keras_p=keras_p, tv_score=tv_score)
        signal = "BUY" if ok else "HOLD"
        payload = {"symbol": symbol, "rf_prob": rf_prob, "tv_score": tv_score, "fused": fused, "signal": signal, "reasons": reasons, "tv_type": tv_type}
        if ADMIN_CHAT:
            try:
                safe_send(ADMIN_CHAT, f"TV alert {symbol} => {signal} p={fused}\nrf={rf_prob} tv={tv_score} type={tv_type}")
            except Exception:
                pass
        return jsonify({"ok": True, "payload": payload})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(e)}), 500

# ---------------- Backtester ----------------
def backtest_strategy(signals: List[Dict[str, Any]], price_series: pd.Series, sl_pct=0.01, tp_pct=0.02):
    returns = []
    equity = [1.0]
    peak = 1.0
    for s in signals:
        t = pd.to_datetime(s["time"])
        if t not in price_series.index:
            pos = price_series.index.get_indexer([t], method="nearest")
            if len(pos) == 0 or pos[0] < 0:
                continue
            t = price_series.index[pos[0]]
        entry = float(price_series.loc[t])
        if s["signal"] == "BUY":
            window = price_series.loc[t:].iloc[:30]
            tp_price = entry * (1 + tp_pct)
            sl_price = entry * (1 - sl_pct)
            result = 0.0
            exited = False
            for p in window:
                if p >= tp_price:
                    result = tp_pct; exited = True; break
                if p <= sl_price:
                    result = -sl_pct; exited = True; break
            if not exited:
                result = (window.iloc[-1] - entry) / entry
            returns.append(result)
            equity.append(equity[-1] * (1 + result))
            peak = max(peak, equity[-1])
    if not returns:
        return {"trades": 0}
    arr = np.array(returns)
    win_rate = (arr > 0).sum() / len(arr)
    avg_ret = arr.mean()
    mdd = (np.maximum.accumulate(equity) - equity).max()
    return {"trades": len(arr), "win_rate": float(win_rate), "avg_return": float(avg_ret), "max_drawdown": float(mdd)}

# ---------------- Label endpoint ----------------
@app.route("/label", methods=["POST"])
def api_label():
    secret = request.headers.get("X-ADMIN-SECRET", "")
    if ADMIN_SECRET and secret != ADMIN_SECRET:
        return jsonify({"ok": False, "error": "unauthorized"}), 401
    data = request.json or {}
    symbol = data.get("symbol")
    time_iso = data.get("time")
    outcome = bool(int(data.get("outcome", 0)))
    try:
        df = fetch_market_candles(symbol, interval="1m", limit=10000)
        if df.empty:
            return jsonify({"ok": False, "reason": "no_data"})
        idx = pd.to_datetime(time_iso)
        if idx not in df.index:
            pos = df.index.get_indexer([idx], method="nearest")
            if len(pos) == 0 or pos[0] < 0:
                return jsonify({"ok": False, "reason": "time_not_found"})
            idx = df.index[pos[0]]
        d = indicator_features(df)
        row = d.loc[idx]
        rec = {k: float(row[k]) for k in row.index}
        rec.update({"symbol": symbol, "time": str(idx), "label": int(outcome)})
        pd.DataFrame([rec]).to_csv(DATA_FILE, mode="a", header=not Path(DATA_FILE).exists(), index=False)
        return jsonify({"ok": True})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(e)}), 500

# ---------------- Telegram handlers ----------------
if bot:
    @bot.message_handler(commands=["start", "help"])
    def cmd_help(msg):
        txt = (
            "Pocket Master AI — commands:\n"
            "/analyze [SYMBOL]\n"
            "/trade [SYMBOL]\n"
            "/scalp SYMBOL TIMEFRAME\n"
            "/swing SYMBOL TIMEFRAME\n"
            "/stopmonitor KEY\n"
            "/urgent (admin)\n"
            "/train (admin)\n"
            "/train_keras (admin)\n"
            "/feedback SYMBOL TIME ISO OUTCOME(1|0) (admin)\n"
            "/status\n"
            "TV webhook: POST /tv_alert with X-TV-SECRET header\n"
            "DRY_RUN: " + str(DRY_RUN)
        )
        safe_send(msg.chat.id, txt)

    @bot.message_handler(commands=["status"])
    def cmd_status(msg):
        rf = load_rf_model()
        km = load_keras_model() if TF_AVAILABLE else None
        mstate = "RF:loaded" if rf else "RF:none"
        mstate += ", Keras:loaded" if km else ", Keras:none"
        safe_send(msg.chat.id, f"Status: {mstate}\nStrategy={_current_strategy}\nSymbols={','.join(PO_SYMBOLS)}\nActive monitors={list(_active_monitors.keys())}")

    @bot.message_handler(commands=["analyze"])
    def cmd_analyze(msg):
        parts = msg.text.split()
        sym = parts[1] if len(parts) > 1 else PO_SYMBOLS[0]
        out = predict_signals_for_symbol(sym)
        safe_send(msg.chat.id, json.dumps(out, default=str))

    @bot.message_handler(commands=["trade"])
    def cmd_trade(msg):
        parts = msg.text.split()
        sym = parts[1] if len(parts) > 1 else PO_SYMBOLS[0]
        out = predict_signals_for_symbol(sym, interval="1m")
        if not out:
            safe_send(msg.chat.id, "No model/data")
            return
        latest = out[-1]
        safe_send(msg.chat.id, f"Trade suggestion for {sym}: {latest['signal']} p={latest['fused']:.2f}\nReasons: {json.dumps(latest['reasons'], default=str)}")

    @bot.message_handler(commands=["scalp"])
    def cmd_scalp(msg):
        parts = msg.text.split()
        if len(parts) < 3:
            safe_send(msg.chat.id, "Usage: /scalp SYMBOL TIMEFRAME (e.g., /scalp BTCUSDT 1m)")
            return
        symbol = parts[1].strip()
        timeframe = parts[2].strip()
        key = f"scalp:{symbol}:{timeframe}"
        res = start_monitor_thread(key, monitor_symbol_for_trade, symbol, timeframe, "scalp")
        safe_send(msg.chat.id, f"Scalp monitor start: {res}")

    @bot.message_handler(commands=["swing"])
    def cmd_swing(msg):
        parts = msg.text.split()
        if len(parts) < 3:
            safe_send(msg.chat.id, "Usage: /swing SYMBOL TIMEFRAME (e.g., /swing BTCUSDT 1h)")
            return
        symbol = parts[1].strip()
        timeframe = parts[2].strip()
        key = f"swing:{symbol}:{timeframe}"
        res = start_monitor_thread(key, monitor_symbol_for_trade, symbol, timeframe, "swing")
        safe_send(msg.chat.id, f"Swing monitor start: {res}")

    @bot.message_handler(commands=["stopmonitor"])
    def cmd_stopmonitor(msg):
        parts = msg.text.split()
        if len(parts) < 2:
            safe_send(msg.chat.id, "Usage: /stopmonitor KEY")
            return
        key = parts[1].strip()
        res = stop_monitor(key)
        safe_send(msg.chat.id, f"Stop monitor: {res}")

    @bot.message_handler(commands=["urgent"])
    def cmd_urgent(msg):
        if not is_admin_chat(msg.chat.id, getattr(msg.from_user, "username", None)):
            safe_send(msg.chat.id, "Unauthorized")
            return
        urgent = []
        for s in PO_SYMBOLS:
            try:
                res = predict_signals_for_symbol(s, interval="1m")
                if not res:
                    continue
                r = res[-1]
                if r["signal"] == "BUY":
                    urgent.append({"symbol": s, "prob": r["fused"], "time": r["time"]})
            except Exception:
                pass
        if not urgent:
            safe_send(msg.chat.id, "No urgent signals")
            return
        urgent = sorted(urgent, key=lambda x: x["prob"], reverse=True)[:10]
        lines = [f"{u['symbol']} p={u['prob']:.3f} @ {u['time']}" for u in urgent]
        safe_send(msg.chat.id, "URGENT signals:\n" + "\n".join(lines))

    @bot.message_handler(commands=["train", "retrain"])
    def cmd_train(msg):
        if not is_admin_chat(msg.chat.id, getattr(msg.from_user, "username", None)):
            safe_send(msg.chat.id, "Unauthorized")
            return
        safe_send(msg.chat.id, "Training RF started")
        def _bg():
            try:
                res = train_rf_model_from_yf(PO_SYMBOLS)
                safe_send(msg.chat.id, f"RF training complete: {res}")
            except Exception as e:
                safe_send(msg.chat.id, f"RF training error: {e}")
                traceback.print_exc()
        threading.Thread(target=_bg, daemon=True).start()

    @bot.message_handler(commands=["train_keras"])
    def cmd_train_keras(msg):
        if not TF_AVAILABLE:
            safe_send(msg.chat.id, "TensorFlow not installed")
            return
        if not is_admin_chat(msg.chat.id, getattr(msg.from_user, "username", None)):
            safe_send(msg.chat.id, "Unauthorized")
            return
        safe_send(msg.chat.id, "Keras training started")
        threading.Thread(target=lambda: train_keras_daily(PO_SYMBOLS), daemon=True).start()

    @bot.message_handler(commands=["feedback"])
    def cmd_feedback(msg):
        parts = msg.text.split()
        if len(parts) < 4:
            safe_send(msg.chat.id, "Usage: /feedback SYMBOL TIME_ISO OUTCOME(1|0)")
            return
        if not is_admin_chat(msg.chat.id, getattr(msg.from_user, "username", None)):
            safe_send(msg.chat.id, "Unauthorized")
            return
        sym, tiso, outcome = parts[1], parts[2], parts[3]
        try:
            df = fetch_market_candles(sym, interval="1m", limit=10000)
            if df.empty:
                safe_send(msg.chat.id, "No data to label")
                return
            idx = pd.to_datetime(tiso)
            if idx not in df.index:
                pos = df.index.get_indexer([idx], method="nearest"); idx = df.index[pos[0]]
            d = indicator_features(df); row = d.loc[idx]
            rec = {k: float(row[k]) for k in row.index}; rec.update({"symbol": sym, "time": str(idx), "label": int(outcome)})
            pd.DataFrame([rec]).to_csv(DATA_FILE, mode="a", header=not Path(DATA_FILE).exists(), index=False)
            safe_send(msg.chat.id, "Feedback recorded")
        except Exception as e:
            safe_send(msg.chat.id, f"Feedback error: {e}")

# ---------------- Keepalive and schedulers ----------------
def keepalive_loop():
    while True:
        time.sleep(300)

def daily_keras_scheduler():
    if not TF_AVAILABLE:
        print("[KERAS] TensorFlow unavailable; scheduler disabled.")
        return
    while True:
        try:
            train_keras_daily(PO_SYMBOLS)
            time.sleep(24 * 3600)
        except Exception as e:
            print("daily_keras_scheduler error:", e)
            time.sleep(60)

# ---------------- Startup ----------------
def startup():
    load_rf_model()
    load_keras_model()
    if bot:
        threading.Thread(target=lambda: bot.polling(non_stop=True, timeout=60), daemon=True).start()
    if TF_AVAILABLE:
        threading.Thread(target=daily_keras_scheduler, daemon=True).start()

if __name__ == "__main__":
    try:
        threading.Thread(target=lambda: app.run(host="0.0.0.0", port=PORT, debug=False, use_reloader=False), daemon=True).start()
        startup()
        while True:
            time.sleep(5)
    except KeyboardInterrupt:
        print("Shutting down.")
    except Exception:
        traceback.print_exc()
        raise
