#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Institutional AI Trading Bot (Full standalone bot.py)
- Deployable on Choreo / Defang
- Live Bitget trading (admin-only)
- Institutional analytics integrated (liquidity, MSS, OFI, Wyckoff, VWAP, VP, etc.)
- /scalp and /swing analysis commands
- Auto take-profit & stop-loss (ATR-based)
- Uses .env for secrets & settings

.env example (create .env in same folder):
-----------------------------------------
TELEGRAM_TOKEN=123456:ABC...
TELEGRAM_ADMIN_ID=123456789
ADMIN_TOKEN=supersecrettoken
BITGET_API_KEY=...
BITGET_SECRET=...
BITGET_PASSWORD=...
BITGET_TESTNET=true            # true / false
DEFAULT_SYMBOL=BTC/USDT

# Risk / trade settings
ENABLE_LIVE_TRADES=true        # true to actually execute
TP_MULTIPLIER=2.0              # TP = entry + ATR * TP_MULTIPLIER (for long)
SL_MULTIPLIER=1.0              # SL = entry - ATR * SL_MULTIPLIER (for long)
POSITION_SIZE_USD=50           # USD notional per trade (or base asset amount if you prefer)
MAX_POSITION_PERCENT=0.02      # Max percent of account balance to risk (safety)
TRADE_LOG_CSV=trade_log.csv

# Institutional settings
INST_LIQUIDITY_LOOKBACK=50
INST_SWING_PERIOD=20
INST_VOLUME_PROFILE_BINS=50
INST_MIN_SCORE=0.75

# Note: test carefully on testnet before running live.
-----------------------------------------
Requirements (use your requirements.txt):
- Flask==2.3.2
- requests==2.31.0
- pandas==2.2.2
- numpy==1.26.4
- scikit-learn==1.3.2
- joblib==1.3.2
- python-dotenv==1.0.0
- pyTelegramBotAPI==4.12.0
- ccxt
"""

import os
import time
import math
import json
import csv
import logging
import threading
from typing import Dict, List, Optional
from functools import wraps
from dotenv import load_dotenv

# Telegram & web
import telebot
from flask import Flask, request, jsonify

# Data / math
import numpy as np
import pandas as pd
from scipy import stats
from collections import deque

# Trading
import ccxt

# Load .env
load_dotenv()

# ---------------------------
# Config / Environment
# ---------------------------
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_ADMIN_ID = int(os.getenv("TELEGRAM_ADMIN_ID", "0"))
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "")
BITGET_API_KEY = os.getenv("BITGET_API_KEY", "")
BITGET_SECRET = os.getenv("BITGET_SECRET", "")
BITGET_PASSWORD = os.getenv("BITGET_PASSWORD", "")
BITGET_TESTNET = os.getenv("BITGET_TESTNET", "true").lower() in ["1", "true", "yes"]
DEFAULT_SYMBOL = os.getenv("DEFAULT_SYMBOL", "BTC/USDT")

ENABLE_LIVE_TRADES = os.getenv("ENABLE_LIVE_TRADES", "true").lower() in ["1", "true", "yes"]
TP_MULTIPLIER = float(os.getenv("TP_MULTIPLIER", "2.0"))
SL_MULTIPLIER = float(os.getenv("SL_MULTIPLIER", "1.0"))
POSITION_SIZE_USD = float(os.getenv("POSITION_SIZE_USD", "50"))
MAX_POSITION_PERCENT = float(os.getenv("MAX_POSITION_PERCENT", "0.02"))
TRADE_LOG_CSV = os.getenv("TRADE_LOG_CSV", "trade_log.csv")

INST_LIQUIDITY_LOOKBACK = int(os.getenv("INST_LIQUIDITY_LOOKBACK", "50"))
INST_SWING_PERIOD = int(os.getenv("INST_SWING_PERIOD", "20"))
INST_VOLUME_PROFILE_BINS = int(os.getenv("INST_VOLUME_PROFILE_BINS", "50"))
INST_MIN_SCORE = float(os.getenv("INST_MIN_SCORE", "0.75"))

# Allowed timeframes for scalp & swing
SCALP_TFS = ["15m", "1h", "4h"]
SWING_TFS = ["8h", "1h", "1d", "1w", "1M"]  # include 8h and 1d etc.

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("institutional-bot")

# ---------------------------
# Validate env
# ---------------------------
if not TELEGRAM_TOKEN:
    logger.error("TELEGRAM_TOKEN not set. Exiting.")
    raise SystemExit("Missing TELEGRAM_TOKEN")

# ---------------------------
# Telegram Bot Setup
# ---------------------------
bot = telebot.TeleBot(TELEGRAM_TOKEN, parse_mode="HTML")

def admin_only(handler):
    @wraps(handler)
    def wrapper(message, *args, **kwargs):
        try:
            user_id = message.from_user.id
        except Exception:
            user_id = None
        if user_id != TELEGRAM_ADMIN_ID:
            bot.send_message(message.chat.id, "⛔ Admin-only command.")
            return
        return handler(message, *args, **kwargs)
    return wrapper

# ---------------------------
# Bitget client helper
# ---------------------------
def get_bitget_client():
    if not BITGET_API_KEY:
        return None
    # ccxt may have bitget; instantiate unified
    params = {
        'apiKey': BITGET_API_KEY,
        'secret': BITGET_SECRET,
        'password': BITGET_PASSWORD,
        'enableRateLimit': True,
    }
    if BITGET_TESTNET:
        # testnet urls for bitget in ccxt
        params['urls'] = {'api': {'public': 'https://api-testnet.bitget.com', 'private':'https://api-testnet.bitget.com'}}
    try:
        client = ccxt.bitget(params)
    except Exception:
        # fallback try constructing exchange by name
        client = getattr(ccxt, 'bitget')(params)
    return client

bitget_client = get_bitget_client()

# ---------------------------
# Utility functions
# ---------------------------
def write_trade_log(row: Dict):
    header = ["timestamp","symbol","side","amount","entry_price","tp_price","sl_price","order_id","status","notes"]
    exists = os.path.exists(TRADE_LOG_CSV)
    with open(TRADE_LOG_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not exists:
            writer.writeheader()
        writer.writerow(row)

# ---------------------------
# Market data fetcher
# ---------------------------
def fetch_market_candles(symbol: str = DEFAULT_SYMBOL, timeframe: str = "1h", limit: int = 500) -> pd.DataFrame:
    """
    Fetch OHLCV candles using ccxt and return pd.DataFrame indexed by timestamp.
    Columns: ['Open','High','Low','Close','Volume']
    """
    global bitget_client
    if bitget_client is None:
        raise RuntimeError("Bitget client not configured (BITGET_API_KEY missing).")
    ohlcv = bitget_client.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['Timestamp','Open','High','Low','Close','Volume'])
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')
    df.set_index('Timestamp', inplace=True)
    for c in ['Open','High','Low','Close','Volume']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    return df

# ---------------------------
# ATR / TP/SL calculations
# ---------------------------
def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df['High']
    low = df['Low']
    close = df['Close']
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_series = tr.rolling(period).mean()
    return atr_series

def calculate_tp_sl(entry_price: float, atr_val: float, side: str, tp_mult: float = TP_MULTIPLIER, sl_mult: float = SL_MULTIPLIER):
    if side.lower() == "buy":
        tp = entry_price + atr_val * tp_mult
        sl = entry_price - atr_val * sl_mult
    else:
        tp = entry_price - atr_val * tp_mult
        sl = entry_price + atr_val * sl_mult
    # Ensure positive
    tp = max(tp, 0.0)
    sl = max(sl, 0.0)
    return float(tp), float(sl)

# ---------------------------
# Institutional analytics (all functions integrated)
# (Adapted from your earlier pasted module)
# ---------------------------

# 1. SMART MONEY CONCEPTS (SMC)
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

# 2. VOLUME PROFILE & AUCTION THEORY
def calculate_volume_profile(df: pd.DataFrame, bins: int = INST_VOLUME_PROFILE_BINS) -> Dict:
    try:
        price_range = df['High'].max() - df['Low'].min()
        if price_range == 0 or np.isnan(price_range):
            return {'poc': None, 'vah': None, 'val': None, 'poc_volume': 0, 'profile': {}}
        bin_size = price_range / bins
        volume_profile = {}
        for i in range(bins):
            price_level = df['Low'].min() + (i * bin_size)
            volume_at_level = df[(df['Low'] <= price_level) & (df['High'] >= price_level)]['Volume'].sum()
            volume_profile[price_level] = volume_at_level
        poc_price = max(volume_profile, key=volume_profile.get)
        poc_volume = volume_profile[poc_price]
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
        return {'poc': poc_price, 'vah': vah, 'val': val, 'poc_volume': poc_volume, 'profile': volume_profile}
    except Exception:
        return {'poc': None, 'vah': None, 'val': None}

def calculate_vwap_bands(df: pd.DataFrame, std_mult: float = 2.0) -> pd.DataFrame:
    result = pd.DataFrame(index=df.index)
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    vwap = (typical_price * df['Volume']).cumsum() / (df['Volume'].cumsum() + 1e-9)
    squared_diff = (typical_price - vwap) ** 2
    variance = (squared_diff * df['Volume']).cumsum() / (df['Volume'].cumsum() + 1e-9)
    std = np.sqrt(variance)
    result['vwap'] = vwap
    result['vwap_upper_1'] = vwap + std * std_mult
    result['vwap_lower_1'] = vwap - std * std_mult
    result['vwap_upper_2'] = vwap + std * (std_mult * 2)
    result['vwap_lower_2'] = vwap - std * (std_mult * 2)
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

# 3. ORDER BOOK DYNAMICS (Simulated)
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
    iceberg = pd.Series(np.tanh(iceberg_score / 10), index=df.index)
    return iceberg

# 4. MARKET MICROSTRUCTURE
def calculate_amihud_illiquidity(df: pd.DataFrame, window: int = 20) -> pd.Series:
    returns = df['Close'].pct_change().abs()
    dollar_volume = df['Close'] * df['Volume']
    illiquidity = returns / (dollar_volume + 1e-9)
    return illiquidity.rolling(window).mean()

def calculate_market_efficiency_ratio(df: pd.DataFrame, period: int = 20) -> pd.Series:
    net_change = (df['Close'] - df['Close'].shift(period)).abs()
    sum_changes = df['Close'].diff().abs().rolling(period).sum()
    mer = net_change / (sum_changes + 1e-9)
    return mer

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

# 5. INSTITUTIONAL ENTRY/EXIT SIGNALS
def detect_accumulation_distribution(df: pd.DataFrame, window: int = 20) -> pd.Series:
    clv = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'] + 1e-9)
    ad_line = (clv * df['Volume']).cumsum()
    ad_smooth = ad_line.rolling(window).mean()
    return ad_smooth

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

# 6. MARKET REGIME DETECTION
def detect_market_regime(df: pd.DataFrame) -> Dict[str, float]:
    returns = df['Close'].pct_change()
    volatility = returns.rolling(20).std() * np.sqrt(252)
    mer = calculate_market_efficiency_ratio(df, period=20).iloc[-1]
    vol_ratio = df['Volume'].iloc[-20:].mean() / (df['Volume'].iloc[-100:].mean() + 1e-9)
    regimes = {
        'trending': float(mer > 0.5),
        'mean_reverting': float(mer < 0.3),
        'volatile': float(volatility.iloc[-1] > volatility.quantile(0.75)) if not volatility.isnull().all() else 0.0,
        'quiet': float(vol_ratio < 0.7),
        'regime_score': float(mer)
    }
    return regimes

def calculate_hurst_exponent(series: pd.Series, max_lag: int = 100) -> float:
    try:
        lags = range(2, min(max_lag, len(series) // 2))
        tau = []
        for lag in lags:
            std = np.std(np.subtract(series[lag:], series[:-lag]))
            tau.append(std)
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        hurst = poly[0]
        return float(np.clip(hurst, 0, 1))
    except Exception:
        return 0.5

# 7. STRATEGY SCORING (kept as in your module)
def strategy_smart_money_score(df: pd.DataFrame) -> float:
    score = 0.0
    liq = detect_liquidity_pools(df)
    score += 0.2 * float(liq['liquidity_above'].iloc[-1] or liq['liquidity_below'].iloc[-1])
    mss = detect_market_structure_shift(df)
    score += 0.25 * float(abs(mss.iloc[-1]))
    ofi = detect_order_flow_imbalance(df)
    score += 0.25 * float(abs(ofi.iloc[-1]))
    inst_candles = detect_institutional_candles(df)
    candle_signal = (inst_candles['bullish_engulf'].iloc[-1] + inst_candles['bullish_rejection'].iloc[-1])
    score += 0.3 * float(candle_signal > 0)
    return float(np.clip(score, 0.0, 1.0))

def strategy_volume_profile_score(df: pd.DataFrame) -> float:
    score = 0.0
    vp = calculate_volume_profile(df)
    if vp['poc'] is None:
        return 0.0
    current_price = df['Close'].iloc[-1]
    distance_to_poc = abs(current_price - vp['poc']) / current_price
    score += 0.4 * (1 - min(distance_to_poc * 100, 1.0))
    vwap_rej = detect_vwap_rejection(df)
    score += 0.3 * float(abs(vwap_rej.iloc[-1]))
    recent_vol = df['Volume'].iloc[-5:].mean()
    avg_vol = df['Volume'].iloc[-100:].mean()
    score += 0.3 * float(recent_vol > avg_vol * 1.2)
    return float(np.clip(score, 0.0, 1.0))

def strategy_wyckoff_score(df: pd.DataFrame) -> float:
    score = 0.0
    phases = detect_wyckoff_phases(df)
    score += 0.4 * float(phases['phase_d_markup'].iloc[-1])
    score += 0.3 * float(phases['phase_c_spring'].iloc[-3:].sum() > 0)
    composite = calculate_composite_index(df)
    score += 0.3 * float(np.tanh(composite.iloc[-1]))
    return float(np.clip(score, 0.0, 1.0))

def strategy_market_maker_score(df: pd.DataFrame) -> float:
    score = 0.0
    manip = detect_price_manipulation(df)
    avoid_manip = 1.0 - float(manip['pump_dump'].iloc[-1])
    score += 0.2 * avoid_manip
    pressure = simulate_orderbook_pressure(df)
    balanced = abs(pressure['pressure_ratio'].iloc[-1] - 1.0)
    score += 0.3 * (1 - min(balanced, 1.0))
    illiq = calculate_amihud_illiquidity(df)
    liquid = 1 - min(illiq.iloc[-1] * 1000, 1.0)
    score += 0.3 * float(liquid)
    mer = calculate_market_efficiency_ratio(df).iloc[-1]
    score += 0.2 * (1 - mer)
    return float(np.clip(score, 0.0, 1.0))

def strategy_institutional_flow_score(df: pd.DataFrame) -> float:
    score = 0.0
    iceberg = detect_iceberg_orders(df)
    score += 0.3 * float(abs(iceberg.iloc[-1]))
    ad = detect_accumulation_distribution(df)
    ad_slope = (ad.iloc[-1] - ad.iloc[-20]) / 20 if len(ad) > 20 else 0.0
    score += 0.4 * float(np.tanh(ad_slope / 1000))
    ofi = detect_order_flow_imbalance(df)
    score += 0.3 * float(abs(ofi.iloc[-1]))
    return float(np.clip(score, 0.0, 1.0))

# 8. Enhanced features builder
def enhanced_institutional_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    liq = detect_liquidity_pools(d)
    d['liquidity_above'] = liq['liquidity_above']
    d['liquidity_below'] = liq['liquidity_below']
    d['market_structure_shift'] = detect_market_structure_shift(d)
    d['order_flow_imbalance'] = detect_order_flow_imbalance(d)
    inst = detect_institutional_candles(d)
    d['bullish_engulf'] = inst['bullish_engulf']
    d['bearish_engulf'] = inst['bearish_engulf']
    d['bullish_rejection'] = inst['bullish_rejection']
    d['bearish_rejection'] = inst['bearish_rejection']
    vp = calculate_volume_profile(d)
    d['poc'] = vp.get('poc', None)
    d['vah'] = vp.get('vah', None)
    d['val'] = vp.get('val', None)
    vwap_bands = calculate_vwap_bands(d)
    d['vwap'] = vwap_bands['vwap']
    d['vwap_upper'] = vwap_bands['vwap_upper_1']
    d['vwap_lower'] = vwap_bands['vwap_lower_1']
    d['vwap_rejection'] = detect_vwap_rejection(d)
    pressure = simulate_orderbook_pressure(d)
    d['buy_pressure'] = pressure['buy_pressure']
    d['sell_pressure'] = pressure['sell_pressure']
    d['pressure_ratio'] = pressure['pressure_ratio']
    d['iceberg_score'] = detect_iceberg_orders(d)
    d['amihud_illiquidity'] = calculate_amihud_illiquidity(d)
    d['market_efficiency'] = calculate_market_efficiency_ratio(d)
    d['accumulation_distribution'] = detect_accumulation_distribution(d)
    d['composite_index'] = calculate_composite_index(d)
    wyckoff = detect_wyckoff_phases(d)
    for phase, series in wyckoff.items():
        d[phase] = series
    regime = detect_market_regime(d)
    for key, val in regime.items():
        d[key] = val
    d['hurst_exponent'] = calculate_hurst_exponent(d['Close'])
    return d.fillna(0)

# ---------------------------
# Trading helpers & execution (conservative)
# ---------------------------
def get_account_balance_usd():
    """
    Try to fetch USD-equivalent account balance. Implementation might vary by Bitget account type.
    For safety, this returns None when we cannot fetch easily and caller should fallback to configured POSITION_SIZE_USD.
    """
    client = bitget_client
    if client is None:
        return None
    try:
        # Try fetch balance; ccxt unified returns total for currencies
        bal = client.fetch_balance()
        # Try to compute approximate USD (USDT) value
        if 'USDT' in bal['free']:
            return float(bal['free']['USDT'])
        # fallback: compute using BTC price if available
        return None
    except Exception as e:
        logger.warning(f"Could not fetch account balance: {e}")
        return None

def place_bitget_market_order_with_tp_sl(symbol: str, side: str, volume: float, tp_price: float, sl_price: float):
    """
    Place a market order and attempt to place TP and SL orders.
    Note: exact params for stop orders depend on exchange; Bitget via ccxt may require 'reduceOnly' or 'stopPrice' param.
    This function will:
     - create market order
     - attempt to place limit TP and stop-limit/stop-market SL (best-effort)
    """
    client = bitget_client
    if client is None:
        raise RuntimeError("Bitget client not configured.")
    result = {"order": None, "tp_order": None, "sl_order": None, "notes": []}
    try:
        order = client.create_order(symbol, 'market', side, volume)
        result['order'] = order
    except Exception as e:
        result['notes'].append(f"market order failed: {e}")
        return result

    # Place TP (limit) opposite side
    try:
        tp_side = "sell" if side.lower() == "buy" else "buy"
        # create limit take-profit order
        tp_order = client.create_order(symbol, 'limit', tp_side, volume, tp_price)
        result['tp_order'] = tp_order
    except Exception as e:
        result['notes'].append(f"tp order failed: {e}")

    # Place SL - try stop market (params may differ)
    try:
        sl_side = "sell" if side.lower() == "buy" else "buy"
        # Many exchanges support stop market via params={'stopPrice': sl_price, 'type': 'STOP_MARKET'}
        params = {'stopPrice': sl_price}
        try:
            sl_order = client.create_order(symbol, 'stop_market', sl_side, volume, None, params)
        except Exception:
            # Fallback to stop-limit
            sl_order = client.create_order(symbol, 'stop_limit', sl_side, volume, sl_price, {'stopPrice': sl_price})
        result['sl_order'] = sl_order
    except Exception as e:
        result['notes'].append(f"sl order failed: {e}")

    return result

# ---------------------------
# Telegram command handlers
# ---------------------------

@bot.message_handler(commands=["start", "help"])
def cmd_start(message):
    txt = (
        "Institutional AI Trading Bot ✅\n\n"
        "Commands (admin-only for trades):\n"
        "/scalp <SYMBOL> <TF> - Short timeframe analysis & (admin) trade\n"
        "/swing <SYMBOL> <TF> - Long timeframe analysis & (admin) trade\n"
        "/institutional <SYMBOL> - Full institutional analysis\n        (shows multi-strategy scores)\n"
        "/liquidity <SYMBOL> - Liquidity pools & POC\n"
        "/wyckoff <SYMBOL> - Wyckoff phases\n"
        "/trade <SYMBOL> <buy|sell> <amount_usd_or_amount> - Place manual trade (admin-only)\n"
        "/status - Bot status\n"
        "/retrain - Trigger retrain (admin-only, placeholder)\n"
    )
    bot.send_message(message.chat.id, txt)

@bot.message_handler(commands=["status"])
def cmd_status(message):
    try:
        balance = get_account_balance_usd()
        txt = (f"Bot Status ✅\nMode: bitget/live\nDefault symbol: {DEFAULT_SYMBOL}\n"
               f"Bitget configured: {'Yes' if bitget_client else 'No'}\n"
               f"Balance(USDT): {balance if balance is not None else 'N/A'}\n"
               f"Live trades enabled: {ENABLE_LIVE_TRADES}\nAdmin ID: {TELEGRAM_ADMIN_ID}")
        bot.send_message(message.chat.id, txt)
    except Exception as e:
        bot.send_message(message.chat.id, f"Error: {e}")

@bot.message_handler(commands=["institutional"])
def cmd_institutional(message):
    parts = message.text.split()
    sym = parts[1] if len(parts) > 1 else DEFAULT_SYMBOL
    tf = parts[2] if len(parts) > 2 else "1h"
    try:
        df = fetch_market_candles(sym, timeframe=tf, limit=500)
        if df.empty:
            bot.send_message(message.chat.id, "No data available")
            return
        scores = {
            "Smart Money": strategy_smart_money_score(df),
            "Volume Profile": strategy_volume_profile_score(df),
            "Wyckoff": strategy_wyckoff_score(df),
            "Market Maker": strategy_market_maker_score(df),
            "Institutional Flow": strategy_institutional_flow_score(df)
        }
        lines = [f"INSTITUTIONAL ANALYSIS - {sym} ({tf})\n"]
        for name, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
            bars = "█" * int(score * 10)
            lines.append(f"{name}: {score:.2f} {bars}")
        regime = detect_market_regime(df)
        lines.append(f"\nMarket Regime:")
        lines.append(f"  Trending: {regime['trending']:.2f}")
        lines.append(f"  Mean Reverting: {regime['mean_reverting']:.2f}")
        lines.append(f"  Volatile: {regime['volatile']:.2f}")
        hurst = calculate_hurst_exponent(df['Close'])
        trend_type = "Trending" if hurst > 0.5 else "Mean-Reverting" if hurst < 0.5 else "Random"
        lines.append(f"\nHurst: {hurst:.3f} ({trend_type})")
        bot.send_message(message.chat.id, "\n".join(lines))
    except Exception as e:
        bot.send_message(message.chat.id, f"Error: {e}")

@bot.message_handler(commands=["liquidity"])
def cmd_liquidity(message):
    parts = message.text.split()
    sym = parts[1] if len(parts) > 1 else DEFAULT_SYMBOL
    tf = parts[2] if len(parts) > 2 else "1h"
    try:
        df = fetch_market_candles(sym, timeframe=tf, limit=300)
        if df.empty:
            bot.send_message(message.chat.id, "No data")
            return
        liq = detect_liquidity_pools(df)
        current = df['Close'].iloc[-1]
        lines = [f"LIQUIDITY ANALYSIS - {sym} ({tf})\n"]
        lines.append(f"Current Price: {current:.2f}")
        high_liq = liq['liquidity_level_high'].iloc[-1]
        low_liq = liq['liquidity_level_low'].iloc[-1]
        if high_liq and not np.isnan(high_liq):
            lines.append(f"Liquidity Above: {high_liq:.2f}  (dist {(high_liq/current-1)*100:.2f}%)")
        if low_liq and not np.isnan(low_liq):
            lines.append(f"Liquidity Below: {low_liq:.2f}  (dist {(current/low_liq-1)*100:.2f}%)")
        vp = calculate_volume_profile(df)
        if vp.get('poc'):
            lines.append(f"POC: {vp['poc']:.2f}  VAH: {vp['vah']:.2f}  VAL: {vp['val']:.2f}")
        bot.send_message(message.chat.id, "\n".join(lines))
    except Exception as e:
        bot.send_message(message.chat.id, f"Error: {e}")

@bot.message_handler(commands=["wyckoff"])
def cmd_wyckoff(message):
    parts = message.text.split()
    sym = parts[1] if len(parts) > 1 else DEFAULT_SYMBOL
    tf = parts[2] if len(parts) > 2 else "1h"
    try:
        df = fetch_market_candles(sym, timeframe=tf, limit=300)
        if df.empty:
            bot.send_message(message.chat.id, "No data")
            return
        phases = detect_wyckoff_phases(df)
        composite = calculate_composite_index(df).iloc[-1]
        lines = [f"WYCKOFF ANALYSIS - {sym} ({tf})\n"]
        recent = 10
        if phases['phase_a_stopping'].iloc[-recent:].sum() > 0:
            lines.append("Phase A: Stopping action detected")
        if phases['phase_b_accumulation'].iloc[-recent:].sum() > 0:
            lines.append("Phase B: Accumulation in progress")
        if phases['phase_c_spring'].iloc[-recent:].sum() > 0:
            lines.append("Phase C: SPRING detected - bullish setup!")
        if phases['phase_d_markup'].iloc[-recent:].sum() > 0:
            lines.append("Phase D: Markup in progress")
        lines.append(f"\nComposite Man Index: {composite:.3f}")
        if composite > 0.3:
            lines.append("Institutional Bias: BULLISH")
        elif composite < -0.3:
            lines.append("Institutional Bias: BEARISH")
        else:
            lines.append("Institutional Bias: NEUTRAL")
        bot.send_message(message.chat.id, "\n".join(lines))
    except Exception as e:
        bot.send_message(message.chat.id, f"Error: {e}")

# /scalp command: short TF analysis and (admin) optional trade
@bot.message_handler(commands=["scalp"])
def cmd_scalp(message):
    """
    /scalp SYMBOL TF [trade]
    TF must be one of SCALP_TFS e.g. 15m,1h,4h
    If admin sends "trade" as 4th token, will attempt trade (admin-only enforced)
    """
    parts = message.text.split()
    if len(parts) < 3:
        bot.send_message(message.chat.id, "Usage: /scalp SYMBOL TF  (TF: 15m,1h,4h)")
        return
    sym = parts[1]
    tf = parts[2]
    want_trade = len(parts) >= 4 and parts[3].lower() in ["trade", "execute", "go"]
    if tf not in SCALP_TFS:
        bot.send_message(message.chat.id, f"Invalid TF. Allowed scalp TFs: {', '.join(SCALP_TFS)}")
        return
    try:
        df = fetch_market_candles(sym, timeframe=tf, limit=500)
        if df.empty:
            bot.send_message(message.chat.id, "No data")
            return
        # Compute features & scores
        score = strategy_smart_money_score(df)
        vp_score = strategy_volume_profile_score(df)
        composite = calculate_composite_index(df).iloc[-1]
        hurst = calculate_hurst_exponent(df['Close'])
        lines = [f"SCALP ANALYSIS - {sym} ({tf})"]
        lines.append(f"SmartMoney: {score:.2f}  VP: {vp_score:.2f}  Composite: {composite:.3f}")
        lines.append(f"Hurst: {hurst:.3f}")
        bot.send_message(message.chat.id, "\n".join(lines))
        # If admin requested trade, require admin_only verification
        if want_trade:
            if message.from_user.id != TELEGRAM_ADMIN_ID:
                bot.send_message(message.chat.id, "⛔ Admin-only trade. Access denied.")
                return
            # Determine side from composite + smart money logic (simple example)
            side = "buy" if composite > 0 else "sell"
            entry_price = float(df['Close'].iloc[-1])
            atr_val = float(atr(df).iloc[-1] if not atr(df).isnull().all() else 0.0)
            if atr_val <= 0:
                bot.send_message(message.chat.id, "ATR calculation failed, aborting trade.")
                return
            tp_price, sl_price = calculate_tp_sl(entry_price, atr_val, side)
            # Determine volume from POSITION_SIZE_USD / entry_price
            amount = float(POSITION_SIZE_USD / entry_price)
            # Safety: check balance / exposure
            balance_usd = get_account_balance_usd()
            if balance_usd is not None and (POSITION_SIZE_USD > balance_usd * MAX_POSITION_PERCENT):
                bot.send_message(message.chat.id, f"Position size exceeds MAX_POSITION_PERCENT of balance. Abort.")
                return
            # Execute trade (live)
            if not ENABLE_LIVE_TRADES:
                bot.send_message(message.chat.id, f"DRY RUN: would {side} {amount:.6f} {sym} at {entry_price:.2f} TP:{tp_price:.2f} SL:{sl_price:.2f}")
                return
            # Place market order and TP/SL
            try:
                result = place_bitget_market_order_with_tp_sl(sym, side, amount, tp_price, sl_price)
                write_trade_log({
                    "timestamp": int(time.time()),
                    "symbol": sym,
                    "side": side,
                    "amount": amount,
                    "entry_price": entry_price,
                    "tp_price": tp_price,
                    "sl_price": sl_price,
                    "order_id": result.get('order', {}).get('id', 'N/A'),
                    "status": "placed" if result.get('order') else "failed",
                    "notes": ";".join(result.get('notes', []))
                })
                bot.send_message(message.chat.id, f"Trade result: {json.dumps(result, default=str)}")
            except Exception as e:
                bot.send_message(message.chat.id, f"Trade error: {e}")

    except Exception as e:
        bot.send_message(message.chat.id, f"Error: {e}")

# /swing command: long TF analysis and optional admin trade
@bot.message_handler(commands=["swing"])
def cmd_swing(message):
    """
    /swing SYMBOL TF [trade]
    TF must be one of SWING_TFS e.g. 8h,1d,1w,1M
    """
    parts = message.text.split()
    if len(parts) < 3:
        bot.send_message(message.chat.id, "Usage: /swing SYMBOL TF  (TF: 8h,1d,1w,1M)")
        return
    sym = parts[1]
    tf = parts[2]
    want_trade = len(parts) >= 4 and parts[3].lower() in ["trade", "execute", "go"]
    if tf not in SWING_TFS:
        bot.send_message(message.chat.id, f"Invalid TF. Allowed swing TFs: {', '.join(SWING_TFS)}")
        return
    try:
        df = fetch_market_candles(sym, timeframe=tf, limit=500)
        if df.empty:
            bot.send_message(message.chat.id, "No data")
            return
        # Scores
        smart = strategy_smart_money_score(df)
        wyck = strategy_wyckoff_score(df)
        composite = calculate_composite_index(df).iloc[-1]
        hurst = calculate_hurst_exponent(df['Close'])
        lines = [f"SWING ANALYSIS - {sym} ({tf})"]
        lines.append(f"SmartMoney: {smart:.2f}  Wyckoff: {wyck:.2f}  Composite: {composite:.3f}")
        lines.append(f"Hurst: {hurst:.3f}")
        bot.send_message(message.chat.id, "\n".join(lines))
        # Optional trade (admin-only)
        if want_trade:
            if message.from_user.id != TELEGRAM_ADMIN_ID:
                bot.send_message(message.chat.id, "⛔ Admin-only trade. Access denied.")
                return
            side = "buy" if composite > 0 else "sell"
            entry_price = float(df['Close'].iloc[-1])
            atr_val = float(atr(df).iloc[-1] if not atr(df).isnull().all() else 0.0)
            if atr_val <= 0:
                bot.send_message(message.chat.id, "ATR calculation failed, aborting trade.")
                return
            tp_price, sl_price = calculate_tp_sl(entry_price, atr_val, side, tp_mult=TP_MULTIPLIER, sl_mult=SL_MULTIPLIER)
            amount = float(POSITION_SIZE_USD / entry_price)
            balance_usd = get_account_balance_usd()
            if balance_usd is not None and (POSITION_SIZE_USD > balance_usd * MAX_POSITION_PERCENT):
                bot.send_message(message.chat.id, f"Position size exceeds MAX_POSITION_PERCENT of balance. Abort.")
                return
            if not ENABLE_LIVE_TRADES:
                bot.send_message(message.chat.id, f"DRY RUN: would {side} {amount:.6f} {sym} at {entry_price:.2f} TP:{tp_price:.2f} SL:{sl_price:.2f}")
                return
            try:
                result = place_bitget_market_order_with_tp_sl(sym, side, amount, tp_price, sl_price)
                write_trade_log({
                    "timestamp": int(time.time()),
                    "symbol": sym,
                    "side": side,
                    "amount": amount,
                    "entry_price": entry_price,
                    "tp_price": tp_price,
                    "sl_price": sl_price,
                    "order_id": result.get('order', {}).get('id', 'N/A'),
                    "status": "placed" if result.get('order') else "failed",
                    "notes": ";".join(result.get('notes', []))
                })
                bot.send_message(message.chat.id, f"Trade result: {json.dumps(result, default=str)}")
            except Exception as e:
                bot.send_message(message.chat.id, f"Trade error: {e}")

    except Exception as e:
        bot.send_message(message.chat.id, f"Error: {e}")

# Manual /trade command (admin-only)
@bot.message_handler(commands=["trade"])
@admin_only
def cmd_trade(message):
    """
    /trade SYMBOL buy|sell amount_usd
    Example: /trade BTC/USDT buy 50
    """
    parts = message.text.split()
    if len(parts) < 4:
        bot.send_message(message.chat.id, "Usage: /trade SYMBOL buy|sell amount_usd")
        return
    sym = parts[1]
    side = parts[2].lower()
    try:
        notional = float(parts[3])
    except Exception:
        bot.send_message(message.chat.id, "Invalid amount.")
        return
    try:
        df = fetch_market_candles(sym, timeframe="1h", limit=50)
        entry_price = float(df['Close'].iloc[-1])
        amount = notional / entry_price
        atr_val = float(atr(df).iloc[-1] if not atr(df).isnull().all() else 0.0)
        if atr_val <= 0:
            bot.send_message(message.chat.id, "ATR calculation failed, aborting trade.")
            return
        tp_price, sl_price = calculate_tp_sl(entry_price, atr_val, side)
        if not ENABLE_LIVE_TRADES:
            bot.send_message(message.chat.id, f"DRY RUN: would {side} {amount:.6f} {sym} at {entry_price:.2f} TP:{tp_price:.2f} SL:{sl_price:.2f}")
            return
        result = place_bitget_market_order_with_tp_sl(sym, side, amount, tp_price, sl_price)
        write_trade_log({
            "timestamp": int(time.time()),
            "symbol": sym,
            "side": side,
            "amount": amount,
            "entry_price": entry_price,
            "tp_price": tp_price,
            "sl_price": sl_price,
            "order_id": result.get('order', {}).get('id', 'N/A'),
            "status": "placed" if result.get('order') else "failed",
            "notes": ";".join(result.get('notes', []))
        })
        bot.send_message(message.chat.id, f"Trade result: {json.dumps(result, default=str)}")
    except Exception as e:
        bot.send_message(message.chat.id, f"Trade error: {e}")

@bot.message_handler(commands=["retrain"])
@admin_only
def cmd_retrain(message):
    # Placeholder for Keras retrain pipeline — implement as needed
    bot.send_message(message.chat.id, "Retrain triggered (placeholder). Implement your retrain routine in the code.")
    # Example: start background thread to retrain models and save artifacts

# ---------------------------
# Background: start Telegram polling (non-blocking for Flask)
# ---------------------------
def start_telegram_polling():
    logger.info("Starting Telegram polling...")
    bot.infinity_polling(timeout=60, long_polling_timeout=60)

polling_thread = threading.Thread(target=start_telegram_polling, daemon=True)
polling_thread.start()

# ---------------------------
# Flask keepalive for Choreo
# ---------------------------
app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return "Institutional AI Trading Bot — Running ✅"

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "bitget_configured": bool(bitget_client),
        "live_trades": ENABLE_LIVE_TRADES
    })

# Admin HTTP endpoint for retrain (requires ADMIN_TOKEN header)
@app.route("/admin/retrain", methods=["POST"])
def http_retrain():
    token = request.headers.get("Authorization", "")
    if token != f"Bearer {ADMIN_TOKEN}":
        return jsonify({"error": "unauthorized"}), 401
    # Kick off retrain in background (placeholder)
    threading.Thread(target=lambda: logger.info("HTTP retrain placeholder started"), daemon=True).start()
    return jsonify({"status": "retrain_started"})

# ---------------------------
# Entrypoint
# ---------------------------
if __name__ == "__main__":
    # When running locally (for testing), Flask will be started (Choreo will run this container)
    port = int(os.getenv("PORT", "8080"))
    logger.info(f"Starting Flask on 0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port)
