# shivshakti_final_enhanced_with_Detailed_Marathi_Reasoning.py - Streamlit App with Advanced Indicators and Enhanced Marathi Logic

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import datetime as dt
import requests
from bs4 import BeautifulSoup
from typing import Tuple, Dict, Any

# *************************************************************************
# 1. NSE-specific import 
# *************************************************************************
try:
    from nsepy import get_quote
except ImportError:
    # Define a dummy function to prevent crash if nsepy is missing
    def get_quote(symbol):
        return pd.DataFrame()


# ---------------- Page Configuration ----------------
st.set_page_config(
    page_title="📊 NSE & YF Hybrid Intraday Scanner (Advanced)",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- Sidebar Settings ----------------
st.sidebar.title("⚙️ Settings")
refresh_sec = st.sidebar.number_input("Auto-refresh (seconds)", min_value=5, max_value=120, value=30, step=5)
history_period = st.sidebar.selectbox("History period (yfinance)", ["1d", "2d", "5d"], index=0)
interval = st.sidebar.selectbox("Chart interval (yfinance)", ["1m", "2m", "5m", "15m"], index=2) # Default 5m
show_download = st.sidebar.checkbox("Show Report Download Button", value=True) 

# --- Stock List (Dynamic Input) ---
default_stocks = ", ".join([
    "INFY.NS", "TCS.NS", "HCLTECH.NS", "LT.NS", "SBIN.NS", "AXISBANK.NS", "RELIANCE.NS",
    "ADANIENT.NS", "HINDUNILVR.NS", "MARUTI.NS", "HDFCBANK.NS"
])
stock_input = st.sidebar.text_area(
    "Stocks (comma separated, e.g., INFY.NS, TCS.NS)",
    value=default_stocks,
    height=150
)
STOCKS = [s.strip().upper() for s in stock_input.split(',') if s.strip()]

if not STOCKS:
    st.error("Please enter at least one stock symbol in the sidebar.")
    st.stop()

# --- Force Refresh Data Now button is removed as requested. ---


# ---------------- Utility Functions (with Caching) ----------------
@st.cache_data(ttl=refresh_sec)
def fetch_ohlcv_yf(ticker: str, period: str, interval: str) -> pd.DataFrame:
    """Fetches OHLCV data from yfinance, optimized with Streamlit Caching."""
    try:
        df = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=False, actions=False, prepost=True)
        if not df.empty:
            df = df.loc[:, ["Open", "High", "Low", "Close", "Volume"]]
            if df.index.tz is None:
                df.index = df.index.tz_localize('UTC').tz_convert('Asia/Kolkata')
            else:
                df.index = df.index.tz_convert('Asia/Kolkata')
            return df.dropna(how="all")
    except Exception:
        pass
    return pd.DataFrame()

@st.cache_data(ttl=5) # Aggressive Caching for NSE data (5 seconds TTL)
def fetch_nse_live_ltp(ticker: str) -> Tuple[float, float]:
    """Fetches the latest LTP and Previous Close from NSE (using nsepy)."""
    try:
        symbol = ticker.replace(".NS", "")
        quote = get_quote(symbol) 
        if not quote.empty:
            ltp = float(quote['lastPrice'].iloc[0])
            prev_close = float(quote['previousClose'].iloc[0])
            return ltp, prev_close
    except Exception:
        pass
    return np.nan, np.nan

@st.cache_data(ttl=3600)
def get_trend_colors(sym: str):
    """Fetches data and calculates trend colors for the last 10 trading days."""
    try:
        df = yf.Ticker(sym).history(period="15d", interval="1d")
        if df.empty or len(df) < 2: return ["#6c757d"] * 10
        df["Change"] = df["Close"].diff()
        changes = df["Change"].iloc[1:].tail(10) 
        colors = []
        for ch in changes:
            if pd.isna(ch):
                colors.append("#adb5bd")
            else:
                colors.append("#28a745" if ch > 0 else ("#dc3545" if ch < 0 else "#adb5bd"))
        if len(colors) < 10:
             colors = ["#6c757d"] * (10 - len(colors)) + colors
        return colors[-10:]
    except Exception: 
        return ["#6c757d"] * 10

def safe_last(df: pd.DataFrame, col: str):
    try:
        val = float(df[col].iloc[-1])
        return val if not pd.isna(val) else np.nan
    except Exception:
        return np.nan

def calc_vwap(df: pd.DataFrame) -> float:
    if df.empty or "Volume" not in df.columns or df["Volume"].sum() == 0:
        return np.nan
    typical = df["Close"] 
    cum_pv = (typical * df["Volume"]).cumsum()
    cum_vol = df["Volume"].cumsum()
    vwap = (cum_pv / cum_vol).ffill()
    return round(float(vwap.iloc[-1]), 2)

def calc_rsi(series: pd.Series, period=14) -> float:
    try:
        s = series.dropna()
        if s.empty or len(s) < period + 1: return np.nan
        delta = s.diff()
        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)
        ma_up = up.ewm(alpha=1/period, adjust=False).mean()
        ma_down = down.ewm(alpha=1/period, adjust=False).mean()
        rs = ma_up / (ma_down + 1e-9)
        rsi = 100 - (100 / (1 + rs))
        return round(float(rsi.iloc[-1]), 2)
    except Exception: return np.nan

def calc_macd(series: pd.Series) -> Tuple[float, float]:
    try:
        s = series.dropna()
        if len(s) < 26: return (np.nan, np.nan)
        ema12 = s.ewm(span=12, adjust=False).mean()
        ema26 = s.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal_line = macd.ewm(span=9, adjust=False).mean()
        return round(float(macd.iloc[-1]), 3), round(float(signal_line.iloc[-1]), 3)
    except Exception: return (np.nan, np.nan)

def calc_atr(df: pd.DataFrame, period=14) -> float:
    try:
        df['H_L'] = df['High'] - df['Low']
        df['H_PC'] = abs(df['High'] - df['Close'].shift(1))
        df['L_PC'] = abs(df['Low'] - df['Close'].shift(1))
        df['TR'] = df[['H_L', 'H_PC', 'L_PC']].max(axis=1, skipna=False)
        atr = df['TR'].ewm(span=period, adjust=False).mean()
        return round(float(atr.iloc[-1]), 2)
    except Exception: return np.nan

def calc_ema(series: pd.Series, period: int) -> float:
    try:
        if len(series) < period: return np.nan
        ema = series.ewm(span=period, adjust=False).mean()
        return round(float(ema.iloc[-1]), 2)
    except Exception:
        return np.nan

def calc_bband(series: pd.Series, period=20, num_std=2) -> Tuple[float, float, float]:
    try:
        if len(series) < period: return np.nan, np.nan, np.nan
        ma = series.rolling(window=period).mean()
        std = series.rolling(window=period).std()
        upper = ma + (std * num_std)
        lower = ma - (std * num_std)
        return round(float(ma.iloc[-1]), 2), round(float(upper.iloc[-1]), 2), round(float(lower.iloc[-1]), 2)
    except Exception:
        return np.nan, np.nan, np.nan

@st.cache_data(ttl=300)
def get_stock_news(symbol: str, limit: int = 2):
    try:
        query = symbol.replace(".NS", "") + " stock"
        url = f"https://news.google.com/rss/search?q={query}+site:moneycontrol.com+OR+site:economictimes.indiatimes.com&hl=en-IN&gl=IN&ceid=IN:en"
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.content, "xml")
        items = soup.find_all("item")[:limit]
        news_list = []
        for item in items:
            title = item.title.text
            link = item.link.text
            pub_date = item.pubDate.text if item.pubDate else "-"
            news_list.append({"title": title, "link": link, "pub_date": pub_date})
        return news_list if news_list else [{"title": "No recent news found.", "link": "#", "pub_date": "-"}]
    except Exception:
        return [{"title": "Unable to fetch news.", "link": "#", "pub_date": "-"}]

def get_signal_background_color(signal: str) -> str:
    if 'BUY' in signal: return '#e8f5e9'
    elif 'SELL' in signal: return '#ffebee'
    else: return '#f5f5f5'

def get_signal_color(signal: str) -> str:
    if 'STRONG BUY' in signal: return '#198754'
    elif 'BUY' in signal: return '#28a745'
    elif 'STRONG SELL' in signal: return '#dc3545'
    elif 'SELL' in signal: return '#ff4d4d'
    else: return '#6c757d'


# ---------------- Analysis Logic ----------------
def analyze_symbol(sym: str, period: str, interval: str) -> Dict[str, Any]:
    
    row = {"Symbol": sym, "Last_Timestamp": "-", "LTP": np.nan,
           "VWAP": np.nan, "RSI": np.nan, "MACD": np.nan, "MACD_SIGNAL": np.nan,
           "Signal": "NEUTRAL", "Reason": "", "Buy_Price": np.nan,
           "Sell_Price": np.nan, "Range": np.nan, "Target": np.nan,
           "Stoploss": np.nan, "Score": np.nan, "Movement": "", "Movement_Detail": "",
           "W52_High": np.nan, "W52_Low": np.nan, "W52_Status_Pct": np.nan,
           "Company_Details": "Company details not available.", "Last_Dividend": np.nan, 
           "Daily_Change_Pct": np.nan, "Volume_Spike": "Normal", "ATR": np.nan,
           "Day_High": np.nan, "Day_Low": np.nan, 
           "PE_Ratio": np.nan, "PB_Ratio": np.nan, "Debt_Equity": np.nan, "EPS": np.nan,
           "Fundamental_Status": "N/A - Data not available.",
           "EMA_20": np.nan, "EMA_50": np.nan, "BB_Mid": np.nan, "BB_Upper": np.nan, "BB_Lower": np.nan
           } 

    # --- 1. Fetch Yfinance OHLCV ---
    df = fetch_ohlcv_yf(sym, period, interval)
    if df.empty:
        row["Reason"] = "No OHLCV data available."
        return row
    
    # --- 2. Fetch NSE Live LTP ---
    nse_ltp, nse_prev_close = fetch_nse_live_ltp(sym)
    
    ltp = nse_ltp if not np.isnan(nse_ltp) else safe_last(df, "Close")

    if not np.isnan(ltp):
        df.loc[df.index[-1], 'Close'] = ltp 
        
    ts = df.index[-1].strftime("%H:%M:%S")

    # --- Day High/Low Calculation ---
    if not df.empty:
        row["Day_High"] = round(df["High"].max(), 2)
        row["Day_Low"] = round(df["Low"].min(), 2)
    

    # --- Daily Percentage Change / Volume Spike ---
    prev_close = nse_prev_close if not np.isnan(nse_prev_close) else df["Close"].iloc[0]
    
    if not np.isnan(ltp) and not np.isnan(prev_close) and prev_close != 0:
        daily_change = (ltp - prev_close) / prev_close * 100
        row["Daily_Change_Pct"] = round(daily_change, 2)
    
    current_volume = safe_last(df, "Volume")
    if len(df) > 20:
        avg_volume = df["Volume"].iloc[-21:-1].mean()
        if not np.isnan(current_volume) and not np.isnan(avg_volume) and avg_volume > 0:
            if current_volume >= (avg_volume * 1.5): 
                row["Volume_Spike"] = "SPIKE"

    # --- Indicators and Company Info ---
    try:
        ticker_info = yf.Ticker(sym).info
        row["W52_High"] = round(ticker_info.get('fiftyTwoWeekHigh', np.nan), 2)
        row["W52_Low"] = round(ticker_info.get('fiftyTwoWeekLow', np.nan), 2)
        row["Last_Dividend"] = round(ticker_info.get('lastDividendValue', np.nan), 3)
        summary = ticker_info.get('longBusinessSummary', 'Company details not available.')
        row["Company_Details"] = summary[:500] + "..." if len(summary) > 500 else summary 
        
        # --- FUNDAMENTAL DATA EXTRACTION ---
        row["PE_Ratio"] = round(ticker_info.get('trailingPE', np.nan), 2)
        row["PB_Ratio"] = round(ticker_info.get('priceToBook', np.nan), 2)
        debt_to_equity_val = ticker_info.get('debtToEquity', np.nan)
        row["Debt_Equity"] = round(debt_to_equity_val, 2) if not np.isnan(debt_to_equity_val) else np.nan
        row["EPS"] = round(ticker_info.get('trailingEps', np.nan), 2)
        
        # --- SIMPLE FUNDAMENTAL STATUS LOGIC ---
        pe = row["PE_Ratio"]
        de = row["Debt_Equity"]
        
        fund_status = "Neutral Status"
        
        if not np.isnan(pe) and not np.isnan(de):
            if pe > 0:
                if pe < 20 and de < 1.0:
                    fund_status = "Strong Value (Low P/E & D/E)"
                elif pe > 35 and de > 1.5:
                    fund_status = "High Risk (Expensive & High Debt)"
                elif pe < 15 and de < 0.5:
                    fund_status = "Deep Value & Low Debt"
                elif pe > 30:
                    fund_status = "Growth Stock (High P/E)"
                elif de > 2.0:
                    fund_status = "High Debt Concern"
            elif pe < 0:
                 fund_status = "Loss Making (Negative P/E)"
        
        row["Fundamental_Status"] = fund_status

        w52_high = row["W52_High"]
        w52_low = row["W52_Low"]
        
        if not np.isnan(ltp) and not np.isnan(w52_high) and not np.isnan(w52_low) and (w52_high - w52_low) > 0:
            status_pct = ((ltp - w52_low) / (w52_high - w52_low)) * 100
            row["W52_Status_Pct"] = round(status_pct, 2)

    except Exception: pass

    # Technical Indicators calculation (existing logic)
    vwap = calc_vwap(df)
    rsi = calc_rsi(df["Close"])
    macd, macd_sig = calc_macd(df["Close"])
    atr = calc_atr(df) 
    bb_mid, bb_upper, bb_lower = calc_bband(df["Close"])
    ema_20 = calc_ema(df["Close"], 20)
    ema_50 = calc_ema(df["Close"], 50)
    
    # Update row with new indicators
    row.update({
        "EMA_20": ema_20, "EMA_50": ema_50, 
        "BB_Mid": bb_mid, "BB_Upper": bb_upper, "BB_Lower": bb_lower,
    })

    # ---------------------------------------------------------------------------------------
    # --- ENHANCED SIGNAL AND REASONING LOGIC (Marathi Detailed) ---
    # ---------------------------------------------------------------------------------------
    
    signal = "NEUTRAL"
    score = 50
    primary_reason = "इंडिकेटर्समध्ये विरोधाभास आहे किंवा सिग्नलची तीव्रता कमी आहे. (उदा. EMA बुलिश, पण MACD बेअरिश)"
    detailed_reason_parts = []

    if not np.isnan(ltp):
        
        # --- 1. VWAP Check (Institutional Strength) ---
        if not np.isnan(vwap) and ltp > vwap:
            detailed_reason_parts.append(f"LTP ({ltp:.2f}) VWAP ({vwap:.2f}) च्या वर आहे, जो **तेजीचा (Bullish) दबाव** दर्शवतो.")
        elif not np.isnan(vwap) and ltp < vwap:
            detailed_reason_parts.append(f"LTP ({ltp:.2f}) VWAP ({vwap:.2f}) च्या खाली आहे, जो **मंदीचा (Bearish) दबाव** दर्शवतो.")
            
        # --- 2. EMA Check (Trend Confirmation) ---
        if not np.isnan(ema_20) and ltp > ema_20:
            detailed_reason_parts.append(f"किंमत (Price) २० EMA ({ema_20:.2f}) च्या वर आहे. याचा अर्थ **शॉर्ट-टर्म ट्रेंड मजबूत** आहे.")
        elif not np.isnan(ema_20) and ltp < ema_20:
            detailed_reason_parts.append(f"किंमत (Price) २० EMA ({ema_20:.2f}) च्या खाली आहे. याचा अर्थ **शॉर्ट-टर्म ट्रेंड कमजोर** आहे.")

        # --- 3. MACD Check (Momentum Direction) ---
        if not np.isnan(macd) and not np.isnan(macd_sig) and macd > macd_sig:
            detailed_reason_parts.append(f"MACD लाईन सिग्नल लाईनच्या वर आहे, जो **खरेदीचा मोमेंटम (Buying Momentum)** वाढल्याचे दर्शवतो.")
        elif not np.isnan(macd) and not np.isnan(macd_sig) and macd < macd_sig:
            detailed_reason_parts.append(f"MACD लाईन सिग्नल लाईनच्या खाली आहे, जो **विक्रीचा मोमेंटम (Selling Momentum)** वाढल्याचे दर्शवतो.")
            
        # --- 4. RSI Check (Strength/Overbought/Oversold) ---
        if not np.isnan(rsi):
            if rsi > 70:
                detailed_reason_parts.append(f"RSI {rsi:.2f} **अति-खरेदीच्या (Overbought)** क्षेत्रात आहे.")
            elif rsi < 30:
                detailed_reason_parts.append(f"RSI {rsi:.2f} **अति-विक्रीच्या (Oversold)** क्षेत्रात आहे.")
            elif rsi > 55:
                detailed_reason_parts.append(f"RSI {rsi:.2f} **सकारात्मक (Positive)** क्षेत्रात आहे.")
            elif rsi < 45:
                detailed_reason_parts.append(f"RSI {rsi:.2f} **नकारात्मक (Negative)** क्षेत्रात आहे.")
        
        # --- Core Signal Logic (Set the signal and primary reason) ---
        
        if ltp > vwap and rsi > 60 and macd > macd_sig and ltp > ema_20:
            signal = "STRONG BUY"
            score = 95
            primary_reason = "चारही महत्त्वाचे इंडिकेटर्स (VWAP, RSI, MACD, EMA) **स्पष्टपणे खरेदीच्या (Bullish)** दिशेने आहेत. हा उच्च विश्वासाचा (High Conviction) सिग्नल आहे."
        elif ltp < vwap and rsi < 40 and macd < macd_sig and ltp < ema_20:
            signal = "STRONG SELL"
            score = 95
            primary_reason = "चारही महत्त्वाचे इंडिकेटर्स (VWAP, RSI, MACD, EMA) **स्पष्टपणे विक्रीच्या (Bearish)** दिशेने आहेत. हा उच्च विश्वासाचा (High Conviction) सिग्नल आहे."
        elif ltp > vwap and rsi > 50:
            signal = "BUY"
            score = 75
            primary_reason = "किंमत VWAP च्या वर आहे आणि RSI ५० च्या वर आहे. मोमेंटम वाढत आहे."
        elif ltp < vwap and rsi < 50:
            signal = "SELL"
            score = 75
            primary_reason = "किंमत VWAP च्या खाली आहे आणि RSI ५० च्या खाली आहे. विक्रीचा दबाव आहे."

    # Combine primary reason and detailed indicator status
    row["Reason"] = primary_reason + " **| इंडिकेटर स्थिती:** " + " ".join(detailed_reason_parts)
    
    # ---------------------------------------------------------------------------------------
    # --- ENHANCED MOVEMENT NATURE LOGIC (Marathi Detailed) ---
    # ---------------------------------------------------------------------------------------

    movement = "अस्पष्ट / अनिर्णय (Indecision)"
    movement_detail = "किंमत (Price) VWAP आणि EMA च्या जवळ ट्रेड करत आहे, ज्यामुळे बाजारात दिशा निश्चित नाही. मोठ्या मुव्हमेंटची वाट पहा."
    
    if not np.isnan(rsi) and not np.isnan(ltp):
        bb_width_pct = (bb_upper - bb_lower) / ltp * 100 if not np.isnan(bb_upper) and not np.isnan(bb_lower) and ltp != 0 else np.nan

        if rsi > 75:
            movement = "🥵 तीव्र खरेदी (Extremely Overbought)"
            movement_detail = f"RSI **७५ च्या वर** आहे. स्टॉक खूप जास्त खरेदी झाला आहे, ज्यामुळे कोणत्याही क्षणी **मोठा रिव्हर्सल** (Reversal) किंवा **प्रॉफिट बुकिंग** होऊ शकते."
        elif rsi > 70:
            movement = "🔥 अति-खरेदी क्षेत्र (Overbought Zone)"
            movement_detail = f"RSI {rsi:.2f} हा ७० च्या वर आहे. सध्याचा ट्रेंड मजबूत आहे, पण तेजी थकू शकते. **ट्रेडिंग करताना सावधगिरी बाळगा.**"
        elif rsi < 25:
            movement = "🥶 तीव्र विक्री (Extremely Oversold)"
            movement_detail = f"RSI **२५ च्या खाली** आहे. स्टॉक खूप जास्त विकला गेला आहे, ज्यामुळे कोणत्याही क्षणी **बाऊन्स बॅक** (Bounce Back) किंवा **शॉर्ट कव्हरिंग** होऊ शकते."
        elif rsi < 30:
            movement = "🧊 अति-विक्री क्षेत्र (Oversold Zone)"
            movement_detail = f"RSI {rsi:.2f} हा ३० च्या खाली आहे. विक्रीचा दबाव जास्त आहे, पण लवकरच बाऊन्स होण्याची शक्यता आहे."
        elif not np.isnan(bb_width_pct) and bb_width_pct < 1.0: # If range is very narrow (less than 1% of LTP)
             movement = "🤏 Low Volatility Squeeze (Breakout Pending)"
             movement_detail = f"**Bollinger Bands खूप अरुंद (Narrow) झाले आहेत** ({bb_width_pct:.2f}% रुंदी). व्होलाटिलिटी कमी आहे, पण लवकरच **मोठा ब्रेकआऊट** (Breakout) दोन्हीपैकी एका दिशेने होण्याची शक्यता आहे."
        elif not np.isnan(vwap) and (ltp > vwap and ltp > ema_20):
             movement = "🚀 मजबूत ट्रेंड (Strong Bullish Trend)"
             movement_detail = "स्टॉक VWAP आणि EMA च्या वर ट्रेड करत आहे. सध्याचा ट्रेंड तेजीचा आहे, ट्रेलिंग स्टॉपलॉसचा (Trailing Stoploss) वापर करा."
        elif not np.isnan(vwap) and (ltp < vwap and ltp < ema_20):
             movement = "📉 मजबूत मंदी (Strong Bearish Trend)"
             movement_detail = "स्टॉक VWAP आणि EMA च्या खाली ट्रेड करत आहे. सध्याचा ट्रेंड मंदीचा आहे, सावधगिरी बाळगा."

    # ---------------------------------------------------------------------------------------
    # --- END OF ENHANCED LOGIC ---
    # ---------------------------------------------------------------------------------------

    # --- Dynamic SL/Target using ATR ---
    if not np.isnan(ltp):
        if not np.isnan(atr) and atr > 0:
            stoploss_val = round(ltp - atr, 2) if "BUY" in signal else round(ltp + atr, 2)
            target_val = round(ltp + (atr * 2), 2) if "BUY" in signal else round(ltp - (atr * 2), 2)
        else:
            stoploss_val = round(ltp * (0.985 if "BUY" in signal else 1.015), 2) 
            target_val = round(ltp * (1.03 if "BUY" in signal else 0.97), 2)     
        
        buy_price = round(ltp * 0.999, 2) if "BUY" in signal else np.nan
        sell_price = round(ltp * 1.001, 2) if "SELL" in signal else np.nan
        range_val = round(abs((target_val if not np.isnan(target_val) else ltp) - (stoploss_val if not np.isnan(stoploss_val) else ltp)), 2)
    else:
        buy_price = sell_price = range_val = target_val = stoploss_val = np.nan

    row.update({
        "Last_Timestamp": ts, "LTP": ltp, "VWAP": vwap, "RSI": rsi, "MACD": macd,
        "MACD_SIGNAL": macd_sig, "Signal": signal, 
        "Buy_Price": buy_price,
        "Sell_Price": sell_price, "Range": range_val, 
        "Target": target_val, "Stoploss": stoploss_val, 
        "Score": score, "Movement": movement, "Movement_Detail": movement_detail,
        "ATR": atr 
    })
    return row

# ---------------- HTML CARD GENERATION HELPER FUNCTION ----------------
def get_card_html(row: pd.Series) -> str:
    """Generates the full HTML string for a single stock card, including all dynamic data."""
    # This section is largely unchanged from the previous version, 
    # ensuring it uses the new detailed 'Movement', 'Movement_Detail', and 'Reason' strings.
    color = get_signal_color(str(row['Signal']))
    bg_color = get_signal_background_color(str(row['Signal']))
    symbol = row['Symbol']
    tradingview_symbol = symbol.replace(".NS", "")
    tradingview_link = f"https://www.tradingview.com/chart/?symbol=NSE%3A{tradingview_symbol}"
    
    # Fetch cached data for trend and news
    trend_colors = get_trend_colors(symbol)
    trend_html = "".join(
        [f"<span style='display:inline-block;width:10px;height:10px;margin-right:2px;background-color:{c};border-radius:2px;'></span>"
            for c in trend_colors]
    )

    stock_news = get_stock_news(symbol)
    news_html = "<ul style='padding-left:15px;margin:5px 0;'>"
    for n in stock_news:
        title_display = n['title'][:60] + '...' if len(n['title']) > 60 else n['title']
        news_html += f"<li style='font-size:12px;'>📢 <a href='{n['link']}' target='_blank' style='color:#0d6efd;text-decoration:none;'>{title_display}</a><br><span style='color:gray;font-size:11px;'>{n['pub_date']}</span></li>"
    news_html += "</ul>"
    
    # Dynamic styling and emojis
    signal_emoji = "🟢" if "BUY" in row['Signal'] else ("🔴" if "SELL" in row['Signal'] else "🟡")
    target_emoji = "🎯"
    stoploss_emoji = "🛑"
    change_color = '#198754' if row['Daily_Change_Pct'] >= 0 else '#dc3545'
    change_text = f" ({row['Daily_Change_Pct']:.2f}%)" if not np.isnan(row['Daily_Change_Pct']) else ""
    volume_emoji = "🚨" if row['Volume_Spike'] == "SPIKE" else "🟢"
    volume_color = '#dc3545' if row['Volume_Spike'] == "SPIKE" else '#212529'
    
    ltp = row['LTP']
    w52_status_pct = row['W52_Status_Pct']
    w52_high = row['W52_High']
    w52_low = row['W52_Low']

    # 52W Status HTML
    w52_status_html = ""
    if not np.isnan(w52_status_pct) and not np.isnan(ltp):
        if w52_status_pct < 20: color_bar = '#007bff'
        elif w52_status_pct < 50: color_bar = '#28a745'
        elif w52_status_pct < 80: color_bar = '#ffc107'
        else: color_bar = '#dc3545'

        w52_status_html = f"""
        <div style="margin-top: 5px; font-size: 13px; padding: 5px; border: 1px dashed #ccc; border-radius: 4px;">
            <p style="margin: 0;">⛰️ <b>52W Status:</b> <b>{w52_status_pct:.2f}%</b> from Low to High</p>
            <div style="width: 100%; background-color: #e9ecef; border-radius: 5px; margin-top: 3px;">
                <div style="width: {w52_status_pct}%; background-color: {color_bar}; height: 8px; border-radius: 5px;"></div>
            </div>
            <p style="margin: 3px 0 0 0; display: flex; justify-content: space-between;">
                <span style="color:#007bff; font-size:11px;">Low: {w52_low:.2f}</span> 
                <span style="color:#212529; font-weight:bold; font-size:11px;">LTP: {ltp:.2f}</span>
                <span style="color:#dc3545; font-size:11px;">High: {w52_high:.2f}</span>
            </p>
        </div>
        """
    else:
         w52_status_html = f"""
         <p style="margin: 0; font-size: 14px; margin-bottom: 5px;">
            ⛰️ <b>52W High:</b> {w52_high:.2f} | ⬇️ <b>52W Low:</b> {w52_low:.2f} 
         </p>
         """
    
    # Last Dividend HTML 
    dividend_html = f"<p style='margin: 0; font-size: 14px;'>💵 <b>Last Dividend:</b> <b>{row['Last_Dividend']:.3f}</b> </p>" if not np.isnan(row['Last_Dividend']) else "<p style='margin: 0; font-size: 14px;'>💵 <b>Last Dividend:</b> Not found </p>"

    # Company Details HTML 
    company_details_html = f"""
    <div style="margin-top: 10px; font-size: 12px; color: #555; max-height: 80px; overflow-y: auto; border: 1px solid #eee; padding: 5px; border-radius: 4px; background-color: #f8f9fa;">
    <b>🏢 Company Snapshot:</b><br>
    {row['Company_Details']}
    </div>
    """

    # Fundamental Status HTML
    fund_status_color = "#6c757d"
    fund_status_bg = "#f5f5f5"
    
    if "Strong Value" in row['Fundamental_Status'] or "Deep Value" in row['Fundamental_Status']:
        fund_status_color = "#198754"
        fund_status_bg = "#e8f5e9"
    elif "High Risk" in row['Fundamental_Status'] or "High Debt" in row['Fundamental_Status'] or "Loss Making" in row['Fundamental_Status']:
        fund_status_color = "#dc3545"
        fund_status_bg = "#ffebee"
    elif "Growth Stock" in row['Fundamental_Status']:
        fund_status_color = "#007bff"
        fund_status_bg = "#f0f8ff"
        
    fundamental_html = f"""
    <div style="
        margin-top: 15px; 
        padding: 10px; 
        border: 1px solid {fund_status_color}; 
        border-radius: 8px; 
        background-color: {fund_status_bg};
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    ">
        <p style="margin: 0; font-size: 14px; font-weight: bold; color: {fund_status_color};">
            🌟 Fundamental Status: {row['Fundamental_Status']}
        </p>
        <hr style="margin: 5px 0; border-top: 1px dashed {fund_status_color}; opacity: 0.5;">
        <div style="display: flex; justify-content: space-between; font-size: 13px;">
            <span>📈 P/E: <b>{row['PE_Ratio']:.2f}</b></span>
            <span>📘 P/B: <b>{row['PB_Ratio']:.2f}</b></span>
            <span>💲 EPS: <b>{row['EPS']:.2f}</b></span>
            <span>⚖️ D/E: <b>{row['Debt_Equity']:.2f}</b></span>
        </div>
    </div>
    """

    # EMA/BBAND HTML 
    bband_html = f"""
    <p style="margin: 0; font-size: 14px;">
        〰️ <b>BB Upper:</b> {row['BB_Upper']:.2f} | ⬇️ <b>BB Lower:</b> {row['BB_Lower']:.2f}
    </p>
    """ if not np.isnan(row['BB_Upper']) else ""
    
    ema_html = f"""
    <p style="margin: 0; font-size: 14px;">
        📉 <b>EMA 20:</b> {row['EMA_20']:.2f} | 📉 <b>EMA 50:</b> {row['EMA_50']:.2f}
    </p>
    """ if not np.isnan(row['EMA_20']) else ""
    
    # The main card content HTML
    card_html = f"""
    <div class="stock-card" style="
    background-color: {bg_color};
    border-top: 6px solid {color};
    border-radius: 12px;
    box-shadow: 0 4px 10px rgba(0,0,0,0.1);
    padding: 20px;
    height: 100%;
    ">
    <h3 style="margin-bottom: 5px;">
    <a href="{tradingview_link}" target="_blank" style="text-decoration:none; color:{color};">📈 {symbol}</a>
    </h3>
    <h5 style="margin-top: 0; color:{color};">
        {signal_emoji} <b>{row['Signal']}</b> 
        <span style="font-size:14px; font-weight:bold; color: {change_color};">{change_text}</span>
    </h5>
    <hr style="margin: 10px 0;">
    
    <p style="margin: 0; font-size: 14px;">
        💰 <b>LTP:</b> {row['LTP']:.2f} | 🌊 <b>VWAP:</b> {row['VWAP']:.2f}
    </p>
    <p style="margin: 0; font-size: 14px;">
        ⬆️ <b>Day High:</b> {row['Day_High']:.2f} | ⬇️ <b>Day Low:</b> {row['Day_Low']:.2f}
    </p>
    
    {ema_html} {bband_html} <p style="margin: 0; font-size: 14px;">
        ⏫ <b>RSI:</b> {row['RSI']:.2f} | 📶 <b>MACD:</b> {row['MACD']:.3f} (Sig: {row['MACD_SIGNAL']:.3f})
    </p>
    
    {dividend_html} {w52_status_html} <p style="margin: 0; font-size: 13px; color: gray;"><b>⌚ Time:</b> {row['Last_Timestamp']}</p>
    
    {fundamental_html} <div style="margin-top: 10px; font-size: 13px; color: #333; background-color: #ffffff; padding: 8px; border-radius: 6px; border: 1px solid #ddd;">
    <p style="margin: 0;"><b>Buy:</b> {row['Buy_Price']} | <b>Sell:</b> {row['Sell_Price']} | <b>Range:</b> {row['Range']}</p>
    <p style="margin: 0;">
        {target_emoji} <b>Target:</b> {row['Target']:.2f} | {stoploss_emoji} <b>Stoploss:</b> {row['Stoploss']:.2f} | ⭐ <b>Score:</b> {row['Score']}
    </p>
    <hr style="margin: 5px 0; border-top: 1px dashed #ccc;">
    <p style="margin: 0;">
        {volume_emoji} <b style="color:{volume_color};">Volume: {row['Volume_Spike']}</b> | 🧭 <b>ATR:</b> {row['ATR']:.2f} (Volatiltiy)
    </p>
    </div>
    
    <div style="margin-top: 10px; font-size: 13px; color: #555;">
    <b>🧭 Movement Nature:</b> {row['Movement']}<br>
    <span style="color:#333;">{row['Movement_Detail']}</span>
    </div>
    
    <div style="margin-top: 10px; font-size: 13px; color: #333; line-height: 1.4;">
    <b>🧠 Detailed Reason:</b> {row['Reason']}
    </div>
    
    <div style="margin-top: 12px;"><b>🗓️ Last 10 Day Trend:</b> {trend_html}</div> 
    
    {company_details_html} 
    
    <div style="margin-top: 10px; font-size: 13px;">
    <b>📰 Latest News:</b> {news_html}
    </div>
    </div>
    """
    return card_html


def generate_html_report(df: pd.DataFrame) -> str:
    """Generates a complete, self-contained HTML file content from the DataFrame."""
    
    # 1. CSS for Card Layout
    css_style = """
    <style>
        body { font-family: Arial, sans-serif; background-color: #f0f2f6; padding: 20px; }
        .report-header { text-align: center; margin-bottom: 25px; padding: 15px; background-color: #fff; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
        .card-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 20px;
        }
        .stock-card {
            border-radius: 12px;
            box-shadow: 0 4px 10px rgba(0,0,0,0.15);
            padding: 20px;
            height: 100%;
        }
        h1 { color: #007bff; }
        h3, h5 { margin-top: 0; }
        hr { border: 0; border-top: 1px solid #ccc; margin: 15px 0; }
        .stock-card a { text-decoration: none; }
    </style>
    """
    
    # 2. Card Content Generation
    cards_html = ""
    for _, row in df.iterrows():
        cards_html += get_card_html(row)
    
    # 3. Final HTML Structure
    timestamp = dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    final_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Intraday Scanner Report - {timestamp}</title>
        {css_style}
    </head>
    <body>
        <div class="report-header">
            <h1>📈 Intraday Signals Report</h1>
            <p>Generated on: <b>{timestamp} IST</b></p>
            <p style="font-size: 12px; color: #6c757d;">
                (डेटा NSE Live LTP + Yfinance Indicators वर आधारित आहे. प्रत्यक्ष ट्रेडिंगसाठी नेहमी तुमच्या ब्रोकरच्या रिअल-टाईम डेटाचा वापर करा.)
            </p>
        </div>
        <div class="card-grid">
            {cards_html}
        </div>
    </body>
    </html>
    """
    return final_html.encode('utf-8')


# ---------------- Run Analysis and Display ----------------
st.markdown("## 📈 Auto Intraday Scanner — Hybrid Card View (Advanced)")
st.caption(f"**NSE Live LTP + Yfinance Indicators.** Interval: **{interval}**, Refresh Rate: **{refresh_sec} sec**.")

if 'last_run' not in st.session_state:
    st.session_state['last_run'] = dt.datetime.now()

time_since_last_run = (dt.datetime.now() - st.session_state['last_run']).seconds
st.info(f"Last data update: {st.session_state['last_run'].strftime('%H:%M:%S')}. Next update expected in approx. {refresh_sec - time_since_last_run if time_since_last_run < refresh_sec else 0} seconds.")

# *************************************************************************
# >>> लॉजिक एक्सप्लेनर बॉक्स (Unchanged - Already detailed) <<<
# *************************************************************************

# मराठीतील सविस्तर स्पष्टीकरण
marathi_logic_explanation = """
## 🧠 कोड लॉजिकचे मराठीत सविस्तर स्पष्टीकरण (Scanner Logic Explained)

हा Streamlit ॲप **NSE Live Price** (LTP), **तांत्रिक इंडिकेटर्स (Technical Indicators)**, आणि **मूलभूत वित्तीय आकडेवारी (Fundamental Metrics)** यांचा वापर करून सिग्नल देतो.

---

### १. तांत्रिक इंडिकेटर्स आणि त्यांचे महत्त्व (Technical Indicators)

| इंडिकेटर | पूर्ण नाव (Full Name) | अर्थ आणि इंट्राडे वापर |
| :--- | :--- | :--- |
| **💰 LTP** | Last Traded Price | बाजारात सध्या चालू असलेली नवीनतम किंमत. |
| **🌊 VWAP** | Volume Weighted Average Price | **संस्थात्मक (Institutional) खरेदीदारांची सरासरी किंमत.** LTP हा **VWAP च्या वर** असल्यास तेजी (Bullish) चा मजबूत संकेत मिळतो. |
| **⏫ RSI** | Relative Strength Index | **मोमेंटम (Momentum) आणि किंमत शक्ती** मोजतो. **RSI > 60** म्हणजे मजबूत खरेदी, **RSI < 40** म्हणजे मजबूत विक्री. |
| **📶 MACD** | Moving Average Convergence Divergence | मोमेंटमची **दिशा आणि सामर्थ्य** दर्शवते. MACD लाईन ही **सिग्नल लाईनच्या वर** गेल्यास BUY सिग्नल कन्फर्म होतो. |
| **📉 EMA (२०/५०)** | Exponential Moving Average | **ट्रेंड कन्फर्मेशनसाठी** वापरला जातो. **LTP EMA २० च्या वर** असल्यास शॉर्ट-टर्म तेजीचा सिग्नल मजबूत होतो. |
| **〰️ BBands** | Bollinger Bands (Upper/Lower) | **व्होलाटिलिटी (Volatility) आणि रेंज** मोजते. LTP Lower Band जवळ असल्यास 'Oversold' परिस्थिती दर्शवते. |
| **🧭 ATR** | Average True Range | स्टॉकच्या **चढ-उताराची (Volatility) सरासरी** मोजतो. याचा वापर **स्टॉपलॉस (SL)** आणि **टार्गेट (Target)** सुरक्षितपणे सेट करण्यासाठी केला जातो. |

---

### २. मूलभूत आकडेवारी (Fundamental Metrics) आणि अर्थ

हे आकडे स्टॉकमधील **गुंतवणुकीचे मूल्य (Value)** आणि **कंपनीचे आरोग्य (Health)** दर्शवतात.

| आकडेवारी | पूर्ण नाव (Full Name) | मराठीतील सोपा अर्थ आणि ॲपमधील वापर |
| :--- | :--- | :--- |
| **📈 P/E Ratio** | **Price to Earnings Ratio** | **'हा स्टॉक किती महाग आहे?'** P/E जितका कमी (उदा. **< २०**) तितका स्टॉक **मूल्यवान (Value)** मानला जातो, कारण तो कंपनीच्या नफ्याच्या तुलनेत स्वस्त असतो. |
| **📘 P/B Ratio** | **Price to Book Ratio** | **'कंपनीच्या पुस्तकी मूल्याच्या तुलनेत स्टॉकची किंमत?'** P/B < १.५ किंवा २.० असल्यास कंपनीचे मूल्य आकर्षक मानले जाते. |
| **💲 EPS** | **Earnings Per Share** | **'प्रत्येक शेअरमागे कंपनी किती नफा कमावते?'** जास्त EPS म्हणजे कंपनीचा नफा जास्त आहे. |
| **⚖️ D/E Ratio** | **Debt to Equity Ratio** | **'कंपनीवर किती कर्ज आहे?'** D/E **> १.०** असल्यास कंपनीवर तिच्या भांडवलापेक्षा जास्त कर्ज आहे, जे **उच्च जोखीम (High Risk)** दर्शवते. |

**Fundamental Status Logic:**
ॲपमध्ये फंडामेंटल स्टेटस देण्यासाठी P/E आणि D/E चा वापर केला जातो:
* **Strong Value:** P/E कमी आणि D/E खूप कमी (Low Debt).
* **High Risk:** P/E खूप जास्त आणि D/E खूप जास्त (Expensive and High Debt).

---

### ३. अंतिम सिग्नल जनरेशन लॉजिक (Decision Making Rules)

स्कॅनर केवळ तेव्हाच **'STRONG'** सिग्नल देतो, जेव्हा **मोमेंटम (RSI, MACD), ट्रेंड (EMA), आणि व्हॉल्यूम (VWAP)** या तिन्ही गोष्टी एकाच दिशेने असतात.

| सिग्नल | उच्च-विश्वास (High Conviction) नियम / कंडिशन |
| :--- | :--- |
| **STRONG BUY** | LTP **>** VWAP **AND** RSI **>** **60** **AND** MACD **>** MACD_SIGNAL **AND** LTP **>** **EMA_20** |
| **STRONG SELL** | LTP **<** VWAP **AND** RSI **<** **40** **AND** MACD **<** MACD_SIGNAL **AND** LTP **<** **EMA_20** |
| **BUY / SELL** | केवळ LTP VWAP च्या वर/खाली असल्यास किंवा RSI ५० च्या वर/खाली असल्यास (Medium Conviction). |
"""

# कोलॅप्सिबल बॉक्समध्ये स्पष्टीकरण दाखवा
with st.expander("🤔 **स्कॅनर कोड लॉजिक आणि मूलभूत आकडेवारीचे मराठीत सविस्तर स्पष्टीकरण (Click Here)**"):
    st.markdown(marathi_logic_explanation, unsafe_allow_html=True)
    st.caption("हे स्पष्टीकरण स्कॅनर कशाप्रकारे डेटा ॲनालाईज करतो हे समजून घेण्यासाठी आहे.")

# *************************************************************************
# >>> लॉजिक एक्सप्लेनर बॉक्सचा शेवट <<<
# *************************************************************************

progress = st.empty()
rows = []
for i, s in enumerate(STOCKS):
    progress.text(f"Analyzing {i+1}/{len(STOCKS)}: {s}")
    rows.append(analyze_symbol(s, history_period, interval))
progress.empty()

# --- DataFrame Processing ---
df_display = pd.DataFrame(rows)

df_display = df_display[df_display["Signal"] != "NEUTRAL"].copy()

signal_order = {"STRONG BUY": 1, "STRONG SELL": 2, "BUY": 3, "SELL": 4}
df_display["SortOrder"] = df_display["Signal"].map(signal_order).fillna(5)
df_display = df_display.sort_values(by="SortOrder", ascending=True).drop(columns=['SortOrder'])

st.markdown("### 💹 Intraday Signals — Hybrid Card View")

# --- Card Layout Display (Streamlit Rendering) ---
cols_per_row = 3
for i in range(0, len(df_display), cols_per_row):
    cols = st.columns(cols_per_row)
    for j, col in enumerate(cols):
        if i + j < len(df_display):
            row = df_display.iloc[i + j]
            # Call the helper function to get the HTML for display
            card_html = get_card_html(row)
            col.markdown(card_html, unsafe_allow_html=True) 

if df_display.empty:
    st.warning("No STRONG BUY, STRONG SELL, BUY, or SELL signals generated.")

# --- HTML Download Section (Unchanged from previous request) ---
if show_download:
    st.markdown("---")
    
    # Generate the complete HTML report content
    html_report_content = generate_html_report(df_display)
    
    st.download_button(
        label="📥 Download HTML Report",
        data=html_report_content,
        file_name=f"intraday_report_{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%d_%H%M%S')}.html",
        mime='text/html'
    )

st.markdown("---")
st.caption(
    "📘 Note: हा स्कॅनर **NSE Live LTP** (NSEPY द्वारे) वापरून **Yfinance** च्या ऐतिहासिक डेटावर आधारित इंडिकेटर्सची गणना करतो. "
    "**Fundamental Status** हा P/E आणि Debt/Equity च्या साध्या नियमांवर आधारित आहे. "
    "**प्रत्यक्ष ट्रेडिंगसाठी नेहमी तुमच्या ब्रोकरच्या रिअल-टाईम डेटाचा आणि सखोल रिसर्चचा वापर करा.**"
)