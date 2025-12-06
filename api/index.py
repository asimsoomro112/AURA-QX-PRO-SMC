from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import ccxt.async_support as ccxt
from datetime import datetime
from typing import List, Dict

app = FastAPI()

# CORS allow taake frontend access kar sake
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- CONFIG ---
DEFAULT_SYMBOL = "BTC/USDT"
LIMIT = 500

# --- HELPER CLASSES (Logic Same as Before) ---
class SMCEngine:
    @staticmethod
    def calculate_structure(df: pd.DataFrame):
        if len(df) < 20: return "RANGING"
        current_close = df['close'].iloc[-1]
        highest_20 = df['high'].iloc[-20:-1].max()
        lowest_20 = df['low'].iloc[-20:-1].min()
        if current_close > highest_20: return "BULLISH"
        elif current_close < lowest_20: return "BEARISH"
        return "RANGING"

    @staticmethod
    def detect_market_structure(df: pd.DataFrame) -> List[Dict]:
        points = []
        window = 5
        for i in range(window, len(df) - window):
            ts = df['timestamp'].iloc[i].strftime('%H:%M')
            if df['high'].iloc[i] == df['high'].iloc[i-window:i+window+1].max():
                points.append({"index": i, "timestamp": ts, "price": float(df['high'].iloc[i]), "type": "HH", "color": "#ef4444"})
            if df['low'].iloc[i] == df['low'].iloc[i-window:i+window+1].min():
                points.append({"index": i, "timestamp": ts, "price": float(df['low'].iloc[i]), "type": "HL", "color": "#10b981"})
        return points[-20:]

    @staticmethod
    def find_fvgs(df: pd.DataFrame, timeframe: str) -> List[Dict]:
        fvgs = []
        for i in range(1, len(df) - 1):
            curr, prev, next_c = df.iloc[i], df.iloc[i-1], df.iloc[i+1]
            if next_c['low'] > prev['high']:
                is_mitigated = any(df.iloc[j]['low'] <= prev['high'] for j in range(i + 2, len(df)))
                if not is_mitigated:
                    fvgs.append({"type": f"{timeframe} Bull FVG", "top": float(next_c['low']), "bottom": float(prev['high']), "color": "#fbbf24", "opacity": 0.2, "index": i})
            elif next_c['high'] < prev['low']:
                is_mitigated = any(df.iloc[j]['high'] >= prev['low'] for j in range(i + 2, len(df)))
                if not is_mitigated:
                    fvgs.append({"type": f"{timeframe} Bear FVG", "top": float(prev['low']), "bottom": float(next_c['high']), "color": "#f87171", "opacity": 0.2, "index": i})
        return fvgs[-10:]

    @staticmethod
    def find_order_blocks(df: pd.DataFrame, timeframe: str) -> List[Dict]:
        obs = []
        current_price = df['close'].iloc[-1]
        
        # Opacity Logic
        tf_map = {"1H": 0.25, "4H": 0.35}
        opacity = tf_map.get(timeframe.upper(), 0.2)

        for i in range(len(df) - 3):
            candle, next_c = df.iloc[i], df.iloc[i+1]
            body = abs(candle['close'] - candle['open'])
            full_range = candle['high'] - candle['low']
            if full_range == 0 or (body / full_range) < 0.20: continue 

            move_size = abs(next_c['close'] - next_c['open'])
            ob_type, top, bottom, color = None, 0, 0, ""

            if candle['close'] < candle['open']: 
                if next_c['close'] > candle['high'] and move_size > body * 1.2:
                    if current_price > candle['high']: ob_type, top, bottom, color = f"{timeframe} Bull OB", float(candle['high']), float(candle['low']), "#10b981"
            elif candle['close'] > candle['open']:
                if next_c['close'] < candle['low'] and move_size > body * 1.2:
                    if current_price < candle['low']: ob_type, top, bottom, color = f"{timeframe} Bear OB", float(candle['high']), float(candle['low']), "#ef4444"
            
            if ob_type:
                is_mitigated = False
                for j in range(i + 2, len(df) - 1):
                    future = df.iloc[j]
                    if "Bull" in ob_type and future['low'] <= top: is_mitigated = True
                    if "Bear" in ob_type and future['high'] >= bottom: is_mitigated = True
                
                if not is_mitigated:
                    is_poi = "4H" in timeframe or "1D" in timeframe
                    obs.append({ "type": ob_type, "top": top, "bottom": bottom, "color": color, "opacity": opacity, "index": i, "is_poi": is_poi })
        return obs[-6:]

    @staticmethod
    def generate_combined_heatmap(df: pd.DataFrame, orderbook: Dict) -> List[Dict]:
        zones = []
        current_price = df['close'].iloc[-1]
        
        # Structure Liquidity
        for i in range(5, len(df)-5):
            window = df.iloc[i-5:i+5]
            if df['high'].iloc[i] == window['high'].max():
                px = float(df['high'].iloc[i])
                if 0.90 * current_price < px < 1.10 * current_price:
                    zones.append({"price": px, "color": "#facc15", "type": "STOP_HUNT_HIGH"})
            if df['low'].iloc[i] == window['low'].min():
                px = float(df['low'].iloc[i])
                if 0.90 * current_price < px < 1.10 * current_price:
                    zones.append({"price": px, "color": "#facc15", "type": "STOP_HUNT_LOW"})

        # Whale Walls
        if orderbook:
            bids = orderbook.get('bids', [])
            asks = orderbook.get('asks', [])
            bids = [b for b in bids if b[0] > current_price * 0.95]
            asks = [a for a in asks if a[0] < current_price * 1.05]

            if len(bids) > 0:
                avg_bid = np.median([b[1] for b in bids])
                top_bids = sorted([b for b in bids if b[1] > avg_bid * 1.5], key=lambda x: x[1], reverse=True)[:3]
                for b in top_bids: zones.append({"price": b[0], "color": "#00ff88", "type": f"🐳 BUY: {int(b[1])}", "style": "solid"})

            if len(asks) > 0:
                avg_ask = np.median([a[1] for a in asks])
                top_asks = sorted([a for a in asks if a[1] > avg_ask * 1.5], key=lambda x: x[1], reverse=True)[:3]
                for a in top_asks: zones.append({"price": a[0], "color": "#ff3355", "type": f"🐻 SELL: {int(a[1])}", "style": "solid"})
        
        return zones

class AIAgent:
    @staticmethod
    def analyze(signal, df):
        if not signal: return None, "Scanning..."
        last = df.iloc[-1]
        body = abs(last['close'] - last['open'])
        wick = (last['high'] - max(last['close'], last['open'])) if "SHORT" in signal['type'] else (min(last['close'], last['open']) - last['low'])
        if wick > body * 0.4: return signal, "✅ APPROVED: Rejection detected."
        return signal, "⚠️ CAUTION: No rejection wick yet."

# --- API ENDPOINT ---
@app.get("/api/scan")
async def scan_market(symbol: str = "BTC/USDT", timeframe: str = "1h", ai_mode: bool = False):
    exchange = ccxt.binance({'enableRateLimit': True, 'options': {'defaultType': 'future'}})
    smc = SMCEngine()
    ai = AIAgent()
    
    try:
        # 1. Fetch Data (Parallel)
        # Only fetch primary TF + 4H for structure
        tfs_to_fetch = [timeframe]
        if timeframe != "4h": tfs_to_fetch.append("4h")
        
        data_store = {}
        orderbook = None
        
        # Simple Loop (Asyncio gather can be complex in serverless sometimes, keep simple)
        for tf in tfs_to_fetch:
            ohlcv = await exchange.fetch_ohlcv(symbol, tf, limit=LIMIT)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            data_store[tf] = df
            
        try:
            orderbook = await exchange.fetch_order_book(symbol, limit=50)
        except:
            orderbook = None

        await exchange.close()

        # 2. Analysis
        df_primary = data_store[timeframe]
        # Slice for display (Last 300)
        chart_df = df_primary.iloc[-300:].copy().reset_index(drop=True)
        
        all_zones = []
        trends = {}
        
        # Analyze Structure & Zones
        for tf, df in data_store.items():
            df_slice = df.iloc[-300:].reset_index(drop=True)
            trends[tf] = smc.calculate_structure(df_slice)
            all_zones.extend(smc.find_order_blocks(df_slice, tf.upper()))
            all_zones.extend(smc.find_fvgs(df_slice, tf.upper()))

        heatmap = smc.generate_combined_heatmap(chart_df, orderbook)
        structure = smc.detect_market_structure(chart_df)
        current_price = float(chart_df['close'].iloc[-1])
        
        signal = None
        for zone in all_zones:
            if zone['bottom'] <= current_price <= zone['top']:
                ltf_trend = trends.get(timeframe, "RANGING")
                is_poi = zone.get('is_poi', False)
                if "Bull" in zone['type']: signal = {"type": f"LONG {'POI' if is_poi else 'TEST'}", "confidence": 85, "reason": f"Testing {zone['type']}", "entry": current_price, "stop": zone['bottom'], "target": current_price*1.03}
                elif "Bear" in zone['type']: signal = {"type": f"SHORT {'POI' if is_poi else 'TEST'}", "confidence": 85, "reason": f"Testing {zone['type']}", "entry": current_price, "stop": zone['top'], "target": current_price*0.97}

        ai_narrative = ""
        if ai_mode:
            signal, ai_narrative = ai.analyze(signal, chart_df)

        # JSON Ready Data
        chart_data_clean = chart_df.replace({np.nan: None})
        chart_data_clean['timestamp'] = chart_data_clean['timestamp'].dt.strftime('%H:%M')

        return {
            "symbol": symbol,
            "price": current_price,
            "trends": trends,
            "zones": all_zones,
            "heatmap": heatmap,
            "structure": structure,
            "candles": chart_data_clean.to_dict(orient='records'),
            "signal": signal,
            "ai_narrative": ai_narrative,
            "is_simulation": False
        }

    except Exception as e:
        await exchange.close()
        return {"error": str(e), "is_simulation": True}
