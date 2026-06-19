#!/usr/bin/env python3
import time, logging, json, os, requests
import pandas as pd
import numpy as np
from datetime import datetime

os.makedirs("logs", exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("logs/bot.log"), logging.StreamHandler()])
log = logging.getLogger(__name__)

with open("config/settings.json") as f:
    CFG = json.load(f)

SYMBOLS = CFG["symbols"]
INTERVAL = CFG["interval"]
CHECK_MIN = CFG["check_every_min"]
TG_TOKEN = CFG["telegram"]["bot_token"]
TG_CHATID = CFG["telegram"]["chat_id"]
last_signal = {}

def send_telegram(text):
    r = requests.post(f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
        json={"chat_id": TG_CHATID, "text": text, "parse_mode": "HTML", "disable_web_page_preview": True}, timeout=10)
    r.raise_for_status()

def fetch_candles(symbol, interval="15m", limit=100):
    r = requests.get("https://api.binance.com/api/v3/klines",
        params={"symbol": symbol, "interval": interval, "limit": limit}, timeout=10)
    r.raise_for_status()
    df = pd.DataFrame(r.json(), columns=["open_time","open","high","low","close","volume","close_time","quote_vol","trades","taker_base","taker_quote","ignore"])
    for col in ["open","high","low","close","volume"]:
        df[col] = df[col].astype(float)
    df["time"] = pd.to_datetime(df["open_time"], unit="ms")
    return df[["time","open","high","low","close","volume"]].copy()

def calc_rsi(s, p=14):
    d = s.diff()
    g = d.clip(lower=0).ewm(com=p-1, min_periods=p).mean()
    l = (-d.clip(upper=0)).ewm(com=p-1, min_periods=p).mean()
    return 100 - (100 / (1 + g / l.replace(0, np.nan)))

def calc_macd(s, fast=12, slow=26, signal=9):
    ef = s.ewm(span=fast, adjust=False).mean()
    es = s.ewm(span=slow, adjust=False).mean()
    ml = ef - es
    sl = ml.ewm(span=signal, adjust=False).mean()
    return ml, sl, ml - sl

def calc_sma(s, p): return s.rolling(p).mean()

def calc_bb(s, p=20, std=2.0):
    m = s.rolling(p).mean(); sg = s.rolling(p).std()
    return m + std*sg, m, m - std*sg

def analyze(symbol):
    try: df = fetch_candles(symbol, INTERVAL)
    except Exception as e: log.error(f"Erro {symbol}: {e}"); return None
    c = df["close"]; score = 0; reasons = []
    rsi = calc_rsi(c); rn = rsi.iloc[-1]
    if rn < 30: score += 3; reasons.append(f"RSI sobrevendido ({rn:.1f})")
    elif rn < 40: score += 1; reasons.append(f"RSI baixo ({rn:.1f})")
    elif rn > 70: score -= 3; reasons.append(f"RSI sobrecomprado ({rn:.1f})")
    elif rn > 60: score -= 1; reasons.append(f"RSI alto ({rn:.1f})")
    ml, sl, hist = calc_macd(c)
    if hist.iloc[-1] > 0 and hist.iloc[-2] <= 0: score += 3; reasons.append("MACD cruzamento altista")
    elif hist.iloc[-1] < 0 and hist.iloc[-2] >= 0: score -= 3; reasons.append("MACD cruzamento baixista")
    elif hist.iloc[-1] > hist.iloc[-2] > 0: score += 1; reasons.append("MACD momentum positivo")
    elif hist.iloc[-1] < hist.iloc[-2] < 0: score -= 1; reasons.append("MACD momentum negativo")
    ma20 = calc_sma(c, 20); ma50 = calc_sma(c, 50); price = c.iloc[-1]
    if ma20.iloc[-1] > ma50.iloc[-1] and ma20.iloc[-2] <= ma50.iloc[-2]: score += 3; reasons.append("Golden Cross MA20/MA50")
    elif ma20.iloc[-1] < ma50.iloc[-1] and ma20.iloc[-2] >= ma50.iloc[-2]: score -= 3; reasons.append("Death Cross MA20/MA50")
    if price > ma20.iloc[-1]: score += 1; reasons.append("Preco acima MA20")
    else: score -= 1; reasons.append("Preco abaixo MA20")
    bbu, _, bbl = calc_bb(c)
    if price < bbl.iloc[-1]: score += 2; reasons.append("Abaixo Bollinger inferior")
    elif price > bbu.iloc[-1]: score -= 2; reasons.append("Acima Bollinger superior")
    va = df["volume"].rolling(20).mean().iloc[-1]; vr = df["volume"].iloc[-1] / va if va > 0 else 1
    if vr > 1.5: reasons.append(f"Volume {vr:.1f}x acima da media")
    sig = "COMPRAR" if score >= 4 else "VENDER" if score <= -4 else "AGUARDAR"
    conf = min(100, int(abs(score) / 12 * 100))
    return {"symbol": symbol, "signal": sig, "score": score, "confidence": conf,
            "price": price, "rsi": rn, "ma20": ma20.iloc[-1], "ma50": ma50.iloc[-1],
            "bb_upper": bbu.iloc[-1], "bb_lower": bbl.iloc[-1],
            "reasons": reasons, "time": df["time"].iloc[-1].strftime("%H:%M %d/%m/%Y")}

def format_msg(r):
    em = {"COMPRAR":"🟢","VENDER":"🔴","AGUARDAR":"🟡"}.get(r["signal"],"⚪")
    coin = r["symbol"].replace("USDT","")
    bar = "█"*int(r["confidence"]/10) + "░"*(10-int(r["confidence"]/10))
    rl = "🔴 Sobrecomprado" if r["rsi"]>70 else "🟢 Sobrevendido" if r["rsi"]<30 else "🟡 Neutro"
    reasons = "\n".join(f"  • {x}" for x in r["reasons"])
    return (f"{em} <b>TRENDSCOPE — {coin}/USDT</b>\n📅 {r['time']}\n"
            f"━━━━━━━━━━━━━━━━━━━━\n🎯 <b>Sinal: {r['signal']}</b> (score: {r['score']:+d})\n"
            f"📊 Confianca: <code>{bar}</code> {r['confidence']}%\n"
            f"━━━━━━━━━━━━━━━━━━━━\n💰 Preco: <b>${r['price']:,.2f}</b>\n"
            f"📈 RSI 14: <b>{r['rsi']:.1f}</b> {rl}\n"
            f"📉 MA20: ${r['ma20']:.2f} | MA50: ${r['ma50']:.2f}\n"
            f"🔼 BB Sup: ${r['bb_upper']:.2f} | BB Inf: ${r['bb_lower']:.2f}\n"
            f"━━━━━━━━━━━━━━━━━━━━\n📌 <b>Razoes:</b>\n{reasons}\n"
            f"━━━━━━━━━━━━━━━━━━━━\n⚠️ <i>Nao e recomendacao financeira.</i>")

def run():
    log.info("TrendScope Bot iniciado")
    try: send_telegram(f"🤖 <b>TrendScope iniciado!</b>\nMonitorando: {', '.join(s.replace('USDT','/USDT') for s in SYMBOLS)}\nCiclo: {CHECK_MIN} min")
    except Exception as e: log.error(e)
    while True:
        for symbol in SYMBOLS:
            r = analyze(symbol)
            if r is None: continue
            log.info(f"{symbol}: {r['signal']} | Score={r['score']:+d} | RSI={r['rsi']:.1f} | ${r['price']:,.2f}")
            if r["signal"] != "AGUARDAR" and last_signal.get(symbol) != r["signal"]:
                try: send_telegram(format_msg(r)); last_signal[symbol] = r["signal"]
                except Exception as e: log.error(e)
            else: last_signal[symbol] = r["signal"]
        log.info(f"Proxima analise em {CHECK_MIN} min")
        time.sleep(CHECK_MIN * 60)

if __name__ == "__main__":
    run()
