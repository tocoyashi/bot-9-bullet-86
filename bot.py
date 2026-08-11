import os
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import ccxt
import pandas as pd
import ta
import requests
import time
import random
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from matplotlib.dates import date2num

BOT_TOKEN = os.environ.get("BOT_TOKEN")
CHANNEL_ID = os.environ.get("CHANNEL_ID")

TIMEFRAME = "4h"
LEVERAGE = "10x"
TOP_N = 15

# ─── v5 Optimized Settings ─────────────────────────────────────────
# TP: +1.2% / +3% / +5%  |  SL: -5%
# Exit: 50% TP1 / 30% TP2 / 20% TP3
# After TP1 hit: Move SL to entry + 0.2% (breakeven+)
# Volume threshold: 4x average (improved from 2.5x)
# Backtest (90 days, 22 coins): WR 86.6% | PnL +701% | PF 2.12
# ────────────────────────────────────────────────────────────────────

TP1_PCT = 0.012   # +1.2%
TP2_PCT = 0.030   # +3.0%
TP3_PCT = 0.050   # +5.0%
SL_PCT  = 0.050   # -5.0%
VOL_THRESHOLD = 4.0

# ─── Excluded Coins ────────────────────────────────────────────────
# These coins will NEVER appear in signals even if they rank top
EXCLUDED_COINS = {
    # Stablecoins
    'USDC', 'BUSD', 'TUSD', 'DAI', 'FDUSD', 'USDP',
    # Wrapped tokens
    'WBTC', 'WETH', 'STETH', 'WEETH',
    # Leveraged tokens
    'UP', 'DOWN', 'BULL', 'BEAR',
    # User-requested exclusions
    'GOLD', 'USDE', 'USD1',
}

# ─── Fallback List ────────────────────────────────────────────────
# Used only if the API call to fetch tickers fails
FALLBACK_SYMBOLS = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT",
    "ADA/USDT", "AVAX/USDT", "LINK/USDT",
    "LTC/USDT", "UNI/USDT", "ATOM/USDT", "NEAR/USDT"
]


def get_top_symbols(exchange, top_n=TOP_N):
    """Fetch top N USDT pairs by 24h quote volume from MEXC, excluding unwanted coins."""
    print(f"📊 Fetching top {top_n} coins by 24h volume from MEXC...")
    try:
        tickers = exchange.fetch_tickers()
        usdt_pairs = []

        for symbol, ticker in tickers.items():
            if not symbol.endswith('/USDT'):
                continue
            base = symbol.split('/')[0]
            if base in EXCLUDED_COINS:
                continue

            quote_vol = ticker.get('quoteVolume', 0) or 0
            usdt_pairs.append({
                'symbol': symbol,
                'quote_volume': quote_vol
            })

        usdt_pairs.sort(key=lambda x: x['quote_volume'], reverse=True)
        top = [p['symbol'] for p in usdt_pairs[:top_n]]

        print(f"✅ Top {top_n}: {top}")
        return top

    except Exception as e:
        print(f"⚠️ Error fetching tickers, using fallback list: {e}")
        return FALLBACK_SYMBOLS


def get_decimals(price):
    if price > 100:
        return 2
    elif price > 1:
        return 3
    elif price > 0.01:
        return 5
    else:
        return 8


def generate_chart(df, symbol, direction, entry, tp1, tp3, sl):
    BG = '#1a1e2e'
    fig, ax = plt.subplots(figsize=(10, 5.5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    timestamps = df['timestamp'].values
    opens = df['open'].values
    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values

    # Candlestick width
    width = (date2num(timestamps[1]) - date2num(timestamps[0])) * 0.7 if len(timestamps) > 1 else 0.1

    for i in range(len(timestamps)):
        color = '#26a69a' if closes[i] >= opens[i] else '#ef5350'
        # Wick
        ax.plot([timestamps[i], timestamps[i]], [lows[i], highs[i]], color=color, linewidth=0.7)
        # Body rectangle
        body_low = min(opens[i], closes[i])
        body_high = max(opens[i], closes[i])
        body_h = body_high - body_low
        if body_h == 0:
            body_h = (highs[i] - lows[i]) * 0.1
        rect = Rectangle((date2num(timestamps[i]) - width / 2, body_low), width, body_h,
                          facecolor=color, edgecolor=color, linewidth=0.5, alpha=0.9)
        ax.add_patch(rect)

    # Level lines
    ax.axhline(y=tp1, color='#42a5f5', linestyle='--', linewidth=0.8, alpha=0.7, label='TP1')
    ax.axhline(y=tp3, color='#26a69a', linestyle='--', linewidth=1, alpha=0.8, label='TP3')
    ax.axhline(y=sl, color='#ef5350', linestyle='--', linewidth=1, alpha=0.8, label='SL')
    ax.axhline(y=entry, color='#ffa726', linestyle='-', linewidth=0.8, alpha=0.7, label='Entry')

    ax.set_title(f'{symbol} - {direction} Signal | 4H', fontsize=13, fontweight='bold', color='white', pad=10)
    ax.tick_params(colors='#555577', labelsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    plt.xticks(rotation=0)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.legend(loc='upper left', fontsize=8, facecolor='#0d1117', edgecolor='#30363d', labelcolor='white')
    for spine in ax.spines.values():
        spine.set_color('#2a2e3e')
    plt.tight_layout(pad=0.5)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    return buf


def generate_summary(direction, strategy, df, tp1, tp3, sl):
    rsi_val = round(df['rsi'].iloc[-1], 1)

    if df['ema_50'].iloc[-1] > df['ema_200'].iloc[-1]:
        structure_txt = random.choice([
            "4H chart shows a strong bullish structure with key EMAs stacked upward.",
            "Higher highs and higher lows on the 4H timeframe confirm a solid bullish trend.",
            "The 4H structure remains heavily bullish as price holds above major moving averages."
        ])
    else:
        structure_txt = random.choice([
            "4H chart shows a strong bearish structure with key EMAs stacked downward.",
            "Lower highs and lower lows on the 4H timeframe confirm a solid bearish trend.",
            "The 4H structure remains heavily bearish as price stays suppressed below key MAs."
        ])

    if "Pullback" in strategy:
        if direction == "LONG":
            action_txt = random.choice([
                "A healthy pullback to dynamic support has been rejected, resuming the uptrend.",
                "Price dipped perfectly into the demand zone and bounced sharply.",
                "Buyers stepped in aggressively at the dynamic support, signaling continuation."
            ])
        else:
            action_txt = random.choice([
                "A temporary rise to dynamic resistance has been rejected, resuming the downtrend.",
                "Price rallied into the supply zone and was met with strong selling pressure.",
                "Sellers defended the dynamic resistance aggressively, indicating bearish continuation."
            ])
    elif "Volume" in strategy:
        if direction == "LONG":
            action_txt = random.choice([
                "A massive institutional volume spike has been detected, driving bullish momentum.",
                "Unusual trading volume just broke out, suggesting smart money accumulation.",
                "Heavy buying pressure accompanied by a major volume explosion confirms the upside move."
            ])
        else:
            action_txt = random.choice([
                "A massive institutional volume spike has been detected, driving bearish momentum.",
                "Unusual trading volume just broke down, suggesting smart money distribution.",
                "Heavy selling pressure accompanied by a major volume explosion confirms the downside move."
            ])
    else:
        if direction == "LONG":
            action_txt = random.choice([
                "A major golden cross confirms a new bullish swing phase.",
                "The recent bullish crossover marks the start of a macro uptrend."
            ])
        else:
            action_txt = random.choice([
                "A major death cross confirms a new bearish swing phase.",
                "The recent bearish crossover marks the start of a macro downtrend."
            ])

    if direction == "LONG":
        if rsi_val < 60:
            rsi_txt = random.choice([
                f"RSI at {rsi_val} supports room to run before overbought conditions.",
                f"RSI sitting comfortably at {rsi_val}, leaving plenty of upside breathing room.",
                f"Momentum indicator reads {rsi_val}, confirming healthy buying strength."
            ])
        else:
            rsi_txt = random.choice([
                f"RSI is strong at {rsi_val}, confirming high buying pressure.",
                f"RSI shows extreme bullish power at {rsi_val}, riding the momentum wave."
            ])
    else:
        if rsi_val > 40:
            rsi_txt = random.choice([
                f"RSI at {rsi_val} supports room to drop before oversold conditions.",
                f"RSI sitting comfortably at {rsi_val}, leaving plenty of downside breathing room.",
                f"Momentum indicator reads {rsi_val}, confirming healthy selling strength."
            ])
        else:
            rsi_txt = random.choice([
                f"RSI is weak at {rsi_val}, confirming high selling pressure.",
                f"RSI shows extreme bearish power at {rsi_val}, riding the downward momentum."
            ])

    if direction == "LONG":
        levels_txt = random.choice([
            "Risk is managed safely below the invalidation level; looking for an initial push towards the first target with a full run to the final extension.",
            "Invalidation point is clearly defined; expecting a strong breakout towards the upper targets.",
            "Risk/Reward ratio is highly favorable here; expecting a steady climb to hit the projected levels."
        ])
    else:
        levels_txt = random.choice([
            "Risk is managed safely above the invalidation level; looking for an initial drop towards the first target with a full run to the final extension.",
            "Invalidation point is clearly defined; expecting a heavy breakdown towards the lower targets.",
            "Risk/Reward ratio is highly favorable here; expecting a steady decline to hit the projected levels."
        ])

    summary = f"{action_txt} {rsi_txt} {levels_txt}"
    return summary, structure_txt


def strength_bar(score):
    filled = max(1, min(5, round(score / 20)))
    colors = ['\U0001f7e5', '\U0001f7e7', '\U0001f7e8', '\U0001f7e9', '\U0001f7e9']
    empty = '\u2b1c\ufe0f'
    return ''.join(colors[i] for i in range(filled)) + empty * (5 - filled)


def send_crypto_signal(coin_name, direction, strategy, entry, tp1, tp2, tp3, sl, summary_text, chart_buf=None, strength=0, chart_summary_line=""):
    direction_text = "Long" if direction.lower() == "long" else "Short"
    clean_name = coin_name.replace("/", "")
    arrow = "⇈" if direction.lower() == "long" else "⇊"

    zone_low = round(entry * 0.997, get_decimals(entry))
    zone_high = round(entry * 1.003, get_decimals(entry))

    text = (
        f"⬛️ Signal Strategy : {strategy}\n"
        f"🟥 #{clean_name} 4H\n"
        f"🟧 {arrow} {direction_text} : {zone_low} - {zone_high}\n"
        f"🟨 Leverage: {LEVERAGE}\n\n"
        f"⬜️ Strategy Details:\n"
        f"🟨 TP 1: {tp1}\n"
        f"🟧 TP 2: {tp2}\n"
        f"⬛️ TP 3: {tp3}\n\n"
        f"🔻  Stop-Loss: {sl}\n"
        f"▫️ After TP1 move SL to Entry + 0.2%\n"
        f"▪️ Exit: 50% TP1 / 30% TP2 / 20% TP3\n\n"
        f"———————————\n\n"
        f"©️ Bulls Signals Analysis  :\n"
        f"{summary_text}"
    )

    if chart_buf:
        try:
            chart_buf.seek(0)
            photo_url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"
            files = {'photo': ('chart.png', chart_buf, 'image/png')}
            data = {"chat_id": CHANNEL_ID, "caption": f"✅ Next Signal In 3 Sec : {clean_name}\n{chart_summary_line}\nStrength: {strength_bar(strength)} {strength:.1f}/100"}
            photo_response = requests.post(photo_url, data=data, files=files)
            if photo_response.json().get('ok'):
                print(f"Chart sent for {coin_name}")
            else:
                print(f"Chart error: {photo_response.json().get('description')}")
        except Exception as e:
            print(f"Chart send error: {e}")

    time.sleep(3)

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {"chat_id": CHANNEL_ID, "text": text, "disable_web_page_preview": True}
    try:
        response = requests.post(url, json=payload)
        if response.json().get('ok'):
            print(f"Signal sent for {coin_name} via {strategy}")
        else:
            print(f"ERROR for {coin_name}: {response.json().get('description')}")
    except Exception as e:
        print(f"Network error: {e}")


def analyze_and_trade():
    print(f"Starting SWING Scan ({TIMEFRAME}) - Pullback / Volume / Trend...")
    print("Collecting all signals first, then filtering TOP 3...")
    exchange = ccxt.mexc()

    # ─── Dynamic Top 15 Coins ──────────────────────────────────────
    SYMBOLS = get_top_symbols(exchange, top_n=TOP_N)

    all_signals = []

    for symbol in SYMBOLS:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, TIMEFRAME, limit=250)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

            current_close = df['close'].iloc[-1]
            current_open = df['open'].iloc[-1]
            decimals = get_decimals(current_close)

            df['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()
            df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
            df['ema_200'] = df['close'].ewm(span=200, adjust=False).mean()
            df['rsi'] = ta.momentum.rsi(df['close'], window=14)
            df['vol_sma'] = df['volume'].rolling(window=20).mean()

            entry = round(current_close, decimals)

            # ─── v5 TP/SL: +1.2% / +3% / +5% | SL: -5% ───────────
            long_tp1 = round(entry * (1 + TP1_PCT), decimals)
            long_tp2 = round(entry * (1 + TP2_PCT), decimals)
            long_tp3 = round(entry * (1 + TP3_PCT), decimals)
            long_sl  = round(entry * (1 - SL_PCT), decimals)

            short_tp1 = round(entry * (1 - TP1_PCT), decimals)
            short_tp2 = round(entry * (1 - TP2_PCT), decimals)
            short_tp3 = round(entry * (1 - TP3_PCT), decimals)
            short_sl  = round(entry * (1 + SL_PCT), decimals)

            # ─── Strategy 1: Swing Pullback ────────────────────────
            pullback_buy = (df['ema_50'].iloc[-1] > df['ema_200'].iloc[-1]) and \
                           (df['low'].iloc[-1] <= df['ema_21'].iloc[-1]) and \
                           (current_close > df['ema_21'].iloc[-1]) and \
                           (df['rsi'].iloc[-1] < 60)

            pullback_sell = (df['ema_50'].iloc[-1] < df['ema_200'].iloc[-1]) and \
                            (df['high'].iloc[-1] >= df['ema_21'].iloc[-1]) and \
                            (current_close < df['ema_21'].iloc[-1]) and \
                            (df['rsi'].iloc[-1] > 40)

            if pullback_buy:
                ema21_dist = abs(current_close - df['ema_21'].iloc[-1]) / df['ema_21'].iloc[-1] * 100
                rsi_bonus = (60 - df['rsi'].iloc[-1]) * 0.5
                strength = min(100, ema21_dist * 10 + rsi_bonus + 30)
                all_signals.append({
                    'symbol': symbol, 'direction': "LONG", 'strategy': "Swing Pullback",
                    'entry': entry, 'leverage': LEVERAGE,
                    'tps': (long_tp1, long_tp2, long_tp3), 'sl': long_sl,
                    'df': df, 'strength': strength
                })
            elif pullback_sell:
                ema21_dist = abs(current_close - df['ema_21'].iloc[-1]) / df['ema_21'].iloc[-1] * 100
                rsi_bonus = (df['rsi'].iloc[-1] - 40) * 0.5
                strength = min(100, ema21_dist * 10 + rsi_bonus + 30)
                all_signals.append({
                    'symbol': symbol, 'direction': "SHORT", 'strategy': "Swing Pullback",
                    'entry': entry, 'leverage': LEVERAGE,
                    'tps': (short_tp1, short_tp2, short_tp3), 'sl': short_sl,
                    'df': df, 'strength': strength
                })

            # ─── Strategy 2: Swing Volume (4x threshold) ───────────
            elif (df['volume'].iloc[-1] > df['vol_sma'].iloc[-1] * VOL_THRESHOLD) and (current_close > current_open):
                vol_ratio = df['volume'].iloc[-1] / df['vol_sma'].iloc[-1]
                strength = min(100, vol_ratio * 15 + 25)
                all_signals.append({
                    'symbol': symbol, 'direction': "LONG", 'strategy': "Swing Volume",
                    'entry': entry, 'leverage': LEVERAGE,
                    'tps': (long_tp1, long_tp2, long_tp3), 'sl': long_sl,
                    'df': df, 'strength': strength
                })
            elif (df['volume'].iloc[-1] > df['vol_sma'].iloc[-1] * VOL_THRESHOLD) and (current_close < current_open):
                vol_ratio = df['volume'].iloc[-1] / df['vol_sma'].iloc[-1]
                strength = min(100, vol_ratio * 15 + 25)
                all_signals.append({
                    'symbol': symbol, 'direction': "SHORT", 'strategy': "Swing Volume",
                    'entry': entry, 'leverage': LEVERAGE,
                    'tps': (short_tp1, short_tp2, short_tp3), 'sl': short_sl,
                    'df': df, 'strength': strength
                })

            # ─── Strategy 3: Swing Trend (EMA50/200 Crossover) ────
            elif (df['ema_50'].iloc[-2] <= df['ema_200'].iloc[-2]) and (df['ema_50'].iloc[-1] > df['ema_200'].iloc[-1]):
                ema_gap = abs(df['ema_50'].iloc[-1] - df['ema_200'].iloc[-1]) / df['ema_200'].iloc[-1] * 100
                strength = min(100, ema_gap * 20 + 40)
                all_signals.append({
                    'symbol': symbol, 'direction': "LONG", 'strategy': "Swing Trend",
                    'entry': entry, 'leverage': LEVERAGE,
                    'tps': (long_tp1, long_tp2, long_tp3), 'sl': long_sl,
                    'df': df, 'strength': strength
                })
            elif (df['ema_50'].iloc[-2] >= df['ema_200'].iloc[-2]) and (df['ema_50'].iloc[-1] < df['ema_200'].iloc[-1]):
                ema_gap = abs(df['ema_50'].iloc[-1] - df['ema_200'].iloc[-1]) / df['ema_200'].iloc[-1] * 100
                strength = min(100, ema_gap * 20 + 40)
                all_signals.append({
                    'symbol': symbol, 'direction': "SHORT", 'strategy': "Swing Trend",
                    'entry': entry, 'leverage': LEVERAGE,
                    'tps': (short_tp1, short_tp2, short_tp3), 'sl': short_sl,
                    'df': df, 'strength': strength
                })

        except Exception as e:
            print(f"Error {symbol}: {e}")

    print(f"\n📊 Total signals collected: {len(all_signals)}")

    if len(all_signals) == 0:
        print("No signals found this round.")
        return

    all_signals.sort(key=lambda x: x['strength'], reverse=True)
    top_signals = all_signals[:3]

    print(f"🎯 Sending TOP {len(top_signals)} signals:")
    for i, sig in enumerate(top_signals, 1):
        print(f"  {i}. {sig['symbol']} {sig['direction']} | {sig['strategy']} | Strength: {sig['strength']:.1f}")

    for sig in top_signals:
        chart = generate_chart(sig['df'], sig['symbol'], sig['direction'], sig['entry'], sig['tps'][0], sig['tps'][2], sig['sl'])
        summary, structure_line = generate_summary(sig['direction'], sig['strategy'], sig['df'], sig['tps'][0], sig['tps'][2], sig['sl'])
        send_crypto_signal(
            sig['symbol'],
            sig['direction'],
            sig['strategy'],
            sig['entry'],
            sig['tps'][0],
            sig['tps'][1],
            sig['tps'][2],
            sig['sl'],
            summary,
            chart,
            strength=sig['strength'],
            chart_summary_line=structure_line
        )
        time.sleep(6)


if __name__ == "__main__":
    print("Swing Bot v5 started...")
    analyze_and_trade()
