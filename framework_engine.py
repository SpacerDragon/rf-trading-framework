# framework_engine.py
# An Object-Oriented, Single-Threaded Framework for Automated Algorithmic Trading
# Architecture designed by Per Idar Rød.

import joblib
import json
import os
import pandas as pd
import time
import MetaTrader5 as mt5
import logging
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# Secure credential extraction via local environment variables
login = int(os.getenv("MT5_LOGIN", 0))
password = os.getenv("MT5_PASSWORD", "")
server = os.getenv("MT5_SERVER", "")

# =====================================================================
# SYSTEM LOGGING ENGINE
# =====================================================================
log_dir = os.path.join(os.getcwd(), "logs")
os.makedirs(log_dir, exist_ok=True)
date_str = datetime.now().strftime('%Y%m%d')

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s')

trade_logger = logging.getLogger("TradeLogger")
trade_logger.setLevel(logging.INFO)
trade_logger.propagate = False
t_handler = logging.FileHandler(os.path.join(
    log_dir, f"TRADES_{date_str}.log"), encoding='utf-8')
t_handler.setFormatter(logging.Formatter('%(asctime)s | %(message)s'))
trade_logger.addHandler(t_handler)
trade_logger.addHandler(logging.StreamHandler())

audit_logger = logging.getLogger("AuditLogger")
audit_logger.setLevel(logging.INFO)
audit_logger.propagate = False
a_handler = logging.FileHandler(os.path.join(
    log_dir, f"AUDIT_{date_str}.log"), encoding='utf-8')
a_handler.setFormatter(logging.Formatter(
    '%(asctime)s | %(levelname)s | %(message)s'))
audit_logger.addHandler(a_handler)
audit_logger.addHandler(logging.StreamHandler())


# =====================================================================
# BASE TIMEFRAME TRADER (PARENT CLASS)
# =====================================================================
class BaseTimeframeTrader:
    """
    Abstract blueprint managing algorithmic execution pipeline layers including:
    - Live MT5 position polling & tracking matrices
    - Multi-state position protection management (Entry -> Break-Even -> Trailing)
    - Dynamic signal calculation loops driven by Scikit-Learn pipelines
    """

    def __init__(self, symbol, config, mt5_timeframe, model_subfolder, base_magic, index, max_positions=1):
        self.symbol = symbol
        self.config = config
        self.mt5_timeframe = mt5_timeframe
        self.max_positions = max_positions
        self.magic = base_magic + index

        # Configured to look for a standardized generic structure in the repository
        self.model_path = f"./models/{model_subfolder}/{symbol}_rf.joblib"
        self.stats_path = f"./models/{model_subfolder}/{symbol}_stats.json"

        self.model = None
        self.is_active = False
        self.last_candle_time = None

        # State tracking matrix mapping live ticket configurations
        self.trade_state = {}   # ticket -> "entry" | "be" | "trail"
        self.sl_updated_this_tick = set()

        self.audit_and_load()

    def audit_and_load(self):
        """Validates backtest statistical expectancy thresholds prior to live boot."""
        try:
            if os.path.exists(self.model_path) and os.path.exists(self.stats_path):
                with open(self.stats_path, "r") as f:
                    stats = json.load(f)

                # Standard filtering barrier to ensure poor optimization loops stay offline
                if stats.get("expectancy", 0) > 0.2:
                    self.model = joblib.load(self.model_path)
                    self.model.set_params(n_jobs=1)
                    self.is_active = True
                    audit_logger.info(
                        f"✅ {self.symbol} ACTIVE (Expectancy: {stats['expectancy']:.2f})")
                else:
                    audit_logger.warning(
                        f"❌ {self.symbol} INACTIVE (Low Expectancy Filtered)")
            else:
                audit_logger.warning(
                    f"⚠️ {self.symbol} tracking files missing structural checks.")
        except Exception as e:
            audit_logger.error(
                f"Audit processing exception for {self.symbol}: {e}")

    def get_atr_price(self, period=14):
        rates = mt5.copy_rates_from_pos(
            self.symbol, self.mt5_timeframe, 0, period + 1)
        if rates is None or len(rates) < period:
            return None

        df = pd.DataFrame(rates)
        tr = pd.concat([
            df['high'] - df['low'],
            abs(df['high'] - df['close'].shift()),
            abs(df['low'] - df['close'].shift())
        ], axis=1).max(axis=1)

        return tr.rolling(period).mean().iloc[-1]

    def get_latest_candle_time(self):
        rates = mt5.copy_rates_from_pos(self.symbol, self.mt5_timeframe, 0, 2)
        if rates is None or len(rates) < 2:
            return None
        return pd.DataFrame(rates)['time'].iloc[-1]

    def compute_indicators(self):
        """Extracts native arrays and engineers specialized feature frames for the model matrix."""
        rates = mt5.copy_rates_from_pos(
            self.symbol, self.mt5_timeframe, 0, 400)
        if rates is None or len(rates) < 200:
            return None

        df = pd.DataFrame(rates)

        # Technical Feature Indicators
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['rsi'] = 100 - (100 / (1 + (gain / loss)))

        short = df['close'].ewm(span=12, adjust=False).mean()
        long = df['close'].ewm(span=26, adjust=False).mean()
        macd = short - long
        signal = macd.ewm(span=9, adjust=False).mean()
        df['macd'] = (macd / df['close']) * 100
        df['macd_signal'] = (signal / df['close']) * 100

        tr = pd.concat([
            df['high'] - df['low'],
            abs(df['high'] - df['close'].shift()),
            abs(df['low'] - df['close'].shift())
        ], axis=1).max(axis=1)
        df['atr'] = (tr.rolling(14).mean() / df['close']) * 100

        ema_fast = df['close'].ewm(span=50, adjust=False).mean()
        ema_slow = df['close'].ewm(span=200, adjust=False).mean()
        df['close_ema'] = (df['close'] - ema_fast) / ema_fast
        df['ema_diff'] = (ema_fast - ema_slow) / ema_slow

        feature_cols = ['close_ema', 'ema_diff',
                        'rsi', 'macd', 'macd_signal', 'atr']
        return df[feature_cols].iloc[[-1]]

    def modify_position_sl(self, ticket, sl, digits):
        if ticket in self.sl_updated_this_tick:
            return False

        pos = mt5.positions_get(ticket=ticket)
        if not pos:
            return False

        p = pos[0]
        rounded_sl = round(float(sl), digits)

        if p.sl == rounded_sl:
            return False

        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "sl": rounded_sl,
            "tp": p.tp
        }

        result = mt5.order_send(request)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            audit_logger.error(
                f"SL modify sequence failed for ticket {ticket}")
            return False

        trade_logger.info(f"SL UPDATED | {ticket} | {rounded_sl}")
        self.sl_updated_this_tick.add(ticket)
        return True

    def monitor_positions(self):
        """Executes full memory cleanup and monitors multi-state risk matrix tracking loops."""
        self.sl_updated_this_tick = set()
        positions = mt5.positions_get(symbol=self.symbol)
        sym = mt5.symbol_info(self.symbol)

        if not positions or sym is None:
            return

        target_points = self.config.BE_TRIGGER_POINTS.get(self.symbol, 150)
        be_trigger_distance = target_points * sym.point
        spread_cushion = 20 * sym.point
        target_trail_points = self.config.TRAILING_STOP_POINTS.get(
            self.symbol, 200)
        trail_distance = target_trail_points * sym.point

        for pos in positions:
            if pos.magic != self.magic:
                continue

            ticket = pos.ticket
            open_price = pos.price_open
            current_price = pos.price_current
            sl = pos.sl
            is_buy = (pos.type == 0)

            # Auto-healing crash protection state recovery layer
            if ticket not in self.trade_state:
                has_sl = (sl and sl != 0)
                rounded_open = round(open_price, sym.digits)
                if is_buy:
                    self.trade_state[ticket] = "trail" if (has_sl and sl > rounded_open) else (
                        "be" if (has_sl and sl == rounded_open) else "entry")
                else:
                    self.trade_state[ticket] = "trail" if (has_sl and sl < rounded_open) else (
                        "be" if (has_sl and sl == rounded_open) else "entry")

            state = self.trade_state[ticket]

            # State 1: Entry Evaluation -> Move to Break-Even Cushion
            if state == "entry":
                price_moved = (
                    current_price - open_price if is_buy else open_price - current_price)
                if price_moved >= be_trigger_distance:
                    be_price = open_price + spread_cushion if is_buy else open_price - spread_cushion
                    if self.modify_position_sl(ticket, be_price, sym.digits):
                        self.trade_state[ticket] = "be"
                        continue

            # State 2: Break-Even Maintenance -> Initiate Trail Target
            elif state == "be":
                if is_buy:
                    if sl < (open_price + spread_cushion):
                        if self.modify_position_sl(ticket, open_price + spread_cushion, sym.digits):
                            continue
                    target_sl = current_price - trail_distance
                    if target_sl > (open_price + spread_cushion) and target_sl > sl:
                        if self.modify_position_sl(ticket, target_sl, sym.digits):
                            self.trade_state[ticket] = "trail"
                            continue
                else:
                    if sl == 0 or sl > (open_price - spread_cushion):
                        if self.modify_position_sl(ticket, open_price - spread_cushion, sym.digits):
                            continue
                    target_sl = current_price + trail_distance
                    if target_sl < (open_price - spread_cushion) and target_sl < sl:
                        if self.modify_position_sl(ticket, target_sl, sym.digits):
                            self.trade_state[ticket] = "trail"
                            continue

            # State 3: Trailing Scale
            elif state == "trail":
                if is_buy:
                    target_sl = current_price - trail_distance
                    if target_sl > sl:
                        self.modify_position_sl(ticket, target_sl, sym.digits)
                else:
                    target_sl = current_price + trail_distance
                    if sl == 0 or target_sl < sl:
                        self.modify_position_sl(ticket, target_sl, sym.digits)

        # Dynamic Garbage Collection for closed positions
        open_tickets = {p.ticket for p in positions if p.magic == self.magic}
        for t in list(self.trade_state.keys()):
            if t not in open_tickets:
                self.trade_state.pop(t, None)

    def place_order(self, action):
        positions = mt5.positions_get(symbol=self.symbol) or ()
        ticker_positions = [p for p in positions if p.magic == self.magic]
        if len(ticker_positions) >= self.max_positions:
            return

        tick = mt5.symbol_info_tick(self.symbol)
        sym = mt5.symbol_info(self.symbol)
        if not tick or not sym:
            return

        order_type = mt5.ORDER_TYPE_BUY if action == 0 else mt5.ORDER_TYPE_SELL
        entry_price = tick.ask if order_type == mt5.ORDER_TYPE_BUY else tick.bid

        target_sl_points = self.config.INITIAL_SL_POINTS.get(self.symbol, 300)
        sl_distance = target_sl_points * sym.point
        sl_price = entry_price - \
            sl_distance if order_type == mt5.ORDER_TYPE_BUY else entry_price + sl_distance

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": self.config.VOLUME,
            "type": order_type,
            "price": entry_price,
            "sl": round(sl_price, sym.digits),
            "magic": self.magic,
            "deviation": 10,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_FOK,
        }

        result = mt5.order_send(request)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            self.trade_state[result.order] = "entry"
            trade_logger.info(
                f"ORDER EXECUTION COMPLETED | {self.symbol} | Ticket: {result.order}")

    def tick_processing_loop(self):
        """Sequential heartbeat pump processing technical metrics and signal evaluation frames."""
        self.monitor_positions()
        candle = self.get_latest_candle_time()
        if candle is None:
            return

        if self.last_candle_time is None:
            self.last_candle_time = candle

        if candle != self.last_candle_time:
            self.last_candle_time = candle

            if self.is_active:
                state = self.compute_indicators()
                if state is not None:
                    probs = self.model.predict_proba(state)[0]
                    threshold = self.config.THRESHOLDS.get(self.symbol, 0.70)

                    if probs[1] >= threshold:
                        self.place_order(0)
                    elif probs[0] >= threshold:
                        self.place_order(1)


# =====================================================================
# CHILD TIME-COMPRESSION CLASS INHERITANCE EXAMPLES
# =====================================================================
class Trader5M(BaseTimeframeTrader):
    def __init__(self, symbol, index):
        import config_sample  # Pointed to sample placeholder maps
        super().__init__(
            symbol=symbol, config=config_sample, mt5_timeframe=mt5.TIMEFRAME_M5,
            model_subfolder="5m_models", base_magic=505000, index=index
        )


class Trader15M(BaseTimeframeTrader):
    def __init__(self, symbol, index):
        import config_sample
        super().__init__(
            symbol=symbol, config=config_sample, mt5_timeframe=mt5.TIMEFRAME_M15,
            model_subfolder="15m_models", base_magic=915000, index=index
        )


# =====================================================================
# SYSTEM CORE BOOTSTRAPPER
# =====================================================================
if __name__ == "__main__":
    audit_logger.info("Initializing system framework template metrics...")

    # Structural framework template checks
    if not mt5.initialize():
        audit_logger.critical(
            "🚨 MetaTrader5 execution environment not found offline.")
        quit()

    symbols_pool = ["EURUSD", "GBPUSD"]
    active_engines = []

    for i, sym in enumerate(symbols_pool):
        active_engines.append(Trader5M(sym, i))
        active_engines.append(Trader15M(sym, i))

    try:
        while True:
            for engine in active_engines:
                engine.tick_processing_loop()
            time.sleep(1)
    except KeyboardInterrupt:
        mt5.shutdown()
        audit_logger.info("🤖 Systems framework stopped cleanly.")
