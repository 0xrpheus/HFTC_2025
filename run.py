import shift
from time import sleep
from datetime import datetime, timedelta
import datetime as dt
from threading import Thread
import numpy as np
import statistics
import math
import random
import os
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("TradingAlgorithm")

# ================= CONFIGURATION PARAMETERS =================
# These parameters can be dynamically adjusted as market conditions change

# Trading thresholds
INITIAL_VOLATILITY_THRESHOLD = 0.0010  # 0.10% movement threshold - starts sensitive
MIN_VOLATILITY_THRESHOLD = 0.0005      # Minimum threshold (0.05%)
MAX_VOLATILITY_THRESHOLD = 0.0050      # Maximum threshold (0.5%)
PRICE_IMPACT_THRESHOLD = 0.0025        # Threshold for considering price impact (0.25%)

# Position management
BASE_POSITION_SIZE = 5                 # Base position size in lots
MIN_POSITION_SIZE = 1                  # Minimum position size
MAX_POSITION_SIZE = 15                 # Maximum position size
MAX_POSITION_VALUE = 100000            # Maximum $ value of any position

# Order management
MIN_ORDERS_TARGET = 250                # Target minimum orders per ticker
MIN_POSITIONS_TARGET = 15              # Target minimum positions per ticker
MAX_ACTIVE_ORDERS = 10                 # Maximum active orders per ticker
ORDER_REFRESH_SECONDS = 15             # How often to refresh limit orders

# Profit targets and stop losses
BASE_PROFIT_TARGET = 0.0035            # Base target (0.35%)
MIN_PROFIT_TARGET = 0.0015             # Minimum target (0.15%)
MAX_PROFIT_TARGET = 0.0100             # Maximum target (1.0%)
BASE_STOP_LOSS = 0.0045                # Base stop loss (0.45%)
MIN_STOP_LOSS = 0.0020                 # Minimum stop loss (0.20%)
MAX_STOP_LOSS = 0.0120                 # Maximum stop loss (1.2%)
TRAIL_STOP_PERCENT = 0.0025            # Trailing stop activation (0.25%)

# Data windows for indicators
PRICE_HISTORY_MAX = 500                # Maximum price history to maintain
VOL_LOOKBACK_WINDOW = 20               # Window for volatility calculation
RSI_PERIOD = 14                        # Period for RSI calculation
MACD_FAST = 12                         # Fast period for MACD
MACD_SLOW = 26                         # Slow period for MACD
MACD_SIGNAL = 9                        # Signal period for MACD
SWING_DETECTION_WINDOW = 10            # Window for swing detection

# Performance assessment
STRATEGY_EVAL_MINUTES = 10             # Evaluate strategy performance every X minutes
MIN_SHARPE_FOR_POSITION = 0.5          # Minimum Sharpe ratio to maintain full position sizing
TARGET_ORDERS_PER_MINUTE = 1.5         # Target orders per minute to meet minimums

# Strategy weights - these will auto-adjust based on what's working
STRATEGY_WEIGHTS = {
    "swing": 1.0,                      # Momentum/swing trading
    "reversion": 1.0,                  # Mean reversion trading
    "mm": 0.5,                         # Market making
    "breakout": 0.8,                   # Breakout/breakdown trading
}

# Risk management
MAX_DRAWDOWN_PCT = 0.02                # Maximum allowed drawdown (2%)
MAX_TICKER_ALLOCATION = 0.4            # Maximum capital allocated to single ticker (40%)

# ================= INDICATOR FUNCTIONS =================

def calculate_rsi(prices, period=14):
    """Calculate Relative Strength Index"""
    if len(prices) < period + 1:
        return 50  # Default to neutral when not enough data
    
    deltas = np.diff(prices)
    seed = deltas[:period]
    up = seed[seed >= 0].sum() / period
    down = -seed[seed < 0].sum() / period
    
    if down == 0:
        return 100
    
    rs = up / down
    rsi = 100 - (100 / (1 + rs))
    
    for delta in deltas[period:]:
        if delta > 0:
            upval = delta
            downval = 0
        else:
            upval = 0
            downval = -delta
            
        up = (up * (period - 1) + upval) / period
        down = (down * (period - 1) + downval) / period
        
        if down == 0:
            return 100
            
        rs = up / down
        rsi = 100 - (100 / (1 + rs))
    
    return rsi

def calculate_ema(prices, period):
    """Calculate Exponential Moving Average"""
    if len(prices) < period:
        return None

    multiplier = 2 / (period + 1)
    ema = [prices[0]]

    for price in prices[1:]:
        ema.append((price - ema[-1]) * multiplier + ema[-1])

    return ema[-1]

def calculate_macd(prices, fast_period=12, slow_period=26, signal_period=9):
    """Calculate MACD and Signal line"""
    if len(prices) < slow_period + signal_period:
        return None, None, None
    
    # Calculate EMAs
    fast_ema = []
    slow_ema = []
    
    # Fast EMA
    multiplier_fast = 2 / (fast_period + 1)
    ema = prices[0]
    fast_ema.append(ema)
    
    for price in prices[1:]:
        ema = (price - ema) * multiplier_fast + ema
        fast_ema.append(ema)
    
    # Slow EMA
    multiplier_slow = 2 / (slow_period + 1)
    ema = prices[0]
    slow_ema.append(ema)
    
    for price in prices[1:]:
        ema = (price - ema) * multiplier_slow + ema
        slow_ema.append(ema)
    
    # Calculate MACD line
    macd_line = [fast - slow for fast, slow in zip(fast_ema, slow_ema)]
    
    # Calculate signal line
    if len(macd_line) < signal_period:
        return macd_line[-1], None, None
    
    multiplier_signal = 2 / (signal_period + 1)
    signal = macd_line[:signal_period]
    signal_line = [sum(signal) / len(signal)]
    
    for macd_val in macd_line[signal_period:]:
        signal = (macd_val - signal_line[-1]) * multiplier_signal + signal_line[-1]
        signal_line.append(signal)
    
    # Calculate histogram
    histogram = macd_line[-1] - signal_line[-1]
    
    return macd_line[-1], signal_line[-1], histogram

def calculate_bollinger_bands(prices, period=20, stdev_factor=2):
    """Calculate Bollinger Bands"""
    if len(prices) < period:
        return None, None, None
    
    # Calculate SMA
    sma = sum(prices[-period:]) / period
    
    # Calculate standard deviation
    squared_diff = [(price - sma) ** 2 for price in prices[-period:]]
    stdev = math.sqrt(sum(squared_diff) / period)
    
    # Calculate upper and lower bands
    upper_band = sma + (stdev * stdev_factor)
    lower_band = sma - (stdev * stdev_factor)
    
    return sma, upper_band, lower_band

def calculate_atr(highs, lows, closes, period=14):
    """Calculate Average True Range"""
    if len(highs) < period or len(lows) < period or len(closes) < period:
        return 0.01  # Default to small value when not enough data
    
    true_ranges = []
    
    for i in range(1, len(closes)):
        high = highs[i]
        low = lows[i]
        prev_close = closes[i-1]
        
        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)
        
        true_range = max(tr1, tr2, tr3)
        true_ranges.append(true_range)
    
    if len(true_ranges) < period:
        return 0.01
        
    atr = sum(true_ranges[-period:]) / period
    return atr

def calculate_support_resistance(prices, window=20, threshold=0.01):
    """Identify support and resistance levels"""
    if len(prices) < window * 2:
        return [], []
    
    supports = []
    resistances = []
    
    for i in range(window, len(prices) - window):
        # Check if this is a local minimum (support)
        is_min = True
        for j in range(i - window, i):
            if prices[j] < prices[i]:
                is_min = False
                break
        for j in range(i + 1, i + window + 1):
            if j < len(prices) and prices[j] < prices[i]:
                is_min = False
                break
        
        if is_min:
            # Check if close to existing support
            is_new = True
            for support in supports:
                if abs(support - prices[i]) / support < threshold:
                    is_new = False
                    break
            
            if is_new:
                supports.append(prices[i])
        
        # Check if this is a local maximum (resistance)
        is_max = True
        for j in range(i - window, i):
            if prices[j] > prices[i]:
                is_max = False
                break
        for j in range(i + 1, i + window + 1):
            if j < len(prices) and prices[j] > prices[i]:
                is_max = False
                break
        
        if is_max:
            # Check if close to existing resistance
            is_new = True
            for resistance in resistances:
                if abs(resistance - prices[i]) / resistance < threshold:
                    is_new = False
                    break
            
            if is_new:
                resistances.append(prices[i])
    
    return supports, resistances

def calculate_volatility(prices, window=20):
    """Calculate historical volatility"""
    if len(prices) < window + 1:
        return 0.01  # Default value
    
    returns = [prices[i] / prices[i-1] - 1 for i in range(1, len(prices))]
    recent_returns = returns[-window:]
    
    return np.std(recent_returns) * np.sqrt(252 * 6.5 * 60)  # Annualized

def detect_trend(prices, short_window=10, long_window=30):
    """Detect price trend using moving averages"""
    if len(prices) < long_window:
        return 0  # No trend
    
    short_ma = sum(prices[-short_window:]) / short_window
    long_ma = sum(prices[-long_window:]) / long_window
    
    # Calculate trend strength as percentage difference
    trend_strength = (short_ma - long_ma) / long_ma
    
    if trend_strength > 0.001:  # 0.1% threshold
        return 1  # Uptrend
    elif trend_strength < -0.001:
        return -1  # Downtrend
    else:
        return 0  # No clear trend

def detect_swing(prices, volumes, window=SWING_DETECTION_WINDOW, threshold=INITIAL_VOLATILITY_THRESHOLD):
    """
    Detect potential swing points in the market
    Returns: 1 for bullish swing, -1 for bearish swing, 0 for no swing
    """
    if len(prices) < window:
        return 0
    
    # Calculate price change
    recent_prices = prices[-window:]
    price_change = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
    
    # Calculate volatility (standard deviation of returns)
    returns = [prices[i]/prices[i-1] - 1 for i in range(max(1, len(prices)-window), len(prices))]
    volatility = np.std(returns) if returns else 0
    
    # Calculate volume change
    if len(volumes) >= window:
        recent_volumes = volumes[-window:]
        avg_volume = sum(recent_volumes) / len(recent_volumes)
        latest_volume = recent_volumes[-1]
        volume_surge = latest_volume > avg_volume * 1.5
    else:
        volume_surge = True  # Assume volume surge if we don't have volume data
    
    # Additional signal strengthening: check for acceleration
    if len(prices) >= window * 2:
        # First half price change
        first_half = prices[-window*2:-window]
        first_change = (first_half[-1] - first_half[0]) / first_half[0] if first_half[0] != 0 else 0
        
        # Second half price change
        second_half = prices[-window:]
        second_change = (second_half[-1] - second_half[0]) / second_half[0] if second_half[0] != 0 else 0
        
        # Acceleration
        acceleration = second_change > first_change
    else:
        acceleration = False
    
    # Detect swing based on price change, volatility and volume
    if price_change > threshold and volatility > threshold*0.5 and (volume_surge or acceleration):
        return 1  # Bullish swing
    elif price_change < -threshold and volatility > threshold*0.5 and (volume_surge or acceleration):
        return -1  # Bearish swing
    
    return 0  # No swing

def is_overbought_oversold(rsi, threshold=70):
    """Determine if a stock is overbought or oversold based on RSI"""
    if rsi >= threshold:
        return -1  # Overbought - potential sell
    elif rsi <= 100 - threshold:
        return 1   # Oversold - potential buy
    
    return 0       # Neither

def calculate_breakout(prices, window=20, threshold=0.005):
    """Detect breakouts and breakdowns"""
    if len(prices) < window:
        return 0
    
    # Calculate the high and low of the lookback period
    period_high = max(prices[-window:-1])
    period_low = min(prices[-window:-1])
    current_price = prices[-1]
    
    # Calculate breakout/breakdown thresholds
    breakout_level = period_high * (1 + threshold)
    breakdown_level = period_low * (1 - threshold)
    
    if current_price > breakout_level:
        return 1  # Breakout
    elif current_price < breakdown_level:
        return -1  # Breakdown
    
    return 0  # No breakout/breakdown

def calculate_market_regime(prices, window=50):
    """Determine market regime (trending, mean-reverting, or random)"""
    if len(prices) < window:
        return "unknown"
    
    # Calculate returns
    returns = [prices[i]/prices[i-1] - 1 for i in range(1, len(prices))]
    
    # Calculate autocorrelation
    if len(returns) < 2:
        return "unknown"
    
    try:
        autocorr = np.corrcoef(returns[:-1], returns[1:])[0, 1]
    except:
        return "unknown"
    
    if autocorr > 0.2:
        return "trending"
    elif autocorr < -0.2:
        return "mean_reverting"
    else:
        return "random"

# ================= UTILITY FUNCTIONS =================

def cancel_orders(trader, ticker):
    """Cancel all waiting orders for a specific ticker"""
    order_ids = []
    
    for order in trader.get_waiting_list():
        if order.symbol == ticker:
            order_ids.append(order.id)
            trader.submit_cancellation(order)
            sleep(0.1)  # Small sleep to avoid overwhelming the system
    
    return order_ids

def close_positions(trader, ticker):
    """Close all positions for a specific ticker"""
    logger.info(f"Closing positions for {ticker}")
    item = trader.get_portfolio_item(ticker)

    long_shares = item.get_long_shares()
    if long_shares > 0:
        logger.info(f"Market selling {ticker} long shares = {long_shares}")
        order = shift.Order(shift.Order.Type.MARKET_SELL, ticker, int(long_shares/100))
        trader.submit_order(order)
        sleep(0.2)

    short_shares = item.get_short_shares()
    if short_shares > 0:
        logger.info(f"Market buying {ticker} short shares = {short_shares}")
        order = shift.Order(shift.Order.Type.MARKET_BUY, ticker, int(short_shares/100))
        trader.submit_order(order)
        sleep(0.2)
        
    return long_shares, short_shares

def get_current_prices(trader, ticker):
    """Get current market prices with error handling"""
    try:
        best_price = trader.get_best_price(ticker)
        if best_price.get_bid_price() <= 0 or best_price.get_ask_price() <= 0:
            logger.warning(f"Invalid prices for {ticker}")
            return None, None, None, None
            
        bid_price = best_price.get_bid_price()
        ask_price = best_price.get_ask_price()
        mid_price = (bid_price + ask_price) / 2
        spread = ask_price - bid_price
        
        return bid_price, ask_price, mid_price, spread
    except Exception as e:
        logger.error(f"Error getting prices for {ticker}: {e}")
        return None, None, None, None

def calculate_optimal_position_size(ticker, trader, current_position, 
                                   signal_strength, volatility, price, 
                                   max_position, risk_factor=1.0):
    """Calculate optimal position size based on multiple factors"""
    
    # Base size calculation
    base_size = BASE_POSITION_SIZE
    
    # Adjust for volatility (inverse relationship)
    if volatility > 0:
        vol_adj = min(1.5, max(0.5, 0.01 / volatility))
    else:
        vol_adj = 1.0
    
    # Adjust for signal strength (0.0 to 1.0)
    signal_adj = abs(signal_strength)
    
    # Adjust for remaining position capacity
    remaining_capacity = max_position - abs(current_position)
    if remaining_capacity <= 0:
        return 0
    
    # Adjust for price (ensure we don't exceed max position value)
    max_shares = int(MAX_POSITION_VALUE / (price * 100))  # Each lot is 100 shares
    
    # Calculate final size
    size = int(base_size * vol_adj * signal_adj * risk_factor)
    
    # Apply constraints
    size = max(MIN_POSITION_SIZE, min(size, int(remaining_capacity), max_shares))
    
    return size

def calculate_order_prices(bid_price, ask_price, mid_price, spread, 
                          direction, volatility, trend_strength):
    """Calculate optimal order prices based on market conditions"""
    
    # Base offset calculation as percentage of spread
    base_offset_pct = min(0.5, max(0.1, 0.2 + (volatility * 5)))
    
    # Adjust for trend strength
    if direction > 0:  # Bullish
        buy_offset_pct = base_offset_pct * (1 - (trend_strength * 0.5))
        sell_offset_pct = base_offset_pct * (1 + (trend_strength * 0.5))
    elif direction < 0:  # Bearish
        buy_offset_pct = base_offset_pct * (1 + (trend_strength * 0.5))
        sell_offset_pct = base_offset_pct * (1 - (trend_strength * 0.5))
    else:  # Neutral
        buy_offset_pct = sell_offset_pct = base_offset_pct
    
    # Calculate prices
    limit_buy_offset = max(0.01, spread * buy_offset_pct)
    limit_sell_offset = max(0.01, spread * sell_offset_pct)
    
    limit_buy_price = round(bid_price - limit_buy_offset, 2)
    limit_sell_price = round(ask_price + limit_sell_offset, 2)
    
    # Calculate aggresive prices (for market making)
    aggressive_buy_price = round(bid_price + (spread * 0.2), 2)
    aggressive_sell_price = round(ask_price - (spread * 0.2), 2)
    
    return {
        "limit_buy": limit_buy_price,
        "limit_sell": limit_sell_price,
        "aggressive_buy": aggressive_buy_price,
        "aggressive_sell": aggressive_sell_price,
        "market_buy": ask_price,
        "market_sell": bid_price
    }

def manage_active_trades(trader, ticker, active_trades, bid_price, ask_price, 
                        mid_price, current_time, adaptive_profit_target, 
                        adaptive_stop_loss):
    """Manage existing open trades - take profit, stop loss, or trailing stop"""
    
    trades_to_remove = []
    order_count = 0
    position_count = 0
    
    for trade_id, trade in active_trades.items():
        try:
            # Check for expiration
            if (current_time - trade["entry_time"]).total_seconds() > 600:  # 10 minute expiration
                if trade["direction"] > 0:  # Long trade
                    order = shift.Order(shift.Order.Type.MARKET_SELL, ticker, trade["size"])
                    trader.submit_order(order)
                else:  # Short trade
                    order = shift.Order(shift.Order.Type.MARKET_BUY, ticker, trade["size"])
                    trader.submit_order(order)
                
                order_count += 1
                position_count += 1
                trades_to_remove.append(trade_id)
                logger.info(f"Time-based exit on {ticker} trade at {mid_price:.2f}")
                continue
            
            # Update trailing stops if needed
            if "trailing_stop" in trade and trade["trailing_stop"]:
                if trade["direction"] > 0:  # Long trade
                    new_stop = mid_price * (1 - TRAIL_STOP_PERCENT)
                    if new_stop > trade["stop_price"]:
                        trade["stop_price"] = new_stop
                        logger.info(f"Updated trailing stop for {ticker} long to {new_stop:.2f}")
                else:  # Short trade
                    new_stop = mid_price * (1 + TRAIL_STOP_PERCENT)
                    if new_stop < trade["stop_price"]:
                        trade["stop_price"] = new_stop
                        logger.info(f"Updated trailing stop for {ticker} short to {new_stop:.2f}")
            
            # Check for exit conditions
            if trade["direction"] > 0:  # Long trade
                # Take profit
                if mid_price >= trade["target_price"]:
                    order = shift.Order(shift.Order.Type.MARKET_SELL, ticker, trade["size"])
                    trader.submit_order(order)
                    order_count += 1
                    position_count += 1
                    trades_to_remove.append(trade_id)
                    logger.info(f"Take profit on {ticker} long trade at {mid_price:.2f}")
                    
                    # Activate trailing stop for remaining position if partial
                    if "partial" in trade and trade["partial"]:
                        remaining_size = trade["original_size"] - trade["size"]
                        if remaining_size > 0:
                            new_trade_id = f"trail_{current_time.strftime('%H%M%S')}"
                            active_trades[new_trade_id] = {
                                "direction": 1,
                                "entry_price": trade["entry_price"],
                                "size": remaining_size,
                                "target_price": trade["target_price"] * 1.005,  # Extended target
                                "stop_price": trade["entry_price"] * 0.995,  # Tight stop
                                "entry_time": trade["entry_time"],
                                "trailing_stop": True
                            }
                
                # Stop loss
                elif mid_price <= trade["stop_price"]:
                    order = shift.Order(shift.Order.Type.MARKET_SELL, ticker, trade["size"])
                    trader.submit_order(order)
                    order_count += 1
                    position_count += 1
                    trades_to_remove.append(trade_id)
                    logger.info(f"Stop loss on {ticker} long trade at {mid_price:.2f}")
                
            elif trade["direction"] < 0:  # Short trade
                # Take profit
                if mid_price <= trade["target_price"]:
                    order = shift.Order(shift.Order.Type.MARKET_BUY, ticker, trade["size"])
                    trader.submit_order(order)
                    order_count += 1
                    position_count += 1
                    trades_to_remove.append(trade_id)
                    logger.info(f"Take profit on {ticker} short trade at {mid_price:.2f}")
                    
                    # Activate trailing stop for remaining position if partial
                    if "partial" in trade and trade["partial"]:
                        remaining_size = trade["original_size"] - trade["size"]
                        if remaining_size > 0:
                            new_trade_id = f"trail_{current_time.strftime('%H%M%S')}"
                            active_trades[new_trade_id] = {
                                "direction": -1,
                                "entry_price": trade["entry_price"],
                                "size": remaining_size,
                                "target_price": trade["target_price"] * 0.995,  # Extended target
                                "stop_price": trade["entry_price"] * 1.005,  # Tight stop
                                "entry_time": trade["entry_time"],
                                "trailing_stop": True
                            }
                
                # Stop loss
                elif mid_price >= trade["stop_price"]:
                    order = shift.Order(shift.Order.Type.MARKET_BUY, ticker, trade["size"])
                    trader.submit_order(order)
                    order_count += 1
                    position_count += 1
                    trades_to_remove.append(trade_id)
                    logger.info(f"Stop loss on {ticker} short trade at {mid_price:.2f}")
        except Exception as e:
            logger.error(f"Error managing trade {trade_id} for {ticker}: {e}")
            trades_to_remove.append(trade_id)
    
    # Remove completed trades
    for trade_id in trades_to_remove:
        active_trades.pop(trade_id, None)
    
    return order_count, position_count

def adjust_strategy_weights(strategy_performance, strategy_weights):
    """Dynamically adjust strategy weights based on performance"""
    total_pnl = sum([perf['pnl'] for perf in strategy_performance.values()])
    
    if total_pnl == 0:
        return strategy_weights
    
    # Calculate new weights
    new_weights = {}
    for strategy, perf in strategy_performance.items():
        # If strategy has a positive PnL, increase its weight
        if perf['pnl'] > 0:
            new_weights[strategy] = strategy_weights[strategy] * (1 + (perf['pnl'] / total_pnl * 0.5))
        # If strategy has a negative PnL, decrease its weight
        elif perf['pnl'] < 0:
            new_weights[strategy] = strategy_weights[strategy] * (1 - min(0.5, abs(perf['pnl'] / total_pnl * 0.5)))
        else:
            new_weights[strategy] = strategy_weights[strategy]
        
        # Ensure weights stay within reasonable bounds
        new_weights[strategy] = max(0.2, min(2.0, new_weights[strategy]))
    
    logger.info(f"Adjusted strategy weights: {new_weights}")
    return new_weights

def calculate_adaptive_thresholds(volatility_history, price_history):
    """Calculate adaptive thresholds based on historical data"""
    if not volatility_history or len(volatility_history) < 10:
        return INITIAL_VOLATILITY_THRESHOLD, BASE_PROFIT_TARGET, BASE_STOP_LOSS
    
    # Calculate median volatility
    median_volatility = sorted(volatility_history)[len(volatility_history)//2]
    
    # Calculate market regime
    market_regime = calculate_market_regime(price_history)
    
    # Adjust thresholds based on volatility and market regime
    if market_regime == "trending":
        volatility_threshold = max(MIN_VOLATILITY_THRESHOLD, min(MAX_VOLATILITY_THRESHOLD, median_volatility * 0.7))
        profit_target = max(MIN_PROFIT_TARGET, min(MAX_PROFIT_TARGET, median_volatility * 3.0))
        stop_loss = max(MIN_STOP_LOSS, min(MAX_STOP_LOSS, median_volatility * 4.0))
    elif market_regime == "mean_reverting":
        volatility_threshold = max(MIN_VOLATILITY_THRESHOLD, min(MAX_VOLATILITY_THRESHOLD, median_volatility * 1.2))
        profit_target = max(MIN_PROFIT_TARGET, min(MAX_PROFIT_TARGET, median_volatility * 2.5))
        stop_loss = max(MIN_STOP_LOSS, min(MAX_STOP_LOSS, median_volatility * 3.0))
    else:  # random or unknown
        volatility_threshold = max(MIN_VOLATILITY_THRESHOLD, min(MAX_VOLATILITY_THRESHOLD, median_volatility))
        profit_target = max(MIN_PROFIT_TARGET, min(MAX_PROFIT_TARGET, median_volatility * 2.8))
        stop_loss = max(MIN_STOP_LOSS, min(MAX_STOP_LOSS, median_volatility * 3.5))
    
    return volatility_threshold, profit_target, stop_loss

def log_trade_metrics(trader, ticker, start_time, order_count, position_count, 
                     pl_change, current_position, indicators):
    """Log trading metrics and indicator values"""
    elapsed_minutes = max(0.1, (trader.get_last_trade_time() - start_time).total_seconds() / 60)
    orders_per_minute = order_count / elapsed_minutes
    
    # Log basic stats
    stats_msg = (f"{ticker} - Position: {current_position:.1f}, "
                f"Orders: {order_count}, Positions: {position_count}, "
                f"P&L: ${pl_change:.2f}, "
                f"Orders/min: {orders_per_minute:.1f}")
    logger.info(stats_msg)
    
    # Log indicators if available
    if indicators:
        indicator_msg = (f"{ticker} Indicators - Trend: {indicators.get('trend', 'N/A')}, "
                       f"RSI: {indicators.get('rsi', 'N/A'):.1f}, "
                       f"Volatility: {indicators.get('volatility', 'N/A'):.6f}, "
                       f"Regime: {indicators.get('regime', 'N/A')}")
        logger.info(indicator_msg)

def should_execute_strategy(strategy, current_position, trend, rsi, strategy_weights, max_position):
    """Determine if a particular strategy should be executed based on current conditions"""
    # Check if strategy is enabled
    if strategy_weights.get(strategy, 0) < 0.2:
        return False
    
    # Check position capacity
    position_available = abs(current_position) < max_position
    if not position_available:
        return False
    
    # Strategy-specific conditions
    if strategy == "swing":
        # Swing strategy works best in trending markets
        return trend != 0
    elif strategy == "reversion":
        # Reversion strategy works when RSI indicates overbought/oversold
        return rsi < 30 or rsi > 70
    elif strategy == "breakout":
        # Breakout strategy needs room to run
        return position_available
    elif strategy == "mm":
        # Market making always active but at lower priority
        return True
    
    return True

# ================= MAIN STRATEGY FUNCTIONS =================

def execute_swing_strategy(trader, ticker, active_trades, price_history, volume_history, 
                         indicators, current_position, bid_price, ask_price, mid_price,
                         adaptive_threshold, adaptive_profit, adaptive_stop,
                         current_time, max_position):
    """Execute swing trading strategy"""
    if len(price_history) < SWING_DETECTION_WINDOW:
        return 0, 0
    
    order_count = 0
    position_count = 0
    
    # Detect swing
    swing_signal = detect_swing(price_history, volume_history, SWING_DETECTION_WINDOW, adaptive_threshold)
    
    if swing_signal != 0 and abs(current_position) < max_position:
        # Signal found
        logger.info(f"Detected {ticker} swing signal: {swing_signal} at price {mid_price:.2f}")
        
        # Calculate trade size
        signal_strength = abs(swing_signal)
        volatility = indicators.get("volatility", 0.01)
        
        trade_size = calculate_optimal_position_size(
            ticker, trader, current_position, signal_strength, 
            volatility, mid_price, max_position, 1.0)
        
        if trade_size > 0:
            if swing_signal > 0:  # Bullish swing
                # Place swing trade
                order = shift.Order(shift.Order.Type.MARKET_BUY, ticker, trade_size)
                trader.submit_order(order)
                order_count += 1
                position_count += 1
                
                # Record this trade for management
                trade_id = f"swing_long_{current_time.strftime('%H%M%S')}"
                target_price = mid_price * (1 + adaptive_profit)
                stop_price = mid_price * (1 - adaptive_stop)
                
                active_trades[trade_id] = {
                    "direction": 1,
                    "entry_price": mid_price,
                    "size": trade_size,
                    "target_price": target_price,
                    "stop_price": stop_price,
                    "entry_time": current_time,
                    "strategy": "swing"
                }
                
                logger.info(f"Opened long swing trade on {ticker} at {mid_price:.2f}, "
                           f"target: {target_price:.2f}, stop: {stop_price:.2f}")
                
            elif swing_signal < 0:  # Bearish swing
                # Place swing trade
                order = shift.Order(shift.Order.Type.MARKET_SELL, ticker, trade_size)
                trader.submit_order(order)
                order_count += 1
                position_count += 1
                
                # Record this trade for management
                trade_id = f"swing_short_{current_time.strftime('%H%M%S')}"
                target_price = mid_price * (1 - adaptive_profit)
                stop_price = mid_price * (1 + adaptive_stop)
                
                active_trades[trade_id] = {
                    "direction": -1,
                    "entry_price": mid_price,
                    "size": trade_size,
                    "target_price": target_price,
                    "stop_price": stop_price,
                    "entry_time": current_time,
                    "strategy": "swing"
                }
                
                logger.info(f"Opened short swing trade on {ticker} at {mid_price:.2f}, "
                           f"target: {target_price:.2f}, stop: {stop_price:.2f}")
    
    return order_count, position_count

def execute_reversion_strategy(trader, ticker, active_trades, price_history,
                             indicators, current_position, bid_price, ask_price, mid_price,
                             adaptive_threshold, adaptive_profit, adaptive_stop,
                             current_time, max_position):
    """Execute mean reversion strategy"""
    if len(price_history) < RSI_PERIOD:
        return 0, 0
    
    order_count = 0
    position_count = 0
    
    # Get oversold/overbought condition
    rsi = indicators.get("rsi", 50)
    
    # Signal based on RSI
    signal = is_overbought_oversold(rsi)
    
    if signal != 0 and abs(current_position) < max_position:
        # Calculate trade size - smaller for reversion trades
        signal_strength = abs(signal) * 0.7  # Reduce strength for reversion
        volatility = indicators.get("volatility", 0.01)
        
        trade_size = calculate_optimal_position_size(
            ticker, trader, current_position, signal_strength, 
            volatility, mid_price, max_position, 0.8)  # Lower risk factor
        
        if trade_size > 0:
            if signal > 0:  # Oversold - buy
                # Use limit order slightly below bid
                limit_price = round(bid_price * 0.998, 2)
                order = shift.Order(shift.Order.Type.LIMIT_BUY, ticker, trade_size, limit_price)
                trader.submit_order(order)
                order_count += 1
                
                # Record this limit order
                trade_id = f"reversion_buy_{current_time.strftime('%H%M%S')}"
                target_price = mid_price * (1 + adaptive_profit * 0.8)  # Lower targets for reversion
                stop_price = mid_price * (1 - adaptive_stop * 0.8)
                
                # We don't track limit orders in active_trades until they're filled
                logger.info(f"Placed reversion limit buy on {ticker} at {limit_price:.2f}")
                
            elif signal < 0:  # Overbought - sell
                # Use limit order slightly above ask
                limit_price = round(ask_price * 1.002, 2)
                order = shift.Order(shift.Order.Type.LIMIT_SELL, ticker, trade_size, limit_price)
                trader.submit_order(order)
                order_count += 1
                
                # Record this limit order
                trade_id = f"reversion_sell_{current_time.strftime('%H%M%S')}"
                target_price = mid_price * (1 - adaptive_profit * 0.8)
                stop_price = mid_price * (1 + adaptive_stop * 0.8)
                
                logger.info(f"Placed reversion limit sell on {ticker} at {limit_price:.2f}")
    
    return order_count, position_count

def execute_breakout_strategy(trader, ticker, active_trades, price_history,
                            indicators, current_position, bid_price, ask_price, mid_price,
                            adaptive_threshold, adaptive_profit, adaptive_stop,
                            current_time, max_position):
    """Execute breakout trading strategy"""
    if len(price_history) < 30:  # Need sufficient history
        return 0, 0
    
    order_count = 0
    position_count = 0
    
    # Detect breakout
    breakout_signal = calculate_breakout(price_history)
    
    if breakout_signal != 0 and abs(current_position) < max_position:
        # Calculate trade size
        signal_strength = abs(breakout_signal) * 0.9
        volatility = indicators.get("volatility", 0.01)
        
        trade_size = calculate_optimal_position_size(
            ticker, trader, current_position, signal_strength, 
            volatility, mid_price, max_position, 0.9)
        
        if trade_size > 0:
            if breakout_signal > 0:  # Bullish breakout
                # Place breakout trade
                order = shift.Order(shift.Order.Type.MARKET_BUY, ticker, trade_size)
                trader.submit_order(order)
                order_count += 1
                position_count += 1
                
                # Record this trade for management
                trade_id = f"breakout_long_{current_time.strftime('%H%M%S')}"
                # Higher profit target for breakouts
                target_price = mid_price * (1 + adaptive_profit * 1.2)
                stop_price = mid_price * (1 - adaptive_stop * 0.9)
                
                active_trades[trade_id] = {
                    "direction": 1,
                    "entry_price": mid_price,
                    "size": trade_size,
                    "target_price": target_price,
                    "stop_price": stop_price,
                    "entry_time": current_time,
                    "strategy": "breakout",
                    "trailing_stop": True  # Use trailing stops for breakouts
                }
                
                logger.info(f"Opened long breakout trade on {ticker} at {mid_price:.2f}, "
                           f"target: {target_price:.2f}, stop: {stop_price:.2f}")
                
            elif breakout_signal < 0:  # Bearish breakout
                # Place breakout trade
                order = shift.Order(shift.Order.Type.MARKET_SELL, ticker, trade_size)
                trader.submit_order(order)
                order_count += 1
                position_count += 1
                
                # Record this trade for management
                trade_id = f"breakout_short_{current_time.strftime('%H%M%S')}"
                target_price = mid_price * (1 - adaptive_profit * 1.2)
                stop_price = mid_price * (1 + adaptive_stop * 0.9)
                
                active_trades[trade_id] = {
                    "direction": -1,
                    "entry_price": mid_price,
                    "size": trade_size,
                    "target_price": target_price,
                    "stop_price": stop_price,
                    "entry_time": current_time,
                    "strategy": "breakout",
                    "trailing_stop": True
                }
                
                logger.info(f"Opened short breakout trade on {ticker} at {mid_price:.2f}, "
                           f"target: {target_price:.2f}, stop: {stop_price:.2f}")
    
    return order_count, position_count

def execute_market_making(trader, ticker, order_prices, spread,
                        current_position, max_position, last_mm_orders):
    """Execute market making strategy"""
    # Only allow a few market making orders at a time
    if len(last_mm_orders) >= 5:
        return 0
    
    order_count = 0
    
    # Calculate trade size - small for market making
    trade_size = 1
    
    # Place limit orders on both sides of the market
    if current_position < max_position:
        buy_order = shift.Order(shift.Order.Type.LIMIT_BUY, ticker, trade_size, order_prices["limit_buy"])
        trader.submit_order(buy_order)
        order_count += 1
        last_mm_orders.append(("buy", buy_order.id))
    
    if current_position > -max_position:
        sell_order = shift.Order(shift.Order.Type.LIMIT_SELL, ticker, trade_size, order_prices["limit_sell"])
        trader.submit_order(sell_order)
        order_count += 1
        last_mm_orders.append(("sell", sell_order.id))
    
    # Remove oldest orders if we have too many
    while len(last_mm_orders) > 5:
        last_mm_orders.pop(0)
    
    return order_count

def ensure_min_orders(trader, ticker, order_count, order_target, time_elapsed, total_time,
                     current_position, bid_price, ask_price, spread):
    """Ensure minimum order count is met by placing additional orders if needed"""
    if time_elapsed <= 0 or total_time <= 0:
        return 0
    
    # Calculate expected order count by this time
    expected_order_rate = order_target / total_time
    expected_orders = time_elapsed * expected_order_rate
    
    # If we're behind on orders
    if order_count < expected_orders - 5:
        orders_to_place = min(10, int(expected_orders - order_count))
        
        # Place small orders with minimal risk
        new_orders = 0
        for i in range(orders_to_place):
            # Alternate buy/sell
            if i % 2 == 0:
                # Buy far from current price
                if current_position < MAX_POSITION_SIZE:
                    limit_price = round(bid_price * 0.98, 2)  # 2% below current bid
                    order = shift.Order(shift.Order.Type.LIMIT_BUY, ticker, 1, limit_price)
                    trader.submit_order(order)
                    new_orders += 1
            else:
                # Sell far from current price
                if current_position > -MAX_POSITION_SIZE:
                    limit_price = round(ask_price * 1.02, 2)  # 2% above current ask
                    order = shift.Order(shift.Order.Type.LIMIT_SELL, ticker, 1, limit_price)
                    trader.submit_order(order)
                    new_orders += 1
        
        if new_orders > 0:
            logger.info(f"Placed {new_orders} additional orders for {ticker} to meet minimum requirements")
        
        return new_orders
    
    return 0

def strategy(trader: shift.Trader, ticker: str, endtime):
    """Main trading strategy function for a single ticker"""
    logger.info(f"Starting advanced trading strategy for {ticker}")
    initial_pl = trader.get_portfolio_item(ticker).get_realized_pl()
    
    # Trading frequency
    check_freq = 0.5  # Check market every 0.5 seconds

    # Data storage
    price_history = []
    bid_history = []
    ask_history = []
    high_history = []
    low_history = []
    volume_history = []
    volatility_history = []
    
    # Trade management
    active_trades = {}
    last_mm_orders = []
    last_order_refresh = trader.get_last_trade_time()
    last_swing_detection = trader.get_last_trade_time()
    
    # Performance tracking
    strategy_performance = {
        "swing": {"pnl": 0, "trades": 0},
        "reversion": {"pnl": 0, "trades": 0},
        "breakout": {"pnl": 0, "trades": 0},
        "mm": {"pnl": 0, "trades": 0}
    }
    strategy_weights = STRATEGY_WEIGHTS.copy()
    last_strategy_eval = trader.get_last_trade_time()
    
    # Thresholds - will be adjusted dynamically
    adaptive_volatility_threshold = INITIAL_VOLATILITY_THRESHOLD
    adaptive_profit_target = BASE_PROFIT_TARGET
    adaptive_stop_loss = BASE_STOP_LOSS
    
    # Counters and timers
    order_count = 0
    position_count = 0
    start_time = trader.get_last_trade_time()
    last_metric_update = start_time
    
    # Position limits
    max_position = MAX_POSITION_SIZE
    
    # Main trading loop
    while trader.get_last_trade_time() < endtime:
        try:
            # First, manage existing orders and positions
            current_time = trader.get_last_trade_time()
            
            # Cancel existing orders periodically
            if (current_time - last_order_refresh).total_seconds() > ORDER_REFRESH_SECONDS:
                cancel_orders(trader, ticker)
                last_order_refresh = current_time
            
            # Get current market data
            bid_price, ask_price, mid_price, spread = get_current_prices(trader, ticker)
            if None in (bid_price, ask_price, mid_price, spread):
                logger.warning(f"Invalid prices for {ticker}, skipping cycle")
                sleep(0.5)
                continue

            # Get current position
            portfolio_item = trader.get_portfolio_item(ticker)
            current_position = (portfolio_item.get_long_shares() - portfolio_item.get_short_shares()) / 100
            
            # Update price history
            price_history.append(mid_price)
            bid_history.append(bid_price)
            ask_history.append(ask_price)
            
            # Update high, low, volume history
            if len(price_history) > 1:
                high_history.append(max(price_history[-2:]))
                low_history.append(min(price_history[-2:]))
                # Approximate volume
                volume_history.append(1)
            
            # Keep history at a reasonable size
            if len(price_history) > PRICE_HISTORY_MAX:
                price_history = price_history[-PRICE_HISTORY_MAX:]
                bid_history = bid_history[-PRICE_HISTORY_MAX:]
                ask_history = ask_history[-PRICE_HISTORY_MAX:]
                high_history = high_history[-PRICE_HISTORY_MAX:]
                low_history = low_history[-PRICE_HISTORY_MAX:]
                volume_history = volume_history[-PRICE_HISTORY_MAX:]
            
            # Calculate indicators
            indicators = {}
            
            # Calculate volatility
            if len(price_history) >= VOL_LOOKBACK_WINDOW:
                volatility = calculate_volatility(price_history, VOL_LOOKBACK_WINDOW)
                indicators["volatility"] = volatility
                volatility_history.append(volatility)
                
                # Keep volatility history at reasonable size
                if len(volatility_history) > 50:
                    volatility_history = volatility_history[-50:]
            
            # Calculate RSI
            if len(price_history) >= RSI_PERIOD:
                rsi = calculate_rsi(price_history, RSI_PERIOD)
                indicators["rsi"] = rsi
            
            # Calculate trend
            trend = detect_trend(price_history)
            indicators["trend"] = trend
            
            # Calculate market regime
            if len(price_history) >= 50:
                regime = calculate_market_regime(price_history)
                indicators["regime"] = regime
            
            # Calculate support/resistance
            if len(price_history) >= 50:
                supports, resistances = calculate_support_resistance(price_history)
                indicators["supports"] = supports
                indicators["resistances"] = resistances
            
            # Calculate adaptive thresholds
            if len(volatility_history) >= 10:
                adaptive_volatility_threshold, adaptive_profit_target, adaptive_stop_loss = \
                    calculate_adaptive_thresholds(volatility_history, price_history)
            
            # Calculate order prices
            order_prices = calculate_order_prices(
                bid_price, ask_price, mid_price, spread, 
                trend, indicators.get("volatility", 0.01), abs(trend))
            
            # ===== Manage existing trades =====
            manage_orders, manage_positions = manage_active_trades(
                trader, ticker, active_trades, bid_price, ask_price, mid_price,
                current_time, adaptive_profit_target, adaptive_stop_loss)
                
            order_count += manage_orders
            position_count += manage_positions
            
            # ===== Execute strategies =====
            # Only check for new entry signals periodically
            if (current_time - last_swing_detection).total_seconds() > 15:  # Every 15 seconds
                # Swing trading strategy
                if should_execute_strategy("swing", current_position, trend, 
                                         indicators.get("rsi", 50), strategy_weights, max_position):
                    swing_orders, swing_positions = execute_swing_strategy(
                        trader, ticker, active_trades, price_history, volume_history, 
                        indicators, current_position, bid_price, ask_price, mid_price,
                        adaptive_volatility_threshold, adaptive_profit_target, adaptive_stop_loss,
                        current_time, max_position)
                    
                    order_count += swing_orders
                    position_count += swing_positions
                
                # Mean reversion strategy
                if should_execute_strategy("reversion", current_position, trend, 
                                         indicators.get("rsi", 50), strategy_weights, max_position):
                    rev_orders, rev_positions = execute_reversion_strategy(
                        trader, ticker, active_trades, price_history,
                        indicators, current_position, bid_price, ask_price, mid_price,
                        adaptive_volatility_threshold, adaptive_profit_target, adaptive_stop_loss,
                        current_time, max_position)
                    
                    order_count += rev_orders
                    position_count += rev_positions
                
                # Breakout strategy
                if should_execute_strategy("breakout", current_position, trend, 
                                         indicators.get("rsi", 50), strategy_weights, max_position):
                    breakout_orders, breakout_positions = execute_breakout_strategy(
                        trader, ticker, active_trades, price_history,
                        indicators, current_position, bid_price, ask_price, mid_price,
                        adaptive_volatility_threshold, adaptive_profit_target, adaptive_stop_loss,
                        current_time, max_position)
                    
                    order_count += breakout_orders
                    position_count += breakout_positions
                
                last_swing_detection = current_time
            
            # Market making - always execute to ensure we meet order minimums
            if should_execute_strategy("mm", current_position, trend, 
                                     indicators.get("rsi", 50), strategy_weights, max_position):
                mm_orders = execute_market_making(
                    trader, ticker, order_prices, spread,
                    current_position, max_position, last_mm_orders)
                
                order_count += mm_orders
            
            # ===== Ensure minimum orders =====
            # Check if we need to place additional orders to meet requirements
            time_elapsed = (current_time - start_time).total_seconds()
            total_time = (endtime - start_time).total_seconds()
            
            additional_orders = ensure_min_orders(
                trader, ticker, order_count, MIN_ORDERS_TARGET, time_elapsed, total_time,
                current_position, bid_price, ask_price, spread)
            
            order_count += additional_orders
            
            # ===== Evaluate strategy performance =====
            if (current_time - last_strategy_eval).total_seconds() > STRATEGY_EVAL_MINUTES * 60:
                # Calculate realized PnL for each strategy
                current_pl = trader.get_portfolio_item(ticker).get_realized_pl()
                pl_change = current_pl - initial_pl
                
                # Simple attribution based on active trades
                strategy_counts = {"swing": 0, "reversion": 0, "breakout": 0, "mm": 0}
                for trade in active_trades.values():
                    if "strategy" in trade:
                        strategy_counts[trade["strategy"]] = strategy_counts.get(trade["strategy"], 0) + 1
                
                total_trades = sum(strategy_counts.values())
                for strategy, count in strategy_counts.items():
                    if total_trades > 0:
                        # Attribute PnL based on number of trades
                        attribution = count / total_trades
                        strategy_performance[strategy]["pnl"] = pl_change * attribution
                        strategy_performance[strategy]["trades"] = count
                
                # Adjust strategy weights
                strategy_weights = adjust_strategy_weights(strategy_performance, strategy_weights)
                
                # Reset evaluation timer
                last_strategy_eval = current_time
            
            # ===== Logging =====
            if (current_time - last_metric_update).total_seconds() >= 60:  # Every minute
                # Calculate P&L
                current_pl = trader.get_portfolio_item(ticker).get_realized_pl()
                pl_change = current_pl - initial_pl
                
                # Log metrics
                log_trade_metrics(trader, ticker, start_time, order_count, position_count, 
                                pl_change, current_position, indicators)
                
                last_metric_update = current_time
            
            # Sleep before next cycle
            sleep(check_freq)
            
        except Exception as e:
            logger.error(f"Error in strategy for {ticker}: {e}", exc_info=True)
            sleep(1)
    
    # ===== End of day cleanup =====
    try:
        logger.info(f"End of day cleanup for {ticker}")
        cancel_orders(trader, ticker)
        close_positions(trader, ticker)
        
        final_pl = trader.get_portfolio_item(ticker).get_realized_pl() - initial_pl
        logger.info(f"Strategy for {ticker} completed with {order_count} orders and {position_count} positions.")
        logger.info(f"Total P&L for {ticker}: ${final_pl:.2f}")
        
        return final_pl, order_count, position_count
    except Exception as e:
        logger.error(f"Error during cleanup for {ticker}: {e}", exc_info=True)
        return 0, order_count, position_count

def main(trader):
    """Main function to run the multi-ticker trading strategy"""
    # Best tickers for swing trading - prioritize high beta stocks that move more
    ticker_priorities = [
        "AAPL",  # Apple - tech leader
        "JPM",   # JP Morgan - financial leader 
        "BA",    # Boeing - industrial/defense with high volatility
        "CSCO",  # Cisco - networking/tech with moderate volatility
        "MSFT",  # Microsoft - tech with lower volatility
        "GS",    # Goldman Sachs - financial with high volatility
        "CAT",   # Caterpillar - industrial/cyclical
        "XOM",   # Exxon - energy/cyclical
        "DIS",   # Disney - consumer/media
        "V"      # Visa - payments/financial
    ]

    # Determine trading hours
    current = trader.get_last_trade_time()
    start_time = current
    
    # End 5 minutes before market close
    end_time = datetime.combine(current.date(), dt.time(15, 55, 0))
    
    logger.info(f"Advanced trading strategy starting at {start_time}, will run until {end_time}")

    # Track initial portfolio values
    initial_total_pl = trader.get_portfolio_summary().get_total_realized_pl()
    initial_bp = trader.get_portfolio_summary().get_total_bp()
    
    logger.info(f"Initial buying power: ${initial_bp:.2f}")

    # Select tickers to trade - for high volatility swing days, focus on 4 high-beta tickers
    active_tickers = ticker_priorities[:4]
    logger.info(f"Trading the following symbols: {active_tickers}")

    # Launch trading threads
    threads = []
    thread_objects = {}
    
    for ticker in active_tickers:
        thread = Thread(target=strategy, args=(trader, ticker, end_time))
        thread.daemon = True  # Make threads daemon to ensure they exit with the main thread
        threads.append(thread)
        thread_objects[ticker] = thread
        thread.start()
        sleep(1)  # Stagger thread starts

    # Wait for threads to complete
    results = {}
    for ticker, thread in thread_objects.items():
        logger.info(f"Waiting for {ticker} thread to complete")
        thread.join(timeout=60*60*7)  # 7-hour timeout
        results[ticker] = {
            "pl": 0,
            "orders": 0,
            "positions": 0
        }

    # Final cleanup
    for ticker in active_tickers:
        try:
            cancel_orders(trader, ticker)
            close_positions(trader, ticker)
        except Exception as e:
            logger.error(f"Error during final cleanup for {ticker}: {e}")

    # Calculate final results
    final_bp = trader.get_portfolio_summary().get_total_bp()
    final_pl = trader.get_portfolio_summary().get_total_realized_pl() - initial_total_pl
    
    total_orders = sum(r.get("orders", 0) for r in results.values())
    total_positions = sum(r.get("positions", 0) for r in results.values())

    logger.info("\n===== STRATEGY COMPLETE =====")
    logger.info(f"Final buying power: ${final_bp:.2f}")
    logger.info(f"Total profit/loss: ${final_pl:.2f}")
    logger.info(f"Return on capital: {(final_pl / initial_bp * 100):.2f}%")
    logger.info(f"Total orders placed: {total_orders}")
    logger.info(f"Total positions opened: {total_positions}")
    
    return final_pl, total_orders

if __name__ == '__main__':
    with shift.Trader("dolla-dolla-bills-yall") as trader:
        try:
            # Connect to the server
            logger.info("Connecting to SHIFT...")
            trader.connect("initiator.cfg", "Zxz7Qxa9")
            sleep(1)
            
            # Subscribe to order book data
            logger.info("Subscribing to order book data...")
            trader.sub_all_order_book()
            sleep(1)

            # Run main strategy
            logger.info("Starting main strategy...")
            main(trader)
            
        except Exception as e:
            logger.error(f"Fatal error in main program: {e}", exc_info=True)
            try:
                # Emergency cleanup
                logger.info("Performing emergency cleanup...")
                tickers = ["AAPL", "JPM", "BA", "CSCO"]
                for ticker in tickers:
                    cancel_orders(trader, ticker)
                    close_positions(trader, ticker)
            except Exception as cleanup_error:
                logger.error(f"Failed to perform emergency cleanup: {cleanup_error}")