import shift
from datetime import datetime, timedelta
import time
import numpy as np
from threading import Thread, Lock
import pandas as pd
import logging
import math
import random
import collections
import traceback

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("ZI_Advanced_Strategy")

# ================= CONFIGURATION =================
# Tickers
TICKER_LIST = ["CS1", "CS2"]

# Minimum requirements
MIN_ORDERS_TARGET = 250       # Target for minimum orders (slightly higher than 200 requirement)
MIN_POSITIONS_TARGET = 20     # Target for minimum positions (higher than 10 requirement)

# Risk parameters
MAX_POSITION_SIZE = 5         # Maximum position size per ticker (reduced from 10)
MAX_POSITION_VALUE = 50000    # Maximum position value per ticker (reduced from 100000)
MAX_DRAWDOWN_PCT = 0.03       # Maximum drawdown allowed (3% - tighter than original 5%)
MAX_EXPOSURE_PCT = 0.25       # Maximum total capital exposure

# Mean Reversion parameters
MR_WINDOW_SHORT = 20          # Short window for mean reversion
MR_WINDOW_LONG = 50           # Long window for mean reversion
MR_ENTRY_THRESHOLD = 2.0      # Entry threshold - increased for stronger signals
MR_EXIT_THRESHOLD = 0.5       # Exit threshold - increased to exit faster
MR_STOP_LOSS_THRESHOLD = 2.0  # Stop loss threshold - tightened

# Liquidity Provision parameters
LP_SPREAD_FACTOR = 0.5        # Increased from 0.3 for more aggressive spread capture
LP_MIN_SPREAD_BPS = 3         # Minimum spread in basis points (reduced to be less selective)
LP_MAX_SPREAD_BPS = 40        # Maximum spread in basis points (reduced to avoid extreme cases)
LP_ORDER_REFRESH_SECONDS = 10 # How often to refresh LP orders (reduced from 30)
LP_MAX_ORDERS_PER_SIDE = 3    # Maximum orders per side for LP

# Rebate Harvesting parameters
RH_MIN_ORDERS = 3             # Minimum orders to place per round
RH_MAX_ORDERS = 10            # Maximum orders to place per round
RH_MIN_DISTANCE = 0.03        # Minimum distance from mid price (3%)
RH_MAX_DISTANCE = 0.10        # Maximum distance from mid price (10%)
RH_ORDER_SIZE = 1             # Size of rebate harvesting orders

# Order Book Imbalance parameters
OBI_THRESHOLD = 0.65          # Threshold for order book imbalance signal
OBI_DEPTH = 5                 # Depth of order book to analyze
OBI_CHECK_INTERVAL = 5        # How often to check order book imbalance

# Order execution parameters
CHECK_INTERVAL = 0.2          # Main loop check interval in seconds
ORDER_CHECK_INTERVAL = 3      # How often to check order status
POSITION_CHECK_INTERVAL = 8   # How often to check positions (reduced from 10)
STATS_UPDATE_INTERVAL = 15    # How often to update market statistics
MAX_POSITION_HOLD_TIME = 300  # Maximum position hold time in seconds (5 minutes)
STRATEGY_UPDATE_INTERVAL = 300 # How often to update strategy weights (5 minutes)

# Strategy weights (adjusted to favor liquidity provision)
STRATEGY_WEIGHTS = {
    "mean_reversion": 0.3,       # Reduced from 0.5
    "liquidity_provision": 0.5,  # Increased from 0.3
    "order_book_imbalance": 0.1, # New strategy
    "rebate_harvesting": 0.1     # New strategy
}

# Statistical parameters
ZSCORE_WINDOW = 50             # Window for z-score calculation
VOL_WINDOW = 20                # Window for volatility calculation
TREND_WINDOW_SHORT = 10        # Short window for trend calculation
TREND_WINDOW_LONG = 30         # Long window for trend calculation
MIN_DATA_POINTS = 20           # Minimum data points before making decisions

# ================= DATA STRUCTURES =================

# Price and statistics tracker
class MarketData:
    def __init__(self, ticker, max_window=200):
        self.ticker = ticker
        self.max_window = max_window
        
        # Price data
        self.timestamps = collections.deque(maxlen=max_window)
        self.prices = collections.deque(maxlen=max_window)
        self.bid_prices = collections.deque(maxlen=max_window)
        self.ask_prices = collections.deque(maxlen=max_window)
        self.spreads = collections.deque(maxlen=max_window)
        self.volumes = collections.deque(maxlen=max_window)
        self.returns = collections.deque(maxlen=max_window)
        
        # Statistics
        self.volatility = 0.01
        self.mean_price = None
        self.std_price = None
        self.zscore = 0
        self.mean_spread = None
        self.ma_short = None
        self.ma_long = None
        self.trend = 0
        
        # Order book metrics
        self.bid_depth = 0
        self.ask_depth = 0
        self.book_imbalance = 0
        
        # Last update
        self.last_update = None
    
    def update(self, timestamp, bid, ask, mid_price, spread, volume=1):
        """Update market data with new prices"""
        self.timestamps.append(timestamp)
        self.prices.append(mid_price)
        self.bid_prices.append(bid)
        self.ask_prices.append(ask)
        self.spreads.append(spread)
        self.volumes.append(volume)
        
        # Calculate return if we have at least 2 prices
        if len(self.prices) >= 2:
            ret = mid_price / self.prices[-2] - 1
            self.returns.append(ret)
        
        self.last_update = timestamp
        
        # Update statistics
        self.update_statistics()
    
    def update_statistics(self):
        """Update calculated statistics"""
        # Need enough data
        if len(self.prices) < 5:
            return
        
        # Calculate basic statistics
        self.mean_price = np.mean(list(self.prices)[-ZSCORE_WINDOW:]) if len(self.prices) >= ZSCORE_WINDOW else np.mean(self.prices)
        self.std_price = np.std(list(self.prices)[-ZSCORE_WINDOW:]) if len(self.prices) >= ZSCORE_WINDOW else np.std(self.prices)
        
        # Z-score (how many standard deviations from mean)
        if self.std_price > 0:
            self.zscore = (self.prices[-1] - self.mean_price) / self.std_price
        else:
            self.zscore = 0
            
        # Moving averages
        if len(self.prices) >= MR_WINDOW_SHORT:
            self.ma_short = np.mean(list(self.prices)[-MR_WINDOW_SHORT:])
        else:
            self.ma_short = self.prices[-1]
            
        if len(self.prices) >= MR_WINDOW_LONG:
            self.ma_long = np.mean(list(self.prices)[-MR_WINDOW_LONG:])
        else:
            self.ma_long = self.prices[-1]
        
        # Calculate trend
        if len(self.prices) >= TREND_WINDOW_LONG:
            short_trend = np.mean(list(self.prices)[-TREND_WINDOW_SHORT:])
            long_trend = np.mean(list(self.prices)[-TREND_WINDOW_LONG:])
            
            if short_trend > long_trend * 1.002:
                self.trend = 1  # Uptrend
            elif short_trend < long_trend * 0.998:
                self.trend = -1  # Downtrend
            else:
                self.trend = 0  # No clear trend
        
        # Volatility (standard deviation of returns)
        if len(self.returns) >= VOL_WINDOW:
            self.volatility = np.std(list(self.returns)[-VOL_WINDOW:])
            # Annualize volatility - approximate for high-frequency context
            self.volatility = self.volatility * np.sqrt(252 * 390)
        else:
            self.volatility = 0.01
        
        # Mean spread
        self.mean_spread = np.mean(list(self.spreads))

# Position tracker
class PositionTracker:
    def __init__(self):
        self.positions = {}  # ticker -> position details
        self.lock = Lock()
    
    def update_position(self, ticker, size, entry_price, entry_time, strategy):
        with self.lock:
            self.positions[ticker] = {
                'size': size,
                'entry_price': entry_price,
                'entry_time': entry_time,
                'strategy': strategy
            }
    
    def get_position(self, ticker):
        with self.lock:
            return self.positions.get(ticker, {})
    
    def clear_position(self, ticker):
        with self.lock:
            if ticker in self.positions:
                del self.positions[ticker]

# ================= UTILITY FUNCTIONS =================

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

def get_current_position(trader, ticker):
    """Get current position for a ticker"""
    try:
        portfolio_item = trader.get_portfolio_item(ticker)
        long_shares = portfolio_item.get_long_shares()
        short_shares = portfolio_item.get_short_shares()
        current_position = (long_shares - short_shares) / 100  # Convert to lots
        return current_position, portfolio_item
    except Exception as e:
        logger.error(f"Error getting position for {ticker}: {e}")
        return 0, None

def calculate_position_value(trader, ticker):
    """Calculate current position value"""
    try:
        # Get prices and position
        bid_price, ask_price, mid_price, spread = get_current_prices(trader, ticker)
        if None in (bid_price, ask_price, mid_price, spread):
            return 0
            
        current_position, portfolio_item = get_current_position(trader, ticker)
        
        # Calculate value
        if current_position > 0:  # Long position
            value = current_position * 100 * bid_price  # Use bid for long positions
        elif current_position < 0:  # Short position
            value = abs(current_position) * 100 * ask_price  # Use ask for short positions
        else:
            value = 0
            
        return value
    except Exception as e:
        logger.error(f"Error calculating position value for {ticker}: {e}")
        return 0

def calculate_unrealized_pnl(trader, ticker):
    """Calculate unrealized P&L for a position"""
    try:
        portfolio_item = trader.get_portfolio_item(ticker)
        current_position, _ = get_current_position(trader, ticker)
        
        # If no position, no unrealized P&L
        if current_position == 0:
            return 0
            
        # Get current prices
        bid_price, ask_price, mid_price, spread = get_current_prices(trader, ticker)
        if None in (bid_price, ask_price, mid_price, spread):
            return 0
        
        # Calculate unrealized P&L
        long_shares = portfolio_item.get_long_shares()
        short_shares = portfolio_item.get_short_shares()
        long_price = portfolio_item.get_long_price()
        short_price = portfolio_item.get_short_price()
        
        unrealized_pnl = 0
        
        if long_shares > 0:
            unrealized_pnl += (bid_price - long_price) * long_shares / 100  # Use bid for long positions
            
        if short_shares > 0:
            unrealized_pnl += (short_price - ask_price) * short_shares / 100  # Use ask for short positions
            
        return unrealized_pnl
    except Exception as e:
        logger.error(f"Error calculating unrealized P&L for {ticker}: {e}")
        return 0

def cancel_orders(trader, ticker):
    """Cancel all waiting orders for a specific ticker"""
    try:
        cancelled_count = 0
        for order in trader.get_waiting_list():
            if order.symbol == ticker:
                trader.submit_cancellation(order)
                cancelled_count += 1
                time.sleep(0.05)  # Small sleep to avoid overwhelming the system
        
        if cancelled_count > 0:
            logger.info(f"Cancelled {cancelled_count} orders for {ticker}")
        
        return cancelled_count
    except Exception as e:
        logger.error(f"Error cancelling orders for {ticker}: {e}")
        return 0

def close_positions(trader, ticker):
    """Close all positions for a specific ticker"""
    try:
        logger.info(f"Closing positions for {ticker}")
        item = trader.get_portfolio_item(ticker)

        long_shares = item.get_long_shares()
        if long_shares > 0:
            logger.info(f"Market selling {ticker} long shares = {long_shares}")
            order = shift.Order(shift.Order.Type.MARKET_SELL, ticker, int(long_shares/100))
            trader.submit_order(order)
            time.sleep(0.2)

        short_shares = item.get_short_shares()
        if short_shares > 0:
            logger.info(f"Market buying {ticker} short shares = {short_shares}")
            order = shift.Order(shift.Order.Type.MARKET_BUY, ticker, int(short_shares/100))
            trader.submit_order(order)
            time.sleep(0.2)
            
        return long_shares, short_shares
    except Exception as e:
        logger.error(f"Error closing positions for {ticker}: {e}")
        return 0, 0

def submit_order(trader, order, retry_attempts=3):
    """Submit an order with retry mechanism"""
    for attempt in range(retry_attempts):
        try:
            trader.submit_order(order)
            return True
        except Exception as e:
            logger.error(f"Error submitting order (attempt {attempt+1}): {e}")
            time.sleep(0.5)
    
    logger.error(f"Failed to submit order after {retry_attempts} attempts")
    return False

def count_executed_orders(trader, ticker, start_time):
    """Count the number of orders that have been executed"""
    try:
        executed_count = 0
        
        for order in trader.get_submitted_orders():
            if order.symbol == ticker and order.status == shift.Order.Status.FILLED:
                executed_count += 1
        
        return executed_count
    except Exception as e:
        logger.error(f"Error counting executed orders: {e}")
        return 0

def calculate_optimal_position_size(ticker, trader, signal_strength, market_data, max_position, current_position):
    """Calculate optimal position size based on signal and risk factors"""
    try:
        # Get prices
        bid_price, ask_price, mid_price, spread = get_current_prices(trader, ticker)
        if None in (bid_price, ask_price, mid_price, spread):
            return 0
            
        # Base size scaled by signal strength (0.0-1.0)
        base_size = abs(signal_strength) * 3  # Reduced max from 5 to 3 lots
        
        # Adjust for volatility (inverse relationship - more conservative in high vol)
        if market_data.volatility > 0:
            vol_factor = min(1.5, max(0.3, 0.01 / market_data.volatility))
        else:
            vol_factor = 1.0
            
        # Adjust for price level (fewer shares for higher priced stocks)
        price_factor = min(1.5, max(0.5, 100 / mid_price)) if mid_price > 0 else 1.0
        
        # Adjust for spread (smaller positions for wider spreads)
        spread_bps = (spread / mid_price) * 10000  # Spread in basis points
        spread_factor = min(1.2, max(0.5, 20 / spread_bps)) if spread_bps > 0 else 1.0
        
        # Adjust for current inventory in the same direction (more aggressive inventory management)
        inventory_factor = 1.0
        if signal_strength > 0 and current_position > 0:
            # Reduce buy size when already long
            inventory_factor = max(0.1, 1.0 - (current_position / max_position))
        elif signal_strength < 0 and current_position < 0:
            # Reduce sell size when already short
            inventory_factor = max(0.1, 1.0 - (abs(current_position) / max_position))
        
        # Calculate final size
        position_size = base_size * vol_factor * price_factor * spread_factor * inventory_factor
        
        # Round to integer and apply constraints
        position_size = int(round(position_size))
        position_size = min(position_size, max_position - abs(current_position))
        position_size = max(1, position_size)  # At least 1 lot
        
        # Check available capital
        max_size_by_value = int(MAX_POSITION_VALUE / (mid_price * 100))
        position_size = min(position_size, max_size_by_value)
        
        return position_size
    except Exception as e:
        logger.error(f"Error calculating position size for {ticker}: {e}")
        return 1  # Default to minimum size on error

def detect_order_book_imbalance(trader, ticker, depth=OBI_DEPTH):
    """Detect imbalance in the order book"""
    try:
        # Get order book
        bid_book = trader.get_order_book(ticker, shift.OrderBookType.GLOBAL_BID, depth)
        ask_book = trader.get_order_book(ticker, shift.OrderBookType.GLOBAL_ASK, depth)
        
        if not bid_book or not ask_book:
            return 0, 0
        
        # Calculate total sizes
        bid_size = sum([order.size for order in bid_book])
        ask_size = sum([order.size for order in ask_book])
        
        # Calculate imbalance
        total_size = bid_size + ask_size
        if total_size == 0:
            return 0, 0
        
        bid_ratio = bid_size / total_size
        ask_ratio = ask_size / total_size
        
        # Imbalance signal
        if bid_ratio > OBI_THRESHOLD:
            return 1, bid_ratio  # Buy signal
        elif ask_ratio > OBI_THRESHOLD:
            return -1, ask_ratio  # Sell signal
        
        return 0, max(bid_ratio, ask_ratio)  # No signal
    except Exception as e:
        logger.error(f"Error detecting order book imbalance for {ticker}: {e}")
        return 0, 0

def check_time_based_exits(trader, ticker, position_tracker, strategy_data, current_time):
    """Check for time-based position exits"""
    try:
        current_position, _ = get_current_position(trader, ticker)
        
        if current_position == 0:
            return 0, 0
        
        # Get position details
        position = position_tracker.get_position(ticker)
        if not position or 'entry_time' not in position:
            return 0, 0
        
        # Check if position held for too long
        hold_time = (current_time - position['entry_time']).total_seconds()
        
        if hold_time > MAX_POSITION_HOLD_TIME:
            # Time to exit position
            if current_position > 0:
                # Exit long position
                order = shift.Order(shift.Order.Type.MARKET_SELL, ticker, int(current_position))
                if submit_order(trader, order):
                    logger.info(f"Time-based exit for {ticker} LONG position, held for {hold_time:.0f} seconds")
                    strategy_data["time_based_orders"] += 1
                    strategy_data["time_based_positions"] += 1
                    position_tracker.clear_position(ticker)
                    return 1, 1
            else:
                # Exit short position
                order = shift.Order(shift.Order.Type.MARKET_BUY, ticker, int(abs(current_position)))
                if submit_order(trader, order):
                    logger.info(f"Time-based exit for {ticker} SHORT position, held for {hold_time:.0f} seconds")
                    strategy_data["time_based_orders"] += 1
                    strategy_data["time_based_positions"] += 1
                    position_tracker.clear_position(ticker)
                    return 1, 1
        
        return 0, 0
    except Exception as e:
        logger.error(f"Error checking time-based exits for {ticker}: {e}")
        return 0, 0

# ================= STRATEGY FUNCTIONS =================

def execute_mean_reversion_strategy(trader, ticker, market_data, position_tracker, strategy_data, current_time):
    """Execute mean reversion strategy"""
    try:
        # Check if we have enough data
        if len(market_data.prices) < MIN_DATA_POINTS:
            return 0, 0
            
        # Get current prices and position
        bid_price, ask_price, mid_price, spread = get_current_prices(trader, ticker)
        if None in (bid_price, ask_price, mid_price, spread):
            return 0, 0
            
        current_position, _ = get_current_position(trader, ticker)
        
        # Calculate mean reversion signal
        zscore = market_data.zscore
        
        # Adjust thresholds based on volatility
        vol_adj_entry = MR_ENTRY_THRESHOLD * (1 + market_data.volatility * 10)
        vol_adj_exit = MR_EXIT_THRESHOLD * (1 + market_data.volatility * 5)
        vol_adj_stop = MR_STOP_LOSS_THRESHOLD * (1 - market_data.volatility * 2)
        
        # No signal within thresholds
        if abs(zscore) < vol_adj_entry:
            # Check for exit conditions if we have a position
            if current_position != 0 and abs(zscore) < vol_adj_exit:
                # Exit position - we've reverted to mean
                if current_position > 0:
                    # Exit long with limit order to earn rebates
                    limit_price = round(bid_price * 0.998, 2)  # Slightly below bid
                    order = shift.Order(shift.Order.Type.LIMIT_SELL, ticker, int(current_position), limit_price)
                    if submit_order(trader, order):
                        logger.info(f"Mean Reversion EXIT LONG for {ticker} at {limit_price:.2f}, Z-score: {zscore:.2f}")
                        strategy_data["mr_orders"] += 1
                        strategy_data["mr_positions"] += 1
                        position_tracker.clear_position(ticker)
                        return 1, 1
                else:
                    # Exit short with limit order to earn rebates
                    limit_price = round(ask_price * 1.002, 2)  # Slightly above ask
                    order = shift.Order(shift.Order.Type.LIMIT_BUY, ticker, int(abs(current_position)), limit_price)
                    if submit_order(trader, order):
                        logger.info(f"Mean Reversion EXIT SHORT for {ticker} at {limit_price:.2f}, Z-score: {zscore:.2f}")
                        strategy_data["mr_orders"] += 1
                        strategy_data["mr_positions"] += 1
                        position_tracker.clear_position(ticker)
                        return 1, 1
            
            return 0, 0
        
        # Check stop loss - use tighter threshold in volatile markets
        if abs(zscore) > vol_adj_stop:
            if (zscore > 0 and current_position > 0) or (zscore < 0 and current_position < 0):
                # Stop loss triggered - we're going against our position
                if current_position > 0:
                    # Exit long position with market order (priority is risk management)
                    order = shift.Order(shift.Order.Type.MARKET_SELL, ticker, int(current_position))
                    if submit_order(trader, order):
                        logger.info(f"Mean Reversion STOP LOSS (LONG) for {ticker} at {bid_price:.2f}, Z-score: {zscore:.2f}")
                        strategy_data["mr_orders"] += 1
                        strategy_data["mr_positions"] += 1
                        position_tracker.clear_position(ticker)
                        return 1, 1
                else:
                    # Exit short position with market order
                    order = shift.Order(shift.Order.Type.MARKET_BUY, ticker, int(abs(current_position)))
                    if submit_order(trader, order):
                        logger.info(f"Mean Reversion STOP LOSS (SHORT) for {ticker} at {ask_price:.2f}, Z-score: {zscore:.2f}")
                        strategy_data["mr_orders"] += 1
                        strategy_data["mr_positions"] += 1
                        position_tracker.clear_position(ticker)
                        return 1, 1
        
        # Entry signals
        if zscore > vol_adj_entry:
            # Overpriced - go short (if not already short)
            if current_position >= 0:
                # Calculate position size
                signal_strength = min(1.0, (zscore - vol_adj_entry) / 3.0)
                position_size = calculate_optimal_position_size(
                    ticker, trader, -signal_strength, market_data, MAX_POSITION_SIZE, current_position)
                
                if position_size <= 0:
                    return 0, 0
                
                # Create sell order - use limit order for better entry and rebates
                limit_price = round(bid_price * 0.995, 2)  # Slightly more aggressive
                order = shift.Order(shift.Order.Type.LIMIT_SELL, ticker, position_size, limit_price)
                
                if submit_order(trader, order):
                    logger.info(f"Mean Reversion SHORT for {ticker} at {limit_price:.2f}, Z-score: {zscore:.2f}, Size: {position_size}")
                    strategy_data["mr_orders"] += 1
                    
                    # Market order as fallback in case limit doesn't fill quickly
                    market_order = shift.Order(shift.Order.Type.MARKET_SELL, ticker, position_size)
                    time.sleep(1.0)  # Give limit order a chance to fill
                    
                    # Check if position changed
                    new_position, _ = get_current_position(trader, ticker)
                    if new_position == current_position:  # Limit didn't fill
                        if submit_order(trader, market_order):
                            logger.info(f"Mean Reversion MARKET SHORT for {ticker} at {bid_price:.2f}, Z-score: {zscore:.2f}, Size: {position_size}")
                            strategy_data["mr_orders"] += 1
                    
                    strategy_data["mr_positions"] += 1
                    position_tracker.update_position(ticker, -position_size, bid_price, current_time, "mean_reversion")
                    return 2, 1  # Count both limit and potential market order
                    
        elif zscore < -vol_adj_entry:
            # Underpriced - go long (if not already long)
            if current_position <= 0:
                # Calculate position size
                signal_strength = min(1.0, (abs(zscore) - vol_adj_entry) / 3.0)
                position_size = calculate_optimal_position_size(
                    ticker, trader, signal_strength, market_data, MAX_POSITION_SIZE, current_position)
                
                if position_size <= 0:
                    return 0, 0
                
                # Create buy order - use limit order for better entry and rebates
                limit_price = round(ask_price * 1.005, 2)  # Slightly more aggressive
                order = shift.Order(shift.Order.Type.LIMIT_BUY, ticker, position_size, limit_price)
                
                if submit_order(trader, order):
                    logger.info(f"Mean Reversion LONG for {ticker} at {limit_price:.2f}, Z-score: {zscore:.2f}, Size: {position_size}")
                    strategy_data["mr_orders"] += 1
                    
                    # Market order as fallback in case limit doesn't fill quickly
                    market_order = shift.Order(shift.Order.Type.MARKET_BUY, ticker, position_size)
                    time.sleep(1.0)  # Give limit order a chance to fill
                    
                    # Check if position changed
                    new_position, _ = get_current_position(trader, ticker)
                    if new_position == current_position:  # Limit didn't fill
                        if submit_order(trader, market_order):
                            logger.info(f"Mean Reversion MARKET LONG for {ticker} at {ask_price:.2f}, Z-score: {zscore:.2f}, Size: {position_size}")
                            strategy_data["mr_orders"] += 1
                    
                    strategy_data["mr_positions"] += 1
                    position_tracker.update_position(ticker, position_size, ask_price, current_time, "mean_reversion")
                    return 2, 1  # Count both limit and potential market order
        
        return 0, 0
    except Exception as e:
        logger.error(f"Error in mean reversion strategy for {ticker}: {e}")
        traceback.print_exc()
        return 0, 0

def execute_liquidity_provision(trader, ticker, market_data, position_tracker, strategy_data, current_time):
    """Execute liquidity provision (market making) strategy"""
    try:
        # Check if we have enough data
        if market_data.mean_spread is None:
            return 0, 0
            
        # Get current prices and position
        bid_price, ask_price, mid_price, spread = get_current_prices(trader, ticker)
        if None in (bid_price, ask_price, mid_price, spread):
            return 0, 0
            
        current_position, _ = get_current_position(trader, ticker)
        
        # Calculate spreads and order sizes
        spread_bps = spread / mid_price * 10000  # Spread in basis points
        
        # Skip if spread is too narrow (not profitable)
        if spread_bps < LP_MIN_SPREAD_BPS:
            return 0, 0
            
        # Cap at maximum spread for safety
        if spread_bps > LP_MAX_SPREAD_BPS:
            spread_bps = LP_MAX_SPREAD_BPS
            
        # Adjust factor based on volatility - more aggressive in volatile markets
        vol_adjusted_factor = LP_SPREAD_FACTOR * (1 + market_data.volatility * 30)
        
        # Calculate limit prices with dynamic offset
        buy_offset = spread * vol_adjusted_factor
        sell_offset = spread * vol_adjusted_factor
        
        # More aggressive inventory management
        if current_position > 0:
            # Already long, increase sell offset and decrease buy offset
            position_ratio = min(1.0, current_position / MAX_POSITION_SIZE)
            buy_offset *= (1 + position_ratio * 1.5)  # Much less aggressive buying when long
            sell_offset *= (1 - position_ratio * 0.8)  # More aggressive selling when long
        elif current_position < 0:
            # Already short, increase buy offset and decrease sell offset
            position_ratio = min(1.0, abs(current_position) / MAX_POSITION_SIZE)
            buy_offset *= (1 - position_ratio * 0.8)  # More aggressive buying when short
            sell_offset *= (1 + position_ratio * 1.5)  # Much less aggressive selling when short
        
        # Consider market trend in pricing
        if market_data.trend > 0:  # Uptrend
            buy_offset *= 0.8  # More aggressive buying in uptrend
            sell_offset *= 1.2  # Less aggressive selling in uptrend
        elif market_data.trend < 0:  # Downtrend
            buy_offset *= 1.2  # Less aggressive buying in downtrend
            sell_offset *= 0.8  # More aggressive selling in downtrend
            
        # Calculate multiple price levels
        buy_prices = []
        sell_prices = []
        
        for i in range(LP_MAX_ORDERS_PER_SIDE):
            level_factor = 1.0 + i * 0.5  # Increase distance for deeper levels
            buy_prices.append(round(bid_price - (buy_offset * level_factor), 2))
            sell_prices.append(round(ask_price + (sell_offset * level_factor), 2))
        
        # Calculate order sizes (smaller for higher volatility and higher levels)
        base_size = max(1, min(3, int(4 * (1 - market_data.volatility * 50))))
        
        # Adjust for inventory
        buy_size = sell_size = base_size
        
        if current_position > MAX_POSITION_SIZE * 0.5:
            # Reduce buy size when position is already significantly long
            buy_size = max(1, buy_size - int(current_position / 2))
        elif current_position < -MAX_POSITION_SIZE * 0.5:
            # Reduce sell size when position is already significantly short
            sell_size = max(1, sell_size - int(abs(current_position) / 2))
            
        # Adjust sizes for capital constraints
        max_size_by_value = int(MAX_POSITION_VALUE / (mid_price * 100))
        buy_size = min(buy_size, max_size_by_value)
        sell_size = min(sell_size, max_size_by_value)
        
        # Place orders
        orders_placed = 0
        
        # Buy orders
        if current_position < MAX_POSITION_SIZE:
            for i, price in enumerate(buy_prices):
                level_size = max(1, buy_size - i)  # Reduce size for deeper levels
                if i == 0 or random.random() < 0.7:  # Place all level 1 orders, 70% chance for deeper levels
                    buy_order = shift.Order(shift.Order.Type.LIMIT_BUY, ticker, level_size, price)
                    if submit_order(trader, buy_order):
                        logger.info(f"LP LIMIT BUY for {ticker} at {price:.2f}, Size: {level_size}")
                        strategy_data["lp_orders"] += 1
                        orders_placed += 1
        
        # Sell orders
        if current_position > -MAX_POSITION_SIZE:
            for i, price in enumerate(sell_prices):
                level_size = max(1, sell_size - i)  # Reduce size for deeper levels
                if i == 0 or random.random() < 0.7:  # Place all level 1 orders, 70% chance for deeper levels
                    sell_order = shift.Order(shift.Order.Type.LIMIT_SELL, ticker, level_size, price)
                    if submit_order(trader, sell_order):
                        logger.info(f"LP LIMIT SELL for {ticker} at {price:.2f}, Size: {level_size}")
                        strategy_data["lp_orders"] += 1
                        orders_placed += 1
        
        return orders_placed, 0
    except Exception as e:
        logger.error(f"Error in liquidity provision strategy for {ticker}: {e}")
        traceback.print_exc()
        return 0, 0

def execute_order_book_imbalance_strategy(trader, ticker, market_data, position_tracker, strategy_data, current_time):
    """Execute order book imbalance strategy"""
    try:
        # Check if we have enough data
        if len(market_data.prices) < MIN_DATA_POINTS:
            return 0, 0
        
        # Get current prices and position
        bid_price, ask_price, mid_price, spread = get_current_prices(trader, ticker)
        if None in (bid_price, ask_price, mid_price, spread):
            return 0, 0
        
        current_position, _ = get_current_position(trader, ticker)
        
        # Get order book imbalance signal
        signal, strength = detect_order_book_imbalance(trader, ticker)
        if signal == 0:
            return 0, 0
        
        # Scale signal by strength
        signal_strength = abs(signal) * (strength - OBI_THRESHOLD) / (1 - OBI_THRESHOLD)
        
        # Calculate position size
        position_size = calculate_optimal_position_size(
            ticker, trader, signal_strength * signal, market_data, MAX_POSITION_SIZE, current_position)
        
        if position_size <= 0:
            return 0, 0
        
        # Execute order based on signal
        if signal > 0 and current_position < MAX_POSITION_SIZE:
            # Buy signal - use aggressive limit order to increase fill probability while earning rebates
            limit_price = round(ask_price * 0.9999, 2)  # Just below ask
            order = shift.Order(shift.Order.Type.LIMIT_BUY, ticker, position_size, limit_price)
            
            if submit_order(trader, order):
                logger.info(f"OBI BUY for {ticker} at {limit_price:.2f}, Imbalance: {strength:.2f}, Size: {position_size}")
                strategy_data["obi_orders"] += 1
                
                # Follow up with market if limit doesn't fill within 0.5 seconds
                time.sleep(0.5)
                new_position, _ = get_current_position(trader, ticker)
                if new_position == current_position:  # Limit didn't fill
                    market_order = shift.Order(shift.Order.Type.MARKET_BUY, ticker, position_size)
                    if submit_order(trader, market_order):
                        logger.info(f"OBI MARKET BUY for {ticker} at {ask_price:.2f}, Imbalance: {strength:.2f}, Size: {position_size}")
                        strategy_data["obi_orders"] += 1
                
                strategy_data["obi_positions"] += 1
                position_tracker.update_position(ticker, position_size, ask_price, current_time, "order_book_imbalance")
                return 2, 1
        
        elif signal < 0 and current_position > -MAX_POSITION_SIZE:
            # Sell signal - use aggressive limit order
            limit_price = round(bid_price * 1.0001, 2)  # Just above bid
            order = shift.Order(shift.Order.Type.LIMIT_SELL, ticker, position_size, limit_price)
            
            if submit_order(trader, order):
                logger.info(f"OBI SELL for {ticker} at {limit_price:.2f}, Imbalance: {strength:.2f}, Size: {position_size}")
                strategy_data["obi_orders"] += 1
                
                # Follow up with market if limit doesn't fill within 0.5 seconds
                time.sleep(0.5)
                new_position, _ = get_current_position(trader, ticker)
                if new_position == current_position:  # Limit didn't fill
                    market_order = shift.Order(shift.Order.Type.MARKET_SELL, ticker, position_size)
                    if submit_order(trader, market_order):
                        logger.info(f"OBI MARKET SELL for {ticker} at {bid_price:.2f}, Imbalance: {strength:.2f}, Size: {position_size}")
                        strategy_data["obi_orders"] += 1
                
                strategy_data["obi_positions"] += 1
                position_tracker.update_position(ticker, -position_size, bid_price, current_time, "order_book_imbalance")
                return 2, 1
        
        return 0, 0
    except Exception as e:
        logger.error(f"Error in order book imbalance strategy for {ticker}: {e}")
        traceback.print_exc()
        return 0, 0

def execute_rebate_harvesting_strategy(trader, ticker, market_data, strategy_data, current_time):
    """Execute rebate harvesting strategy - place limit orders far from market for rebates"""
    try:
        # Get current prices and position
        bid_price, ask_price, mid_price, spread = get_current_prices(trader, ticker)
        if None in (bid_price, ask_price, mid_price, spread):
            return 0, 0
            
        current_position, _ = get_current_position(trader, ticker)
        
        # Determine number of orders based on volatility and spread
        vol_factor = max(0.5, min(2.0, market_data.volatility * 50)) if market_data.volatility else 1.0
        spread_bps = spread / mid_price * 10000 if mid_price > 0 else 20
        
        max_orders = int(min(RH_MAX_ORDERS, max(RH_MIN_ORDERS, spread_bps / 5)))
        
        # Calculate order distances - wider during high volatility
        min_distance = RH_MIN_DISTANCE * vol_factor
        max_distance = RH_MAX_DISTANCE * vol_factor
        
        # Place orders
        orders_placed = 0
        
        # Buy orders - place fewer when we already have long position
        buy_orders = max_orders
        if current_position > 0:
            buy_orders = max(1, int(buy_orders * (1 - current_position / MAX_POSITION_SIZE)))
        
        for i in range(buy_orders):
            # Calculate distance factor - increases with each order
            distance_factor = min_distance + (max_distance - min_distance) * (i / max(1, buy_orders - 1))
            limit_price = round(bid_price * (1 - distance_factor), 2)
            
            # Place small order
            order = shift.Order(shift.Order.Type.LIMIT_BUY, ticker, RH_ORDER_SIZE, limit_price)
            if submit_order(trader, order):
                logger.info(f"Rebate Harvesting BUY for {ticker} at {limit_price:.2f}, Distance: {distance_factor*100:.1f}%")
                strategy_data["rh_orders"] += 1
                orders_placed += 1
        
        # Sell orders - place fewer when we already have short position
        sell_orders = max_orders
        if current_position < 0:
            sell_orders = max(1, int(sell_orders * (1 - abs(current_position) / MAX_POSITION_SIZE)))
        
        for i in range(sell_orders):
            # Calculate distance factor - increases with each order
            distance_factor = min_distance + (max_distance - min_distance) * (i / max(1, sell_orders - 1))
            limit_price = round(ask_price * (1 + distance_factor), 2)
            
            # Place small order
            order = shift.Order(shift.Order.Type.LIMIT_SELL, ticker, RH_ORDER_SIZE, limit_price)
            if submit_order(trader, order):
                logger.info(f"Rebate Harvesting SELL for {ticker} at {limit_price:.2f}, Distance: {distance_factor*100:.1f}%")
                strategy_data["rh_orders"] += 1
                orders_placed += 1
        
        return orders_placed, 0
    except Exception as e:
        logger.error(f"Error in rebate harvesting strategy for {ticker}: {e}")
        traceback.print_exc()
        return 0, 0

def ensure_minimum_requirements(trader, ticker, shared_data, remaining_minutes, strategy_data):
    """Ensure we meet minimum requirements for orders and positions"""
    try:
        # Get counts from shared data
        with shared_data["lock"]:
            total_orders = shared_data[f"{ticker}_total_orders"]
            total_positions = shared_data[f"{ticker}_total_positions"]
        
        # Calculate how aggressive we should be based on time remaining
        # More aggressive as we get closer to deadline
        urgency = max(0, min(1, (180 - remaining_minutes) / 180))
        
        # Ensure minimum positions
        positions_needed = max(0, MIN_POSITIONS_TARGET - total_positions)
        if positions_needed > 0 and (urgency > 0.7 or random.random() < urgency * 0.3):
            # Get current position
            current_position, _ = get_current_position(trader, ticker)
            
            # Determine direction (opposite of current position to reduce risk)
            is_buy = current_position <= 0
            
            # Execute a small market order
            position_size = 1
            if is_buy:
                order = shift.Order(shift.Order.Type.MARKET_BUY, ticker, position_size)
                if submit_order(trader, order):
                    logger.info(f"Minimum requirement: Market buy for {ticker}, Size: {position_size}")
                    strategy_data["min_req_orders"] += 1
                    strategy_data["min_req_positions"] += 1
                    
                    with shared_data["lock"]:
                        shared_data[f"{ticker}_total_orders"] += 1
                        shared_data[f"{ticker}_total_positions"] += 1
                    
                    return 1, 1
            else:
                order = shift.Order(shift.Order.Type.MARKET_SELL, ticker, position_size)
                if submit_order(trader, order):
                    logger.info(f"Minimum requirement: Market sell for {ticker}, Size: {position_size}")
                    strategy_data["min_req_orders"] += 1
                    strategy_data["min_req_positions"] += 1
                    
                    with shared_data["lock"]:
                        shared_data[f"{ticker}_total_orders"] += 1
                        shared_data[f"{ticker}_total_positions"] += 1
                    
                    return 1, 1
        
        # Ensure minimum orders
        orders_needed = max(0, MIN_ORDERS_TARGET - total_orders)
        if orders_needed > 0 and (urgency > 0.5 or random.random() < urgency * 0.2):
            # Get current prices
            bid_price, ask_price, mid_price, spread = get_current_prices(trader, ticker)
            if None in (bid_price, ask_price, mid_price, spread):
                return 0, 0
            
            # Place aggressive limit orders with small size
            orders_to_place = min(5, max(1, int(orders_needed * urgency / 10)))
            orders_placed = 0
            
            for i in range(orders_to_place):
                is_buy = (i % 2 == 0)  # Alternate buy/sell
                position_size = 1
                
                if is_buy:
                    # Place buy slightly below ask (more likely to execute)
                    limit_price = round(ask_price * 0.999, 2)
                    order = shift.Order(shift.Order.Type.LIMIT_BUY, ticker, position_size, limit_price)
                    if submit_order(trader, order):
                        logger.info(f"Minimum requirement: Limit buy for {ticker} at {limit_price:.2f}, Size: {position_size}")
                        strategy_data["min_req_orders"] += 1
                        orders_placed += 1
                        
                        with shared_data["lock"]:
                            shared_data[f"{ticker}_total_orders"] += 1
                else:
                    # Place sell slightly above bid
                    limit_price = round(bid_price * 1.001, 2)
                    order = shift.Order(shift.Order.Type.LIMIT_SELL, ticker, position_size, limit_price)
                    if submit_order(trader, order):
                        logger.info(f"Minimum requirement: Limit sell for {ticker} at {limit_price:.2f}, Size: {position_size}")
                        strategy_data["min_req_orders"] += 1
                        orders_placed += 1
                        
                        with shared_data["lock"]:
                            shared_data[f"{ticker}_total_orders"] += 1
            
            return orders_placed, 0
        
        return 0, 0
    except Exception as e:
        logger.error(f"Error ensuring minimum requirements for {ticker}: {e}")
        traceback.print_exc()
        return 0, 0

def check_risk_limits(trader, ticker, market_data, strategy_data, position_tracker):
    """Check if risk limits are exceeded and reduce positions if needed"""
    try:
        # Get current position and value
        current_position, _ = get_current_position(trader, ticker)
        position_value = calculate_position_value(trader, ticker)
        unrealized_pnl = calculate_unrealized_pnl(trader, ticker)
        
        # Get buying power
        total_bp = trader.get_portfolio_summary().get_total_bp()
        
        # Check for maximum drawdown
        if unrealized_pnl < -MAX_DRAWDOWN_PCT * total_bp:
            # Drawdown exceeds threshold - reduce position
            logger.warning(f"Maximum drawdown exceeded for {ticker}. Reducing position.")
            
            # Reduce position by half
            reduction_size = max(1, int(abs(current_position) / 2))
            
            if current_position > 0:
                # Reduce long position
                order = shift.Order(shift.Order.Type.MARKET_SELL, ticker, reduction_size)
                if submit_order(trader, order):
                    logger.info(f"Risk management: Reducing long position for {ticker}, Size: {reduction_size}")
                    strategy_data["risk_orders"] += 1
                    strategy_data["risk_positions"] += 1
                    
                    # Update position tracker
                    position = position_tracker.get_position(ticker)
                    if position and 'size' in position:
                        position_tracker.update_position(
                            ticker, 
                            position.get('size', 0) - reduction_size,
                            position.get('entry_price', 0),
                            position.get('entry_time', datetime.now()),
                            position.get('strategy', 'risk_management')
                        )
                    
                    return True
            elif current_position < 0:
                # Reduce short position
                order = shift.Order(shift.Order.Type.MARKET_BUY, ticker, reduction_size)
                if submit_order(trader, order):
                    logger.info(f"Risk management: Reducing short position for {ticker}, Size: {reduction_size}")
                    strategy_data["risk_orders"] += 1
                    strategy_data["risk_positions"] += 1
                    
                    # Update position tracker
                    position = position_tracker.get_position(ticker)
                    if position and 'size' in position:
                        position_tracker.update_position(
                            ticker, 
                            position.get('size', 0) + reduction_size,
                            position.get('entry_price', 0),
                            position.get('entry_time', datetime.now()),
                            position.get('strategy', 'risk_management')
                        )
                    
                    return True
                    
        # Check for excessive position size
        if abs(current_position) > MAX_POSITION_SIZE:
            # Position too large - reduce to maximum
            reduction_size = abs(current_position) - MAX_POSITION_SIZE
            
            if current_position > 0:
                # Reduce long position
                order = shift.Order(shift.Order.Type.MARKET_SELL, ticker, reduction_size)
                if submit_order(trader, order):
                    logger.info(f"Risk management: Reducing oversized long position for {ticker}, Size: {reduction_size}")
                    strategy_data["risk_orders"] += 1
                    strategy_data["risk_positions"] += 1
                    return True
            elif current_position < 0:
                # Reduce short position
                order = shift.Order(shift.Order.Type.MARKET_BUY, ticker, reduction_size)
                if submit_order(trader, order):
                    logger.info(f"Risk management: Reducing oversized short position for {ticker}, Size: {reduction_size}")
                    strategy_data["risk_orders"] += 1
                    strategy_data["risk_positions"] += 1
                    return True
                    
        # Check for excessive position value
        if position_value > MAX_POSITION_VALUE:
            # Position value too high - reduce
            reduction_factor = MAX_POSITION_VALUE / position_value
            reduction_size = max(1, int(abs(current_position) * (1 - reduction_factor)))
            
            if current_position > 0:
                # Reduce long position
                order = shift.Order(shift.Order.Type.MARKET_SELL, ticker, reduction_size)
                if submit_order(trader, order):
                    logger.info(f"Risk management: Reducing high-value long position for {ticker}, Size: {reduction_size}")
                    strategy_data["risk_orders"] += 1
                    strategy_data["risk_positions"] += 1
                    return True
            elif current_position < 0:
                # Reduce short position
                order = shift.Order(shift.Order.Type.MARKET_BUY, ticker, reduction_size)
                if submit_order(trader, order):
                    logger.info(f"Risk management: Reducing high-value short position for {ticker}, Size: {reduction_size}")
                    strategy_data["risk_orders"] += 1
                    strategy_data["risk_positions"] += 1
                    return True
                    
        # Check max exposure
        if position_value > total_bp * MAX_EXPOSURE_PCT:
            # Exposure too high - reduce position
            reduction_factor = (total_bp * MAX_EXPOSURE_PCT) / position_value
            reduction_size = max(1, int(abs(current_position) * (1 - reduction_factor)))
            
            if current_position > 0:
                # Reduce long position
                order = shift.Order(shift.Order.Type.MARKET_SELL, ticker, reduction_size)
                if submit_order(trader, order):
                    logger.info(f"Risk management: Reducing high-exposure long position for {ticker}, Size: {reduction_size}")
                    strategy_data["risk_orders"] += 1
                    strategy_data["risk_positions"] += 1
                    return True
            elif current_position < 0:
                # Reduce short position
                order = shift.Order(shift.Order.Type.MARKET_BUY, ticker, reduction_size)
                if submit_order(trader, order):
                    logger.info(f"Risk management: Reducing high-exposure short position for {ticker}, Size: {reduction_size}")
                    strategy_data["risk_orders"] += 1
                    strategy_data["risk_positions"] += 1
                    return True
                    
        return False
    except Exception as e:
        logger.error(f"Error checking risk limits for {ticker}: {e}")
        traceback.print_exc()
        return False

def update_strategy_weights(shared_data, ticker):
    """Update strategy weights based on performance"""
    try:
        # Get strategy performance
        strategy_data = shared_data.get(f"{ticker}_strategy_data", {})
        
        # Calculate P&L by strategy
        mr_pnl = strategy_data.get("mr_pnl", 0)
        lp_pnl = strategy_data.get("lp_pnl", 0)
        obi_pnl = strategy_data.get("obi_pnl", 0)
        rh_pnl = strategy_data.get("rh_pnl", 0)
        
        # Calculate total P&L
        total_pnl = mr_pnl + lp_pnl + obi_pnl + rh_pnl
        
        # If no significant P&L yet, don't adjust
        if abs(total_pnl) < 10:
            return STRATEGY_WEIGHTS
            
        # Calculate new weights based on performance
        new_weights = {}
        
        # Start with base weights
        new_weights["mean_reversion"] = STRATEGY_WEIGHTS["mean_reversion"]
        new_weights["liquidity_provision"] = STRATEGY_WEIGHTS["liquidity_provision"]
        new_weights["order_book_imbalance"] = STRATEGY_WEIGHTS["order_book_imbalance"]
        new_weights["rebate_harvesting"] = STRATEGY_WEIGHTS["rebate_harvesting"]
        
        # Adjust weights based on P&L contribution
        if total_pnl != 0:
            # For profitable strategies, increase weight; for unprofitable, decrease
            for strategy, pnl in [("mean_reversion", mr_pnl),
                                 ("liquidity_provision", lp_pnl),
                                 ("order_book_imbalance", obi_pnl),
                                 ("rebate_harvesting", rh_pnl)]:
                
                if pnl > 0:
                    # Increase weight for profitable strategies
                    new_weights[strategy] *= (1 + (pnl / max(1, abs(total_pnl))) * 0.5)
                else:
                    # Decrease weight for unprofitable strategies
                    new_weights[strategy] *= (1 - min(0.5, (abs(pnl) / max(1, abs(total_pnl))) * 0.5))
        
        # Normalize weights to sum to 1.0
        total_weight = sum(new_weights.values())
        if total_weight > 0:
            new_weights = {k: v / total_weight for k, v in new_weights.items()}
            
        # Enforce minimum weights
        for strategy in new_weights:
            new_weights[strategy] = max(0.05, min(0.7, new_weights[strategy]))
            
        # Re-normalize after enforcing minimums
        total_weight = sum(new_weights.values())
        new_weights = {k: v / total_weight for k, v in new_weights.items()}
            
        logger.info(f"Updated strategy weights for {ticker}: {new_weights}")
        return new_weights
    except Exception as e:
        logger.error(f"Error updating strategy weights for {ticker}: {e}")
        traceback.print_exc()
        return STRATEGY_WEIGHTS

# ================= MAIN STRATEGY FUNCTION =================

def ticker_strategy(trader, ticker, end_time, shared_data):
    """Main strategy function for a single ticker"""
    logger.info(f"Starting advanced trading strategy for {ticker}")
    
    # Initialize market data tracker
    market_data = MarketData(ticker)
    
    # Initialize position tracker
    position_tracker = PositionTracker()
    
    # Initialize strategy data
    strategy_data = {
        "mr_orders": 0,        # Mean reversion orders
        "mr_positions": 0,      # Mean reversion positions
        "mr_pnl": 0,            # Mean reversion P&L
        
        "lp_orders": 0,         # Liquidity provision orders
        "lp_positions": 0,      # Liquidity provision positions
        "lp_pnl": 0,            # Liquidity provision P&L
        
        "obi_orders": 0,        # Order book imbalance orders
        "obi_positions": 0,     # Order book imbalance positions
        "obi_pnl": 0,           # Order book imbalance P&L
        
        "rh_orders": 0,         # Rebate harvesting orders
        "rh_positions": 0,      # Rebate harvesting positions
        "rh_pnl": 0,            # Rebate harvesting P&L
        
        "risk_orders": 0,       # Risk management orders
        "risk_positions": 0,    # Risk management positions
        
        "time_based_orders": 0, # Time-based exit orders
        "time_based_positions": 0, # Time-based exit positions
        
        "min_req_orders": 0,    # Minimum requirement orders
        "min_req_positions": 0  # Minimum requirement positions
    }
    
    # Initialize timing variables
    start_time = trader.get_last_trade_time()
    last_stats_update = start_time
    last_mr_check = start_time
    last_lp_check = start_time
    last_obi_check = start_time
    last_rh_check = start_time
    last_risk_check = start_time
    last_time_check = start_time
    last_order_refresh = start_time
    last_strategy_update = start_time
    
    # Initialize strategy weights
    strategy_weights = STRATEGY_WEIGHTS.copy()
    
    # Initialize P&L tracking
    initial_pl = trader.get_portfolio_item(ticker).get_realized_pl()
    last_pl_check = initial_pl
    last_pl_time = start_time
    
    # Track whether we've initialized the shared data
    initialized_shared_data = False
    
    # Main trading loop
    while trader.get_last_trade_time() < end_time:
        try:
            current_time = trader.get_last_trade_time()
            
            # Calculate time remaining
            minutes_remaining = (end_time - current_time).total_seconds() / 60
            
            # Get current market data
            bid_price, ask_price, mid_price, spread = get_current_prices(trader, ticker)
            if None in (bid_price, ask_price, mid_price, spread):
                time.sleep(CHECK_INTERVAL)
                continue
                
            # Update market data tracker
            market_data.update(current_time, bid_price, ask_price, mid_price, spread)
            
            # Initialize shared data if needed
            if not initialized_shared_data:
                with shared_data["lock"]:
                    shared_data[f"{ticker}_total_orders"] = 0
                    shared_data[f"{ticker}_total_positions"] = 0
                    shared_data[f"{ticker}_strategy_data"] = strategy_data
                initialized_shared_data = True
            
            # Update statistics periodically
            if (current_time - last_stats_update).total_seconds() > STATS_UPDATE_INTERVAL:
                # Update market stats
                market_data.update_statistics()
                
                # Update strategy performance
                current_pl = trader.get_portfolio_item(ticker).get_realized_pl()
                pl_change = current_pl - last_pl_check
                time_elapsed = (current_time - last_pl_time).total_seconds()
                
                if time_elapsed > 0:
                    # Attribute P&L based on orders executed by each strategy
                    total_orders = sum([
                        strategy_data["mr_orders"],
                        strategy_data["lp_orders"],
                        strategy_data["obi_orders"],
                        strategy_data["rh_orders"],
                        strategy_data["risk_orders"],
                        strategy_data["time_based_orders"],
                        strategy_data["min_req_orders"]
                    ])
                    
                    if total_orders > 0:
                        # Attribute proportionally to orders
                        strategy_data["mr_pnl"] += pl_change * (strategy_data["mr_orders"] / total_orders)
                        strategy_data["lp_pnl"] += pl_change * (strategy_data["lp_orders"] / total_orders)
                        strategy_data["obi_pnl"] += pl_change * (strategy_data["obi_orders"] / total_orders)
                        strategy_data["rh_pnl"] += pl_change * (strategy_data["rh_orders"] / total_orders)
                    
                    # Reset for next period
                    last_pl_check = current_pl
                    last_pl_time = current_time
                    
                    # Update shared data
                    with shared_data["lock"]:
                        shared_data[f"{ticker}_strategy_data"] = strategy_data
                
                last_stats_update = current_time
            
            # Refresh orders periodically - more frequently for better liquidity provision
            if (current_time - last_order_refresh).total_seconds() > LP_ORDER_REFRESH_SECONDS:
                cancel_orders(trader, ticker)
                last_order_refresh = current_time
            
            # Execute mean reversion strategy
            if (current_time - last_mr_check).total_seconds() > 10:
                if random.random() < strategy_weights["mean_reversion"]:
                    mr_orders, mr_positions = execute_mean_reversion_strategy(
                        trader, ticker, market_data, position_tracker, strategy_data, current_time)
                    
                    # Update counters
                    with shared_data["lock"]:
                        shared_data[f"{ticker}_total_orders"] += mr_orders
                        shared_data[f"{ticker}_total_positions"] += mr_positions
                    
                last_mr_check = current_time
            
            # Execute liquidity provision strategy
            if (current_time - last_lp_check).total_seconds() > 15:
                if random.random() < strategy_weights["liquidity_provision"]:
                    lp_orders, lp_positions = execute_liquidity_provision(
                        trader, ticker, market_data, position_tracker, strategy_data, current_time)
                    
                    # Update counters
                    with shared_data["lock"]:
                        shared_data[f"{ticker}_total_orders"] += lp_orders
                        shared_data[f"{ticker}_total_positions"] += lp_positions
                    
                last_lp_check = current_time
            
            # Execute order book imbalance strategy
            if (current_time - last_obi_check).total_seconds() > OBI_CHECK_INTERVAL:
                if random.random() < strategy_weights["order_book_imbalance"]:
                    obi_orders, obi_positions = execute_order_book_imbalance_strategy(
                        trader, ticker, market_data, position_tracker, strategy_data, current_time)
                    
                    # Update counters
                    with shared_data["lock"]:
                        shared_data[f"{ticker}_total_orders"] += obi_orders
                        shared_data[f"{ticker}_total_positions"] += obi_positions
                    
                last_obi_check = current_time
            
            # Execute rebate harvesting strategy
            if (current_time - last_rh_check).total_seconds() > 45:  # Less frequent - focus on quality
                if random.random() < strategy_weights["rebate_harvesting"]:
                    rh_orders, rh_positions = execute_rebate_harvesting_strategy(
                        trader, ticker, market_data, strategy_data, current_time)
                    
                    # Update counters
                    with shared_data["lock"]:
                        shared_data[f"{ticker}_total_orders"] += rh_orders
                        shared_data[f"{ticker}_total_positions"] += rh_positions
                    
                last_rh_check = current_time
            
            # Check risk limits
            if (current_time - last_risk_check).total_seconds() > POSITION_CHECK_INTERVAL:
                risk_action = check_risk_limits(trader, ticker, market_data, strategy_data, position_tracker)
                
                # Check time-based exits
                time_orders, time_positions = check_time_based_exits(
                    trader, ticker, position_tracker, strategy_data, current_time)
                
                # Update counters
                with shared_data["lock"]:
                    shared_data[f"{ticker}_total_orders"] += time_orders
                    shared_data[f"{ticker}_total_positions"] += time_positions
                
                # Ensure minimum requirements
                min_orders, min_positions = ensure_minimum_requirements(
                    trader, ticker, shared_data, minutes_remaining, strategy_data)
                
                # Update counters
                with shared_data["lock"]:
                    shared_data[f"{ticker}_total_orders"] += min_orders
                    shared_data[f"{ticker}_total_positions"] += min_positions
                
                # Log current status
                current_position, _ = get_current_position(trader, ticker)
                current_pl = trader.get_portfolio_item(ticker).get_realized_pl() - initial_pl
                
                with shared_data["lock"]:
                    total_orders = shared_data[f"{ticker}_total_orders"]
                    total_positions = shared_data[f"{ticker}_total_positions"]
                
                logger.info(f"{ticker} Status: Position={current_position}, "
                           f"Orders={total_orders}, Positions={total_positions}, "
                           f"P&L=${current_pl:.2f}, Z-score={market_data.zscore:.2f}, "
                           f"Vol={market_data.volatility:.4f}")
                
                last_risk_check = current_time
            
            # Update strategy weights based on performance
            if (current_time - last_strategy_update).total_seconds() > STRATEGY_UPDATE_INTERVAL:
                strategy_weights = update_strategy_weights(shared_data, ticker)
                last_strategy_update = current_time
            
            # Brief pause
            time.sleep(CHECK_INTERVAL)
            
        except Exception as e:
            logger.error(f"Error in main loop for {ticker}: {e}")
            traceback.print_exc()
            time.sleep(1)
    
    # End of day cleanup
    try:
        logger.info(f"End of day cleanup for {ticker}")
        
        # Cancel remaining orders
        cancel_orders(trader, ticker)
        
        # Close positions
        close_positions(trader, ticker)
        
        # Calculate final P&L
        final_pl = trader.get_portfolio_item(ticker).get_realized_pl() - initial_pl
        
        with shared_data["lock"]:
            total_orders = shared_data[f"{ticker}_total_orders"]
            total_positions = shared_data[f"{ticker}_total_positions"]
        
        logger.info(f"Strategy complete for {ticker}: "
                   f"Orders={total_orders}, Positions={total_positions}, P&L=${final_pl:.2f}")
        
        with shared_data["lock"]:
            shared_data[f"{ticker}_final_pl"] = final_pl
            shared_data[f"{ticker}_final_orders"] = total_orders
            shared_data[f"{ticker}_final_positions"] = total_positions
            shared_data[f"{ticker}_completed"] = True
        
        return True
    except Exception as e:
        logger.error(f"Error during cleanup for {ticker}: {e}")
        traceback.print_exc()
        
        with shared_data["lock"]:
            shared_data[f"{ticker}_completed"] = True
            shared_data[f"{ticker}_error"] = str(e)
        
        return False

def main(trader):
    """Main function to run the strategy"""
    logger.info("Starting advanced ZI-optimized trading strategy")
    
    # Tickers to trade
    active_tickers = TICKER_LIST
    
    # Determine trading hours
    current = trader.get_last_trade_time()
    start_time = current
    
    # End time - 5 minutes before market close
    current_date = current.date()
    end_time = datetime(current_date.year, current_date.month, current_date.day, 15, 55, 0)
    
    logger.info(f"Trading strategy starting at {start_time}, will run until {end_time}")
    
    # Track initial portfolio values
    initial_total_pl = trader.get_portfolio_summary().get_total_realized_pl()
    initial_bp = trader.get_portfolio_summary().get_total_bp()
    
    logger.info(f"Initial buying power: ${initial_bp:.2f}")
    logger.info(f"Trading the following symbols: {active_tickers}")
    
    # Shared data between threads
    shared_data = {
        "lock": Lock()
    }
    
    # Launch trading threads
    threads = []
    for ticker in active_tickers:
        thread = Thread(target=ticker_strategy, args=(trader, ticker, end_time, shared_data))
        thread.daemon = True
        threads.append(thread)
        thread.start()
        time.sleep(0.5)  # Stagger thread starts
    
    # Monitor progress while threads are running
    while any(thread.is_alive() for thread in threads):
        try:
            # Calculate current metrics
            current_bp = trader.get_portfolio_summary().get_total_bp()
            current_pl = trader.get_portfolio_summary().get_total_realized_pl() - initial_total_pl
            
            total_orders = 0
            total_positions = 0
            
            with shared_data["lock"]:
                for ticker in active_tickers:
                    total_orders += shared_data.get(f"{ticker}_total_orders", 0)
                    total_positions += shared_data.get(f"{ticker}_total_positions", 0)
            
            # Print status update
            logger.info(f"Status: P&L=${current_pl:.2f}, "
                       f"Orders={total_orders}, Positions={total_positions}, "
                       f"BP=${current_bp:.2f}")
            
            # Check if all threads have completed
            all_completed = True
            with shared_data["lock"]:
                for ticker in active_tickers:
                    if not shared_data.get(f"{ticker}_completed", False):
                        all_completed = False
                        break
            
            if all_completed:
                logger.info("All trading threads have completed")
                break
            
            time.sleep(60)  # Update every minute
            
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt, stopping threads...")
            break
        except Exception as e:
            logger.error(f"Error in monitoring loop: {e}")
            time.sleep(60)
    
    # Wait for threads to complete (with timeout)
    for thread in threads:
        thread.join(timeout=30)
    
    # Final cleanup for any remaining positions
    for ticker in active_tickers:
        try:
            cancel_orders(trader, ticker)
            close_positions(trader, ticker)
        except Exception as e:
            logger.error(f"Error during final cleanup for {ticker}: {e}")
    
    # Calculate final results
    try:
        final_bp = trader.get_portfolio_summary().get_total_bp()
        final_pl = trader.get_portfolio_summary().get_total_realized_pl() - initial_total_pl
        
        # Gather per-ticker results
        ticker_results = {}
        with shared_data["lock"]:
            for ticker in active_tickers:
                ticker_results[ticker] = {
                    "pl": shared_data.get(f"{ticker}_final_pl", 0),
                    "orders": shared_data.get(f"{ticker}_final_orders", 0),
                    "positions": shared_data.get(f"{ticker}_final_positions", 0)
                }
        
        # Print final results
        logger.info("\n===== STRATEGY COMPLETE =====")
        logger.info(f"Final buying power: ${final_bp:.2f}")
        logger.info(f"Total profit/loss: ${final_pl:.2f}")
        logger.info(f"Return on capital: {(final_pl / initial_bp * 100):.2f}%")
        
        # Print ticker-specific results
        logger.info("\n===== TICKER RESULTS =====")
        for ticker, results in ticker_results.items():
            logger.info(f"{ticker}: P&L=${results['pl']:.2f}, "
                       f"Orders={results['orders']}, Positions={results['positions']}")
        
        return final_pl
    except Exception as e:
        logger.error(f"Error calculating final results: {e}")
        return 0

if __name__ == '__main__':
    with shift.Trader("dolla-dolla-bills-yall") as trader:
        try:
            # Connect to the server
            logger.info("Connecting to SHIFT...")
            trader.connect("initiator.cfg", "Zxz7Qxa9")
            time.sleep(1)
            
            # Subscribe to order book data
            logger.info("Subscribing to order book data...")
            trader.sub_all_order_book()
            time.sleep(1)
            
            # Run main strategy
            logger.info("Starting main strategy...")
            main(trader)
            
        except Exception as e:
            logger.error(f"Fatal error in main program: {e}")
            try:
                # Emergency cleanup
                logger.info("Performing emergency cleanup...")
                for ticker in TICKER_LIST:
                    cancel_orders(trader, ticker)
                    close_positions(trader, ticker)
            except Exception as cleanup_error:
                logger.error(f"Failed to perform emergency cleanup: {cleanup_error}")