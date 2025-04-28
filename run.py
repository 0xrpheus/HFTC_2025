import shift
import time
import numpy as np
import pandas as pd
import datetime as dt
from datetime import datetime, timedelta
import logging
import math
import random
from threading import Thread, Lock
import statistics
import traceback
import collections
import os
import sys

# ================ CONFIGURATION PARAMETERS ================

# Set up logging
LOG_LEVEL = logging.INFO
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
LOG_FILE = 'rl_strategy.log'

# Tickers to trade - in RL environment we'll have 2 stocks
TICKERS = ["CS1", "CS2"]

# Minimum requirements
MIN_ORDERS = 200       # Minimum 200 orders per day requirement
MIN_POSITIONS = 10     # Minimum 10 positions per day requirement
TARGET_ORDERS = 250    # Target slightly above minimum for safety
TARGET_POSITIONS = 15  # Target slightly above minimum for safety

# Capital allocation
MAX_CAPITAL_PER_TICKER = 0.4  # Maximum % of capital to allocate to a single ticker
MAX_POSITION_SIZE = 5         # Maximum position size in lots (1 lot = 100 shares)

# Trading thresholds
ENTRY_THRESHOLD = 0.002    # 0.2% price move threshold for entry
EXIT_THRESHOLD = 0.004     # 0.4% price move threshold for exit
STOP_LOSS_THRESHOLD = 0.01 # 1% stop loss threshold

# Time parameters
OBSERVATION_PERIOD = 180   # 3 minutes observation at start (in seconds) - reduced from 15 minutes
ORDER_REFRESH_SECONDS = 30 # How often to refresh orders
CHECK_INTERVAL = 0.2       # Main loop check interval in seconds
MAX_HOLDING_TIME = 300     # Maximum position holding time (5 minutes)
STRATEGY_UPDATE_INTERVAL = 300  # How often to update strategy (5 minutes)

# Strategy weights - will dynamically adjust based on performance
STRATEGY_WEIGHTS = {
    "adaptive_market_making": 0.40,  # Primary strategy
    "momentum_surfing": 0.25,        # Secondary strategy
    "pattern_recognition": 0.20,     # Supporting strategy
    "agent_manipulation": 0.15       # Experimental strategy
}

# Order book parameters
BOOK_DEPTH = 5            # Depth of order book to analyze
IMBALANCE_THRESHOLD = 0.7 # Order book imbalance threshold for signals

# Technical indicators
LOOKBACK_SHORT = 20       # Short lookback period for indicators
LOOKBACK_MEDIUM = 50      # Medium lookback period for indicators
LOOKBACK_LONG = 100       # Long lookback period for indicators
ZSCORE_WINDOW = 50        # Window for z-score calculation
VOLATILITY_WINDOW = 20    # Window for volatility calculation

# Risk management
MAX_DRAWDOWN_PCT = 0.03   # Maximum drawdown allowed (3%)
POSITION_SIZING_FACTOR = 0.7  # Position sizing conservatism factor

# Market regime detection
TRENDING_THRESHOLD = 0.6  # Threshold for trending market detection
MEAN_REVERTING_THRESHOLD = 0.6  # Threshold for mean-reverting market detection

# ================ INITIALIZATION ================

# Set up logging
def setup_logging():
    logging.basicConfig(
        level=LOG_LEVEL,
        format=LOG_FORMAT,
        handlers=[
            logging.FileHandler(LOG_FILE),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("RL_Strategy")

# Price and statistics tracker
class MarketData:
    """Class to track market data and calculate statistics"""
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
        
        # Order book data
        self.bid_depths = collections.deque(maxlen=max_window)
        self.ask_depths = collections.deque(maxlen=max_window)
        self.book_imbalances = collections.deque(maxlen=max_window)
        
        # Statistics
        self.volatility = 0.01
        self.mean_price = None
        self.std_price = None
        self.zscore = 0
        self.mean_spread = None
        self.trend = 0
        self.market_regime = "unknown"  # "trending", "mean_reverting", "random", "unknown"
        
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
        
        # Mean spread
        self.mean_spread = np.mean(list(self.spreads))
        
        # Volatility (standard deviation of returns)
        if len(self.returns) >= VOLATILITY_WINDOW:
            self.volatility = np.std(list(self.returns)[-VOLATILITY_WINDOW:])
            # Annualize volatility for context
            self.volatility = self.volatility * np.sqrt(252 * 390)
        else:
            self.volatility = 0.01
        
        # Detect trend
        if len(self.prices) >= LOOKBACK_MEDIUM:
            short_ma = np.mean(list(self.prices)[-LOOKBACK_SHORT:])
            medium_ma = np.mean(list(self.prices)[-LOOKBACK_MEDIUM:])
            
            if short_ma > medium_ma * 1.002:
                self.trend = 1  # Uptrend
            elif short_ma < medium_ma * 0.998:
                self.trend = -1  # Downtrend
            else:
                self.trend = 0  # No clear trend
                
        # Detect market regime
        self.detect_market_regime()
    
    def detect_market_regime(self):
        """Detect current market regime (trending, mean-reverting, or random)"""
        if len(self.returns) < LOOKBACK_MEDIUM:
            self.market_regime = "unknown"
            return
        
        # Calculate autocorrelation of returns
        returns_list = list(self.returns)[-LOOKBACK_MEDIUM:]
        
        try:
            # Return autocorrelation (lag 1)
            returns_lag_1 = returns_list[:-1]
            returns_t = returns_list[1:]
            
            if len(returns_lag_1) > 1 and np.std(returns_lag_1) > 0 and np.std(returns_t) > 0:
                autocorr = np.corrcoef(returns_lag_1, returns_t)[0, 1]
                
                if autocorr > TRENDING_THRESHOLD:
                    self.market_regime = "trending"
                elif autocorr < -MEAN_REVERTING_THRESHOLD:
                    self.market_regime = "mean_reverting"
                else:
                    self.market_regime = "random"
            else:
                self.market_regime = "unknown"
        except:
            self.market_regime = "unknown"
    
    def update_book_data(self, bid_depth, ask_depth, imbalance):
        """Update order book data"""
        self.bid_depths.append(bid_depth)
        self.ask_depths.append(ask_depth)
        self.book_imbalances.append(imbalance)
    
    def get_book_imbalance_signal(self):
        """Get signal from order book imbalance"""
        if len(self.book_imbalances) < 3:
            return 0
            
        # Get recent imbalances
        recent_imbalances = list(self.book_imbalances)[-3:]
        avg_imbalance = sum(recent_imbalances) / len(recent_imbalances)
        
        if avg_imbalance > IMBALANCE_THRESHOLD:
            return 1  # Buy signal
        elif avg_imbalance < (1 - IMBALANCE_THRESHOLD):
            return -1  # Sell signal
        return 0  # No signal
    
    def get_momentum_signal(self):
        """Get momentum signal from recent price action"""
        if len(self.prices) < LOOKBACK_SHORT:
            return 0
            
        # Calculate short-term momentum
        short_return = list(self.prices)[-1] / list(self.prices)[-LOOKBACK_SHORT] - 1
        
        # Scale by volatility
        if self.volatility > 0:
            vol_adjusted_return = short_return / self.volatility
        else:
            vol_adjusted_return = short_return / 0.01
            
        if vol_adjusted_return > 1.5:  # Strong upward momentum
            return 1
        elif vol_adjusted_return < -1.5:  # Strong downward momentum
            return -1
        return 0  # No signal
    
    def get_mean_reversion_signal(self):
        """Get mean reversion signal based on z-score"""
        if abs(self.zscore) < 2:
            return 0
            
        if self.zscore > 2:  # Price significantly above mean
            return -1  # Sell signal
        elif self.zscore < -2:  # Price significantly below mean
            return 1  # Buy signal
        return 0  # No signal
    
    def get_adaptive_signal(self):
        """Get adaptive signal based on detected market regime"""
        if self.market_regime == "trending":
            return self.get_momentum_signal()
        elif self.market_regime == "mean_reverting":
            return self.get_mean_reversion_signal()
        else:
            book_signal = self.get_book_imbalance_signal()
            if book_signal != 0:
                return book_signal
            return 0

# Position tracker
class PositionTracker:
    """Class to track open positions and their details"""
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

# Strategy tracker
class StrategyPerformance:
    """Class to track strategy performance metrics"""
    def __init__(self):
        self.strategies = {
            "adaptive_market_making": {"pnl": 0, "positions": 0, "success": 0},
            "momentum_surfing": {"pnl": 0, "positions": 0, "success": 0},
            "pattern_recognition": {"pnl": 0, "positions": 0, "success": 0},
            "agent_manipulation": {"pnl": 0, "positions": 0, "success": 0}
        }
        self.lock = Lock()
    
    def record_trade(self, strategy, pnl, success=True):
        with self.lock:
            if strategy in self.strategies:
                self.strategies[strategy]["pnl"] += pnl
                self.strategies[strategy]["positions"] += 1
                if success:
                    self.strategies[strategy]["success"] += 1
    
    def get_success_rate(self, strategy):
        with self.lock:
            if strategy in self.strategies and self.strategies[strategy]["positions"] > 0:
                return self.strategies[strategy]["success"] / self.strategies[strategy]["positions"]
            return 0
    
    def get_adjusted_weights(self, base_weights):
        with self.lock:
            adj_weights = base_weights.copy()
            
            # Calculate total absolute PnL
            total_pnl = sum(abs(s["pnl"]) for s in self.strategies.values())
            
            if total_pnl > 0:
                # Adjust weights based on performance
                for strategy in adj_weights:
                    if self.strategies[strategy]["positions"] > 0:
                        # Success rate adjustment
                        success_rate = self.get_success_rate(strategy)
                        
                        # PnL adjustment
                        pnl_contrib = self.strategies[strategy]["pnl"] / total_pnl if total_pnl > 0 else 0
                        
                        # Combined adjustment factor
                        adj_factor = 1 + (success_rate * 0.5 + pnl_contrib * 0.5)
                        
                        # Apply adjustment (limit to reasonable range)
                        adj_weights[strategy] *= min(2, max(0.5, adj_factor))
            
            # Normalize weights to sum to 1.0
            total_weight = sum(adj_weights.values())
            if total_weight > 0:
                adj_weights = {k: v/total_weight for k, v in adj_weights.items()}
            
            return adj_weights

# ================ UTILITY FUNCTIONS ================

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

def get_order_book_data(trader, ticker, depth=BOOK_DEPTH):
    """Get order book data and calculate imbalance"""
    try:
        # Get order book
        bid_book = trader.get_order_book(ticker, shift.OrderBookType.GLOBAL_BID, depth)
        ask_book = trader.get_order_book(ticker, shift.OrderBookType.GLOBAL_ASK, depth)
        
        if not bid_book or not ask_book:
            return 0, 0, 0.5  # Default to neutral
        
        # Calculate total sizes
        bid_size = sum([order.size for order in bid_book])
        ask_size = sum([order.size for order in ask_book])
        
        # Calculate imbalance
        total_size = bid_size + ask_size
        if total_size == 0:
            return 0, 0, 0.5
        
        # Imbalance ratio (0-1 scale)
        imbalance = bid_size / total_size if total_size > 0 else 0.5
        
        return bid_size, ask_size, imbalance
    except Exception as e:
        logger.error(f"Error getting order book data for {ticker}: {e}")
        return 0, 0, 0.5

def cancel_orders(trader, ticker):
    """Cancel all waiting orders for a specific ticker"""
    try:
        cancelled_count = 0
        for order in trader.get_waiting_list():
            if order.symbol == ticker:
                trader.submit_cancellation(order)
                cancelled_count += 1
                time.sleep(0.01)  # Small sleep to avoid overwhelming the system
        
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
            time.sleep(0.2)
    
    logger.error(f"Failed to submit order after {retry_attempts} attempts")
    return False

def calculate_optimal_position_size(trader, ticker, signal_strength, market_data, max_position, current_position):
    """Calculate optimal position size based on multiple factors"""
    try:
        # Get prices
        bid_price, ask_price, mid_price, spread = get_current_prices(trader, ticker)
        if None in (bid_price, ask_price, mid_price, spread):
            return 1  # Default to minimum size on error
            
        # Base size scaled by signal strength (0.0-1.0)
        signal_strength = abs(signal_strength)
        base_size = max_position * signal_strength * POSITION_SIZING_FACTOR
        
        # Adjust for volatility (inverse relationship - more conservative in high vol)
        if market_data.volatility > 0:
            vol_factor = min(1.5, max(0.3, 0.01 / market_data.volatility))
        else:
            vol_factor = 1.0
            
        # Adjust for spread (smaller positions for wider spreads)
        spread_factor = 1.0
        if mid_price > 0:
            spread_bps = (spread / mid_price) * 10000  # Spread in basis points
            spread_factor = min(1.2, max(0.5, 20 / spread_bps)) if spread_bps > 0 else 1.0
            
        # Adjust for current inventory in the same direction
        inventory_factor = 1.0
        if signal_strength > 0 and current_position > 0:
            # Reduce buy size when already long
            inventory_factor = max(0.1, 1.0 - (current_position / max_position))
        elif signal_strength < 0 and current_position < 0:
            # Reduce sell size when already short
            inventory_factor = max(0.1, 1.0 - (abs(current_position) / max_position))
        
        # Calculate final size
        position_size = base_size * vol_factor * spread_factor * inventory_factor
        
        # Round to integer and apply constraints
        position_size = int(round(position_size))
        position_size = min(position_size, max_position - abs(current_position))
        position_size = max(1, position_size)  # At least 1 lot
        
        # Check trading capital
        capital = trader.get_portfolio_summary().get_total_bp()
        max_size_by_capital = int((capital * MAX_CAPITAL_PER_TICKER) / (mid_price * 100))
        position_size = min(position_size, max_size_by_capital)
        
        return position_size
    except Exception as e:
        logger.error(f"Error calculating position size for {ticker}: {e}")
        return 1  # Default to minimum size on error

def check_risk_limits(trader, ticker, market_data, position_tracker):
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
                    position_tracker.clear_position(ticker)
                    return True
            elif current_position < 0:
                # Reduce short position
                order = shift.Order(shift.Order.Type.MARKET_BUY, ticker, reduction_size)
                if submit_order(trader, order):
                    logger.info(f"Risk management: Reducing short position for {ticker}, Size: {reduction_size}")
                    position_tracker.clear_position(ticker)
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
                    return True
            elif current_position < 0:
                # Reduce short position
                order = shift.Order(shift.Order.Type.MARKET_BUY, ticker, reduction_size)
                if submit_order(trader, order):
                    logger.info(f"Risk management: Reducing oversized short position for {ticker}, Size: {reduction_size}")
                    return True
                    
        # Check max exposure
        position_percentage = position_value / total_bp if total_bp > 0 else 0
        if position_percentage > MAX_CAPITAL_PER_TICKER:
            # Exposure too high - reduce position
            reduction_factor = MAX_CAPITAL_PER_TICKER / position_percentage
            reduction_size = max(1, int(abs(current_position) * (1 - reduction_factor)))
            
            if current_position > 0:
                # Reduce long position
                order = shift.Order(shift.Order.Type.MARKET_SELL, ticker, reduction_size)
                if submit_order(trader, order):
                    logger.info(f"Risk management: Reducing high-exposure long position for {ticker}, Size: {reduction_size}")
                    return True
            elif current_position < 0:
                # Reduce short position
                order = shift.Order(shift.Order.Type.MARKET_BUY, ticker, reduction_size)
                if submit_order(trader, order):
                    logger.info(f"Risk management: Reducing high-exposure short position for {ticker}, Size: {reduction_size}")
                    return True
                    
        return False
    except Exception as e:
        logger.error(f"Error checking risk limits for {ticker}: {e}")
        return False

def check_time_based_exits(trader, ticker, position_tracker, current_time):
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
        
        if hold_time > MAX_HOLDING_TIME:
            # Time to exit position
            if current_position > 0:
                # Exit long position
                order = shift.Order(shift.Order.Type.MARKET_SELL, ticker, int(current_position))
                if submit_order(trader, order):
                    logger.info(f"Time-based exit for {ticker} LONG position, held for {hold_time:.0f} seconds")
                    position_tracker.clear_position(ticker)
                    return 1, 1
            else:
                # Exit short position
                order = shift.Order(shift.Order.Type.MARKET_BUY, ticker, int(abs(current_position)))
                if submit_order(trader, order):
                    logger.info(f"Time-based exit for {ticker} SHORT position, held for {hold_time:.0f} seconds")
                    position_tracker.clear_position(ticker)
                    return 1, 1
        
        return 0, 0
    except Exception as e:
        logger.error(f"Error checking time-based exits for {ticker}: {e}")
        return 0, 0

def ensure_minimum_requirements(trader, ticker, shared_data, remaining_minutes):
    """Ensure we meet minimum requirements for orders and positions"""
    try:
        # Get counts from shared data
        with shared_data["lock"]:
            total_orders = shared_data.get(f"{ticker}_total_orders", 0)
            total_positions = shared_data.get(f"{ticker}_total_positions", 0)
        
        # Calculate how aggressive we should be based on time remaining
        # More aggressive as we get closer to deadline
        urgency = max(0, min(1, (180 - remaining_minutes) / 180))
        
        # Ensure minimum positions
        positions_needed = max(0, MIN_POSITIONS - total_positions)
        orders_placed = 0
        positions_opened = 0
        
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
                    orders_placed += 1
                    positions_opened += 1
                    
                    with shared_data["lock"]:
                        shared_data[f"{ticker}_total_orders"] = total_orders + 1
                        shared_data[f"{ticker}_total_positions"] = total_positions + 1
            else:
                order = shift.Order(shift.Order.Type.MARKET_SELL, ticker, position_size)
                if submit_order(trader, order):
                    logger.info(f"Minimum requirement: Market sell for {ticker}, Size: {position_size}")
                    orders_placed += 1
                    positions_opened += 1
                    
                    with shared_data["lock"]:
                        shared_data[f"{ticker}_total_orders"] = total_orders + 1
                        shared_data[f"{ticker}_total_positions"] = total_positions + 1
        
        # Ensure minimum orders
        orders_needed = max(0, MIN_ORDERS - total_orders)
        if orders_needed > 0 and (urgency > 0.5 or random.random() < urgency * 0.2):
            # Get current prices
            bid_price, ask_price, mid_price, spread = get_current_prices(trader, ticker)
            if None in (bid_price, ask_price, mid_price, spread):
                return orders_placed, positions_opened
            
            # Place aggressive limit orders with small size
            orders_to_place = min(5, max(1, int(orders_needed * urgency / 10)))
            
            for i in range(orders_to_place):
                is_buy = (i % 2 == 0)  # Alternate buy/sell
                position_size = 1
                
                if is_buy:
                    # Place buy slightly below ask (more likely to execute)
                    limit_price = round(ask_price * 0.999, 2)
                    order = shift.Order(shift.Order.Type.LIMIT_BUY, ticker, position_size, limit_price)
                    if submit_order(trader, order):
                        logger.info(f"Minimum requirement: Limit buy for {ticker} at {limit_price:.2f}, Size: {position_size}")
                        orders_placed += 1
                        
                        with shared_data["lock"]:
                            shared_data[f"{ticker}_total_orders"] = total_orders + orders_placed
                else:
                    # Place sell slightly above bid
                    limit_price = round(bid_price * 1.001, 2)
                    order = shift.Order(shift.Order.Type.LIMIT_SELL, ticker, position_size, limit_price)
                    if submit_order(trader, order):
                        logger.info(f"Minimum requirement: Limit sell for {ticker} at {limit_price:.2f}, Size: {position_size}")
                        orders_placed += 1
                        
                        with shared_data["lock"]:
                            shared_data[f"{ticker}_total_orders"] = total_orders + orders_placed
        
        return orders_placed, positions_opened
    except Exception as e:
        logger.error(f"Error ensuring minimum requirements for {ticker}: {e}")
        return 0, 0

# ================ STRATEGY FUNCTIONS ================

def execute_adaptive_market_making(trader, ticker, market_data, position_tracker, current_time, max_position):
    """Execute adaptive market making strategy that adjusts based on market regime"""
    try:
        # Check if we have enough data
        if len(market_data.prices) < 5:
            return 0, 0
            
        # Get current prices and position
        bid_price, ask_price, mid_price, spread = get_current_prices(trader, ticker)
        if None in (bid_price, ask_price, mid_price, spread):
            return 0, 0
            
        current_position, _ = get_current_position(trader, ticker)
        
        # Check current market regime
        regime = market_data.market_regime
        
        # Adjust spread factors based on market regime
        if regime == "trending":
            # In trending markets, be more aggressive on the trend side
            bid_offset_factor = 0.3 if market_data.trend > 0 else 0.7
            ask_offset_factor = 0.7 if market_data.trend > 0 else 0.3
        elif regime == "mean_reverting":
            # In mean reverting markets, be more aggressive against recent moves
            # If price is high (positive z-score), be more aggressive on the sell side
            bid_offset_factor = 0.7 if market_data.zscore > 0 else 0.3
            ask_offset_factor = 0.3 if market_data.zscore > 0 else 0.7
        else:
            # In random/unknown markets, use default balanced approach
            bid_offset_factor = 0.5
            ask_offset_factor = 0.5
        
        # Adjust for volatility - wider spreads in high volatility
        vol_adjustment = max(1.0, min(3.0, market_data.volatility * 100))
        
        # Calculate spreads for limit orders
        bid_offset = spread * bid_offset_factor * vol_adjustment
        ask_offset = spread * ask_offset_factor * vol_adjustment
        
        # Adjust for inventory
        inventory_adjustment = current_position / (max_position * 2) if max_position > 0 else 0
        bid_offset *= (1 + inventory_adjustment)  # Increase bid offset (less aggressive buying) when long
        ask_offset *= (1 - inventory_adjustment)  # Decrease ask offset (more aggressive selling) when long
        
        # Calculate limit prices
        limit_buy_price = round(bid_price - bid_offset, 2)
        limit_sell_price = round(ask_price + ask_offset, 2)
        
        # Calculate order sizes
        base_size = 1
        
        # Scale size based on confidence in market regime
        if len(market_data.prices) > LOOKBACK_MEDIUM:
            base_size = 2
        
        # Reduce size when we already have inventory in that direction
        buy_size = max(1, base_size - max(0, int(current_position)))
        sell_size = max(1, base_size - max(0, int(-current_position)))
        
        # Place orders - use multiple small orders rather than single large ones
        orders_placed = 0
        positions_opened = 0
        
        # Only submit orders if there's room for inventory
        if current_position < max_position:
            order = shift.Order(shift.Order.Type.LIMIT_BUY, ticker, buy_size, limit_buy_price)
            if submit_order(trader, order):
                logger.info(f"AMM: Limit BUY for {ticker} at {limit_buy_price:.2f}, Size: {buy_size}, Regime: {regime}")
                orders_placed += 1
        
        if current_position > -max_position:
            order = shift.Order(shift.Order.Type.LIMIT_SELL, ticker, sell_size, limit_sell_price)
            if submit_order(trader, order):
                logger.info(f"AMM: Limit SELL for {ticker} at {limit_sell_price:.2f}, Size: {sell_size}, Regime: {regime}")
                orders_placed += 1
        
        return orders_placed, positions_opened
    except Exception as e:
        logger.error(f"Error in adaptive market making for {ticker}: {e}")
        traceback.print_exc()
        return 0, 0

def execute_momentum_surfing(trader, ticker, market_data, position_tracker, current_time, max_position):
    """Execute momentum surfing strategy - catch and ride trending movements"""
    try:
        # Check if we have enough data
        if len(market_data.prices) < LOOKBACK_SHORT:
            return 0, 0
            
        # Get current prices and position
        bid_price, ask_price, mid_price, spread = get_current_prices(trader, ticker)
        if None in (bid_price, ask_price, mid_price, spread):
            return 0, 0
            
        current_position, _ = get_current_position(trader, ticker)
        
        # Check for trend
        if market_data.trend == 0:
            return 0, 0
            
        # Check for momentum in the trend direction
        momentum_signal = market_data.get_momentum_signal()
        
        # Only act on momentum in line with trend
        if momentum_signal * market_data.trend <= 0:
            return 0, 0
            
        # Calculate position size
        signal_strength = abs(momentum_signal) * (0.5 + abs(market_data.trend) * 0.5)
        position_size = calculate_optimal_position_size(
            trader, ticker, signal_strength, market_data, max_position, current_position)
            
        if position_size <= 0:
            return 0, 0
            
        orders_placed = 0
        positions_opened = 0
        
        # Execute the trade - use market orders for momentum to ensure execution
        if momentum_signal > 0 and current_position < max_position:
            # Buy signal
            order = shift.Order(shift.Order.Type.MARKET_BUY, ticker, position_size)
            if submit_order(trader, order):
                logger.info(f"Momentum SURF: Market BUY for {ticker} at {ask_price:.2f}, Size: {position_size}")
                orders_placed += 1
                positions_opened += 1
                
                # Update position tracker
                position_tracker.update_position(
                    ticker, current_position + position_size, ask_price, current_time, "momentum_surfing")
                
        elif momentum_signal < 0 and current_position > -max_position:
            # Sell signal
            order = shift.Order(shift.Order.Type.MARKET_SELL, ticker, position_size)
            if submit_order(trader, order):
                logger.info(f"Momentum SURF: Market SELL for {ticker} at {bid_price:.2f}, Size: {position_size}")
                orders_placed += 1
                positions_opened += 1
                
                # Update position tracker
                position_tracker.update_position(
                    ticker, current_position - position_size, bid_price, current_time, "momentum_surfing")
        
        return orders_placed, positions_opened
    except Exception as e:
        logger.error(f"Error in momentum surfing for {ticker}: {e}")
        traceback.print_exc()
        return 0, 0

def execute_pattern_recognition(trader, ticker, market_data, position_tracker, current_time, max_position):
    """Execute pattern recognition strategy - look for specific technical patterns"""
    try:
        # Check if we have enough data
        if len(market_data.prices) < LOOKBACK_MEDIUM:
            return 0, 0
            
        # Get current prices and position
        bid_price, ask_price, mid_price, spread = get_current_prices(trader, ticker)
        if None in (bid_price, ask_price, mid_price, spread):
            return 0, 0
            
        current_position, _ = get_current_position(trader, ticker)
        
        # Pattern 1: Mean reversion in oversold/overbought conditions
        mean_reversion_signal = 0
        
        # Check for extreme z-scores with confirming RSI
        if market_data.zscore > 2 and market_data.trend < 0:
            mean_reversion_signal = -1  # Sell signal
        elif market_data.zscore < -2 and market_data.trend > 0:
            mean_reversion_signal = 1  # Buy signal
            
        # Pattern 2: Order book imbalance
        imbalance_signal = market_data.get_book_imbalance_signal()
        
        # Pattern 3: Volatility breakout
        volatility_breakout = False
        if len(market_data.returns) >= VOLATILITY_WINDOW:
            current_vol = np.std(list(market_data.returns)[-5:])
            historical_vol = np.std(list(market_data.returns)[-VOLATILITY_WINDOW:-5])
            
            if current_vol > historical_vol * 2:
                volatility_breakout = True
                
        # Combine signals - prioritize based on market regime
        trading_signal = 0
        
        if market_data.market_regime == "mean_reverting" and mean_reversion_signal != 0:
            trading_signal = mean_reversion_signal
        elif volatility_breakout and market_data.trend != 0:
            trading_signal = market_data.trend  # Trade in direction of trend on volatility breakout
        elif imbalance_signal != 0:
            trading_signal = imbalance_signal
            
        if trading_signal == 0:
            return 0, 0
            
        # Calculate position size
        signal_strength = abs(trading_signal) * 0.7  # Be conservative with pattern recognition
        position_size = calculate_optimal_position_size(
            trader, ticker, signal_strength, market_data, max_position, current_position)
            
        if position_size <= 0:
            return 0, 0
            
        orders_placed = 0
        positions_opened = 0
        
        # Execute the trade - use limit orders close to market for better execution
        if trading_signal > 0 and current_position < max_position:
            # Buy signal
            limit_price = round(ask_price * 1.001, 2)  # Slightly above ask
            order = shift.Order(shift.Order.Type.LIMIT_BUY, ticker, position_size, limit_price)
            if submit_order(trader, order):
                logger.info(f"Pattern Recognition: Limit BUY for {ticker} at {limit_price:.2f}, Size: {position_size}")
                orders_placed += 1
                positions_opened += 1
                
                # Update position tracker
                position_tracker.update_position(
                    ticker, current_position + position_size, ask_price, current_time, "pattern_recognition")
                
        elif trading_signal < 0 and current_position > -max_position:
            # Sell signal
            limit_price = round(bid_price * 0.999, 2)  # Slightly below bid
            order = shift.Order(shift.Order.Type.LIMIT_SELL, ticker, position_size, limit_price)
            if submit_order(trader, order):
                logger.info(f"Pattern Recognition: Limit SELL for {ticker} at {limit_price:.2f}, Size: {position_size}")
                orders_placed += 1
                positions_opened += 1
                
                # Update position tracker
                position_tracker.update_position(
                    ticker, current_position - position_size, bid_price, current_time, "pattern_recognition")
        
        return orders_placed, positions_opened
    except Exception as e:
        logger.error(f"Error in pattern recognition for {ticker}: {e}")
        traceback.print_exc()
        return 0, 0

def execute_agent_manipulation(trader, ticker, market_data, position_tracker, current_time, max_position):
    """Execute agent manipulation strategy - attempt to influence RL agent behavior"""
    try:
        # Check if we have enough data
        if len(market_data.prices) < 10:
            return 0, 0
            
        # Get current prices and position
        bid_price, ask_price, mid_price, spread = get_current_prices(trader, ticker)
        if None in (bid_price, ask_price, mid_price, spread):
            return 0, 0
            
        current_position, _ = get_current_position(trader, ticker)
        
        # This strategy attempts to influence the RL agents by creating patterns they might learn from
        # It's more experimental and relies on understanding how RL agents typically behave
        
        orders_placed = 0
        positions_opened = 0
        
        # Technique 1: Create false order book pressure
        # Place and quickly cancel orders to create impression of buying/selling pressure
        
        # Decide on direction based on current position (try to move market favorable to our position)
        manipulation_direction = -1 if current_position > 0 else 1
        
        # Place a series of orders on one side of the book
        order_count = random.randint(2, 4)
        order_size = 1
        
        for i in range(order_count):
            if manipulation_direction > 0 and current_position < max_position:
                # Create buying pressure
                limit_price = round(bid_price * (1 + 0.001 * (i+1)), 2)
                order = shift.Order(shift.Order.Type.LIMIT_BUY, ticker, order_size, limit_price)
                if submit_order(trader, order):
                    orders_placed += 1
            elif manipulation_direction < 0 and current_position > -max_position:
                # Create selling pressure
                limit_price = round(ask_price * (1 - 0.001 * (i+1)), 2)
                order = shift.Order(shift.Order.Type.LIMIT_SELL, ticker, order_size, limit_price)
                if submit_order(trader, order):
                    orders_placed += 1
        
        # Technique 2: Small probing orders to test agent reactions
        if random.random() < 0.3:  # Only do this occasionally
            if current_position < max_position:
                # Small buy
                order = shift.Order(shift.Order.Type.MARKET_BUY, ticker, 1)
                if submit_order(trader, order):
                    logger.info(f"Agent Manipulation: Probing BUY for {ticker}")
                    orders_placed += 1
                    positions_opened += 1
                    
                    # Update position tracker
                    position_tracker.update_position(
                        ticker, current_position + 1, ask_price, current_time, "agent_manipulation")
            elif current_position > -max_position:
                # Small sell
                order = shift.Order(shift.Order.Type.MARKET_SELL, ticker, 1)
                if submit_order(trader, order):
                    logger.info(f"Agent Manipulation: Probing SELL for {ticker}")
                    orders_placed += 1
                    positions_opened += 1
                    
                    # Update position tracker
                    position_tracker.update_position(
                        ticker, current_position - 1, bid_price, current_time, "agent_manipulation")
        
        return orders_placed, positions_opened
    except Exception as e:
        logger.error(f"Error in agent manipulation for {ticker}: {e}")
        traceback.print_exc()
        return 0, 0

# ================ MAIN STRATEGY FUNCTION ================

def run_strategy(trader, ticker, end_time, shared_data):
    """Main strategy function for each ticker"""
    logger.info(f"Starting RL-optimized strategy for {ticker}")
    
    # Initialize tracking objects
    market_data = MarketData(ticker)
    position_tracker = PositionTracker()
    strategy_performance = StrategyPerformance()
    
    # Trading state
    observation_phase = True
    initial_pl = trader.get_portfolio_item(ticker).get_realized_pl()
    last_pl_check = initial_pl
    last_pl_check_time = trader.get_last_trade_time()
    
    # Initialize timing variables
    start_time = trader.get_last_trade_time()
    last_order_refresh = start_time
    last_amm_check = start_time
    last_momentum_check = start_time
    last_pattern_check = start_time
    last_manipulation_check = start_time
    last_risk_check = start_time
    last_strategy_update = start_time
    
    # Order and position counters
    total_orders = 0
    total_positions = 0
    
    # Initialize strategy weights
    strategy_weights = STRATEGY_WEIGHTS.copy()
    
    # IMPORTANT: Test for date comparison problems
    logger.info(f"Start time: {start_time}, End time: {end_time}")
    if start_time > end_time:
        logger.error(f"ERROR: Start time ({start_time}) is after end time ({end_time})!")
        # Fix end time if needed
        current_date = start_time.date()
        corrected_end_time = datetime(current_date.year, current_date.month, current_date.day, 15, 55, 0)
        
        if start_time.hour >= 15 and start_time.minute >= 55:
            # Too late in the day already, use tomorrow
            corrected_end_time += timedelta(days=1)
            
        logger.info(f"Correcting end time to: {corrected_end_time}")
        end_time = corrected_end_time
    
    # Main trading loop - add additional safety checks
    while True:
        try:
            current_time = trader.get_last_trade_time()
            
            # Safety check for loop termination
            if current_time >= end_time:
                logger.info(f"Reached end time. Exiting trading loop for {ticker}")
                break
                
            # Safety check for disconnection
            if not trader.is_connected():
                logger.error(f"Lost connection to SHIFT server for {ticker}. Attempting to reconnect...")
                # Attempt reconnection or exit loop
                break
                
            elapsed_seconds = (current_time - start_time).total_seconds()
            minutes_remaining = (end_time - current_time).total_seconds() / 60
            
            # In observation phase, just collect data and don't trade
            # This gives time for the RL agents to begin learning
            if observation_phase:
                if elapsed_seconds < OBSERVATION_PERIOD:
                    # Get current market data
                    bid_price, ask_price, mid_price, spread = get_current_prices(trader, ticker)
                    if None not in (bid_price, ask_price, mid_price, spread):
                        # Update market data
                        market_data.update(current_time, bid_price, ask_price, mid_price, spread)
                        
                        # Get order book data
                        bid_depth, ask_depth, imbalance = get_order_book_data(trader, ticker)
                        market_data.update_book_data(bid_depth, ask_depth, imbalance)
                        
                        # Log market state during observation
                        if elapsed_seconds % 60 < CHECK_INTERVAL:
                            logger.info(f"Observation phase for {ticker}: elapsed={elapsed_seconds:.0f}s, " +
                                       f"price={mid_price:.2f}, spread={spread:.2f}, " +
                                       f"volatility={market_data.volatility:.4f}")
                    
                    time.sleep(CHECK_INTERVAL)
                    continue
                else:
                    # End observation phase
                    observation_phase = False
                    logger.info(f"Ending observation phase for {ticker}. Starting active trading.")
            
            # Get current market data
            bid_price, ask_price, mid_price, spread = get_current_prices(trader, ticker)
            if None in (bid_price, ask_price, mid_price, spread):
                time.sleep(CHECK_INTERVAL)
                continue
                
            # Update market data
            market_data.update(current_time, bid_price, ask_price, mid_price, spread)
            
            # Get order book data
            bid_depth, ask_depth, imbalance = get_order_book_data(trader, ticker)
            market_data.update_book_data(bid_depth, ask_depth, imbalance)
            
            # Refresh orders periodically
            if (current_time - last_order_refresh).total_seconds() > ORDER_REFRESH_SECONDS:
                cancel_orders(trader, ticker)
                last_order_refresh = current_time
            
            # Check risk limits and manage positions
            if (current_time - last_risk_check).total_seconds() > 5:  # Check frequently
                check_risk_limits(trader, ticker, market_data, position_tracker)
                
                # Check for time-based exits
                time_orders, time_positions = check_time_based_exits(
                    trader, ticker, position_tracker, current_time)
                
                total_orders += time_orders
                total_positions += time_positions
                
                # Update P&L attribution
                current_pl = trader.get_portfolio_item(ticker).get_realized_pl()
                pl_change = current_pl - last_pl_check
                
                if pl_change != 0:
                    # Attribute P&L to the last used strategy
                    position = position_tracker.get_position(ticker)
                    if position and 'strategy' in position:
                        strategy_name = position['strategy']
                        success = pl_change > 0
                        strategy_performance.record_trade(strategy_name, pl_change, success)
                
                last_pl_check = current_pl
                last_pl_check_time = current_time
                last_risk_check = current_time
            
            # Execute strategies based on weights and timing
            
            # Adaptive Market Making
            if (current_time - last_amm_check).total_seconds() > 5:
                if random.random() < strategy_weights["adaptive_market_making"]:
                    amm_orders, amm_positions = execute_adaptive_market_making(
                        trader, ticker, market_data, position_tracker, current_time, MAX_POSITION_SIZE)
                    
                    total_orders += amm_orders
                    total_positions += amm_positions
                
                last_amm_check = current_time
            
            # Momentum Surfing
            if (current_time - last_momentum_check).total_seconds() > 10:
                if random.random() < strategy_weights["momentum_surfing"]:
                    mom_orders, mom_positions = execute_momentum_surfing(
                        trader, ticker, market_data, position_tracker, current_time, MAX_POSITION_SIZE)
                    
                    total_orders += mom_orders
                    total_positions += mom_positions
                
                last_momentum_check = current_time
            
            # Pattern Recognition
            if (current_time - last_pattern_check).total_seconds() > 15:
                if random.random() < strategy_weights["pattern_recognition"]:
                    pat_orders, pat_positions = execute_pattern_recognition(
                        trader, ticker, market_data, position_tracker, current_time, MAX_POSITION_SIZE)
                    
                    total_orders += pat_orders
                    total_positions += pat_positions
                
                last_pattern_check = current_time
            
            # Agent Manipulation
            if (current_time - last_manipulation_check).total_seconds() > 20:
                if random.random() < strategy_weights["agent_manipulation"]:
                    manip_orders, manip_positions = execute_agent_manipulation(
                        trader, ticker, market_data, position_tracker, current_time, MAX_POSITION_SIZE)
                    
                    total_orders += manip_orders
                    total_positions += manip_positions
                
                last_manipulation_check = current_time
            
            # Ensure minimum requirements - start early to be safe
            # For Day 5, ensure we get 200 orders ASAP 
            min_orders, min_positions = ensure_minimum_requirements(
                trader, ticker, shared_data, minutes_remaining)
            
            total_orders += min_orders
            total_positions += min_positions
            
            # Update strategy weights based on performance
            if (current_time - last_strategy_update).total_seconds() > STRATEGY_UPDATE_INTERVAL:
                strategy_weights = strategy_performance.get_adjusted_weights(STRATEGY_WEIGHTS)
                
                logger.info(f"Updated strategy weights for {ticker}: {strategy_weights}")
                last_strategy_update = current_time
            
            # Update shared data
            with shared_data["lock"]:
                shared_data[f"{ticker}_total_orders"] = total_orders
                shared_data[f"{ticker}_total_positions"] = total_positions
                shared_data[f"{ticker}_market_regime"] = market_data.market_regime
                shared_data[f"{ticker}_volatility"] = market_data.volatility
            
            # Log status periodically
            if elapsed_seconds % 60 < CHECK_INTERVAL:
                current_position, _ = get_current_position(trader, ticker)
                current_pl = trader.get_portfolio_item(ticker).get_realized_pl() - initial_pl
                
                logger.info(f"{ticker} Status: Position={current_position}, " +
                           f"Orders={total_orders}, Positions={total_positions}, " +
                           f"P&L=${current_pl:.2f}, Regime={market_data.market_regime}, " +
                           f"Volatility={market_data.volatility:.4f}")
            
            # Short pause in main loop
            time.sleep(CHECK_INTERVAL)
            
        except Exception as e:
            logger.error(f"Error in main loop for {ticker}: {e}")
            traceback.print_exc()
            time.sleep(1)
    
    # End of day cleanup
    try:
        logger.info(f"End of day cleanup for {ticker}")
        
        # Cancel all orders
        cancel_orders(trader, ticker)
        
        # Close all positions
        close_positions(trader, ticker)
        
        # Final P&L calculation
        final_pl = trader.get_portfolio_item(ticker).get_realized_pl() - initial_pl
        
        logger.info(f"Strategy complete for {ticker}. Orders: {total_orders}, " +
                   f"Positions: {total_positions}, P&L: ${final_pl:.2f}")
        
        # Update shared data with final results
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
            shared_data[f"{ticker}_error"] = str(e)
            shared_data[f"{ticker}_completed"] = True
        
        return False

def main():
    """Main function to initialize and run the trading strategy"""
    global logger
    logger = setup_logging()
    
    logger.info("Starting RL-optimized trading strategy")
    
    with shift.Trader("dolla-dolla-bills-yall") as trader:
        try:
            # Connect to the server
            logger.info("Connecting to SHIFT server...")
            trader.connect("initiator.cfg", "Zxz7Qxa9")
            time.sleep(1)
            
            # Subscribe to order book data
            logger.info("Subscribing to order book data...")
            trader.sub_all_order_book()
            time.sleep(1)
            
            # Determine trading hours
            current = trader.get_last_trade_time()
            start_time = current
            
            # End time - 5 minutes before market close
            # Fix the date/time issues we saw in the error logs
            current_date = current.date()
            end_time = datetime(current_date.year, current_date.month, current_date.day, 15, 55, 0)
            
            # If it's already after the end time, set end time to tomorrow
            if current.hour >= 15 and current.minute >= 55:
                end_time += timedelta(days=1)
                logger.info(f"Already past market close, setting end time to tomorrow: {end_time}")
            
            logger.info(f"Trading strategy starting at {start_time}, will run until {end_time}")
            
            # Track initial portfolio values
            initial_bp = trader.get_portfolio_summary().get_total_bp()
            initial_pl = trader.get_portfolio_summary().get_total_realized_pl()
            
            logger.info(f"Initial buying power: ${initial_bp:.2f}")
            
            # Shared data between threads
            shared_data = {
                "lock": Lock()  # Lock for thread-safe access to shared data
            }
            
            # Initialize shared data
            for ticker in TICKERS:
                shared_data[f"{ticker}_total_orders"] = 0
                shared_data[f"{ticker}_total_positions"] = 0
                shared_data[f"{ticker}_completed"] = False
            
            # Launch trading threads
            threads = []
            for ticker in TICKERS:
                thread = Thread(target=run_strategy, args=(trader, ticker, end_time, shared_data))
                thread.daemon = True
                threads.append(thread)
                thread.start()
                time.sleep(1)  # Stagger thread starts
            
            # Monitor progress while threads are running
            monitoring_active = True
            while monitoring_active and any(thread.is_alive() for thread in threads):
                try:
                    # Wait a bit
                    time.sleep(60)
                    
                    # Calculate current metrics
                    current_bp = trader.get_portfolio_summary().get_total_bp()
                    current_pl = trader.get_portfolio_summary().get_total_realized_pl() - initial_pl
                    
                    # Gather stats from all tickers
                    ticker_stats = {}
                    total_orders = 0
                    total_positions = 0
                    
                    with shared_data["lock"]:
                        for ticker in TICKERS:
                            orders = shared_data.get(f"{ticker}_total_orders", 0)
                            positions = shared_data.get(f"{ticker}_total_positions", 0)
                            
                            ticker_stats[ticker] = {
                                "orders": orders,
                                "positions": positions,
                                "regime": shared_data.get(f"{ticker}_market_regime", "unknown"),
                                "volatility": shared_data.get(f"{ticker}_volatility", 0),
                                "completed": shared_data.get(f"{ticker}_completed", False)
                            }
                            
                            total_orders += orders
                            total_positions += positions
                    
                    # Print status update
                    logger.info(f"Overall Status: P&L=${current_pl:.2f}, " +
                               f"Orders={total_orders}, Positions={total_positions}, " +
                               f"BP=${current_bp:.2f}")
                    
                    # Check if all threads have completed
                    all_completed = all(stat["completed"] for stat in ticker_stats.values())
                    
                    if all_completed:
                        logger.info("All trading threads have completed")
                        monitoring_active = False
                    
                except Exception as e:
                    logger.error(f"Error in monitoring loop: {e}")
            
            # Final cleanup for any remaining positions
            for ticker in TICKERS:
                try:
                    cancel_orders(trader, ticker)
                    close_positions(trader, ticker)
                except Exception as e:
                    logger.error(f"Error during final cleanup for {ticker}: {e}")
            
            # Calculate final results
            final_bp = trader.get_portfolio_summary().get_total_bp()
            final_pl = trader.get_portfolio_summary().get_total_realized_pl() - initial_pl
            
            # Get per-ticker results
            ticker_results = {}
            total_orders = 0
            total_positions = 0
            
            with shared_data["lock"]:
                for ticker in TICKERS:
                    orders = shared_data.get(f"{ticker}_final_orders", 0)
                    positions = shared_data.get(f"{ticker}_final_positions", 0)
                    
                    ticker_results[ticker] = {
                        "pl": shared_data.get(f"{ticker}_final_pl", 0),
                        "orders": orders,
                        "positions": positions
                    }
                    
                    total_orders += orders
                    total_positions += positions
            
            # Print final results
            logger.info("\n===== STRATEGY COMPLETE =====")
            logger.info(f"Final buying power: ${final_bp:.2f}")
            logger.info(f"Total profit/loss: ${final_pl:.2f}")
            logger.info(f"Return on capital: {(final_pl / initial_bp * 100):.2f}%")
            
            # Print ticker-specific results
            logger.info("\n===== TICKER RESULTS =====")
            for ticker, results in ticker_results.items():
                logger.info(f"{ticker}: P&L=${results['pl']:.2f}, " +
                           f"Orders={results['orders']}, Positions={results['positions']}")
            
            # Final success assessment
            if total_orders >= MIN_ORDERS and total_positions >= MIN_POSITIONS:
                logger.info("\n✅ SUCCESS: Minimum requirements met!")
                logger.info(f"Total orders: {total_orders}/{MIN_ORDERS} required")
                logger.info(f"Total positions: {total_positions}/{MIN_POSITIONS} required")
            else:
                logger.info("\n❌ WARNING: Minimum requirements not met!")
                logger.info(f"Total orders: {total_orders}/{MIN_ORDERS} required")
                logger.info(f"Total positions: {total_positions}/{MIN_POSITIONS} required")
            
            return final_pl
            
        except Exception as e:
            logger.error(f"Fatal error in main program: {e}")
            traceback.print_exc()
            
            # Emergency cleanup
            try:
                logger.info("Performing emergency cleanup...")
                for ticker in TICKERS:
                    cancel_orders(trader, ticker)
                    close_positions(trader, ticker)
            except Exception as cleanup_error:
                logger.error(f"Failed to perform emergency cleanup: {cleanup_error}")
            
            return 0

if __name__ == "__main__":
    main()

# final round