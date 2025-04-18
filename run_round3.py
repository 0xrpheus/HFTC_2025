# import shift
# from time import sleep
# from datetime import datetime, timedelta
# import datetime as dt
# from threading import Thread
# import numpy as np
# import math
# import logging
# import random

# # Set up logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#     handlers=[logging.StreamHandler()]
# )
# logger = logging.getLogger("VolatilityTrader")

# # ================= CONFIGURATION PARAMETERS =================
# # These parameters are optimized for high-VIX trading days

# # Strategy weightings (will be dynamically adjusted)
# STRATEGY_WEIGHTS = {
#     "mean_reversion": 1.0,     # Primary strategy - works well in high volatility
#     "momentum": 0.6,           # Secondary strategy for capturing trends
#     "volatility_breakout": 0.8, # Specifically for high-VIX days
#     "rebate_harvesting": 0.3,   # For meeting order minimums
# }

# # Position management
# MAX_POSITION_SIZE = 10         # Max position size per ticker in lots (conservative for high volatility)
# MAX_POSITION_VALUE = 75000     # Maximum $ value of any position
# MAX_TOTAL_POSITIONS = 30       # Maximum total positions across all tickers
# MAX_TICKER_EXPOSURE = 0.25     # Maximum capital in any one ticker (25%)

# # Order management
# ORDER_TARGET = 250             # Target orders per ticker (to meet 200 minimum with safety margin)
# LIMIT_ORDER_RATIO = 0.85       # Portion of orders that should be limit orders for rebates
# MIN_POSITIONS_TARGET = 15      # Target minimum positions per ticker
# CANCEL_REFRESH_SECONDS = 10    # How often to refresh limit orders

# # Risk management (critical for high VIX)
# MAX_DRAWDOWN_PCT = 0.015       # Maximum allowed drawdown (1.5% - conservative)
# INITIAL_VOLATILITY_THRESHOLD = 0.004  # Initial volatility threshold (0.4% - will adapt)
# MAX_VOLATILITY_THRESHOLD = 0.05      # Maximum volatility threshold (5%)
# BASE_STOP_LOSS = 0.01          # Base stop loss (1% - wider for high volatility)

# # Technical parameters
# RSI_PERIOD = 14                # RSI calculation period
# EMA_SHORT = 9                  # Short EMA period
# EMA_MEDIUM = 21                # Medium EMA period 
# EMA_LONG = 50                  # Long EMA period
# VOLATILITY_WINDOW = 20         # Window for volatility calculation
# BB_PERIOD = 20                 # Bollinger Bands period
# BB_STD = 2.0                   # Bollinger Bands standard deviation
# PRICE_HISTORY_MAX = 500        # Max price history to maintain

# # Ticker prioritization - focusing on liquid stocks with high volatility profile
# TICKER_PRIORITY = [
#     "SPY",  # S&P 500 ETF - market barometer
#     "AAPL", # High liquidity, often has good volatility
#     "MSFT", # Tech leader, usually liquid
#     "JPM",  # Financial sector, responsive to volatility
#     "BA",   # Industrial with high beta, swings a lot
#     "DIS",  # Consumer/Media with good movement
#     "GS",   # Financial with high volatility 
#     "CAT",  # Industrial/cyclical with volatility
#     "CSCO", # Tech with moderate volatility
#     "XOM"   # Energy sector, different correlation
# ]

# # ================= INDICATOR FUNCTIONS =================

# def calculate_volatility(prices, window=VOLATILITY_WINDOW):
#     """Calculate historical volatility"""
#     if len(prices) < window + 1:
#         return 0.01  # Default value
    
#     returns = [prices[i] / prices[i-1] - 1 for i in range(1, len(prices))]
#     recent_returns = returns[-window:]
    
#     return np.std(recent_returns) * np.sqrt(252 * 6.5 * 60)  # Annualized

# def calculate_rsi(prices, period=RSI_PERIOD):
#     """Calculate Relative Strength Index"""
#     if len(prices) < period + 1:
#         return 50  # Default to neutral when not enough data
    
#     deltas = np.diff(prices)
#     seed = deltas[:period]
#     up = seed[seed >= 0].sum() / period
#     down = -seed[seed < 0].sum() / period
    
#     if down == 0:
#         return 100
    
#     rs = up / down
#     rsi = 100 - (100 / (1 + rs))
    
#     for delta in deltas[period:]:
#         if delta > 0:
#             upval = delta
#             downval = 0
#         else:
#             upval = 0
#             downval = -delta
            
#         up = (up * (period - 1) + upval) / period
#         down = (down * (period - 1) + downval) / period
        
#         if down == 0:
#             return 100
            
#         rs = up / down
#         rsi = 100 - (100 / (1 + rs))
    
#     return rsi

# def calculate_ema(prices, period):
#     """Calculate Exponential Moving Average"""
#     if len(prices) < period:
#         return None
    
#     multiplier = 2 / (period + 1)
#     ema = prices[0]
    
#     for price in prices[1:]:
#         ema = (price - ema) * multiplier + ema
    
#     return ema

# def calculate_bollinger_bands(prices, period=BB_PERIOD, stdev_factor=BB_STD):
#     """Calculate Bollinger Bands"""
#     if len(prices) < period:
#         return None, None, None
    
#     # Calculate SMA
#     sma = sum(prices[-period:]) / period
    
#     # Calculate standard deviation
#     squared_diff = [(price - sma) ** 2 for price in prices[-period:]]
#     stdev = math.sqrt(sum(squared_diff) / period)
    
#     # Calculate upper and lower bands
#     upper_band = sma + (stdev * stdev_factor)
#     lower_band = sma - (stdev * stdev_factor)
    
#     return sma, upper_band, lower_band

# def detect_trend(prices, short_window=EMA_SHORT, long_window=EMA_LONG):
#     """Detect price trend using EMAs"""
#     if len(prices) < long_window:
#         return 0  # No trend
    
#     short_ema = calculate_ema(prices, short_window)
#     long_ema = calculate_ema(prices, long_window)
    
#     if short_ema is None or long_ema is None:
#         return 0
    
#     # Calculate trend strength as percentage difference
#     trend_strength = (short_ema - long_ema) / long_ema
    
#     if trend_strength > 0.002:  # 0.2% threshold - lower for high-vol environment
#         return 1  # Uptrend
#     elif trend_strength < -0.002:
#         return -1  # Downtrend
#     else:
#         return 0  # No clear trend

# def detect_mean_reversion_signal(prices, rsi, bollinger_bands, volatility):
#     """Detect mean reversion signals with volatility adjustment"""
#     if len(prices) < 5 or bollinger_bands[0] is None:
#         return 0
    
#     sma, upper_band, lower_band = bollinger_bands
#     current_price = prices[-1]
    
#     # Calculate band penetration percentage
#     upper_penetration = (current_price - upper_band) / upper_band if current_price > upper_band else 0
#     lower_penetration = (lower_band - current_price) / lower_band if current_price < lower_band else 0
    
#     # Adjust signal strength based on volatility
#     volatility_factor = min(2.0, max(0.5, 0.01 / volatility if volatility > 0 else 1.0))
    
#     # Combine RSI and Bollinger Band signals
#     signal = 0
    
#     # Oversold conditions - buy signal
#     if (rsi < 30 and current_price < lower_band) or (lower_penetration > 0.005):
#         # Stronger signal when deeper penetration
#         signal_strength = (0.5 + lower_penetration * 10) * volatility_factor
#         signal = min(1.0, signal_strength)  # Buy signal, capped at 1.0
    
#     # Overbought conditions - sell signal
#     elif (rsi > 70 and current_price > upper_band) or (upper_penetration > 0.005):
#         # Stronger signal when deeper penetration
#         signal_strength = (0.5 + upper_penetration * 10) * volatility_factor
#         signal = max(-1.0, -signal_strength)  # Sell signal, capped at -1.0
    
#     return signal

# def detect_momentum_signal(prices, trend, rsi, volatility):
#     """Detect momentum signals with volatility adjustment"""
#     if len(prices) < 5:
#         return 0
    
#     # Calculate short-term returns
#     short_return = prices[-1] / prices[-3] - 1 if len(prices) >= 3 else 0
    
#     # Scale the momentum based on volatility - stronger in high vol environments
#     volatility_factor = min(2.0, max(0.5, volatility * 50))
    
#     # Combine trend, RSI and recent returns for momentum signal
#     signal = 0
    
#     # Strong uptrend with momentum
#     if trend > 0 and short_return > 0.002 * volatility_factor and rsi < 80:
#         signal_strength = min(1.0, (short_return / (0.002 * volatility_factor)) * 0.5)
#         signal = signal_strength  # Buy signal
    
#     # Strong downtrend with momentum
#     elif trend < 0 and short_return < -0.002 * volatility_factor and rsi > 20:
#         signal_strength = min(1.0, (abs(short_return) / (0.002 * volatility_factor)) * 0.5)
#         signal = -signal_strength  # Sell signal
    
#     return signal

# def detect_volatility_breakout(prices, volumes, volatility, bollinger_bands):
#     """Detect volatility breakouts - specifically for high VIX environments"""
#     if len(prices) < 15 or bollinger_bands[0] is None:
#         return 0
    
#     # Unpack Bollinger Bands
#     sma, upper_band, lower_band = bollinger_bands
    
#     # Calculate price change rate
#     price_change = (prices[-1] - prices[-5]) / prices[-5] if len(prices) >= 5 else 0
    
#     # Volatility breakout occurs when price moves significantly along with rising volatility
#     vol_signal = 0
    
#     # Enhanced volume indicator (if volume data available)
#     volume_surge = False
#     if len(volumes) >= 10:
#         avg_volume = sum(volumes[-10:-1]) / 9  # Average of previous 9 periods
#         current_volume = volumes[-1]
#         volume_surge = current_volume > avg_volume * 1.5
    
#     # Upside breakout
#     if price_change > volatility * 2 and prices[-1] > upper_band and (volume_surge or len(volumes) < 10):
#         vol_signal = min(1.0, price_change / (volatility * 2))
    
#     # Downside breakout
#     elif price_change < -volatility * 2 and prices[-1] < lower_band and (volume_surge or len(volumes) < 10):
#         vol_signal = max(-1.0, price_change / (volatility * 2))
    
#     return vol_signal

# # ================= UTILITY FUNCTIONS =================

# def cancel_orders(trader, ticker):
#     """Cancel all waiting orders for a specific ticker"""
#     order_ids = []
    
#     for order in trader.get_waiting_list():
#         if order.symbol == ticker:
#             order_ids.append(order.id)
#             trader.submit_cancellation(order)
#             sleep(0.1)  # Small sleep to avoid overwhelming the system
    
#     if order_ids:
#         logger.info(f"Cancelled {len(order_ids)} orders for {ticker}")
    
#     return order_ids

# def close_positions(trader, ticker):
#     """Close all positions for a specific ticker"""
#     logger.info(f"Closing positions for {ticker}")
#     item = trader.get_portfolio_item(ticker)

#     long_shares = item.get_long_shares()
#     if long_shares > 0:
#         logger.info(f"Market selling {ticker} long shares = {long_shares}")
#         order = shift.Order(shift.Order.Type.MARKET_SELL, ticker, int(long_shares/100))
#         trader.submit_order(order)
#         sleep(0.2)

#     short_shares = item.get_short_shares()
#     if short_shares > 0:
#         logger.info(f"Market buying {ticker} short shares = {short_shares}")
#         order = shift.Order(shift.Order.Type.MARKET_BUY, ticker, int(short_shares/100))
#         trader.submit_order(order)
#         sleep(0.2)
        
#     return long_shares, short_shares

# def get_current_prices(trader, ticker):
#     """Get current market prices with error handling"""
#     try:
#         best_price = trader.get_best_price(ticker)
#         if best_price.get_bid_price() <= 0 or best_price.get_ask_price() <= 0:
#             logger.warning(f"Invalid prices for {ticker}")
#             return None, None, None, None
            
#         bid_price = best_price.get_bid_price()
#         ask_price = best_price.get_ask_price()
#         mid_price = (bid_price + ask_price) / 2
#         spread = ask_price - bid_price
        
#         return bid_price, ask_price, mid_price, spread
#     except Exception as e:
#         logger.error(f"Error getting prices for {ticker}: {e}")
#         return None, None, None, None

# def get_current_position(trader, ticker):
#     """Get current position for a ticker"""
#     try:
#         portfolio_item = trader.get_portfolio_item(ticker)
#         long_shares = portfolio_item.get_long_shares()
#         short_shares = portfolio_item.get_short_shares()
#         current_position = (long_shares - short_shares) / 100  # Convert to lots
#         return current_position, portfolio_item
#     except Exception as e:
#         logger.error(f"Error getting position for {ticker}: {e}")
#         return 0, None

# def calculate_adaptive_thresholds(volatility, market_volatility):
#     """Calculate adaptive thresholds based on current volatility"""
#     # Adjust for high VIX environment - wider thresholds overall
#     vol_factor = min(2.0, max(1.0, market_volatility / volatility if volatility > 0 else 1.5))
    
#     # Scale thresholds based on current volatility
#     entry_threshold = min(MAX_VOLATILITY_THRESHOLD, max(INITIAL_VOLATILITY_THRESHOLD, volatility * 1.5))
#     profit_target = max(0.003, min(0.03, volatility * 3.0 * vol_factor))
#     stop_loss = max(BASE_STOP_LOSS, min(0.05, volatility * 4.0 * vol_factor))
    
#     return entry_threshold, profit_target, stop_loss

# def calculate_position_size(ticker, trader, signal_strength, volatility, price, max_position, current_position):
#     """Calculate optimal position size based on multiple factors"""
    
#     # Base size (inverse relationship with volatility - smaller positions in high vol)
#     if volatility > 0:
#         vol_adjustment = min(1.0, max(0.2, 0.005 / volatility))
#     else:
#         vol_adjustment = 0.5

#     # Scale with signal strength
#     signal_adjustment = abs(signal_strength)
    
#     # Calculate dollar value per lot
#     dollar_per_lot = price * 100  # Each lot is 100 shares
    
#     # Maximum lots based on risk limit
#     max_size_by_value = int(MAX_POSITION_VALUE / dollar_per_lot)
    
#     # Calculate position size
#     base_size = 3  # Start with a base size of 3 lots
#     position_size = max(1, int(base_size * vol_adjustment * signal_adjustment))
    
#     # Adjust size based on remaining capacity
#     remaining_capacity = max_position - abs(current_position)
#     if remaining_capacity <= 0:
#         return 0
    
#     # Apply constraints
#     position_size = min(position_size, max_size_by_value, int(remaining_capacity))
    
#     return max(1, position_size)  # Minimum size of 1

# def adjust_strategy_weights(trader, strategies_performance, weights):
#     """Dynamically adjust strategy weights based on performance"""
#     # Calculate total absolute PnL
#     total_abs_pnl = sum([abs(perf) for perf in strategies_performance.values()])
    
#     # If no significant PnL, don't adjust weights
#     if total_abs_pnl < 10:
#         return weights
    
#     # Calculate new weights
#     new_weights = {}
#     for strategy, pnl in strategies_performance.items():
#         if pnl > 0:
#             # Increase weight for profitable strategies
#             new_weights[strategy] = weights[strategy] * (1 + pnl / total_abs_pnl)
#         else:
#             # Decrease weight for unprofitable strategies, but don't eliminate
#             new_weights[strategy] = weights[strategy] * max(0.3, 1 + pnl / total_abs_pnl)
    
#     # Normalize weights
#     total_weight = sum(new_weights.values())
#     if total_weight > 0:
#         new_weights = {k: v / total_weight * len(weights) for k, v in new_weights.items()}
    
#     # Ensure no weight falls below minimum threshold
#     for strategy in new_weights:
#         new_weights[strategy] = max(0.2, min(2.0, new_weights[strategy]))
    
#     logger.info(f"Adjusted strategy weights: {new_weights}")
#     return new_weights

# def create_limit_order(trader, ticker, is_buy, size, price, active_trades, current_time):
#     """Create and submit a limit order with tracking"""
#     order_type = shift.Order.Type.LIMIT_BUY if is_buy else shift.Order.Type.LIMIT_SELL
    
#     order = shift.Order(order_type, ticker, size, price)
#     trader.submit_order(order)
    
#     # Track this order
#     direction = 1 if is_buy else -1
#     order_id = f"limit_{direction}_{current_time.strftime('%H%M%S')}_{random.randint(0, 999)}"
    
#     active_trades[order_id] = {
#         "order_id": order.id,
#         "direction": direction,
#         "size": size,
#         "price": price,
#         "entry_time": current_time,
#         "type": "limit",
#         "status": "pending"
#     }
    
#     return order_id

# def create_market_order(trader, ticker, is_buy, size, active_trades, current_time, strategy_name="unknown"):
#     """Create and submit a market order with tracking"""
#     order_type = shift.Order.Type.MARKET_BUY if is_buy else shift.Order.Type.MARKET_SELL
    
#     order = shift.Order(order_type, ticker, size)
#     trader.submit_order(order)
    
#     # Track this order
#     direction = 1 if is_buy else -1
#     order_id = f"market_{direction}_{current_time.strftime('%H%M%S')}_{random.randint(0, 999)}"
    
#     active_trades[order_id] = {
#         "order_id": order.id,
#         "direction": direction,
#         "size": size,
#         "entry_time": current_time,
#         "type": "market",
#         "status": "executed",
#         "strategy": strategy_name
#     }
    
#     return order_id

# def manage_active_trades(trader, ticker, active_trades, bid_price, ask_price, mid_price,
#                         current_time, profit_target, stop_loss, market_volatility):
#     """Manage existing trades - take profit, stop loss, etc."""
#     trades_to_remove = []
#     executions = 0
    
#     for trade_id, trade in list(active_trades.items()):
#         try:
#             # Skip pending limit orders
#             if trade["type"] == "limit" and trade["status"] == "pending":
#                 continue
            
#             # Check for trade expiration (10 minute max hold time)
#             if (current_time - trade["entry_time"]).total_seconds() > 600:
#                 if trade["direction"] > 0:  # Long trade
#                     create_market_order(trader, ticker, False, trade["size"], active_trades, current_time, "expiration")
#                 else:  # Short trade
#                     create_market_order(trader, ticker, True, trade["size"], active_trades, current_time, "expiration")
                
#                 trades_to_remove.append(trade_id)
#                 executions += 1
#                 logger.info(f"Time-based exit on {ticker} trade at {mid_price:.2f}")
#                 continue
            
#             # For executed trades, check for exit conditions
#             if "entry_price" in trade:
#                 entry_price = trade["entry_price"]
                
#                 # Adjust profit target and stop loss based on volatility
#                 volatility_factor = min(1.5, max(0.8, market_volatility / 0.02))  # Normalize to expected VIX
#                 adjusted_profit = profit_target * volatility_factor
#                 adjusted_stop = stop_loss * volatility_factor
                
#                 if trade["direction"] > 0:  # Long trade
#                     # Take profit
#                     if mid_price >= entry_price * (1 + adjusted_profit):
#                         create_market_order(trader, ticker, False, trade["size"], active_trades, current_time, "take_profit")
#                         trades_to_remove.append(trade_id)
#                         executions += 1
#                         logger.info(f"Take profit on {ticker} long trade at {mid_price:.2f}")
                    
#                     # Stop loss
#                     elif mid_price <= entry_price * (1 - adjusted_stop):
#                         create_market_order(trader, ticker, False, trade["size"], active_trades, current_time, "stop_loss")
#                         trades_to_remove.append(trade_id)
#                         executions += 1
#                         logger.info(f"Stop loss on {ticker} long trade at {mid_price:.2f}")
                
#                 elif trade["direction"] < 0:  # Short trade
#                     # Take profit
#                     if mid_price <= entry_price * (1 - adjusted_profit):
#                         create_market_order(trader, ticker, True, trade["size"], active_trades, current_time, "take_profit")
#                         trades_to_remove.append(trade_id)
#                         executions += 1
#                         logger.info(f"Take profit on {ticker} short trade at {mid_price:.2f}")
                    
#                     # Stop loss
#                     elif mid_price >= entry_price * (1 + adjusted_stop):
#                         create_market_order(trader, ticker, True, trade["size"], active_trades, current_time, "stop_loss")
#                         trades_to_remove.append(trade_id)
#                         executions += 1
#                         logger.info(f"Stop loss on {ticker} short trade at {mid_price:.2f}")
            
#         except Exception as e:
#             logger.error(f"Error managing trade {trade_id} for {ticker}: {e}")
#             trades_to_remove.append(trade_id)
    
#     # Remove completed trades
#     for trade_id in trades_to_remove:
#         active_trades.pop(trade_id, None)
    
#     return executions

# def update_limit_order_status(trader, ticker, active_trades):
#     """Update status of limit orders"""
#     filled_orders = 0
    
#     # Check for filled limit orders
#     for order in trader.get_submitted_orders():
#         if order.symbol == ticker and order.status == shift.Order.Status.FILLED:
#             # Find matching orders in our active_trades
#             for trade_id, trade in list(active_trades.items()):
#                 if "order_id" in trade and trade["order_id"] == order.id and trade["status"] == "pending":
#                     # Update the trade with execution details
#                     trade["status"] = "executed"
#                     trade["entry_price"] = order.executed_price
#                     filled_orders += 1
                    
#                     logger.info(f"Limit order filled for {ticker} direction={trade['direction']} at {order.executed_price:.2f}")
    
#     return filled_orders

# def execute_mean_reversion_strategy(trader, ticker, active_trades, indicators, 
#                                    current_position, max_position, 
#                                    bid_price, ask_price, mid_price,
#                                    adaptive_threshold, entry_signal, current_time):
#     """Execute mean reversion strategy"""
#     if entry_signal == 0:
#         return 0
    
#     order_count = 0
    
#     # Calculate position size based on signal strength and volatility
#     position_size = calculate_position_size(
#         ticker, trader, entry_signal, indicators["volatility"], 
#         mid_price, max_position, current_position
#     )
    
#     if position_size <= 0:
#         return 0
    
#     # Entry logic
#     if entry_signal > 0 and current_position < max_position:  # Buy signal
#         # Use limit order slightly below current bid
#         limit_price = max(0.01, round(bid_price * 0.998, 2))
        
#         create_limit_order(
#             trader, ticker, True, position_size, 
#             limit_price, active_trades, current_time
#         )
        
#         logger.info(f"Mean reversion BUY signal for {ticker} at {limit_price:.2f}, size={position_size}")
#         order_count += 1
        
#     elif entry_signal < 0 and current_position > -max_position:  # Sell signal
#         # Use limit order slightly above current ask
#         limit_price = round(ask_price * 1.002, 2)
        
#         create_limit_order(
#             trader, ticker, False, position_size, 
#             limit_price, active_trades, current_time
#         )
        
#         logger.info(f"Mean reversion SELL signal for {ticker} at {limit_price:.2f}, size={position_size}")
#         order_count += 1
    
#     return order_count

# def execute_momentum_strategy(trader, ticker, active_trades, indicators,
#                              current_position, max_position,
#                              bid_price, ask_price, mid_price,
#                              adaptive_threshold, entry_signal, current_time):
#     """Execute momentum strategy"""
#     if entry_signal == 0:
#         return 0
    
#     order_count = 0
    
#     # Calculate position size - generally smaller for momentum in high-vol
#     position_size = max(1, int(calculate_position_size(
#         ticker, trader, entry_signal, indicators["volatility"],
#         mid_price, max_position, current_position
#     ) * 0.7))  # Scale down for momentum strategies
    
#     if position_size <= 0:
#         return 0
    
#     # Entry logic - using market orders for momentum to ensure execution
#     if entry_signal > 0 and current_position < max_position:  # Buy signal
#         # Market order for immediate execution
#         create_market_order(
#             trader, ticker, True, position_size, 
#             active_trades, current_time, "momentum"
#         )
        
#         logger.info(f"Momentum BUY signal for {ticker} at market, size={position_size}")
#         order_count += 1
        
#     elif entry_signal < 0 and current_position > -max_position:  # Sell signal
#         create_market_order(
#             trader, ticker, False, position_size, 
#             active_trades, current_time, "momentum"
#         )
        
#         logger.info(f"Momentum SELL signal for {ticker} at market, size={position_size}")
#         order_count += 1
    
#     return order_count

# def execute_volatility_breakout_strategy(trader, ticker, active_trades, indicators,
#                                         current_position, max_position,
#                                         bid_price, ask_price, mid_price,
#                                         adaptive_threshold, entry_signal, current_time):
#     """Execute volatility breakout strategy - specifically for high VIX days"""
#     if entry_signal == 0:
#         return 0
    
#     order_count = 0
    
#     # Calculate position size - can be larger for breakout trades
#     position_size = calculate_position_size(
#         ticker, trader, entry_signal, indicators["volatility"],
#         mid_price, max_position, current_position
#     )
    
#     if position_size <= 0:
#         return 0
    
#     # Entry logic - using market orders for breakouts to ensure we catch the move
#     if entry_signal > 0 and current_position < max_position:  # Buy signal
#         create_market_order(
#             trader, ticker, True, position_size, 
#             active_trades, current_time, "vol_breakout"
#         )
        
#         logger.info(f"Volatility breakout BUY for {ticker} at market, size={position_size}")
#         order_count += 1
        
#     elif entry_signal < 0 and current_position > -max_position:  # Sell signal
#         create_market_order(
#             trader, ticker, False, position_size, 
#             active_trades, current_time, "vol_breakout"
#         )
        
#         logger.info(f"Volatility breakout SELL for {ticker} at market, size={position_size}")
#         order_count += 1
    
#     return order_count

# def execute_rebate_harvesting(trader, ticker, active_trades, current_position, max_position,
#                              bid_price, ask_price, mid_price, spread, current_time):
#     """Place limit orders far from market to harvest rebates"""
#     if abs(current_position) >= max_position:
#         return 0
    
#     order_count = 0
#     position_size = 1  # Always use minimum size for rebate harvesting
    
#     # Calculate prices far from current market
#     buy_buffer = max(0.1, spread * 5)  # Place buy orders far below market
#     sell_buffer = max(0.1, spread * 5)  # Place sell orders far above market
    
#     buy_price = round(bid_price * 0.97, 2)  # 3% below bid
#     sell_price = round(ask_price * 1.03, 2)  # 3% above ask
    
#     # Only place buys if we have room for long positions
#     if current_position < max_position:
#         create_limit_order(
#             trader, ticker, True, position_size,
#             buy_price, active_trades, current_time
#         )
#         order_count += 1
    
#     # Only place sells if we have room for short positions
#     if current_position > -max_position:
#         create_limit_order(
#             trader, ticker, False, position_size,
#             sell_price, active_trades, current_time
#         )
#         order_count += 1
    
#     return order_count

# def ensure_minimum_orders(trader, ticker, order_count, order_target, target_time, elapsed_time,
#                          active_trades, current_position, max_position,
#                          bid_price, ask_price, mid_price, current_time):
#     """Ensure we're on track to meet minimum order requirements"""
#     if elapsed_time <= 0 or target_time <= 0:
#         return 0
    
#     # Calculate expected orders by this time
#     order_rate = order_target / target_time
#     expected_orders = order_rate * elapsed_time
    
#     # If we're behind on orders
#     if order_count < expected_orders * 0.8:
#         orders_needed = min(5, int(expected_orders - order_count))
#         new_orders = 0
        
#         # Place additional limit orders
#         for i in range(orders_needed):
#             if i % 2 == 0 and current_position < max_position:
#                 # Buy far from market
#                 price = round(bid_price * 0.96, 2)
#                 create_limit_order(trader, ticker, True, 1, price, active_trades, current_time)
#                 new_orders += 1
#             elif current_position > -max_position:
#                 # Sell far from market
#                 price = round(ask_price * 1.04, 2)
#                 create_limit_order(trader, ticker, False, 1, price, active_trades, current_time)
#                 new_orders += 1
        
#         if new_orders > 0:
#             logger.info(f"Added {new_orders} additional orders for {ticker} to meet minimum requirements")
        
#         return new_orders
    
#     return 0

# def calculate_market_volatility(ticker_indicators, spy_rsi=None):
#     """Calculate overall market volatility based on multiple tickers"""
#     volatilities = [indic["volatility"] for ticker, indic in ticker_indicators.items() 
#                    if "volatility" in indic and indic["volatility"] > 0]
    
#     if not volatilities:
#         return 0.02  # Default for high VIX environment
    
#     # Emphasize higher volatilities which is likely in a high-VIX environment
#     avg_vol = sum(volatilities) / len(volatilities)
#     max_vol = max(volatilities)
    
#     # Blend average and max, emphasizing max more in high-VIX environments
#     market_vol = avg_vol * 0.3 + max_vol * 0.7
    
#     # Adjust with SPY RSI if available (higher RSI = lower perceived vol)
#     if spy_rsi:
#         rsi_factor = (100 - spy_rsi) / 50  # 1.0 at RSI 50, higher when RSI lower
#         market_vol *= min(1.5, max(0.7, rsi_factor))
    
#     return market_vol

# # ================= MAIN STRATEGY FUNCTION =================

# def strategy(trader: shift.Trader, ticker: str, endtime, market_data, strategy_weights):
#     """Main trading strategy for a single ticker"""
#     logger.info(f"Starting volatility-based trading strategy for {ticker}")
#     initial_pl = trader.get_portfolio_item(ticker).get_realized_pl()
    
#     # Trading frequency
#     check_freq = 0.5  # Check market every 0.5 seconds
    
#     # Data storage
#     price_history = []
#     bid_history = []
#     ask_history = []
#     spread_history = []
#     volume_history = []
    
#     # Trade management
#     active_trades = {}
    
#     # Counters
#     order_count = 0
#     execution_count = 0
#     last_order_refresh = trader.get_last_trade_time()
#     last_strategy_check = trader.get_last_trade_time()
#     start_time = trader.get_last_trade_time()
    
#     # Performance tracking
#     strategy_performance = {
#         "mean_reversion": 0,
#         "momentum": 0,
#         "volatility_breakout": 0,
#         "rebate_harvesting": 0
#     }
    
#     entry_threshold = INITIAL_VOLATILITY_THRESHOLD
#     profit_target = 0.01  # Initial 1% target
#     stop_loss = BASE_STOP_LOSS
    
#     # Position sizing
#     max_position = MAX_POSITION_SIZE
    
#     # Main trading loop
#     while trader.get_last_trade_time() < endtime:
#         try:
#             current_time = trader.get_last_trade_time()
            
#             # Get current market data
#             bid_price, ask_price, mid_price, spread = get_current_prices(trader, ticker)
#             if None in (bid_price, ask_price, mid_price, spread):
#                 sleep(0.5)
#                 continue
            
#             # Get current position
#             current_position, portfolio_item = get_current_position(trader, ticker)
            
#             # Update price history
#             price_history.append(mid_price)
#             bid_history.append(bid_price)
#             ask_history.append(ask_price)
#             spread_history.append(spread)
#             volume_history.append(1)  # Placeholder for volume
            
#             # Keep history at reasonable size
#             if len(price_history) > PRICE_HISTORY_MAX:
#                 price_history = price_history[-PRICE_HISTORY_MAX:]
#                 bid_history = bid_history[-PRICE_HISTORY_MAX:]
#                 ask_history = ask_history[-PRICE_HISTORY_MAX:]
#                 spread_history = spread_history[-PRICE_HISTORY_MAX:]
#                 volume_history = volume_history[-PRICE_HISTORY_MAX:]
            
#             # Calculate indicators
#             indicators = {}
            
#             # Volatility (critical for high-VIX environment)
#             if len(price_history) >= VOLATILITY_WINDOW:
#                 volatility = calculate_volatility(price_history, VOLATILITY_WINDOW)
#                 indicators["volatility"] = volatility
#             else:
#                 indicators["volatility"] = 0.02  # Default for high-VIX environment
            
#             # RSI
#             if len(price_history) >= RSI_PERIOD:
#                 rsi = calculate_rsi(price_history, RSI_PERIOD)
#                 indicators["rsi"] = rsi
#             else:
#                 indicators["rsi"] = 50
            
#             # EMAs for trend detection
#             if len(price_history) >= EMA_LONG:
#                 short_ema = calculate_ema(price_history, EMA_SHORT)
#                 medium_ema = calculate_ema(price_history, EMA_MEDIUM)
#                 long_ema = calculate_ema(price_history, EMA_LONG)
                
#                 indicators["ema_short"] = short_ema
#                 indicators["ema_medium"] = medium_ema
#                 indicators["ema_long"] = long_ema
            
#             # Trend detection
#             trend = detect_trend(price_history)
#             indicators["trend"] = trend
            
#             # Bollinger Bands
#             if len(price_history) >= BB_PERIOD:
#                 bollinger_bands = calculate_bollinger_bands(price_history)
#                 indicators["bollinger_bands"] = bollinger_bands
#             else:
#                 indicators["bollinger_bands"] = (None, None, None)
            
#             # Get market volatility from shared data
#             market_volatility = market_data.get("market_volatility", 0.02)
            
#             # Update local ticker indicators in shared data
#             market_data["ticker_indicators"][ticker] = indicators
            
#             # Calculate adaptive thresholds based on volatility
#             entry_threshold, profit_target, stop_loss = calculate_adaptive_thresholds(
#                 indicators.get("volatility", 0.02), market_volatility)
            
#             # Refresh limit orders periodically
#             if (current_time - last_order_refresh).total_seconds() > CANCEL_REFRESH_SECONDS:
#                 cancel_orders(trader, ticker)
#                 last_order_refresh = current_time
            
#             # Update status of limit orders
#             filled_orders = update_limit_order_status(trader, ticker, active_trades)
#             execution_count += filled_orders
            
#             # Manage active trades (check for exits)
#             executions = manage_active_trades(
#                 trader, ticker, active_trades, bid_price, ask_price, mid_price,
#                 current_time, profit_target, stop_loss, market_volatility)
            
#             execution_count += executions
            
#             # Execute strategies periodically
#             if (current_time - last_strategy_check).total_seconds() > 15:  # Every 15 seconds
#                 # Calculate entry signals
#                 mean_reversion_signal = detect_mean_reversion_signal(
#                     price_history, indicators.get("rsi", 50), 
#                     indicators.get("bollinger_bands", (None, None, None)),
#                     indicators.get("volatility", 0.02))
                
#                 momentum_signal = detect_momentum_signal(
#                     price_history, indicators.get("trend", 0),
#                     indicators.get("rsi", 50), indicators.get("volatility", 0.02))
                
#                 volatility_breakout_signal = detect_volatility_breakout(
#                     price_history, volume_history, indicators.get("volatility", 0.02),
#                     indicators.get("bollinger_bands", (None, None, None)))
                
#                 # Apply strategy weights
#                 weighted_mean_reversion = mean_reversion_signal * strategy_weights.get("mean_reversion", 1.0)
#                 weighted_momentum = momentum_signal * strategy_weights.get("momentum", 0.6)
#                 weighted_vol_breakout = volatility_breakout_signal * strategy_weights.get("volatility_breakout", 0.8)
                
#                 # Execute strategies - focusing on high-vol responsive strategies
#                 if abs(weighted_mean_reversion) > entry_threshold * 0.7:
#                     mr_orders = execute_mean_reversion_strategy(
#                         trader, ticker, active_trades, indicators,
#                         current_position, max_position,
#                         bid_price, ask_price, mid_price,
#                         entry_threshold, weighted_mean_reversion, current_time)
                    
#                     order_count += mr_orders
                
#                 if abs(weighted_momentum) > entry_threshold:
#                     mom_orders = execute_momentum_strategy(
#                         trader, ticker, active_trades, indicators,
#                         current_position, max_position,
#                         bid_price, ask_price, mid_price,
#                         entry_threshold, weighted_momentum, current_time)
                    
#                     order_count += mom_orders
                
#                 if abs(weighted_vol_breakout) > entry_threshold * 0.8:
#                     vb_orders = execute_volatility_breakout_strategy(
#                         trader, ticker, active_trades, indicators,
#                         current_position, max_position,
#                         bid_price, ask_price, mid_price,
#                         entry_threshold, weighted_vol_breakout, current_time)
                    
#                     order_count += vb_orders
                
#                 # Execute rebate harvesting strategy if enabled and we need more orders
#                 if strategy_weights.get("rebate_harvesting", 0.3) > 0.2:
#                     # Only place these orders if we're behind on order count
#                     elapsed_seconds = (current_time - start_time).total_seconds()
#                     total_seconds = (endtime - start_time).total_seconds()
                    
#                     expected_orders = (elapsed_seconds / total_seconds) * ORDER_TARGET
#                     if order_count < expected_orders:
#                         rb_orders = execute_rebate_harvesting(
#                             trader, ticker, active_trades, current_position, max_position,
#                             bid_price, ask_price, mid_price, spread, current_time)
                        
#                         order_count += rb_orders
                
#                 last_strategy_check = current_time
            
#             # Ensure minimum orders and positions
#             elapsed_seconds = (current_time - start_time).total_seconds()
#             total_seconds = (endtime - start_time).total_seconds()
            
#             additional_orders = ensure_minimum_orders(
#                 trader, ticker, order_count, ORDER_TARGET, total_seconds, elapsed_seconds,
#                 active_trades, current_position, max_position,
#                 bid_price, ask_price, mid_price, current_time)
            
#             order_count += additional_orders
            
#             # Log status periodically
#             if order_count % 20 == 0 and "last_log_time" not in locals():
#                 last_log_time = current_time
#                 logger.info(f"{ticker} - Position: {current_position}, Orders: {order_count}, "
#                            f"Volatility: {indicators.get('volatility', 0):.4f}, "
#                            f"RSI: {indicators.get('rsi', 0):.1f}, "
#                            f"Trend: {indicators.get('trend', 0)}")
#             elif "last_log_time" in locals() and (current_time - last_log_time).total_seconds() > 60:
#                 last_log_time = current_time
#                 logger.info(f"{ticker} - Position: {current_position}, Orders: {order_count}, "
#                            f"Volatility: {indicators.get('volatility', 0):.4f}, "
#                            f"RSI: {indicators.get('rsi', 0):.1f}, "
#                            f"Trend: {indicators.get('trend', 0)}")
            
#             sleep(check_freq)
            
#         except Exception as e:
#             logger.error(f"Error in strategy for {ticker}: {e}", exc_info=True)
#             sleep(1)
    
#     # End of day cleanup
#     cancel_orders(trader, ticker)
#     close_positions(trader, ticker)
    
#     # Calculate final P&L
#     final_pl = trader.get_portfolio_item(ticker).get_realized_pl() - initial_pl
#     logger.info(f"Strategy for {ticker} completed. Orders: {order_count}, Executions: {execution_count}")
#     logger.info(f"Final P&L for {ticker}: ${final_pl:.2f}")
    
#     return final_pl, order_count, execution_count

# def main(trader):
#     """Main function to run the multi-ticker trading strategy"""
#     logger.info("Starting volatility-optimized trading strategy for high VIX environment")
    
#     # Select tickers to trade - prioritizing high liquidity & volatility tickers in high VIX
#     ticker_list = TICKER_PRIORITY[:5]  # Use top 5 tickers
    
#     # Determine trading hours
#     current = trader.get_last_trade_time()
#     start_time = current
    
#     # End 5 minutes before market close to ensure all positions are closed
#     end_time = datetime.combine(current.date(), dt.time(15, 55, 0))
    
#     logger.info(f"Trading strategy starting at {start_time}, will run until {end_time}")
    
#     # Initial portfolio values
#     initial_total_pl = trader.get_portfolio_summary().get_total_realized_pl()
#     initial_bp = trader.get_portfolio_summary().get_total_bp()
    
#     logger.info(f"Initial buying power: ${initial_bp:.2f}")
#     logger.info(f"Trading the following symbols: {ticker_list}")
    
#     # Shared market data
#     market_data = {
#         "market_volatility": 0.02,  # Default high VIX value
#         "ticker_indicators": {},
#         "strategy_weights": STRATEGY_WEIGHTS.copy()
#     }
    
#     # Launch trading threads
#     threads = []
#     thread_objects = {}
    
#     for ticker in ticker_list:
#         thread = Thread(target=strategy, 
#                       args=(trader, ticker, end_time, market_data, market_data["strategy_weights"]))
#         thread.daemon = True
#         threads.append(thread)
#         thread_objects[ticker] = thread
#         thread.start()
#         sleep(1)  # Stagger thread starts
    
#     # Market volatility update thread
#     def update_market_data():
#         while trader.get_last_trade_time() < end_time:
#             try:
#                 # Update market volatility based on all tickers
#                 market_data["market_volatility"] = calculate_market_volatility(
#                     market_data["ticker_indicators"])
                
#                 # Update strategy weights periodically
#                 if "last_weight_update" not in locals():
#                     locals()["last_weight_update"] = trader.get_last_trade_time()
                
#                 current_time = trader.get_last_trade_time()
#                 if "last_weight_update" in locals() and (current_time - locals()["last_weight_update"]).total_seconds() > 300:
#                     # Adjust strategy weights based on performance
#                     # (Simple placeholder - in reality would track performance by strategy)
#                     market_data["strategy_weights"] = adjust_strategy_weights(
#                         trader, {"mean_reversion": 10, "momentum": 5, 
#                               "volatility_breakout": 8, "rebate_harvesting": 2}, 
#                         market_data["strategy_weights"])
                    
#                     locals()["last_weight_update"] = current_time
                
#                 sleep(10)  # Update every 10 seconds
#             except Exception as e:
#                 logger.error(f"Error updating market data: {e}")
#                 sleep(10)
    
#     # Start market data update thread
#     market_thread = Thread(target=update_market_data)
#     market_thread.daemon = True
#     market_thread.start()
    
#     # Wait for threads to complete
#     results = {}
#     for ticker, thread in thread_objects.items():
#         logger.info(f"Waiting for {ticker} thread to complete")
#         thread.join(timeout=60*60*7)  # 7-hour timeout
#         results[ticker] = {
#             "pl": 0,
#             "orders": 0,
#             "executions": 0
#         }
    
#     # Final cleanup
#     for ticker in ticker_list:
#         try:
#             cancel_orders(trader, ticker)
#             close_positions(trader, ticker)
#         except Exception as e:
#             logger.error(f"Error during final cleanup for {ticker}: {e}")
    
#     # Calculate final results
#     final_bp = trader.get_portfolio_summary().get_total_bp()
#     final_pl = trader.get_portfolio_summary().get_total_realized_pl() - initial_total_pl
    
#     total_orders = sum(r.get("orders", 0) for r in results.values())
#     total_executions = sum(r.get("executions", 0) for r in results.values())
    
#     logger.info("\n===== STRATEGY COMPLETE =====")
#     logger.info(f"Final buying power: ${final_bp:.2f}")
#     logger.info(f"Total profit/loss: ${final_pl:.2f}")
#     logger.info(f"Return on capital: {(final_pl / initial_bp * 100):.2f}%")
#     logger.info(f"Total orders placed: {total_orders}")
#     logger.info(f"Total executions: {total_executions}")
    
#     return final_pl, total_orders

# if __name__ == '__main__':
#     with shift.Trader("dolla-dolla-bills-yall") as trader:
#         try:
#             # Connect to the server
#             logger.info("Connecting to SHIFT...")
#             trader.connect("initiator.cfg", "Zxz7Qxa9")
#             sleep(1)
            
#             # Subscribe to order book data
#             logger.info("Subscribing to order book data...")
#             trader.sub_all_order_book()
#             sleep(1)
            
#             # Run main strategy
#             logger.info("Starting main strategy...")
#             main(trader)
            
#         except Exception as e:
#             logger.error(f"Fatal error in main program: {e}", exc_info=True)
#             try:
#                 # Emergency cleanup
#                 logger.info("Performing emergency cleanup...")
#                 tickers = TICKER_PRIORITY[:5]
#                 for ticker in tickers:
#                     cancel_orders(trader, ticker)
#                     close_positions(trader, ticker)
#             except Exception as cleanup_error:
#                 logger.error(f"Failed to perform emergency cleanup: {cleanup_error}")

import shift
from time import sleep
from datetime import datetime, timedelta
import datetime as dt
from threading import Thread, Lock
import numpy as np
import math
import logging
import random
import traceback
import statistics

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("HighVIX_Trader")

# ================= CONFIGURATION PARAMETERS =================
# High-VIX specific settings - optimized for high volatility environments

# Core strategy parameters
TICKER_PRIORITY = [
    "SPY",  # Market index (use as barometer)
    "AAPL", # Tech leader with good liquidity
    "JPM",  # Financial with responsive volatility
    "BA",   # Industrial with high beta
    "MSFT", # Tech with moderate volatility
]

# Position management
MAX_POSITION_SIZE = 10         # Max position size per ticker (conservative)
MAX_POSITION_VALUE = 80000     # Maximum $ value of any position
MAX_TOTAL_POSITIONS = 20       # Maximum open positions across all tickers
MAX_TICKER_ALLOCATION = 0.25   # Maximum capital in any one ticker (25%)

# Order management
MIN_ORDERS_TARGET = 250        # Target orders per ticker (>200 for safety)
MIN_POSITIONS_TARGET = 15      # Target minimum positions per ticker (>10 for safety)
LIMIT_ORDER_RATIO = 0.7        # Portion of orders that should be limit orders (for rebates)
ORDER_REFRESH_SECONDS = 8      # How often to refresh limit orders
EXECUTION_CHECK_SECONDS = 120  # How often to check execution rates
RAPID_FIRE_THRESHOLD = 180     # Minutes before end when we ensure minimums
FAILSAFE_THRESHOLD = 140       # Minutes remaining when we activate failsafe if under targets

# Risk management (critical for high VIX)
BASE_VOLATILITY_THRESHOLD = 0.003   # Base volatility threshold (0.3%)
MAX_VOLATILITY_THRESHOLD = 0.05     # Maximum volatility threshold (5%)
BASE_PROFIT_TARGET = 0.008          # Base profit target (0.8%)
BASE_STOP_LOSS = 0.012              # Base stop loss (1.2%)

# Technical parameters
PRICE_HISTORY_MAX = 300        # Max price history to maintain
VOL_WINDOW = 20                # Window for volatility calculation
RSI_PERIOD = 14                # RSI calculation period
EMA_SHORT = 10                 # Short EMA period
EMA_MEDIUM = 25                # Medium EMA period 
EMA_LONG = 50                  # Long EMA period
BOLLINGER_PERIOD = 20          # Bollinger Bands period
BOLLINGER_STD = 2.0            # Bollinger Bands standard deviation
MACD_FAST = 12                 # MACD fast period
MACD_SLOW = 26                 # MACD slow period
MACD_SIGNAL = 9                # MACD signal period

# Strategy weights - these will dynamically adjust
STRATEGY_WEIGHTS = {
    "mean_reversion": 1.0,   # Primary strategy for high-VIX
    "momentum": 0.7,         # Secondary for strong trends in volatility
    "market_making": 0.5,    # For rebate collection
    "rebate_harvesting": 0.4 # For ensuring order minimums
}

# ================= UTILITY FUNCTIONS =================

def safe_divide(a, b, default=0):
    """Safe division that returns default value if denominator is 0"""
    return a / b if b != 0 else default

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
    """Get current position for a ticker with error handling"""
    try:
        portfolio_item = trader.get_portfolio_item(ticker)
        long_shares = portfolio_item.get_long_shares()
        short_shares = portfolio_item.get_short_shares()
        current_position = (long_shares - short_shares) / 100  # Convert to lots
        
        return current_position, portfolio_item
    except Exception as e:
        logger.error(f"Error getting position for {ticker}: {e}")
        return 0, None

def cancel_orders(trader, ticker):
    """Cancel all waiting orders for a specific ticker"""
    try:
        cancelled_count = 0
        for order in trader.get_waiting_list():
            if order.symbol == ticker:
                trader.submit_cancellation(order)
                cancelled_count += 1
                sleep(0.05)  # Small sleep to avoid overwhelming the system
        
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
            sleep(0.2)

        short_shares = item.get_short_shares()
        if short_shares > 0:
            logger.info(f"Market buying {ticker} short shares = {short_shares}")
            order = shift.Order(shift.Order.Type.MARKET_BUY, ticker, int(short_shares/100))
            trader.submit_order(order)
            sleep(0.2)
            
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
            sleep(0.5)
    
    logger.error(f"Failed to submit order after {retry_attempts} attempts")
    return False

# ================= INDICATOR FUNCTIONS =================

def calculate_volatility(prices, window=20):
    """Calculate historical volatility"""
    if len(prices) < window + 1:
        return 0.01  # Default value for high-VIX environment
    
    returns = [prices[i] / prices[i-1] - 1 for i in range(1, len(prices))]
    recent_returns = returns[-window:]
    
    return np.std(recent_returns) * np.sqrt(252 * 6.5 * 60)  # Annualized

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
    ema = prices[0]
    
    for price in prices[1:]:
        ema = (price - ema) * multiplier + ema
    
    return ema

def calculate_bollinger_bands(prices, period=20, std_dev=2.0):
    """Calculate Bollinger Bands"""
    if len(prices) < period:
        return None, None, None
    
    # Calculate SMA
    sma = sum(prices[-period:]) / period
    
    # Calculate standard deviation
    squared_diff = [(price - sma) ** 2 for price in prices[-period:]]
    stdev = math.sqrt(sum(squared_diff) / period)
    
    # Calculate bands
    upper_band = sma + (stdev * std_dev)
    lower_band = sma - (stdev * std_dev)
    
    return sma, upper_band, lower_band

def calculate_macd(prices, fast_period=12, slow_period=26, signal_period=9):
    """Calculate MACD and Signal Line"""
    if len(prices) < slow_period + signal_period:
        return None, None, None
    
    # Calculate EMAs
    fast_ema = calculate_ema(prices, fast_period)
    slow_ema = calculate_ema(prices, slow_period)
    
    if fast_ema is None or slow_ema is None:
        return None, None, None
    
    # Calculate MACD line
    macd_line = fast_ema - slow_ema
    
    # Calculate signal line
    # For simplicity, we'll calculate the signal line from the last N MACD values
    macd_values = []
    for i in range(signal_period):
        if len(prices) > slow_period + i:
            fast_ema_i = calculate_ema(prices[:-i] if i > 0 else prices, fast_period)
            slow_ema_i = calculate_ema(prices[:-i] if i > 0 else prices, slow_period)
            if fast_ema_i is not None and slow_ema_i is not None:
                macd_values.append(fast_ema_i - slow_ema_i)
    
    if len(macd_values) >= signal_period:
        signal_line = calculate_ema(macd_values, signal_period)
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    return macd_line, None, None

def detect_trend(prices, short_window=EMA_SHORT, long_window=EMA_LONG):
    """Detect price trend using EMAs"""
    if len(prices) < long_window:
        return 0  # No trend detected yet
    
    short_ema = calculate_ema(prices, short_window)
    long_ema = calculate_ema(prices, long_window)
    
    if short_ema is None or long_ema is None:
        return 0
    
    # Calculate trend strength as percentage difference
    trend_strength = (short_ema - long_ema) / long_ema
    
    # Adjusting thresholds for high-VIX - allowing for more sensitivity
    if trend_strength > 0.002:  # 0.2% threshold
        return 1  # Uptrend
    elif trend_strength < -0.002:
        return -1  # Downtrend
    else:
        return 0  # No clear trend

def calculate_vix_adjustment(volatility):
    """Calculate adjustment factor based on volatility (proxy for VIX)"""
    # Higher volatility = higher adjustment factor
    # This will be used to adjust various parameters
    base_volatility = 0.01  # 1% is considered normal
    return min(3.0, max(1.0, volatility / base_volatility))

def calculate_adaptive_thresholds(volatility):
    """Calculate adaptive thresholds based on current volatility"""
    vix_adjustment = calculate_vix_adjustment(volatility)
    
    # Scale thresholds with volatility
    volatility_threshold = min(
        MAX_VOLATILITY_THRESHOLD, 
        max(BASE_VOLATILITY_THRESHOLD, volatility * 0.8)
    )
    
    # Higher profit target and stop loss in high volatility
    profit_target = max(BASE_PROFIT_TARGET, min(BASE_PROFIT_TARGET * 2, volatility * 2.5))
    stop_loss = max(BASE_STOP_LOSS, min(BASE_STOP_LOSS * 2, volatility * 3.5))
    
    return volatility_threshold, profit_target, stop_loss

def detect_mean_reversion_signal(prices, rsi, bollinger_bands, volatility_threshold):
    """Detect mean reversion signals - effective in high volatility"""
    if len(prices) < 5 or bollinger_bands[0] is None:
        return 0
    
    sma, upper_band, lower_band = bollinger_bands
    current_price = prices[-1]
    
    # Calculate distance from bands
    upper_distance = (upper_band - current_price) / current_price if upper_band else 0
    lower_distance = (current_price - lower_band) / current_price if lower_band else 0
    
    # Mean reversion signals
    signal = 0
    
    # Oversold conditions (buy signal)
    if (rsi < 30 and current_price < lower_band) or (lower_distance < -0.005):
        signal_strength = 0.5
        if rsi < 25:  # Stronger signal for more extreme RSI
            signal_strength = 0.7
        if current_price < lower_band * 0.99:  # Stronger signal for deeper penetration
            signal_strength = 0.9
        
        signal = signal_strength  # Buy signal
    
    # Overbought conditions (sell signal)
    elif (rsi > 70 and current_price > upper_band) or (upper_distance > 0.005):
        signal_strength = 0.5
        if rsi > 75:  # Stronger signal for more extreme RSI
            signal_strength = 0.7
        if current_price > upper_band * 1.01:  # Stronger signal for deeper penetration
            signal_strength = 0.9
        
        signal = -signal_strength  # Sell signal
    
    return signal

def detect_momentum_signal(prices, trend, macd_data, volatility_threshold):
    """Detect momentum signals - good for riding volatility swings"""
    if len(prices) < 5 or macd_data[0] is None:
        return 0
    
    macd_line, signal_line, histogram = macd_data
    
    # No signal line yet
    if signal_line is None:
        return 0
    
    # Price momentum
    price_change = prices[-1] / prices[-3] - 1 if len(prices) >= 3 else 0
    
    # Momentum signals
    signal = 0
    
    # Bullish momentum
    if histogram > 0 and trend > 0 and price_change > volatility_threshold * 0.5:
        signal_strength = 0.5
        if histogram > 0.1:  # Stronger signal for more divergence
            signal_strength = 0.7
        if price_change > volatility_threshold:  # Stronger signal for bigger price moves
            signal_strength = 0.9
        
        signal = signal_strength  # Buy signal
    
    # Bearish momentum
    elif histogram < 0 and trend < 0 and price_change < -volatility_threshold * 0.5:
        signal_strength = 0.5
        if histogram < -0.1:  # Stronger signal for more divergence
            signal_strength = 0.7
        if price_change < -volatility_threshold:  # Stronger signal for bigger price moves
            signal_strength = 0.9
        
        signal = -signal_strength  # Sell signal
    
    return signal

def calculate_position_size(trader, ticker, signal_strength, volatility, 
                          price, max_position, current_position):
    """Calculate optimal position size based on multiple factors"""
    
    # Base size (inverse relationship with volatility - smaller positions in high vol)
    if volatility > 0:
        vol_adjustment = min(1.0, max(0.3, 0.01 / volatility))
    else:
        vol_adjustment = 0.5

    # Scale with signal strength
    signal_adjustment = abs(signal_strength)
    
    # Calculate dollar value per lot
    dollar_per_lot = price * 100  # Each lot is 100 shares
    
    # Maximum lots based on risk limit
    max_size_by_value = int(MAX_POSITION_VALUE / dollar_per_lot)
    
    # Calculate position size
    base_size = 3  # Start with a base size of 3 lots
    position_size = max(1, int(base_size * vol_adjustment * signal_adjustment))
    
    # Adjust size based on remaining capacity
    remaining_capacity = max_position - abs(current_position)
    if remaining_capacity <= 0:
        return 0
    
    # Apply constraints
    position_size = min(position_size, max_size_by_value, int(remaining_capacity))
    
    return max(1, position_size)  # Minimum size of 1

# ================= STRATEGY FUNCTIONS =================

def execute_mean_reversion_strategy(trader, ticker, indicators, current_position, 
                                  max_position, bid_price, ask_price, mid_price, 
                                  strategy_data, current_time):
    """Execute mean reversion strategy - primary strategy for high-VIX"""
    try:
        # Check if we have enough data
        if "bollinger_bands" not in indicators or indicators["bollinger_bands"][0] is None:
            return 0, 0
        
        # Get mean reversion signal
        signal = detect_mean_reversion_signal(
            indicators["prices"], 
            indicators["rsi"], 
            indicators["bollinger_bands"],
            indicators["volatility_threshold"]
        )
        
        order_count = 0
        position_count = 0
        
        if abs(signal) > 0.3:  # Signal strength threshold
            # Calculate position size
            position_size = calculate_position_size(
                trader, ticker, signal, indicators["volatility"],
                mid_price, max_position, current_position
            )
            
            if position_size <= 0:
                return 0, 0
                
            # Create unique trade ID
            trade_id = f"mr_{current_time.strftime('%H%M%S')}_{random.randint(100, 999)}"
                
            if signal > 0:  # Buy signal
                # Only go long if we have capacity
                if current_position < max_position:
                    # Use limit order below current bid for better entry
                    limit_price = round(bid_price * 0.997, 2)
                    
                    # Create order
                    order = shift.Order(shift.Order.Type.LIMIT_BUY, ticker, position_size, limit_price)
                    if submit_order(trader, order):
                        order_count += 1
                        
                        # Store order info
                        strategy_data["pending_orders"][order.id] = {
                            "direction": 1,
                            "size": position_size,
                            "limit_price": limit_price,
                            "entry_time": current_time,
                            "strategy": "mean_reversion",
                            "trade_id": trade_id
                        }
                        
                        logger.info(f"Mean reversion BUY signal for {ticker}: Placed limit buy order at {limit_price:.2f}, size={position_size}")
            
            elif signal < 0:  # Sell signal
                # Only go short if we have capacity
                if current_position > -max_position:
                    # Use limit order above current ask for better entry
                    limit_price = round(ask_price * 1.003, 2)
                    
                    # Create order
                    order = shift.Order(shift.Order.Type.LIMIT_SELL, ticker, position_size, limit_price)
                    if submit_order(trader, order):
                        order_count += 1
                        
                        # Store order info
                        strategy_data["pending_orders"][order.id] = {
                            "direction": -1,
                            "size": position_size, 
                            "limit_price": limit_price,
                            "entry_time": current_time,
                            "strategy": "mean_reversion",
                            "trade_id": trade_id
                        }
                        
                        logger.info(f"Mean reversion SELL signal for {ticker}: Placed limit sell order at {limit_price:.2f}, size={position_size}")
        
        return order_count, position_count
        
    except Exception as e:
        logger.error(f"Error in mean reversion strategy for {ticker}: {e}")
        traceback.print_exc()
        return 0, 0

def execute_momentum_strategy(trader, ticker, indicators, current_position, 
                            max_position, bid_price, ask_price, mid_price, 
                            strategy_data, current_time):
    """Execute momentum strategy - for riding volatility swings"""
    try:
        # Check if we have enough data
        if "macd" not in indicators or indicators["macd"][0] is None:
            return 0, 0
        
        # Get momentum signal
        signal = detect_momentum_signal(
            indicators["prices"], 
            indicators["trend"],
            indicators["macd"],
            indicators["volatility_threshold"]
        )
        
        order_count = 0
        position_count = 0
        
        if abs(signal) > 0.4:  # Signal strength threshold - higher for momentum
            # Calculate position size - smaller for momentum trades
            position_size = calculate_position_size(
                trader, ticker, signal, indicators["volatility"],
                mid_price, max_position, current_position
            )
            
            # Reduce size for momentum trades in high-VIX
            position_size = max(1, int(position_size * 0.7))
            
            if position_size <= 0:
                return 0, 0
                
            # Create unique trade ID
            trade_id = f"mom_{current_time.strftime('%H%M%S')}_{random.randint(100, 999)}"
                
            if signal > 0:  # Buy signal
                # Only go long if we have capacity
                if current_position < max_position:
                    # Use market order for momentum to ensure execution
                    order = shift.Order(shift.Order.Type.MARKET_BUY, ticker, position_size)
                    if submit_order(trader, order):
                        order_count += 1
                        position_count += 1
                        
                        # Calculate profit target and stop loss
                        target_price = mid_price * (1 + indicators["profit_target"])
                        stop_price = mid_price * (1 - indicators["stop_loss"])
                        
                        # Store in active trades
                        strategy_data["active_trades"][trade_id] = {
                            "direction": 1,
                            "entry_price": mid_price,
                            "size": position_size,
                            "target_price": target_price,
                            "stop_price": stop_price,
                            "entry_time": current_time,
                            "strategy": "momentum"
                        }
                        
                        logger.info(f"Momentum BUY signal for {ticker}: Executed market buy at ~{mid_price:.2f}, size={position_size}")
                        logger.info(f"Target: {target_price:.2f}, Stop: {stop_price:.2f}")
            
            elif signal < 0:  # Sell signal
                # Only go short if we have capacity
                if current_position > -max_position:
                    # Use market order for momentum to ensure execution
                    order = shift.Order(shift.Order.Type.MARKET_SELL, ticker, position_size)
                    if submit_order(trader, order):
                        order_count += 1
                        position_count += 1
                        
                        # Calculate profit target and stop loss
                        target_price = mid_price * (1 - indicators["profit_target"])
                        stop_price = mid_price * (1 + indicators["stop_loss"])
                        
                        # Store in active trades
                        strategy_data["active_trades"][trade_id] = {
                            "direction": -1,
                            "entry_price": mid_price,
                            "size": position_size,
                            "target_price": target_price,
                            "stop_price": stop_price,
                            "entry_time": current_time,
                            "strategy": "momentum"
                        }
                        
                        logger.info(f"Momentum SELL signal for {ticker}: Executed market sell at ~{mid_price:.2f}, size={position_size}")
                        logger.info(f"Target: {target_price:.2f}, Stop: {stop_price:.2f}")
        
        return order_count, position_count
        
    except Exception as e:
        logger.error(f"Error in momentum strategy for {ticker}: {e}")
        traceback.print_exc()
        return 0, 0

def execute_market_making(trader, ticker, indicators, current_position, 
                        max_position, bid_price, ask_price, mid_price, spread,
                        strategy_data, current_time, strategy_weights):
    """Execute market making strategy - for collecting spread and rebates"""
    try:
        # Weight for market making
        mm_weight = strategy_weights.get("market_making", 0.5)
        
        # Skip if weight too low
        if mm_weight < 0.2:
            return 0
        
        # Only run if we have enough data
        if "volatility" not in indicators:
            return 0
        
        order_count = 0
        
        # Calculate lot size (smaller for market making)
        lot_size = 1
        
        # Calculate order prices based on spread
        # In high-VIX environments, use wider spreads for safety
        mm_spread_factor = min(2.5, max(1.2, indicators["volatility"] * 50))
        
        # Base spread multiplier for orders
        base_spread_pct = min(0.003, max(0.001, spread / mid_price)) * mm_spread_factor
        
        # Adjust based on trend
        if indicators.get("trend", 0) > 0:  # Uptrend
            buy_spread_pct = base_spread_pct * 0.8  # More aggressive buys
            sell_spread_pct = base_spread_pct * 1.2  # Less aggressive sells
        elif indicators.get("trend", 0) < 0:  # Downtrend
            buy_spread_pct = base_spread_pct * 1.2  # Less aggressive buys
            sell_spread_pct = base_spread_pct * 0.8  # More aggressive sells
        else:  # No trend
            buy_spread_pct = sell_spread_pct = base_spread_pct
        
        # Calculate order prices
        buy_price = round(bid_price * (1 - buy_spread_pct), 2)
        sell_price = round(ask_price * (1 + sell_spread_pct), 2)
        
        # Place buy order if we have capacity
        if current_position < max_position:
            # Create order
            buy_order = shift.Order(shift.Order.Type.LIMIT_BUY, ticker, lot_size, buy_price)
            if submit_order(trader, buy_order):
                order_count += 1
                
                # Store order info
                strategy_data["pending_orders"][buy_order.id] = {
                    "direction": 1,
                    "size": lot_size,
                    "limit_price": buy_price,
                    "entry_time": current_time,
                    "strategy": "market_making",
                    "trade_id": f"mm_buy_{current_time.strftime('%H%M%S')}"
                }
        
        # Place sell order if we have capacity
        if current_position > -max_position:
            # Create order
            sell_order = shift.Order(shift.Order.Type.LIMIT_SELL, ticker, lot_size, sell_price)
            if submit_order(trader, sell_order):
                order_count += 1
                
                # Store order info
                strategy_data["pending_orders"][sell_order.id] = {
                    "direction": -1,
                    "size": lot_size,
                    "limit_price": sell_price,
                    "entry_time": current_time,
                    "strategy": "market_making",
                    "trade_id": f"mm_sell_{current_time.strftime('%H%M%S')}"
                }
        
        return order_count
        
    except Exception as e:
        logger.error(f"Error in market making strategy for {ticker}: {e}")
        traceback.print_exc()
        return 0

def execute_rebate_harvesting(trader, ticker, current_position, max_position, 
                             bid_price, ask_price, strategy_data, current_time,
                             strategy_weights, remaining_minutes):
    """Place limit orders to harvest rebates - ensures minimum order requirements are met"""
    try:
        # Skip if weight too low
        rebate_weight = strategy_weights.get("rebate_harvesting", 0.4)
        if rebate_weight < 0.2:
            return 0
        
        order_count = 0
        
        # Calculate how aggressive to be based on time remaining
        # More aggressive as we get closer to end of day
        aggressiveness = min(1.0, max(0.2, (240 - remaining_minutes) / 240))
        
        # Determine number of orders to place
        base_orders = 2  # Base number of orders
        extra_orders = int(aggressiveness * 8)  # Up to 8 extra orders near end of day
        orders_to_place = base_orders + extra_orders
        
        # Use minimum size
        lot_size = 1
        
        # Calculate far away prices
        # Place orders far enough to not execute under normal circumstances
        # but close enough that they might execute in extreme volatility
        buy_discount = min(0.05, max(0.01, 0.02 * aggressiveness))  # 1-5% below bid
        sell_premium = min(0.05, max(0.01, 0.02 * aggressiveness))  # 1-5% above ask
        
        # Place buy orders
        if current_position < max_position:
            for i in range(orders_to_place):
                # Vary the distance for each order
                variation = random.uniform(0.8, 1.2)
                buy_price = round(bid_price * (1 - buy_discount * variation), 2)
                
                # Create order
                buy_order = shift.Order(shift.Order.Type.LIMIT_BUY, ticker, lot_size, buy_price)
                if submit_order(trader, buy_order):
                    order_count += 1
                    
                    # Store order info
                    strategy_data["pending_orders"][buy_order.id] = {
                        "direction": 1,
                        "size": lot_size,
                        "limit_price": buy_price,
                        "entry_time": current_time,
                        "strategy": "rebate_harvesting",
                        "trade_id": f"rebate_buy_{current_time.strftime('%H%M%S')}_{i}"
                    }
        
        # Place sell orders
        if current_position > -max_position:
            for i in range(orders_to_place):
                # Vary the distance for each order
                variation = random.uniform(0.8, 1.2)
                sell_price = round(ask_price * (1 + sell_premium * variation), 2)
                
                # Create order
                sell_order = shift.Order(shift.Order.Type.LIMIT_SELL, ticker, lot_size, sell_price)
                if submit_order(trader, sell_order):
                    order_count += 1
                    
                    # Store order info
                    strategy_data["pending_orders"][sell_order.id] = {
                        "direction": -1,
                        "size": lot_size,
                        "limit_price": sell_price,
                        "entry_time": current_time,
                        "strategy": "rebate_harvesting",
                        "trade_id": f"rebate_sell_{current_time.strftime('%H%M%S')}_{i}"
                    }
        
        return order_count
        
    except Exception as e:
        logger.error(f"Error in rebate harvesting strategy for {ticker}: {e}")
        traceback.print_exc()
        return 0

def execute_failsafe_positions(trader, ticker, current_position, bid_price, ask_price, 
                             strategy_data, current_time, min_positions):
    """Failsafe to ensure we meet minimum position requirements"""
    try:
        # Count current positions opened
        position_count = strategy_data.get("position_count", 0)
        
        # If we've already met the minimum, no need for failsafe
        if position_count >= min_positions:
            return 0, 0
        
        # Calculate how many positions we're short
        positions_needed = min_positions - position_count
        
        # Don't try to open all at once
        positions_to_open = min(3, positions_needed)
        
        # Minimum size
        lot_size = 1
        
        order_count = 0
        position_count = 0
        
        # Alternate between buy and sell
        for i in range(positions_to_open):
            if i % 2 == 0:  # Buy
                order = shift.Order(shift.Order.Type.MARKET_BUY, ticker, lot_size)
                if submit_order(trader, order):
                    order_count += 1
                    position_count += 1
                    
                    # Store the trade
                    trade_id = f"failsafe_buy_{current_time.strftime('%H%M%S')}_{i}"
                    strategy_data["active_trades"][trade_id] = {
                        "direction": 1,
                        "entry_price": ask_price,  # Approximate execution price
                        "size": lot_size,
                        "entry_time": current_time,
                        "strategy": "failsafe"
                    }
                    
                    logger.info(f"Failsafe: Opened long position for {ticker} at ~{ask_price:.2f}")
            else:  # Sell
                order = shift.Order(shift.Order.Type.MARKET_SELL, ticker, lot_size)
                if submit_order(trader, order):
                    order_count += 1
                    position_count += 1
                    
                    # Store the trade
                    trade_id = f"failsafe_sell_{current_time.strftime('%H%M%S')}_{i}"
                    strategy_data["active_trades"][trade_id] = {
                        "direction": -1,
                        "entry_price": bid_price,  # Approximate execution price
                        "size": lot_size,
                        "entry_time": current_time,
                        "strategy": "failsafe"
                    }
                    
                    logger.info(f"Failsafe: Opened short position for {ticker} at ~{bid_price:.2f}")
        
        return order_count, position_count
        
    except Exception as e:
        logger.error(f"Error in failsafe positions for {ticker}: {e}")
        traceback.print_exc()
        return 0, 0

def execute_failsafe_orders(trader, ticker, current_position, bid_price, ask_price, 
                          strategy_data, current_time, min_orders, max_position):
    """Failsafe to ensure we meet minimum order requirements"""
    try:
        # Count current orders placed
        order_count = strategy_data.get("order_count", 0)
        
        # If we've already met the minimum, no need for failsafe
        if order_count >= min_orders:
            return 0
        
        # Calculate how many orders we're short
        orders_needed = min_orders - order_count
        
        # Don't try to place all at once
        orders_to_place = min(10, orders_needed)
        
        # Minimum size
        lot_size = 1
        
        placed_orders = 0
        
        # Place a mix of market and limit orders
        market_orders = min(2, orders_to_place // 5)  # 20% market orders
        limit_orders = orders_to_place - market_orders
        
        # Place market orders to ensure position count
        for i in range(market_orders):
            if i % 2 == 0 and current_position < max_position:  # Buy
                order = shift.Order(shift.Order.Type.MARKET_BUY, ticker, lot_size)
                if submit_order(trader, order):
                    placed_orders += 1
                    logger.info(f"Failsafe: Placed market buy order for {ticker}")
            elif current_position > -max_position:  # Sell
                order = shift.Order(shift.Order.Type.MARKET_SELL, ticker, lot_size)
                if submit_order(trader, order):
                    placed_orders += 1
                    logger.info(f"Failsafe: Placed market sell order for {ticker}")
        
        # Place limit orders for the rest
        for i in range(limit_orders):
            if i % 2 == 0 and current_position < max_position:  # Buy
                # Place buy order far from market (unlikely to execute)
                buy_price = round(bid_price * 0.9, 2)  # 10% below current bid
                order = shift.Order(shift.Order.Type.LIMIT_BUY, ticker, lot_size, buy_price)
                if submit_order(trader, order):
                    placed_orders += 1
                    
                    # Store order info
                    strategy_data["pending_orders"][order.id] = {
                        "direction": 1,
                        "size": lot_size,
                        "limit_price": buy_price,
                        "entry_time": current_time,
                        "strategy": "failsafe",
                        "trade_id": f"failsafe_limit_buy_{current_time.strftime('%H%M%S')}_{i}"
                    }
                    
                    logger.info(f"Failsafe: Placed limit buy order for {ticker} at {buy_price:.2f}")
            elif current_position > -max_position:  # Sell
                # Place sell order far from market (unlikely to execute)
                sell_price = round(ask_price * 1.1, 2)  # 10% above current ask
                order = shift.Order(shift.Order.Type.LIMIT_SELL, ticker, lot_size, sell_price)
                if submit_order(trader, order):
                    placed_orders += 1
                    
                    # Store order info
                    strategy_data["pending_orders"][order.id] = {
                        "direction": -1,
                        "size": lot_size,
                        "limit_price": sell_price,
                        "entry_time": current_time,
                        "strategy": "failsafe",
                        "trade_id": f"failsafe_limit_sell_{current_time.strftime('%H%M%S')}_{i}"
                    }
                    
                    logger.info(f"Failsafe: Placed limit sell order for {ticker} at {sell_price:.2f}")
        
        return placed_orders
        
    except Exception as e:
        logger.error(f"Error in failsafe orders for {ticker}: {e}")
        traceback.print_exc()
        return 0

def check_limit_order_executions(trader, ticker, strategy_data):
    """Check if any limit orders have been executed and update active trades"""
    try:
        filled_orders = []
        
        # Get all submitted orders
        for order in trader.get_submitted_orders():
            # Only look at orders for this ticker
            if order.symbol != ticker:
                continue
                
            # Check if this is a pending order that has been filled
            if order.id in strategy_data["pending_orders"] and order.status == shift.Order.Status.FILLED:
                order_info = strategy_data["pending_orders"][order.id]
                
                # Add to active trades
                trade_id = order_info["trade_id"]
                
                # Calculate profit target and stop loss based on entry price
                profit_target = order_info.get("profit_target", 0.01)
                stop_loss = order_info.get("stop_loss", 0.015)
                
                strategy_data["active_trades"][trade_id] = {
                    "direction": order_info["direction"],
                    "entry_price": order.executed_price,
                    "size": order_info["size"],
                    "target_price": order.executed_price * (1 + profit_target * order_info["direction"]),
                    "stop_price": order.executed_price * (1 - stop_loss * order_info["direction"]),
                    "entry_time": order_info["entry_time"],
                    "strategy": order_info["strategy"]
                }
                
                # Mark this order as filled
                filled_orders.append(order.id)
                
                # Increment position count
                strategy_data["position_count"] = strategy_data.get("position_count", 0) + 1
                
                logger.info(f"Limit order filled for {ticker}: Direction={order_info['direction']}, "
                           f"Price={order.executed_price:.2f}, Size={order_info['size']}, "
                           f"Strategy={order_info['strategy']}")
        
        # Remove filled orders from pending
        for order_id in filled_orders:
            if order_id in strategy_data["pending_orders"]:
                strategy_data["pending_orders"].pop(order_id)
        
        return len(filled_orders)
        
    except Exception as e:
        logger.error(f"Error checking limit order executions for {ticker}: {e}")
        traceback.print_exc()
        return 0

def manage_active_trades(trader, ticker, strategy_data, current_time, mid_price):
    """Manage active trades - take profit, stop loss, or time-based exit"""
    try:
        completed_trades = []
        order_count = 0
        position_count = 0
        
        # Check each active trade
        for trade_id, trade in strategy_data["active_trades"].items():
            # Check for expiration (maximum hold time of 15 minutes for high-VIX)
            if (current_time - trade["entry_time"]).total_seconds() > 900:  # 15 minutes
                # Close position
                if trade["direction"] > 0:  # Long
                    order = shift.Order(shift.Order.Type.MARKET_SELL, ticker, trade["size"])
                    if submit_order(trader, order):
                        order_count += 1
                        position_count += 1
                        completed_trades.append(trade_id)
                        logger.info(f"Time-based exit for {ticker} long position at ~{mid_price:.2f}")
                else:  # Short
                    order = shift.Order(shift.Order.Type.MARKET_BUY, ticker, trade["size"])
                    if submit_order(trader, order):
                        order_count += 1
                        position_count += 1
                        completed_trades.append(trade_id)
                        logger.info(f"Time-based exit for {ticker} short position at ~{mid_price:.2f}")
                
                continue
            
            # Check for exit conditions
            if trade["direction"] > 0:  # Long position
                # Take profit
                if mid_price >= trade["target_price"]:
                    order = shift.Order(shift.Order.Type.MARKET_SELL, ticker, trade["size"])
                    if submit_order(trader, order):
                        order_count += 1
                        position_count += 1
                        completed_trades.append(trade_id)
                        logger.info(f"Taking profit on {ticker} long position at ~{mid_price:.2f}")
                
                # Stop loss
                elif mid_price <= trade["stop_price"]:
                    order = shift.Order(shift.Order.Type.MARKET_SELL, ticker, trade["size"])
                    if submit_order(trader, order):
                        order_count += 1
                        position_count += 1
                        completed_trades.append(trade_id)
                        logger.info(f"Stop loss on {ticker} long position at ~{mid_price:.2f}")
            
            else:  # Short position
                # Take profit
                if mid_price <= trade["target_price"]:
                    order = shift.Order(shift.Order.Type.MARKET_BUY, ticker, trade["size"])
                    if submit_order(trader, order):
                        order_count += 1
                        position_count += 1
                        completed_trades.append(trade_id)
                        logger.info(f"Taking profit on {ticker} short position at ~{mid_price:.2f}")
                
                # Stop loss
                elif mid_price >= trade["stop_price"]:
                    order = shift.Order(shift.Order.Type.MARKET_BUY, ticker, trade["size"])
                    if submit_order(trader, order):
                        order_count += 1
                        position_count += 1
                        completed_trades.append(trade_id)
                        logger.info(f"Stop loss on {ticker} short position at ~{mid_price:.2f}")
        
        # Remove completed trades
        for trade_id in completed_trades:
            if trade_id in strategy_data["active_trades"]:
                strategy_data["active_trades"].pop(trade_id)
        
        return order_count, position_count
        
    except Exception as e:
        logger.error(f"Error managing active trades for {ticker}: {e}")
        traceback.print_exc()
        return 0, 0

def adjust_strategy_weights(strategy_data, weights):
    """Dynamically adjust strategy weights based on performance"""
    try:
        # Calculate performance metrics
        strategy_performance = strategy_data.get("strategy_performance", {})
        
        # If no performance data yet, return original weights
        if not strategy_performance:
            return weights
        
        # Calculate total absolute P&L
        total_abs_pnl = sum([abs(perf.get("pnl", 0)) for perf in strategy_performance.values()])
        
        # If no significant P&L, don't adjust weights
        if total_abs_pnl < 10:
            return weights
        
        # Calculate new weights
        new_weights = {}
        for strategy, perf in strategy_performance.items():
            pnl = perf.get("pnl", 0)
            
            if strategy in weights:
                if pnl > 0:
                    # Increase weight for profitable strategies
                    new_weights[strategy] = weights[strategy] * (1 + pnl / total_abs_pnl)
                else:
                    # Decrease weight for unprofitable strategies, but don't eliminate
                    new_weights[strategy] = weights[strategy] * max(0.4, 1 + pnl / total_abs_pnl)
        
        # Ensure all strategies have weights
        for strategy in weights:
            if strategy not in new_weights:
                new_weights[strategy] = weights[strategy]
        
        # Normalize weights
        total_weight = sum(new_weights.values())
        if total_weight > 0:
            normalize_factor = sum(weights.values()) / total_weight
            new_weights = {k: v * normalize_factor for k, v in new_weights.items()}
        
        # Ensure no weight falls below minimum threshold
        for strategy in new_weights:
            new_weights[strategy] = max(0.2, min(2.0, new_weights[strategy]))
        
        logger.info(f"Adjusted strategy weights: {new_weights}")
        return new_weights
        
    except Exception as e:
        logger.error(f"Error adjusting strategy weights: {e}")
        traceback.print_exc()
        return weights

# ================= MAIN STRATEGY FUNCTION =================

def strategy(trader: shift.Trader, ticker: str, endtime, shared_data):
    """Main trading strategy for a single ticker"""
    logger.info(f"Starting high-VIX trading strategy for {ticker}")
    
    # Initialize strategy data
    strategy_data = {
        "ticker": ticker,
        "active_trades": {},
        "pending_orders": {},
        "price_history": [],
        "bid_history": [],
        "ask_history": [],
        "spread_history": [],
        "volume_history": [],
        "order_count": 0,
        "position_count": 0,
        "strategy_performance": {
            "mean_reversion": {"pnl": 0, "count": 0},
            "momentum": {"pnl": 0, "count": 0},
            "market_making": {"pnl": 0, "count": 0},
            "rebate_harvesting": {"pnl": 0, "count": 0}
        }
    }
    
    # Get initial P&L for tracking
    try:
        initial_pl = trader.get_portfolio_item(ticker).get_realized_pl()
    except:
        initial_pl = 0
        logger.warning(f"Unable to get initial P&L for {ticker}, defaulting to 0")
    
    # Trading parameters
    max_position = MAX_POSITION_SIZE
    check_freq = 0.5  # Check market every 0.5 seconds
    
    # Timing variables
    start_time = trader.get_last_trade_time()
    last_order_refresh = start_time
    last_strategy_check = start_time
    last_exec_check = start_time
    last_weight_update = start_time
    last_log_time = start_time
    
    # Strategy weights
    strategy_weights = STRATEGY_WEIGHTS.copy()
    
    # Main trading loop
    while trader.get_last_trade_time() < endtime:
        try:
            current_time = trader.get_last_trade_time()
            
            # Calculate time remaining
            minutes_remaining = (endtime - current_time).total_seconds() / 60
            
            # Cancel existing orders periodically
            if (current_time - last_order_refresh).total_seconds() > ORDER_REFRESH_SECONDS:
                cancel_orders(trader, ticker)
                last_order_refresh = current_time
            
            # Get current market data
            bid_price, ask_price, mid_price, spread = get_current_prices(trader, ticker)
            if None in (bid_price, ask_price, mid_price, spread):
                sleep(0.5)
                continue
            
            # Get current position
            current_position, portfolio_item = get_current_position(trader, ticker)
            
            # Update price history
            strategy_data["price_history"].append(mid_price)
            strategy_data["bid_history"].append(bid_price)
            strategy_data["ask_history"].append(ask_price)
            strategy_data["spread_history"].append(spread)
            
            # Keep history at reasonable size
            if len(strategy_data["price_history"]) > PRICE_HISTORY_MAX:
                strategy_data["price_history"] = strategy_data["price_history"][-PRICE_HISTORY_MAX:]
                strategy_data["bid_history"] = strategy_data["bid_history"][-PRICE_HISTORY_MAX:]
                strategy_data["ask_history"] = strategy_data["ask_history"][-PRICE_HISTORY_MAX:]
                strategy_data["spread_history"] = strategy_data["spread_history"][-PRICE_HISTORY_MAX:]
            
            # Calculate indicators
            indicators = {}
            
            # Calculate volatility (critical for high-VIX environment)
            if len(strategy_data["price_history"]) >= VOL_WINDOW:
                volatility = calculate_volatility(strategy_data["price_history"], VOL_WINDOW)
                indicators["volatility"] = volatility
                
                # Store in shared data for cross-ticker analysis
                ticker_vol_key = f"{ticker}_volatility"
                shared_data[ticker_vol_key] = volatility
                
                # Calculate adaptive thresholds based on volatility
                volatility_threshold, profit_target, stop_loss = calculate_adaptive_thresholds(volatility)
                indicators["volatility_threshold"] = volatility_threshold
                indicators["profit_target"] = profit_target
                indicators["stop_loss"] = stop_loss
            else:
                # Default values for early trading
                indicators["volatility"] = 0.02  # Higher default for high-VIX
                indicators["volatility_threshold"] = BASE_VOLATILITY_THRESHOLD
                indicators["profit_target"] = BASE_PROFIT_TARGET
                indicators["stop_loss"] = BASE_STOP_LOSS
            
            # Calculate RSI
            if len(strategy_data["price_history"]) >= RSI_PERIOD:
                rsi = calculate_rsi(strategy_data["price_history"], RSI_PERIOD)
                indicators["rsi"] = rsi
            else:
                indicators["rsi"] = 50  # Neutral
            
            # Calculate trend
            if len(strategy_data["price_history"]) >= EMA_LONG:
                trend = detect_trend(strategy_data["price_history"], EMA_SHORT, EMA_LONG)
                indicators["trend"] = trend
            else:
                indicators["trend"] = 0  # No trend
            
            # Calculate Bollinger Bands
            if len(strategy_data["price_history"]) >= BOLLINGER_PERIOD:
                bollinger_bands = calculate_bollinger_bands(
                    strategy_data["price_history"], BOLLINGER_PERIOD, BOLLINGER_STD
                )
                indicators["bollinger_bands"] = bollinger_bands
            else:
                indicators["bollinger_bands"] = (None, None, None)
            
            # Calculate MACD
            if len(strategy_data["price_history"]) >= MACD_SLOW + MACD_SIGNAL:
                macd = calculate_macd(
                    strategy_data["price_history"], MACD_FAST, MACD_SLOW, MACD_SIGNAL
                )
                indicators["macd"] = macd
            else:
                indicators["macd"] = (None, None, None)
            
            # Store prices for indicator calculations
            indicators["prices"] = strategy_data["price_history"]
            
            # Check for limit order executions
            if (current_time - last_exec_check).total_seconds() > 5:  # Every 5 seconds
                filled_count = check_limit_order_executions(trader, ticker, strategy_data)
                last_exec_check = current_time
            
            # Manage active trades
            manage_orders, manage_positions = manage_active_trades(
                trader, ticker, strategy_data, current_time, mid_price
            )
            
            # Update counters
            strategy_data["order_count"] += manage_orders
            strategy_data["position_count"] += manage_positions
            
            # Execute strategies periodically
            if (current_time - last_strategy_check).total_seconds() > 15:  # Every 15 seconds
                # Execute mean reversion strategy (primary for high-VIX)
                mr_orders, mr_positions = execute_mean_reversion_strategy(
                    trader, ticker, indicators, current_position, max_position,
                    bid_price, ask_price, mid_price, strategy_data, current_time
                )
                
                strategy_data["order_count"] += mr_orders
                strategy_data["position_count"] += mr_positions
                
                # Execute momentum strategy
                mom_orders, mom_positions = execute_momentum_strategy(
                    trader, ticker, indicators, current_position, max_position,
                    bid_price, ask_price, mid_price, strategy_data, current_time
                )
                
                strategy_data["order_count"] += mom_orders
                strategy_data["position_count"] += mom_positions
                
                # Market making for rebate collection
                mm_orders = execute_market_making(
                    trader, ticker, indicators, current_position, max_position,
                    bid_price, ask_price, mid_price, spread, strategy_data, current_time,
                    strategy_weights
                )
                
                strategy_data["order_count"] += mm_orders
                
                last_strategy_check = current_time
            
            # Rebate harvesting - ensure minimum orders
            if minutes_remaining < RAPID_FIRE_THRESHOLD:
                # Execute more frequently as we get closer to end of day
                check_interval = max(30, min(120, minutes_remaining * 0.5))
                
                if (current_time - last_order_refresh).total_seconds() > check_interval:
                    rh_orders = execute_rebate_harvesting(
                        trader, ticker, current_position, max_position,
                        bid_price, ask_price, strategy_data, current_time,
                        strategy_weights, minutes_remaining
                    )
                    
                    strategy_data["order_count"] += rh_orders
                    last_order_refresh = current_time
            
            # Activate failsafe if needed - ensure minimum requirements
            if minutes_remaining < FAILSAFE_THRESHOLD:
                # Only check occasionally
                failsafe_check_interval = max(60, min(180, minutes_remaining))
                
                if (current_time - last_exec_check).total_seconds() > failsafe_check_interval:
                    # Check if we need to ensure minimum positions
                    if strategy_data["position_count"] < MIN_POSITIONS_TARGET * 0.7:
                        fs_orders, fs_positions = execute_failsafe_positions(
                            trader, ticker, current_position, bid_price, ask_price,
                            strategy_data, current_time, MIN_POSITIONS_TARGET
                        )
                        
                        strategy_data["order_count"] += fs_orders
                        strategy_data["position_count"] += fs_positions
                    
                    # Check if we need to ensure minimum orders
                    if strategy_data["order_count"] < MIN_ORDERS_TARGET * 0.7:
                        fs_orders = execute_failsafe_orders(
                            trader, ticker, current_position, bid_price, ask_price,
                            strategy_data, current_time, MIN_ORDERS_TARGET, max_position
                        )
                        
                        strategy_data["order_count"] += fs_orders
                    
                    last_exec_check = current_time
            
            # Update strategy weights periodically
            if (current_time - last_weight_update).total_seconds() > 300:  # Every 5 minutes
                strategy_weights = adjust_strategy_weights(strategy_data, strategy_weights)
                last_weight_update = current_time
            
            # Log status periodically
            if (current_time - last_log_time).total_seconds() > 60:  # Every minute
                # Calculate P&L change
                try:
                    current_pl = trader.get_portfolio_item(ticker).get_realized_pl()
                    pl_change = current_pl - initial_pl
                except:
                    pl_change = 0
                
                # Calculate time elapsed
                elapsed_minutes = max(0.1, (current_time - start_time).total_seconds() / 60)
                
                # Log stats
                logger.info(f"{ticker} - Position: {current_position:.1f}, "
                           f"Orders: {strategy_data['order_count']}, "
                           f"Positions: {strategy_data['position_count']}, "
                           f"P&L: ${pl_change:.2f}, "
                           f"Vol: {indicators.get('volatility', 0):.4f}, "
                           f"RSI: {indicators.get('rsi', 0):.1f}, "
                           f"Trend: {indicators.get('trend', 0)}")
                
                # Store in shared data
                shared_data[f"{ticker}_stats"] = {
                    "position": current_position,
                    "orders": strategy_data["order_count"],
                    "positions": strategy_data["position_count"],
                    "pl": pl_change
                }
                
                last_log_time = current_time
            
            # Small delay between cycles
            sleep(check_freq)
            
        except Exception as e:
            logger.error(f"Error in main loop for {ticker}: {str(e)}")
            traceback.print_exc()
            sleep(1)
    
    # End of day cleanup
    try:
        logger.info(f"End of day cleanup for {ticker}")
        
        # Cancel all orders
        cancel_orders(trader, ticker)
        
        # Close all positions
        close_positions(trader, ticker)
        
        # Calculate final P&L
        final_pl = trader.get_portfolio_item(ticker).get_realized_pl() - initial_pl
        
        logger.info(f"Strategy for {ticker} completed. "
                   f"Orders: {strategy_data['order_count']}, "
                   f"Positions: {strategy_data['position_count']}, "
                   f"P&L: ${final_pl:.2f}")
        
        # Return results
        return {
            "ticker": ticker,
            "pl": final_pl,
            "orders": strategy_data["order_count"],
            "positions": strategy_data["position_count"]
        }
        
    except Exception as e:
        logger.error(f"Error during cleanup for {ticker}: {e}")
        traceback.print_exc()
        
        # Return best-effort results
        return {
            "ticker": ticker,
            "pl": 0,
            "orders": strategy_data.get("order_count", 0),
            "positions": strategy_data.get("position_count", 0)
        }

def main(trader):
    """Main function to run the multi-ticker high-VIX trading strategy"""
    logger.info("Starting high-VIX optimized trading strategy")
    
    # Select tickers to trade - focusing on high liquidity stocks
    # For high-VIX, we want to concentrate on fewer tickers to focus attention
    # but ensure we meet minimum order requirements
    active_tickers = TICKER_PRIORITY[:4]  # Top 4 tickers
    
    # Determine trading hours
    current = trader.get_last_trade_time()
    start_time = current
    
    # End 5 minutes before market close to ensure all positions are closed
    end_time = datetime.combine(current.date(), dt.time(15, 55, 0))
    
    logger.info(f"Trading strategy starting at {start_time}, will run until {end_time}")
    
    # Track initial portfolio values
    initial_total_pl = trader.get_portfolio_summary().get_total_realized_pl()
    initial_bp = trader.get_portfolio_summary().get_total_bp()
    
    logger.info(f"Initial buying power: ${initial_bp:.2f}")
    logger.info(f"Trading the following symbols: {active_tickers}")
    
    # Shared data between threads
    shared_data = {}
    shared_data_lock = Lock()
    
    # Launch trading threads
    threads = []
    thread_objects = {}
    
    for ticker in active_tickers:
        thread = Thread(target=strategy, args=(trader, ticker, end_time, shared_data))
        thread.daemon = True
        threads.append(thread)
        thread_objects[ticker] = thread
        thread.start()
        sleep(1)  # Stagger thread starts
    
    # Wait for threads to complete
    results = {}
    for ticker, thread in thread_objects.items():
        logger.info(f"Waiting for {ticker} thread to complete")
        thread.join(timeout=60*60*7)  # 7-hour timeout
        results[ticker] = shared_data.get(f"{ticker}_stats", {
            "position": 0,
            "orders": 0,
            "positions": 0,
            "pl": 0
        })
    
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
                tickers = TICKER_PRIORITY[:4]
                for ticker in tickers:
                    cancel_orders(trader, ticker)
                    close_positions(trader, ticker)
            except Exception as cleanup_error:
                logger.error(f"Failed to perform emergency cleanup: {cleanup_error}")