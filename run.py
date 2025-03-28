import shift
from time import sleep
from datetime import datetime, timedelta
import datetime as dt
from threading import Thread
import numpy as np

def cancel_orders(trader, ticker):
    """Cancel all waiting orders for a specific ticker"""
    for order in trader.get_waiting_list():
        if order.symbol == ticker:
            trader.submit_cancellation(order)
            sleep(0.1)

def close_positions(trader, ticker):
    """Close all positions for a specific ticker"""
    print(f"Closing positions for {ticker}")
    item = trader.get_portfolio_item(ticker)

    long_shares = item.get_long_shares()
    if long_shares > 0:
        print(f"Market selling {ticker} long shares = {long_shares}")
        order = shift.Order(shift.Order.Type.MARKET_SELL, ticker, int(long_shares/100))
        trader.submit_order(order)
        sleep(0.2)

    short_shares = item.get_short_shares()
    if short_shares > 0:
        print(f"Market buying {ticker} short shares = {short_shares}")
        order = shift.Order(shift.Order.Type.MARKET_BUY, ticker, int(short_shares/100))
        trader.submit_order(order)
        sleep(0.2)

def calculate_ema(prices, period):
    """Calculate Exponential Moving Average"""
    if len(prices) < period:
        return None

    multiplier = 2 / (period + 1)
    ema = [prices[0]]

    for price in prices[1:]:
        ema.append((price - ema[-1]) * multiplier + ema[-1])

    return ema[-1]

def strategy(trader: shift.Trader, ticker: str, endtime):
    """Trading strategy for a specific ticker"""
    print(f"Starting strategy for {ticker}")
    initial_pl = trader.get_portfolio_item(ticker).get_realized_pl()

    check_freq = 0.75
    max_position = 15
    order_count_target = 250

    short_period = 10
    long_period = 25

    price_history = []
    bid_history = []
    ask_history = []
    spread_history = []
    return_history = []
    last_portfolio_value = None

    trend_direction = 0
    market_volatility = 0
    order_count = 0
    position_count = 0

    position_sizing_factor = 1.0
    
    start_time = trader.get_last_trade_time()
    last_position_check = start_time

    while trader.get_last_trade_time() < endtime:
        try:
            cancel_orders(trader, ticker)

            best_price = trader.get_best_price(ticker)
            if best_price.get_bid_price() <= 0 or best_price.get_ask_price() <= 0:
                print(f"Invalid prices for {ticker}, skipping cycle")
                sleep(0.5)
                continue

            bid_price = best_price.get_bid_price()
            ask_price = best_price.get_ask_price()
            mid_price = (bid_price + ask_price) / 2
            spread = ask_price - bid_price

            portfolio_item = trader.get_portfolio_item(ticker)
            current_position = (portfolio_item.get_long_shares() - portfolio_item.get_short_shares()) / 100
            
            current_portfolio_value = trader.get_portfolio_summary().get_total_bp() + portfolio_item.get_long_shares() * bid_price / 100 - portfolio_item.get_short_shares() * ask_price / 100
            
            if last_portfolio_value is None:
                last_portfolio_value = current_portfolio_value
            
            period_return = (current_portfolio_value - last_portfolio_value) / last_portfolio_value
            return_history.append(period_return)
            last_portfolio_value = current_portfolio_value

            price_history.append(mid_price)
            bid_history.append(bid_price)
            ask_history.append(ask_price)
            spread_history.append(spread)

            if len(price_history) > long_period * 2:
                price_history.pop(0)
                bid_history.pop(0)
                ask_history.pop(0)
                spread_history.pop(0)

            short_ema = calculate_ema(price_history, short_period)
            long_ema = calculate_ema(price_history, long_period)

            if len(spread_history) > 10:
                market_volatility = np.std(spread_history[-10:])

            if len(return_history) > 20:
                avg_return = np.mean(return_history[-20:])
                std_return = np.std(return_history[-20:])
                if std_return > 0:
                    sharpe_ratio = avg_return / std_return * np.sqrt(252)

                    if sharpe_ratio < 0.5:
                        position_sizing_factor = 0.7
                    elif sharpe_ratio > 1.5:
                        position_sizing_factor = min(position_sizing_factor * 1.05, 1.2)

            if short_ema is not None and long_ema is not None:
                if short_ema > long_ema * 1.0003:
                    trend_direction = 1
                elif short_ema < long_ema * 0.9997:
                    trend_direction = -1
                else:
                    trend_direction = 0

            actual_max_position = max(5, int(max_position * position_sizing_factor))
            
            base_size = max(1, min(4, int((spread + market_volatility) * 40)))
            
            trend_bias = 0
            if trend_direction == 1:
                trend_bias = 0.7
            elif trend_direction == -1:
                trend_bias = -0.7

            buy_size = max(1, int(base_size * (1 + trend_bias)))
            sell_size = max(1, int(base_size * (1 - trend_bias)))

            position_ratio = abs(current_position) / actual_max_position if actual_max_position > 0 else 0
            if position_ratio > 0.6:
                if current_position > 0:
                    buy_size = max(1, int(buy_size * (1 - position_ratio)))
                else:
                    sell_size = max(1, int(sell_size * (1 - position_ratio)))

            if current_position >= actual_max_position:
                buy_size = 0
            elif current_position <= -actual_max_position:
                sell_size = 0

            volatility_factor = 1 + (market_volatility * 6 if market_volatility > 0 else 0.1)
            limit_buy_offset = max(0.01, spread * 0.28 * volatility_factor)
            limit_sell_offset = max(0.01, spread * 0.28 * volatility_factor)

            if trend_direction == 1:
                limit_buy_offset *= 0.75
                limit_sell_offset *= 1.2
            elif trend_direction == -1:
                limit_buy_offset *= 1.2
                limit_sell_offset *= 0.75

            limit_buy_price = round(bid_price - limit_buy_offset, 2)
            limit_sell_price = round(ask_price + limit_sell_offset, 2)

            if buy_size > 0:
                buy_order = shift.Order(shift.Order.Type.LIMIT_BUY, ticker, buy_size, limit_buy_price)
                trader.submit_order(buy_order)
                order_count += 1

            if sell_size > 0:
                sell_order = shift.Order(shift.Order.Type.LIMIT_SELL, ticker, sell_size, limit_sell_price)
                trader.submit_order(sell_order)
                order_count += 1

            current_time = trader.get_last_trade_time()
            time_elapsed = (current_time - start_time).total_seconds() / 60.0

            required_position_rate = 12 / (6.5 * 60)
            actual_position_rate = position_count / max(1, time_elapsed)
            
            market_order_freq = 15 if actual_position_rate >= required_position_rate else 10
            
            if (endtime - current_time).total_seconds() < 1800 and order_count < order_count_target:
                market_order_freq = 5
            
            if order_count % market_order_freq == 0:
                market_size = 1
                
                if trend_direction == 1 and current_position < actual_max_position * 0.8:
                    market_order = shift.Order(shift.Order.Type.MARKET_BUY, ticker, market_size)
                    trader.submit_order(market_order)
                    position_count += 1
                    order_count += 1
                elif trend_direction == -1 and current_position > -actual_max_position * 0.8:
                    market_order = shift.Order(shift.Order.Type.MARKET_SELL, ticker, market_size)
                    trader.submit_order(market_order)
                    position_count += 1
                    order_count += 1
                elif actual_position_rate < required_position_rate:
                    if abs(current_position + market_size) < abs(current_position - market_size):
                        market_order = shift.Order(shift.Order.Type.MARKET_BUY, ticker, market_size)
                    else:
                        market_order = shift.Order(shift.Order.Type.MARKET_SELL, ticker, market_size)
                    trader.submit_order(market_order)
                    position_count += 1
                    order_count += 1

            if (current_time - last_position_check).total_seconds() > 60:
                if (trend_direction == 1 and current_position < -3) or (trend_direction == -1 and current_position > 3):
                    reduction_size = min(2, abs(int(current_position)) // 2)
                    if reduction_size > 0:
                        if trend_direction == 1:
                            market_order = shift.Order(shift.Order.Type.MARKET_BUY, ticker, reduction_size)
                            trader.submit_order(market_order)
                            position_count += 1
                            order_count += 1
                        else:
                            market_order = shift.Order(shift.Order.Type.MARKET_SELL, ticker, reduction_size)
                            trader.submit_order(market_order)
                            position_count += 1
                            order_count += 1
                last_position_check = current_time

            if order_count % 30 == 0:
                elapsed_minutes = (trader.get_last_trade_time() - start_time).total_seconds() / 60
                if elapsed_minutes > 0:
                    print(f"{ticker} - Position: {current_position:.1f}, Trend: {trend_direction}, "
                        f"Orders: {order_count}, Positions: {position_count}, "
                        f"Orders/min: {order_count/max(1, elapsed_minutes):.1f}")

            sleep(check_freq)

        except Exception as e:
            print(f"Error in strategy for {ticker}: {e}")
            sleep(1)

    cancel_orders(trader, ticker)
    close_positions(trader, ticker)

    final_pl = trader.get_portfolio_item(ticker).get_realized_pl() - initial_pl
    print(f"Strategy for {ticker} completed. Total P&L: ${final_pl:.2f}")
    return final_pl

def main(trader):
    """Main function to initialize and run trading strategies"""

    ticker_priorities = [
        "AAPL",
        "MSFT",
        "JPM",
        "XOM",
        "V",
        "WMT",
        "JNJ",
        "PG"
    ]

    current = trader.get_last_trade_time()
    start_time = current

    end_time = datetime.combine(current.date(), dt.time(15, 55, 0))

    print(f"Strategy starting at {start_time}, will run until {end_time}")

    initial_total_pl = trader.get_portfolio_summary().get_total_realized_pl()
    initial_bp = trader.get_portfolio_summary().get_total_bp()
    
    print(f"Initial buying power: ${initial_bp:.2f}")

    active_tickers = ticker_priorities[:5]
    print(f"Trading the following symbols: {active_tickers}")

    threads = []
    
    for ticker in active_tickers:
        thread = Thread(target=strategy, args=(trader, ticker, end_time))
        threads.append(thread)
        thread.start()
        sleep(1)

    for thread in threads:
        thread.join(timeout=60*60*7)

    for ticker in active_tickers:
        try:
            cancel_orders(trader, ticker)
            close_positions(trader, ticker)
        except Exception as e:
            print(f"Error during final cleanup for {ticker}: {e}")

    final_bp = trader.get_portfolio_summary().get_total_bp()
    final_pl = trader.get_portfolio_summary().get_total_realized_pl() - initial_total_pl

    print("---Strategy Complete---")
    print(f"Final buying power: ${final_bp:.2f}")
    print(f"Total profit/loss: ${final_pl:.2f}")
    print(f"Return on capital: {(final_pl / initial_bp * 100):.2f}%")

if __name__ == '__main__':
    with shift.Trader("dolla-dolla-bills-yall") as trader:
        try:
            trader.connect("initiator.cfg", "Zxz7Qxa9")
            sleep(1)
            trader.sub_all_order_book()
            sleep(1)

            main(trader)
        except Exception as e:
            print(f"Fatal error in main program: {e}")
            try:
                tickers = ["AAPL", "MSFT", "JPM", "XOM", "V"]
                for ticker in tickers:
                    cancel_orders(trader, ticker)
                    close_positions(trader, ticker)
            except:
                print("Failed to perform emergency cleanup")
