#!/usr/bin/env python3
"""
Enhanced Publisher-Specific Backtesting with Detailed Trade Logging
"""

import argparse
import pandas as pd
import numpy as np
import uuid
from datetime import datetime, timedelta
from postgres_data_manager import PostgresDataManager

def simulate_trade_execution(news_row, initial_capital, position_size_pct=0.01):
    """
    Simulate a detailed trade execution based on news and price movement
    
    Args:
        news_row: Row containing news and price data
        initial_capital: Current capital amount
        position_size_pct: Percentage of capital to risk per trade
        
    Returns:
        Dictionary with detailed trade information
    """
    
    # Extract data
    ticker = news_row['ticker']
    published_date = news_row['published_date']
    direction_prediction = news_row['predicted_side']  # 'UP' or 'DOWN'
    actual_direction = news_row['actual_side']
    price_change_pct = news_row['price_change_percentage']
    link = news_row.get('link', '')
    event = news_row.get('event', 'unknown')
    begin_price = news_row.get('begin_price', 100.0)
    end_price = news_row.get('end_price', 100.0)
    
    # Simulate entry price (use actual begin price)
    entry_price = float(begin_price) if begin_price else 100.0
    
    # Calculate position size
    risk_amount = initial_capital * position_size_pct
    
    # Determine trade direction
    if direction_prediction == 'UP':
        direction = 'LONG'
        shares = int(risk_amount / entry_price)
        target_price = entry_price * 1.01  # 1% target
        stop_price = entry_price * 0.995   # 0.5% stop
    else:  # 'DOWN'
        direction = 'SHORT'
        shares = int(risk_amount / entry_price)
        target_price = entry_price * 0.99   # 1% target (for short)
        stop_price = entry_price * 1.005    # 0.5% stop (for short)
    
    # Simulate exit based on actual price movement
    actual_exit_price = float(end_price) if end_price else entry_price
    
    if direction == 'LONG':
        if price_change_pct >= 1.0:  # Hit target
            exit_price = target_price
            hit_target = True
            hit_stop = False
        elif price_change_pct <= -0.5:  # Hit stop
            exit_price = stop_price
            hit_target = False
            hit_stop = True
        else:  # Exit at actual end price
            exit_price = actual_exit_price
            hit_target = False
            hit_stop = False
    else:  # SHORT
        if price_change_pct <= -1.0:  # Hit target (price went down)
            exit_price = target_price
            hit_target = True
            hit_stop = False
        elif price_change_pct >= 0.5:  # Hit stop (price went up)
            exit_price = stop_price
            hit_target = False
            hit_stop = True
        else:  # Exit at actual end price
            exit_price = actual_exit_price
            hit_target = False
            hit_stop = False
    
    # Calculate P&L
    if direction == 'LONG':
        pnl = shares * (exit_price - entry_price)
    else:  # SHORT
        pnl = shares * (entry_price - exit_price)
    
    pnl_pct = (pnl / risk_amount) * 100 if risk_amount > 0 else 0
    capital_after = initial_capital + pnl
    
    # Determine market session
    hour = published_date.hour if hasattr(published_date, 'hour') else 9
    if hour < 9:
        market = 'pre_market'
    elif hour >= 16:
        market = 'after_market'
    else:
        market = 'regular_market'
    
    # Create trade record
    trade_data = {
        'published_date': published_date,
        'market': market,
        'entry_time': published_date + timedelta(hours=1),  # Simulate 1 hour delay
        'exit_time': published_date + timedelta(hours=6),   # Simulate 6 hour holding
        'ticker': ticker,
        'direction': direction,
        'shares': shares,
        'entry_price': round(entry_price, 6),
        'exit_price': round(exit_price, 6),
        'target_price': round(target_price, 6),
        'stop_price': round(stop_price, 6),
        'hit_target': hit_target,
        'hit_stop': hit_stop,
        'pnl': round(pnl, 6),
        'pnl_pct': round(pnl_pct, 6),
        'capital_after': round(capital_after, 6),
        'news_event': event,
        'link': link
    }
    
    return trade_data, capital_after

def run_detailed_backtest(dm, publisher, initial_capital, limit=None, log_trades=True):
    """
    Run detailed backtest with individual trade logging
    
    Args:
        dm: PostgreSQL data manager instance
        publisher: Publisher to filter by ('all' for all publishers)
        initial_capital: Starting capital
        limit: Maximum number of trades to process
        log_trades: Whether to log individual trades to database
        
    Returns:
        Dictionary with backtest results and trade list
    """
    
    print(f"🚀 Running detailed backtest for publisher: {publisher}")
    
    # Generate unique run ID
    run_id = str(uuid.uuid4())[:8]
    run_date = datetime.now()
    
    # Get data
    if publisher == 'all':
        data = dm.get_news_with_price_moves(limit=limit)
        model_name = f"detailed_backtest_all_publishers"
    else:
        data = dm.get_news_with_price_moves(publisher=publisher, limit=limit)
        model_name = f"detailed_backtest_{publisher}"
    
    if data.empty:
        print(f"❌ No data found for publisher: {publisher}")
        return None
    
    print(f"📊 Processing {len(data)} news articles with price data")
    
    # Initialize tracking variables
    current_capital = initial_capital
    trades = []
    
    # Process each news item
    for idx, row in data.iterrows():
        # Simulate trade
        trade_data, new_capital = simulate_trade_execution(row, current_capital)
        
        # Add run metadata
        trade_data['runid'] = run_id
        trade_data['rundate'] = run_date
        
        # Log trade to database if enabled
        if log_trades:
            success = dm.log_trade(trade_data)
            if not success:
                print(f"⚠️  Failed to log trade {idx + 1}")
        
        # Add to trades list
        trades.append(trade_data)
        current_capital = new_capital
        
        # Progress update
        if (idx + 1) % 50 == 0:
            print(f"   Processed {idx + 1}/{len(data)} trades, Capital: ${current_capital:,.2f}")
    
    # Calculate summary statistics
    trades_df = pd.DataFrame(trades)
    
    total_trades = len(trades)
    winning_trades = len(trades_df[trades_df['pnl'] > 0])
    losing_trades = len(trades_df[trades_df['pnl'] < 0])
    total_pnl = trades_df['pnl'].sum()
    return_pct = ((current_capital - initial_capital) / initial_capital) * 100
    win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
    
    target_hits = len(trades_df[trades_df['hit_target'] == True])
    stop_hits = len(trades_df[trades_df['hit_stop'] == True])
    
    # Create results summary
    results = {
        'run_id': run_id,
        'model_name': model_name,
        'publisher': publisher,
        'start_date': data['published_date'].min(),
        'end_date': data['published_date'].max(),
        'initial_capital': initial_capital,
        'final_capital': current_capital,
        'total_pnl': total_pnl,
        'return_percent': return_pct,
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'losing_trades': losing_trades,
        'win_rate_percent': win_rate,
        'target_hits': target_hits,
        'stop_hits': stop_hits,
        'news_articles_used': len(data),
        'price_moves_used': len(data),
        'notes': f'Detailed backtest with trade logging for {publisher}',
        'trades': trades
    }
    
    # Save backtest summary
    dm.save_backtest_results(results, run_id)
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Run detailed publisher-specific backtests with trade logging')
    parser.add_argument('--capital', type=float, default=100000, help='Initial capital (default: 100000)')
    parser.add_argument('--limit', type=int, help='Limit number of trades to process')
    parser.add_argument('--publishers', nargs='+', default=['globenewswire_biotech', 'all'], 
                       help='Publishers to test (default: globenewswire_biotech all)')
    parser.add_argument('--output', help='CSV file to save detailed results')
    parser.add_argument('--no-db-logging', action='store_true', help='Disable database trade logging')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    print("🧪 Detailed Publisher Backtesting with Trade Logging")
    print("=" * 60)
    print(f"💰 Initial Capital: ${args.capital:,.2f}")
    print(f"📊 Publishers: {', '.join(args.publishers)}")
    if args.limit:
        print(f"🔢 Trade Limit: {args.limit}")
    print(f"💾 Database Logging: {'Disabled' if args.no_db_logging else 'Enabled'}")
    print()
    
    try:
        # Initialize data manager
        dm = PostgresDataManager()
        
        all_results = []
        all_trades = []
        
        # Run backtests for each publisher
        for publisher in args.publishers:
            print(f"\n{'='*20} {publisher.upper()} {'='*20}")
            
            results = run_detailed_backtest(
                dm, 
                publisher, 
                args.capital, 
                limit=args.limit,
                log_trades=not args.no_db_logging
            )
            
            if results:
                all_results.append(results)
                all_trades.extend(results['trades'])
                
                # Print summary
                print(f"\n📈 Results for {publisher}:")
                print(f"   Run ID: {results['run_id']}")
                print(f"   Total Trades: {results['total_trades']}")
                print(f"   Winning Trades: {results['winning_trades']}")
                print(f"   Losing Trades: {results['losing_trades']}")
                print(f"   Win Rate: {results['win_rate_percent']:.2f}%")
                print(f"   Target Hits: {results['target_hits']}")
                print(f"   Stop Hits: {results['stop_hits']}")
                print(f"   Total P&L: ${results['total_pnl']:,.2f}")
                print(f"   Return: {results['return_percent']:.2f}%")
                print(f"   Final Capital: ${results['final_capital']:,.2f}")
        
        # Save detailed trades to CSV if requested
        if args.output and all_trades:
            trades_df = pd.DataFrame(all_trades)
            trades_df.to_csv(args.output, index=False)
            print(f"\n💾 Detailed trades saved to: {args.output}")
            print(f"   Total trades exported: {len(trades_df)}")
        
        # Print overall summary
        if all_results:
            print(f"\n{'='*60}")
            print("📊 OVERALL SUMMARY")
            print(f"{'='*60}")
            
            for result in all_results:
                print(f"{result['publisher']:20} | "
                      f"Trades: {result['total_trades']:4d} | "
                      f"Win Rate: {result['win_rate_percent']:6.2f}% | "
                      f"Return: {result['return_percent']:7.2f}% | "
                      f"Run ID: {result['run_id']}")
        
        print(f"\n✅ Detailed backtesting completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during backtesting: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())

