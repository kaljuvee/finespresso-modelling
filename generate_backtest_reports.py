#!/usr/bin/env python3
"""
Comprehensive Backtest Reporting Script
Generates detailed trades, summaries, and exports to organized CSV files
"""

import argparse
import pandas as pd
import numpy as np
import uuid
import os
from datetime import datetime, timedelta
from postgres_data_manager import PostgresDataManager

def simulate_trade_execution(news_row, initial_capital, position_size_pct=0.01):
    """
    Simulate a detailed trade execution based on news and price movement
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

def run_comprehensive_backtest(dm, publisher, initial_capital, limit=None, save_to_db=True):
    """
    Run comprehensive backtest with detailed logging and summary generation
    """
    
    print(f"🚀 Running comprehensive backtest for publisher: {publisher}")
    
    # Generate unique run ID and timestamp
    run_id = str(uuid.uuid4())[:8]
    run_date = datetime.now()
    timestamp_str = run_date.strftime("%Y%m%d_%H%M%S")
    
    # Get data
    if publisher == 'all':
        data = dm.get_news_with_price_moves(limit=limit)
        model_name = f"comprehensive_backtest_all_publishers"
    else:
        data = dm.get_news_with_price_moves(publisher=publisher, limit=limit)
        model_name = f"comprehensive_backtest_{publisher}"
    
    if data.empty:
        print(f"❌ No data found for publisher: {publisher}")
        return None
    
    print(f"📊 Processing {len(data)} news articles with price data")
    
    # Initialize tracking variables
    current_capital = initial_capital
    trades = []
    capital_history = [initial_capital]
    
    # Process each news item
    for idx, row in data.iterrows():
        # Simulate trade
        trade_data, new_capital = simulate_trade_execution(row, current_capital)
        
        # Add run metadata
        trade_data['runid'] = run_id
        trade_data['rundate'] = run_date
        
        # Log trade to database if enabled
        if save_to_db:
            try:
                # Use psycopg2 directly for trade logging since the data manager has issues
                import psycopg2
                DATABASE_URL = 'postgresql://finespresso_db_user:XZ0o6UkxcV0poBcLDQf6RGXwEfWmBlnb@dpg-ctj7u2lumphs73f8t9qg-a.frankfurt-postgres.render.com/finespresso_db'
                conn = psycopg2.connect(DATABASE_URL)
                cursor = conn.cursor()
                
                query = """
                INSERT INTO backtest_trades (
                    published_date, market, entry_time, exit_time, ticker, direction,
                    shares, entry_price, exit_price, target_price, stop_price,
                    hit_target, hit_stop, pnl, pnl_pct, capital_after,
                    news_event, link, runid, rundate
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """
                
                values = (
                    trade_data['published_date'], trade_data['market'], trade_data['entry_time'],
                    trade_data['exit_time'], trade_data['ticker'], trade_data['direction'],
                    trade_data['shares'], trade_data['entry_price'], trade_data['exit_price'],
                    trade_data['target_price'], trade_data['stop_price'], trade_data['hit_target'],
                    trade_data['hit_stop'], trade_data['pnl'], trade_data['pnl_pct'],
                    trade_data['capital_after'], trade_data['news_event'], trade_data['link'],
                    trade_data['runid'], trade_data['rundate']
                )
                
                cursor.execute(query, values)
                conn.commit()
                cursor.close()
                conn.close()
                
            except Exception as e:
                print(f"⚠️  Failed to log trade {idx + 1}: {e}")
        
        # Add to trades list
        trades.append(trade_data)
        current_capital = new_capital
        capital_history.append(current_capital)
        
        # Progress update
        if (idx + 1) % 25 == 0:
            print(f"   Processed {idx + 1}/{len(data)} trades, Capital: ${current_capital:,.2f}")
    
    # Calculate comprehensive statistics
    trades_df = pd.DataFrame(trades)
    
    total_trades = len(trades)
    winning_trades = len(trades_df[trades_df['pnl'] > 0])
    losing_trades = len(trades_df[trades_df['pnl'] < 0])
    total_pnl = trades_df['pnl'].sum()
    return_pct = ((current_capital - initial_capital) / initial_capital) * 100
    win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
    
    target_hits = len(trades_df[trades_df['hit_target'] == True])
    stop_hits = len(trades_df[trades_df['hit_stop'] == True])
    
    # Calculate additional metrics
    capital_series = pd.Series(capital_history)
    peak_capital = capital_series.cummax()
    drawdown = (capital_series - peak_capital) / peak_capital * 100
    max_drawdown = drawdown.min()
    
    avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
    avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0
    profit_factor = abs(avg_win * winning_trades / (avg_loss * losing_trades)) if losing_trades > 0 and avg_loss != 0 else float('inf')
    
    # Create comprehensive results summary
    results = {
        'run_id': run_id,
        'timestamp': run_date,
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
        'max_drawdown': max_drawdown,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'news_articles_used': len(data),
        'price_moves_used': len(data),
        'notes': f'Comprehensive backtest with detailed logging for {publisher}',
        'trades': trades,
        'capital_history': capital_history
    }
    
    # Save backtest summary to database
    if save_to_db:
        try:
            # Save to backtest_summary table using direct SQL
            import psycopg2
            conn = psycopg2.connect(DATABASE_URL)
            cursor = conn.cursor()
            
            summary_query = """
            INSERT INTO backtest_summary (
                run_id, timestamp, model_name, start_date, end_date,
                initial_capital, final_capital, total_pnl, return_percent,
                total_trades, winning_trades, losing_trades, win_rate_percent,
                max_drawdown, news_articles_used, price_moves_used, notes
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            summary_values = (
                run_id, run_date, model_name, results['start_date'], results['end_date'],
                initial_capital, current_capital, total_pnl, return_pct,
                total_trades, winning_trades, losing_trades, win_rate,
                max_drawdown, len(data), len(data), results['notes']
            )
            
            cursor.execute(summary_query, summary_values)
            conn.commit()
            cursor.close()
            conn.close()
            print(f"✅ Saved backtest summary to database (run_id: {run_id})")
            
        except Exception as e:
            print(f"⚠️  Failed to save backtest summary: {e}")
    
    return results

def export_reports(results_list, timestamp_str):
    """
    Export comprehensive reports to organized CSV files
    """
    
    if not results_list:
        print("❌ No results to export")
        return
    
    # Create reports directory
    os.makedirs("reports/backtesting", exist_ok=True)
    
    # Export detailed trades
    all_trades = []
    for result in results_list:
        all_trades.extend(result['trades'])
    
    if all_trades:
        trades_df = pd.DataFrame(all_trades)
        trades_file = f"reports/backtesting/backtest_trades_{timestamp_str}.csv"
        trades_df.to_csv(trades_file, index=False)
        print(f"📄 Exported detailed trades: {trades_file}")
    
    # Export backtest summary
    summary_data = []
    for result in results_list:
        summary_row = {
            'run_id': result['run_id'],
            'timestamp': result['timestamp'],
            'model_name': result['model_name'],
            'publisher': result['publisher'],
            'start_date': result['start_date'],
            'end_date': result['end_date'],
            'initial_capital': result['initial_capital'],
            'final_capital': result['final_capital'],
            'total_pnl': result['total_pnl'],
            'return_percent': result['return_percent'],
            'total_trades': result['total_trades'],
            'winning_trades': result['winning_trades'],
            'losing_trades': result['losing_trades'],
            'win_rate_percent': result['win_rate_percent'],
            'target_hits': result['target_hits'],
            'stop_hits': result['stop_hits'],
            'max_drawdown': result['max_drawdown'],
            'avg_win': result['avg_win'],
            'avg_loss': result['avg_loss'],
            'profit_factor': result['profit_factor'],
            'news_articles_used': result['news_articles_used'],
            'price_moves_used': result['price_moves_used'],
            'notes': result['notes']
        }
        summary_data.append(summary_row)
    
    summary_df = pd.DataFrame(summary_data)
    summary_file = f"reports/backtesting/backtest_summary_{timestamp_str}.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"📊 Exported backtest summary: {summary_file}")
    
    # Export performance comparison
    comparison_data = []
    for result in results_list:
        comparison_row = {
            'Publisher': result['publisher'],
            'Run_ID': result['run_id'],
            'Total_Trades': result['total_trades'],
            'Win_Rate_Percent': result['win_rate_percent'],
            'Return_Percent': result['return_percent'],
            'Total_PnL_USD': result['total_pnl'],
            'Final_Capital_USD': result['final_capital'],
            'Max_Drawdown_Percent': result['max_drawdown'],
            'Target_Hits': result['target_hits'],
            'Stop_Hits': result['stop_hits'],
            'Profit_Factor': result['profit_factor'],
            'Avg_Win_USD': result['avg_win'],
            'Avg_Loss_USD': result['avg_loss']
        }
        comparison_data.append(comparison_row)
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_file = f"reports/backtesting/publisher_comparison_{timestamp_str}.csv"
    comparison_df.to_csv(comparison_file, index=False)
    print(f"📈 Exported publisher comparison: {comparison_file}")
    
    return {
        'trades_file': trades_file,
        'summary_file': summary_file,
        'comparison_file': comparison_file
    }

def main():
    parser = argparse.ArgumentParser(description='Generate comprehensive backtest reports')
    parser.add_argument('--capital', type=float, default=100000, help='Initial capital (default: 100000)')
    parser.add_argument('--limit', type=int, default=100, help='Limit number of trades to process (default: 100)')
    parser.add_argument('--publishers', nargs='+', default=['globenewswire_biotech', 'all'], 
                       help='Publishers to test (default: globenewswire_biotech all)')
    parser.add_argument('--no-db', action='store_true', help='Disable database saving')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("📊 Comprehensive Backtest Report Generation")
    print("=" * 60)
    print(f"💰 Initial Capital: ${args.capital:,.2f}")
    print(f"📊 Publishers: {', '.join(args.publishers)}")
    print(f"🔢 Trade Limit: {args.limit}")
    print(f"💾 Database Saving: {'Disabled' if args.no_db else 'Enabled'}")
    print(f"📅 Timestamp: {timestamp_str}")
    print()
    
    try:
        # Initialize data manager
        dm = PostgresDataManager()
        
        all_results = []
        
        # Run backtests for each publisher
        for publisher in args.publishers:
            print(f"\n{'='*20} {publisher.upper()} {'='*20}")
            
            results = run_comprehensive_backtest(
                dm, 
                publisher, 
                args.capital, 
                limit=args.limit,
                save_to_db=not args.no_db
            )
            
            if results:
                all_results.append(results)
                
                # Print detailed summary
                print(f"\n📈 Results for {publisher}:")
                print(f"   Run ID: {results['run_id']}")
                print(f"   Total Trades: {results['total_trades']}")
                print(f"   Winning Trades: {results['winning_trades']} ({results['win_rate_percent']:.1f}%)")
                print(f"   Losing Trades: {results['losing_trades']}")
                print(f"   Target Hits: {results['target_hits']}")
                print(f"   Stop Hits: {results['stop_hits']}")
                print(f"   Total P&L: ${results['total_pnl']:,.2f}")
                print(f"   Return: {results['return_percent']:.2f}%")
                print(f"   Final Capital: ${results['final_capital']:,.2f}")
                print(f"   Max Drawdown: {results['max_drawdown']:.2f}%")
                print(f"   Profit Factor: {results['profit_factor']:.2f}")
                print(f"   Avg Win: ${results['avg_win']:.2f}")
                print(f"   Avg Loss: ${results['avg_loss']:.2f}")
        
        # Export comprehensive reports
        if all_results:
            print(f"\n{'='*60}")
            print("📄 EXPORTING REPORTS")
            print(f"{'='*60}")
            
            files = export_reports(all_results, timestamp_str)
            
            print(f"\n📁 Reports saved to:")
            for file_type, file_path in files.items():
                print(f"   {file_type}: {file_path}")
        
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
                      f"Drawdown: {result['max_drawdown']:6.2f}% | "
                      f"Run ID: {result['run_id']}")
        
        print(f"\n✅ Comprehensive backtest reporting completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during backtesting: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())

