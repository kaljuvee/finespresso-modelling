#!/usr/bin/env python3
"""
Command-line script for running realistic backtests

Usage:
    python run_realistic_backtest.py --capital 100000 --limit 500
    python run_realistic_backtest.py --publisher globenewswire_biotech --output results.csv
    python run_realistic_backtest.py --start-date 2024-01-01 --end-date 2024-12-31
"""

import argparse
import pandas as pd
from datetime import datetime
import os

from realistic_backtester import RealisticBacktester
from postgres_data_manager import PostgresDataManager

def main():
    parser = argparse.ArgumentParser(description='Run realistic backtesting with ML predictions')
    
    # Basic parameters
    parser.add_argument('--capital', type=float, default=100000,
                       help='Initial capital amount (default: 100000)')
    parser.add_argument('--limit', type=int, default=300,
                       help='Number of news items to process (default: 300)')
    parser.add_argument('--publisher', type=str, default='all',
                       help='Publisher filter (default: all)')
    
    # Date range
    parser.add_argument('--start-date', type=str,
                       help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str,
                       help='End date (YYYY-MM-DD)')
    
    # Risk management
    parser.add_argument('--stop-loss', type=float, default=0.02,
                       help='Stop loss percentage (default: 0.02 = 2%)')
    parser.add_argument('--take-profit', type=float, default=0.04,
                       help='Take profit percentage (default: 0.04 = 4%)')
    parser.add_argument('--max-position', type=float, default=0.05,
                       help='Max position size as fraction of capital (default: 0.05 = 5%)')
    parser.add_argument('--min-confidence', type=float, default=0.6,
                       help='Minimum ML confidence threshold (default: 0.6 = 60%)')
    
    # Output
    parser.add_argument('--output', type=str,
                       help='Output CSV file for detailed trades')
    parser.add_argument('--summary-output', type=str,
                       help='Output CSV file for summary results')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    print("🚀 REALISTIC BACKTESTING ENGINE")
    print("=" * 50)
    print(f"💰 Initial Capital: ${args.capital:,.2f}")
    print(f"📊 Data Limit: {args.limit} news items")
    print(f"📰 Publisher: {args.publisher}")
    print(f"🛡️  Stop Loss: {args.stop_loss*100:.1f}%")
    print(f"🎯 Take Profit: {args.take_profit*100:.1f}%")
    print(f"📈 Max Position: {args.max_position*100:.1f}%")
    print(f"🤖 Min Confidence: {args.min_confidence*100:.1f}%")
    print()
    
    # Initialize data manager
    print("🔌 Connecting to PostgreSQL database...")
    dm = PostgresDataManager()
    
    # Get data
    print(f"📊 Fetching news data...")
    if args.publisher == 'all':
        news_data = dm.get_news_with_price_moves(limit=args.limit)
    else:
        news_data = dm.get_news_with_price_moves(
            limit=args.limit, 
            publisher=args.publisher
        )
    
    print(f"✅ Retrieved {len(news_data)} news items")
    
    if len(news_data) < 100:
        print("⚠️  Warning: Less than 100 samples may result in poor ML model performance")
    
    # Initialize backtester
    print("🤖 Initializing realistic backtester...")
    backtester = RealisticBacktester(
        initial_capital=args.capital,
        data_manager=dm  # Pass data manager for database saving
    )
    
    # Configure risk management
    backtester.stop_loss_pct = args.stop_loss
    backtester.take_profit_pct = args.take_profit
    backtester.max_position_size = args.max_position
    backtester.min_confidence = args.min_confidence
    
    # Run backtest
    print("🔄 Running realistic backtest...")
    results = backtester.run_backtest(
        news_data, 
        start_date=args.start_date,
        end_date=args.end_date
    )
    
    # Display results
    print("\n📊 BACKTEST RESULTS SUMMARY")
    print("=" * 40)
    print(f"💰 Initial Capital: ${results['initial_capital']:,.2f}")
    print(f"💰 Final Capital: ${results['final_capital']:,.2f}")
    print(f"📈 Total P&L: ${results['total_pnl']:,.2f}")
    print(f"📊 Total Return: {results['total_return_pct']:.2f}%")
    print(f"🎯 Total Trades: {results['total_trades']}")
    print(f"✅ Winning Trades: {results['winning_trades']}")
    print(f"❌ Losing Trades: {results['losing_trades']}")
    print(f"🏆 Win Rate: {results['win_rate_pct']:.1f}%")
    print(f"⚡ Profit Factor: {results['profit_factor']:.2f}")
    print(f"📊 Execution Rate: {results['successful_trades']/results['news_items_processed']*100:.1f}%")
    
    # Save detailed trades
    if results['trade_log']:
        trades_df = pd.DataFrame(results['trade_log'])
        
        # Default output file
        if not args.output:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            args.output = f"reports/backtesting/realistic_backtest_{timestamp}.csv"
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        
        # Save trades
        trades_df.to_csv(args.output, index=False)
        print(f"\n💾 Detailed trades saved to: {args.output}")
        
        # Save summary
        if args.summary_output:
            summary_data = {
                'metric': [
                    'Initial Capital', 'Final Capital', 'Total P&L', 'Return %',
                    'Total Trades', 'Winning Trades', 'Losing Trades', 'Win Rate %',
                    'Profit Factor', 'Execution Rate %', 'News Items Processed'
                ],
                'value': [
                    results['initial_capital'], results['final_capital'],
                    results['total_pnl'], results['total_return_pct'],
                    results['total_trades'], results['winning_trades'],
                    results['losing_trades'], results['win_rate_pct'],
                    results['profit_factor'], 
                    results['successful_trades']/results['news_items_processed']*100,
                    results['news_items_processed']
                ]
            }
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_csv(args.summary_output, index=False)
            print(f"📊 Summary saved to: {args.summary_output}")
        
        # Show sample trades if verbose
        if args.verbose and len(trades_df) > 0:
            print("\n💼 SAMPLE TRADES:")
            print("-" * 50)
            for i, trade in enumerate(trades_df.head(5).to_dict('records')):
                print(f"Trade {i+1}: {trade['ticker']} {trade['direction']}")
                print(f"  Entry: ${trade['entry_price']:.2f} @ {trade['entry_time']}")
                print(f"  Exit: ${trade['exit_price']:.2f} @ {trade['exit_time']}")
                print(f"  P&L: ${trade['pnl']:.2f} ({trade['pnl_pct']:.1f}%)")
                print(f"  Reason: {trade['exit_reason']}")
                print(f"  Confidence: {trade['confidence']:.2f}")
                print()
    
    else:
        print("\n⚠️  No trades were executed. Consider:")
        print("   - Lowering --min-confidence threshold")
        print("   - Increasing --limit for more data")
        print("   - Checking data quality and date ranges")
    
    print("\n🎉 Realistic backtesting completed!")
    return results

if __name__ == "__main__":
    main()

