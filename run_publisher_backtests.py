#!/usr/bin/env python3
"""
Publisher-specific backtesting script for Finespresso Analytics

This script runs separate backtests for different publishers and stores results
with separate run IDs in the database.

Usage:
    python run_publisher_backtests.py --capital 100000
    python run_publisher_backtests.py --help
"""

import argparse
import sys
import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import uuid
from typing import Dict, List, Any

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from postgres_data_manager import PostgresDataManager


class PublisherBacktester:
    def __init__(self, initial_capital: float = 100000.0):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.positions = {}
        self.trade_log = []
        self.data_manager = PostgresDataManager()
        
    def reset(self):
        """Reset backtester state for new run"""
        self.capital = self.initial_capital
        self.positions = {}
        self.trade_log = []
    
    def execute_trade(self, ticker: str, date: datetime, action: str, quantity: int, price: float, 
                     news_id: int = None, event: str = None):
        """Execute a trade and log it"""
        if action == 'BUY':
            cost = quantity * price
            if cost <= self.capital:
                self.capital -= cost
                self.positions[ticker] = self.positions.get(ticker, 0) + quantity
                
                trade = {
                    'date': date,
                    'ticker': ticker,
                    'action': action,
                    'quantity': quantity,
                    'price': price,
                    'cost': cost,
                    'news_id': news_id,
                    'event': event,
                    'capital_after': self.capital
                }
                self.trade_log.append(trade)
                return True
        
        elif action == 'SELL':
            if ticker in self.positions and self.positions[ticker] >= quantity:
                revenue = quantity * price
                self.capital += revenue
                self.positions[ticker] -= quantity
                
                if self.positions[ticker] == 0:
                    del self.positions[ticker]
                
                trade = {
                    'date': date,
                    'ticker': ticker,
                    'action': action,
                    'quantity': quantity,
                    'price': price,
                    'revenue': revenue,
                    'news_id': news_id,
                    'event': event,
                    'capital_after': self.capital
                }
                self.trade_log.append(trade)
                return True
        
        return False
    
    def run_direction_strategy(self, data: pd.DataFrame, publisher: str) -> Dict[str, Any]:
        """
        Run direction-based trading strategy
        """
        
        print(f"🚀 Running direction strategy for {publisher} on {len(data)} data points")
        
        trades_executed = 0
        correct_predictions = 0
        total_predictions = 0
        
        # Sort by published date
        data = data.sort_values('published_date').reset_index(drop=True)
        
        for idx, row in data.iterrows():
            if pd.isna(row['actual_side']) or pd.isna(row['begin_price']) or pd.isna(row['end_price']):
                continue
                
            ticker = row['ticker']
            if not ticker or ticker.strip() == '':
                continue
                
            news_id = row['news_id']
            event = row['event']
            published_date = row['published_date']
            begin_price = row['begin_price']
            end_price = row['end_price']
            actual_side = row['actual_side']
            daily_alpha = row.get('daily_alpha', 0)
            
            # Skip if prices are invalid
            if begin_price <= 0 or end_price <= 0:
                continue
            
            total_predictions += 1
            
            # Use the actual predicted_side from ML models in the database
            predicted_side = row['predicted_side']
            
            # Skip if no prediction available
            if pd.isna(predicted_side) or predicted_side not in ['UP', 'DOWN']:
                continue
            
            if predicted_side == 'UP':
                # Buy at begin_price, sell at end_price
                quantity = int((self.capital * 0.1) / begin_price)  # Use 10% of capital
                if quantity > 0:
                    if self.execute_trade(ticker, published_date, 'BUY', quantity, begin_price, news_id, event):
                        # Sell at end price
                        self.execute_trade(ticker, published_date, 'SELL', quantity, end_price, news_id, event)
                        trades_executed += 1
                        
                        if actual_side == 'UP':
                            correct_predictions += 1
            
            elif predicted_side == 'DOWN':
                # Short strategy: sell high, buy low (simplified)
                quantity = int((self.capital * 0.1) / begin_price)  # Use 10% of capital
                if quantity > 0:
                    # Simulate short selling by reversing the trade logic
                    pnl = quantity * (begin_price - end_price)  # Profit if price goes down
                    self.capital += pnl
                    
                    trade = {
                        'date': published_date,
                        'ticker': ticker,
                        'action': 'SHORT',
                        'quantity': quantity,
                        'entry_price': begin_price,
                        'exit_price': end_price,
                        'pnl': pnl,
                        'news_id': news_id,
                        'event': event,
                        'capital_after': self.capital
                    }
                    self.trade_log.append(trade)
                    trades_executed += 1
                    
                    if actual_side == 'DOWN':
                        correct_predictions += 1
        
        # Calculate final metrics
        final_capital = self.capital
        total_pnl = final_capital - self.initial_capital
        return_pct = (total_pnl / self.initial_capital) * 100
        win_rate = (correct_predictions / total_predictions * 100) if total_predictions > 0 else 0
        
        return {
            'publisher': publisher,
            'strategy': 'direction_based',
            'initial_capital': self.initial_capital,
            'final_capital': final_capital,
            'total_pnl': total_pnl,
            'return_percent': return_pct,
            'total_trades': trades_executed,
            'total_predictions': total_predictions,
            'correct_predictions': correct_predictions,
            'win_rate_percent': win_rate,
            'trade_log': self.trade_log
        }


def run_publisher_backtest(publisher: str, capital: float, start_date: datetime, end_date: datetime, 
                          limit: int = None, save_to_db: bool = True) -> Dict[str, Any]:
    """Run backtest for a specific publisher"""
    
    print(f"\n📰 Running backtest for publisher: {publisher}")
    print("=" * 60)
    
    # Initialize data manager and backtester
    dm = PostgresDataManager()
    backtester = PublisherBacktester(initial_capital=capital)
    
    # Get data for this publisher
    print(f"📊 Fetching data...")
    data = dm.get_news_with_price_moves(
        start_date=start_date, 
        end_date=end_date, 
        limit=limit,
        publisher=publisher if publisher != 'all' else None
    )
    
    if data.empty:
        print(f"❌ No data found for publisher: {publisher}")
        return None
    
    print(f"📈 Processing {len(data)} news articles with price moves")
    
    # Run backtest
    result = backtester.run_direction_strategy(data, publisher)
    
    # Add metadata
    run_id = str(uuid.uuid4())
    result.update({
        'run_id': run_id,
        'start_date': start_date,
        'end_date': end_date,
        'data_points': len(data),
        'timestamp': datetime.now()
    })
    
    # Print results
    print(f"  💰 PnL: ${result['total_pnl']:,.2f}")
    print(f"  📊 Trades: {result['total_trades']} (Win Rate: {result['win_rate_percent']:.1f}%)")
    print(f"  📈 Return: {result['return_percent']:.2f}%")
    print(f"  🆔 Run ID: {run_id}")
    
    # Save to database if requested
    if save_to_db:
        db_result = {
            'model_name': f"direction_based_{publisher}",
            'start_date': start_date,
            'end_date': end_date,
            'initial_capital': capital,
            'final_capital': result['final_capital'],
            'total_pnl': result['total_pnl'],
            'return_percent': result['return_percent'],
            'total_trades': result['total_trades'],
            'winning_trades': result['correct_predictions'],
            'losing_trades': result['total_trades'] - result['correct_predictions'],
            'win_rate_percent': result['win_rate_percent'],
            'news_articles_used': len(data),
            'price_moves_used': len(data),
            'notes': f"Publisher-specific backtest for {publisher}"
        }
        
        success = dm.save_backtest_results(db_result, run_id)
        if success:
            print(f"  💾 Results saved to database")
        else:
            print(f"  ❌ Failed to save results to database")
    
    return result


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Run publisher-specific backtests for Finespresso Analytics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_publisher_backtests.py --capital 100000
  python run_publisher_backtests.py --capital 50000 --start-date 2024-01-01
  python run_publisher_backtests.py --list-publishers
        """
    )
    
    parser.add_argument(
        "--capital", "-c",
        type=float,
        default=100000.0,
        help="Initial capital for backtesting (default: 100000)"
    )
    
    parser.add_argument(
        "--start-date", "-s",
        type=str,
        default=None,
        help="Start date for backtesting (YYYY-MM-DD format). Default: 30 days ago"
    )
    
    parser.add_argument(
        "--end-date", "-e",
        type=str,
        default=None,
        help="End date for backtesting (YYYY-MM-DD format). Default: today"
    )
    
    parser.add_argument(
        "--list-publishers", "-l",
        action="store_true",
        help="List all available publishers and exit"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output file for results (CSV format). Default: print to console"
    )
    
    parser.add_argument(
        "--limit", "-lim",
        type=int,
        default=None,
        help="Limit number of records to process per publisher (for testing)"
    )
    
    parser.add_argument(
        "--no-save-db",
        action="store_true",
        help="Don't save results to database"
    )
    
    args = parser.parse_args()
    
    print("🎯 Finespresso Analytics - Publisher-Specific Backtesting")
    print("=" * 70)
    
    try:
        # Initialize data manager
        dm = PostgresDataManager()
        
        # List publishers if requested
        if args.list_publishers:
            publishers = dm.get_unique_publishers()
            print("📋 Available Publishers:")
            print("=" * 50)
            for i, publisher in enumerate(publishers, 1):
                print(f"{i:2d}. {publisher}")
            print("=" * 50)
            return
        
        # Set default dates if not provided
        if args.start_date is None:
            start_date = datetime.now() - timedelta(days=30)
        else:
            start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
        
        if args.end_date is None:
            end_date = datetime.now()
        else:
            end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
        
        # Validate date range
        if start_date >= end_date:
            print("❌ Start date must be before end date")
            return
        
        print(f"💰 Initial Capital: ${args.capital:,.2f}")
        print(f"📅 Date Range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        # Define publishers to test
        publishers_to_test = ['globenewswire_biotech', 'all']
        
        print(f"📰 Testing publishers: {publishers_to_test}")
        
        # Run backtests for each publisher
        all_results = []
        
        for publisher in publishers_to_test:
            result = run_publisher_backtest(
                publisher=publisher,
                capital=args.capital,
                start_date=start_date,
                end_date=end_date,
                limit=args.limit,
                save_to_db=not args.no_save_db
            )
            
            if result:
                all_results.append(result)
        
        # Display summary results
        if all_results:
            print("\n📊 PUBLISHER BACKTEST RESULTS SUMMARY")
            print("=" * 80)
            
            # Create results DataFrame
            df_results = pd.DataFrame([{
                'Publisher': r['publisher'],
                'Total_PnL_USD': r['total_pnl'],
                'Return_Percent': r['return_percent'],
                'Total_Trades': r['total_trades'],
                'Win_Rate_Percent': r['win_rate_percent'],
                'Final_Capital_USD': r['final_capital'],
                'Run_ID': r['run_id']
            } for r in all_results])
            
            df_results = df_results.sort_values('Return_Percent', ascending=False)
            
            # Print formatted results
            print(f"{'Publisher':<25} {'PnL':<12} {'Return %':<10} {'Trades':<8} {'Win Rate':<10}")
            print("-" * 80)
            
            for _, row in df_results.iterrows():
                print(f"{row['Publisher']:<25} ${row['Total_PnL_USD']:>10,.2f} {row['Return_Percent']:>8.2f}% "
                      f"{row['Total_Trades']:>6} {row['Win_Rate_Percent']:>8.1f}%")
            
            print("-" * 80)
            
            # Summary statistics
            best_publisher = df_results.iloc[0]
            total_pnl = df_results['Total_PnL_USD'].sum()
            avg_return = df_results['Return_Percent'].mean()
            
            print(f"\n🏆 Best Publisher: {best_publisher['Publisher']} ({best_publisher['Return_Percent']:.2f}% return)")
            print(f"💰 Total PnL: ${total_pnl:,.2f}")
            print(f"📈 Average Return: {avg_return:.2f}%")
            
            # Save to file if requested
            if args.output:
                df_results.to_csv(args.output, index=False)
                print(f"💾 Results saved to: {args.output}")
            
            print(f"\n✅ Publisher backtesting completed successfully!")
            
            if not args.no_save_db:
                print(f"💾 All results saved to database with separate run IDs")
        
        else:
            print("❌ No results generated")
        
    except Exception as e:
        print(f"❌ Backtesting failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

