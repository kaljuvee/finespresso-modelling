#!/usr/bin/env python3
"""
PostgreSQL-based backtesting script for Finespresso Analytics

Usage:
    python run_postgres_backtest.py --capital 100000 --start-date 2024-01-01 --end-date 2024-12-31
    python run_postgres_backtest.py --help
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


class PostgresBacktester:
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
    
    def calculate_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """Calculate current portfolio value"""
        portfolio_value = self.capital
        
        for ticker, quantity in self.positions.items():
            if ticker in current_prices:
                portfolio_value += quantity * current_prices[ticker]
        
        return portfolio_value
    
    def run_simple_strategy(self, data: pd.DataFrame, strategy: str = "direction_based") -> Dict[str, Any]:
        """
        Run a simple backtesting strategy based on actual vs predicted price movements
        
        Args:
            data: DataFrame with news and price move data
            strategy: Strategy type ("direction_based", "alpha_based", "confidence_based")
        """
        
        print(f"🚀 Running {strategy} strategy on {len(data)} data points")
        
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
            
            # Strategy 1: Direction-based trading
            if strategy == "direction_based":
                # Use actual_side as our "prediction" for now (this simulates perfect prediction)
                # In real implementation, this would come from ML models
                predicted_side = actual_side  # Simulating perfect prediction for testing
                
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
            
            # Strategy 2: Alpha-based trading (trade based on daily alpha)
            elif strategy == "alpha_based":
                if abs(daily_alpha) > 0.02:  # Only trade if alpha > 2%
                    quantity = int((self.capital * 0.05) / begin_price)  # Use 5% of capital
                    if quantity > 0:
                        if daily_alpha > 0:
                            # Positive alpha - buy
                            if self.execute_trade(ticker, published_date, 'BUY', quantity, begin_price, news_id, event):
                                self.execute_trade(ticker, published_date, 'SELL', quantity, end_price, news_id, event)
                                trades_executed += 1
                                if actual_side == 'UP':
                                    correct_predictions += 1
                        else:
                            # Negative alpha - short
                            pnl = quantity * (begin_price - end_price)
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
            'strategy': strategy,
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


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Run PostgreSQL-based backtesting for Finespresso Analytics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_postgres_backtest.py --capital 100000
  python run_postgres_backtest.py --capital 50000 --start-date 2024-01-01 --end-date 2024-12-31
  python run_postgres_backtest.py --events financial_results earnings_releases_and_operating_results
  python run_postgres_backtest.py --list-events
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
        "--publishers", "-p",
        nargs="+",
        default=None,
        help="Specific publishers to filter for backtesting. Default: all publishers"
    )
    
    parser.add_argument(
        "--list-publishers", "-lp",
        action="store_true",
        help="List all available publishers and exit"
    )
    
    parser.add_argument(
        "--events", "-ev",
        nargs="+",
        default=None,
        help="Specific events to filter for backtesting. Default: all events"
    )
    
    parser.add_argument(
        "--strategies", "-st",
        nargs="+",
        default=["direction_based"],
        choices=["direction_based", "alpha_based"],
        help="Trading strategies to test"
    )
    
    parser.add_argument(
        "--list-events", "-l",
        action="store_true",
        help="List all available events and exit"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output file for results (CSV format). Default: print to console"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--limit", "-lim",
        type=int,
        default=None,
        help="Limit number of records to process (for testing)"
    )
    
    parser.add_argument(
        "--save-to-db", "-db",
        action="store_true",
        help="Save results to database backtest_summary table"
    )
    
    return parser.parse_args()


def validate_date(date_string):
    """Validate and parse date string"""
    try:
        return datetime.strptime(date_string, "%Y-%m-%d")
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid date format: {date_string}. Use YYYY-MM-DD")


def main():
    """Main function"""
    args = parse_arguments()
    
    print("🎯 Finespresso Analytics - PostgreSQL Backtesting")
    print("=" * 60)
    
    try:
        # Initialize data manager
        dm = PostgresDataManager()
        
        # List events if requested
        if args.list_events:
            events = dm.get_unique_events()
            print("📋 Available Events:")
            print("=" * 50)
            for i, event in enumerate(events, 1):
                print(f"{i:2d}. {event}")
            print("=" * 50)
            return
        
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
            start_date = validate_date(args.start_date)
        
        if args.end_date is None:
            end_date = datetime.now()
        else:
            end_date = validate_date(args.end_date)
        
        # Validate date range
        if start_date >= end_date:
            print("❌ Start date must be before end date")
            return
        
        print(f"💰 Initial Capital: ${args.capital:,.2f}")
        print(f"📅 Date Range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        print(f"🎯 Strategies: {args.strategies}")
        
        # Get data
        print(f"\n📊 Fetching data from PostgreSQL...")
        data = dm.get_news_with_price_moves(start_date=start_date, end_date=end_date, limit=args.limit)
        
        if data.empty:
            print("❌ No data found for the specified criteria")
            return
        
        # Filter by events if specified
        if args.events:
            data = data[data['event'].isin(args.events)]
            print(f"🔍 Filtered to events: {args.events}")
        
        print(f"📈 Processing {len(data)} news articles with price moves")
        
        # Run backtesting for each strategy
        results = []
        run_id = str(uuid.uuid4())
        
        for strategy in args.strategies:
            print(f"\n🚀 Running {strategy} strategy...")
            
            # Initialize backtester
            backtester = PostgresBacktester(initial_capital=args.capital)
            
            # Run backtest
            result = backtester.run_simple_strategy(data, strategy=strategy)
            
            # Add metadata
            result.update({
                'run_id': run_id,
                'start_date': start_date,
                'end_date': end_date,
                'data_points': len(data),
                'events_filter': args.events,
                'timestamp': datetime.now()
            })
            
            results.append(result)
            
            # Print results
            print(f"  💰 PnL: ${result['total_pnl']:,.2f}")
            print(f"  📊 Trades: {result['total_trades']} (Win Rate: {result['win_rate_percent']:.1f}%)")
            print(f"  📈 Return: {result['return_percent']:.2f}%")
            
            # Save to database if requested
            if args.save_to_db:
                db_result = {
                    'model_name': strategy,
                    'start_date': start_date,
                    'end_date': end_date,
                    'initial_capital': args.capital,
                    'final_capital': result['final_capital'],
                    'total_pnl': result['total_pnl'],
                    'return_percent': result['return_percent'],
                    'total_trades': result['total_trades'],
                    'winning_trades': result['correct_predictions'],
                    'losing_trades': result['total_trades'] - result['correct_predictions'],
                    'win_rate_percent': result['win_rate_percent'],
                    'news_articles_used': len(data),
                    'price_moves_used': len(data),
                    'notes': f"Strategy: {strategy}, Events: {args.events or 'all'}"
                }
                dm.save_backtest_results(db_result, run_id)
        
        # Display summary results
        if results:
            print("\n📊 BACKTEST RESULTS SUMMARY")
            print("=" * 80)
            
            # Create results DataFrame
            df_results = pd.DataFrame([{
                'Strategy': r['strategy'],
                'Total_PnL_USD': r['total_pnl'],
                'Return_Percent': r['return_percent'],
                'Total_Trades': r['total_trades'],
                'Win_Rate_Percent': r['win_rate_percent'],
                'Final_Capital_USD': r['final_capital']
            } for r in results])
            
            df_results = df_results.sort_values('Return_Percent', ascending=False)
            
            # Print formatted results
            print(f"{'Strategy':<20} {'PnL':<12} {'Return %':<10} {'Trades':<8} {'Win Rate':<10}")
            print("-" * 80)
            
            for _, row in df_results.iterrows():
                print(f"{row['Strategy']:<20} ${row['Total_PnL_USD']:>10,.2f} {row['Return_Percent']:>8.2f}% "
                      f"{row['Total_Trades']:>6} {row['Win_Rate_Percent']:>8.1f}%")
            
            print("-" * 80)
            
            # Summary statistics
            best_strategy = df_results.iloc[0]
            total_pnl = df_results['Total_PnL_USD'].sum()
            avg_return = df_results['Return_Percent'].mean()
            
            print(f"\n🏆 Best Strategy: {best_strategy['Strategy']} ({best_strategy['Return_Percent']:.2f}% return)")
            print(f"💰 Total PnL: ${total_pnl:,.2f}")
            print(f"📈 Average Return: {avg_return:.2f}%")
            
            # Save to file if requested
            if args.output:
                df_results.to_csv(args.output, index=False)
                print(f"💾 Results saved to: {args.output}")
            
            print(f"\n✅ Backtesting completed successfully!")
            
            if args.save_to_db:
                print(f"💾 Results saved to database with run_id: {run_id}")
        
    except Exception as e:
        print(f"❌ Backtesting failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

