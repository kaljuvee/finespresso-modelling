#!/usr/bin/env python3
"""
Command-line backtesting script for Finespresso Analytics

Usage:
    python run_backtest.py --capital 100000 --start-date 2024-01-01 --end-date 2024-12-31
    python run_backtest.py --help
"""

import argparse
import sys
import os
from datetime import datetime, timedelta
import pandas as pd

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_manager import DataManager
from backtest_util import Backtester


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Run backtesting for Finespresso Analytics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_backtest.py --capital 100000
  python run_backtest.py --capital 50000 --start-date 2024-01-01 --end-date 2024-12-31
  python run_backtest.py --capital 100000 --models all_events partnerships
  python run_backtest.py --list-models
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
        "--models", "-m",
        nargs="+",
        default=None,
        help="Specific models to use for backtesting. Default: all available models"
    )
    
    parser.add_argument(
        "--list-models", "-l",
        action="store_true",
        help="List all available models and exit"
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
        "--database", "-d",
        type=str,
        default="finespresso.db",
        help="Database file path (default: finespresso.db)"
    )
    
    return parser.parse_args()


def validate_date(date_string):
    """Validate and parse date string"""
    try:
        return datetime.strptime(date_string, "%Y-%m-%d")
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid date format: {date_string}. Use YYYY-MM-DD")


def list_available_models():
    """List all available trained models"""
    models_dir = "models"
    if not os.path.exists(models_dir):
        print("❌ Models directory not found. Please train models first.")
        return []
    
    model_files = [f for f in os.listdir(models_dir) if f.endswith("_model.joblib")]
    models = [f.replace("_model.joblib", "") for f in model_files]
    
    if models:
        print("📋 Available Models:")
        print("=" * 50)
        for i, model in enumerate(models, 1):
            print(f"{i:2d}. {model}")
        print("=" * 50)
    else:
        print("❌ No trained models found. Please train models first.")
    
    return models


def check_database(db_path):
    """Check if database exists and has data"""
    if not os.path.exists(db_path):
        print(f"❌ Database not found: {db_path}")
        print("💡 Please run data collection first or specify correct database path with --database")
        return False
    
    try:
        dm = DataManager(f"sqlite:///{db_path}")
        
        # Check for news articles
        news_count = dm.get_news_count()
        if news_count == 0:
            print("❌ No news articles found in database")
            print("💡 Please run data collection first")
            return False
        
        # Check for price data
        price_count = dm.get_price_data_count()
        if price_count == 0:
            print("❌ No price data found in database")
            print("💡 Please run data collection first")
            return False
        
        print(f"✅ Database ready: {news_count} articles, {price_count} price records")
        return True
        
    except Exception as e:
        print(f"❌ Database error: {e}")
        return False


def run_backtest(args):
    """Run the backtesting with given parameters"""
    
    # Check database
    if not check_database(args.database):
        return False
    
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
        return False
    
    print(f"🚀 Starting Backtest")
    print("=" * 60)
    print(f"💰 Initial Capital: ${args.capital:,.2f}")
    print(f"📅 Date Range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"🗄️  Database: {args.database}")
    
    # Initialize backtester
    try:
        backtester = Backtester(initial_capital=args.capital, model_dir="models/")
        
        # Set up database connection for backtester
        backtester.data_manager = DataManager(f"sqlite:///{args.database}")
        
        # Get available models
        available_models = backtester.models.keys()
        
        if args.models:
            # Use specified models
            models_to_use = []
            for model in args.models:
                if model in available_models:
                    models_to_use.append(model)
                else:
                    print(f"⚠️  Model '{model}' not found. Available models: {list(available_models)}")
            
            if not models_to_use:
                print("❌ No valid models specified")
                return False
        else:
            # Use all available models
            models_to_use = list(available_models)
        
        print(f"🤖 Using Models: {models_to_use}")
        print("=" * 60)
        
        # Run backtesting
        results = []
        
        # Prepare data for backtesting
        dm = DataManager(f"sqlite:///{args.database}")
        
        # Get news data
        news_articles = dm.get_all_news()
        news_data = []
        for article in news_articles:
            if article.published >= start_date and article.published <= end_date and article.ticker:
                news_data.append({
                    'published': article.published,
                    'ticker': article.ticker,
                    'text': article.title + " " + (article.summary or ""),
                    'event_category': 'all_events'  # Simplified for now
                })
        
        if not news_data:
            print("❌ No news data found for the specified date range")
            return False
        
        news_df = pd.DataFrame(news_data)
        
        # Get price data
        unique_tickers = news_df['ticker'].unique()
        price_data = []
        for ticker in unique_tickers:
            ticker_prices = dm.get_price_data_for_ticker(ticker, start_date, end_date)
            for price in ticker_prices:
                price_data.append({
                    'date': price.date,
                    'ticker': price.ticker,
                    'open': price.open,
                    'close': price.close
                })
        
        if not price_data:
            print("❌ No price data found for the specified date range")
            return False
        
        price_df = pd.DataFrame(price_data)
        
        print(f"📊 Data prepared: {len(news_df)} news articles, {len(price_df)} price records")
        
        for model_name in models_to_use:
            if args.verbose:
                print(f"\n🔄 Running backtest with model: {model_name}")
            
            try:
                # Update news data to use the specific model
                news_df_model = news_df.copy()
                news_df_model['event_category'] = model_name
                
                # Run backtest
                final_pnl, trade_log = backtester.run_backtest(news_df_model, price_df)
                
                # Calculate metrics
                total_trades = len(trade_log)
                winning_trades = len([t for t in trade_log if 'BUY' in str(t) and 'profit' in str(t).lower()])
                losing_trades = total_trades - winning_trades
                win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
                final_capital = args.capital + final_pnl
                
                results.append({
                    'model': model_name,
                    'total_pnl': final_pnl,
                    'total_trades': total_trades,
                    'winning_trades': winning_trades,
                    'losing_trades': losing_trades,
                    'win_rate': win_rate,
                    'final_capital': final_capital,
                    'return_pct': (final_pnl / args.capital) * 100
                })
                
                if args.verbose:
                    print(f"  💰 PnL: ${final_pnl:,.2f}")
                    print(f"  📊 Trades: {total_trades} (Win Rate: {win_rate:.1f}%)")
                    print(f"  📈 Return: {(final_pnl / args.capital) * 100:.2f}%")
                
                # Reset backtester for next model
                backtester.capital = args.capital
                backtester.positions = {}
                backtester.trade_log = []
                
            except Exception as e:
                print(f"❌ Error running backtest with {model_name}: {e}")
                continue
        
        # Display results
        if results:
            print("\n📊 BACKTEST RESULTS")
            print("=" * 80)
            
            # Create results DataFrame
            df_results = pd.DataFrame(results)
            df_results = df_results.sort_values('return_pct', ascending=False)
            
            # Rename columns for better readability
            df_results = df_results.rename(columns={
                'model': 'Model_Name',
                'total_pnl': 'Total_PnL_USD',
                'total_trades': 'Total_Trades',
                'winning_trades': 'Winning_Trades',
                'losing_trades': 'Losing_Trades',
                'win_rate': 'Win_Rate_Percent',
                'final_capital': 'Final_Capital_USD',
                'return_pct': 'Return_Percent'
            })
            
            # Print formatted results
            print(f"{'Model':<30} {'PnL':<12} {'Trades':<8} {'Win Rate':<10} {'Return %':<10}")
            print("-" * 80)
            
            for _, row in df_results.iterrows():
                print(f"{row['Model_Name']:<30} ${row['Total_PnL_USD']:>10,.2f} {row['Total_Trades']:>6} "
                      f"{row['Win_Rate_Percent']:>8.1f}% {row['Return_Percent']:>8.2f}%")
            
            print("-" * 80)
            
            # Summary statistics
            best_model = df_results.iloc[0]
            total_pnl = df_results['Total_PnL_USD'].sum()
            avg_return = df_results['Return_Percent'].mean()
            
            print(f"\n🏆 Best Model: {best_model['Model_Name']} ({best_model['Return_Percent']:.2f}% return)")
            print(f"💰 Total PnL: ${total_pnl:,.2f}")
            print(f"📈 Average Return: {avg_return:.2f}%")
            
            # Save to file if requested
            if args.output:
                df_results.to_csv(args.output, index=False)
                print(f"💾 Results saved to: {args.output}")
            
            return True
        else:
            print("❌ No successful backtests completed")
            return False
            
    except Exception as e:
        print(f"❌ Backtesting error: {e}")
        return False


def main():
    """Main function"""
    args = parse_arguments()
    
    print("🎯 Finespresso Analytics - Command Line Backtesting")
    print("=" * 60)
    
    # List models if requested
    if args.list_models:
        list_available_models()
        return
    
    # Run backtesting
    success = run_backtest(args)
    
    if success:
        print("\n✅ Backtesting completed successfully!")
    else:
        print("\n❌ Backtesting failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()

