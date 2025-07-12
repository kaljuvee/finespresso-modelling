#!/usr/bin/env python3
"""
Check backtest results stored in the PostgreSQL database
"""

import sys
import os
from datetime import datetime
import pandas as pd

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from postgres_data_manager import PostgresDataManager


def main():
    print("🔍 Checking Backtest Results in Database")
    print("=" * 50)
    
    try:
        # Initialize data manager
        dm = PostgresDataManager()
        
        # Get recent backtest results
        results = dm.get_recent_backtest_results(limit=20)
        
        if results.empty:
            print("❌ No backtest results found in database")
            return
        
        print(f"📊 Found {len(results)} recent backtest results:")
        print("=" * 100)
        
        # Display results in a formatted table
        for _, row in results.iterrows():
            print(f"🆔 Run ID: {row['run_id']}")
            print(f"📰 Model: {row['model_name']}")
            print(f"📅 Date: {row['timestamp']}")
            print(f"💰 Capital: ${row['initial_capital']:,.2f} → ${row['final_capital']:,.2f}")
            print(f"📈 PnL: ${row['total_pnl']:,.2f} ({row['return_percent']:.2f}%)")
            print(f"📊 Trades: {row['total_trades']} (Win Rate: {row['win_rate_percent']:.1f}%)")
            print(f"📰 Data: {row['news_articles_used']} articles, {row['price_moves_used']} price moves")
            print(f"📝 Notes: {row['notes']}")
            print("-" * 100)
        
        # Summary statistics
        print(f"\n📊 SUMMARY STATISTICS")
        print("=" * 50)
        print(f"Total runs: {len(results)}")
        print(f"Average return: {results['return_percent'].mean():.2f}%")
        print(f"Best return: {results['return_percent'].max():.2f}%")
        print(f"Worst return: {results['return_percent'].min():.2f}%")
        print(f"Total PnL: ${results['total_pnl'].sum():,.2f}")
        
        # Publisher-specific analysis
        publisher_results = results[results['model_name'].str.contains('_')]
        if not publisher_results.empty:
            print(f"\n📰 PUBLISHER ANALYSIS")
            print("=" * 50)
            
            # Extract publisher from model name
            publisher_results['publisher'] = publisher_results['model_name'].str.split('_').str[-1]
            publisher_summary = publisher_results.groupby('publisher').agg({
                'return_percent': 'mean',
                'total_pnl': 'sum',
                'total_trades': 'sum',
                'win_rate_percent': 'mean'
            }).round(2)
            
            print(publisher_summary)
        
        print(f"\n✅ Database check completed successfully!")
        
    except Exception as e:
        print(f"❌ Error checking database: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

