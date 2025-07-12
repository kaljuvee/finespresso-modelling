#!/usr/bin/env python3
"""
Export Database Summaries to CSV
Exports backtest_summary and backtest_trades data from PostgreSQL to CSV files
"""

import psycopg2
import pandas as pd
import os
from datetime import datetime

def export_database_summaries():
    """Export database summaries to CSV files"""
    
    DATABASE_URL = 'postgresql://finespresso_db_user:XZ0o6UkxcV0poBcLDQf6RGXwEfWmBlnb@dpg-ctj7u2lumphs73f8t9qg-a.frankfurt-postgres.render.com/finespresso_db'
    
    # Create timestamp for file naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Ensure reports directory exists
    os.makedirs("reports/backtesting", exist_ok=True)
    
    try:
        # Connect to database
        conn = psycopg2.connect(DATABASE_URL)
        
        print("🔗 Connected to PostgreSQL database")
        
        # Export backtest_summary table
        print("📊 Exporting backtest_summary table...")
        summary_query = """
        SELECT run_id, timestamp, model_name, start_date, end_date,
               initial_capital, final_capital, total_pnl, return_percent,
               total_trades, winning_trades, losing_trades, win_rate_percent,
               max_drawdown, sharpe_ratio, news_articles_used, price_moves_used, 
               database_version, notes
        FROM backtest_summary 
        ORDER BY timestamp DESC
        """
        
        summary_df = pd.read_sql(summary_query, conn)
        summary_file = f"reports/backtesting/db_backtest_summary_{timestamp}.csv"
        summary_df.to_csv(summary_file, index=False)
        print(f"✅ Exported {len(summary_df)} backtest summaries to: {summary_file}")
        
        # Export backtest_trades table (recent trades only to avoid huge files)
        print("📈 Exporting recent backtest_trades...")
        trades_query = """
        SELECT published_date, market, entry_time, exit_time, ticker, direction,
               shares, entry_price, exit_price, target_price, stop_price,
               hit_target, hit_stop, pnl, pnl_pct, capital_after,
               news_event, link, runid, rundate
        FROM backtest_trades 
        WHERE created_at >= NOW() - INTERVAL '7 days'
        ORDER BY created_at DESC
        """
        
        trades_df = pd.read_sql(trades_query, conn)
        if not trades_df.empty:
            trades_file = f"reports/backtesting/db_backtest_trades_{timestamp}.csv"
            trades_df.to_csv(trades_file, index=False)
            print(f"✅ Exported {len(trades_df)} recent trades to: {trades_file}")
        else:
            print("ℹ️  No recent trades found in database")
        
        # Create summary statistics
        print("📋 Creating summary statistics...")
        
        # Get summary statistics
        stats_query = """
        SELECT 
            COUNT(*) as total_summaries,
            COUNT(DISTINCT model_name) as unique_models,
            MIN(timestamp) as earliest_run,
            MAX(timestamp) as latest_run,
            AVG(return_percent) as avg_return,
            MAX(return_percent) as best_return,
            MIN(return_percent) as worst_return,
            AVG(win_rate_percent) as avg_win_rate
        FROM backtest_summary
        """
        
        cursor = conn.cursor()
        cursor.execute(stats_query)
        stats = cursor.fetchone()
        
        # Get trade statistics
        trade_stats_query = """
        SELECT 
            COUNT(*) as total_trades,
            COUNT(DISTINCT runid) as unique_runs,
            SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
            SUM(CASE WHEN hit_target THEN 1 ELSE 0 END) as target_hits,
            SUM(CASE WHEN hit_stop THEN 1 ELSE 0 END) as stop_hits,
            AVG(pnl) as avg_pnl,
            SUM(pnl) as total_pnl
        FROM backtest_trades
        """
        
        cursor.execute(trade_stats_query)
        trade_stats = cursor.fetchone()
        
        # Create statistics summary
        stats_data = {
            'Metric': [
                'Total Backtest Summaries',
                'Unique Models',
                'Earliest Run',
                'Latest Run',
                'Average Return %',
                'Best Return %',
                'Worst Return %',
                'Average Win Rate %',
                'Total Individual Trades',
                'Unique Run IDs',
                'Total Winning Trades',
                'Total Target Hits',
                'Total Stop Hits',
                'Average P&L per Trade',
                'Total P&L All Trades'
            ],
            'Value': [
                stats[0] if stats[0] else 0,
                stats[1] if stats[1] else 0,
                stats[2].strftime('%Y-%m-%d %H:%M') if stats[2] else 'N/A',
                stats[3].strftime('%Y-%m-%d %H:%M') if stats[3] else 'N/A',
                f"{stats[4]:.4f}" if stats[4] else '0.0000',
                f"{stats[5]:.4f}" if stats[5] else '0.0000',
                f"{stats[6]:.4f}" if stats[6] else '0.0000',
                f"{stats[7]:.2f}" if stats[7] else '0.00',
                trade_stats[0] if trade_stats[0] else 0,
                trade_stats[1] if trade_stats[1] else 0,
                trade_stats[2] if trade_stats[2] else 0,
                trade_stats[3] if trade_stats[3] else 0,
                trade_stats[4] if trade_stats[4] else 0,
                f"${trade_stats[5]:.2f}" if trade_stats[5] else '$0.00',
                f"${trade_stats[6]:.2f}" if trade_stats[6] else '$0.00'
            ]
        }
        
        stats_df = pd.DataFrame(stats_data)
        stats_file = f"reports/backtesting/db_statistics_{timestamp}.csv"
        stats_df.to_csv(stats_file, index=False)
        print(f"✅ Exported database statistics to: {stats_file}")
        
        # Get model performance comparison
        print("🏆 Creating model performance comparison...")
        model_query = """
        SELECT 
            model_name,
            COUNT(*) as run_count,
            AVG(return_percent) as avg_return,
            MAX(return_percent) as best_return,
            AVG(win_rate_percent) as avg_win_rate,
            AVG(total_trades) as avg_trades,
            MAX(timestamp) as latest_run
        FROM backtest_summary 
        GROUP BY model_name
        ORDER BY avg_return DESC
        """
        
        model_df = pd.read_sql(model_query, conn)
        model_file = f"reports/backtesting/db_model_performance_{timestamp}.csv"
        model_df.to_csv(model_file, index=False)
        print(f"✅ Exported model performance comparison to: {model_file}")
        
        cursor.close()
        conn.close()
        
        print(f"\n📁 All database exports completed successfully!")
        print(f"   Files saved to: reports/backtesting/")
        
        return {
            'summary_file': summary_file,
            'trades_file': trades_file if not trades_df.empty else None,
            'stats_file': stats_file,
            'model_file': model_file
        }
        
    except Exception as e:
        print(f"❌ Error exporting database summaries: {e}")
        return None

if __name__ == "__main__":
    export_database_summaries()

