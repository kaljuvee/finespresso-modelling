#!/usr/bin/env python3
"""
Script to verify backtest trades in the database and show summaries
"""

import psycopg2
import pandas as pd
from datetime import datetime

def verify_backtest_trades():
    """Verify and display backtest trades from the database"""
    
    DATABASE_URL = 'postgresql://finespresso_db_user:XZ0o6UkxcV0poBcLDQf6RGXwEfWmBlnb@dpg-ctj7u2lumphs73f8t9qg-a.frankfurt-postgres.render.com/finespresso_db'
    
    try:
        # Connect to database
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()
        
        # Check if backtest_trades table exists and has data
        cursor.execute("SELECT COUNT(*) FROM backtest_trades;")
        total_trades = cursor.fetchone()[0]
        
        print(f"📊 Total trades in database: {total_trades}")
        
        if total_trades > 0:
            # Get recent trades
            cursor.execute("""
                SELECT runid, COUNT(*) as trade_count, 
                       SUM(pnl) as total_pnl,
                       AVG(pnl_pct) as avg_return_pct,
                       SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
                       SUM(CASE WHEN hit_target THEN 1 ELSE 0 END) as target_hits,
                       SUM(CASE WHEN hit_stop THEN 1 ELSE 0 END) as stop_hits,
                       MIN(rundate) as run_date
                FROM backtest_trades 
                GROUP BY runid 
                ORDER BY MIN(rundate) DESC 
                LIMIT 10;
            """)
            
            results = cursor.fetchall()
            
            print(f"\n📈 Recent Backtest Runs:")
            print(f"{'Run ID':<10} {'Trades':<7} {'Total P&L':<12} {'Avg Return %':<12} {'Win Rate':<10} {'Targets':<8} {'Stops':<6} {'Date'}")
            print("-" * 80)
            
            for row in results:
                runid, trade_count, total_pnl, avg_return, winning_trades, targets, stops, run_date = row
                win_rate = (winning_trades / trade_count * 100) if trade_count > 0 else 0
                
                print(f"{runid:<10} {trade_count:<7} ${total_pnl:<11.2f} {avg_return:<11.4f}% {win_rate:<9.1f}% {targets:<8} {stops:<6} {run_date.strftime('%m/%d %H:%M')}")
            
            # Get sample trades
            print(f"\n🔍 Sample Recent Trades:")
            cursor.execute("""
                SELECT published_date, ticker, direction, shares, entry_price, exit_price, 
                       pnl, pnl_pct, hit_target, hit_stop, news_event, runid
                FROM backtest_trades 
                ORDER BY created_at DESC 
                LIMIT 5;
            """)
            
            trades = cursor.fetchall()
            
            for trade in trades:
                pub_date, ticker, direction, shares, entry, exit, pnl, pnl_pct, target, stop, event, runid = trade
                status = "🎯 TARGET" if target else "🛑 STOP" if stop else "📊 CLOSE"
                print(f"  {ticker:<6} {direction:<5} {shares:>4} shares @ ${entry:.2f} → ${exit:.2f} = ${pnl:>7.2f} ({pnl_pct:>6.2f}%) {status} [{runid}]")
                print(f"         {event} | {pub_date.strftime('%Y-%m-%d %H:%M')}")
                print()
        
        # Check backtest_summary table
        cursor.execute("SELECT COUNT(*) FROM backtest_summary;")
        summary_count = cursor.fetchone()[0]
        
        print(f"📋 Backtest summaries in database: {summary_count}")
        
        if summary_count > 0:
            cursor.execute("""
                SELECT model_name, total_trades, win_rate_percent, return_percent, 
                       total_pnl, final_capital, timestamp
                FROM backtest_summary 
                ORDER BY timestamp DESC 
                LIMIT 5;
            """)
            
            summaries = cursor.fetchall()
            
            print(f"\n📊 Recent Backtest Summaries:")
            for summary in summaries:
                model, trades, win_rate, return_pct, pnl, final_cap, timestamp = summary
                print(f"  {model}: {trades} trades, {win_rate:.1f}% win rate, {return_pct:.3f}% return (${pnl:.2f})")
                print(f"    Final Capital: ${final_cap:,.2f} | {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
                print()
        
        cursor.close()
        conn.close()
        
        print("✅ Database verification completed!")
        
    except Exception as e:
        print(f"❌ Error verifying database: {e}")

if __name__ == "__main__":
    verify_backtest_trades()

