#!/usr/bin/env python3
"""
Script to create the backtest_trades table in PostgreSQL database
"""

import psycopg2

def create_backtest_trades_table():
    """Create the backtest_trades table with the exact format from the CSV"""
    
    DATABASE_URL = 'postgresql://finespresso_db_user:XZ0o6UkxcV0poBcLDQf6RGXwEfWmBlnb@dpg-ctj7u2lumphs73f8t9qg-a.frankfurt-postgres.render.com/finespresso_db'
    
    try:
        # Connect to database
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()
        
        # Create backtest_trades table
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS backtest_trades (
            id SERIAL PRIMARY KEY,
            published_date TIMESTAMP WITH TIME ZONE,
            market VARCHAR(20),
            entry_time TIMESTAMP WITH TIME ZONE,
            exit_time TIMESTAMP WITH TIME ZONE,
            ticker VARCHAR(10),
            direction VARCHAR(10),
            shares INTEGER,
            entry_price DECIMAL(10, 6),
            exit_price DECIMAL(10, 6),
            target_price DECIMAL(10, 6),
            stop_price DECIMAL(10, 6),
            hit_target BOOLEAN,
            hit_stop BOOLEAN,
            pnl DECIMAL(12, 6),
            pnl_pct DECIMAL(8, 6),
            capital_after DECIMAL(15, 6),
            news_event VARCHAR(100),
            link TEXT,
            runid VARCHAR(50),
            rundate TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        
        cursor.execute(create_table_sql)
        
        # Create indexes for better performance
        indexes_sql = [
            "CREATE INDEX IF NOT EXISTS idx_backtest_trades_runid ON backtest_trades(runid);",
            "CREATE INDEX IF NOT EXISTS idx_backtest_trades_ticker ON backtest_trades(ticker);",
            "CREATE INDEX IF NOT EXISTS idx_backtest_trades_entry_time ON backtest_trades(entry_time);",
            "CREATE INDEX IF NOT EXISTS idx_backtest_trades_news_event ON backtest_trades(news_event);",
        ]
        
        for index_sql in indexes_sql:
            cursor.execute(index_sql)
        
        conn.commit()
        print("✅ backtest_trades table created successfully!")
        
        # Check if table exists and show structure
        cursor.execute("""
            SELECT column_name, data_type, is_nullable 
            FROM information_schema.columns 
            WHERE table_name = 'backtest_trades' 
            ORDER BY ordinal_position;
        """)
        
        columns = cursor.fetchall()
        print("\n📋 Table structure:")
        for col in columns:
            print(f"  {col[0]}: {col[1]} ({'NULL' if col[2] == 'YES' else 'NOT NULL'})")
            
    except Exception as e:
        print(f"❌ Error creating table: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    create_backtest_trades_table()

