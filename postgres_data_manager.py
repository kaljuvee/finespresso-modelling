#!/usr/bin/env python3
"""
PostgreSQL Data Manager for Finespresso Analytics
Connects to the production PostgreSQL database and provides data access methods.
"""

import pandas as pd
import psycopg2
from sqlalchemy import create_engine, text
from datetime import datetime, timedelta
import os
from typing import Optional, List, Dict, Any
import uuid


class PostgresDataManager:
    def __init__(self, database_url: str = None):
        """Initialize connection to PostgreSQL database"""
        self.database_url = database_url or os.getenv('DATABASE_URL', 
            'postgresql://finespresso_db_user:XZ0o6UkxcV0poBcLDQf6RGXwEfWmBlnb@dpg-ctj7u2lumphs73f8t9qg-a.frankfurt-postgres.render.com/finespresso_db')
        
        try:
            self.engine = create_engine(self.database_url)
            print(f"✅ Connected to PostgreSQL database")
            self._test_connection()
        except Exception as e:
            print(f"❌ Failed to connect to database: {e}")
            raise

    def _test_connection(self):
        """Test database connection and show basic stats"""
        try:
            with self.engine.connect() as conn:
                # Test basic queries
                news_count = conn.execute(text("SELECT COUNT(*) FROM news")).scalar()
                price_moves_count = conn.execute(text("SELECT COUNT(*) FROM price_moves")).scalar()
                
                print(f"📊 Database Stats:")
                print(f"   News articles: {news_count:,}")
                print(f"   Price moves: {price_moves_count:,}")
                
        except Exception as e:
            print(f"❌ Connection test failed: {e}")
            raise

    def get_news_with_price_moves(self, start_date: datetime = None, end_date: datetime = None, 
                                  limit: int = None, publisher: str = None) -> pd.DataFrame:
        """
        Get news articles joined with their corresponding price moves
        """
        query = """
        SELECT 
            n.id as news_id,
            n.title,
            n.company,
            n.published_date,
            n.event,
            n.ticker,
            n.yf_ticker,
            n.publisher,
            n.predicted_side,
            n.predicted_move,
            pm.id as price_move_id,
            pm.begin_price,
            pm.end_price,
            pm.price_change,
            pm.price_change_percentage,
            pm.daily_alpha,
            pm.actual_side,
            pm.volume,
            pm.market
        FROM news n
        INNER JOIN price_moves pm ON n.id = pm.news_id
        WHERE n.ticker IS NOT NULL 
        AND n.event IS NOT NULL
        AND pm.actual_side IS NOT NULL
        """
        
        params = {}
        
        if start_date:
            query += " AND n.published_date >= %(start_date)s"
            params['start_date'] = start_date
            
        if end_date:
            query += " AND n.published_date <= %(end_date)s"
            params['end_date'] = end_date
            
        if publisher and publisher.lower() != 'all':
            query += " AND n.publisher = %(publisher)s"
            params['publisher'] = publisher
            
        query += " ORDER BY n.published_date DESC"
        
        if limit:
            query += f" LIMIT {limit}"
        
        try:
            df = pd.read_sql(query, self.engine, params=params)
            publisher_filter = publisher if publisher else "all"
            print(f"📊 Retrieved {len(df)} news articles with price moves (publisher: {publisher_filter})")
            return df
        except Exception as e:
            print(f"❌ Error retrieving news with price moves: {e}")
            return pd.DataFrame()

    def get_unique_events(self) -> List[str]:
        """Get list of unique events from news table"""
        query = "SELECT DISTINCT event FROM news WHERE event IS NOT NULL ORDER BY event"
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(query))
                events = [row[0] for row in result]
                print(f"📋 Found {len(events)} unique events")
                return events
        except Exception as e:
            print(f"❌ Error retrieving events: {e}")
            return []

    def get_unique_publishers(self) -> List[str]:
        """Get list of unique publishers from news table"""
        query = "SELECT DISTINCT publisher FROM news WHERE publisher IS NOT NULL ORDER BY publisher"
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(query))
                publishers = [row[0] for row in result]
                print(f"📋 Found {len(publishers)} unique publishers")
                return publishers
        except Exception as e:
            print(f"❌ Error retrieving publishers: {e}")
            return []

    def get_unique_tickers(self) -> List[str]:
        """Get list of unique tickers from news table"""
        query = "SELECT DISTINCT ticker FROM news WHERE ticker IS NOT NULL ORDER BY ticker"
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(query))
                tickers = [row[0] for row in result]
                print(f"📋 Found {len(tickers)} unique tickers")
                return tickers
        except Exception as e:
            print(f"❌ Error retrieving tickers: {e}")
            return []

    def get_data_summary(self) -> Dict[str, Any]:
        """Get summary statistics about the database"""
        try:
            with self.engine.connect() as conn:
                # Basic counts
                news_count = conn.execute(text("SELECT COUNT(*) FROM news")).scalar()
                price_moves_count = conn.execute(text("SELECT COUNT(*) FROM price_moves")).scalar()
                
                # Date ranges
                date_range = conn.execute(text("""
                    SELECT 
                        MIN(published_date) as earliest_date,
                        MAX(published_date) as latest_date
                    FROM news 
                    WHERE published_date IS NOT NULL
                """)).fetchone()
                
                # Event distribution
                event_dist = pd.read_sql("""
                    SELECT event, COUNT(*) as count 
                    FROM news 
                    WHERE event IS NOT NULL 
                    GROUP BY event 
                    ORDER BY count DESC 
                    LIMIT 10
                """, self.engine)
                
                # Ticker distribution
                ticker_dist = pd.read_sql("""
                    SELECT ticker, COUNT(*) as count 
                    FROM news 
                    WHERE ticker IS NOT NULL 
                    GROUP BY ticker 
                    ORDER BY count DESC 
                    LIMIT 10
                """, self.engine)
                
                return {
                    'news_count': news_count,
                    'price_moves_count': price_moves_count,
                    'earliest_date': date_range[0],
                    'latest_date': date_range[1],
                    'top_events': event_dist.to_dict('records'),
                    'top_tickers': ticker_dist.to_dict('records')
                }
                
        except Exception as e:
            print(f"❌ Error getting data summary: {e}")
            return {}

    def save_model_results(self, results: Dict[str, Any], run_id: str = None) -> bool:
        """Save binary model results to eq_model_results_binary table"""
        if not run_id:
            run_id = str(uuid.uuid4())
            
        try:
            # Prepare data for insertion
            insert_data = {
                'run_id': run_id,
                'timestamp': datetime.now(),
                'event': results.get('event', 'unknown'),
                'accuracy': results.get('accuracy', 0.0),
                'precision': results.get('precision'),
                'recall': results.get('recall'),
                'f1_score': results.get('f1_score'),
                'auc_roc': results.get('auc_roc'),
                'test_sample': results.get('test_sample', 0),
                'training_sample': results.get('training_sample', 0),
                'total_sample': results.get('total_sample', 0),
                'up_accuracy': results.get('up_accuracy'),
                'down_accuracy': results.get('down_accuracy'),
                'total_up': results.get('total_up', 0),
                'total_down': results.get('total_down', 0),
                'correct_up': results.get('correct_up', 0),
                'correct_down': results.get('correct_down', 0),
                'up_predictions_pct': results.get('up_predictions_pct'),
                'down_predictions_pct': results.get('down_predictions_pct')
            }
            
            # Insert into database
            df = pd.DataFrame([insert_data])
            df.to_sql('eq_model_results_binary', self.engine, if_exists='append', index=False)
            
            print(f"✅ Saved model results for event: {results.get('event')} (run_id: {run_id})")
            return True
            
        except Exception as e:
            print(f"❌ Error saving model results: {e}")
            return False

    def create_backtest_summary_table(self):
        """Create backtest_summary table if it doesn't exist"""
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS backtest_summary (
            id SERIAL PRIMARY KEY,
            run_id UUID NOT NULL DEFAULT gen_random_uuid(),
            timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            model_name VARCHAR(255) NOT NULL,
            start_date TIMESTAMP NOT NULL,
            end_date TIMESTAMP NOT NULL,
            initial_capital DOUBLE PRECISION NOT NULL,
            final_capital DOUBLE PRECISION NOT NULL,
            total_pnl DOUBLE PRECISION NOT NULL,
            return_percent DOUBLE PRECISION NOT NULL,
            total_trades INTEGER NOT NULL,
            winning_trades INTEGER NOT NULL,
            losing_trades INTEGER NOT NULL,
            win_rate_percent DOUBLE PRECISION NOT NULL,
            max_drawdown DOUBLE PRECISION,
            sharpe_ratio DOUBLE PRECISION,
            news_articles_used INTEGER,
            price_moves_used INTEGER,
            database_version VARCHAR(100),
            notes TEXT
        );
        """
        
        try:
            with self.engine.connect() as conn:
                conn.execute(text(create_table_sql))
                conn.commit()
                print("✅ Created/verified backtest_summary table")
                return True
        except Exception as e:
            print(f"❌ Error creating backtest_summary table: {e}")
            return False

    def save_backtest_results(self, results: Dict[str, Any], run_id: str = None) -> bool:
        """Save backtest results to backtest_summary table"""
        if not run_id:
            run_id = str(uuid.uuid4())
            
        try:
            # Prepare data for insertion
            insert_data = {
                'run_id': run_id,
                'timestamp': datetime.now(),
                'model_name': results.get('model_name', 'unknown'),
                'start_date': results.get('start_date'),
                'end_date': results.get('end_date'),
                'initial_capital': results.get('initial_capital', 0.0),
                'final_capital': results.get('final_capital', 0.0),
                'total_pnl': results.get('total_pnl', 0.0),
                'return_percent': results.get('return_percent', 0.0),
                'total_trades': results.get('total_trades', 0),
                'winning_trades': results.get('winning_trades', 0),
                'losing_trades': results.get('losing_trades', 0),
                'win_rate_percent': results.get('win_rate_percent', 0.0),
                'max_drawdown': results.get('max_drawdown'),
                'sharpe_ratio': results.get('sharpe_ratio'),
                'news_articles_used': results.get('news_articles_used', 0),
                'price_moves_used': results.get('price_moves_used', 0),
                'database_version': 'postgresql_production',
                'notes': results.get('notes', '')
            }
            
            # Insert into database
            df = pd.DataFrame([insert_data])
            df.to_sql('backtest_summary', self.engine, if_exists='append', index=False)
            
            print(f"✅ Saved backtest results for model: {results.get('model_name')} (run_id: {run_id})")
            return True
            
        except Exception as e:
            print(f"❌ Error saving backtest results: {e}")
            return False

    def get_recent_backtest_results(self, limit: int = 10) -> pd.DataFrame:
        """Get recent backtest results from backtest_summary table"""
        query = f"""
        SELECT * FROM backtest_summary 
        ORDER BY timestamp DESC 
        LIMIT {limit}
        """
        
        try:
            df = pd.read_sql(query, self.engine)
            print(f"📊 Retrieved {len(df)} recent backtest results")
            return df
        except Exception as e:
            print(f"❌ Error retrieving backtest results: {e}")
            return pd.DataFrame()

    def log_trade(self, trade_data: dict) -> bool:
        """
        Log a detailed trade to the backtest_trades table
        
        Args:
            trade_data: Dictionary containing trade information
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            query = """
            INSERT INTO backtest_trades (
                published_date, market, entry_time, exit_time, ticker, direction,
                shares, entry_price, exit_price, target_price, stop_price,
                hit_target, hit_stop, pnl, pnl_pct, capital_after,
                news_event, link, runid, rundate
            ) VALUES (
                :published_date, :market, :entry_time, :exit_time, :ticker, :direction,
                :shares, :entry_price, :exit_price, :target_price, :stop_price,
                :hit_target, :hit_stop, :pnl, :pnl_pct, :capital_after,
                :news_event, :link, :runid, :rundate
            )
            """
            
            with self.engine.connect() as conn:
                conn.execute(text(query), trade_data)
                conn.commit()
            return True
            
        except Exception as e:
            print(f"Error logging trade: {e}")
            return False
    
    def get_trades_by_runid(self, runid: str) -> List[dict]:
        """
        Get all trades for a specific run ID
        
        Args:
            runid: The run ID to filter by
            
        Returns:
            List of trade dictionaries
        """
        try:
            query = """
            SELECT * FROM backtest_trades 
            WHERE runid = %s 
            ORDER BY entry_time
            """
            
            cursor = self.connection.cursor()
            cursor.execute(query, (runid,))
            
            columns = [desc[0] for desc in cursor.description]
            trades = []
            
            for row in cursor.fetchall():
                trade = dict(zip(columns, row))
                trades.append(trade)
            
            cursor.close()
            return trades
            
        except Exception as e:
            print(f"Error getting trades: {e}")
            return []
    
    def get_trade_summary_by_runid(self, runid: str) -> dict:
        """
        Get trade summary statistics for a specific run ID
        
        Args:
            runid: The run ID to analyze
            
        Returns:
            Dictionary with summary statistics
        """
        try:
            query = """
            SELECT 
                COUNT(*) as total_trades,
                SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
                SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) as losing_trades,
                SUM(CASE WHEN hit_target THEN 1 ELSE 0 END) as target_hits,
                SUM(CASE WHEN hit_stop THEN 1 ELSE 0 END) as stop_hits,
                SUM(pnl) as total_pnl,
                AVG(pnl_pct) as avg_return_pct,
                MAX(capital_after) as final_capital,
                MIN(capital_after) as min_capital,
                MAX(capital_after) - MIN(capital_after) as capital_range
            FROM backtest_trades 
            WHERE runid = %s
            """
            
            cursor = self.connection.cursor()
            cursor.execute(query, (runid,))
            
            result = cursor.fetchone()
            cursor.close()
            
            if result:
                return {
                    'total_trades': result[0] or 0,
                    'winning_trades': result[1] or 0,
                    'losing_trades': result[2] or 0,
                    'target_hits': result[3] or 0,
                    'stop_hits': result[4] or 0,
                    'total_pnl': float(result[5]) if result[5] else 0.0,
                    'avg_return_pct': float(result[6]) if result[6] else 0.0,
                    'final_capital': float(result[7]) if result[7] else 0.0,
                    'min_capital': float(result[8]) if result[8] else 0.0,
                    'capital_range': float(result[9]) if result[9] else 0.0
                }
            else:
                return {}
                
        except Exception as e:
            print(f"Error getting trade summary: {e}")
            return {}

    def get_news_count(self) -> int:
        """Get total count of news articles"""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT COUNT(*) FROM news")).scalar()
                return result or 0
        except Exception as e:
            print(f"Error getting news count: {e}")
            return 0

    def get_price_moves_count(self) -> int:
        """Get total count of price moves"""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT COUNT(*) FROM price_moves")).scalar()
                return result or 0
        except Exception as e:
            print(f"Error getting price moves count: {e}")
            return 0

    def get_publisher_stats(self) -> pd.DataFrame:
        """Get statistics by publisher"""
        try:
            query = """
            SELECT publisher, COUNT(*) as count
            FROM news 
            WHERE publisher IS NOT NULL
            GROUP BY publisher 
            ORDER BY count DESC
            LIMIT 10
            """
            
            with self.engine.connect() as conn:
                result = pd.read_sql(query, conn)
                return result
                
        except Exception as e:
            print(f"Error getting publisher stats: {e}")
            return pd.DataFrame()

    def save_backtest_summary(self, summary_data: Dict) -> None:
        """Save backtest summary to database"""
        try:
            # Insert into backtest_summary table (matching actual table structure)
            insert_query = """
            INSERT INTO backtest_summary (
                run_id, model_name, start_date, end_date, initial_capital, 
                final_capital, total_pnl, return_percent, total_trades, 
                winning_trades, losing_trades, win_rate_percent, 
                news_articles_used, price_moves_used, database_version, notes
            ) VALUES (
                :run_id, :model_name, :start_date, :end_date, :initial_capital,
                :final_capital, :total_pnl, :return_percent, :total_trades,
                :winning_trades, :losing_trades, :win_rate_percent,
                :news_articles_used, :price_moves_used, :database_version, :notes
            )
            """
            
            # Map the data to match table columns and convert numpy types
            mapped_data = {
                'run_id': summary_data['run_id'],
                'model_name': summary_data['model_name'],
                'start_date': summary_data['start_date'],
                'end_date': summary_data['end_date'],
                'initial_capital': float(summary_data['initial_capital']),
                'final_capital': float(summary_data['final_capital']),
                'total_pnl': float(summary_data['total_pnl']),
                'return_percent': float(summary_data['total_return_pct']),  # Map to correct column name
                'total_trades': int(summary_data['total_trades']),
                'winning_trades': int(summary_data['winning_trades']),
                'losing_trades': int(summary_data['losing_trades']),
                'win_rate_percent': float(summary_data['win_rate_pct']),  # Map to correct column name
                'news_articles_used': int(summary_data['news_items_processed']),
                'price_moves_used': int(summary_data['total_trades']),  # Approximate
                'database_version': summary_data['data_source'],
                'notes': summary_data['notes']
            }
            
            with self.engine.connect() as conn:
                conn.execute(text(insert_query), mapped_data)
                conn.commit()
                
            print(f"✅ Backtest summary saved to database")
            
        except Exception as e:
            print(f"❌ Error saving backtest summary: {e}")
            raise


if __name__ == "__main__":
    # Test the PostgreSQL connection
    print("🔗 Testing PostgreSQL Data Manager")
    print("=" * 50)
    
    try:
        # Initialize data manager
        dm = PostgresDataManager()
        
        # Get data summary
        summary = dm.get_data_summary()
        print(f"\n📊 Data Summary:")
        print(f"   Total news articles: {summary.get('news_count', 0):,}")
        print(f"   Total price moves: {summary.get('price_moves_count', 0):,}")
        print(f"   Date range: {summary.get('earliest_date')} to {summary.get('latest_date')}")
        
        # Show top events
        print(f"\n🏷️  Top Events:")
        for event in summary.get('top_events', [])[:5]:
            print(f"   {event['event']}: {event['count']} articles")
        
        # Show top tickers
        print(f"\n📈 Top Tickers:")
        for ticker in summary.get('top_tickers', [])[:5]:
            print(f"   {ticker['ticker']}: {ticker['count']} articles")
        
        # Test data retrieval
        print(f"\n🔍 Testing data retrieval...")
        sample_data = dm.get_news_with_price_moves(limit=5)
        print(f"   Sample data shape: {sample_data.shape}")
        if not sample_data.empty:
            print(f"   Columns: {list(sample_data.columns)}")
        
        # Create backtest summary table
        dm.create_backtest_summary_table()
        
        print(f"\n✅ PostgreSQL Data Manager test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")

