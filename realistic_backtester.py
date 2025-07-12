#!/usr/bin/env python3
"""
Realistic Backtesting Engine for Finespresso Analytics

This backtester implements:
1. ML model predictions based on news content
2. Realistic trade timing (market open, 1-minute delays)
3. Intraday data from yfinance
4. Stop loss and profit taking rules
5. Market close exits
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta, time
import pytz
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

from postgres_data_manager import PostgresDataManager

class NewsClassifier:
    """ML model for predicting price direction based on news content"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2
        )
        self.model = LogisticRegression(
            random_state=42,
            max_iter=1000,
            class_weight='balanced'
        )
        self.is_trained = False
        
    def prepare_features(self, texts: List[str]) -> np.ndarray:
        """Extract features from news text"""
        # Clean and prepare text
        cleaned_texts = []
        for text in texts:
            if pd.isna(text):
                text = ""
            # Basic text cleaning
            text = str(text).lower()
            text = text.replace('\n', ' ').replace('\r', ' ')
            cleaned_texts.append(text)
        
        return cleaned_texts
    
    def train(self, news_data: pd.DataFrame, min_samples: int = 100) -> Dict:
        """Train the classifier on news data"""
        
        print("🤖 Training ML model for news classification...")
        
        # Prepare training data
        valid_data = news_data.dropna(subset=['title', 'actual_side'])
        
        if len(valid_data) < min_samples:
            raise ValueError(f"Insufficient training data: {len(valid_data)} < {min_samples}")
        
        # Combine title and company for features
        texts = []
        for _, row in valid_data.iterrows():
            title = str(row.get('title', ''))
            company = str(row.get('company', ''))
            event = str(row.get('event', ''))
            combined_text = f"{title} {company} {event}"
            texts.append(combined_text)
        
        # Prepare features and labels
        cleaned_texts = self.prepare_features(texts)
        labels = valid_data['actual_side'].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            cleaned_texts, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Fit vectorizer and transform
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)
        
        # Train model
        self.model.fit(X_train_vec, y_train)
        
        # Evaluate
        train_pred = self.model.predict(X_train_vec)
        test_pred = self.model.predict(X_test_vec)
        
        train_acc = accuracy_score(y_train, train_pred)
        test_acc = accuracy_score(y_test, test_pred)
        
        self.is_trained = True
        
        results = {
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'feature_count': X_train_vec.shape[1]
        }
        
        print(f"✅ Model trained successfully!")
        print(f"   Training accuracy: {train_acc:.3f}")
        print(f"   Test accuracy: {test_acc:.3f}")
        print(f"   Features: {X_train_vec.shape[1]}")
        
        return results
    
    def predict(self, news_text: str, company: str = "", event: str = "") -> Tuple[str, float]:
        """Predict direction and confidence for a news item"""
        
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Prepare text
        combined_text = f"{news_text} {company} {event}"
        cleaned_text = self.prepare_features([combined_text])
        
        # Vectorize and predict
        text_vec = self.vectorizer.transform(cleaned_text)
        prediction = self.model.predict(text_vec)[0]
        probabilities = self.model.predict_proba(text_vec)[0]
        
        # Get confidence (max probability)
        confidence = max(probabilities)
        
        return prediction, confidence
    
    def save_model(self, filepath: str):
        """Save trained model to disk"""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        model_data = {
            'vectorizer': self.vectorizer,
            'model': self.model,
            'is_trained': self.is_trained
        }
        joblib.dump(model_data, filepath)
        print(f"💾 Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model from disk"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model_data = joblib.load(filepath)
        self.vectorizer = model_data['vectorizer']
        self.model = model_data['model']
        self.is_trained = model_data['is_trained']
        print(f"📂 Model loaded from {filepath}")


class RealisticBacktester:
    """Realistic backtesting engine with proper trade timing and risk management"""
    
    def __init__(self, initial_capital: float = 100000, data_manager=None):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.positions = {}
        self.trade_log = []
        self.daily_pnl = []
        self.data_manager = data_manager  # PostgreSQL data manager for saving results
        
        # Risk management parameters
        self.max_position_size = 0.05  # 5% of capital per trade
        self.stop_loss_pct = 0.02      # 2% stop loss
        self.take_profit_pct = 0.04    # 4% take profit
        self.min_confidence = 0.6      # Minimum model confidence
        
        # Market timing
        self.market_open = time(9, 30)   # 9:30 AM
        self.market_close = time(16, 0)  # 4:00 PM
        self.timezone = pytz.timezone('US/Eastern')
        
        # ML model
        self.classifier = NewsClassifier()
        
    def is_market_hours(self, dt: datetime) -> bool:
        """Check if datetime is during market hours"""
        if dt.tzinfo is None:
            dt = self.timezone.localize(dt)
        
        market_time = dt.time()
        return self.market_open <= market_time <= self.market_close
    
    def get_entry_time(self, news_time: datetime) -> datetime:
        """Calculate realistic entry time based on news timing"""
        
        if news_time.tzinfo is None:
            news_time = self.timezone.localize(news_time)
        
        news_time_only = news_time.time()
        
        # Pre-market news (before 9:30 AM) -> enter at market open
        if news_time_only < self.market_open:
            entry_time = news_time.replace(
                hour=self.market_open.hour,
                minute=self.market_open.minute,
                second=0,
                microsecond=0
            )
        
        # After-market news (after 4:00 PM) -> enter at next market open
        elif news_time_only > self.market_close:
            next_day = news_time + timedelta(days=1)
            # Skip weekends (simplified)
            while next_day.weekday() >= 5:  # Saturday = 5, Sunday = 6
                next_day += timedelta(days=1)
            
            entry_time = next_day.replace(
                hour=self.market_open.hour,
                minute=self.market_open.minute,
                second=0,
                microsecond=0
            )
        
        # During market hours -> enter 1 minute later
        else:
            entry_time = news_time + timedelta(minutes=1)
        
        return entry_time
    
    def get_intraday_data(self, ticker: str, date: datetime, 
                         entry_time: datetime) -> Optional[pd.DataFrame]:
        """Get intraday price data from yfinance"""
        
        try:
            # Convert to date string
            date_str = date.strftime('%Y-%m-%d')
            
            # Get 1-minute data for the day
            stock = yf.Ticker(ticker)
            data = stock.history(
                start=date_str,
                end=(date + timedelta(days=1)).strftime('%Y-%m-%d'),
                interval='1m'
            )
            
            if data.empty:
                return None
            
            # Filter data from entry time onwards
            if entry_time.tzinfo is None:
                entry_time = self.timezone.localize(entry_time)
            
            # Convert data index to same timezone
            if data.index.tz is None:
                data.index = data.index.tz_localize('US/Eastern')
            else:
                data.index = data.index.tz_convert('US/Eastern')
            
            # Filter from entry time
            intraday_data = data[data.index >= entry_time]
            
            return intraday_data
            
        except Exception as e:
            print(f"⚠️  Error getting intraday data for {ticker}: {e}")
            return None
    
    def calculate_position_size(self, price: float, confidence: float) -> int:
        """Calculate position size based on capital and confidence"""
        
        # Base position size as percentage of capital
        base_size = self.capital * self.max_position_size
        
        # Adjust by confidence (higher confidence = larger position)
        confidence_multiplier = confidence  # 0.6-1.0 range
        adjusted_size = base_size * confidence_multiplier
        
        # Convert to number of shares
        shares = int(adjusted_size / price)
        
        # Ensure we don't exceed available capital
        max_shares = int(self.capital / price)
        shares = min(shares, max_shares)
        
        return max(0, shares)
    
    def execute_realistic_trade(self, news_row: pd.Series) -> Optional[Dict]:
        """Execute a realistic trade based on news and ML prediction"""
        
        # Extract news information
        ticker = news_row['ticker']
        news_time = pd.to_datetime(news_row['published_date'])
        title = news_row.get('title', '')
        company = news_row.get('company', '')
        event = news_row.get('event', '')
        
        # Skip if no ticker
        if pd.isna(ticker) or ticker.strip() == '':
            return None
        
        # Make ML prediction
        try:
            prediction, confidence = self.classifier.predict(title, company, event)
        except Exception as e:
            print(f"⚠️  Prediction error for {ticker}: {e}")
            return None
        
        # Skip if confidence too low
        if confidence < self.min_confidence:
            return None
        
        # Calculate entry time
        entry_time = self.get_entry_time(news_time)
        
        # Get intraday data
        intraday_data = self.get_intraday_data(ticker, entry_time.date(), entry_time)
        
        if intraday_data is None or intraday_data.empty:
            return None
        
        # Get entry price (first available price after entry time)
        entry_price = intraday_data.iloc[0]['Open']
        
        # Calculate position size
        shares = self.calculate_position_size(entry_price, confidence)
        
        if shares == 0:
            return None
        
        # Execute entry
        trade_cost = shares * entry_price
        if trade_cost > self.capital:
            return None
        
        self.capital -= trade_cost
        
        # Calculate stop loss and take profit levels
        if prediction == 'UP':
            stop_loss_price = entry_price * (1 - self.stop_loss_pct)
            take_profit_price = entry_price * (1 + self.take_profit_pct)
            direction = 'LONG'
        else:  # DOWN
            stop_loss_price = entry_price * (1 + self.stop_loss_pct)
            take_profit_price = entry_price * (1 - self.take_profit_pct)
            direction = 'SHORT'
        
        # Simulate intraday price action and exit logic
        exit_info = self.simulate_exit(
            intraday_data, entry_price, stop_loss_price, 
            take_profit_price, direction, entry_time
        )
        
        # Calculate P&L
        exit_price = exit_info['exit_price']
        exit_time = exit_info['exit_time']
        exit_reason = exit_info['exit_reason']
        
        if direction == 'LONG':
            pnl = shares * (exit_price - entry_price)
        else:  # SHORT
            pnl = shares * (entry_price - exit_price)
        
        self.capital += shares * exit_price  # Return capital from sale
        
        # Create trade record
        trade = {
            'entry_time': entry_time,
            'exit_time': exit_time,
            'ticker': ticker,
            'direction': direction,
            'prediction': prediction,
            'confidence': confidence,
            'shares': shares,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'stop_loss_price': stop_loss_price,
            'take_profit_price': take_profit_price,
            'exit_reason': exit_reason,
            'pnl': pnl,
            'pnl_pct': (pnl / trade_cost) * 100,
            'capital_after': self.capital,
            'news_title': title,
            'news_event': event,
            'holding_minutes': (exit_time - entry_time).total_seconds() / 60
        }
        
        self.trade_log.append(trade)
        
        return trade
    
    def simulate_exit(self, intraday_data: pd.DataFrame, entry_price: float,
                     stop_loss_price: float, take_profit_price: float,
                     direction: str, entry_time: datetime) -> Dict:
        """Simulate realistic exit based on stop loss, take profit, or market close"""
        
        # Default exit at market close
        market_close_time = entry_time.replace(
            hour=self.market_close.hour,
            minute=self.market_close.minute,
            second=0,
            microsecond=0
        )
        
        exit_price = intraday_data.iloc[-1]['Close']  # Default to last price
        exit_time = market_close_time
        exit_reason = 'MARKET_CLOSE'
        
        # Check each minute for stop loss or take profit
        for timestamp, row in intraday_data.iterrows():
            current_price = row['Close']
            low_price = row['Low']
            high_price = row['High']
            
            if direction == 'LONG':
                # Check stop loss (price went down)
                if low_price <= stop_loss_price:
                    exit_price = stop_loss_price
                    exit_time = timestamp
                    exit_reason = 'STOP_LOSS'
                    break
                
                # Check take profit (price went up)
                if high_price >= take_profit_price:
                    exit_price = take_profit_price
                    exit_time = timestamp
                    exit_reason = 'TAKE_PROFIT'
                    break
            
            else:  # SHORT
                # Check stop loss (price went up)
                if high_price >= stop_loss_price:
                    exit_price = stop_loss_price
                    exit_time = timestamp
                    exit_reason = 'STOP_LOSS'
                    break
                
                # Check take profit (price went down)
                if low_price <= take_profit_price:
                    exit_price = take_profit_price
                    exit_time = timestamp
                    exit_reason = 'TAKE_PROFIT'
                    break
        
        return {
            'exit_price': exit_price,
            'exit_time': exit_time,
            'exit_reason': exit_reason
        }
    
    def run_backtest(self, news_data: pd.DataFrame, 
                    start_date: str = None, end_date: str = None) -> Dict:
        """Run complete realistic backtest"""
        
        print("🚀 Starting Realistic Backtesting...")
        print("=" * 50)
        
        # Filter by date if specified
        if start_date or end_date:
            news_data = news_data.copy()
            news_data['published_date'] = pd.to_datetime(news_data['published_date'])
            
            if start_date:
                news_data = news_data[news_data['published_date'] >= start_date]
            if end_date:
                news_data = news_data[news_data['published_date'] <= end_date]
        
        print(f"📊 Processing {len(news_data)} news items...")
        
        # Train ML model if not already trained
        if not self.classifier.is_trained:
            print("🤖 Training ML model...")
            self.classifier.train(news_data)
        
        # Process each news item
        successful_trades = 0
        skipped_predictions = 0
        skipped_data = 0
        
        for idx, row in news_data.iterrows():
            if idx % 50 == 0:
                print(f"📈 Processed {idx}/{len(news_data)} items...")
            
            trade = self.execute_realistic_trade(row)
            
            if trade is not None:
                successful_trades += 1
                print(f"✅ Trade executed: {trade['ticker']} {trade['direction']} "
                      f"${trade['pnl']:.2f} ({trade['exit_reason']})")
            else:
                # Could be skipped for various reasons
                continue
        
        # Calculate final results
        final_capital = self.capital
        total_pnl = final_capital - self.initial_capital
        total_return_pct = (total_pnl / self.initial_capital) * 100
        
        if self.trade_log:
            winning_trades = len([t for t in self.trade_log if t['pnl'] > 0])
            losing_trades = len([t for t in self.trade_log if t['pnl'] < 0])
            win_rate = (winning_trades / len(self.trade_log)) * 100
            
            avg_win = np.mean([t['pnl'] for t in self.trade_log if t['pnl'] > 0]) if winning_trades > 0 else 0
            avg_loss = np.mean([t['pnl'] for t in self.trade_log if t['pnl'] < 0]) if losing_trades > 0 else 0
            
            profit_factor = abs(avg_win * winning_trades / (avg_loss * losing_trades)) if avg_loss != 0 and losing_trades > 0 else float('inf')
        else:
            winning_trades = losing_trades = 0
            win_rate = 0
            avg_win = avg_loss = 0
            profit_factor = 0
        
        results = {
            'initial_capital': self.initial_capital,
            'final_capital': final_capital,
            'total_pnl': total_pnl,
            'total_return_pct': total_return_pct,
            'total_trades': len(self.trade_log),
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate_pct': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'news_items_processed': len(news_data),
            'successful_trades': successful_trades,
            'trade_log': self.trade_log
        }
        
        print("\n📊 REALISTIC BACKTEST RESULTS")
        print("=" * 40)
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        print(f"Final Capital: ${final_capital:,.2f}")
        print(f"Total P&L: ${total_pnl:,.2f}")
        print(f"Total Return: {total_return_pct:.2f}%")
        print(f"Total Trades: {len(self.trade_log)}")
        print(f"Win Rate: {win_rate:.1f}%")
        print(f"Profit Factor: {profit_factor:.2f}")
        print(f"News Items Processed: {len(news_data)}")
        print(f"Trade Execution Rate: {successful_trades/len(news_data)*100:.1f}%")
        
        # Save results to database if data manager is available
        if self.data_manager:
            try:
                run_id = self.save_backtest_results(results, start_date, end_date)
                results['run_id'] = run_id
                print(f"💾 Results saved to database with run_id: {run_id}")
            except Exception as e:
                print(f"⚠️  Failed to save results to database: {e}")
        
        return results

    def save_backtest_results(self, results: Dict, start_date: str = None, end_date: str = None) -> str:
        """Save backtest results to PostgreSQL database"""
        
        if not self.data_manager:
            raise ValueError("Data manager not available for saving results")
        
        import uuid
        from datetime import datetime
        
        # Generate unique run ID
        run_id = str(uuid.uuid4())
        
        # Prepare backtest summary data
        summary_data = {
            'run_id': run_id,
            'model_name': 'realistic_ml_backtester',
            'start_date': start_date or (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'),
            'end_date': end_date or datetime.now().strftime('%Y-%m-%d'),
            'initial_capital': results['initial_capital'],
            'final_capital': results['final_capital'],
            'total_pnl': results['total_pnl'],
            'total_return_pct': results['total_return_pct'],
            'total_trades': results['total_trades'],
            'winning_trades': results['winning_trades'],
            'losing_trades': results['losing_trades'],
            'win_rate_pct': results['win_rate_pct'],
            'profit_factor': results['profit_factor'],
            'news_items_processed': results['news_items_processed'],
            'execution_rate_pct': (results['successful_trades'] / results['news_items_processed']) * 100,
            'avg_win': results['avg_win'],
            'avg_loss': results['avg_loss'],
            'data_source': 'postgresql_production',
            'notes': f"Realistic backtester with ML predictions. Stop loss: {self.stop_loss_pct*100:.1f}%, Take profit: {self.take_profit_pct*100:.1f}%, Min confidence: {self.min_confidence*100:.1f}%"
        }
        
        # Save summary to database
        self.data_manager.save_backtest_summary(summary_data)
        
        # Save individual trades to database
        if results['trade_log']:
            for trade in results['trade_log']:
                trade_data = {
                    'published_date': trade['entry_time'].strftime('%Y-%m-%d'),
                    'market': self.classify_market_session(trade['entry_time']),
                    'entry_time': trade['entry_time'].strftime('%Y-%m-%d %H:%M:%S'),
                    'exit_time': trade['exit_time'].strftime('%Y-%m-%d %H:%M:%S'),
                    'ticker': trade['ticker'],
                    'direction': trade['direction'],
                    'shares': int(trade['shares']),
                    'entry_price': float(trade['entry_price']),
                    'exit_price': float(trade['exit_price']),
                    'target_price': float(trade['take_profit_price']),
                    'stop_price': float(trade['stop_loss_price']),
                    'hit_target': trade['exit_reason'] == 'TAKE_PROFIT',
                    'hit_stop': trade['exit_reason'] == 'STOP_LOSS',
                    'pnl': float(trade['pnl']),
                    'pnl_pct': float(trade['pnl_pct']),
                    'capital_after': float(trade['capital_after']),
                    'news_event': trade.get('news_event', ''),
                    'link': '',  # Could add news link if available
                    'runid': run_id,
                    'rundate': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                
                self.data_manager.log_trade(trade_data)
        
        return run_id
    
    def classify_market_session(self, dt: datetime) -> str:
        """Classify market session based on time"""
        if dt.tzinfo is None:
            dt = self.timezone.localize(dt)
        
        market_time = dt.time()
        
        if market_time < self.market_open:
            return 'pre_market'
        elif market_time > self.market_close:
            return 'after_market'
        else:
            return 'regular_market'


if __name__ == "__main__":
    # Example usage
    dm = PostgresDataManager()
    
    # Get sample data
    news_data = dm.get_news_with_price_moves(limit=100)
    
    # Run realistic backtest
    backtester = RealisticBacktester(initial_capital=100000)
    results = backtester.run_backtest(news_data)
    
    # Save results
    trades_df = pd.DataFrame(results['trade_log'])
    if not trades_df.empty:
        trades_df.to_csv('reports/backtesting/realistic_backtest_trades.csv', index=False)
        print(f"💾 Trade log saved to realistic_backtest_trades.csv")

