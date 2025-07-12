#!/usr/bin/env python3
"""
Test script for the realistic backtester
"""

import pandas as pd
from datetime import datetime, timedelta
from realistic_backtester import RealisticBacktester, NewsClassifier
from postgres_data_manager import PostgresDataManager

def test_realistic_backtester():
    """Test the realistic backtester with a small dataset"""
    
    print("🧪 TESTING REALISTIC BACKTESTER")
    print("=" * 50)
    
    # Initialize data manager
    dm = PostgresDataManager()
    
    # Get sufficient data for testing (need 100+ for training)
    print("📊 Fetching test data...")
    news_data = dm.get_news_with_price_moves(limit=150)
    
    print(f"✅ Retrieved {len(news_data)} news items")
    print()
    
    # Show sample data
    print("📋 SAMPLE DATA:")
    print("-" * 30)
    sample_cols = ['ticker', 'title', 'published_date', 'actual_side', 'price_change_percentage']
    available_cols = [col for col in sample_cols if col in news_data.columns]
    print(news_data[available_cols].head(3))
    print()
    
    # Initialize backtester
    print("🚀 Initializing realistic backtester...")
    backtester = RealisticBacktester(initial_capital=50000)  # Smaller capital for testing
    
    # Adjust parameters for testing
    backtester.min_confidence = 0.5  # Lower confidence threshold for testing
    backtester.max_position_size = 0.1  # Larger positions for testing
    
    print(f"💰 Initial capital: ${backtester.initial_capital:,}")
    print(f"🎯 Min confidence: {backtester.min_confidence}")
    print(f"📊 Max position size: {backtester.max_position_size*100}%")
    print()
    
    # Run backtest
    try:
        print("🔄 Running realistic backtest...")
        results = backtester.run_backtest(news_data)
        
        print("\n✅ BACKTEST COMPLETED SUCCESSFULLY!")
        print("=" * 40)
        
        # Display detailed results
        print(f"📈 Performance Summary:")
        print(f"   Initial Capital: ${results['initial_capital']:,.2f}")
        print(f"   Final Capital: ${results['final_capital']:,.2f}")
        print(f"   Total P&L: ${results['total_pnl']:,.2f}")
        print(f"   Return: {results['total_return_pct']:.2f}%")
        print()
        
        print(f"📊 Trading Statistics:")
        print(f"   Total Trades: {results['total_trades']}")
        print(f"   Winning Trades: {results['winning_trades']}")
        print(f"   Losing Trades: {results['losing_trades']}")
        print(f"   Win Rate: {results['win_rate_pct']:.1f}%")
        print(f"   Profit Factor: {results['profit_factor']:.2f}")
        print()
        
        print(f"🔍 Execution Analysis:")
        print(f"   News Items Processed: {results['news_items_processed']}")
        print(f"   Successful Trades: {results['successful_trades']}")
        print(f"   Execution Rate: {results['successful_trades']/results['news_items_processed']*100:.1f}%")
        print()
        
        # Show sample trades
        if results['trade_log']:
            print("💼 SAMPLE TRADES:")
            print("-" * 50)
            trades_df = pd.DataFrame(results['trade_log'])
            
            # Show first few trades
            for i, trade in enumerate(trades_df.head(3).to_dict('records')):
                print(f"Trade {i+1}:")
                print(f"  {trade['ticker']} {trade['direction']} - {trade['prediction']}")
                print(f"  Entry: ${trade['entry_price']:.2f} @ {trade['entry_time']}")
                print(f"  Exit: ${trade['exit_price']:.2f} @ {trade['exit_time']} ({trade['exit_reason']})")
                print(f"  P&L: ${trade['pnl']:.2f} ({trade['pnl_pct']:.1f}%)")
                print(f"  Confidence: {trade['confidence']:.2f}")
                print(f"  Holding: {trade['holding_minutes']:.0f} minutes")
                print()
            
            # Save detailed results
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            trades_file = f"reports/backtesting/realistic_test_trades_{timestamp}.csv"
            trades_df.to_csv(trades_file, index=False)
            print(f"💾 Detailed trades saved to: {trades_file}")
            
            # Create summary
            summary_data = {
                'metric': ['Initial Capital', 'Final Capital', 'Total P&L', 'Return %', 
                          'Total Trades', 'Win Rate %', 'Profit Factor', 'Execution Rate %'],
                'value': [results['initial_capital'], results['final_capital'], 
                         results['total_pnl'], results['total_return_pct'],
                         results['total_trades'], results['win_rate_pct'], 
                         results['profit_factor'], 
                         results['successful_trades']/results['news_items_processed']*100]
            }
            summary_df = pd.DataFrame(summary_data)
            summary_file = f"reports/backtesting/realistic_test_summary_{timestamp}.csv"
            summary_df.to_csv(summary_file, index=False)
            print(f"📊 Summary saved to: {summary_file}")
        
        return results
        
    except Exception as e:
        print(f"❌ Error during backtesting: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_ml_model():
    """Test the ML model component separately"""
    
    print("\n🤖 TESTING ML MODEL")
    print("=" * 30)
    
    # Get training data
    dm = PostgresDataManager()
    training_data = dm.get_news_with_price_moves(limit=200)
    
    print(f"📊 Training data: {len(training_data)} samples")
    
    # Test classifier
    classifier = NewsClassifier()
    
    try:
        # Train model
        results = classifier.train(training_data)
        
        print(f"✅ Model training successful!")
        print(f"   Training accuracy: {results['train_accuracy']:.3f}")
        print(f"   Test accuracy: {results['test_accuracy']:.3f}")
        print(f"   Features: {results['feature_count']}")
        
        # Test prediction
        sample_news = training_data.iloc[0]
        prediction, confidence = classifier.predict(
            sample_news['title'], 
            sample_news.get('company', ''), 
            sample_news.get('event', '')
        )
        
        print(f"\n🎯 Sample prediction:")
        print(f"   News: {sample_news['title'][:100]}...")
        print(f"   Predicted: {prediction}")
        print(f"   Confidence: {confidence:.3f}")
        print(f"   Actual: {sample_news['actual_side']}")
        
        return True
        
    except Exception as e:
        print(f"❌ ML model error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Test ML model first
    ml_success = test_ml_model()
    
    if ml_success:
        # Test full backtester
        results = test_realistic_backtester()
        
        if results:
            print("\n🎉 ALL TESTS PASSED!")
            print("The realistic backtester is working correctly.")
        else:
            print("\n❌ BACKTEST TEST FAILED!")
    else:
        print("\n❌ ML MODEL TEST FAILED!")
        print("Cannot proceed with backtesting without working ML model.")

