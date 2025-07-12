#!/usr/bin/env python3
"""
Diagnostic script to analyze backtesting logic and identify issues
"""

import pandas as pd
from postgres_data_manager import PostgresDataManager
from datetime import datetime, timedelta

def diagnose_backtest_data():
    """Analyze the data being used for backtesting"""
    
    print("🔍 BACKTESTING LOGIC DIAGNOSIS")
    print("=" * 50)
    
    dm = PostgresDataManager()
    
    # Get sample data
    data = dm.get_news_with_price_moves(limit=20)
    
    print(f"📊 Total records retrieved: {len(data)}")
    print()
    
    # Analyze predictions
    print("🎯 PREDICTION ANALYSIS:")
    print("-" * 30)
    print("Predicted side distribution:")
    print(data['predicted_side'].value_counts(dropna=False))
    print()
    print("Actual side distribution:")
    print(data['actual_side'].value_counts(dropna=False))
    print()
    
    # Check for null values that would be filtered out
    null_checks = {
        'actual_side': data['actual_side'].isnull().sum(),
        'begin_price': data['begin_price'].isnull().sum(),
        'end_price': data['end_price'].isnull().sum(),
        'predicted_side': data['predicted_side'].isnull().sum(),
        'ticker': data['ticker'].isnull().sum()
    }
    
    print("❌ NULL VALUES (would be filtered out):")
    print("-" * 40)
    for field, count in null_checks.items():
        print(f"{field}: {count}")
    print()
    
    # Simulate the filtering logic
    print("🔄 SIMULATING BACKTEST FILTERING:")
    print("-" * 40)
    
    total_records = len(data)
    print(f"Starting records: {total_records}")
    
    # Filter 1: Remove null actual_side, begin_price, end_price
    filtered_1 = data.dropna(subset=['actual_side', 'begin_price', 'end_price'])
    print(f"After null filtering: {len(filtered_1)} (removed {total_records - len(filtered_1)})")
    
    # Filter 2: Remove empty tickers
    filtered_2 = filtered_1[filtered_1['ticker'].notna() & (filtered_1['ticker'].str.strip() != '')]
    print(f"After ticker filtering: {len(filtered_2)} (removed {len(filtered_1) - len(filtered_2)})")
    
    # Filter 3: Remove invalid prices
    filtered_3 = filtered_2[(filtered_2['begin_price'] > 0) & (filtered_2['end_price'] > 0)]
    print(f"After price filtering: {len(filtered_3)} (removed {len(filtered_2) - len(filtered_3)})")
    
    # Filter 4: Remove invalid predictions
    filtered_4 = filtered_3[filtered_3['predicted_side'].isin(['UP', 'DOWN'])]
    print(f"After prediction filtering: {len(filtered_4)} (removed {len(filtered_3) - len(filtered_4)})")
    
    print()
    print("🎯 FINAL TRADEABLE RECORDS:")
    print("-" * 30)
    print(f"Records that would become trades: {len(filtered_4)}")
    print(f"Percentage of original data: {len(filtered_4)/total_records*100:.1f}%")
    
    # Analyze prediction accuracy
    if len(filtered_4) > 0:
        print()
        print("📈 PREDICTION ACCURACY ANALYSIS:")
        print("-" * 35)
        
        correct_predictions = (filtered_4['predicted_side'] == filtered_4['actual_side']).sum()
        total_predictions = len(filtered_4)
        accuracy = correct_predictions / total_predictions * 100
        
        print(f"Correct predictions: {correct_predictions}")
        print(f"Total predictions: {total_predictions}")
        print(f"Accuracy: {accuracy:.1f}%")
        
        # Break down by prediction type
        up_predictions = filtered_4[filtered_4['predicted_side'] == 'UP']
        down_predictions = filtered_4[filtered_4['predicted_side'] == 'DOWN']
        
        if len(up_predictions) > 0:
            up_correct = (up_predictions['predicted_side'] == up_predictions['actual_side']).sum()
            up_accuracy = up_correct / len(up_predictions) * 100
            print(f"UP predictions: {len(up_predictions)} (accuracy: {up_accuracy:.1f}%)")
        
        if len(down_predictions) > 0:
            down_correct = (down_predictions['predicted_side'] == down_predictions['actual_side']).sum()
            down_accuracy = down_correct / len(down_predictions) * 100
            print(f"DOWN predictions: {len(down_predictions)} (accuracy: {down_accuracy:.1f}%)")
    
    print()
    print("🚨 POTENTIAL ISSUES IDENTIFIED:")
    print("-" * 35)
    
    issues = []
    
    if len(filtered_4) == total_records:
        issues.append("❌ No filtering happening - all records become trades")
    
    if len(filtered_4) == 20:  # Our limit
        issues.append("❌ Exactly limit number of trades - suspicious")
    
    if len(data['predicted_side'].unique()) <= 2:
        issues.append("✅ Predictions look valid (UP/DOWN)")
    
    if data['predicted_side'].isnull().sum() == 0:
        issues.append("❌ No null predictions - every record has a prediction")
    
    for issue in issues:
        print(issue)
    
    print()
    print("💡 RECOMMENDATIONS:")
    print("-" * 20)
    print("1. Add confidence thresholds for predictions")
    print("2. Add minimum price movement filters")
    print("3. Add liquidity/volume filters")
    print("4. Add time-based filters (market hours)")
    print("5. Add position sizing limits")
    print("6. Add realistic trade execution delays")

if __name__ == "__main__":
    diagnose_backtest_data()

