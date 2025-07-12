# Backtesting Reports

This directory contains comprehensive backtesting reports and analysis for the Finespresso Analytics system.

## 📁 File Structure

### **Detailed Trade Reports**
- `backtest_trades_YYYYMMDD_HHMMSS.csv` - Individual trade execution details
- `detailed_backtest_trades.csv` - Sample detailed trades from testing

### **Summary Reports**
- `backtest_summary_YYYYMMDD_HHMMSS.csv` - Comprehensive backtest summaries
- `backtest_summary.csv` - Manual summary with metadata
- `publisher_comparison_YYYYMMDD_HHMMSS.csv` - Publisher performance comparison

### **Legacy Reports**
- `backtest_results.csv` - Initial backtest results
- `backtest_results_improved.csv` - Enhanced results with better column names
- `postgres_backtest_results.csv` - PostgreSQL-based backtest results
- `publisher_backtest_results.csv` - Publisher-specific results

## 📊 Report Formats

### **Detailed Trades CSV Columns**
```
published_date, market, entry_time, exit_time, ticker, direction,
shares, entry_price, exit_price, target_price, stop_price,
hit_target, hit_stop, pnl, pnl_pct, capital_after,
news_event, link, runid, rundate
```

### **Backtest Summary CSV Columns**
```
run_id, timestamp, model_name, publisher, start_date, end_date,
initial_capital, final_capital, total_pnl, return_percent,
total_trades, winning_trades, losing_trades, win_rate_percent,
target_hits, stop_hits, max_drawdown, avg_win, avg_loss,
profit_factor, news_articles_used, price_moves_used, notes
```

### **Publisher Comparison CSV Columns**
```
Publisher, Run_ID, Total_Trades, Win_Rate_Percent, Return_Percent,
Total_PnL_USD, Final_Capital_USD, Max_Drawdown_Percent,
Target_Hits, Stop_Hits, Profit_Factor, Avg_Win_USD, Avg_Loss_USD
```

## 🚀 Generating New Reports

### **Comprehensive Reports**
```bash
python generate_backtest_reports.py --capital 100000 --limit 100
```

### **Publisher-Specific Reports**
```bash
python run_detailed_publisher_backtests.py --publishers globenewswire_biotech all
```

### **Database Verification**
```bash
python verify_backtest_trades.py
```

## 📈 Key Metrics

### **Performance Indicators**
- **Return Percent**: Total return on initial capital
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Ratio of gross profit to gross loss
- **Max Drawdown**: Maximum peak-to-trough decline
- **Target/Stop Hits**: Risk management effectiveness

### **Publisher Comparison**
- **globenewswire_biotech**: Biotech-focused news analysis
- **all**: All publishers combined analysis

## 🔍 Analysis Tips

1. **Compare Publishers**: Use publisher comparison reports to identify best-performing news sources
2. **Trade Analysis**: Review detailed trades to understand entry/exit patterns
3. **Risk Management**: Monitor target/stop hit ratios for strategy optimization
4. **Drawdown Analysis**: Track maximum drawdown for risk assessment
5. **Profit Factor**: Values > 1.0 indicate profitable strategies

## 📅 Report Timestamps

All reports include timestamps in the format `YYYYMMDD_HHMMSS` for version tracking and historical analysis.

## 🗄️ Database Integration

Reports are automatically saved to PostgreSQL database tables:
- `backtest_trades` - Individual trade records
- `backtest_summary` - Aggregated backtest results

Use `verify_backtest_trades.py` to check database consistency and retrieve stored results.

