
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib
import os

# Placeholder for DataManager and other necessary imports
# from .data_manager import DataManager
# from .rss_parser import extract_company_and_ticker

class Backtester:
    def __init__(self, initial_capital=100000, model_dir="/home/ubuntu/finespresso-modelling/models/"):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.positions = {}
        self.trade_log = []
        self.model_dir = model_dir
        self.models = {}
        self.vectorizers = {}
        self._load_models()

    def _load_models(self):
        # Load all event-specific models and the all_events fallback model
        for filename in os.listdir(self.model_dir):
            if filename.endswith("_model.joblib"):
                event_name = filename.replace("_model.joblib", "")
                self.models[event_name] = joblib.load(os.path.join(self.model_dir, filename))
            elif filename.endswith("_vectorizer.joblib"):
                event_name = filename.replace("_vectorizer.joblib", "")
                self.vectorizers[event_name] = joblib.load(os.path.join(self.model_dir, filename))
        print(f"Loaded models: {list(self.models.keys())}")

    def _get_model_and_vectorizer(self, event_category):
        model = self.models.get(event_category)
        vectorizer = self.vectorizers.get(event_category)
        if model and vectorizer:
            return model, vectorizer
        else:
            # Fallback to all_events model if event-specific model not found
            print(f"No specific model for {event_category}. Using all_events model.")
            return self.models.get("all_events"), self.vectorizers.get("all_events")

    def predict_price_direction(self, news_text, event_category):
        model, vectorizer = self._get_model_and_vectorizer(event_category)
        if not model or not vectorizer:
            print(f"No model or vectorizer found for {event_category} or all_events.")
            return None

        # Preprocess text (assuming preprocess_text is available, e.g., from rss_parser)
        # For now, a dummy preprocess_text
        def dummy_preprocess_text(text):
            return text.lower() # Simple lowercasing for demonstration

        processed_text = dummy_preprocess_text(news_text)
        text_vec = vectorizer.transform([processed_text])
        prediction = model.predict(text_vec)[0]
        return prediction

    def execute_trade(self, ticker, trade_date, trade_type, quantity, price):
        # trade_type: 'BUY' or 'SELL'
        cost = quantity * price
        if trade_type == 'BUY':
            if self.capital >= cost:
                self.capital -= cost
                self.positions[ticker] = self.positions.get(ticker, 0) + quantity
                self.trade_log.append({
                    'date': trade_date,
                    'ticker': ticker,
                    'type': trade_type,
                    'quantity': quantity,
                    'price': price,
                    'cost': cost
                })
                print(f"{trade_date}: BUY {quantity} of {ticker} at {price}. Capital: {self.capital:.2f}")
            else:
                print(f"{trade_date}: Insufficient capital to BUY {quantity} of {ticker}.")
        elif trade_type == 'SELL':
            if self.positions.get(ticker, 0) >= quantity:
                self.capital += cost
                self.positions[ticker] -= quantity
                self.trade_log.append({
                    'date': trade_date,
                    'ticker': ticker,
                    'type': trade_type,
                    'quantity': quantity,
                    'price': price,
                    'cost': cost
                })
                print(f"{trade_date}: SELL {quantity} of {ticker} at {price}. Capital: {self.capital:.2f}")
            else:
                print(f"{trade_date}: Insufficient shares to SELL {quantity} of {ticker}.")

    def calculate_pnl(self, current_prices):
        # current_prices: dict of {ticker: current_price}
        current_portfolio_value = self.capital
        for ticker, quantity in self.positions.items():
            if quantity > 0 and ticker in current_prices:
                current_portfolio_value += quantity * current_prices[ticker]
        pnl = current_portfolio_value - self.initial_capital
        return pnl, current_portfolio_value

    def run_backtest(self, news_data, price_data):
        # news_data: DataFrame with columns: 'published', 'ticker', 'text', 'event_category'
        # price_data: DataFrame with columns: 'date', 'ticker', 'open', 'close'

        # Ensure 'published' column in news_data is datetime objects
        news_data['published'] = pd.to_datetime(news_data['published'])
        # Ensure 'date' column in price_data is datetime objects
        price_data['date'] = pd.to_datetime(price_data['date'])

        # Sort news data by published date
        news_data = news_data.sort_values(by='published').reset_index(drop=True)

        for index, news_event in news_data.iterrows():
            news_date = news_event['published'].date()
            ticker = news_event['ticker']
            news_text = news_event['text']
            event_category = news_event['event_category']

            # Predict price direction
            predicted_direction = self.predict_price_direction(news_text, event_category)

            if predicted_direction:
                # Find the relevant price data for the next trading day
                # This is a simplified approach. A more robust system would handle market hours, holidays, etc.
                next_day_prices = price_data[(price_data['ticker'] == ticker) & (price_data['date'].dt.date > news_date)]
                if not next_day_prices.empty:
                    trade_date = next_day_prices.iloc[0]['date'].date()
                    open_price = next_day_prices.iloc[0]['open']
                    close_price = next_day_prices.iloc[0]['close']

                    if predicted_direction == 'UP':
                        # Buy at open, sell at close (simplified strategy)
                        quantity_to_buy = int(self.capital * 0.1 / open_price) # Allocate 10% of capital
                        if quantity_to_buy > 0:
                            self.execute_trade(ticker, trade_date, 'BUY', quantity_to_buy, open_price)
                            self.execute_trade(ticker, trade_date, 'SELL', quantity_to_buy, close_price)
                    elif predicted_direction == 'DOWN':
                        # Short sell at open, buy back at close (simplified strategy)
                        # For simplicity, we'll treat shorting as a regular sell for PnL calculation
                        # In a real scenario, this would involve more complex shorting logic
                        quantity_to_sell = int(self.capital * 0.1 / open_price) # Allocate 10% of capital
                        if quantity_to_sell > 0:
                            self.execute_trade(ticker, trade_date, 'SELL', quantity_to_sell, open_price)
                            self.execute_trade(ticker, trade_date, 'BUY', quantity_to_sell, close_price)

        # Final PnL calculation
        final_prices = {}
        for ticker in self.positions.keys():
            latest_price_data = price_data[price_data['ticker'] == ticker].sort_values(by='date', ascending=False)
            if not latest_price_data.empty:
                final_prices[ticker] = latest_price_data.iloc[0]['close']

        final_pnl, final_portfolio_value = self.calculate_pnl(final_prices)
        print(f"\n--- Backtest Results ---")
        print(f"Initial Capital: {self.initial_capital:.2f}")
        print(f"Final Portfolio Value: {final_portfolio_value:.2f}")
        print(f"Total PnL: {final_pnl:.2f}")
        print(f"Trade Log: {self.trade_log}")
        return final_pnl, self.trade_log

if __name__ == '__main__':
    # Create dummy data for demonstration
    # In a real scenario, this data would come from the database
    dummy_news_data = pd.DataFrame({
        'published': [
            datetime(2025, 1, 1, 9, 0, 0),
            datetime(2025, 1, 2, 17, 0, 0),
            datetime(2025, 1, 3, 10, 0, 0),
            datetime(2025, 1, 4, 8, 0, 0)
        ],
        'ticker': ['AAPL', 'MSFT', 'AAPL', 'MSFT'],
        'text': [
            "Apple announces strong Q4 earnings, beating estimates.",
            "Microsoft faces antitrust investigation in Europe.",
            "Apple unveils new iPhone with advanced AI features.",
            "Microsoft partners with OpenAI for new cloud services."
        ],
        'event_category': [
            'earnings_releases_and_operating_results',
            'regulatory_filings',
            'product_services_announcement',
            'partnerships'
        ]
    })

    dummy_price_data = pd.DataFrame({
        'date': [
            datetime(2025, 1, 1), datetime(2025, 1, 1),
            datetime(2025, 1, 2), datetime(2025, 1, 2),
            datetime(2025, 1, 3), datetime(2025, 1, 3),
            datetime(2025, 1, 6), datetime(2025, 1, 6) # Skip weekend
        ],
        'ticker': ['AAPL', 'MSFT', 'AAPL', 'MSFT', 'AAPL', 'MSFT', 'AAPL', 'MSFT'],
        'open': [170.0, 400.0, 172.0, 395.0, 175.0, 390.0, 178.0, 392.0],
        'close': [171.0, 398.0, 174.0, 392.0, 177.0, 388.0, 180.0, 395.0]
    })

    # Create dummy models and vectorizers for testing
    # This is needed because the backtester tries to load them
    if not os.path.exists("/home/ubuntu/finespresso-modelling/models/"):
        os.makedirs("/home/ubuntu/finespresso-modelling/models/")

    # Dummy data for fitting models
    X_dummy = ["positive news", "negative news"]
    y_dummy = ["UP", "DOWN"]

    # Dummy model and vectorizer for 'earnings_releases_and_operating_results'
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_extraction.text import TfidfVectorizer
    dummy_vectorizer_er = TfidfVectorizer()
    X_dummy_vec_er = dummy_vectorizer_er.fit_transform(X_dummy)
    dummy_model_er = RandomForestClassifier()
    dummy_model_er.fit(X_dummy_vec_er, y_dummy)
    joblib.dump(dummy_model_er, "/home/ubuntu/finespresso-modelling/models/earnings_releases_and_operating_results_model.joblib")
    joblib.dump(dummy_vectorizer_er, "/home/ubuntu/finespresso-modelling/models/earnings_releases_and_operating_results_vectorizer.joblib")

    # Dummy model and vectorizer for 'regulatory_filings'
    dummy_vectorizer_rf = TfidfVectorizer()
    X_dummy_vec_rf = dummy_vectorizer_rf.fit_transform(X_dummy)
    dummy_model_rf = RandomForestClassifier()
    dummy_model_rf.fit(X_dummy_vec_rf, y_dummy)
    joblib.dump(dummy_model_rf, "/home/ubuntu/finespresso-modelling/models/regulatory_filings_model.joblib")
    joblib.dump(dummy_vectorizer_rf, "/home/ubuntu/finespresso-modelling/models/regulatory_filings_vectorizer.joblib")

    # Dummy model and vectorizer for 'product_services_announcement'
    dummy_vectorizer_psa = TfidfVectorizer()
    X_dummy_vec_psa = dummy_vectorizer_psa.fit_transform(X_dummy)
    dummy_model_psa = RandomForestClassifier()
    dummy_model_psa.fit(X_dummy_vec_psa, y_dummy)
    joblib.dump(dummy_model_psa, "/home/ubuntu/finespresso-modelling/models/product_services_announcement_model.joblib")
    joblib.dump(dummy_vectorizer_psa, "/home/ubuntu/finespresso-modelling/models/product_services_announcement_vectorizer.joblib")

    # Dummy model and vectorizer for 'partnerships'
    dummy_vectorizer_p = TfidfVectorizer()
    X_dummy_vec_p = dummy_vectorizer_p.fit_transform(X_dummy)
    dummy_model_p = RandomForestClassifier()
    dummy_model_p.fit(X_dummy_vec_p, y_dummy)
    joblib.dump(dummy_model_p, "/home/ubuntu/finespresso-modelling/models/partnerships_model.joblib")
    joblib.dump(dummy_vectorizer_p, "/home/ubuntu/finespresso-modelling/models/partnerships_vectorizer.joblib")

    # Dummy model and vectorizer for 'all_events'
    dummy_vectorizer_all = TfidfVectorizer()
    X_dummy_vec_all = dummy_vectorizer_all.fit_transform(X_dummy)
    dummy_model_all = RandomForestClassifier()
    dummy_model_all.fit(X_dummy_vec_all, y_dummy)
    joblib.dump(dummy_model_all, "/home/ubuntu/finespresso-modelling/models/all_events_model.joblib")
    joblib.dump(dummy_vectorizer_all, "/home/ubuntu/finespresso-modelling/models/all_events_vectorizer.joblib")

    backtester = Backtester(initial_capital=100000)
    final_pnl, trade_log = backtester.run_backtest(dummy_news_data, dummy_price_data)

    print(f"\nFinal PnL from main: {final_pnl:.2f}")
    # print("Trade Log from main:")
    # for trade in trade_log:
    #     print(trade)


