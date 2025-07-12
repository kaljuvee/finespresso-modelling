
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from datetime import datetime
from database import Base, News, Price # Changed to absolute import

class DataManager:
    def __init__(self, database_url="sqlite:///finespresso.db"):
        self.engine = create_engine(database_url)
        Base.metadata.create_all(self.engine) # Create tables if they don't exist
        self.Session = sessionmaker(bind=self.engine)

    def add_news_article(self, title, link, published, summary, company_name=None, ticker=None, industry=None):
        session = self.Session()
        try:
            # Check if news article already exists to prevent duplicates
            existing_news = session.query(News).filter_by(link=link).first()
            if existing_news:
                print(f"News article with link {link} already exists. Skipping.")
                return existing_news

            news = News(
                title=title,
                link=link,
                published=published,
                summary=summary,
                company_name=company_name,
                ticker=ticker,
                industry=industry
            )
            session.add(news)
            session.commit()
            print(f"Added news: {news.title}")
            return news
        except Exception as e:
            session.rollback()
            print(f"Error adding news article: {e}")
            return None
        finally:
            session.close()

    def add_price_data(self, ticker, date, open_price, high_price, low_price, close_price, volume):
        session = self.Session()
        try:
            # Check if price data for this ticker and date already exists
            existing_price = session.query(Price).filter_by(ticker=ticker, date=date).first()
            if existing_price:
                print(f"Price data for {ticker} on {date} already exists. Skipping.")
                return existing_price

            price = Price(
                ticker=ticker,
                date=date,
                open=open_price,
                high=high_price,
                low=low_price,
                close=close_price,
                volume=volume
            )
            session.add(price)
            session.commit()
            print(f"Added price data for {ticker} on {date}")
            return price
        except Exception as e:
            session.rollback()
            print(f"Error adding price data: {e}")
            return None
        finally:
            session.close()

    def add_price_data_bulk(self, price_data_list):
        session = self.Session()
        try:
            new_prices = []
            for data in price_data_list:
                existing_price = session.query(Price).filter_by(ticker=data["ticker"], date=data["date"]).first()
                if not existing_price:
                    new_prices.append(Price(
                        ticker=data["ticker"],
                        date=data["date"],
                        open=data["open_price"],
                        high=data["high_price"],
                        low=data["low_price"],
                        close=data["close_price"],
                        volume=data["volume"]
                    ))
            if new_prices:
                session.bulk_save_objects(new_prices)
                session.commit()
                print(f"Added {len(new_prices)} new price data records.")
            else:
                print("No new price data records to add.")
        except Exception as e:
            session.rollback()
            print(f"Error adding price data in bulk: {e}")
        finally:
            session.close()

    def get_all_news(self):
        session = self.Session()
        try:
            return session.query(News).all()
        finally:
            session.close()

    def get_price_data_for_ticker(self, ticker, start_date=None, end_date=None):
        session = self.Session()
        try:
            query = session.query(Price).filter_by(ticker=ticker)
            if start_date:
                query = query.filter(Price.date >= start_date)
            if end_date:
                query = query.filter(Price.date <= end_date)
            return query.order_by(Price.date).all()
        finally:
            session.close()

    def get_news_count(self):
        """Get total count of news articles"""
        session = self.Session()
        try:
            return session.query(News).count()
        finally:
            session.close()

    def get_price_data_count(self):
        """Get total count of price data records"""
        session = self.Session()
        try:
            return session.query(Price).count()
        finally:
            session.close()

    def get_unique_tickers(self):
        """Get list of unique tickers in the database"""
        session = self.Session()
        try:
            return [ticker[0] for ticker in session.query(Price.ticker).distinct().all()]
        finally:
            session.close()

if __name__ == '__main__':
    dm = DataManager()

    # Add dummy news
    from datetime import datetime
    dm.add_news_article(
        title="Sample News 1",
        link="http://example.com/news1",
        published=datetime(2025, 7, 10, 10, 0, 0),
        summary="Summary of sample news 1.",
        company_name="Sample Corp",
        ticker="SMPL",
        industry="Tech"
    )

    # Add dummy price data
    dm.add_price_data(
        ticker="SMPL",
        date=datetime(2025, 7, 10),
        open_price=100.0,
        high_price=102.0,
        low_price=99.0,
        close_price=101.5,
        volume=100000
    )

    # Test bulk add
    bulk_data = [
        {
            "ticker": "SMPL",
            "date": datetime(2025, 7, 11),
            "open_price": 101.0,
            "high_price": 103.0,
            "low_price": 100.0,
            "close_price": 102.5,
            "volume": 120000
        },
        {
            "ticker": "SMPL",
            "date": datetime(2025, 7, 12),
            "open_price": 102.0,
            "high_price": 104.0,
            "low_price": 101.0,
            "close_price": 103.5,
            "volume": 150000
        }
    ]
    dm.add_price_data_bulk(bulk_data)

    # Retrieve and print data
    print("\nAll News:")
    for news in dm.get_all_news():
        print(news)

    print("\nPrice data for SMPL:")
    for price in dm.get_price_data_for_ticker("SMPL"):
        print(price)

    # Test duplicate entry
    dm.add_news_article(
        title="Sample News 1",
        link="http://example.com/news1",
        published=datetime(2025, 7, 10, 10, 0, 0),
        summary="Summary of sample news 1.",
        company_name="Sample Corp",
        ticker="SMPL",
        industry="Tech"
    )




