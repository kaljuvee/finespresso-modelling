
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()

class News(Base):
    __tablename__ = 'news'

    id = Column(Integer, primary_key=True)
    title = Column(String, nullable=False)
    link = Column(String, unique=True, nullable=False)
    published = Column(DateTime)
    summary = Column(Text)
    company_name = Column(String)
    ticker = Column(String)
    industry = Column(String)

    def __repr__(self):
        return f"<News(title='{self.title}', ticker='{self.ticker}')>"

class Price(Base):
    __tablename__ = 'price'

    id = Column(Integer, primary_key=True)
    ticker = Column(String, nullable=False)
    date = Column(DateTime, nullable=False)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Integer)

    def __repr__(self):
        return f"<Price(ticker='{self.ticker}', date='{self.date}')>"

def init_db(database_url='sqlite:///finespresso.db'):
    engine = create_engine(database_url)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    return Session()

if __name__ == '__main__':
    # Example usage: Initialize the database and add some dummy data
    session = init_db()
    print("Database initialized and tables created.")

    # Add a dummy news entry
    from datetime import datetime
    dummy_news = News(
        title="Test News Title",
        link="http://example.com/test-news",
        published=datetime.now(),
        summary="This is a summary of the test news.",
        company_name="Test Corp",
        ticker="TEST",
        industry="Technology"
    )
    session.add(dummy_news)
    session.commit()
    print(f"Added news: {dummy_news}")

    # Add a dummy price entry
    dummy_price = Price(
        ticker="TEST",
        date=datetime.now(),
        open=100.0,
        high=105.0,
        low=99.0,
        close=103.5,
        volume=100000
    )
    session.add(dummy_price)
    session.commit()
    print(f"Added price: {dummy_price}")

    # Query and print data
    print("\nQuerying news:")
    for news in session.query(News).all():
        print(news)

    print("\nQuerying prices:")
    for price in session.query(Price).all():
        print(price)

    session.close()


