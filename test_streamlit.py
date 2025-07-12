import streamlit as st
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

st.title("Database Connection Test")

# Test environment variables
database_url = os.getenv('DATABASE_URL')
st.write(f"DATABASE_URL found: {database_url is not None}")

if database_url:
    st.write(f"DATABASE_URL (first 50 chars): {database_url[:50]}...")
    
    # Test database connection
    try:
        from postgres_data_manager import PostgresDataManager
        dm = PostgresDataManager(database_url=database_url)
        st.success("✅ Database connection successful!")
        
        # Test basic queries
        news_count = dm.get_news_count()
        price_moves_count = dm.get_price_moves_count()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("📰 News Articles", f"{news_count:,}")
        with col2:
            st.metric("📈 Price Moves", f"{price_moves_count:,}")
            
        # Test recent data
        st.subheader("Recent Data Sample")
        recent_data = dm.get_news_with_price_moves(limit=5)
        if not recent_data.empty:
            st.dataframe(recent_data[['ticker', 'title', 'published_date']])
        else:
            st.warning("No recent data available")
            
    except Exception as e:
        st.error(f"❌ Database connection failed: {e}")
        st.write(f"Error type: {type(e).__name__}")
else:
    st.error("❌ DATABASE_URL not found in environment variables")

