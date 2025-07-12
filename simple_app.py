import streamlit as st
import pandas as pd
import plotly.express as px
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import custom modules
from postgres_data_manager import PostgresDataManager

st.set_page_config(
    page_title="Finespresso Analytics",
    page_icon="📈",
    layout="wide"
)

@st.cache_resource
def get_data_manager():
    """Initialize and cache the PostgreSQL data manager"""
    try:
        database_url = os.getenv('DATABASE_URL')
        if not database_url:
            st.error("❌ DATABASE_URL not found in environment variables.")
            return None
        
        dm = PostgresDataManager(database_url=database_url)
        return dm
    except Exception as e:
        st.error(f"Failed to connect to database: {e}")
        return None

def show_database_status():
    st.header("📊 Database Status")
    
    dm = get_data_manager()
    if not dm:
        st.error("Database not connected. Please check your configuration.")
        return
    
    # Database statistics
    st.subheader("📈 Database Statistics")
    
    try:
        # Get basic stats
        news_count = dm.get_news_count()
        price_moves_count = dm.get_price_moves_count()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📰 News Articles", f"{news_count:,}")
        
        with col2:
            st.metric("📈 Price Moves", f"{price_moves_count:,}")
        
        with col3:
            coverage = (price_moves_count / news_count * 100) if news_count > 0 else 0
            st.metric("📊 Price Coverage", f"{coverage:.1f}%")
        
        with col4:
            st.metric("🔗 Database", "PostgreSQL", "Live")
        
        # Recent data sample
        st.subheader("📋 Recent Data Sample")
        
        recent_data = dm.get_news_with_price_moves(limit=10)
        if not recent_data.empty:
            # Display key columns
            display_cols = ['ticker', 'title', 'published_date', 'actual_side', 'price_change_percentage']
            available_cols = [col for col in display_cols if col in recent_data.columns]
            st.dataframe(recent_data[available_cols], use_container_width=True)
        else:
            st.warning("No recent data available")
        
        # Publisher distribution
        st.subheader("📊 Publisher Distribution")
        
        publisher_data = dm.get_publisher_stats()
        if not publisher_data.empty:
            fig = px.bar(
                publisher_data, 
                x='publisher', 
                y='count',
                title="News Articles by Publisher",
                color='count',
                color_continuous_scale='Blues'
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error fetching database statistics: {e}")

def main():
    st.title("📈 Finespresso Analytics Dashboard")
    st.subheader("Professional Financial News Analysis & Realistic Backtesting Platform")
    
    # Navigation
    page = st.sidebar.selectbox(
        "Navigate",
        ["📊 Database Status", "🔧 Test Connection"]
    )
    
    if page == "📊 Database Status":
        show_database_status()
    elif page == "🔧 Test Connection":
        st.header("🔧 Database Connection Test")
        dm = get_data_manager()
        if dm:
            st.success("✅ Database connection successful!")
            st.write("Connection details verified.")
        else:
            st.error("❌ Database connection failed!")

if __name__ == "__main__":
    main()

