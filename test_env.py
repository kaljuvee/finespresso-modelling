#!/usr/bin/env python3
"""
Test script to verify environment variable loading
"""
import os
from dotenv import load_dotenv

print("=== ENVIRONMENT VARIABLE TEST ===")

# Test 1: Load .env file
print("1. Loading .env file...")
load_dotenv()

# Test 2: Check if DATABASE_URL is available
database_url = os.getenv('DATABASE_URL')
print(f"2. DATABASE_URL found: {database_url is not None}")
if database_url:
    print(f"   DATABASE_URL (first 50 chars): {database_url[:50]}...")

# Test 3: Test database connection
print("3. Testing database connection...")
try:
    from postgres_data_manager import PostgresDataManager
    dm = PostgresDataManager(database_url=database_url)
    print("   ✅ Database connection successful!")
    
    # Test basic queries
    news_count = dm.get_news_count()
    print(f"   📰 News articles: {news_count:,}")
    
except Exception as e:
    print(f"   ❌ Database connection failed: {e}")
    print(f"   Error type: {type(e).__name__}")

print("=== TEST COMPLETE ===")

