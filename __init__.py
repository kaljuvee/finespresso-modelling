"""
Finespresso Modelling System

A comprehensive data collection, modeling, and backtesting system for financial news analysis.
"""

__version__ = "1.0.0"
__author__ = "Finespresso Team"
__email__ = "team@finespresso.com"

from .data_manager import DataManager
from .rss_parser import parse_rss_feed, extract_company_and_ticker
from .price_collector import get_price_data
from .train_classifier import train_model, preprocess_text
from .backtest_util import Backtester

__all__ = [
    "DataManager",
    "parse_rss_feed",
    "extract_company_and_ticker", 
    "get_price_data",
    "train_model",
    "preprocess_text",
    "Backtester"
]

