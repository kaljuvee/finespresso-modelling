# 📈 Finespresso Analytics - Streamlit Dashboard

A comprehensive financial news analysis and backtesting platform with an interactive web interface built using Streamlit.

## 🌟 Features

### 📊 Interactive Dashboard
- **System Overview**: Real-time statistics and system components
- **Data Collection**: Automated RSS feed parsing and price data gathering
- **Model Training**: Machine learning model training with performance metrics
- **Backtesting**: Strategy simulation with PnL calculation
- **System Monitoring**: Health checks and status monitoring

### 🎨 Modern Interface
- Responsive design with sidebar navigation
- Interactive charts and visualizations using Plotly
- Real-time progress indicators
- Professional styling with custom CSS

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/kaljuvee/finespresso-modelling.git
   cd finespresso-modelling
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Access the dashboard**
   Open your browser and navigate to `http://localhost:8501`

## 🌐 Deployment

### Streamlit Cloud Deployment

1. **Fork this repository** to your GitHub account

2. **Connect to Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with your GitHub account
   - Click "New app"
   - Select your repository and branch
   - Set main file path to `app.py`
   - Click "Deploy"

3. **Environment Setup**
   - Streamlit Cloud will automatically install dependencies from `requirements.txt`
   - The app will be available at a public URL

### Local Development

```bash
# Start the development server
streamlit run app.py --server.port 8501

# Access the application
open http://localhost:8501
```

## 📁 Project Structure

```
finespresso-modelling/
├── app.py                    # Main Streamlit application
├── main.py                   # Alternative entry point
├── requirements.txt          # Python dependencies
├── .streamlit/
│   └── config.toml          # Streamlit configuration
├── models/                   # Trained ML models
│   ├── *_model.joblib       # Classifier models
│   ├── *_vectorizer.joblib  # Text vectorizers
│   └── model_training_results.csv
├── data/                     # Data storage
├── rss_feeds.txt            # RSS feed URLs
├── database.py              # Database schema
├── data_manager.py          # Database operations
├── rss_parser.py            # RSS feed parsing
├── price_collector.py       # Price data collection
├── train_classifier.py      # ML model training
├── backtest_util.py         # Backtesting engine
└── README.md                # This file
```

## 🔧 Configuration

### RSS Feeds
Edit `rss_feeds.txt` to modify data sources:
```
Energy RSS: https://www.globenewswire.com/rss/sector/energy
Biotechnology RSS: https://www.globenewswire.com/rss/sector/biotechnology
```

### Streamlit Settings
Modify `.streamlit/config.toml` for custom configuration:
```toml
[server]
port = 8501
address = "0.0.0.0"

[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
```

## 📊 Usage Guide

### 1. Data Collection
- Navigate to the "Data Collection" section
- Click "Start Data Collection" to gather news and price data
- Monitor progress through the real-time progress bar
- View collected data summary and statistics

### 2. Model Training
- Go to the "Model Training" section
- Review training configuration and parameters
- Click "Train Models" to build ML classification models
- Analyze model performance metrics and accuracy charts

### 3. Backtesting
- Access the "Backtesting" section
- Configure initial capital and strategy parameters
- Click "Run Backtest" to simulate trading strategies
- Review PnL results and trade logs

### 4. System Monitoring
- Check the "System Status" section for health monitoring
- Verify database connections and API access
- Monitor system performance and file status

## 🤖 Machine Learning Models

The system uses binary classification models for price direction prediction:

- **Algorithm**: Random Forest Classifier
- **Features**: TF-IDF vectorized news text
- **Target**: Binary price direction (UP/DOWN)
- **Validation**: 5-fold cross-validation
- **Performance**: Up to 100% accuracy for specific event categories

## 📈 Data Sources

- **News Data**: Globenewswire RSS feeds (Energy & Biotechnology sectors)
- **Price Data**: yfinance API for historical and real-time stock prices
- **Storage**: SQLite database for efficient data management

## 🔍 System Requirements

- **Python**: 3.8+
- **Memory**: 2GB+ recommended
- **Storage**: 1GB+ for data and models
- **Network**: Internet connection for data collection

## 🛠️ Development

### Adding New Features
1. Create feature branch: `git checkout -b feature/new-feature`
2. Implement changes in the appropriate modules
3. Test locally with `streamlit run app.py`
4. Commit and push: `git push origin feature/new-feature`
5. Create pull request

### Debugging
- Check Streamlit logs in the terminal
- Use `st.write()` for debugging output
- Monitor browser console for JavaScript errors

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📞 Support

For questions or issues:
- Open an issue on GitHub
- Check the Streamlit documentation
- Review the system logs

## 🔗 Links

- [Streamlit Documentation](https://docs.streamlit.io)
- [Plotly Documentation](https://plotly.com/python/)
- [yfinance Documentation](https://pypi.org/project/yfinance/)
- [scikit-learn Documentation](https://scikit-learn.org/)

---

**Built with ❤️ using Streamlit and Python**

