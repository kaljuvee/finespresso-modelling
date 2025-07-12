# Environment Configuration Guide

**Date**: July 12, 2025  
**Version**: Finespresso Analytics v2.1  
**Update**: Environment Variables Configuration (.env)

---

## 🔧 **Environment Variables Setup**

The Finespresso Analytics application now uses `.env` files for configuration instead of Streamlit secrets. This provides better flexibility for local development and deployment.

### **1. ✅ Local Development Setup**

1. **Copy the example file**:
   ```bash
   cp .env.example .env
   ```

2. **Edit the .env file** with your actual credentials:
   ```bash
   # Environment variables for Finespresso Analytics
   
   # Database Configuration
   DATABASE_URL=postgresql://your_user:your_password@your_host/your_database
   
   # API Keys
   OPENAI_API_KEY=sk-proj-your_openai_api_key_here
   POLYGON_API_KEY=your_polygon_api_key_here
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**:
   ```bash
   streamlit run app.py
   ```

### **2. ✅ Production Deployment**

#### **For Streamlit Cloud:**

1. **Deploy from GitHub** as usual
2. **Add environment variables** in the Streamlit Cloud dashboard:
   - Go to your app settings
   - Navigate to "Secrets" section
   - Add each variable:
     ```
     DATABASE_URL = "postgresql://your_user:your_password@your_host/your_database"
     OPENAI_API_KEY = "sk-proj-your_openai_api_key_here"
     POLYGON_API_KEY = "your_polygon_api_key_here"
     ```

#### **For Other Platforms:**

Set environment variables according to your platform's documentation:

- **Heroku**: Use `heroku config:set`
- **Railway**: Use environment variables in dashboard
- **Vercel**: Use environment variables in project settings
- **Docker**: Use `-e` flags or docker-compose environment section

---

## 📋 **Required Environment Variables**

| Variable | Description | Example |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection string | `postgresql://user:pass@host/db` |
| `OPENAI_API_KEY` | OpenAI API key for ML features | `sk-proj-...` |
| `POLYGON_API_KEY` | Polygon.io API key for market data | `your_key_here` |

---

## 🔒 **Security Best Practices**

### **✅ DO:**
- Use `.env` files for local development
- Add `.env` to `.gitignore` (already done)
- Use platform-specific environment variable systems for production
- Rotate API keys regularly
- Use different credentials for development and production

### **❌ DON'T:**
- Commit `.env` files to version control
- Share API keys in plain text
- Use production credentials in development
- Hard-code credentials in source code

---

## 🚀 **Application Features**

The application automatically loads environment variables using `python-dotenv`:

```python
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Access variables
database_url = os.getenv('DATABASE_URL')
openai_key = os.getenv('OPENAI_API_KEY')
```

### **Database Connection**
- Automatic PostgreSQL connection using `DATABASE_URL`
- Connection pooling and error handling
- Real-time database statistics and health checks

### **API Integration**
- OpenAI API for ML model training and predictions
- Polygon.io API for market data (if needed)
- Graceful fallback when APIs are unavailable

---

## 🛠️ **Troubleshooting**

### **Common Issues:**

1. **"DATABASE_URL not found"**
   - Check that `.env` file exists
   - Verify variable name spelling
   - Ensure no extra spaces around `=`

2. **Database connection failed**
   - Verify PostgreSQL credentials
   - Check network connectivity
   - Confirm database server is running

3. **API key errors**
   - Verify API key format and validity
   - Check API key permissions
   - Ensure sufficient API credits

### **Debug Steps:**

1. **Check environment loading**:
   ```python
   import os
   from dotenv import load_dotenv
   
   load_dotenv()
   print("DATABASE_URL:", os.getenv('DATABASE_URL'))
   ```

2. **Test database connection**:
   ```bash
   python -c "from postgres_data_manager import PostgresDataManager; dm = PostgresDataManager()"
   ```

3. **Verify file permissions**:
   ```bash
   ls -la .env
   ```

---

## 📁 **File Structure**

```
finespresso-streamlit/
├── .env                    # Your actual credentials (not in git)
├── .env.example           # Template file (in git)
├── .gitignore            # Includes .env
├── app.py                # Main Streamlit application
├── requirements.txt      # Includes python-dotenv
└── ...
```

---

## 🎯 **Migration from Streamlit Secrets**

If you were previously using Streamlit secrets, the migration is automatic:

### **Before (secrets.toml)**:
```toml
[database]
DATABASE_URL = "postgresql://..."

[api_keys]
OPENAI_API_KEY = "sk-proj-..."
```

### **After (.env)**:
```bash
DATABASE_URL=postgresql://...
OPENAI_API_KEY=sk-proj-...
```

The application now uses `os.getenv()` instead of `st.secrets[]`, providing better compatibility across different deployment platforms.

---

**System Status**: ✅ **PRODUCTION READY**  
**Configuration**: Environment Variables (.env)  
**Security**: API keys protected from version control  
**Deployment**: Compatible with all major platforms

