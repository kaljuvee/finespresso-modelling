#!/usr/bin/env python3
"""
Main entry point for the Finespresso Streamlit application
"""

import subprocess
import sys
import os

def main():
    """Run the Streamlit application"""
    # Change to the directory containing the app
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Run streamlit with the correct configuration
    cmd = [
        sys.executable, "-m", "streamlit", "run", "app.py",
        "--server.port", "8501",
        "--server.address", "0.0.0.0",
        "--server.headless", "true",
        "--server.enableCORS", "false",
        "--server.enableXsrfProtection", "false"
    ]
    
    subprocess.run(cmd)

if __name__ == "__main__":
    main()

