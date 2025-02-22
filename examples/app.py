#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   app.py
@Time    :   2024/02/20
@Author  :   Taylor Firman
@Version :   1.0
@Contact :   tefirman@gmail.com
@Desc    :   Main entry point for March Madness bracket app
'''

from shiny import App
import logging
from datetime import datetime
from pathlib import Path

# Import our modular components
from ui import app_ui
from server import server
from data import initialize_tournament_data

# Set up logging configuration
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(Path('logs') / f'app_{datetime.now().strftime("%Y%m%d")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    """Main entry point for the application"""
    try:
        # Initialize data
        logger.info("Initializing tournament data...")
        initialize_tournament_data()
        
        # Create and configure app
        logger.info("Creating Shiny app...")
        app = App(app_ui, server)
        
        # Run the app
        logger.info("Starting application...")
        app.run()
        
    except Exception as e:
        logger.error(f"Error running application: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
