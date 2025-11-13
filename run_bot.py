#!/usr/bin/env python3
"""
Main entry point for running the Telegram bot
Run this file from project root: python run_bot.py
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import and run the bot
from src.bot_simple import UniversityDocumentBot

if __name__ == "__main__":
    bot = UniversityDocumentBot()
    bot.run()
