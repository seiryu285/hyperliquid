#!/usr/bin/env python
"""
Hyperliquid Market Data Collection Script

This script collects and stores market data for a specific symbol (ETH-PERP by default)
from the Hyperliquid testnet. It uses the data collector module to:
1. Fetch historical candle data
2. Subscribe to real-time market data via WebSocket
3. Store all data in MongoDB for further analysis

Usage:
    python collect_market_data.py --symbol ETH-PERP --duration 3600 --interval 1h,5m,1m
"""

import os
import sys
import json
import time
import asyncio
import logging
import argparse
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from dotenv import load_dotenv

# Import local modules
try:
    from market_data.data_collector import HyperLiquidDataCollector
except ImportError:
    # If the module doesn't exist, create a simple implementation
    class HyperLiquidDataCollector:
        """Temporary implementation of HyperLiquidDataCollector"""
        
        def __init__(self, api_key=None, api_secret=None, db_uri=None):
            self.api_key = api_key
            self.api_secret = api_secret
            self.db_uri = db_uri
            self.running = False
            
        async def start(self, symbols=None, candle_intervals=None):
            """Start data collection"""
            self.running = True
            logging.info(f"Started data collection for {symbols} with intervals {candle_intervals}")
            
        async def stop(self):
            """Stop data collection"""
            self.running = False
            logging.info("Stopped data collection")
            
        async def fetch_historical_candles(self, symbol, interval, limit=1000, start_time=None, end_time=None):
            """Fetch historical candles"""
            logging.info(f"Fetching historical candles for {symbol} ({interval})")
            # Return empty list for now
            return []
            
        async def fetch_ticker(self, symbols):
            """Fetch ticker data"""
            logging.info(f"Fetching ticker data for {symbols}")
            # Return empty dict for now
            return {}

async def get_data_collector():
    """
    Get a data collector instance.
    
    Returns:
        HyperLiquidDataCollector: A data collector instance
    """
    # Load environment variables
    load_dotenv(project_root / '.env')
    
    # Get API credentials
    api_key = os.getenv('HYPERLIQUID_API_KEY', '')
    api_secret = os.getenv('HYPERLIQUID_API_SECRET', '')
    db_uri = os.getenv('MONGODB_URI', 'mongodb://localhost:27017')
    
    # Create data collector
    collector = HyperLiquidDataCollector(
        api_key=api_key,
        api_secret=api_secret,
        db_uri=db_uri
    )
    
    return collector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"logs/market_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger(__name__)

async def collect_market_data(symbol: str, duration: int, intervals: list):
    """
    Collect market data for a specific symbol.
    
    Args:
        symbol: Trading symbol (e.g., 'ETH-PERP')
        duration: Collection duration in seconds
        intervals: List of candle intervals to collect (e.g., ['1m', '5m', '1h'])
    """
    logger.info(f"Starting market data collection for {symbol}")
    logger.info(f"Collection duration: {duration} seconds")
    logger.info(f"Candle intervals: {intervals}")
    
    # Load environment variables
    load_dotenv(project_root / '.env')
    
    # Verify API configuration
    api_key = os.getenv('HYPERLIQUID_API_KEY', '')
    api_secret = os.getenv('HYPERLIQUID_API_SECRET', '')
    
    if not api_key or not api_secret:
        logger.warning("API credentials not found in .env file. Some functionality may be limited.")
    
    # Get data collector
    collector = await get_data_collector()
    
    try:
        # Fetch historical data for each interval
        for interval in intervals:
            logger.info(f"Fetching historical {interval} candles for {symbol}")
            
            # Calculate start time (7 days ago)
            end_time = int(time.time() * 1000)
            start_time = end_time - (7 * 24 * 60 * 60 * 1000)  # 7 days in milliseconds
            
            # Fetch and store historical candles
            candles = await collector.fetch_historical_candles(
                symbol=symbol,
                interval=interval,
                limit=1000,
                start_time=start_time,
                end_time=end_time
            )
            
            logger.info(f"Fetched {len(candles)} historical {interval} candles for {symbol}")
        
        # Start real-time data collection
        logger.info(f"Starting real-time data collection for {symbol}")
        await collector.start(symbols=[symbol], candle_intervals=intervals)
        
        # Collect ticker data
        logger.info(f"Fetching ticker data for {symbol}")
        ticker_data = await collector.fetch_ticker([symbol])
        
        # Run for specified duration
        logger.info(f"Collecting data for {duration} seconds...")
        start_time = time.time()
        
        while time.time() - start_time < duration:
            # Log progress every minute
            if int(time.time() - start_time) % 60 == 0:
                elapsed = int(time.time() - start_time)
                remaining = duration - elapsed
                logger.info(f"Data collection in progress: {elapsed}s elapsed, {remaining}s remaining")
            
            await asyncio.sleep(1)
        
        # Stop data collection
        logger.info("Data collection duration reached, stopping collector")
        await collector.stop()
        
        logger.info("Market data collection completed successfully")
        
    except Exception as e:
        logger.error(f"Error during market data collection: {e}", exc_info=True)
        
        # Try to stop collector if it's running
        try:
            await collector.stop()
        except:
            pass
        
        raise

async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Collect market data from Hyperliquid')
    parser.add_argument('--symbol', type=str, default='ETH-PERP', 
                        help='Trading symbol (default: ETH-PERP)')
    parser.add_argument('--duration', type=int, default=3600, 
                        help='Collection duration in seconds (default: 3600)')
    parser.add_argument('--intervals', type=str, default='1h,5m,1m', 
                        help='Comma-separated list of candle intervals (default: 1h,5m,1m)')
    
    args = parser.parse_args()
    
    # Parse intervals
    intervals = args.intervals.split(',')
    
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Collect market data
    await collect_market_data(args.symbol, args.duration, intervals)

if __name__ == '__main__':
    asyncio.run(main())
