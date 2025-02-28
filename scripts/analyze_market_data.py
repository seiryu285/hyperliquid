#!/usr/bin/env python
"""
Hyperliquid Market Data Analysis Script

This script analyzes market data collected for a specific symbol (ETH-PERP by default)
from MongoDB. It provides various analyses and visualizations to help understand
market behavior and validate data collection.

Usage:
    python analyze_market_data.py --symbol ETH-PERP --interval 1h --days 7
"""

import os
import sys
import json
import logging
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import motor.motor_asyncio
import asyncio
from dotenv import load_dotenv
from core.config import settings
from market_data.data_processor import DataProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def analyze_market_data(symbol: str, interval: str, days: int, output_dir: str):
    """
    Analyze market data for a specific symbol.
    
    Args:
        symbol: Trading symbol (e.g., 'ETH-PERP')
        interval: Candle interval to analyze (e.g., '1h')
        days: Number of days of data to analyze
        output_dir: Directory to save analysis outputs
    """
    logger.info(f"Starting market data analysis for {symbol} ({interval} candles, {days} days)")
    
    # Load environment variables
    load_dotenv(project_root / '.env')
    
    # Connect to MongoDB
    client = motor.motor_asyncio.AsyncIOMotorClient(settings.MONGODB_URL)
    db = client[settings.MONGO_DB]
    candles_collection = db.candles
    
    # Calculate time range
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=days)
    
    logger.info(f"Fetching data from {start_time} to {end_time}")
    
    # Query candles from MongoDB
    cursor = candles_collection.find({
        'symbol': symbol,
        'interval': interval,
        'timestamp': {'$gte': start_time, '$lte': end_time}
    }).sort('timestamp', 1)
    
    # Convert to list and then DataFrame
    candles = await cursor.to_list(length=None)
    
    if not candles:
        logger.warning(f"No candle data found for {symbol} with interval {interval}")
        return
    
    logger.info(f"Found {len(candles)} candles")
    
    # Create DataFrame
    df = pd.DataFrame(candles)
    
    # Basic data validation
    logger.info("Performing basic data validation...")
    
    # Check for missing timestamps
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)
    
    # Resample to the correct interval to identify gaps
    if interval == '1m':
        expected_df = df.resample('1T').asfreq()
    elif interval == '5m':
        expected_df = df.resample('5T').asfreq()
    elif interval == '15m':
        expected_df = df.resample('15T').asfreq()
    elif interval == '1h':
        expected_df = df.resample('1H').asfreq()
    elif interval == '4h':
        expected_df = df.resample('4H').asfreq()
    elif interval == '1d':
        expected_df = df.resample('1D').asfreq()
    
    missing_count = expected_df.shape[0] - df.shape[0]
    missing_pct = (missing_count / expected_df.shape[0]) * 100 if expected_df.shape[0] > 0 else 0
    
    logger.info(f"Data completeness: {df.shape[0]} out of {expected_df.shape[0]} expected candles")
    logger.info(f"Missing candles: {missing_count} ({missing_pct:.2f}%)")
    
    # Create DataProcessor instance
    processor = DataProcessor()
    
    # Add technical indicators
    df_with_indicators = processor.add_technical_indicators(df)
    
    # Create analysis directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save processed data to CSV
    csv_path = os.path.join(output_dir, f"{symbol}_{interval}_{days}days.csv")
    df_with_indicators.to_csv(csv_path)
    logger.info(f"Saved processed data to {csv_path}")
    
    # Generate price chart
    plt.figure(figsize=(12, 6))
    plt.plot(df_with_indicators.index, df_with_indicators['close'], label='Close Price')
    plt.plot(df_with_indicators.index, df_with_indicators['sma_7'], label='SMA 7')
    plt.plot(df_with_indicators.index, df_with_indicators['sma_25'], label='SMA 25')
    plt.title(f"{symbol} Price Chart ({interval} candles, {days} days)")
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    
    # Save price chart
    price_chart_path = os.path.join(output_dir, f"{symbol}_{interval}_{days}days_price.png")
    plt.savefig(price_chart_path)
    logger.info(f"Saved price chart to {price_chart_path}")
    
    # Generate volume chart
    plt.figure(figsize=(12, 4))
    plt.bar(df_with_indicators.index, df_with_indicators['volume'], alpha=0.7)
    plt.title(f"{symbol} Volume Chart ({interval} candles, {days} days)")
    plt.xlabel('Date')
    plt.ylabel('Volume')
    plt.grid(True)
    
    # Save volume chart
    volume_chart_path = os.path.join(output_dir, f"{symbol}_{interval}_{days}days_volume.png")
    plt.savefig(volume_chart_path)
    logger.info(f"Saved volume chart to {volume_chart_path}")
    
    # Generate RSI chart
    plt.figure(figsize=(12, 4))
    plt.plot(df_with_indicators.index, df_with_indicators['rsi_14'], label='RSI 14')
    plt.axhline(y=70, color='r', linestyle='--', alpha=0.5)
    plt.axhline(y=30, color='g', linestyle='--', alpha=0.5)
    plt.title(f"{symbol} RSI Chart ({interval} candles, {days} days)")
    plt.xlabel('Date')
    plt.ylabel('RSI')
    plt.legend()
    plt.grid(True)
    
    # Save RSI chart
    rsi_chart_path = os.path.join(output_dir, f"{symbol}_{interval}_{days}days_rsi.png")
    plt.savefig(rsi_chart_path)
    logger.info(f"Saved RSI chart to {rsi_chart_path}")
    
    # Generate MACD chart
    plt.figure(figsize=(12, 4))
    plt.plot(df_with_indicators.index, df_with_indicators['macd_line'], label='MACD Line')
    plt.plot(df_with_indicators.index, df_with_indicators['macd_signal'], label='Signal Line')
    plt.bar(df_with_indicators.index, df_with_indicators['macd_histogram'], alpha=0.5, label='Histogram')
    plt.title(f"{symbol} MACD Chart ({interval} candles, {days} days)")
    plt.xlabel('Date')
    plt.ylabel('MACD')
    plt.legend()
    plt.grid(True)
    
    # Save MACD chart
    macd_chart_path = os.path.join(output_dir, f"{symbol}_{interval}_{days}days_macd.png")
    plt.savefig(macd_chart_path)
    logger.info(f"Saved MACD chart to {macd_chart_path}")
    
    # Generate Bollinger Bands chart
    plt.figure(figsize=(12, 6))
    plt.plot(df_with_indicators.index, df_with_indicators['close'], label='Close Price')
    plt.plot(df_with_indicators.index, df_with_indicators['bb_upper'], label='Upper Band')
    plt.plot(df_with_indicators.index, df_with_indicators['bb_middle'], label='Middle Band')
    plt.plot(df_with_indicators.index, df_with_indicators['bb_lower'], label='Lower Band')
    plt.title(f"{symbol} Bollinger Bands ({interval} candles, {days} days)")
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    
    # Save Bollinger Bands chart
    bb_chart_path = os.path.join(output_dir, f"{symbol}_{interval}_{days}days_bb.png")
    plt.savefig(bb_chart_path)
    logger.info(f"Saved Bollinger Bands chart to {bb_chart_path}")
    
    # Generate basic statistics
    stats = {
        'symbol': symbol,
        'interval': interval,
        'days': days,
        'start_date': df.index.min().strftime('%Y-%m-%d %H:%M:%S'),
        'end_date': df.index.max().strftime('%Y-%m-%d %H:%M:%S'),
        'candle_count': len(df),
        'missing_candles': missing_count,
        'missing_pct': missing_pct,
        'price_stats': {
            'min': df['low'].min(),
            'max': df['high'].max(),
            'mean': df['close'].mean(),
            'std': df['close'].std(),
            'current': df['close'].iloc[-1],
            'change_pct': ((df['close'].iloc[-1] / df['close'].iloc[0]) - 1) * 100
        },
        'volume_stats': {
            'min': df['volume'].min(),
            'max': df['volume'].max(),
            'mean': df['volume'].mean(),
            'std': df['volume'].std(),
            'total': df['volume'].sum()
        }
    }
    
    # Save statistics to JSON
    stats_path = os.path.join(output_dir, f"{symbol}_{interval}_{days}days_stats.json")
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Saved statistics to {stats_path}")
    
    # Print summary
    logger.info("\n=== Analysis Summary ===")
    logger.info(f"Symbol: {symbol}")
    logger.info(f"Interval: {interval}")
    logger.info(f"Period: {stats['start_date']} to {stats['end_date']} ({days} days)")
    logger.info(f"Candles: {stats['candle_count']} (Missing: {stats['missing_candles']} or {stats['missing_pct']:.2f}%)")
    logger.info(f"Price range: {stats['price_stats']['min']:.2f} to {stats['price_stats']['max']:.2f}")
    logger.info(f"Current price: {stats['price_stats']['current']:.2f}")
    logger.info(f"Price change: {stats['price_stats']['change_pct']:.2f}%")
    logger.info(f"Average volume: {stats['volume_stats']['mean']:.2f}")
    logger.info(f"Total volume: {stats['volume_stats']['total']:.2f}")
    logger.info("=====================")
    
    logger.info(f"Market data analysis completed for {symbol}")
    return df_with_indicators

async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Analyze market data from Hyperliquid')
    parser.add_argument('--symbol', type=str, default=settings.PRIMARY_SYMBOL, 
                        help=f'Trading symbol (default: {settings.PRIMARY_SYMBOL})')
    parser.add_argument('--interval', type=str, default='1h', 
                        help='Candle interval to analyze (default: 1h)')
    parser.add_argument('--days', type=int, default=7, 
                        help='Number of days of data to analyze (default: 7)')
    parser.add_argument('--output', type=str, default='analysis_output', 
                        help='Directory to save analysis outputs (default: analysis_output)')
    
    args = parser.parse_args()
    
    # Analyze market data
    await analyze_market_data(args.symbol, args.interval, args.days, args.output)

if __name__ == '__main__':
    asyncio.run(main())
