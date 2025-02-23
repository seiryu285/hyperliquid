#!/usr/bin/env python3
"""
Data loader module for fetching and preprocessing market data from HyperLiquid API
"""

import os
import time
import json
import logging
import yaml
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any

import requests
import pandas as pd
import numpy as np
from retrying import retry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class HyperLiquidAPI:
    """HyperLiquid API wrapper"""
    
    BASE_URL = "https://api.hyperliquid.xyz"
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize HyperLiquid API client
        
        Args:
            api_key: Optional API key for authenticated endpoints
        """
        self.api_key = api_key
        self.session = requests.Session()
        if api_key:
            self.session.headers.update({
                'Authorization': f'Bearer {api_key}'
            })
    
    @retry(stop_max_attempt_number=3, wait_exponential_multiplier=1000)
    def _request(self, method: str, endpoint: str, **kwargs) -> Any:
        """Make API request with retry logic"""
        url = f"{self.BASE_URL}{endpoint}"
        try:
            response = self.session.request(method, url, **kwargs)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {str(e)}")
            raise
    
    def get_meta(self) -> Dict:
        """Get exchange metadata"""
        return self._request('GET', '/info/meta')
    
    def get_market_status(self, coin: str) -> Dict:
        """Get current market status"""
        return self._request('GET', '/info/status', params={'coin': coin})
    
    def get_orderbook(self, coin: str, depth: int = 10) -> Dict:
        """Get orderbook data"""
        return self._request('GET', '/info/l2book', 
                           params={'coin': coin, 'depth': depth})
    
    def get_trades(self, coin: str, limit: int = 100) -> List[Dict]:
        """Get recent trades"""
        return self._request('GET', '/info/trades', 
                           params={'coin': coin, 'limit': limit})
    
    def get_funding_rate(self, coin: str) -> Dict:
        """Get current funding rate"""
        return self._request('GET', '/info/funding', params={'coin': coin})
    
    def get_candles(self, coin: str, interval: str, 
                    start_time: Optional[int] = None,
                    end_time: Optional[int] = None) -> List[Dict]:
        """Get historical candle data"""
        params = {
            'coin': coin,
            'interval': interval
        }
        if start_time:
            params['startTime'] = start_time
        if end_time:
            params['endTime'] = end_time
            
        return self._request('GET', '/info/candles', params=params)

class DataLoader:
    def __init__(self, config_path: str = 'config/data_sources.yaml'):
        """
        Initialize data loader with configuration
        
        Args:
            config_path: Path to data source configuration file
        """
        self.config = self._load_config(config_path)
        self.api = HyperLiquidAPI(self.config.get('hyperliquid', {}).get('api_key'))
        
        # Initialize data storage directories
        self._init_storage()
        
        # Cache for market data
        self.market_cache = {}
        
    def _load_config(self, path: str) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load config from {path}: {str(e)}")
            return {
                'hyperliquid': {
                    'api_key': None
                }
            }
    
    def _init_storage(self) -> None:
        """Create required data directories"""
        try:
            for dir_path in [
                'data/raw/market',
                'data/raw/trades',
                'data/processed/historical',
                'data/processed/realtime'
            ]:
                os.makedirs(dir_path, exist_ok=True)
                logger.info(f"Created directory: {dir_path}")
        except Exception as e:
            logger.error(f"Failed to create directories: {str(e)}")
    
    def _save_raw_data(self, data: Union[Dict, List], 
                       category: str, timestamp: str) -> None:
        """Save raw data to file"""
        try:
            path = os.path.join(
                'data/raw',
                category,
                f"{timestamp}.json"
            )
            with open(path, 'w') as f:
                json.dump(data, f)
            logger.info(f"Saved raw data to {path}")
        except Exception as e:
            logger.error(f"Failed to save raw data: {str(e)}")
    
    def _process_market_data(self, data: Dict) -> pd.DataFrame:
        """Process raw market data into DataFrame"""
        try:
            df = pd.DataFrame({
                'timestamp': [pd.Timestamp.now()],
                'price': [float(data.get('markPrice', 0))],
                'index_price': [float(data.get('indexPrice', 0))],
                'funding_rate': [float(data.get('fundingRate', 0))],
                'open_interest': [float(data.get('openInterest', 0))],
                'volume_24h': [float(data.get('volume24h', 0))]
            })
            return df
        except Exception as e:
            logger.error(f"Failed to process market data: {str(e)}")
            return pd.DataFrame()
    
    def _process_orderbook(self, data: Dict) -> pd.DataFrame:
        """Process orderbook data into DataFrame"""
        try:
            bids = pd.DataFrame(data.get('bids', []), columns=['price', 'size'])
            asks = pd.DataFrame(data.get('asks', []), columns=['price', 'size'])
            
            if len(bids) > 0 and len(asks) > 0:
                df = pd.DataFrame({
                    'timestamp': [pd.Timestamp.now()],
                    'bid_price': [float(bids['price'].iloc[0])],
                    'bid_size': [float(bids['size'].iloc[0])],
                    'ask_price': [float(asks['price'].iloc[0])],
                    'ask_size': [float(asks['size'].iloc[0])],
                    'spread': [float(asks['price'].iloc[0]) - float(bids['price'].iloc[0])],
                    'mid_price': [(float(asks['price'].iloc[0]) + float(bids['price'].iloc[0])) / 2]
                })
                return df
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Failed to process orderbook: {str(e)}")
            return pd.DataFrame()
    
    def _process_trades(self, trades: List[Dict]) -> pd.DataFrame:
        """Process trade data into DataFrame"""
        try:
            df = pd.DataFrame(trades)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['price'] = df['price'].astype(float)
            df['size'] = df['size'].astype(float)
            return df
        except Exception as e:
            logger.error(f"Failed to process trades: {str(e)}")
            return pd.DataFrame()
    
    def _process_candles(self, candles: List[Dict]) -> pd.DataFrame:
        """Process candle data into DataFrame"""
        try:
            df = pd.DataFrame(candles)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
            return df
        except Exception as e:
            logger.error(f"Failed to process candles: {str(e)}")
            return pd.DataFrame()
    
    def load_market_data(self, coin: str = 'BTC') -> pd.DataFrame:
        """
        Load current market data
        
        Args:
            coin: Trading pair symbol
        """
        try:
            # Get market status
            market_data = self.api.get_market_status(coin)
            self._save_raw_data(
                market_data, 
                'market', 
                datetime.now().strftime('%Y%m%d_%H%M%S')
            )
            
            # Process market data
            df = self._process_market_data(market_data)
            if df.empty:
                logger.warning("No market data available")
                return df
            
            # Get and process orderbook
            orderbook = self.api.get_orderbook(coin)
            ob_df = self._process_orderbook(orderbook)
            
            # Merge data if orderbook is available
            if not ob_df.empty:
                df = pd.concat([df, ob_df], axis=1)
            
            # Save processed data
            self._save_processed_data(df, 'realtime')
            
            return df
        except Exception as e:
            logger.error(f"Failed to load market data: {str(e)}")
            return pd.DataFrame()
    
    def load_historical_data(self, 
                           coin: str = 'BTC',
                           days: int = 30,
                           interval: str = '1h') -> pd.DataFrame:
        """
        Load historical market data
        
        Args:
            coin: Trading pair symbol
            days: Number of days of historical data
            interval: Candle interval
        """
        try:
            end_time = int(time.time() * 1000)
            start_time = end_time - (days * 24 * 60 * 60 * 1000)
            
            # Get historical candles
            candles = self.api.get_candles(
                coin=coin,
                interval=interval,
                start_time=start_time,
                end_time=end_time
            )
            
            # Process candles
            df = self._process_candles(candles)
            if df.empty:
                logger.warning("No historical data available")
                return df
            
            # Save processed data
            self._save_processed_data(df, 'historical')
            
            return df
        except Exception as e:
            logger.error(f"Failed to load historical data: {str(e)}")
            return pd.DataFrame()
    
    def _save_processed_data(self, df: pd.DataFrame, data_type: str) -> None:
        """Save processed data to parquet file"""
        try:
            if not df.empty:
                path = os.path.join(
                    'data/processed',
                    data_type,
                    f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
                )
                df.to_parquet(path)
                logger.info(f"Saved processed data to {path}")
        except Exception as e:
            logger.error(f"Failed to save processed data: {str(e)}")

if __name__ == '__main__':
    # Example usage
    loader = DataLoader()
    
    # Load current market data
    market_data = loader.load_market_data('BTC')
    print("\nCurrent Market Data:")
    print(market_data.head())
    
    # Load historical data
    historical_data = loader.load_historical_data('BTC', days=7)
    print("\nHistorical Data:")
    print(historical_data.head())
