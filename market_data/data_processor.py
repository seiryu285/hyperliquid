"""
Data preprocessing module for HyperLiquid trading agent.

This module is responsible for processing market data, calculating technical indicators,
and preparing data for the prediction model.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta

from core.config import settings
from market_data.data_collector import HyperLiquidDataCollector

# Configure logging
logger = logging.getLogger(__name__)

class DataProcessor:
    """
    Data processor for market data.
    
    This class is responsible for:
    1. Fetching data from MongoDB
    2. Calculating technical indicators
    3. Preparing data for the prediction model
    4. Feature engineering and normalization
    """
    
    def __init__(self, db_client=None):
        """
        Initialize data processor.
        
        Args:
            db_client: MongoDB client to use for data retrieval
        """
        self.db_client = db_client
        logger.info("Data processor initialized")
    
    async def get_ohlcv_data(self, 
                           symbol: str, 
                           timeframe: str = '1h', 
                           limit: int = 1000) -> pd.DataFrame:
        """
        Get OHLCV data for a specific symbol and timeframe.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe for the data (e.g., '1m', '5m', '1h', '1d')
            limit: Maximum number of candles to retrieve
            
        Returns:
            DataFrame with OHLCV data
        """
        # This is a placeholder - in a real implementation, you would retrieve data from MongoDB
        # For now, we'll simulate this with random data
        logger.info(f"Getting OHLCV data for {symbol} on {timeframe} timeframe")
        
        # Create datetime index
        end_time = datetime.utcnow()
        
        # Define timeframe in minutes
        timeframe_minutes = {
            '1m': 1,
            '5m': 5,
            '15m': 15,
            '30m': 30,
            '1h': 60,
            '4h': 240,
            '1d': 1440
        }
        
        minutes = timeframe_minutes.get(timeframe, 60)
        
        # Create timestamps
        timestamps = [end_time - timedelta(minutes=i * minutes) for i in range(limit)]
        timestamps.reverse()
        
        # Create simulated data
        base_price = 50000  # Example base price for BTC
        volatility = 0.002  # 0.2% price movement per candle on average
        
        data = []
        current_price = base_price
        
        for timestamp in timestamps:
            # Create random price movement
            price_change = np.random.normal(0, volatility)
            current_price *= (1 + price_change)
            
            # Generate OHLCV values
            high = current_price * (1 + abs(np.random.normal(0, volatility/2)))
            low = current_price * (1 - abs(np.random.normal(0, volatility/2)))
            open_price = current_price * (1 + np.random.normal(0, volatility/4))
            close = current_price
            volume = np.random.lognormal(10, 1)
            
            data.append({
                'timestamp': timestamp,
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume
            })
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        
        return df
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators to the dataframe.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added technical indicators
        """
        # Make a copy to avoid modifying the original dataframe
        df_with_indicators = df.copy()
        
        # Simple Moving Averages
        df_with_indicators['sma_7'] = df_with_indicators['close'].rolling(window=7).mean()
        df_with_indicators['sma_25'] = df_with_indicators['close'].rolling(window=25).mean()
        df_with_indicators['sma_99'] = df_with_indicators['close'].rolling(window=99).mean()
        
        # Exponential Moving Averages
        df_with_indicators['ema_9'] = df_with_indicators['close'].ewm(span=9, adjust=False).mean()
        df_with_indicators['ema_21'] = df_with_indicators['close'].ewm(span=21, adjust=False).mean()
        
        # Bollinger Bands (20, 2)
        window = 20
        std_dev = 2
        
        df_with_indicators['bb_middle'] = df_with_indicators['close'].rolling(window=window).mean()
        df_with_indicators['bb_std'] = df_with_indicators['close'].rolling(window=window).std()
        df_with_indicators['bb_upper'] = df_with_indicators['bb_middle'] + (df_with_indicators['bb_std'] * std_dev)
        df_with_indicators['bb_lower'] = df_with_indicators['bb_middle'] - (df_with_indicators['bb_std'] * std_dev)
        
        # RSI (14)
        window = 14
        delta = df_with_indicators['close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        
        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()
        
        # Skip division by zero using np.finfo(float).eps as a small epsilon value
        rs = avg_gain / (avg_loss + np.finfo(float).eps)
        df_with_indicators['rsi_14'] = 100 - (100 / (1 + rs))
        
        # MACD
        short_window = 12
        long_window = 26
        signal_window = 9
        
        df_with_indicators['macd_line'] = df_with_indicators['close'].ewm(span=short_window, adjust=False).mean() - \
                                       df_with_indicators['close'].ewm(span=long_window, adjust=False).mean()
        df_with_indicators['macd_signal'] = df_with_indicators['macd_line'].ewm(span=signal_window, adjust=False).mean()
        df_with_indicators['macd_histogram'] = df_with_indicators['macd_line'] - df_with_indicators['macd_signal']
        
        # Average True Range (ATR)
        high_low = df_with_indicators['high'] - df_with_indicators['low']
        high_close = np.abs(df_with_indicators['high'] - df_with_indicators['close'].shift())
        low_close = np.abs(df_with_indicators['low'] - df_with_indicators['close'].shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df_with_indicators['atr_14'] = true_range.rolling(window=14).mean()
        
        # Price change
        df_with_indicators['price_change_1'] = df_with_indicators['close'].pct_change(1)
        df_with_indicators['price_change_5'] = df_with_indicators['close'].pct_change(5)
        
        # Volume change
        df_with_indicators['volume_change_1'] = df_with_indicators['volume'].pct_change(1)
        
        # Fill NaN values
        df_with_indicators.fillna(method='bfill', inplace=True)
        
        logger.info(f"Added technical indicators to dataframe with shape {df_with_indicators.shape}")
        return df_with_indicators
    
    def normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize features for model input.
        
        Args:
            df: DataFrame with features
            
        Returns:
            DataFrame with normalized features
        """
        # Create a copy to avoid modifying the original
        df_normalized = df.copy()
        
        # List price-based columns to normalize relative to the current price
        price_columns = ['open', 'high', 'low', 'close', 'sma_7', 'sma_25', 'sma_99', 
                         'ema_9', 'ema_21', 'bb_middle', 'bb_upper', 'bb_lower']
        
        # Normalize price columns relative to the current close price
        for col in price_columns:
            if col in df_normalized.columns:
                df_normalized[f'{col}_rel'] = df_normalized[col] / df_normalized['close']
                
        # Min-max scaling for other indicators
        indicator_columns = ['rsi_14', 'macd_line', 'macd_signal', 'macd_histogram', 'atr_14']
        
        for col in indicator_columns:
            if col in df_normalized.columns:
                min_val = df_normalized[col].min()
                max_val = df_normalized[col].max()
                if max_val > min_val:  # Avoid division by zero
                    df_normalized[f'{col}_norm'] = (df_normalized[col] - min_val) / (max_val - min_val)
        
        logger.info(f"Normalized features in dataframe with shape {df_normalized.shape}")
        return df_normalized
    
    def create_training_data(self, 
                            df: pd.DataFrame, 
                            target_column: str = 'close', 
                            window_size: int = 60,
                            forecast_horizon: int = 10,
                            feature_columns: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create training data with sliding window for sequence models.
        
        Args:
            df: DataFrame with normalized features
            target_column: Column to use as the prediction target
            window_size: Size of the sliding window (lookback period)
            forecast_horizon: How many steps ahead to predict
            feature_columns: List of columns to use as features (if None, use all)
            
        Returns:
            Tuple of (X, y) where X is the feature array and y is the target array
        """
        if feature_columns is None:
            # Use all columns except the target for features
            feature_columns = [col for col in df.columns if col != target_column]
        
        # Extract the features and target
        data = df[feature_columns + [target_column]].values
        
        X, y = [], []
        
        for i in range(len(data) - window_size - forecast_horizon + 1):
            # Extract the window of features
            feature_window = data[i:i+window_size, :-1]  # All columns except target
            
            # Extract the target value at the forecast horizon
            target_value = data[i+window_size+forecast_horizon-1, -1]  
            
            X.append(feature_window)
            y.append(target_value)
        
        logger.info(f"Created training data with {len(X)} samples, window_size={window_size}, forecast_horizon={forecast_horizon}")
        return np.array(X), np.array(y)
    
    def prepare_latest_data(self, 
                           df: pd.DataFrame, 
                           window_size: int = 60,
                           feature_columns: Optional[List[str]] = None) -> np.ndarray:
        """
        Prepare the latest data window for model prediction.
        
        Args:
            df: DataFrame with normalized features
            window_size: Size of the sliding window (lookback period)
            feature_columns: List of columns to use as features
            
        Returns:
            Feature array for the latest data window
        """
        if feature_columns is None:
            # Use all columns
            feature_columns = df.columns.tolist()
        
        # Get the latest window of data
        latest_data = df[feature_columns].values[-window_size:]
        
        # Reshape for model input (adding batch dimension)
        return latest_data.reshape(1, window_size, len(feature_columns))

if __name__ == '__main__':
    # Example usage
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create processor
    processor = DataProcessor()
    
    # Run in an async wrapper
    import asyncio
    
    async def test_processor():
        # Get OHLCV data
        df = await processor.get_ohlcv_data(symbol='BTC', timeframe='1h', limit=200)
        print(f"Original dataframe shape: {df.shape}")
        
        # Add technical indicators
        df_with_indicators = processor.add_technical_indicators(df)
        print(f"Dataframe with indicators shape: {df_with_indicators.shape}")
        
        # Normalize features
        df_normalized = processor.normalize_features(df_with_indicators)
        print(f"Normalized dataframe shape: {df_normalized.shape}")
        
        # Create training data
        X, y = processor.create_training_data(
            df=df_normalized,
            window_size=60,
            forecast_horizon=10,
        )
        print(f"Training data shapes: X={X.shape}, y={y.shape}")
        
        # Prepare latest data for prediction
        latest_features = processor.prepare_latest_data(df_normalized, window_size=60)
        print(f"Latest features shape: {latest_features.shape}")
        
    # Run the async function
    asyncio.run(test_processor())
