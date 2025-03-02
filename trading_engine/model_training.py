"""
Model training script for LSTM and Transformer models.

This script provides functionality for training and evaluating LSTM and Transformer
models for price prediction.
"""

import os
import sys
import logging
import json
import asyncio
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trading_engine.model_training.model_trainer import ModelTrainer
from market_data.data_processor import DataProcessor
from core.config import get_settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/model_training.log')
    ]
)
logger = logging.getLogger(__name__)

# Get settings
settings = get_settings()


async def train_models(config_path: str, symbols: Optional[List[str]] = None, timeframes: Optional[List[str]] = None):
    """
    Train models based on configuration.
    
    Args:
        config_path: Path to configuration file
        symbols: List of symbols to train on (if None, use all symbols in config)
        timeframes: List of timeframes to train on (if None, use all timeframes in config)
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Get model configurations
    model_configs = config.get('models', [])
    if not model_configs:
        logger.error("No model configurations found in config file")
        return
    
    # Get symbols and timeframes
    all_symbols = config.get('symbols', [])
    all_timeframes = config.get('timeframes', ['1h'])
    
    # Use provided symbols and timeframes if available
    symbols = symbols or all_symbols
    timeframes = timeframes or all_timeframes
    
    # Create data processor
    data_processor = DataProcessor()
    
    # Create model trainer
    model_trainer = ModelTrainer(data_processor)
    
    # Train models
    for symbol in symbols:
        for timeframe in timeframes:
            for model_config in model_configs:
                logger.info(f"Training {model_config['type']} model for {symbol} {timeframe}")
                
                # Set training period
                end_date = datetime.now().strftime('%Y-%m-%d')
                days_back = model_config.get('training_days', 365)
                start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
                
                try:
                    # Train model
                    model, metrics = await model_trainer.train_model(
                        model_config=model_config,
                        symbol=symbol,
                        timeframe=timeframe,
                        start_date=start_date,
                        end_date=end_date
                    )
                    
                    # Save metrics
                    model_id = model_config.get('model_id', f"{model_config['type']}_{symbol}_{timeframe}")
                    metrics_path = f"models/metrics/{model_id}.json"
                    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
                    
                    with open(metrics_path, 'w') as f:
                        json.dump(metrics, f, indent=4)
                    
                    logger.info(f"Saved metrics to {metrics_path}")
                    
                except Exception as e:
                    logger.error(f"Error training model for {symbol} {timeframe}: {e}")


def main():
    """
    Main function for model training.
    """
    # Create parser
    parser = argparse.ArgumentParser(description='Train models for trading')
    parser.add_argument('--config', type=str, default='config.json', help='Path to configuration file')
    parser.add_argument('--symbols', type=str, nargs='+', help='Symbols to train on')
    parser.add_argument('--timeframes', type=str, nargs='+', help='Timeframes to train on')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Train models
    asyncio.run(train_models(args.config, args.symbols, args.timeframes))


if __name__ == '__main__':
    main()
