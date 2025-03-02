import logging
from typing import Dict, Any, Optional
import numpy as np
from datetime import datetime
import yaml

logger = logging.getLogger(__name__)

class DataValidator:
    def __init__(self, config_path: str = "config/market_data.yaml"):
        self.config = self._load_config(config_path)
        self.required_fields = {
            'trades': ['timestamp', 'price', 'size', 'side'],
            'orderbook': ['timestamp', 'bids', 'asks'],
            'ticker': ['timestamp', 'price', 'volume', 'open', 'high', 'low', 'close']
        }

    def _load_config(self, config_path: str) -> dict:
        """Load market data configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Successfully loaded market data configuration from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load market data configuration: {e}")
            raise

    async def validate_message(self, message: Dict) -> bool:
        """Validate incoming market data message."""
        try:
            if not isinstance(message, dict):
                logger.warning(f"Invalid message format: {message}")
                return False

            # Check message type
            if 'type' not in message:
                logger.warning("Message missing 'type' field")
                return False

            message_type = message['type']
            if message_type not in self.required_fields:
                logger.warning(f"Unknown message type: {message_type}")
                return False

            # Check required fields
            for field in self.required_fields[message_type]:
                if field not in message:
                    logger.warning(f"Missing required field '{field}' in {message_type} message")
                    return False

            # Validate timestamp
            if not self._validate_timestamp(message['timestamp']):
                return False

            # Validate specific message types
            if message_type == 'trades':
                return await self._validate_trade(message)
            elif message_type == 'orderbook':
                return await self._validate_orderbook(message)
            elif message_type == 'ticker':
                return await self._validate_ticker(message)

            return True

        except Exception as e:
            logger.error(f"Error validating message: {e}")
            return False

    def _validate_timestamp(self, timestamp: Any) -> bool:
        """Validate message timestamp."""
        try:
            if isinstance(timestamp, (int, float)):
                # Convert to datetime for range checking
                dt = datetime.fromtimestamp(timestamp)
            elif isinstance(timestamp, str):
                # Parse ISO format string
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            else:
                logger.warning(f"Invalid timestamp format: {timestamp}")
                return False

            # Check if timestamp is within reasonable range
            now = datetime.now()
            if dt > now or dt < now.replace(year=now.year - 1):
                logger.warning(f"Timestamp out of reasonable range: {dt}")
                return False

            return True

        except Exception as e:
            logger.error(f"Error validating timestamp: {e}")
            return False

    async def _validate_trade(self, message: Dict) -> bool:
        """Validate trade message."""
        try:
            # Validate price
            if not isinstance(message['price'], (int, float)) or message['price'] <= 0:
                logger.warning(f"Invalid trade price: {message['price']}")
                return False

            # Validate size
            if not isinstance(message['size'], (int, float)) or message['size'] <= 0:
                logger.warning(f"Invalid trade size: {message['size']}")
                return False

            # Validate side
            if message['side'] not in ['buy', 'sell']:
                logger.warning(f"Invalid trade side: {message['side']}")
                return False

            return True

        except Exception as e:
            logger.error(f"Error validating trade message: {e}")
            return False

    async def _validate_orderbook(self, message: Dict) -> bool:
        """Validate orderbook message."""
        try:
            # Validate bids
            if not isinstance(message['bids'], list):
                logger.warning("Invalid bids format")
                return False

            # Validate asks
            if not isinstance(message['asks'], list):
                logger.warning("Invalid asks format")
                return False

            # Validate orderbook depth
            max_depth = self.config['collection']['orderbook_depth']
            if len(message['bids']) > max_depth or len(message['asks']) > max_depth:
                logger.warning(f"Orderbook depth exceeds maximum: {max_depth}")
                return False

            # Validate price levels
            for bid in message['bids']:
                if not self._validate_price_level(bid):
                    return False

            for ask in message['asks']:
                if not self._validate_price_level(ask):
                    return False

            # Validate price ordering
            if not self._validate_price_ordering(message['bids'], message['asks']):
                return False

            return True

        except Exception as e:
            logger.error(f"Error validating orderbook message: {e}")
            return False

    async def _validate_ticker(self, message: Dict) -> bool:
        """Validate ticker message."""
        try:
            # Validate price fields
            price_fields = ['price', 'open', 'high', 'low', 'close']
            for field in price_fields:
                if not isinstance(message[field], (int, float)) or message[field] <= 0:
                    logger.warning(f"Invalid {field}: {message[field]}")
                    return False

            # Validate volume
            if not isinstance(message['volume'], (int, float)) or message['volume'] < 0:
                logger.warning(f"Invalid volume: {message['volume']}")
                return False

            # Validate price relationships
            if not (message['low'] <= message['price'] <= message['high']):
                logger.warning("Invalid price relationships in ticker")
                return False

            return True

        except Exception as e:
            logger.error(f"Error validating ticker message: {e}")
            return False

    def _validate_price_level(self, level: Any) -> bool:
        """Validate a single price level in the orderbook."""
        try:
            if not isinstance(level, (list, tuple)) or len(level) != 2:
                logger.warning(f"Invalid price level format: {level}")
                return False

            price, size = level

            if not isinstance(price, (int, float)) or price <= 0:
                logger.warning(f"Invalid price in price level: {price}")
                return False

            if not isinstance(size, (int, float)) or size < 0:
                logger.warning(f"Invalid size in price level: {size}")
                return False

            return True

        except Exception as e:
            logger.error(f"Error validating price level: {e}")
            return False

    def _validate_price_ordering(self, bids: list, asks: list) -> bool:
        """Validate price ordering in the orderbook."""
        try:
            if not bids or not asks:
                return True

            highest_bid = max(bid[0] for bid in bids)
            lowest_ask = min(ask[0] for ask in asks)

            if highest_bid >= lowest_ask:
                logger.warning(f"Invalid price ordering: highest bid ({highest_bid}) >= lowest ask ({lowest_ask})")
                return False

            return True

        except Exception as e:
            logger.error(f"Error validating price ordering: {e}")
            return False

    def _check_outliers(self, value: float, mean: float, std: float) -> bool:
        """Check if a value is an outlier based on standard deviation."""
        if std == 0:
            return True
        z_score = abs(value - mean) / std
        return z_score <= self.config['processing']['outlier_std_threshold']
