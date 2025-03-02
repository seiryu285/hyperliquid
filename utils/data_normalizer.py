import logging
from typing import Dict, Any, List, Optional
import numpy as np
from datetime import datetime
import yaml

logger = logging.getLogger(__name__)

class DataNormalizer:
    def __init__(self, config_path: str = "config/market_data.yaml"):
        self.config = self._load_config(config_path)
        self.price_history: Dict[str, List[float]] = {}
        self.volume_history: Dict[str, List[float]] = {}
        self.window_size = 100  # Rolling window size for normalization

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

    async def normalize_data(self, message: Dict) -> Dict:
        """Normalize market data based on message type."""
        try:
            if not self.config['processing']['normalization_enabled']:
                return message

            message_type = message.get('type')
            if message_type == 'trades':
                return await self._normalize_trade(message)
            elif message_type == 'orderbook':
                return await self._normalize_orderbook(message)
            elif message_type == 'ticker':
                return await self._normalize_ticker(message)
            else:
                logger.warning(f"Unknown message type for normalization: {message_type}")
                return message

        except Exception as e:
            logger.error(f"Error normalizing data: {e}")
            return message

    async def _normalize_trade(self, message: Dict) -> Dict:
        """Normalize trade data."""
        try:
            symbol = message.get('symbol')
            if not symbol:
                return message

            # Update price history
            if symbol not in self.price_history:
                self.price_history[symbol] = []
            self.price_history[symbol].append(float(message['price']))
            if len(self.price_history[symbol]) > self.window_size:
                self.price_history[symbol].pop(0)

            # Update volume history
            if symbol not in self.volume_history:
                self.volume_history[symbol] = []
            self.volume_history[symbol].append(float(message['size']))
            if len(self.volume_history[symbol]) > self.window_size:
                self.volume_history[symbol].pop(0)

            # Normalize price and size
            normalized_message = message.copy()
            normalized_message['normalized_price'] = self._normalize_value(
                message['price'],
                self.price_history[symbol]
            )
            normalized_message['normalized_size'] = self._normalize_value(
                message['size'],
                self.volume_history[symbol]
            )

            return normalized_message

        except Exception as e:
            logger.error(f"Error normalizing trade data: {e}")
            return message

    async def _normalize_orderbook(self, message: Dict) -> Dict:
        """Normalize orderbook data."""
        try:
            normalized_message = message.copy()
            
            # Normalize bids
            normalized_bids = []
            for price, size in message['bids']:
                normalized_price = self._normalize_value(price, self.price_history.get(message['symbol'], []))
                normalized_size = self._normalize_value(size, self.volume_history.get(message['symbol'], []))
                normalized_bids.append([normalized_price, normalized_size])
            normalized_message['normalized_bids'] = normalized_bids

            # Normalize asks
            normalized_asks = []
            for price, size in message['asks']:
                normalized_price = self._normalize_value(price, self.price_history.get(message['symbol'], []))
                normalized_size = self._normalize_value(size, self.volume_history.get(message['symbol'], []))
                normalized_asks.append([normalized_price, normalized_size])
            normalized_message['normalized_asks'] = normalized_asks

            return normalized_message

        except Exception as e:
            logger.error(f"Error normalizing orderbook data: {e}")
            return message

    async def _normalize_ticker(self, message: Dict) -> Dict:
        """Normalize ticker data."""
        try:
            symbol = message.get('symbol')
            if not symbol:
                return message

            normalized_message = message.copy()
            price_history = self.price_history.get(symbol, [])
            volume_history = self.volume_history.get(symbol, [])

            # Normalize price fields
            price_fields = ['price', 'open', 'high', 'low', 'close']
            for field in price_fields:
                normalized_message[f'normalized_{field}'] = self._normalize_value(
                    message[field],
                    price_history
                )

            # Normalize volume
            normalized_message['normalized_volume'] = self._normalize_value(
                message['volume'],
                volume_history
            )

            return normalized_message

        except Exception as e:
            logger.error(f"Error normalizing ticker data: {e}")
            return message

    def _normalize_value(self, value: float, history: List[float]) -> float:
        """Normalize a single value using z-score normalization."""
        try:
            if not history:
                return value

            mean = np.mean(history)
            std = np.std(history)

            if std == 0:
                return 0.0

            return (value - mean) / std

        except Exception as e:
            logger.error(f"Error normalizing value: {e}")
            return value

    def _min_max_normalize(self, value: float, min_val: float, max_val: float) -> float:
        """Normalize a value to the range [0, 1] using min-max normalization."""
        try:
            if max_val == min_val:
                return 0.5
            return (value - min_val) / (max_val - min_val)

        except Exception as e:
            logger.error(f"Error performing min-max normalization: {e}")
            return value

    def _log_normalize(self, value: float) -> float:
        """Apply log normalization to a value."""
        try:
            if value <= 0:
                return 0.0
            return np.log1p(value)

        except Exception as e:
            logger.error(f"Error performing log normalization: {e}")
            return value

    def reset_history(self, symbol: Optional[str] = None) -> None:
        """Reset price and volume history for a symbol or all symbols."""
        try:
            if symbol:
                self.price_history[symbol] = []
                self.volume_history[symbol] = []
            else:
                self.price_history = {}
                self.volume_history = {}
            logger.info(f"Reset history for {'all symbols' if symbol is None else symbol}")

        except Exception as e:
            logger.error(f"Error resetting history: {e}")

    def get_statistics(self, symbol: str) -> Dict[str, Any]:
        """Get normalization statistics for a symbol."""
        try:
            price_history = self.price_history.get(symbol, [])
            volume_history = self.volume_history.get(symbol, [])

            return {
                'price_mean': np.mean(price_history) if price_history else None,
                'price_std': np.std(price_history) if price_history else None,
                'volume_mean': np.mean(volume_history) if volume_history else None,
                'volume_std': np.std(volume_history) if volume_history else None,
                'price_history_length': len(price_history),
                'volume_history_length': len(volume_history)
            }

        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {}
