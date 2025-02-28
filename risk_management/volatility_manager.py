import logging
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from collections import deque

logger = logging.getLogger(__name__)

class VolatilityManager:
    def __init__(self, config: dict):
        self.config = config
        self.price_history: Dict[str, deque] = {}
        self.volatility_history: Dict[str, deque] = {}
        self.last_update = datetime.now()
        self.lookback_period = self.config['volatility']['lookback_period'] * 60  # Convert hours to minutes

    async def update_price_history(self, symbol: str, price: float, timestamp: datetime) -> None:
        """Update price history for volatility calculation."""
        try:
            if symbol not in self.price_history:
                self.price_history[symbol] = deque(maxlen=self.lookback_period)
                self.volatility_history[symbol] = deque(maxlen=self.lookback_period)

            self.price_history[symbol].append((timestamp, price))

            # Calculate and store volatility if we have enough data
            if len(self.price_history[symbol]) > 1:
                volatility = self._calculate_rolling_volatility(symbol)
                self.volatility_history[symbol].append((timestamp, volatility))
                
        except Exception as e:
            logger.error(f"Error updating price history for {symbol}: {e}")
            raise

    async def calculate_volatility(self, market_data: Dict) -> Dict[str, float]:
        """Calculate current volatility for all symbols."""
        try:
            current_time = datetime.now()
            volatilities = {}

            for symbol, data in market_data.items():
                if 'price' in data:
                    await self.update_price_history(symbol, data['price'], current_time)
                    if symbol in self.volatility_history and self.volatility_history[symbol]:
                        volatilities[symbol] = self.volatility_history[symbol][-1][1]
                    else:
                        volatilities[symbol] = 0.0

            return volatilities
            
        except Exception as e:
            logger.error(f"Error calculating volatilities: {e}")
            raise

    def _calculate_rolling_volatility(self, symbol: str) -> float:
        """Calculate rolling volatility using price history."""
        try:
            prices = [p[1] for p in self.price_history[symbol]]
            if len(prices) < 2:
                return 0.0

            # Calculate log returns
            log_returns = np.log(np.array(prices[1:]) / np.array(prices[:-1]))
            
            # Calculate annualized volatility
            volatility = np.std(log_returns) * np.sqrt(525600 / self.lookback_period)  # Annualize (525600 minutes in a year)
            return volatility
            
        except Exception as e:
            logger.error(f"Error calculating rolling volatility: {e}")
            return 0.0

    async def get_volatility_forecast(self, symbol: str) -> Optional[float]:
        """Calculate volatility forecast using EWMA."""
        try:
            if symbol not in self.volatility_history or len(self.volatility_history[symbol]) < 2:
                return None

            # Get historical volatilities
            volatilities = [v[1] for v in self.volatility_history[symbol]]
            
            # Calculate EWMA volatility forecast
            lambda_param = 0.94  # Standard EWMA parameter
            weights = np.array([(1 - lambda_param) * lambda_param**i for i in range(len(volatilities))])
            weights = weights / weights.sum()  # Normalize weights
            
            forecast = np.sum(weights * np.array(volatilities))
            return forecast
            
        except Exception as e:
            logger.error(f"Error calculating volatility forecast: {e}")
            return None

    async def calculate_volatility_adjusted_position_size(
        self,
        symbol: str,
        base_position_size: float,
        current_volatility: Optional[float] = None
    ) -> float:
        """Calculate volatility-adjusted position size."""
        try:
            if current_volatility is None:
                if symbol not in self.volatility_history or not self.volatility_history[symbol]:
                    return base_position_size
                current_volatility = self.volatility_history[symbol][-1][1]

            target_volatility = self.config['volatility']['max_volatility']
            
            if current_volatility == 0:
                return base_position_size

            # Calculate volatility scaling factor
            scaling_factor = target_volatility / current_volatility
            
            # Apply limits to scaling factor
            scaling_factor = min(
                2.0,  # Maximum double the position
                max(0.25, scaling_factor)  # Minimum quarter the position
            )
            
            return base_position_size * scaling_factor
            
        except Exception as e:
            logger.error(f"Error calculating volatility adjusted position size: {e}")
            return base_position_size

    async def get_volatility_metrics(self, symbol: str) -> Dict:
        """Get comprehensive volatility metrics for a symbol."""
        try:
            if symbol not in self.volatility_history or not self.volatility_history[symbol]:
                return {
                    'current_volatility': 0.0,
                    'forecast_volatility': None,
                    'volatility_trend': 'stable',
                    'is_high_volatility': False
                }

            current_volatility = self.volatility_history[symbol][-1][1]
            forecast = await self.get_volatility_forecast(symbol)
            
            # Calculate volatility trend
            if len(self.volatility_history[symbol]) > 1:
                prev_volatility = self.volatility_history[symbol][-2][1]
                if current_volatility > prev_volatility * 1.1:
                    trend = 'increasing'
                elif current_volatility < prev_volatility * 0.9:
                    trend = 'decreasing'
                else:
                    trend = 'stable'
            else:
                trend = 'stable'

            return {
                'current_volatility': current_volatility,
                'forecast_volatility': forecast,
                'volatility_trend': trend,
                'is_high_volatility': current_volatility > self.config['volatility']['max_volatility']
            }
            
        except Exception as e:
            logger.error(f"Error getting volatility metrics: {e}")
            return {
                'current_volatility': 0.0,
                'forecast_volatility': None,
                'volatility_trend': 'error',
                'is_high_volatility': False
            }

    def get_volatility_history(self, symbol: str) -> List:
        """Get volatility history for a symbol."""
        return list(self.volatility_history.get(symbol, []))

    def clear_history(self, symbol: Optional[str] = None) -> None:
        """Clear price and volatility history for a symbol or all symbols."""
        if symbol:
            if symbol in self.price_history:
                del self.price_history[symbol]
            if symbol in self.volatility_history:
                del self.volatility_history[symbol]
        else:
            self.price_history.clear()
            self.volatility_history.clear()
