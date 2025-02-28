import logging
from typing import Dict, List, Optional
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)

class PositionManager:
    def __init__(self, config: dict):
        self.config = config
        self.positions: Dict[str, Dict] = {}
        self.position_history: List[Dict] = []

    async def update_position(self, symbol: str, position_data: Dict) -> None:
        """Update position information for a symbol."""
        try:
            self.positions[symbol] = position_data
            self.position_history.append({
                'timestamp': datetime.now(),
                'symbol': symbol,
                'position': position_data
            })
            
            # Trim history to keep only recent data
            if len(self.position_history) > 1000:
                self.position_history = self.position_history[-1000:]
                
            logger.info(f"Updated position for {symbol}: {position_data}")
            
        except Exception as e:
            logger.error(f"Error updating position for {symbol}: {e}")
            raise

    async def calculate_exposure(self, positions: Dict) -> float:
        """Calculate total position exposure."""
        try:
            total_exposure = sum(
                abs(pos['position_value']) 
                for pos in positions.values()
            )
            return total_exposure
            
        except Exception as e:
            logger.error(f"Error calculating exposure: {e}")
            raise

    async def check_position_limits(self, symbol: str, size: float) -> bool:
        """Check if position size is within limits."""
        try:
            # Check absolute position size limit
            if abs(size) > self.config['position_limits']['max_position_size']:
                logger.warning(f"Position size {size} exceeds maximum limit")
                return False

            # Check position concentration
            total_position_value = sum(
                abs(pos['position_value']) 
                for pos in self.positions.values()
            )
            
            if total_position_value > 0:
                concentration = abs(size) / total_position_value
                if concentration > self.config['position_limits']['position_concentration_limit']:
                    logger.warning(f"Position concentration {concentration} exceeds limit")
                    return False

            return True
            
        except Exception as e:
            logger.error(f"Error checking position limits: {e}")
            raise

    async def calculate_position_adjustments(
        self,
        symbol: str,
        current_price: float,
        volatility: float
    ) -> Optional[Dict]:
        """Calculate required position adjustments based on market conditions."""
        try:
            if symbol not in self.positions:
                return None

            position = self.positions[symbol]
            adjustments = {}

            # Adjust for volatility
            if self.config['volatility']['volatility_scaling']:
                vol_ratio = volatility / self.config['volatility']['max_volatility']
                if vol_ratio > 1:
                    adjustments['volatility_reduction'] = 1 - (1 / vol_ratio)

            # Adjust for concentration
            total_exposure = await self.calculate_exposure(self.positions)
            if total_exposure > 0:
                concentration = abs(position['position_value']) / total_exposure
                if concentration > self.config['position_limits']['position_concentration_limit']:
                    adjustments['concentration_reduction'] = (
                        concentration - 
                        self.config['position_limits']['position_concentration_limit']
                    ) / concentration

            return adjustments if adjustments else None
            
        except Exception as e:
            logger.error(f"Error calculating position adjustments: {e}")
            raise

    async def get_position_history(
        self,
        symbol: Optional[str] = None,
        start_time: Optional[datetime] = None
    ) -> List[Dict]:
        """Get position history for analysis."""
        try:
            history = self.position_history
            
            if symbol:
                history = [h for h in history if h['symbol'] == symbol]
                
            if start_time:
                history = [h for h in history if h['timestamp'] >= start_time]
                
            return history
            
        except Exception as e:
            logger.error(f"Error getting position history: {e}")
            raise

    async def calculate_portfolio_metrics(self) -> Dict:
        """Calculate portfolio-wide metrics."""
        try:
            metrics = {
                'total_exposure': await self.calculate_exposure(self.positions),
                'position_count': len(self.positions),
                'largest_position': max(
                    (abs(pos['position_value']) for pos in self.positions.values()),
                    default=0
                ),
                'total_margin_used': sum(
                    pos.get('margin_used', 0) for pos in self.positions.values()
                ),
                'total_unrealized_pnl': sum(
                    pos.get('unrealized_pnl', 0) for pos in self.positions.values()
                )
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating portfolio metrics: {e}")
            raise

    def get_position(self, symbol: str) -> Optional[Dict]:
        """Get current position for a symbol."""
        return self.positions.get(symbol)

    def get_all_positions(self) -> Dict:
        """Get all current positions."""
        return self.positions.copy()
