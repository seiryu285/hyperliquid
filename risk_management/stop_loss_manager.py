import logging
from typing import Dict, Optional
from datetime import datetime, timedelta
import numpy as np

logger = logging.getLogger(__name__)

class StopLossManager:
    def __init__(self, config: dict):
        self.config = config
        self.stop_losses: Dict[str, Dict] = {}
        self.trailing_stops: Dict[str, Dict] = {}
        self.daily_pnl = 0.0
        self.last_reset = datetime.now()

    async def update_stop_loss(
        self,
        symbol: str,
        current_price: float,
        position_size: float,
        entry_price: float
    ) -> Optional[Dict]:
        """Update and check stop loss levels."""
        try:
            # Initialize stop loss if not exists
            if symbol not in self.stop_losses:
                self.stop_losses[symbol] = {
                    'level': entry_price * (
                        1 - self.config['stop_loss']['initial_stop_loss'] * np.sign(position_size)
                    ),
                    'trailing_level': entry_price,
                    'triggered': False
                }

            stop_loss = self.stop_losses[symbol]
            
            # Update trailing stop if price moved in favorable direction
            if position_size > 0:  # Long position
                if current_price > stop_loss['trailing_level']:
                    new_stop = current_price * (1 - self.config['stop_loss']['trailing_stop_loss'])
                    if new_stop > stop_loss['level']:
                        stop_loss['level'] = new_stop
                        stop_loss['trailing_level'] = current_price
                        logger.info(f"Updated trailing stop for {symbol} to {new_stop}")
                        
            else:  # Short position
                if current_price < stop_loss['trailing_level']:
                    new_stop = current_price * (1 + self.config['stop_loss']['trailing_stop_loss'])
                    if new_stop < stop_loss['level']:
                        stop_loss['level'] = new_stop
                        stop_loss['trailing_level'] = current_price
                        logger.info(f"Updated trailing stop for {symbol} to {new_stop}")

            # Check if stop loss is triggered
            if position_size > 0:  # Long position
                if current_price <= stop_loss['level']:
                    stop_loss['triggered'] = True
                    
            else:  # Short position
                if current_price >= stop_loss['level']:
                    stop_loss['triggered'] = True

            if stop_loss['triggered']:
                logger.warning(f"Stop loss triggered for {symbol} at {current_price}")
                return {
                    'action': 'close',
                    'reason': 'stop_loss_triggered',
                    'level': stop_loss['level']
                }

            return None
            
        except Exception as e:
            logger.error(f"Error updating stop loss for {symbol}: {e}")
            raise

    async def update_daily_pnl(self, pnl_change: float) -> Optional[Dict]:
        """Update daily PnL and check daily loss limit."""
        try:
            # Reset daily PnL if new day
            current_time = datetime.now()
            if current_time.date() > self.last_reset.date():
                self.daily_pnl = 0.0
                self.last_reset = current_time

            self.daily_pnl += pnl_change

            # Check daily loss limit
            max_daily_loss = self.config['stop_loss']['max_daily_loss']
            if self.daily_pnl < -max_daily_loss:
                logger.warning(f"Daily loss limit exceeded: {self.daily_pnl}")
                return {
                    'action': 'halt_trading',
                    'reason': 'daily_loss_limit_exceeded',
                    'current_loss': self.daily_pnl
                }

            return None
            
        except Exception as e:
            logger.error(f"Error updating daily PnL: {e}")
            raise

    async def calculate_dynamic_stop_loss(
        self,
        symbol: str,
        current_price: float,
        volatility: float,
        position_size: float
    ) -> float:
        """Calculate dynamic stop loss based on market conditions."""
        try:
            base_stop = self.config['stop_loss']['initial_stop_loss']
            
            # Adjust stop loss based on volatility
            vol_adjustment = min(
                2.0,  # Maximum volatility multiplier
                max(
                    0.5,  # Minimum volatility multiplier
                    volatility / self.config['volatility']['min_volatility']
                )
            )
            
            # Calculate dynamic stop loss percentage
            dynamic_stop = base_stop * vol_adjustment
            
            # Calculate actual stop loss level
            if position_size > 0:  # Long position
                stop_level = current_price * (1 - dynamic_stop)
            else:  # Short position
                stop_level = current_price * (1 + dynamic_stop)

            return stop_level
            
        except Exception as e:
            logger.error(f"Error calculating dynamic stop loss: {e}")
            raise

    async def get_stop_loss_info(self, symbol: str) -> Optional[Dict]:
        """Get current stop loss information for a symbol."""
        return self.stop_losses.get(symbol)

    def reset_daily_pnl(self) -> None:
        """Reset daily PnL counter."""
        self.daily_pnl = 0.0
        self.last_reset = datetime.now()

    def get_daily_pnl(self) -> float:
        """Get current daily PnL."""
        return self.daily_pnl
