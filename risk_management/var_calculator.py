import logging
import numpy as np
from typing import Dict, List
from datetime import datetime, timedelta
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

class VaRCalculator:
    def __init__(self, config: dict):
        self.config = config
        self.price_history: Dict[str, List[float]] = {}
        self.returns_history: Dict[str, List[float]] = {}
        self.last_update = datetime.now()

    async def update_price_history(self, symbol: str, price: float) -> None:
        """Update price history for VaR calculation."""
        try:
            if symbol not in self.price_history:
                self.price_history[symbol] = []
                self.returns_history[symbol] = []

            self.price_history[symbol].append(price)

            # Calculate returns if we have at least 2 prices
            if len(self.price_history[symbol]) > 1:
                returns = np.log(price / self.price_history[symbol][-2])
                self.returns_history[symbol].append(returns)

            # Keep only required historical window
            window = self.config['var_settings']['historical_window'] * 24 * 60  # Convert days to minutes
            if len(self.price_history[symbol]) > window:
                self.price_history[symbol] = self.price_history[symbol][-window:]
                self.returns_history[symbol] = self.returns_history[symbol][-window:]
                
        except Exception as e:
            logger.error(f"Error updating price history for {symbol}: {e}")
            raise

    async def calculate_var(self, market_data: Dict, positions: Dict) -> float:
        """Calculate Value at Risk using historical simulation method."""
        try:
            portfolio_returns = []
            
            # Update price histories
            for symbol, data in market_data.items():
                if 'price' in data:
                    await self.update_price_history(symbol, data['price'])

            # Calculate portfolio returns
            total_portfolio_value = sum(abs(pos['position_value']) for pos in positions.values())
            
            if total_portfolio_value == 0:
                return 0.0

            # Calculate weighted returns for each position
            for symbol, position in positions.items():
                if symbol in self.returns_history and self.returns_history[symbol]:
                    weight = abs(position['position_value']) / total_portfolio_value
                    position_returns = np.array(self.returns_history[symbol]) * weight
                    
                    if len(portfolio_returns) == 0:
                        portfolio_returns = position_returns
                    else:
                        # Pad shorter array with zeros if histories have different lengths
                        if len(position_returns) < len(portfolio_returns):
                            position_returns = np.pad(
                                position_returns,
                                (len(portfolio_returns) - len(position_returns), 0)
                            )
                        elif len(portfolio_returns) < len(position_returns):
                            portfolio_returns = np.pad(
                                portfolio_returns,
                                (len(position_returns) - len(portfolio_returns), 0)
                            )
                        portfolio_returns += position_returns

            if len(portfolio_returns) == 0:
                return 0.0

            # Calculate VaR
            confidence_level = self.config['var_settings']['confidence_level']
            var = -np.percentile(portfolio_returns, (1 - confidence_level) * 100)

            # Scale VaR to the specified time horizon
            time_horizon = self.config['var_settings']['time_horizon']
            var_scaled = var * np.sqrt(time_horizon)

            return var_scaled
            
        except Exception as e:
            logger.error(f"Error calculating VaR: {e}")
            raise

    async def calculate_expected_shortfall(self, market_data: Dict, positions: Dict) -> float:
        """Calculate Expected Shortfall (CVaR) using historical simulation method."""
        try:
            portfolio_returns = []
            
            # Calculate portfolio returns (similar to VaR calculation)
            total_portfolio_value = sum(abs(pos['position_value']) for pos in positions.values())
            
            if total_portfolio_value == 0:
                return 0.0

            for symbol, position in positions.items():
                if symbol in self.returns_history and self.returns_history[symbol]:
                    weight = abs(position['position_value']) / total_portfolio_value
                    position_returns = np.array(self.returns_history[symbol]) * weight
                    
                    if len(portfolio_returns) == 0:
                        portfolio_returns = position_returns
                    else:
                        if len(position_returns) < len(portfolio_returns):
                            position_returns = np.pad(
                                position_returns,
                                (len(portfolio_returns) - len(position_returns), 0)
                            )
                        elif len(portfolio_returns) < len(position_returns):
                            portfolio_returns = np.pad(
                                portfolio_returns,
                                (len(position_returns) - len(portfolio_returns), 0)
                            )
                        portfolio_returns += position_returns

            if len(portfolio_returns) == 0:
                return 0.0

            # Calculate Expected Shortfall
            confidence_level = self.config['var_settings']['confidence_level']
            var_cutoff = np.percentile(portfolio_returns, (1 - confidence_level) * 100)
            es = -np.mean(portfolio_returns[portfolio_returns <= var_cutoff])

            # Scale ES to the specified time horizon
            time_horizon = self.config['var_settings']['time_horizon']
            es_scaled = es * np.sqrt(time_horizon)

            return es_scaled
            
        except Exception as e:
            logger.error(f"Error calculating Expected Shortfall: {e}")
            raise

    async def calculate_component_var(self, market_data: Dict, positions: Dict) -> Dict[str, float]:
        """Calculate Component VaR for each position."""
        try:
            component_vars = {}
            total_var = await self.calculate_var(market_data, positions)
            
            if total_var == 0:
                return {symbol: 0.0 for symbol in positions.keys()}

            total_portfolio_value = sum(abs(pos['position_value']) for pos in positions.values())
            
            if total_portfolio_value == 0:
                return {symbol: 0.0 for symbol in positions.keys()}

            for symbol, position in positions.items():
                weight = abs(position['position_value']) / total_portfolio_value
                if symbol in self.returns_history and self.returns_history[symbol]:
                    beta = self._calculate_beta(symbol, positions)
                    component_vars[symbol] = weight * beta * total_var

            return component_vars
            
        except Exception as e:
            logger.error(f"Error calculating Component VaR: {e}")
            raise

    def _calculate_beta(self, symbol: str, positions: Dict) -> float:
        """Calculate beta of a position relative to the portfolio."""
        try:
            if symbol not in self.returns_history:
                return 1.0

            position_returns = np.array(self.returns_history[symbol])
            portfolio_returns = []
            
            total_portfolio_value = sum(abs(pos['position_value']) for pos in positions.values())
            
            if total_portfolio_value == 0:
                return 1.0

            # Calculate portfolio returns
            for sym, position in positions.items():
                if sym in self.returns_history:
                    weight = abs(position['position_value']) / total_portfolio_value
                    weighted_returns = np.array(self.returns_history[sym]) * weight
                    
                    if len(portfolio_returns) == 0:
                        portfolio_returns = weighted_returns
                    else:
                        if len(weighted_returns) < len(portfolio_returns):
                            weighted_returns = np.pad(
                                weighted_returns,
                                (len(portfolio_returns) - len(weighted_returns), 0)
                            )
                        elif len(portfolio_returns) < len(weighted_returns):
                            portfolio_returns = np.pad(
                                portfolio_returns,
                                (len(weighted_returns) - len(portfolio_returns), 0)
                            )
                        portfolio_returns += weighted_returns

            if len(portfolio_returns) == 0:
                return 1.0

            # Calculate beta using covariance method
            covariance = np.cov(position_returns, portfolio_returns)[0][1]
            variance = np.var(portfolio_returns)
            
            if variance == 0:
                return 1.0
                
            beta = covariance / variance
            return beta
            
        except Exception as e:
            logger.error(f"Error calculating beta: {e}")
            return 1.0  # Return neutral beta on error
