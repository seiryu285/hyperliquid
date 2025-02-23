# Volatility Adjustment Strategy
# Code to adjust strategies based on volatility.

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import pandas as pd
from scipy.stats import norm

@dataclass
class VolatilityMetrics:
    current_volatility: float
    historical_volatility: float
    volatility_ratio: float
    adjusted_position_size: float
    confidence_level: float

class VolatilityAdjustment:
    def __init__(
        self,
        lookback_period: int = 30,
        volatility_threshold: float = 2.0,
        min_position_size: float = 0.1,
        confidence_level: float = 0.95,
        use_garch: bool = False
    ):
        """Initialize Volatility Adjustment strategy.
        
        Args:
            lookback_period (int): Period for historical volatility calculation
            volatility_threshold (float): Threshold for volatility ratio adjustment
            min_position_size (float): Minimum allowed position size as fraction
            confidence_level (float): Confidence level for VaR calculation
            use_garch (bool): Whether to use GARCH model for volatility forecasting
        """
        self.lookback_period = lookback_period
        self.volatility_threshold = volatility_threshold
        self.min_position_size = min_position_size
        self.confidence_level = confidence_level
        self.use_garch = use_garch
        
        if use_garch:
            try:
                from arch import arch_model
                self.arch_model = arch_model
            except ImportError:
                print("Warning: arch package not found. Falling back to simple volatility calculation.")
                self.use_garch = False

    def calculate_historical_volatility(
        self,
        returns: List[float],
        annualize: bool = True
    ) -> float:
        """Calculate historical volatility from returns.
        
        Args:
            returns (List[float]): List of historical returns
            annualize (bool): Whether to annualize the volatility
            
        Returns:
            float: Historical volatility
        """
        if len(returns) < 2:
            return 0.0
            
        volatility = np.std(returns, ddof=1)
        
        if annualize:
            # Assume daily data, annualize by multiplying by sqrt(252)
            volatility *= np.sqrt(252)
            
        return volatility

    def calculate_garch_volatility(
        self,
        returns: List[float],
        forecast_horizon: int = 1
    ) -> float:
        """Calculate volatility forecast using GARCH(1,1) model.
        
        Args:
            returns (List[float]): Historical returns
            forecast_horizon (int): Number of periods to forecast
            
        Returns:
            float: Forecasted volatility
        """
        if not self.use_garch or len(returns) < self.lookback_period:
            return self.calculate_historical_volatility(returns)
            
        try:
            # Fit GARCH(1,1) model
            model = self.arch_model(returns, vol='Garch', p=1, q=1)
            result = model.fit(disp='off')
            
            # Get volatility forecast
            forecast = result.forecast(horizon=forecast_horizon)
            return np.sqrt(forecast.variance.values[-1, -1])
        except Exception as e:
            print(f"GARCH model failed: {e}. Using simple volatility calculation.")
            return self.calculate_historical_volatility(returns)

    def calculate_value_at_risk(
        self,
        returns: List[float],
        position_size: float,
        confidence_level: Optional[float] = None
    ) -> float:
        """Calculate Value at Risk for the position.
        
        Args:
            returns (List[float]): Historical returns
            position_size (float): Current position size
            confidence_level (float, optional): Confidence level for VaR
            
        Returns:
            float: Value at Risk
        """
        if confidence_level is None:
            confidence_level = self.confidence_level
            
        if len(returns) < 2:
            return 0.0
            
        # Calculate volatility and mean return
        volatility = self.calculate_historical_volatility(returns, annualize=False)
        mean_return = np.mean(returns)
        
        # Calculate VaR using normal distribution
        z_score = norm.ppf(1 - confidence_level)
        var = position_size * (mean_return + z_score * volatility)
        
        return abs(var)

    def adjust_position_size(
        self,
        current_position_size: float,
        historical_returns: List[float],
        recent_returns: List[float]
    ) -> VolatilityMetrics:
        """Adjust position size based on volatility comparison.
        
        Args:
            current_position_size (float): Current position size
            historical_returns (List[float]): Long-term historical returns
            recent_returns (List[float]): Recent returns for current volatility
            
        Returns:
            VolatilityMetrics: Calculated volatility metrics and adjusted position
        """
        # Calculate volatilities
        if self.use_garch:
            current_vol = self.calculate_garch_volatility(recent_returns)
            historical_vol = self.calculate_garch_volatility(historical_returns)
        else:
            current_vol = self.calculate_historical_volatility(recent_returns)
            historical_vol = self.calculate_historical_volatility(historical_returns)
            
        # Calculate volatility ratio
        vol_ratio = current_vol / historical_vol if historical_vol > 0 else 1.0
        
        # Adjust position size based on volatility ratio
        adjustment_factor = 1.0
        if vol_ratio > self.volatility_threshold:
            # Reduce position size when volatility is high
            adjustment_factor = self.volatility_threshold / vol_ratio
        elif vol_ratio < 1.0 / self.volatility_threshold:
            # Increase position size when volatility is low
            adjustment_factor = min(2.0, 1.0 / vol_ratio)
            
        adjusted_size = max(
            self.min_position_size,
            current_position_size * adjustment_factor
        )
        
        return VolatilityMetrics(
            current_volatility=current_vol,
            historical_volatility=historical_vol,
            volatility_ratio=vol_ratio,
            adjusted_position_size=adjusted_size,
            confidence_level=self.confidence_level
        )

    def get_risk_metrics(
        self,
        position_size: float,
        historical_returns: List[float]
    ) -> Dict[str, float]:
        """Calculate various risk metrics for the position.
        
        Args:
            position_size (float): Position size
            historical_returns (List[float]): Historical returns
            
        Returns:
            Dict[str, float]: Dictionary of risk metrics
        """
        volatility = self.calculate_historical_volatility(historical_returns)
        var = self.calculate_value_at_risk(historical_returns, position_size)
        
        # Calculate additional risk metrics
        sharpe_ratio = 0.0
        if volatility > 0:
            risk_free_rate = 0.02  # Assume 2% risk-free rate
            excess_returns = np.mean(historical_returns) - risk_free_rate
            sharpe_ratio = excess_returns / volatility
            
        return {
            'volatility': volatility,
            'value_at_risk': var,
            'sharpe_ratio': sharpe_ratio,
            'position_size': position_size
        }

if __name__ == '__main__':
    print('Volatility Adjustment Strategy')