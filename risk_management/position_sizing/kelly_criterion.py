#!/usr/bin/env python3
"""
Kelly Criterion implementation for optimal position sizing in trading,
incorporating historical returns and real-time market data.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import pandas as pd
from scipy import stats

@dataclass
class MarketState:
    """Container for current market state information."""
    price: float
    volatility: float
    volume: float
    funding_rate: float
    margin_ratio: float
    leverage_limit: float

@dataclass
class KellyParams:
    win_rate: float
    profit_ratio: float  # Average profit on winning trades
    loss_ratio: float   # Average loss on losing trades
    kelly_fraction: float = 0.5  # Conservative adjustment

class KellyCriterion:
    def __init__(
        self,
        window_size: int = 100,
        confidence_level: float = 0.95,
        max_leverage: float = 10.0,
        min_samples: int = 50,
        kelly_fraction: float = 0.5
    ):
        """
        Initialize Kelly Criterion calculator.
        
        Args:
            window_size: Number of historical samples to consider
            confidence_level: Confidence level for statistical estimates
            max_leverage: Maximum allowed leverage
            min_samples: Minimum number of samples required for calculation
            kelly_fraction: Fraction of Kelly to use (0.5 = "Half Kelly")
        """
        self.window_size = window_size
        self.confidence_level = confidence_level
        self.max_leverage = max_leverage
        self.min_samples = min_samples
        self.kelly_fraction = kelly_fraction
        
        # Initialize storage for historical data
        self.returns_history: List[float] = []
        self.volatility_history: List[float] = []
        self.volume_history: List[float] = []
        
    def update_history(
        self,
        returns: float,
        volatility: float,
        volume: float
    ) -> None:
        """
        Update historical data with new observations.
        
        Args:
            returns: New return observation
            volatility: New volatility observation
            volume: New volume observation
        """
        # Add new data
        self.returns_history.append(returns)
        self.volatility_history.append(volatility)
        self.volume_history.append(volume)
        
        # Maintain window size
        if len(self.returns_history) > self.window_size:
            self.returns_history = self.returns_history[-self.window_size:]
            self.volatility_history = self.volatility_history[-self.window_size:]
            self.volume_history = self.volume_history[-self.window_size:]
            
    def estimate_parameters(
        self,
        market_state: MarketState
    ) -> Tuple[float, float, float]:
        """
        Estimate return distribution parameters using historical data.
        
        Args:
            market_state: Current market state information
            
        Returns:
            Tuple of (expected return, variance, win probability)
        """
        if len(self.returns_history) < self.min_samples:
            return 0.0, 0.0, 0.5
        
        # Calculate basic statistics
        returns = np.array(self.returns_history)
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        # Adjust expected return based on funding rate
        adjusted_return = mean_return - market_state.funding_rate
        
        # Calculate win probability (probability of positive return)
        win_prob = np.mean(returns > 0)
        
        # Adjust variance based on current volatility relative to historical
        current_vol = market_state.volatility
        hist_vol = np.mean(self.volatility_history)
        vol_ratio = current_vol / hist_vol if hist_vol > 0 else 1.0
        adjusted_variance = (std_return * vol_ratio) ** 2
        
        return adjusted_return, adjusted_variance, win_prob
        
    def calculate_kelly_fraction(
        self, 
        win_rate: float, 
        profit_ratio: float, 
        loss_ratio: float
    ) -> float:
        """Calculate the optimal Kelly fraction.
        
        Args:
            win_rate (float): Probability of winning (0-1)
            profit_ratio (float): Average profit on winning trades (positive)
            loss_ratio (float): Average loss on losing trades (positive)
            
        Returns:
            float: Optimal fraction of capital to risk
        """
        if not 0 <= win_rate <= 1:
            raise ValueError("Win rate must be between 0 and 1")
        if profit_ratio <= 0 or loss_ratio <= 0:
            raise ValueError("Profit and loss ratios must be positive")
            
        # Kelly Formula: f = (p(b+1) - 1)/b
        # where p is probability of win, b is odds received (profit/loss ratio)
        b = profit_ratio / loss_ratio
        f = (win_rate * (b + 1) - 1) / b
        
        # Apply conservative fraction and ensure non-negative
        return max(0, f * self.kelly_fraction)

    def calculate_position_size(
        self,
        market_state: MarketState,
        portfolio_value: float,
        risk_adjustment: float = 1.0
    ) -> Dict[str, float]:
        """
        Calculate optimal position size using Kelly Criterion.
        
        Args:
            market_state: Current market state information
            portfolio_value: Current portfolio value
            risk_adjustment: Factor to adjust risk (1.0 = full Kelly)
            
        Returns:
            Dictionary containing position sizing information
        """
        # Get parameter estimates
        exp_return, variance, win_prob = self.estimate_parameters(market_state)
        
        # Calculate Kelly fraction
        kelly_fraction = self.calculate_kelly_fraction(
            win_rate=win_prob,
            profit_ratio=exp_return / variance if variance != 0 else 0,
            loss_ratio=1.0
        )
        
        # Apply risk adjustment (fractional Kelly)
        kelly_fraction *= risk_adjustment
        
        # Apply leverage and margin constraints
        max_allowed_fraction = min(
            market_state.leverage_limit,
            1.0 / market_state.margin_ratio,
            self.max_leverage
        )
        
        # Constrain position size
        position_fraction = np.clip(kelly_fraction, 0.0, max_allowed_fraction)
        
        # Calculate actual position size
        position_size = position_fraction * portfolio_value
        
        # Calculate risk metrics
        value_at_risk = self._calculate_var(
            position_size,
            exp_return,
            variance,
            self.confidence_level
        )
        
        return {
            'position_size': position_size,
            'position_fraction': position_fraction,
            'kelly_fraction': kelly_fraction,
            'expected_return': exp_return,
            'volatility': np.sqrt(variance),
            'win_probability': win_prob,
            'value_at_risk': value_at_risk
        }
        
    def _calculate_var(
        self,
        position_size: float,
        exp_return: float,
        variance: float,
        confidence_level: float
    ) -> float:
        """
        Calculate Value at Risk for the position.
        
        Args:
            position_size: Size of the position
            exp_return: Expected return
            variance: Return variance
            confidence_level: Confidence level for VaR
            
        Returns:
            Value at Risk estimate
        """
        z_score = stats.norm.ppf(1 - confidence_level)
        var = position_size * (exp_return + z_score * np.sqrt(variance))
        return abs(var)
        
    def calculate_dynamic_leverage(
        self,
        market_state: MarketState,
        recent_performance: float,
        volatility_threshold: float = 0.02
    ) -> float:
        """
        Calculate dynamic leverage limit based on market conditions.
        
        Args:
            market_state: Current market state information
            recent_performance: Recent Sharpe ratio or similar performance metric
            volatility_threshold: Threshold for volatility scaling
            
        Returns:
            Adjusted leverage limit
        """
        # Base leverage limit
        base_leverage = market_state.leverage_limit
        
        # Volatility scaling
        vol_scale = min(1.0, volatility_threshold / market_state.volatility)
        
        # Performance scaling (reduce leverage if performing poorly)
        perf_scale = np.clip(recent_performance / 2.0, 0.0, 1.0)
        
        # Volume scaling (reduce leverage in low liquidity)
        avg_volume = np.mean(self.volume_history) if self.volume_history else 0
        vol_ratio = market_state.volume / avg_volume if avg_volume > 0 else 0
        liquidity_scale = np.clip(vol_ratio, 0.5, 1.0)
        
        # Combine all scaling factors
        adjusted_leverage = base_leverage * vol_scale * perf_scale * liquidity_scale
        
        return min(adjusted_leverage, self.max_leverage)
        
    def get_risk_metrics(
        self,
        position_size: float,
        market_state: MarketState
    ) -> Dict[str, float]:
        """
        Calculate various risk metrics for the current position.
        
        Args:
            position_size: Current position size
            market_state: Current market state
            
        Returns:
            Dictionary of risk metrics
        """
        exp_return, variance, _ = self.estimate_parameters(market_state)
        
        # Calculate various risk metrics
        volatility = np.sqrt(variance)
        var_95 = self._calculate_var(position_size, exp_return, variance, 0.95)
        var_99 = self._calculate_var(position_size, exp_return, variance, 0.99)
        
        # Calculate expected shortfall (CVaR)
        z_score_95 = stats.norm.ppf(0.05)
        cvar_95 = position_size * (exp_return + volatility * stats.norm.pdf(z_score_95) / 0.05)
        
        return {
            'volatility': volatility,
            'var_95': var_95,
            'var_99': var_99,
            'cvar_95': abs(cvar_95),
            'leverage_ratio': position_size / market_state.price,
            'margin_ratio': market_state.margin_ratio,
            'funding_cost': position_size * market_state.funding_rate
        }

if __name__ == '__main__':
    # Example usage
    kelly = KellyCriterion()
    
    # Example market state
    market_state = MarketState(
        price=50000.0,
        volatility=0.02,
        volume=1000.0,
        funding_rate=0.001,
        margin_ratio=0.1,
        leverage_limit=5.0
    )
    
    # Update with some historical data
    for _ in range(100):
        kelly.update_history(
            returns=np.random.normal(0.001, 0.02),
            volatility=np.random.normal(0.02, 0.005),
            volume=np.random.normal(1000, 100)
        )
    
    # Calculate position size
    position_info = kelly.calculate_position_size(
        market_state=market_state,
        portfolio_value=100000.0,
        risk_adjustment=0.5  # Half-Kelly for more conservative sizing
    )
    
    print("Position Information:")
    for key, value in position_info.items():
        print(f"{key}: {value:.4f}")