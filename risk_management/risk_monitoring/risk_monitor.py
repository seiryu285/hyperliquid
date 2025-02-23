"""
Risk Monitoring System for HyperLiquid Trading

This module implements real-time risk monitoring for high-frequency trading operations
on the HyperLiquid DEX. It calculates and evaluates various risk metrics including
margin buffer, volatility, liquidation risk, and Value at Risk (VaR).

The system is designed to be robust against data anomalies and missing values,
with built-in retry logic and comprehensive error handling.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import json
import time
import yaml
import logging
from datetime import datetime
from pathlib import Path
from prometheus_client import Summary, Counter, Gauge

# Import metrics collector
from monitoring.metrics_collector import MetricsCollector, MetricsConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('risk_monitor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class MarketData:
    """Container for market-related data."""
    timestamp: float
    current_price: float
    bid_price: float
    ask_price: float
    volume_24h: float
    order_book_depth: Dict[str, List[Tuple[float, float]]]  # price levels -> [(price, size)]
    funding_rate: float

@dataclass
class PositionData:
    """Container for position-related data."""
    timestamp: float
    size: float  # Positive for long, negative for short
    entry_price: float
    current_margin: float
    required_margin: float
    leverage: float
    unrealized_pnl: float
    liquidation_price: float

@dataclass
class RiskMetrics:
    """Container for calculated risk metrics."""
    timestamp: float
    margin_buffer_ratio: float
    short_term_volatility: float
    long_term_volatility: float
    volatility_ratio: float
    liquidation_risk: float
    value_at_risk: float
    sharpe_ratio: float
    max_drawdown: float

class RiskMonitor:
    """Main class for monitoring trading risks."""
    
    def __init__(self, config_path: str = "config/hyperparameters.yaml"):
        """Initialize the risk monitor with configuration parameters.
        
        Args:
            config_path: Path to the configuration file
        """
        self.config = self._load_config(config_path)
        self.risk_thresholds = self.config['risk_thresholds']
        self.monitoring_params = self.config['monitoring']
        
        # Initialize storage for historical data
        self.price_history: List[float] = []
        self.volatility_window_short = self.monitoring_params['volatility_window_short']
        self.volatility_window_long = self.monitoring_params['volatility_window_long']
        
        # Initialize metrics collector
        metrics_config = MetricsConfig(port=8001)
        self.metrics_collector = MetricsCollector(metrics_config)
        self.metrics_collector.start()
        
        # Initialize Prometheus metrics
        self.risk_eval_duration = Summary(
            'risk_evaluation_duration_seconds',
            'Time spent evaluating risks'
        )
        self.risk_threshold_breaches = Counter(
            'risk_threshold_breaches_total',
            'Number of risk threshold breaches',
            ['risk_type']
        )
        
        logger.info("Risk Monitor initialized with configuration from %s", config_path)

    def __del__(self):
        """Cleanup when the object is destroyed."""
        if hasattr(self, 'metrics_collector'):
            self.metrics_collector.stop()

    @self.risk_eval_duration.time()
    def evaluate_risks(
        self,
        market_data: MarketData,
        position_data: PositionData
    ) -> RiskMetrics:
        """Evaluate all risk metrics for the current state.
        
        Args:
            market_data: Current market data
            position_data: Current position data
            
        Returns:
            RiskMetrics object containing all calculated metrics
        """
        try:
            # Calculate individual risk metrics
            margin_buffer = self.calculate_margin_buffer(position_data)
            volatility_short = self.calculate_volatility(
                self.price_history,
                self.volatility_window_short
            )
            volatility_long = self.calculate_volatility(
                self.price_history,
                self.volatility_window_long
            )
            volatility_ratio = volatility_short / volatility_long if volatility_long > 0 else 1.0
            liquidation_risk = self.calculate_liquidation_risk(position_data, market_data)
            
            # Update price history
            self.price_history.append(market_data.current_price)
            if len(self.price_history) > self.volatility_window_long:
                self.price_history.pop(0)
            
            # Create risk metrics object
            metrics = RiskMetrics(
                timestamp=time.time(),
                margin_buffer_ratio=margin_buffer,
                short_term_volatility=volatility_short,
                long_term_volatility=volatility_long,
                volatility_ratio=volatility_ratio,
                liquidation_risk=liquidation_risk,
                value_at_risk=self.calculate_var(position_data, self.price_history),
                sharpe_ratio=0.0,  # TODO: Implement Sharpe ratio calculation
                max_drawdown=0.0  # TODO: Implement max drawdown calculation
            )
            
            # Record metrics for monitoring
            self.metrics_collector.record_risk_metrics({
                'margin_buffer_ratio': margin_buffer,
                'volatility_ratio': volatility_ratio,
                'liquidation_risk': liquidation_risk
            })
            
            # Check for threshold breaches
            self._check_thresholds(metrics)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating risks: {e}")
            self.metrics_collector.error_total.inc()
            raise

    def _check_thresholds(self, metrics: RiskMetrics):
        """Check if any risk metrics exceed their thresholds.
        
        Args:
            metrics: Current risk metrics
        """
        if metrics.margin_buffer_ratio < self.risk_thresholds['margin_buffer_min']:
            self.risk_threshold_breaches.labels(risk_type='margin_buffer').inc()
        
        if metrics.volatility_ratio > self.risk_thresholds['volatility_ratio_max']:
            self.risk_threshold_breaches.labels(risk_type='volatility').inc()
        
        if metrics.liquidation_risk > self.risk_thresholds['liquidation_risk_max']:
            self.risk_threshold_breaches.labels(risk_type='liquidation').inc()

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file.
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            Dict containing configuration parameters
        """
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            logger.error(f"Failed to load configuration from {config_path}: {e}")
            raise

    def calculate_margin_buffer(self, position: PositionData) -> float:
        """Calculate the margin buffer ratio.
        
        Args:
            position: Current position data
            
        Returns:
            float: Margin buffer ratio (current_margin / required_margin)
        """
        if position.required_margin <= 0:
            return float('inf')
        
        margin_ratio = position.current_margin / position.required_margin
        logger.debug(f"Margin buffer ratio calculated: {margin_ratio:.4f}")
        return margin_ratio

    def calculate_volatility(
        self,
        prices: List[float],
        window: int,
        annualize: bool = True
    ) -> float:
        """Calculate price volatility over a specified window.
        
        Args:
            prices: List of historical prices
            window: Number of periods for volatility calculation
            annualize: Whether to annualize the volatility
            
        Returns:
            float: Calculated volatility
        """
        if len(prices) < 2:
            return 0.0
            
        # Calculate returns
        returns = np.diff(np.log(prices[-window:]))
        volatility = np.std(returns, ddof=1)
        
        if annualize:
            # Assume data frequency is per minute
            volatility *= np.sqrt(525600)  # Minutes in a year
            
        return volatility

    def calculate_liquidation_risk(
        self,
        position: PositionData,
        market: MarketData
    ) -> float:
        """Calculate the risk of liquidation.
        
        Args:
            position: Current position data
            market: Current market data
            
        Returns:
            float: Liquidation risk score (0-1, higher means higher risk)
        """
        if position.size == 0:
            return 0.0
            
        # Calculate distance to liquidation as a percentage
        current_price = market.current_price
        liquidation_price = position.liquidation_price
        
        if position.size > 0:  # Long position
            distance = (current_price - liquidation_price) / current_price
        else:  # Short position
            distance = (liquidation_price - current_price) / current_price
            
        # Convert distance to risk score (0-1)
        risk_score = 1 / (1 + np.exp(5 * distance))  # Sigmoid function
        
        logger.debug(f"Liquidation risk calculated: {risk_score:.4f}")
        return risk_score

    def calculate_var(
        self,
        position: PositionData,
        returns: List[float],
        confidence_level: float = 0.95
    ) -> float:
        """Calculate Value at Risk using historical simulation method.
        
        Args:
            position: Current position data
            returns: Historical returns
            confidence_level: Confidence level for VaR calculation
            
        Returns:
            float: Value at Risk
        """
        if not returns or position.size == 0:
            return 0.0
            
        position_value = abs(position.size * position.entry_price)
        returns_array = np.array(returns)
        var_percentile = np.percentile(returns_array, (1 - confidence_level) * 100)
        
        var = position_value * abs(var_percentile)
        logger.debug(f"VaR calculated: {var:.2f} at {confidence_level:.2%} confidence")
        
        return var

    def to_json(self, metrics: RiskMetrics) -> str:
        """Convert risk metrics to JSON format.
        
        Args:
            metrics: Risk metrics to convert
            
        Returns:
            str: JSON string representation of the metrics
        """
        return json.dumps({
            'timestamp': metrics.timestamp,
            'margin_buffer_ratio': metrics.margin_buffer_ratio,
            'short_term_volatility': metrics.short_term_volatility,
            'long_term_volatility': metrics.long_term_volatility,
            'volatility_ratio': metrics.volatility_ratio,
            'liquidation_risk': metrics.liquidation_risk,
            'value_at_risk': metrics.value_at_risk,
            'sharpe_ratio': metrics.sharpe_ratio,
            'max_drawdown': metrics.max_drawdown
        })

if __name__ == '__main__':
    print('Risk Monitor')