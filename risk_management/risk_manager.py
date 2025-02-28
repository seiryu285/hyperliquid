import asyncio
import logging
import yaml
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from pydantic import BaseModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RiskMetrics(BaseModel):
    margin_buffer: float
    current_leverage: float
    position_exposure: float
    var_ratio: float
    volatility: float
    drawdown: float
    daily_pnl: float

class RiskAlert(BaseModel):
    alert_type: str
    severity: str
    message: str
    timestamp: datetime
    metrics: Dict[str, float]

class RiskManager:
    def __init__(self, config_path: str = "config/risk_management.yaml"):
        self.config = self._load_config(config_path)
        self.position_manager = PositionManager(self.config)
        self.stop_loss_manager = StopLossManager(self.config)
        self.var_calculator = VaRCalculator(self.config)
        self.volatility_manager = VolatilityManager(self.config)
        self.alerts: List[RiskAlert] = []
        self.last_update = datetime.now()

    def _load_config(self, config_path: str) -> dict:
        """Load risk management configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Successfully loaded risk configuration from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load risk configuration: {e}")
            raise

    async def update_risk_metrics(self, market_data: Dict, positions: Dict) -> RiskMetrics:
        """Update all risk metrics based on current market data and positions."""
        try:
            metrics = RiskMetrics(
                margin_buffer=await self.calculate_margin_buffer(positions),
                current_leverage=await self.calculate_leverage(positions),
                position_exposure=await self.position_manager.calculate_exposure(positions),
                var_ratio=await self.var_calculator.calculate_var(market_data, positions),
                volatility=await self.volatility_manager.calculate_volatility(market_data),
                drawdown=await self.calculate_drawdown(positions),
                daily_pnl=await self.calculate_daily_pnl(positions)
            )
            
            await self.check_risk_limits(metrics)
            return metrics
            
        except Exception as e:
            logger.error(f"Error updating risk metrics: {e}")
            raise

    async def calculate_margin_buffer(self, positions: Dict) -> float:
        """Calculate current margin buffer."""
        try:
            total_margin_used = sum(pos['margin_used'] for pos in positions.values())
            total_margin_available = sum(pos['margin_available'] for pos in positions.values())
            
            if total_margin_available == 0:
                return 1.0
                
            margin_buffer = (total_margin_available - total_margin_used) / total_margin_available
            return margin_buffer
            
        except Exception as e:
            logger.error(f"Error calculating margin buffer: {e}")
            raise

    async def calculate_leverage(self, positions: Dict) -> float:
        """Calculate current leverage."""
        try:
            total_position_value = sum(pos['position_value'] for pos in positions.values())
            total_margin_available = sum(pos['margin_available'] for pos in positions.values())
            
            if total_margin_available == 0:
                return 0.0
                
            leverage = total_position_value / total_margin_available
            return leverage
            
        except Exception as e:
            logger.error(f"Error calculating leverage: {e}")
            raise

    async def calculate_drawdown(self, positions: Dict) -> float:
        """Calculate current drawdown."""
        try:
            peak_equity = max(pos.get('peak_equity', 0) for pos in positions.values())
            current_equity = sum(pos['current_equity'] for pos in positions.values())
            
            if peak_equity == 0:
                return 0.0
                
            drawdown = (peak_equity - current_equity) / peak_equity
            return drawdown
            
        except Exception as e:
            logger.error(f"Error calculating drawdown: {e}")
            raise

    async def calculate_daily_pnl(self, positions: Dict) -> float:
        """Calculate daily PnL."""
        try:
            daily_pnl = sum(pos.get('daily_pnl', 0) for pos in positions.values())
            return daily_pnl
        except Exception as e:
            logger.error(f"Error calculating daily PnL: {e}")
            raise

    async def check_risk_limits(self, metrics: RiskMetrics) -> None:
        """Check if any risk limits are breached and generate alerts."""
        try:
            # Check margin buffer
            if metrics.margin_buffer < self.config['margin_settings']['critical_margin_buffer']:
                await self._generate_alert("CRITICAL", "Margin buffer critically low", metrics)
            elif metrics.margin_buffer < self.config['margin_settings']['warning_margin_buffer']:
                await self._generate_alert("WARNING", "Margin buffer low", metrics)

            # Check leverage
            if metrics.current_leverage > self.config['margin_settings']['max_leverage']:
                await self._generate_alert("CRITICAL", "Leverage exceeds maximum limit", metrics)

            # Check VaR
            if metrics.var_ratio > self.config['var_settings']['max_var_ratio']:
                await self._generate_alert("WARNING", "VaR ratio exceeds limit", metrics)

            # Check drawdown
            if metrics.drawdown > self.config['stop_loss']['max_drawdown']:
                await self._generate_alert("CRITICAL", "Maximum drawdown exceeded", metrics)

            # Check daily loss
            if metrics.daily_pnl < -self.config['stop_loss']['max_daily_loss']:
                await self._generate_alert("CRITICAL", "Maximum daily loss exceeded", metrics)

        except Exception as e:
            logger.error(f"Error checking risk limits: {e}")
            raise

    async def _generate_alert(self, severity: str, message: str, metrics: RiskMetrics) -> None:
        """Generate and log a risk alert."""
        try:
            alert = RiskAlert(
                alert_type="RISK_LIMIT_BREACH",
                severity=severity,
                message=message,
                timestamp=datetime.now(),
                metrics=metrics.dict()
            )
            
            self.alerts.append(alert)
            logger.warning(f"Risk Alert: {severity} - {message}")
            
            # Implement additional alert notifications here (e.g., email, Slack, etc.)
            
        except Exception as e:
            logger.error(f"Error generating alert: {e}")
            raise

    async def get_position_adjustments(self, metrics: RiskMetrics) -> Dict:
        """Calculate required position adjustments based on risk metrics."""
        try:
            adjustments = {}
            
            # Check if risk reduction is needed
            if (metrics.margin_buffer < self.config['margin_settings']['warning_margin_buffer'] or
                metrics.var_ratio > self.config['var_settings']['max_var_ratio']):
                
                # Calculate reduction ratio
                reduction_ratio = min(
                    1.0,
                    max(
                        0.1,  # Minimum reduction
                        (self.config['margin_settings']['warning_margin_buffer'] - metrics.margin_buffer) /
                        self.config['margin_settings']['warning_margin_buffer']
                    )
                )
                
                adjustments['reduction_ratio'] = reduction_ratio
                adjustments['reason'] = "Risk metrics exceeded limits"
                
            return adjustments
            
        except Exception as e:
            logger.error(f"Error calculating position adjustments: {e}")
            raise

    def get_active_alerts(self) -> List[RiskAlert]:
        """Get list of active risk alerts."""
        return self.alerts

    def clear_alerts(self) -> None:
        """Clear all active alerts."""
        self.alerts = []
