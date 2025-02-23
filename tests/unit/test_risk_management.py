import pytest
import asyncio
from datetime import datetime, timedelta
import numpy as np
from risk_management.risk_manager import RiskManager
from risk_management.position_manager import PositionManager
from risk_management.stop_loss_manager import StopLossManager
from risk_management.var_calculator import VaRCalculator
from risk_management.volatility_manager import VolatilityManager

@pytest.fixture
def risk_config():
    return {
        'margin_settings': {
            'minimum_margin_buffer': 0.15,
            'warning_margin_buffer': 0.25,
            'critical_margin_buffer': 0.20,
            'max_leverage': 10.0
        },
        'position_limits': {
            'max_position_size': 1000000,
            'max_position_ratio': 0.1,
            'position_concentration_limit': 0.3
        },
        'stop_loss': {
            'initial_stop_loss': 0.02,
            'trailing_stop_loss': 0.015,
            'max_daily_loss': 0.05,
            'max_drawdown': 0.15
        },
        'volatility': {
            'lookback_period': 24,
            'volatility_scaling': True,
            'max_volatility': 0.5,
            'min_volatility': 0.05
        },
        'var_settings': {
            'confidence_level': 0.99,
            'time_horizon': 24,
            'historical_window': 30,
            'max_var_ratio': 0.1
        },
        'monitoring': {
            'update_interval': 60,
            'alert_threshold': 0.8,
            'log_level': 'INFO',
            'retry_attempts': 3,
            'retry_delay': 5
        }
    }

@pytest.fixture
def sample_positions():
    return {
        'BTC-USD': {
            'position_value': 100000,
            'margin_used': 10000,
            'margin_available': 50000,
            'current_equity': 45000,
            'peak_equity': 50000,
            'daily_pnl': -2000,
            'unrealized_pnl': -1000
        },
        'ETH-USD': {
            'position_value': 50000,
            'margin_used': 5000,
            'margin_available': 25000,
            'current_equity': 23000,
            'peak_equity': 25000,
            'daily_pnl': -1000,
            'unrealized_pnl': -500
        }
    }

@pytest.fixture
def sample_market_data():
    return {
        'BTC-USD': {'price': 50000},
        'ETH-USD': {'price': 3000}
    }

class TestRiskManager:
    @pytest.mark.asyncio
    async def test_calculate_margin_buffer(self, risk_config, sample_positions):
        risk_manager = RiskManager(config_path=None)
        risk_manager.config = risk_config
        
        margin_buffer = await risk_manager.calculate_margin_buffer(sample_positions)
        expected_buffer = (75000 - 15000) / 75000  # (total_available - total_used) / total_available
        
        assert abs(margin_buffer - expected_buffer) < 1e-6

    @pytest.mark.asyncio
    async def test_calculate_leverage(self, risk_config, sample_positions):
        risk_manager = RiskManager(config_path=None)
        risk_manager.config = risk_config
        
        leverage = await risk_manager.calculate_leverage(sample_positions)
        expected_leverage = 150000 / 75000  # total_position_value / total_margin_available
        
        assert abs(leverage - expected_leverage) < 1e-6

    @pytest.mark.asyncio
    async def test_risk_alerts(self, risk_config, sample_positions):
        risk_manager = RiskManager(config_path=None)
        risk_manager.config = risk_config
        
        metrics = await risk_manager.update_risk_metrics(
            market_data={'BTC-USD': {'price': 45000}},
            positions=sample_positions
        )
        
        alerts = risk_manager.get_active_alerts()
        assert len(alerts) > 0  # Should generate alerts due to low margin buffer

class TestPositionManager:
    @pytest.mark.asyncio
    async def test_calculate_exposure(self, risk_config, sample_positions):
        position_manager = PositionManager(risk_config)
        
        exposure = await position_manager.calculate_exposure(sample_positions)
        expected_exposure = 150000  # Sum of absolute position values
        
        assert abs(exposure - expected_exposure) < 1e-6

    @pytest.mark.asyncio
    async def test_check_position_limits(self, risk_config):
        position_manager = PositionManager(risk_config)
        
        # Test position within limits
        result = await position_manager.check_position_limits('BTC-USD', 500000)
        assert result is True
        
        # Test position exceeding limits
        result = await position_manager.check_position_limits('BTC-USD', 1500000)
        assert result is False

class TestStopLossManager:
    @pytest.mark.asyncio
    async def test_update_stop_loss(self, risk_config):
        stop_loss_manager = StopLossManager(risk_config)
        
        # Test long position stop loss
        result = await stop_loss_manager.update_stop_loss(
            symbol='BTC-USD',
            current_price=50000,
            position_size=1.0,
            entry_price=51000
        )
        
        assert result is None  # Stop loss not triggered
        
        # Test stop loss trigger
        result = await stop_loss_manager.update_stop_loss(
            symbol='BTC-USD',
            current_price=49000,
            position_size=1.0,
            entry_price=51000
        )
        
        assert result is not None
        assert result['action'] == 'close'

    @pytest.mark.asyncio
    async def test_daily_pnl_limit(self, risk_config):
        stop_loss_manager = StopLossManager(risk_config)
        
        # Test within daily loss limit
        result = await stop_loss_manager.update_daily_pnl(-1000)
        assert result is None
        
        # Test exceeding daily loss limit
        result = await stop_loss_manager.update_daily_pnl(-5000)
        assert result is not None
        assert result['action'] == 'halt_trading'

class TestVaRCalculator:
    @pytest.mark.asyncio
    async def test_calculate_var(self, risk_config, sample_positions, sample_market_data):
        var_calculator = VaRCalculator(risk_config)
        
        # Add some price history
        for i in range(100):
            price = 50000 * (1 + np.random.normal(0, 0.01))
            await var_calculator.update_price_history('BTC-USD', price)
        
        var = await var_calculator.calculate_var(sample_market_data, sample_positions)
        assert var > 0  # VaR should be positive
        assert var < 1  # VaR should be less than 100%

class TestVolatilityManager:
    @pytest.mark.asyncio
    async def test_calculate_volatility(self, risk_config, sample_market_data):
        volatility_manager = VolatilityManager(risk_config)
        
        # Add price history
        current_time = datetime.now()
        for i in range(100):
            time = current_time + timedelta(minutes=i)
            price = 50000 * (1 + np.random.normal(0, 0.01))
            await volatility_manager.update_price_history('BTC-USD', price, time)
        
        volatilities = await volatility_manager.calculate_volatility(sample_market_data)
        assert 'BTC-USD' in volatilities
        assert volatilities['BTC-USD'] > 0
        assert volatilities['BTC-USD'] < 1

    @pytest.mark.asyncio
    async def test_volatility_adjusted_position_size(self, risk_config):
        volatility_manager = VolatilityManager(risk_config)
        
        # Test position size adjustment
        base_size = 100000
        adjusted_size = await volatility_manager.calculate_volatility_adjusted_position_size(
            'BTC-USD',
            base_size,
            current_volatility=0.25  # Half of max_volatility
        )
        
        assert adjusted_size > base_size  # Should increase position size due to lower volatility
