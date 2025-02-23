"""
Integration tests for the risk monitoring and alert system.

This module tests the integration between RiskMonitor and AlertSystem,
ensuring that risk threshold breaches correctly trigger alerts.
"""

import pytest
import time
from unittest.mock import Mock, patch
from pathlib import Path
import sys
import os

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from risk_management.risk_monitoring.risk_monitor import (
    RiskMonitor,
    MarketData,
    PositionData,
    RiskMetrics
)
from risk_management.risk_monitoring.alert_system import (
    AlertSystem,
    AlertMessage
)

@pytest.fixture
def risk_monitor():
    """Create a RiskMonitor instance for testing."""
    return RiskMonitor()

@pytest.fixture
def alert_system():
    """Create an AlertSystem instance for testing."""
    return AlertSystem()

@pytest.fixture
def sample_market_data():
    """Create sample market data for testing."""
    return MarketData(
        timestamp=time.time(),
        current_price=50000.0,
        bid_price=49990.0,
        ask_price=50010.0,
        volume_24h=1000000.0,
        order_book_depth={
            'bids': [(49990, 1.0), (49980, 2.0)],
            'asks': [(50010, 1.0), (50020, 2.0)]
        },
        funding_rate=0.0001
    )

@pytest.fixture
def sample_position_data():
    """Create sample position data for testing."""
    return PositionData(
        timestamp=time.time(),
        size=1.0,
        entry_price=50000.0,
        current_margin=5000.0,
        required_margin=3000.0,
        leverage=10.0,
        unrealized_pnl=0.0,
        liquidation_price=45000.0
    )

def test_risk_monitor_initialization(risk_monitor):
    """Test that RiskMonitor initializes correctly."""
    assert risk_monitor is not None
    assert hasattr(risk_monitor, 'risk_thresholds')
    assert hasattr(risk_monitor, 'monitoring_params')

def test_alert_system_initialization(alert_system):
    """Test that AlertSystem initializes correctly."""
    assert alert_system is not None
    assert hasattr(alert_system, 'alert_config')
    assert hasattr(alert_system, 'env_config')

def test_risk_evaluation(risk_monitor, sample_market_data, sample_position_data):
    """Test that risk evaluation produces valid metrics."""
    metrics = risk_monitor.evaluate_risks(sample_market_data, sample_position_data)
    
    assert isinstance(metrics, RiskMetrics)
    assert metrics.margin_buffer_ratio > 0
    assert metrics.liquidation_risk >= 0 and metrics.liquidation_risk <= 1
    assert metrics.value_at_risk >= 0

@patch('risk_management.risk_monitoring.alert_system.requests.post')
@patch('risk_management.risk_monitoring.alert_system.smtplib.SMTP')
def test_alert_triggering(mock_smtp, mock_post, risk_monitor, alert_system,
                         sample_market_data, sample_position_data):
    """Test that risk threshold breaches trigger alerts."""
    # Modify position data to trigger alerts
    sample_position_data.current_margin = 1000.0  # Lower margin to trigger alert
    
    # Evaluate risks
    metrics = risk_monitor.evaluate_risks(sample_market_data, sample_position_data)
    threshold_breaches = risk_monitor.check_thresholds(metrics)
    
    # Check if any thresholds were breached
    assert any(threshold_breaches.values()), "Expected at least one threshold breach"
    
    # Create and send alert
    alert_message = AlertMessage(
        severity="HIGH",
        title="Risk Threshold Breach",
        description="Multiple risk thresholds exceeded",
        metrics={k: v for k, v in vars(metrics).items() if k != 'timestamp'},
        timestamp=time.time()
    )
    
    # Configure mocks
    mock_post.return_value.status_code = 200
    mock_smtp_instance = Mock()
    mock_smtp.return_value.__enter__.return_value = mock_smtp_instance
    
    # Send alert
    results = alert_system.send_alert(alert_message, channels=['slack', 'email'])
    
    # Verify alerts were sent
    assert results['slack'] is True
    assert results['email'] is True
    assert mock_post.called
    assert mock_smtp_instance.send_message.called

def test_risk_monitor_error_handling(risk_monitor):
    """Test that RiskMonitor handles invalid inputs gracefully."""
    with pytest.raises(Exception):
        risk_monitor.evaluate_risks(None, None)

def test_alert_system_rate_limiting(alert_system):
    """Test that AlertSystem properly implements rate limiting."""
    message = AlertMessage(
        severity="LOW",
        title="Test Alert",
        description="Testing rate limiting",
        metrics={'test': 1.0},
        timestamp=time.time()
    )
    
    # Send first alert
    first_result = alert_system.send_alert(message, channels=['slack'])
    assert first_result['slack'] is True
    
    # Immediate second alert should be blocked by rate limiting
    second_result = alert_system.send_alert(message, channels=['slack'])
    assert second_result['slack'] is False

def test_integration_workflow(risk_monitor, alert_system, sample_market_data,
                            sample_position_data):
    """Test the complete integration workflow from risk evaluation to alert sending."""
    # 1. Evaluate initial risks
    initial_metrics = risk_monitor.evaluate_risks(sample_market_data, sample_position_data)
    initial_breaches = risk_monitor.check_thresholds(initial_metrics)
    
    # 2. Modify position to trigger alert
    sample_position_data.current_margin = 1000.0
    sample_position_data.required_margin = 2000.0
    
    # 3. Re-evaluate risks
    updated_metrics = risk_monitor.evaluate_risks(sample_market_data, sample_position_data)
    updated_breaches = risk_monitor.check_thresholds(updated_metrics)
    
    # 4. Verify that risk status changed
    assert any(updated_breaches.values()), "Expected risk threshold breach"
    assert updated_metrics.margin_buffer_ratio < initial_metrics.margin_buffer_ratio
    
    # 5. Test alert creation and sending
    with patch('risk_management.risk_monitoring.alert_system.requests.post') as mock_post:
        mock_post.return_value.status_code = 200
        
        alert_message = AlertMessage(
            severity="HIGH",
            title="Margin Buffer Critical",
            description="Margin buffer ratio has fallen below critical threshold",
            metrics={'margin_buffer_ratio': updated_metrics.margin_buffer_ratio},
            timestamp=time.time()
        )
        
        results = alert_system.send_alert(alert_message, channels=['slack'])
        assert results['slack'] is True
        assert mock_post.called

if __name__ == '__main__':
    pytest.main([__file__])
