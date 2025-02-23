"""
Performance tests for the risk monitoring and alert system.

This module evaluates the performance characteristics of the risk monitoring system,
including response times, system load, and scalability under various conditions.
"""

import pytest
import time
import numpy as np
import psutil
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import os
import logging
from typing import List, Tuple, Dict
import statistics
import json

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from risk_management.risk_monitoring.risk_monitor import (
    RiskMonitor,
    MarketData,
    PositionData
)
from risk_management.risk_monitoring.alert_system import AlertSystem, AlertMessage

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('performance_test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PerformanceMetrics:
    """Container for performance test results."""
    
    def __init__(self):
        self.response_times: List[float] = []
        self.cpu_usage: List[float] = []
        self.memory_usage: List[float] = []
        self.throughput: List[float] = []
        self.error_rates: List[float] = []
        
    def to_dict(self) -> Dict:
        """Convert metrics to dictionary format."""
        return {
            'response_time': {
                'avg': self.avg_response_time,
                'p95': self.p95_response_time,
                'max': self.max_response_time
            },
            'cpu_usage': {
                'avg': statistics.mean(self.cpu_usage) if self.cpu_usage else 0.0,
                'max': max(self.cpu_usage) if self.cpu_usage else 0.0
            },
            'memory_usage': {
                'avg': statistics.mean(self.memory_usage) if self.memory_usage else 0.0,
                'max': max(self.memory_usage) if self.memory_usage else 0.0
            },
            'throughput': {
                'avg': statistics.mean(self.throughput) if self.throughput else 0.0
            },
            'error_rate': statistics.mean(self.error_rates) if self.error_rates else 0.0
        }
    
    def save_to_file(self, filename: str):
        """Save metrics to JSON file."""
        with open(filename, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @property
    def avg_response_time(self) -> float:
        return statistics.mean(self.response_times) if self.response_times else 0.0
    
    @property
    def p95_response_time(self) -> float:
        return np.percentile(self.response_times, 95) if self.response_times else 0.0
    
    @property
    def max_response_time(self) -> float:
        return max(self.response_times) if self.response_times else 0.0

class SystemMonitor(threading.Thread):
    """Monitor system resources during tests."""
    
    def __init__(self, interval: float = 1.0):
        super().__init__()
        self.interval = interval
        self.metrics = PerformanceMetrics()
        self._stop_event = threading.Event()
    
    def run(self):
        while not self._stop_event.is_set():
            self.metrics.cpu_usage.append(psutil.cpu_percent())
            self.metrics.memory_usage.append(psutil.Process().memory_info().rss / 1024 / 1024)
            time.sleep(self.interval)
    
    def stop(self):
        self._stop_event.set()
        self.join()

def generate_test_data(num_samples: int) -> List[Tuple[MarketData, PositionData]]:
    """Generate test data for performance testing.
    
    Args:
        num_samples: Number of test data points to generate
        
    Returns:
        List of (MarketData, PositionData) tuples
    """
    base_price = 50000.0
    base_time = time.time()
    
    test_data = []
    for i in range(num_samples):
        # Generate realistic price movements using random walk
        price_change = np.random.normal(0, 100)
        current_price = base_price + price_change
        
        market_data = MarketData(
            timestamp=base_time + i,
            current_price=current_price,
            bid_price=current_price - 5,
            ask_price=current_price + 5,
            volume_24h=1000000.0 + np.random.normal(0, 100000),
            order_book_depth={
                'bids': [(current_price - 5 - j, 1.0) for j in range(5)],
                'asks': [(current_price + 5 + j, 1.0) for j in range(5)]
            },
            funding_rate=0.0001 + np.random.normal(0, 0.0001)
        )
        
        position_data = PositionData(
            timestamp=base_time + i,
            size=1.0 + np.random.normal(0, 0.1),
            entry_price=base_price,
            current_margin=5000.0 + np.random.normal(0, 100),
            required_margin=3000.0,
            leverage=10.0,
            unrealized_pnl=(current_price - base_price),
            liquidation_price=base_price * 0.9
        )
        
        test_data.append((market_data, position_data))
    
    return test_data

def measure_execution_time(func, *args) -> Tuple[float, any]:
    """Measure execution time of a function.
    
    Args:
        func: Function to measure
        args: Arguments to pass to the function
        
    Returns:
        Tuple of (execution time, function result)
    """
    start_time = time.perf_counter()
    result = func(*args)
    end_time = time.perf_counter()
    return end_time - start_time, result

@pytest.fixture
def risk_monitor():
    """Create a RiskMonitor instance for testing."""
    return RiskMonitor()

@pytest.fixture
def alert_system():
    """Create an AlertSystem instance for testing."""
    return AlertSystem()

def test_risk_evaluation_performance(risk_monitor):
    """Test the performance of risk evaluation under various loads."""
    test_sizes = [100, 1000, 10000]
    metrics = PerformanceMetrics()
    system_monitor = SystemMonitor()
    system_monitor.start()
    
    try:
        for size in test_sizes:
            logger.info(f"Testing risk evaluation with {size} samples")
            test_data = generate_test_data(size)
            
            start_time = time.time()
            success_count = 0
            
            # Measure individual evaluation times
            for market_data, position_data in test_data:
                try:
                    exec_time, _ = measure_execution_time(
                        risk_monitor.evaluate_risks,
                        market_data,
                        position_data
                    )
                    metrics.response_times.append(exec_time)
                    success_count += 1
                except Exception as e:
                    logger.error(f"Error in risk evaluation: {e}")
            
            # Calculate throughput and error rate
            elapsed_time = time.time() - start_time
            metrics.throughput.append(success_count / elapsed_time)
            metrics.error_rates.append(1 - (success_count / size))
            
            logger.info(f"Average response time: {metrics.avg_response_time:.6f} seconds")
            logger.info(f"95th percentile response time: {metrics.p95_response_time:.6f} seconds")
            logger.info(f"Maximum response time: {metrics.max_response_time:.6f} seconds")
            logger.info(f"Throughput: {metrics.throughput[-1]:.2f} requests/second")
            logger.info(f"Error rate: {metrics.error_rates[-1]:.2%}")
            
            # Assert performance requirements
            assert metrics.avg_response_time < 0.05, "Average response time too high"
            assert metrics.p95_response_time < 0.1, "P95 response time too high"
            assert metrics.error_rates[-1] < 0.01, "Error rate too high"
    
    finally:
        system_monitor.stop()
        metrics.cpu_usage.extend(system_monitor.metrics.cpu_usage)
        metrics.memory_usage.extend(system_monitor.metrics.memory_usage)
        
        # Save metrics to file
        metrics.save_to_file('risk_evaluation_performance.json')

def test_alert_system_performance(alert_system):
    """Test the performance of alert sending under load."""
    num_alerts = 100
    metrics = PerformanceMetrics()
    system_monitor = SystemMonitor()
    system_monitor.start()
    
    try:
        # Create test alerts
        test_alerts = [
            AlertMessage(
                severity="HIGH",
                title=f"Test Alert {i}",
                description="Performance test alert",
                metrics={'test_metric': float(i)},
                timestamp=time.time()
            )
            for i in range(num_alerts)
        ]
        
        # Test sequential alert sending
        logger.info("Testing sequential alert sending")
        start_time = time.time()
        success_count = 0
        
        for alert in test_alerts[:10]:  # Test with subset for sequential
            try:
                exec_time, _ = measure_execution_time(
                    alert_system.send_alert,
                    alert,
                    ['slack']  # Use only Slack for performance testing
                )
                metrics.response_times.append(exec_time)
                success_count += 1
            except Exception as e:
                logger.error(f"Error in alert sending: {e}")
        
        elapsed_time = time.time() - start_time
        metrics.throughput.append(success_count / elapsed_time)
        metrics.error_rates.append(1 - (success_count / 10))
        
        logger.info(f"Sequential average response time: {metrics.avg_response_time:.6f} seconds")
        logger.info(f"Throughput: {metrics.throughput[-1]:.2f} requests/second")
        logger.info(f"Error rate: {metrics.error_rates[-1]:.2%}")
        
        # Test parallel alert sending
        logger.info("Testing parallel alert sending")
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            start_time = time.time()
            
            for alert in test_alerts:
                future = executor.submit(
                    alert_system.send_alert,
                    alert,
                    ['slack']
                )
                futures.append(future)
            
            success_count = 0
            for future in futures:
                try:
                    future.result()
                    success_count += 1
                except Exception as e:
                    logger.error(f"Error in alert sending: {e}")
            
            elapsed_time = time.time() - start_time
            metrics.throughput.append(success_count / elapsed_time)
            metrics.error_rates.append(1 - (success_count / num_alerts))
            
            logger.info(f"Parallel average response time: {metrics.avg_response_time:.6f} seconds")
            logger.info(f"Throughput: {metrics.throughput[-1]:.2f} requests/second")
            logger.info(f"Error rate: {metrics.error_rates[-1]:.2%}")
        
        # Assert performance requirements
        assert metrics.avg_response_time < 0.5, "Alert sending too slow"
        assert metrics.error_rates[-1] < 0.01, "Error rate too high"
    
    finally:
        system_monitor.stop()
        metrics.cpu_usage.extend(system_monitor.metrics.cpu_usage)
        metrics.memory_usage.extend(system_monitor.metrics.memory_usage)
        
        # Save metrics to file
        metrics.save_to_file('alert_system_performance.json')

def test_system_scalability(risk_monitor, alert_system):
    """Test the scalability of the entire system."""
    num_iterations = 1000
    metrics = PerformanceMetrics()
    system_monitor = SystemMonitor()
    system_monitor.start()
    
    try:
        logger.info(f"Testing system scalability with {num_iterations} iterations")
        test_data = generate_test_data(num_iterations)
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            start_time = time.time()
            
            for market_data, position_data in test_data:
                future = executor.submit(
                    lambda: (
                        risk_monitor.evaluate_risks(market_data, position_data),
                        alert_system.send_alert(
                            AlertMessage(
                                severity="INFO",
                                title="Test Alert",
                                description="Scalability test",
                                metrics={'test_metric': 1.0},
                                timestamp=time.time()
                            ),
                            ['slack']
                        )
                    )
                )
                futures.append(future)
            
            success_count = 0
            for future in futures:
                try:
                    future.result()
                    success_count += 1
                except Exception as e:
                    logger.error(f"Error in scalability test: {e}")
            
            elapsed_time = time.time() - start_time
            metrics.throughput.append(success_count / elapsed_time)
            metrics.error_rates.append(1 - (success_count / num_iterations))
            
            logger.info(f"System throughput: {metrics.throughput[-1]:.2f} operations/second")
            logger.info(f"Error rate: {metrics.error_rates[-1]:.2%}")
            
            # Assert scalability requirements
            assert metrics.throughput[-1] >= 10.0, "System throughput too low"
            assert metrics.error_rates[-1] < 0.01, "Error rate too high"
    
    finally:
        system_monitor.stop()
        metrics.cpu_usage.extend(system_monitor.metrics.cpu_usage)
        metrics.memory_usage.extend(system_monitor.metrics.memory_usage)
        
        # Save metrics to file
        metrics.save_to_file('system_scalability_performance.json')

def test_memory_usage():
    """Test memory usage under sustained load."""
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # Create test load
    risk_monitor = RiskMonitor()
    test_data = generate_test_data(10000)
    
    # Run operations
    for market_data, position_data in test_data:
        risk_monitor.evaluate_risks(market_data, position_data)
    
    final_memory = process.memory_info().rss / 1024 / 1024  # MB
    memory_increase = final_memory - initial_memory
    
    logger.info(f"Memory usage increased by {memory_increase:.2f} MB")
    assert memory_increase < 100, "Memory usage increased too much"

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
