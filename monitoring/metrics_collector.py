"""
Metrics Collector for HyperLiquid Trading System

This module implements a Prometheus metrics collector that exposes various
performance and risk metrics from the trading system.
"""

import time
from typing import Dict, List
from dataclasses import dataclass
import logging
from prometheus_client import start_http_server, Gauge, Counter, Histogram
from prometheus_client.core import CollectorRegistry
import threading

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class MetricsConfig:
    """Configuration for metrics collection."""
    port: int = 8001
    collection_interval: float = 5.0
    metrics_prefix: str = 'hyperliquid_'

class MetricsCollector:
    """Collects and exposes metrics for Prometheus."""
    
    def __init__(self, config: MetricsConfig):
        """Initialize the metrics collector.
        
        Args:
            config: Configuration for metrics collection
        """
        self.config = config
        self.registry = CollectorRegistry()
        
        # Risk metrics
        self.margin_buffer_ratio = Gauge(
            f'{config.metrics_prefix}margin_buffer_ratio',
            'Current margin buffer ratio',
            registry=self.registry
        )
        
        self.volatility_ratio = Gauge(
            f'{config.metrics_prefix}volatility_ratio',
            'Ratio of short-term to long-term volatility',
            registry=self.registry
        )
        
        self.liquidation_risk = Gauge(
            f'{config.metrics_prefix}liquidation_risk',
            'Current liquidation risk score',
            registry=self.registry
        )
        
        # Performance metrics
        self.request_duration = Histogram(
            f'{config.metrics_prefix}request_duration_seconds',
            'Request duration in seconds',
            buckets=(0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0),
            registry=self.registry
        )
        
        self.request_total = Counter(
            f'{config.metrics_prefix}request_total',
            'Total number of requests',
            registry=self.registry
        )
        
        self.error_total = Counter(
            f'{config.metrics_prefix}error_total',
            'Total number of errors',
            registry=self.registry
        )
        
        # System metrics
        self.memory_usage = Gauge(
            f'{config.metrics_prefix}memory_usage_bytes',
            'Current memory usage in bytes',
            registry=self.registry
        )
        
        self.cpu_usage = Gauge(
            f'{config.metrics_prefix}cpu_usage_percent',
            'Current CPU usage percentage',
            registry=self.registry
        )
        
        self._stop_event = threading.Event()
        
    def start(self):
        """Start the metrics server and collection."""
        try:
            start_http_server(self.config.port, registry=self.registry)
            logger.info(f"Metrics server started on port {self.config.port}")
            
            # Start metrics collection in a separate thread
            self._collection_thread = threading.Thread(target=self._collect_metrics)
            self._collection_thread.start()
            
        except Exception as e:
            logger.error(f"Failed to start metrics server: {e}")
            raise
    
    def stop(self):
        """Stop the metrics collection."""
        self._stop_event.set()
        if hasattr(self, '_collection_thread'):
            self._collection_thread.join()
    
    def _collect_metrics(self):
        """Continuously collect metrics."""
        while not self._stop_event.is_set():
            try:
                self._update_system_metrics()
                time.sleep(self.config.collection_interval)
            except Exception as e:
                logger.error(f"Error collecting metrics: {e}")
                self.error_total.inc()
    
    def _update_system_metrics(self):
        """Update system metrics (CPU, memory usage)."""
        import psutil
        
        process = psutil.Process()
        self.memory_usage.set(process.memory_info().rss)
        self.cpu_usage.set(process.cpu_percent())
    
    def record_risk_metrics(self, metrics: Dict):
        """Record risk-related metrics.
        
        Args:
            metrics: Dictionary containing risk metrics
        """
        try:
            self.margin_buffer_ratio.set(metrics.get('margin_buffer_ratio', 0))
            self.volatility_ratio.set(metrics.get('volatility_ratio', 0))
            self.liquidation_risk.set(metrics.get('liquidation_risk', 0))
        except Exception as e:
            logger.error(f"Error recording risk metrics: {e}")
            self.error_total.inc()
    
    def record_request_duration(self, duration: float):
        """Record the duration of a request.
        
        Args:
            duration: Request duration in seconds
        """
        try:
            self.request_duration.observe(duration)
            self.request_total.inc()
        except Exception as e:
            logger.error(f"Error recording request duration: {e}")
            self.error_total.inc()

if __name__ == '__main__':
    # Example usage
    config = MetricsConfig(port=8001)
    collector = MetricsCollector(config)
    
    try:
        collector.start()
        # Keep the main thread running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        collector.stop()
