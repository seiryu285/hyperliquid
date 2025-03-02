import pytest
import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List
from unittest.mock import AsyncMock, patch
import websockets
import numpy as np
from market_data.websocket_client import WebSocketClient
from market_data.stream_manager import StreamManager
from utils.data_validator import DataValidator
from utils.data_normalizer import DataNormalizer
from storage.market_data_store import MarketDataStore

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PerformanceMetrics:
    def __init__(self):
        self.message_processing_times = []
        self.validation_times = []
        self.normalization_times = []
        self.storage_times = []
        self.total_messages = 0
        self.start_time = None
        self.end_time = None

    def add_message_time(self, processing_time: float):
        self.message_processing_times.append(processing_time)

    def add_validation_time(self, validation_time: float):
        self.validation_times.append(validation_time)

    def add_normalization_time(self, normalization_time: float):
        self.normalization_times.append(normalization_time)

    def add_storage_time(self, storage_time: float):
        self.storage_times.append(storage_time)

    def start(self):
        self.start_time = time.time()

    def stop(self):
        self.end_time = time.time()

    def get_summary(self) -> Dict:
        total_time = self.end_time - self.start_time
        return {
            'total_messages': len(self.message_processing_times),
            'total_time_seconds': total_time,
            'messages_per_second': len(self.message_processing_times) / total_time,
            'average_processing_time_ms': np.mean(self.message_processing_times) * 1000,
            'p95_processing_time_ms': np.percentile(self.message_processing_times, 95) * 1000,
            'p99_processing_time_ms': np.percentile(self.message_processing_times, 99) * 1000,
            'average_validation_time_ms': np.mean(self.validation_times) * 1000,
            'average_normalization_time_ms': np.mean(self.normalization_times) * 1000,
            'average_storage_time_ms': np.mean(self.storage_times) * 1000,
        }

class PerformanceTestWebSocket:
    def __init__(self, message_generator, message_delay: float = 0.001):
        self.message_generator = message_generator
        self.message_delay = message_delay
        self.closed = False

    async def send(self, message):
        if self.closed:
            raise websockets.exceptions.ConnectionClosed(1000, "Mock connection closed")
        return None

    async def recv(self):
        if self.closed:
            raise websockets.exceptions.ConnectionClosed(1000, "Mock connection closed")
        await asyncio.sleep(self.message_delay)
        return json.dumps(next(self.message_generator))

    async def close(self):
        self.closed = True

def generate_test_messages(num_messages: int):
    current_time = datetime.now()
    message_types = ['trades', 'orderbook', 'ticker']
    symbols = ['BTC-USD', 'ETH-USD', 'SOL-USD']
    
    for i in range(num_messages):
        message_type = message_types[i % len(message_types)]
        symbol = symbols[i % len(symbols)]
        timestamp = (current_time + timedelta(milliseconds=i*100)).timestamp()
        
        if message_type == 'trades':
            yield {
                'type': message_type,
                'symbol': symbol,
                'timestamp': timestamp,
                'price': 50000.0 + np.random.normal(0, 100),
                'size': np.random.uniform(0.1, 5.0),
                'side': 'buy' if np.random.random() > 0.5 else 'sell'
            }
        elif message_type == 'orderbook':
            num_levels = 10
            mid_price = 50000.0 + np.random.normal(0, 100)
            
            bids = [[mid_price - i*10, np.random.uniform(0.1, 5.0)] for i in range(num_levels)]
            asks = [[mid_price + i*10, np.random.uniform(0.1, 5.0)] for i in range(num_levels)]
            
            yield {
                'type': message_type,
                'symbol': symbol,
                'timestamp': timestamp,
                'bids': bids,
                'asks': asks
            }
        else:  # ticker
            price = 50000.0 + np.random.normal(0, 100)
            yield {
                'type': message_type,
                'symbol': symbol,
                'timestamp': timestamp,
                'price': price,
                'volume': np.random.uniform(100, 1000),
                'open': price - np.random.uniform(0, 100),
                'high': price + np.random.uniform(0, 100),
                'low': price - np.random.uniform(0, 100),
                'close': price
            }

@pytest.fixture
def performance_metrics():
    return PerformanceMetrics()

@pytest.fixture
def config():
    return {
        'api': {
            'ws_url': 'wss://api.hyperliquid.xyz/ws',
            'timeout': 30,
            'max_retries': 3,
            'retry_delay': 5
        },
        'websocket': {
            'ping_interval': 30,
            'reconnect_attempts': 5,
            'reconnect_delay': 5,
            'connection_timeout': 60,
            'heartbeat_timeout': 30
        },
        'collection': {
            'symbols': ['BTC-USD', 'ETH-USD', 'SOL-USD'],
            'channels': ['trades', 'orderbook', 'ticker'],
            'orderbook_depth': 20,
            'trade_history_limit': 1000,
            'update_interval': 1
        },
        'processing': {
            'batch_size': 100,
            'max_queue_size': 10000,
            'validation_enabled': True,
            'normalization_enabled': True,
            'missing_data_threshold': 0.1,
            'outlier_std_threshold': 3.0
        },
        'storage': {
            'mongodb': {
                'uri': 'mongodb://localhost:27017',
                'database': 'market_data_test_performance',
                'collections': {
                    'trades': 'trades_test_performance',
                    'orderbook': 'orderbook_test_performance',
                    'ticker': 'ticker_test_performance'
                }
            },
            'compression_enabled': True,
            'retention_days': 30,
            'backup_enabled': True
        }
    }

class TestMarketDataPerformance:
    @pytest.mark.asyncio
    async def test_message_throughput(self, config, performance_metrics):
        """Test system performance with high message throughput."""
        
        num_messages = 10000
        message_delay = 0.0001  # 0.1ms between messages
        
        # Create performance test websocket
        mock_ws = PerformanceTestWebSocket(
            generate_test_messages(num_messages),
            message_delay
        )
        
        with patch('websockets.connect', new_callable=AsyncMock) as mock_connect:
            mock_connect.return_value = mock_ws
            
            # Initialize components with performance tracking
            stream_manager = StreamManager(config_path=None)
            stream_manager.ws_client.config = config
            stream_manager.validator.config = config
            stream_manager.normalizer.config = config
            stream_manager.data_store.config = config
            
            try:
                performance_metrics.start()
                
                # Start processing
                await stream_manager.start()
                
                # Process messages and collect metrics
                while performance_metrics.total_messages < num_messages:
                    message_start = time.time()
                    
                    # Get message from queue
                    message = await stream_manager.ws_client.get_queue().get()
                    
                    # Validate
                    validation_start = time.time()
                    is_valid = await stream_manager.validator.validate_message(message)
                    validation_time = time.time() - validation_start
                    performance_metrics.add_validation_time(validation_time)
                    
                    if is_valid:
                        # Normalize
                        normalization_start = time.time()
                        normalized_data = await stream_manager.normalizer.normalize_data(message)
                        normalization_time = time.time() - normalization_start
                        performance_metrics.add_normalization_time(normalization_time)
                        
                        # Store
                        storage_start = time.time()
                        await stream_manager.data_store.store_data(normalized_data)
                        storage_time = time.time() - storage_start
                        performance_metrics.add_storage_time(storage_time)
                    
                    message_time = time.time() - message_start
                    performance_metrics.add_message_time(message_time)
                    performance_metrics.total_messages += 1
                
                performance_metrics.stop()
                
                # Get and log performance summary
                summary = performance_metrics.get_summary()
                logger.info("Performance Test Results:")
                for metric, value in summary.items():
                    logger.info(f"{metric}: {value:.2f}")
                
                # Assert performance requirements
                assert summary['messages_per_second'] > 1000, "Processing speed below 1000 messages/second"
                assert summary['p95_processing_time_ms'] < 10, "95th percentile processing time above 10ms"
                assert summary['p99_processing_time_ms'] < 20, "99th percentile processing time above 20ms"
                
            finally:
                await stream_manager.stop()

    @pytest.mark.asyncio
    async def test_batch_processing_performance(self, config, performance_metrics):
        """Test batch processing performance."""
        
        batch_sizes = [10, 50, 100, 500]
        messages_per_batch = 1000
        
        results = {}
        
        for batch_size in batch_sizes:
            config['processing']['batch_size'] = batch_size
            
            mock_ws = PerformanceTestWebSocket(
                generate_test_messages(messages_per_batch),
                message_delay=0.0001
            )
            
            with patch('websockets.connect', new_callable=AsyncMock) as mock_connect:
                mock_connect.return_value = mock_ws
                
                stream_manager = StreamManager(config_path=None)
                stream_manager.ws_client.config = config
                stream_manager.validator.config = config
                stream_manager.normalizer.config = config
                stream_manager.data_store.config = config
                
                try:
                    performance_metrics.start()
                    await stream_manager.start()
                    await asyncio.sleep(5)  # Allow time for processing
                    performance_metrics.stop()
                    
                    results[batch_size] = performance_metrics.get_summary()
                    
                finally:
                    await stream_manager.stop()
        
        # Log results
        logger.info("\nBatch Processing Performance Results:")
        for batch_size, metrics in results.items():
            logger.info(f"\nBatch Size: {batch_size}")
            for metric, value in metrics.items():
                logger.info(f"{metric}: {value:.2f}")
        
        # Assert batch processing efficiency
        best_throughput = max(metrics['messages_per_second'] for metrics in results.values())
        logger.info(f"\nBest throughput achieved: {best_throughput:.2f} messages/second")
        assert best_throughput > 1000, "Batch processing throughput below target"

    @pytest.mark.asyncio
    async def test_memory_usage(self, config):
        """Test memory usage under load."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        num_messages = 5000
        mock_ws = PerformanceTestWebSocket(
            generate_test_messages(num_messages),
            message_delay=0.0001
        )
        
        with patch('websockets.connect', new_callable=AsyncMock) as mock_connect:
            mock_connect.return_value = mock_ws
            
            stream_manager = StreamManager(config_path=None)
            stream_manager.ws_client.config = config
            
            try:
                await stream_manager.start()
                await asyncio.sleep(5)  # Allow time for processing
                
                final_memory = process.memory_info().rss / 1024 / 1024  # MB
                memory_increase = final_memory - initial_memory
                
                logger.info(f"\nMemory Usage Test Results:")
                logger.info(f"Initial memory usage: {initial_memory:.2f} MB")
                logger.info(f"Final memory usage: {final_memory:.2f} MB")
                logger.info(f"Memory increase: {memory_increase:.2f} MB")
                
                # Assert reasonable memory usage
                assert memory_increase < 500, "Memory usage increased by more than 500MB"
                
            finally:
                await stream_manager.stop()
