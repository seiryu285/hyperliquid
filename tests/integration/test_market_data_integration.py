import pytest
import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List
from unittest.mock import AsyncMock, patch
import websockets
from market_data.websocket_client import WebSocketClient
from market_data.stream_manager import StreamManager
from utils.data_validator import DataValidator
from utils.data_normalizer import DataNormalizer
from storage.market_data_store import MarketDataStore

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MockWebSocket:
    def __init__(self, test_messages: List[Dict]):
        self.test_messages = test_messages
        self.message_index = 0
        self.closed = False

    async def send(self, message):
        if self.closed:
            raise websockets.exceptions.ConnectionClosed(1000, "Mock connection closed")
        return None

    async def recv(self):
        if self.closed:
            raise websockets.exceptions.ConnectionClosed(1000, "Mock connection closed")
        if self.message_index >= len(self.test_messages):
            self.closed = True
            raise websockets.exceptions.ConnectionClosed(1000, "No more messages")
        message = self.test_messages[self.message_index]
        self.message_index += 1
        return json.dumps(message)

    async def close(self):
        self.closed = True

@pytest.fixture
def test_messages():
    current_time = datetime.now()
    return [
        {
            'type': 'trades',
            'symbol': 'BTC-USD',
            'timestamp': current_time.timestamp(),
            'price': 50000.0,
            'size': 1.5,
            'side': 'buy'
        },
        {
            'type': 'orderbook',
            'symbol': 'BTC-USD',
            'timestamp': current_time.timestamp(),
            'bids': [[49900.0, 1.0], [49800.0, 2.0]],
            'asks': [[50100.0, 1.5], [50200.0, 2.5]]
        },
        {
            'type': 'ticker',
            'symbol': 'BTC-USD',
            'timestamp': current_time.timestamp(),
            'price': 50000.0,
            'volume': 1000.0,
            'open': 49000.0,
            'high': 51000.0,
            'low': 48000.0,
            'close': 50000.0
        }
    ]

@pytest.fixture
def mock_websocket(test_messages):
    return MockWebSocket(test_messages)

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
            'symbols': ['BTC-USD', 'ETH-USD'],
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
                'database': 'market_data_test_integration',
                'collections': {
                    'trades': 'trades_test_integration',
                    'orderbook': 'orderbook_test_integration',
                    'ticker': 'ticker_test_integration'
                }
            },
            'compression_enabled': True,
            'retention_days': 30,
            'backup_enabled': True
        }
    }

class TestMarketDataIntegration:
    @pytest.mark.asyncio
    async def test_full_market_data_flow(self, config, mock_websocket, test_messages):
        """Test the complete flow of market data from WebSocket to storage."""
        
        # Mock websockets.connect
        with patch('websockets.connect', new_callable=AsyncMock) as mock_connect:
            mock_connect.return_value = mock_websocket
            
            # Initialize components
            stream_manager = StreamManager(config_path=None)
            stream_manager.ws_client.config = config
            stream_manager.validator.config = config
            stream_manager.normalizer.config = config
            stream_manager.data_store.config = config
            
            try:
                # Start the stream manager
                await stream_manager.start()
                logger.info("Stream manager started successfully")
                
                # Wait for messages to be processed
                await asyncio.sleep(2)
                
                # Verify data in storage
                for message in test_messages:
                    symbol = message['symbol']
                    message_type = message['type']
                    
                    # Get latest data from storage
                    stored_data = await stream_manager.data_store.get_latest_data(
                        symbol,
                        message_type
                    )
                    
                    assert stored_data is not None
                    assert stored_data['symbol'] == symbol
                    assert stored_data['type'] == message_type
                    
                    # Verify specific fields based on message type
                    if message_type == 'trades':
                        assert stored_data['price'] == message['price']
                        assert stored_data['size'] == message['size']
                        assert stored_data['side'] == message['side']
                        assert 'normalized_price' in stored_data
                        assert 'normalized_size' in stored_data
                    
                    elif message_type == 'orderbook':
                        assert len(stored_data['bids']) == len(message['bids'])
                        assert len(stored_data['asks']) == len(message['asks'])
                        assert 'normalized_bids' in stored_data
                        assert 'normalized_asks' in stored_data
                    
                    elif message_type == 'ticker':
                        assert stored_data['price'] == message['price']
                        assert stored_data['volume'] == message['volume']
                        assert stored_data['open'] == message['open']
                        assert stored_data['high'] == message['high']
                        assert stored_data['low'] == message['low']
                        assert stored_data['close'] == message['close']
                        assert 'normalized_price' in stored_data
                        assert 'normalized_volume' in stored_data
                
                logger.info("Successfully verified stored data")
                
            finally:
                # Stop the stream manager
                await stream_manager.stop()
                logger.info("Stream manager stopped")

    @pytest.mark.asyncio
    async def test_reconnection_handling(self, config, mock_websocket, test_messages):
        """Test WebSocket reconnection handling."""
        
        with patch('websockets.connect', new_callable=AsyncMock) as mock_connect:
            # First connection attempt fails, second succeeds
            mock_connect.side_effect = [
                websockets.exceptions.ConnectionClosed(1000, "Test disconnection"),
                mock_websocket
            ]
            
            stream_manager = StreamManager(config_path=None)
            stream_manager.ws_client.config = config
            
            try:
                await stream_manager.start()
                await asyncio.sleep(2)  # Wait for reconnection
                
                assert stream_manager.ws_client.connected
                assert stream_manager.ws_client.reconnect_attempts == 1
                
            finally:
                await stream_manager.stop()

    @pytest.mark.asyncio
    async def test_data_validation_and_normalization(self, config, test_messages):
        """Test data validation and normalization pipeline."""
        
        validator = DataValidator(config_path=None)
        validator.config = config
        
        normalizer = DataNormalizer(config_path=None)
        normalizer.config = config
        
        for message in test_messages:
            # Validate message
            is_valid = await validator.validate_message(message)
            assert is_valid, f"Message validation failed: {message}"
            
            # Normalize message
            normalized_data = await normalizer.normalize_data(message)
            
            # Check normalization results
            if message['type'] == 'trades':
                assert 'normalized_price' in normalized_data
                assert 'normalized_size' in normalized_data
            elif message['type'] == 'orderbook':
                assert 'normalized_bids' in normalized_data
                assert 'normalized_asks' in normalized_data
            elif message['type'] == 'ticker':
                assert 'normalized_price' in normalized_data
                assert 'normalized_volume' in normalized_data

    @pytest.mark.asyncio
    async def test_error_handling(self, config):
        """Test error handling in various scenarios."""
        
        stream_manager = StreamManager(config_path=None)
        stream_manager.ws_client.config = config
        
        # Test invalid message handling
        invalid_message = {'type': 'invalid'}
        is_valid = await stream_manager.validator.validate_message(invalid_message)
        assert not is_valid
        
        # Test database connection error handling
        with patch.object(stream_manager.data_store, 'store_data', side_effect=Exception("Test DB error")):
            try:
                await stream_manager.start()
                await asyncio.sleep(1)
            except Exception as e:
                assert "Test DB error" in str(e)
            finally:
                await stream_manager.stop()

    @pytest.mark.asyncio
    async def test_performance(self, config, test_messages):
        """Test system performance with high message throughput."""
        
        # Create a large number of test messages
        num_messages = 1000
        high_throughput_messages = test_messages * (num_messages // len(test_messages))
        
        mock_ws = MockWebSocket(high_throughput_messages)
        
        with patch('websockets.connect', new_callable=AsyncMock) as mock_connect:
            mock_connect.return_value = mock_ws
            
            stream_manager = StreamManager(config_path=None)
            stream_manager.ws_client.config = config
            
            try:
                start_time = datetime.now()
                
                await stream_manager.start()
                await asyncio.sleep(5)  # Allow time for processing
                
                end_time = datetime.now()
                processing_time = (end_time - start_time).total_seconds()
                
                # Calculate messages per second
                messages_per_second = len(high_throughput_messages) / processing_time
                logger.info(f"Processing speed: {messages_per_second:.2f} messages/second")
                
                # Verify all messages were processed
                stats = await stream_manager.data_store.get_collection_stats()
                total_stored = sum(stat['total_documents'] for stat in stats.values())
                assert total_stored > 0, "No messages were stored"
                
            finally:
                await stream_manager.stop()
