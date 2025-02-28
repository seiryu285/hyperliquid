import asyncio
import json
import logging
import os
import websockets
import yaml
from datetime import datetime
from dotenv import load_dotenv
from typing import Dict, List, Optional, Tuple
from websockets.exceptions import ConnectionClosed
from pathlib import Path
from prometheus_client import Counter, Gauge, Histogram, REGISTRY

logger = logging.getLogger(__name__)

class DummyMetric:
    """Dummy metric class that implements the same interface as Prometheus metrics but does nothing."""
    def inc(self, *args, **kwargs):
        pass
        
    def set(self, *args, **kwargs):
        pass
        
    def observe(self, *args, **kwargs):
        pass
        
    def labels(self, *args, **kwargs):
        return self
        
    def time(self):
        class DummyTimer:
            def __enter__(self):
                return self
                
            def __exit__(self, exc_type, exc_val, exc_tb):
                pass
        return DummyTimer()

class WebSocketClient:
    def __init__(self, config_path: str = "config/market_data.yaml"):
        # Load environment variables
        load_dotenv()
        self.config = self._load_config(config_path)
        self.ws = None
        self.connected = False
        self.last_ping = datetime.now()
        self.message_queue = asyncio.Queue(maxsize=self.config['processing']['max_queue_size'])
        self.reconnect_attempts = 0
        self.subscriptions = set()
        self.connection_lock = asyncio.Lock()
        
        # Initialize metrics with unique names to avoid conflicts
        metric_prefix = 'hyperliquid_ws_'
        try:
            # Try to unregister existing metrics to avoid duplicates
            for metric in list(REGISTRY._names_to_collectors.values()):
                if metric.name.startswith(metric_prefix):
                    REGISTRY.unregister(metric)
                    
            self.ws_connected = Gauge(f'{metric_prefix}connected', 'WebSocket connection status')
            self.ws_messages_received = Counter(f'{metric_prefix}messages_received', 'Number of messages received')
            self.ws_messages_processed = Counter(f'{metric_prefix}messages_processed', 'Number of messages processed')
            self.ws_processing_time = Histogram(f'{metric_prefix}processing_time', 'Message processing time')
            self.ws_reconnects = Counter(f'{metric_prefix}reconnects', 'Number of reconnection attempts')
            self.ws_errors = Counter(f'{metric_prefix}errors', 'Number of WebSocket errors', ['error_type'])
            
            logger.info("Prometheus metrics initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Prometheus metrics: {e}")
            # Fallback to dummy metrics that don't actually record anything
            self.ws_connected = DummyMetric()
            self.ws_messages_received = DummyMetric()
            self.ws_messages_processed = DummyMetric()
            self.ws_processing_time = DummyMetric()
            self.ws_reconnects = DummyMetric()
            self.ws_errors = DummyMetric()

    def _load_config(self, config_path: str) -> dict:
        """Load market data configuration from YAML file and process environment variables."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Process environment variables in config
            required_env_vars = {
                'HYPERLIQUID_API_KEY': 'api.api_key',
                'HYPERLIQUID_API_SECRET': 'api.api_secret',
                'MONGODB_URI': 'storage.mongodb.uri'
            }
            
            for env_var, config_path in required_env_vars.items():
                env_value = os.getenv(env_var)
                if not env_value:
                    logger.warning(f"Environment variable {env_var} not found. Please set it in .env file.")
                    # Set a placeholder value for development
                    if 'MONGODB_URI' in env_var:
                        env_value = 'mongodb://localhost:27017'
                    else:
                        env_value = 'development_key'
                
                # Update nested config value
                config_keys = config_path.split('.')
                current = config
                for key in config_keys[:-1]:
                    current = current[key]
                current[config_keys[-1]] = env_value
            
            logger.info(f"Successfully loaded market data configuration from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load market data configuration: {e}")
            raise

    async def handle_connection_error(self, error: Exception) -> None:
        """Handle WebSocket connection errors with exponential backoff."""
        error_type = type(error).__name__
        self.ws_errors.labels(error_type=error_type).inc()
        
        if self.reconnect_attempts >= self.config['retry']['max_attempts']:
            logger.error(f"Max reconnection attempts reached: {self.reconnect_attempts}")
            raise Exception("Failed to establish WebSocket connection after maximum attempts")
            
        delay = min(
            self.config['retry']['initial_delay'] * (2 ** self.reconnect_attempts),
            self.config['retry']['max_delay']
        )
        
        logger.warning(f"Connection error: {error}. Retrying in {delay} seconds...")
        await asyncio.sleep(delay)
        self.reconnect_attempts += 1
        self.ws_reconnects.inc()
        await self.reconnect()

    def validate_market_data(self, data: Dict) -> Tuple[bool, Optional[str]]:
        """
        Validate incoming market data.
        
        Returns:
            Tuple[bool, Optional[str]]: (is_valid, error_message)
        """
        try:
            required_fields = {
                'timestamp': (int, float),
                'price': (int, float),
                'volume': (int, float),
                'symbol': str
            }
            
            for field, expected_type in required_fields.items():
                if field not in data:
                    return False, f"Missing required field: {field}"
                if not isinstance(data[field], expected_type):
                    return False, f"Invalid type for {field}: expected {expected_type}, got {type(data[field])}"
            
            # Additional validation rules
            if data['price'] <= 0 or data['volume'] < 0:
                return False, "Invalid price or volume value"
                
            return True, None
            
        except Exception as e:
            logger.error(f"Data validation error: {e}")
            return False, str(e)

    async def process_message(self, message: str) -> None:
        """Process incoming WebSocket messages with validation and metrics."""
        try:
            with self.ws_processing_time.time():
                data = json.loads(message)
                self.ws_messages_received.inc()
                
                # Validate data
                is_valid, error_message = self.validate_market_data(data)
                if not is_valid:
                    logger.error(f"Invalid market data: {error_message}")
                    self.ws_errors.labels(error_type='validation_error').inc()
                    return
                
                # Process valid data
                await self.message_queue.put(data)
                self.ws_messages_processed.inc()
                
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            self.ws_errors.labels(error_type='json_error').inc()
        except Exception as e:
            logger.error(f"Message processing error: {e}")
            self.ws_errors.labels(error_type='processing_error').inc()

    async def connect(self) -> None:
        """Establish WebSocket connection with error handling and metrics."""
        async with self.connection_lock:
            try:
                if self.ws is not None:
                    await self.ws.close()
                
                self.ws = await websockets.connect(
                    self.config['api']['ws_url'],
                    ping_interval=self.config['websocket']['ping_interval'],
                    close_timeout=self.config['websocket']['connection_timeout']
                )
                
                self.connected = True
                self.ws_connected.set(1)
                self.reconnect_attempts = 0
                logger.info("WebSocket connection established")
                
            except Exception as e:
                self.ws_connected.set(0)
                await self.handle_connection_error(e)

    async def _resubscribe(self) -> None:
        """Resubscribe to all previous channels after reconnection."""
        try:
            for subscription in self.subscriptions:
                await self.subscribe(subscription['channel'], subscription.get('symbols', []))
            logger.info("Successfully resubscribed to all channels")
        except Exception as e:
            logger.error(f"Error during resubscription: {e}")
            raise

    async def subscribe(self, channel: str, symbols: List[str]) -> None:
        """Subscribe to a market data channel."""
        try:
            subscription_message = {
                "op": "subscribe",
                "channel": channel,
                "symbols": symbols
            }
            
            await self.ws.send(json.dumps(subscription_message))
            self.subscriptions.add(frozenset({'channel': channel, 'symbols': tuple(symbols)}.items()))
            logger.info(f"Subscribed to channel {channel} for symbols {symbols}")
            
        except Exception as e:
            logger.error(f"Error subscribing to channel {channel}: {e}")
            raise

    async def unsubscribe(self, channel: str, symbols: List[str]) -> None:
        """Unsubscribe from a market data channel."""
        try:
            unsubscription_message = {
                "op": "unsubscribe",
                "channel": channel,
                "symbols": symbols
            }
            
            await self.ws.send(json.dumps(unsubscription_message))
            self.subscriptions.remove(frozenset({'channel': channel, 'symbols': tuple(symbols)}.items()))
            logger.info(f"Unsubscribed from channel {channel} for symbols {symbols}")
            
        except Exception as e:
            logger.error(f"Error unsubscribing from channel {channel}: {e}")
            raise

    async def receive(self) -> Optional[Dict]:
        """Receive and process incoming WebSocket messages."""
        while True:
            try:
                if not self.connected:
                    await self.connect()

                if not self.ws:
                    logger.error("WebSocket connection is None")
                    await asyncio.sleep(1)
                    continue

                message = await self.ws.recv()
                
                try:
                    await self.process_message(message)
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    
            except ConnectionClosed as e:
                logger.warning(f"WebSocket connection closed: {e}")
                self.connected = False
                await self.connect()
                
            except asyncio.CancelledError:
                logger.info("Receive task cancelled")
                raise
                
            except Exception as e:
                logger.error(f"Error receiving message: {e}")
                self.connected = False
                await asyncio.sleep(1)

    async def send(self, message: Dict) -> None:
        """Send a message through the WebSocket connection."""
        try:
            if not self.connected:
                await self.connect()
                
            if not self.ws:
                raise ConnectionError("WebSocket connection is None")
                
            await self.ws.send(json.dumps(message))
            logger.debug(f"Sent message: {message}")
            
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            self.connected = False
            raise

    async def heartbeat(self) -> None:
        """Send periodic heartbeat messages to keep the connection alive."""
        while True:
            try:
                if self.connected and self.ws:
                    await self.send({"op": "ping"})
                    self.last_ping = datetime.now()
                    
                await asyncio.sleep(self.config['websocket']['ping_interval'])
                
            except asyncio.CancelledError:
                logger.info("Heartbeat task cancelled")
                raise
                
            except Exception as e:
                logger.error(f"Error in heartbeat: {e}")
                self.connected = False
                await asyncio.sleep(1)

    async def monitor_connection(self) -> None:
        """Monitor WebSocket connection health."""
        while True:
            try:
                if self.connected and self.ws:
                    # Check if we haven't received a pong in too long
                    time_since_last_ping = (datetime.now() - self.last_ping).total_seconds()
                    if time_since_last_ping > self.config['websocket']['heartbeat_timeout']:
                        logger.warning(f"Connection appears to be stale (no pong for {time_since_last_ping}s), reconnecting...")
                        self.connected = False
                        await self.connect()
                        
                await asyncio.sleep(1)
                
            except asyncio.CancelledError:
                logger.info("Connection monitor task cancelled")
                raise
                
            except Exception as e:
                logger.error(f"Error in connection monitor: {e}")
                self.connected = False
                await asyncio.sleep(1)

    async def close(self) -> None:
        """Close the WebSocket connection."""
        try:
            if self.connected and self.ws:
                await self.ws.close()
                self.connected = False
                logger.info("WebSocket connection closed")
                
        except Exception as e:
            logger.error(f"Error closing WebSocket connection: {e}")
            self.connected = False
            raise

    def get_queue(self) -> asyncio.Queue:
        """Get the message queue for consuming market data."""
        return self.message_queue

    async def clear_queue(self) -> None:
        """Clear all messages from the queue."""
        try:
            while not self.message_queue.empty():
                try:
                    self.message_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
            logger.debug("Message queue cleared")
        except Exception as e:
            logger.error(f"Error clearing message queue: {e}")
            raise
