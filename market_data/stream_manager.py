import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional
from .websocket_client import WebSocketClient
from ..utils.data_validator import DataValidator
from ..utils.data_normalizer import DataNormalizer
from ..storage.market_data_store import MarketDataStore
from risk_management.risk_monitoring.alert_system import alert_system, AlertType, AlertLevel

logger = logging.getLogger(__name__)

class StreamManager:
    def __init__(self, config_path: str = "config/market_data.yaml"):
        self.ws_client = WebSocketClient(config_path)
        self.validator = DataValidator(config_path)
        self.normalizer = DataNormalizer(config_path)
        self.data_store = MarketDataStore(config_path)
        self.running = False
        self.tasks = []
        self.error_count = 0
        self.max_errors = 10
        self.error_reset_interval = 300  # 5 minutes

    async def start(self) -> None:
        """Start the market data streaming service."""
        try:
            await self.ws_client.connect()
            
            # Start background tasks
            self.tasks = [
                asyncio.create_task(self.ws_client.heartbeat()),
                asyncio.create_task(self.ws_client.monitor_connection()),
                asyncio.create_task(self._process_messages()),
                asyncio.create_task(self._store_data()),
                asyncio.create_task(self._reset_error_count())
            ]
            
            self.running = True
            logger.info("Stream manager started successfully")
            
            # Subscribe to configured channels
            await self._subscribe_to_channels()
            
        except Exception as e:
            logger.error(f"Error starting stream manager: {e}")
            # Trigger alert for startup failure
            alert_system.trigger_alert(
                alert_type=AlertType.SYSTEM_ERROR,
                level=AlertLevel.CRITICAL,
                message=f"Failed to start stream manager: {str(e)}",
                data={"component": "StreamManager", "error": str(e)}
            )
            await self.stop()
            raise

    async def stop(self) -> None:
        """Stop the market data streaming service."""
        try:
            self.running = False
            
            # Cancel all background tasks
            for task in self.tasks:
                task.cancel()
                
            await asyncio.gather(*self.tasks, return_exceptions=True)
            await self.ws_client.close()
            
            logger.info("Stream manager stopped successfully")
            
        except Exception as e:
            logger.error(f"Error stopping stream manager: {e}")
            # Trigger alert for shutdown failure
            alert_system.trigger_alert(
                alert_type=AlertType.SYSTEM_ERROR,
                level=AlertLevel.WARNING,
                message=f"Error during stream manager shutdown: {str(e)}",
                data={"component": "StreamManager", "error": str(e)}
            )
            raise

    async def _subscribe_to_channels(self) -> None:
        """Subscribe to all configured market data channels."""
        try:
            config = self.ws_client.config
            
            for channel in config['collection']['channels']:
                await self.ws_client.subscribe(channel, config['collection']['symbols'])
                logger.info(f"Subscribed to {channel} for symbols {config['collection']['symbols']}")
                
        except Exception as e:
            logger.error(f"Error subscribing to channels: {e}")
            # Trigger alert for subscription failure
            alert_system.trigger_alert(
                alert_type=AlertType.SYSTEM_ERROR,
                level=AlertLevel.WARNING,
                message=f"Failed to subscribe to market data channels: {str(e)}",
                data={"component": "StreamManager", "error": str(e)}
            )
            raise

    async def _process_messages(self) -> None:
        """Process incoming market data messages."""
        while self.running:
            try:
                # Get message from queue
                message = await self.ws_client.get_queue().get()
                
                # Validate message
                if not await self.validator.validate_message(message):
                    logger.warning(f"Invalid message received: {message}")
                    continue
                    
                # Normalize data
                normalized_data = await self.normalizer.normalize_data(message)
                
                # Store processed data
                await self.data_store.store_data(normalized_data)
                
                # Reset error count on successful processing
                self.error_count = 0
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                self.error_count += 1
                
                # Check if error threshold is exceeded
                if self.error_count >= self.max_errors:
                    alert_system.trigger_alert(
                        alert_type=AlertType.SYSTEM_ERROR,
                        level=AlertLevel.CRITICAL,
                        message=f"Excessive message processing errors: {self.error_count} in the last interval",
                        data={"component": "StreamManager", "error_count": self.error_count, "last_error": str(e)}
                    )
                continue

    async def _reset_error_count(self) -> None:
        """Periodically reset the error count to avoid false positives."""
        while self.running:
            try:
                await asyncio.sleep(self.error_reset_interval)
                if self.error_count > 0:
                    logger.info(f"Resetting error count from {self.error_count} to 0")
                    self.error_count = 0
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in error count reset task: {e}")

    async def _store_data(self) -> None:
        """Periodically store accumulated data in batches."""
        while self.running:
            try:
                await self.data_store.flush()
                await asyncio.sleep(1)  # Adjust based on requirements
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error storing data: {e}")
                continue

    async def get_latest_data(self, symbol: str, channel: str) -> Optional[Dict]:
        """Get the latest market data for a symbol and channel."""
        try:
            return await self.data_store.get_latest_data(symbol, channel)
        except Exception as e:
            logger.error(f"Error getting latest data: {e}")
            return None

    async def get_historical_data(
        self,
        symbol: str,
        channel: str,
        start_time: datetime,
        end_time: datetime
    ) -> List[Dict]:
        """Get historical market data for a symbol and channel."""
        try:
            return await self.data_store.get_historical_data(symbol, channel, start_time, end_time)
        except Exception as e:
            logger.error(f"Error getting historical data: {e}")
            return []

    def is_running(self) -> bool:
        """Check if the stream manager is running."""
        return self.running

    async def add_symbol(self, symbol: str) -> None:
        """Add a new symbol to all subscribed channels."""
        try:
            config = self.ws_client.config
            
            for channel in config['collection']['channels']:
                await self.ws_client.subscribe(channel, [symbol])
                logger.info(f"Added {symbol} to channel {channel}")
                
        except Exception as e:
            logger.error(f"Error adding symbol {symbol}: {e}")
            raise

    async def remove_symbol(self, symbol: str) -> None:
        """Remove a symbol from all subscribed channels."""
        try:
            config = self.ws_client.config
            
            for channel in config['collection']['channels']:
                await self.ws_client.unsubscribe(channel, [symbol])
                logger.info(f"Removed {symbol} from channel {channel}")
                
        except Exception as e:
            logger.error(f"Error removing symbol {symbol}: {e}")
            raise

    async def get_statistics(self) -> Dict:
        """Get statistics about the market data streaming service."""
        try:
            return {
                'running': self.running,
                'queue_size': self.ws_client.get_queue().qsize(),
                'connected': self.ws_client.connected,
                'reconnect_attempts': self.ws_client.reconnect_attempts,
                'subscriptions': list(self.ws_client.subscriptions),
                'last_ping': self.ws_client.last_ping.isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {}
