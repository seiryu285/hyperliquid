import asyncio
import json
import logging
import websockets
from datetime import datetime
from typing import Set, Dict, Any, Optional
from dataclasses import dataclass, asdict
from risk_management.risk_monitoring.risk_monitor import RiskMonitor
from market_data.market_data_service import MarketDataService
from position_tracking.position_tracker import PositionTracker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class WebSocketMessage:
    type: str
    payload: Dict[str, Any]
    timestamp: float

class WebSocketServer:
    def __init__(self, host: str = 'localhost', port: int = 8000):
        self.host = host
        self.port = port
        self.clients: Set[websockets.WebSocketServerProtocol] = set()
        self.risk_monitor = RiskMonitor()
        self.market_data_service = MarketDataService()
        self.position_tracker = PositionTracker()
        
        # Initialize data buffers
        self.market_data_buffer: Dict[str, Any] = {}
        self.position_data_buffer: Dict[str, Any] = {}
        self.risk_metrics_buffer: Dict[str, Any] = {}

    async def start(self):
        """Start the WebSocket server."""
        try:
            async with websockets.serve(self.handle_client, self.host, self.port):
                logger.info(f"WebSocket server started on ws://{self.host}:{self.port}")
                await asyncio.Future()  # run forever
        except Exception as e:
            logger.error(f"Failed to start WebSocket server: {e}")
            raise

    async def handle_client(self, websocket: websockets.WebSocketServerProtocol, path: str):
        """Handle individual client connections."""
        try:
            # Register client
            self.clients.add(websocket)
            client_info = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
            logger.info(f"New client connected: {client_info}")

            # Send initial data
            await self.send_initial_data(websocket)

            # Start client message handling
            await self.handle_client_messages(websocket)

        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client disconnected normally: {client_info}")
        except Exception as e:
            logger.error(f"Error handling client {client_info}: {e}")
        finally:
            # Cleanup on disconnect
            self.clients.remove(websocket)
            logger.info(f"Client removed: {client_info}")

    async def send_initial_data(self, websocket: websockets.WebSocketServerProtocol):
        """Send initial data to newly connected client."""
        try:
            # Send cached data if available
            if self.market_data_buffer:
                await self.send_message(websocket, "MARKET_DATA", self.market_data_buffer)
            if self.position_data_buffer:
                await self.send_message(websocket, "POSITION_DATA", self.position_data_buffer)
            if self.risk_metrics_buffer:
                await self.send_message(websocket, "RISK_METRICS", self.risk_metrics_buffer)
        except Exception as e:
            logger.error(f"Error sending initial data: {e}")
            await self.send_error(websocket, str(e))

    async def handle_client_messages(self, websocket: websockets.WebSocketServerProtocol):
        """Handle incoming messages from a client."""
        async for message in websocket:
            try:
                data = json.loads(message)
                message_type = data.get('type', '')

                if message_type == 'PING':
                    await self.send_message(websocket, 'PONG', {'timestamp': datetime.now().timestamp()})
                elif message_type == 'SUBSCRIBE':
                    await self.handle_subscription(websocket, data)
                else:
                    logger.warning(f"Unknown message type: {message_type}")

            except json.JSONDecodeError:
                logger.error("Invalid JSON received")
                await self.send_error(websocket, "Invalid JSON format")
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                await self.send_error(websocket, str(e))

    async def handle_subscription(self, websocket: websockets.WebSocketServerProtocol, data: Dict):
        """Handle client subscription requests."""
        try:
            channels = data.get('channels', [])
            for channel in channels:
                if channel == 'market_data':
                    await self.send_message(websocket, "MARKET_DATA", self.market_data_buffer)
                elif channel == 'position_data':
                    await self.send_message(websocket, "POSITION_DATA", self.position_data_buffer)
                elif channel == 'risk_metrics':
                    await self.send_message(websocket, "RISK_METRICS", self.risk_metrics_buffer)
        except Exception as e:
            logger.error(f"Error handling subscription: {e}")
            await self.send_error(websocket, str(e))

    async def broadcast(self, message_type: str, payload: Dict):
        """Broadcast message to all connected clients."""
        if not self.clients:
            return

        message = self._create_message(message_type, payload)
        
        # Update appropriate buffer
        if message_type == "MARKET_DATA":
            self.market_data_buffer = payload
        elif message_type == "POSITION_DATA":
            self.position_data_buffer = payload
        elif message_type == "RISK_METRICS":
            self.risk_metrics_buffer = payload

        # Broadcast to all clients
        disconnected_clients = set()
        for client in self.clients:
            try:
                await client.send(json.dumps(asdict(message)))
            except websockets.exceptions.ConnectionClosed:
                disconnected_clients.add(client)
            except Exception as e:
                logger.error(f"Error broadcasting to client: {e}")
                disconnected_clients.add(client)

        # Clean up disconnected clients
        for client in disconnected_clients:
            self.clients.remove(client)

    async def send_message(self, websocket: websockets.WebSocketServerProtocol, 
                         message_type: str, payload: Dict):
        """Send a message to a specific client."""
        try:
            message = self._create_message(message_type, payload)
            await websocket.send(json.dumps(asdict(message)))
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            await self.send_error(websocket, str(e))

    async def send_error(self, websocket: websockets.WebSocketServerProtocol, error_message: str):
        """Send an error message to a client."""
        try:
            message = self._create_message("ERROR", {"message": error_message})
            await websocket.send(json.dumps(asdict(message)))
        except Exception as e:
            logger.error(f"Error sending error message: {e}")

    def _create_message(self, message_type: str, payload: Dict) -> WebSocketMessage:
        """Create a WebSocketMessage instance."""
        return WebSocketMessage(
            type=message_type,
            payload=payload,
            timestamp=datetime.now().timestamp()
        )

    async def update_market_data(self):
        """Update and broadcast market data."""
        try:
            market_data = await self.market_data_service.get_latest_data()
            await self.broadcast("MARKET_DATA", market_data)
        except Exception as e:
            logger.error(f"Error updating market data: {e}")

    async def update_position_data(self):
        """Update and broadcast position data."""
        try:
            position_data = await self.position_tracker.get_current_position()
            await self.broadcast("POSITION_DATA", position_data)
        except Exception as e:
            logger.error(f"Error updating position data: {e}")

    async def update_risk_metrics(self):
        """Update and broadcast risk metrics."""
        try:
            risk_metrics = self.risk_monitor.calculate_metrics(
                self.market_data_buffer,
                self.position_data_buffer
            )
            await self.broadcast("RISK_METRICS", risk_metrics)
        except Exception as e:
            logger.error(f"Error updating risk metrics: {e}")

    async def start_data_updates(self):
        """Start periodic data updates."""
        while True:
            try:
                await asyncio.gather(
                    self.update_market_data(),
                    self.update_position_data(),
                    self.update_risk_metrics()
                )
                await asyncio.sleep(1)  # Update every second
            except Exception as e:
                logger.error(f"Error in data update loop: {e}")
                await asyncio.sleep(5)  # Wait before retrying

if __name__ == "__main__":
    server = WebSocketServer()
    loop = asyncio.get_event_loop()
    
    try:
        loop.create_task(server.start_data_updates())
        loop.run_until_complete(server.start())
    except KeyboardInterrupt:
        logger.info("Server shutting down...")
    except Exception as e:
        logger.error(f"Server error: {e}")
    finally:
        loop.close()
