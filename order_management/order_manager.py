"""
Order manager for handling order lifecycle.

This module provides the base OrderManager class for managing orders
throughout their lifecycle.
"""

import logging
import asyncio
import uuid
import time
import random
from typing import Dict, List, Optional, Any, Callable, Coroutine, Union, Tuple
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

from .order_types import (
    Order, OrderRequest, OrderCancelRequest, OrderFill, 
    OrderStatus, OrderSide, OrderType, Position
)
from risk_management.risk_monitoring.alert_system import alert_system, AlertType, AlertLevel

# Configure logging
logger = logging.getLogger(__name__)


class OrderManager:
    """
    Base class for managing orders.
    
    This class provides the basic functionality for managing orders,
    including order creation, cancellation, and tracking.
    """
    
    def __init__(self, max_workers: int = 10, retry_attempts: int = 3, retry_delay: float = 1.0,
                 max_retry_attempts: int = 5, max_retry_delay: float = 30.0):
        """
        Initialize the order manager.
        
        Args:
            max_workers: Maximum number of worker threads for parallel processing
            retry_attempts: Number of retry attempts for failed operations
            retry_delay: Delay between retry attempts (seconds)
            max_retry_attempts: Maximum number of retry attempts for critical operations
            max_retry_delay: Maximum delay between retry attempts (seconds)
        """
        self.orders: Dict[str, Order] = {}
        self.positions: Dict[str, Position] = {}
        self.order_fills: Dict[str, List[OrderFill]] = {}
        
        # Callbacks
        self.on_order_update_callbacks: List[Callable[[Order], None]] = []
        self.on_fill_callbacks: List[Callable[[OrderFill], None]] = []
        self.on_position_update_callbacks: List[Callable[[Position], None]] = []
        
        # Concurrency and retry settings
        self.max_workers = max_workers
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        self.max_retry_attempts = max_retry_attempts
        self.max_retry_delay = max_retry_delay
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Locks for thread safety
        self.order_lock = asyncio.Lock()
        self.position_lock = asyncio.Lock()
        
        # Error tracking
        self.error_counts = {}
        self.error_threshold = 5
        
        logger.info(f"Initialized order manager with max_workers={max_workers}, retry_attempts={retry_attempts}")
    
    def add_order(self, order: Order) -> Order:
        """
        Add an order to the manager.
        
        Args:
            order: Order to add
            
        Returns:
            The added order
        """
        # Generate an ID if not provided
        if not order.id:
            order.id = str(uuid.uuid4())
        
        # Store the order
        self.orders[order.id] = order
        
        # Initialize order fills list
        self.order_fills[order.id] = []
        
        logger.info(f"Added order {order.id}: {order.side.value} {order.quantity} {order.symbol} @ {order.price}")
        
        # Notify callbacks
        self._notify_order_update(order)
        
        return order
    
    def get_order(self, order_id: str) -> Optional[Order]:
        """
        Get an order by ID.
        
        Args:
            order_id: Order ID
            
        Returns:
            Order if found, None otherwise
        """
        return self.orders.get(order_id)
    
    def get_orders(self, 
                  symbol: Optional[str] = None, 
                  status: Optional[OrderStatus] = None,
                  side: Optional[OrderSide] = None) -> List[Order]:
        """
        Get orders filtered by criteria.
        
        Args:
            symbol: Filter by symbol
            status: Filter by status
            side: Filter by side
            
        Returns:
            List of matching orders
        """
        result = list(self.orders.values())
        
        if symbol:
            result = [o for o in result if o.symbol == symbol]
        
        if status:
            result = [o for o in result if o.status == status]
        
        if side:
            result = [o for o in result if o.side == side]
        
        return result
    
    def get_active_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """
        Get active orders.
        
        Args:
            symbol: Filter by symbol
            
        Returns:
            List of active orders
        """
        active_statuses = [OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIAL]
        result = [o for o in self.orders.values() if o.status in active_statuses]
        
        if symbol:
            result = [o for o in result if o.symbol == symbol]
        
        return result
    
    def update_order(self, order_id: str, updates: Dict[str, Any]) -> Optional[Order]:
        """
        Update an order.
        
        Args:
            order_id: Order ID
            updates: Dictionary of updates to apply
            
        Returns:
            Updated order if found, None otherwise
        """
        order = self.get_order(order_id)
        if not order:
            logger.warning(f"Cannot update order {order_id}: not found")
            return None
        
        # Apply updates
        for key, value in updates.items():
            if hasattr(order, key):
                setattr(order, key, value)
        
        # Update timestamp
        order.updated_at = datetime.utcnow()
        
        logger.info(f"Updated order {order_id}: {updates}")
        
        # Notify callbacks
        self._notify_order_update(order)
        
        return order
    
    def add_fill(self, fill: OrderFill) -> Optional[Order]:
        """
        Add a fill to an order.
        
        Args:
            fill: Order fill
            
        Returns:
            Updated order if found, None otherwise
        """
        order = self.get_order(fill.order_id)
        if not order:
            logger.warning(f"Cannot add fill to order {fill.order_id}: not found")
            return None
        
        # Add fill to list
        self.order_fills[order.id].append(fill)
        
        # Update order
        new_filled_quantity = order.filled_quantity + fill.quantity
        
        # Calculate new average fill price
        if order.average_fill_price is None:
            new_avg_price = fill.price
        else:
            new_avg_price = (
                (order.average_fill_price * order.filled_quantity) + (fill.price * fill.quantity)
            ) / new_filled_quantity
        
        # Update order
        updates = {
            'filled_quantity': new_filled_quantity,
            'average_fill_price': new_avg_price,
            'updated_at': datetime.utcnow()
        }
        
        # Update status if fully filled
        if abs(new_filled_quantity - order.quantity) < 1e-10:
            updates['status'] = OrderStatus.FILLED
        elif new_filled_quantity > 0:
            updates['status'] = OrderStatus.PARTIAL
        
        # Apply updates
        for key, value in updates.items():
            setattr(order, key, value)
        
        logger.info(f"Added fill to order {order.id}: {fill.quantity} @ {fill.price}")
        
        # Update position
        self._update_position_from_fill(fill)
        
        # Notify callbacks
        self._notify_order_update(order)
        self._notify_fill(fill)
        
        return order
    
    def cancel_order(self, order_id: str) -> Optional[Order]:
        """
        Cancel an order.
        
        Args:
            order_id: Order ID
            
        Returns:
            Cancelled order if found and cancellable, None otherwise
        """
        order = self.get_order(order_id)
        if not order:
            logger.warning(f"Cannot cancel order {order_id}: not found")
            return None
        
        # Check if order can be cancelled
        if not order.is_active():
            logger.warning(f"Cannot cancel order {order_id}: not active (status: {order.status.value})")
            return None
        
        # Update order
        order.status = OrderStatus.CANCELLED
        order.updated_at = datetime.utcnow()
        
        logger.info(f"Cancelled order {order_id}")
        
        # Notify callbacks
        self._notify_order_update(order)
        
        return order
    
    def get_position(self, symbol: str) -> Position:
        """
        Get position for a symbol.
        
        Args:
            symbol: Symbol
            
        Returns:
            Position for the symbol (creates a new one if not found)
        """
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol=symbol)
        
        return self.positions[symbol]
    
    def get_all_positions(self) -> List[Position]:
        """
        Get all positions.
        
        Returns:
            List of all positions
        """
        return list(self.positions.values())
    
    def update_position(self, symbol: str, updates: Dict[str, Any]) -> Position:
        """
        Update a position.
        
        Args:
            symbol: Symbol
            updates: Dictionary of updates to apply
            
        Returns:
            Updated position
        """
        position = self.get_position(symbol)
        
        # Apply updates
        for key, value in updates.items():
            if hasattr(position, key):
                setattr(position, key, value)
        
        # Update timestamp
        position.updated_at = datetime.utcnow()
        
        logger.info(f"Updated position for {symbol}: {updates}")
        
        # Notify callbacks
        self._notify_position_update(position)
        
        return position
    
    def update_position_price(self, symbol: str, price: float) -> Position:
        """
        Update position with current price.
        
        Args:
            symbol: Symbol
            price: Current price
            
        Returns:
            Updated position
        """
        position = self.get_position(symbol)
        position.update_unrealized_pnl(price)
        
        # Notify callbacks
        self._notify_position_update(position)
        
        return position
    
    def register_order_update_callback(self, callback: Callable[[Order], None]):
        """
        Register a callback for order updates.
        
        Args:
            callback: Callback function
        """
        self.on_order_update_callbacks.append(callback)
    
    def register_fill_callback(self, callback: Callable[[OrderFill], None]):
        """
        Register a callback for order fills.
        
        Args:
            callback: Callback function
        """
        self.on_fill_callbacks.append(callback)
    
    def register_position_update_callback(self, callback: Callable[[Position], None]):
        """
        Register a callback for position updates.
        
        Args:
            callback: Callback function
        """
        self.on_position_update_callbacks.append(callback)
    
    def _notify_order_update(self, order: Order):
        """
        Notify order update callbacks.
        
        Args:
            order: Updated order
        """
        for callback in self.on_order_update_callbacks:
            try:
                callback(order)
            except Exception as e:
                logger.error(f"Error in order update callback: {e}")
    
    def _notify_fill(self, fill: OrderFill):
        """
        Notify fill callbacks.
        
        Args:
            fill: Order fill
        """
        for callback in self.on_fill_callbacks:
            try:
                callback(fill)
            except Exception as e:
                logger.error(f"Error in fill callback: {e}")
    
    def _notify_position_update(self, position: Position):
        """
        Notify position update callbacks.
        
        Args:
            position: Updated position
        """
        for callback in self.on_position_update_callbacks:
            try:
                callback(position)
            except Exception as e:
                logger.error(f"Error in position update callback: {e}")
    
    def _update_position_from_fill(self, fill: OrderFill):
        """
        Update position based on a fill.
        
        Args:
            fill: Order fill
        """
        position = self.get_position(fill.symbol)
        
        # Calculate quantity change (positive for buy, negative for sell)
        quantity_change = fill.quantity if fill.side == OrderSide.BUY else -fill.quantity
        
        # Calculate new position quantity
        new_quantity = position.quantity + quantity_change
        
        # Calculate new entry price (weighted average)
        if abs(position.quantity) < 1e-10:
            # No existing position, entry price is fill price
            new_entry_price = fill.price
        elif (position.quantity > 0 and quantity_change > 0) or (position.quantity < 0 and quantity_change < 0):
            # Adding to existing position (long + buy or short + sell)
            new_entry_price = (
                (position.entry_price * abs(position.quantity)) + (fill.price * abs(quantity_change))
            ) / abs(new_quantity)
        elif abs(new_quantity) < 1e-10:
            # Position closed, entry price doesn't matter
            new_entry_price = 0.0
        elif (position.quantity > 0 and quantity_change < 0) or (position.quantity < 0 and quantity_change > 0):
            # Reducing position (long + sell or short + buy)
            # Calculate realized PnL
            realized_pnl = abs(quantity_change) * (fill.price - position.entry_price)
            if position.quantity < 0:
                realized_pnl = -realized_pnl
            
            # Update realized PnL
            position.realized_pnl += realized_pnl
            
            # Keep the same entry price if reducing
            new_entry_price = position.entry_price
            
            # If position flipped, new entry price is fill price
            if (position.quantity > 0 and new_quantity < 0) or (position.quantity < 0 and new_quantity > 0):
                new_entry_price = fill.price
        else:
            # Shouldn't happen, but just in case
            new_entry_price = fill.price
        
        # Update position
        updates = {
            'quantity': new_quantity,
            'entry_price': new_entry_price,
            'current_price': fill.price
        }
        
        # Apply updates
        self.update_position(fill.symbol, updates)
        
        # Update unrealized PnL
        position.update_unrealized_pnl(fill.price)

    async def add_order_async(self, order: Order) -> Order:
        """
        Add an order to the manager asynchronously.
        
        Args:
            order: Order to add
            
        Returns:
            The added order
        """
        async with self.order_lock:
            # Generate an ID if not provided
            if not order.id:
                order.id = str(uuid.uuid4())
            
            # Store the order
            self.orders[order.id] = order
            
            # Initialize order fills list
            self.order_fills[order.id] = []
            
            logger.info(f"Added order {order.id}: {order.side.value} {order.quantity} {order.symbol} @ {order.price}")
            
            # Notify callbacks
            await self._notify_order_update_async(order)
            
            return order

    async def update_order_async(self, order_id: str, updates: Dict[str, Any]) -> Optional[Order]:
        """
        Update an order asynchronously.
        
        Args:
            order_id: Order ID
            updates: Dictionary of updates to apply
            
        Returns:
            Updated order if found, None otherwise
        """
        async with self.order_lock:
            order = self.get_order(order_id)
            if not order:
                logger.warning(f"Cannot update order {order_id}: not found")
                return None
            
            # Apply updates
            for key, value in updates.items():
                if hasattr(order, key):
                    setattr(order, key, value)
            
            # Update timestamp
            order.updated_at = datetime.utcnow()
            
            logger.info(f"Updated order {order_id}: {updates}")
            
            # Notify callbacks
            await self._notify_order_update_async(order)
            
            return order

    async def add_fill_async(self, fill: OrderFill) -> Optional[Order]:
        """
        Add a fill to an order asynchronously.
        
        Args:
            fill: Order fill
            
        Returns:
            Updated order if found, None otherwise
        """
        async with self.order_lock:
            order = self.get_order(fill.order_id)
            if not order:
                logger.warning(f"Cannot add fill to order {fill.order_id}: not found")
                return None
            
            # Add fill to list
            self.order_fills[order.id].append(fill)
            
            # Update order
            new_filled_quantity = order.filled_quantity + fill.quantity
            
            # Calculate new average fill price
            if order.average_fill_price is None:
                new_avg_price = fill.price
            else:
                new_avg_price = (
                    (order.average_fill_price * order.filled_quantity) + (fill.price * fill.quantity)
                ) / new_filled_quantity
            
            # Update order
            updates = {
                'filled_quantity': new_filled_quantity,
                'average_fill_price': new_avg_price,
                'updated_at': datetime.utcnow()
            }
            
            # Update status if fully filled
            if abs(new_filled_quantity - order.quantity) < 1e-10:
                updates['status'] = OrderStatus.FILLED
            elif new_filled_quantity > 0:
                updates['status'] = OrderStatus.PARTIAL
            
            # Apply updates
            for key, value in updates.items():
                setattr(order, key, value)
            
            logger.info(f"Added fill to order {order.id}: {fill.quantity} @ {fill.price}")
            
            # Notify callbacks
            await self._notify_fill_async(fill)
            await self._notify_order_update_async(order)
            
            # Update position
            await self._update_position_from_fill_async(fill)
            
            return order

    async def cancel_order_async(self, order_id: str) -> Optional[Order]:
        """
        Cancel an order asynchronously.
        
        Args:
            order_id: Order ID
            
        Returns:
            Cancelled order if found and cancellable, None otherwise
        """
        async with self.order_lock:
            order = self.get_order(order_id)
            if not order:
                logger.warning(f"Cannot cancel order {order_id}: not found")
                return None
            
            # Check if order can be cancelled
            if order.status not in [OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIAL]:
                logger.warning(f"Cannot cancel order {order_id}: status is {order.status.value}")
                return None
            
            # Update order
            order.status = OrderStatus.CANCELLED
            order.updated_at = datetime.utcnow()
            
            logger.info(f"Cancelled order {order_id}")
            
            # Notify callbacks
            await self._notify_order_update_async(order)
            
            return order

    async def update_position_async(self, symbol: str, updates: Dict[str, Any]) -> Position:
        """
        Update a position asynchronously.
        
        Args:
            symbol: Symbol
            updates: Dictionary of updates to apply
            
        Returns:
            Updated position
        """
        async with self.position_lock:
            position = self.get_position(symbol)
            
            # Apply updates
            for key, value in updates.items():
                if hasattr(position, key):
                    setattr(position, key, value)
            
            # Update timestamp
            position.updated_at = datetime.utcnow()
            
            logger.info(f"Updated position for {symbol}: {updates}")
            
            # Notify callbacks
            await self._notify_position_update_async(position)
            
            return position

    async def update_position_price_async(self, symbol: str, price: float) -> Position:
        """
        Update position with current price asynchronously.
        
        Args:
            symbol: Symbol
            price: Current price
            
        Returns:
            Updated position
        """
        async with self.position_lock:
            position = self.get_position(symbol)
            
            # Update price
            position.current_price = price
            
            # Calculate unrealized PnL
            if position.quantity != 0 and position.entry_price is not None:
                if position.quantity > 0:
                    # Long position
                    position.unrealized_pnl = (price - position.entry_price) * position.quantity
                else:
                    # Short position
                    position.unrealized_pnl = (position.entry_price - price) * abs(position.quantity)
            else:
                position.unrealized_pnl = 0.0
            
            # Update timestamp
            position.updated_at = datetime.utcnow()
            
            # Notify callbacks
            await self._notify_position_update_async(position)
            
            return position

    async def retry_async(self, func, *args, critical: bool = False, **kwargs) -> Any:
        """
        Retry an asynchronous function with exponential backoff.
        
        Args:
            func: Function to retry
            *args: Arguments to pass to the function
            critical: Whether this is a critical operation that should use max retry settings
            **kwargs: Keyword arguments to pass to the function
            
        Returns:
            Result of the function
        """
        attempt = 0
        last_error = None
        
        # Determine retry settings based on criticality
        retry_attempts = self.max_retry_attempts if critical else self.retry_attempts
        base_delay = self.retry_delay
        max_delay = self.max_retry_delay
        
        # Track function name for error reporting
        func_name = getattr(func, "__name__", str(func))
        
        while attempt < retry_attempts:
            try:
                result = await func(*args, **kwargs)
                
                # Reset error count on success
                if func_name in self.error_counts:
                    self.error_counts[func_name] = 0
                
                return result
                
            except Exception as e:
                attempt += 1
                last_error = e
                
                # Track error count
                self.error_counts[func_name] = self.error_counts.get(func_name, 0) + 1
                
                # Log error with increasing severity
                if attempt == 1:
                    logger.warning(f"Error in {func_name}: {e}, will retry")
                elif attempt < retry_attempts:
                    logger.error(f"Retry attempt {attempt}/{retry_attempts} for {func_name} failed: {e}")
                else:
                    logger.critical(f"All {retry_attempts} retry attempts for {func_name} failed: {e}")
                    
                    # Trigger alert for critical failures
                    if self.error_counts.get(func_name, 0) >= self.error_threshold:
                        alert_system.trigger_alert(
                            alert_type=AlertType.SYSTEM_ERROR,
                            level=AlertLevel.CRITICAL,
                            message=f"Operation {func_name} failed after {retry_attempts} attempts",
                            data={"error": str(e), "attempts": attempt}
                        )
                
                if attempt < retry_attempts:
                    # Calculate backoff delay with jitter (exponential backoff with full jitter)
                    delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
                    actual_delay = random.uniform(0, delay)
                    logger.warning(f"Retrying {func_name} in {actual_delay:.2f}s (attempt {attempt+1}/{retry_attempts})")
                    await asyncio.sleep(actual_delay)
        
        # Re-raise the last error
        raise last_error

    async def _notify_order_update_async(self, order: Order):
        """
        Notify order update callbacks asynchronously.
        
        Args:
            order: Updated order
        """
        for callback in self.on_order_update_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(order)
                else:
                    await asyncio.to_thread(callback, order)
            except Exception as e:
                logger.error(f"Error in order update callback: {e}")

    async def _notify_fill_async(self, fill: OrderFill):
        """
        Notify fill callbacks asynchronously.
        
        Args:
            fill: Order fill
        """
        for callback in self.on_fill_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(fill)
                else:
                    await asyncio.to_thread(callback, fill)
            except Exception as e:
                logger.error(f"Error in fill callback: {e}")

    async def _notify_position_update_async(self, position: Position):
        """
        Notify position update callbacks asynchronously.
        
        Args:
            position: Updated position
        """
        for callback in self.on_position_update_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(position)
                else:
                    await asyncio.to_thread(callback, position)
            except Exception as e:
                logger.error(f"Error in position update callback: {e}")

    async def _update_position_from_fill_async(self, fill: OrderFill):
        """
        Update position based on a fill asynchronously.
        
        Args:
            fill: Order fill
        """
        async with self.position_lock:
            position = self.get_position(fill.symbol)
            
            # Calculate new quantity
            new_quantity = position.quantity
            if fill.side == OrderSide.BUY:
                new_quantity += fill.quantity
            else:
                new_quantity -= fill.quantity
            
            # Calculate new entry price
            if position.quantity == 0:
                # New position
                new_entry_price = fill.price
            elif (position.quantity > 0 and fill.side == OrderSide.BUY) or (position.quantity < 0 and fill.side == OrderSide.SELL):
                # Adding to existing position
                new_entry_price = (
                    (position.entry_price * abs(position.quantity)) + (fill.price * fill.quantity)
                ) / (abs(position.quantity) + fill.quantity)
            elif abs(new_quantity) < 1e-10:
                # Position closed
                new_entry_price = None
            else:
                # Reducing position
                new_entry_price = position.entry_price
            
            # Calculate realized PnL
            realized_pnl = 0.0
            if (position.quantity > 0 and fill.side == OrderSide.SELL) or (position.quantity < 0 and fill.side == OrderSide.BUY):
                # Closing or reducing position
                if position.entry_price is not None:
                    if position.quantity > 0:
                        # Long position
                        realized_pnl = (fill.price - position.entry_price) * min(abs(position.quantity), fill.quantity)
                    else:
                        # Short position
                        realized_pnl = (position.entry_price - fill.price) * min(abs(position.quantity), fill.quantity)
            
            # Update position
            updates = {
                'quantity': new_quantity,
                'entry_price': new_entry_price,
                'realized_pnl': position.realized_pnl + realized_pnl,
                'updated_at': datetime.utcnow()
            }
            
            # Apply updates
            for key, value in updates.items():
                setattr(position, key, value)
            
            # Notify callbacks
            await self._notify_position_update_async(position)
