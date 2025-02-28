"""
Order types and related data structures for the order management system.

This module defines the various order types and related data structures
used in the order management system.
"""

from enum import Enum, auto
from typing import Dict, Any, List, Union, Optional
from datetime import datetime
from pydantic import BaseModel, Field


class OrderSide(str, Enum):
    """Order side (buy or sell)."""
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    """Order type."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderStatus(str, Enum):
    """Order status."""
    PENDING = "pending"          # Order is pending submission
    SUBMITTED = "submitted"      # Order has been submitted to the exchange
    PARTIAL = "partial"          # Order has been partially filled
    FILLED = "filled"            # Order has been completely filled
    CANCELLED = "cancelled"      # Order has been cancelled
    REJECTED = "rejected"        # Order has been rejected by the exchange
    EXPIRED = "expired"          # Order has expired


class OrderTimeInForce(str, Enum):
    """Time in force for orders."""
    GTC = "gtc"  # Good Till Cancelled
    IOC = "ioc"  # Immediate Or Cancel
    FOK = "fok"  # Fill Or Kill


class Order(BaseModel):
    """
    Base order model.
    
    This class represents an order in the system.
    """
    id: Optional[str] = None
    exchange_id: Optional[str] = None
    symbol: str
    side: OrderSide
    type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: OrderTimeInForce = OrderTimeInForce.GTC
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    average_fill_price: Optional[float] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def is_active(self) -> bool:
        """Check if the order is active."""
        return self.status in [OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIAL]
    
    def is_complete(self) -> bool:
        """Check if the order is complete."""
        return self.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED, OrderStatus.EXPIRED]
    
    def remaining_quantity(self) -> float:
        """Get the remaining quantity to be filled."""
        return self.quantity - self.filled_quantity


class OrderFill(BaseModel):
    """
    Order fill information.
    
    This class represents a fill (execution) of an order.
    """
    order_id: str
    exchange_fill_id: Optional[str] = None
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    fee: float = 0.0
    fee_currency: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Position(BaseModel):
    """
    Position information.
    
    This class represents a position in a particular symbol.
    """
    symbol: str
    quantity: float = 0.0
    entry_price: float = 0.0
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    def is_long(self) -> bool:
        """Check if the position is long."""
        return self.quantity > 0
    
    def is_short(self) -> bool:
        """Check if the position is short."""
        return self.quantity < 0
    
    def is_flat(self) -> bool:
        """Check if the position is flat (no position)."""
        return abs(self.quantity) < 1e-10
    
    def update_unrealized_pnl(self, current_price: float):
        """
        Update the unrealized PnL based on the current price.
        
        Args:
            current_price: Current price of the symbol
        """
        self.current_price = current_price
        if abs(self.quantity) < 1e-10:
            self.unrealized_pnl = 0.0
        else:
            # For long positions, PnL is positive if current price > entry price
            # For short positions, PnL is positive if current price < entry price
            self.unrealized_pnl = self.quantity * (current_price - self.entry_price)
        self.updated_at = datetime.utcnow()


class OrderRequest(BaseModel):
    """
    Order request model.
    
    This class represents a request to create an order.
    """
    symbol: str
    side: OrderSide
    type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: OrderTimeInForce = OrderTimeInForce.GTC
    client_order_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def to_order(self) -> Order:
        """Convert the request to an Order object."""
        return Order(
            id=self.client_order_id,
            symbol=self.symbol,
            side=self.side,
            type=self.type,
            quantity=self.quantity,
            price=self.price,
            stop_price=self.stop_price,
            time_in_force=self.time_in_force,
            metadata=self.metadata.copy()
        )


class OrderCancelRequest(BaseModel):
    """
    Order cancel request model.
    
    This class represents a request to cancel an order.
    """
    order_id: str
    symbol: Optional[str] = None
    exchange_id: Optional[str] = None
