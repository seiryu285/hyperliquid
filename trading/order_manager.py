import logging
import asyncio
from typing import Dict, List, Optional
from datetime import datetime
from core.rest_client import HyperLiquidRestClient

logger = logging.getLogger(__name__)

class OrderManager:
    """注文管理システム"""
    
    def __init__(self):
        self.client = HyperLiquidRestClient()
        self.active_orders: Dict[str, Dict] = {}
        self.positions: Dict[str, Dict] = {}
        self.order_history: List[Dict] = []
        
    async def initialize(self):
        """
        初期化処理：現在のポジションと注文を取得
        """
        try:
            # 現在のポジションを取得
            positions = await self.client.get_positions()
            self.positions = {p['symbol']: p for p in positions.get('positions', [])}
            
            # アクティブな注文を取得
            orders = await self.client.get_order_status('')  # 全注文取得
            self.active_orders = {o['order_id']: o for o in orders.get('orders', [])}
            
            logger.info(f"Initialized with {len(self.positions)} positions and {len(self.active_orders)} active orders")
        except Exception as e:
            logger.error(f"Error initializing OrderManager: {e}")
            raise
            
    async def place_market_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        reduce_only: bool = False
    ) -> Dict:
        """
        成行注文の発注
        
        Args:
            symbol (str): 取引ペア
            side (str): 'buy' or 'sell'
            quantity (float): 取引量
            reduce_only (bool): ポジション縮小のみの注文かどうか
        """
        try:
            order_data = {
                'symbol': symbol,
                'side': side,
                'type': 'market',
                'quantity': quantity,
                'reduce_only': reduce_only
            }
            
            response = await self.client.place_order(order_data)
            order_id = response.get('order_id')
            
            if order_id:
                self.active_orders[order_id] = {
                    **order_data,
                    'order_id': order_id,
                    'status': 'new',
                    'timestamp': datetime.now().isoformat()
                }
                
            return response
        except Exception as e:
            logger.error(f"Error placing market order: {e}")
            raise
            
    async def place_limit_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        post_only: bool = False,
        reduce_only: bool = False
    ) -> Dict:
        """
        指値注文の発注
        
        Args:
            symbol (str): 取引ペア
            side (str): 'buy' or 'sell'
            quantity (float): 取引量
            price (float): 指値価格
            post_only (bool): ポストオンリー注文かどうか
            reduce_only (bool): ポジション縮小のみの注文かどうか
        """
        try:
            order_data = {
                'symbol': symbol,
                'side': side,
                'type': 'limit',
                'quantity': quantity,
                'price': price,
                'post_only': post_only,
                'reduce_only': reduce_only
            }
            
            response = await self.client.place_order(order_data)
            order_id = response.get('order_id')
            
            if order_id:
                self.active_orders[order_id] = {
                    **order_data,
                    'order_id': order_id,
                    'status': 'new',
                    'timestamp': datetime.now().isoformat()
                }
                
            return response
        except Exception as e:
            logger.error(f"Error placing limit order: {e}")
            raise
            
    async def cancel_order(self, order_id: str) -> Dict:
        """注文のキャンセル"""
        try:
            response = await self.client.cancel_order(order_id)
            
            if order_id in self.active_orders:
                self.active_orders[order_id]['status'] = 'cancelled'
                self.order_history.append(self.active_orders.pop(order_id))
                
            return response
        except Exception as e:
            logger.error(f"Error cancelling order: {e}")
            raise
            
    async def update_order_status(self, order_id: str):
        """注文ステータスの更新"""
        try:
            status = await self.client.get_order_status(order_id)
            
            if order_id in self.active_orders:
                if status.get('status') in ['filled', 'cancelled']:
                    self.order_history.append(self.active_orders.pop(order_id))
                else:
                    self.active_orders[order_id].update(status)
                    
            return status
        except Exception as e:
            logger.error(f"Error updating order status: {e}")
            raise
            
    async def get_position(self, symbol: str) -> Optional[Dict]:
        """特定の取引ペアのポジション情報を取得"""
        try:
            positions = await self.client.get_positions()
            
            for position in positions.get('positions', []):
                if position['symbol'] == symbol:
                    self.positions[symbol] = position
                    return position
                    
            return None
        except Exception as e:
            logger.error(f"Error getting position: {e}")
            raise
            
    def get_active_orders(self, symbol: Optional[str] = None) -> List[Dict]:
        """アクティブな注文のリストを取得"""
        if symbol:
            return [order for order in self.active_orders.values() if order['symbol'] == symbol]
        return list(self.active_orders.values())
