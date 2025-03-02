import asyncio
import logging
from typing import Dict, Optional
from datetime import datetime
from ...core.rest_client import HyperLiquidRestClient
from ..order_manager import OrderManager

logger = logging.getLogger(__name__)

class SimpleMarketMaker:
    """
    シンプルなマーケットメイキング戦略
    - 指定された価格範囲で売買注文を出す
    - ポジションが一定量を超えたら反対側に注文を出してバランスを取る
    """
    
    def __init__(
        self,
        symbol: str,
        order_size: float,
        spread_percentage: float = 0.002,  # 0.2%
        max_position: float = 1.0,
        position_threshold: float = 0.8
    ):
        self.symbol = symbol
        self.order_size = order_size
        self.spread_percentage = spread_percentage
        self.max_position = max_position
        self.position_threshold = position_threshold
        
        self.client = HyperLiquidRestClient()
        self.order_manager = OrderManager()
        self.running = False
        self.last_price: Optional[float] = None
        
    async def initialize(self):
        """戦略の初期化"""
        await self.order_manager.initialize()
        prices = await self.client.get_market_prices()
        self.last_price = float(prices[self.symbol]['last'])
        logger.info(f"Initialized market maker for {self.symbol} at price {self.last_price}")
        
    async def update_market_data(self):
        """市場データの更新"""
        try:
            prices = await self.client.get_market_prices()
            self.last_price = float(prices[self.symbol]['last'])
        except Exception as e:
            logger.error(f"Error updating market data: {e}")
            
    async def manage_orders(self):
        """注文の管理"""
        try:
            # 現在のポジションを取得
            position = await self.order_manager.get_position(self.symbol)
            position_size = float(position['size']) if position else 0
            
            # 既存の注文をキャンセル
            for order in self.order_manager.get_active_orders(self.symbol):
                await self.order_manager.cancel_order(order['order_id'])
                
            if abs(position_size) >= self.max_position * self.position_threshold:
                # ポジションが大きすぎる場合、反対側に成行注文
                side = 'sell' if position_size > 0 else 'buy'
                reduce_size = abs(position_size) - (self.max_position * 0.5)
                if reduce_size > 0:
                    await self.order_manager.place_market_order(
                        self.symbol,
                        side,
                        reduce_size,
                        reduce_only=True
                    )
                    logger.info(f"Placed position reduction order: {side} {reduce_size}")
                    return
                    
            # 新しい指値注文を配置
            bid_price = self.last_price * (1 - self.spread_percentage/2)
            ask_price = self.last_price * (1 + self.spread_percentage/2)
            
            # 買い注文
            await self.order_manager.place_limit_order(
                self.symbol,
                'buy',
                self.order_size,
                bid_price,
                post_only=True
            )
            
            # 売り注文
            await self.order_manager.place_limit_order(
                self.symbol,
                'sell',
                self.order_size,
                ask_price,
                post_only=True
            )
            
            logger.info(f"Placed new orders at {bid_price}/{ask_price}")
            
        except Exception as e:
            logger.error(f"Error managing orders: {e}")
            
    async def run(self):
        """戦略の実行"""
        try:
            await self.initialize()
            self.running = True
            
            while self.running:
                await self.update_market_data()
                await self.manage_orders()
                await asyncio.sleep(5)  # 5秒ごとに更新
                
        except Exception as e:
            logger.error(f"Error running market maker: {e}")
            self.running = False
            raise
        finally:
            # 終了時に全ての注文をキャンセル
            for order in self.order_manager.get_active_orders(self.symbol):
                try:
                    await self.order_manager.cancel_order(order['order_id'])
                except Exception as e:
                    logger.error(f"Error cancelling order during shutdown: {e}")
                    
    def stop(self):
        """戦略の停止"""
        self.running = False
        logger.info("Market maker stopped")
