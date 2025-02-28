from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Dict, Any
import asyncio
import uvicorn
import os
import sys
import logging
from datetime import datetime, timedelta
import pandas as pd
from pydantic import BaseModel

# プロジェクトルートをPythonパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 自作モジュールのインポート
from market_data.data_collector import HyperLiquidDataCollector
from market_data.data_processor import DataProcessor
from order_management.hyperliquid_execution import HyperLiquidExecution
from core.config import Config

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'api.log'))
    ]
)
logger = logging.getLogger(__name__)

# FastAPIアプリケーションの作成
app = FastAPI(title="HyperLiquid Trading API", version="1.0.0")

# CORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 本番環境では特定のオリジンのみを許可するように変更する
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 設定の読み込み
config = Config()

# データコレクターとデータプロセッサーの初期化
data_collector = HyperLiquidDataCollector(config)
data_processor = DataProcessor(config)

# 注文実行エンジンの初期化
execution_engine = HyperLiquidExecution(config)

# モデルの定義
class OrderRequest(BaseModel):
    symbol: str
    side: str  # "buy" or "sell"
    type: str  # "market" or "limit"
    size: float
    price: Optional[float] = None
    reduce_only: Optional[bool] = False
    post_only: Optional[bool] = False
    time_in_force: Optional[str] = "GTC"  # GTC, IOC, FOK

class OrderResponse(BaseModel):
    id: str
    symbol: str
    side: str
    type: str
    size: float
    price: Optional[float]
    status: str
    created_at: str

# ヘルスチェックエンドポイント
@app.get("/api/health")
async def health_check():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}

# 市場データエンドポイント
@app.get("/api/market-data/{symbol}")
async def get_market_data(symbol: str):
    try:
        # オーダーブックの取得
        orderbook = await data_collector.get_orderbook(symbol)
        
        # 最近の取引の取得
        trades = await data_collector.get_recent_trades(symbol, limit=50)
        
        # ポジション情報の取得
        positions = await execution_engine.get_positions()
        
        # ティッカー情報の取得
        ticker = await data_collector.get_ticker(symbol)
        
        return {
            "symbol": symbol,
            "orderBook": orderbook,
            "trades": trades,
            "positions": positions,
            "ticker": ticker,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error fetching market data for {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# OHLCVデータエンドポイント
@app.get("/api/ohlcv/{symbol}")
async def get_ohlcv_data(
    symbol: str, 
    timeframe: str = Query("1h", description="Timeframe (1m, 5m, 15m, 1h, 4h, 1d)"),
    limit: int = Query(100, description="Number of candles to return")
):
    try:
        # OHLCVデータの取得
        ohlcv_data = await data_collector.get_ohlcv(symbol, timeframe, limit)
        
        # テクニカル指標の追加
        if ohlcv_data is not None and len(ohlcv_data) > 0:
            df = pd.DataFrame(ohlcv_data)
            df = data_processor.add_technical_indicators(df)
            return df.to_dict('records')
        
        return []
    except Exception as e:
        logger.error(f"Error fetching OHLCV data for {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ティッカーエンドポイント
@app.get("/api/ticker/{symbol}")
async def get_ticker(symbol: str):
    try:
        ticker = await data_collector.get_ticker(symbol)
        return ticker
    except Exception as e:
        logger.error(f"Error fetching ticker for {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# アカウント情報エンドポイント
@app.get("/api/account")
async def get_account_info():
    try:
        account_info = await execution_engine.get_account_info()
        return account_info
    except Exception as e:
        logger.error(f"Error fetching account info: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# 注文作成エンドポイント
@app.post("/api/orders", response_model=OrderResponse)
async def create_order(order_request: OrderRequest):
    try:
        order = await execution_engine.place_order(
            symbol=order_request.symbol,
            side=order_request.side,
            order_type=order_request.type,
            size=order_request.size,
            price=order_request.price,
            reduce_only=order_request.reduce_only,
            post_only=order_request.post_only,
            time_in_force=order_request.time_in_force
        )
        return order
    except Exception as e:
        logger.error(f"Error creating order: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# 注文キャンセルエンドポイント
@app.delete("/api/orders/{order_id}")
async def cancel_order(order_id: str):
    try:
        result = await execution_engine.cancel_order(order_id)
        return result
    except Exception as e:
        logger.error(f"Error canceling order {order_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# アクティブな注文の取得エンドポイント
@app.get("/api/orders/active")
async def get_active_orders(symbol: Optional[str] = None):
    try:
        orders = await execution_engine.get_open_orders(symbol)
        return orders
    except Exception as e:
        logger.error(f"Error fetching active orders: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# 注文履歴の取得エンドポイント
@app.get("/api/orders/history")
async def get_order_history(
    symbol: Optional[str] = None,
    limit: int = Query(50, description="Number of orders to return")
):
    try:
        history = await execution_engine.get_order_history(symbol, limit)
        return history
    except Exception as e:
        logger.error(f"Error fetching order history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# トレード履歴の取得エンドポイント
@app.get("/api/trades/history")
async def get_trade_history(
    symbol: Optional[str] = None,
    limit: int = Query(50, description="Number of trades to return")
):
    try:
        trades = await execution_engine.get_trade_history(symbol, limit)
        return trades
    except Exception as e:
        logger.error(f"Error fetching trade history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# WebSocket接続の初期化
@app.on_event("startup")
async def startup_event():
    try:
        # WebSocket接続の開始
        asyncio.create_task(data_collector.start_websocket())
        logger.info("WebSocket connection started")
    except Exception as e:
        logger.error(f"Error starting WebSocket connection: {str(e)}")

# アプリケーションのシャットダウン
@app.on_event("shutdown")
async def shutdown_event():
    try:
        # WebSocket接続の終了
        await data_collector.stop_websocket()
        logger.info("WebSocket connection stopped")
    except Exception as e:
        logger.error(f"Error stopping WebSocket connection: {str(e)}")

# メイン関数
if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=5000, reload=True)
