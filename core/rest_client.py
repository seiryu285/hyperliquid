import json
import logging
import aiohttp
from typing import Dict, Optional, Any
from .auth import HyperLiquidAuth

logger = logging.getLogger(__name__)

class HyperLiquidRestClient:
    """HyperLiquid REST APIクライアント"""
    
    def __init__(self, base_url: str = "https://api.hyperliquid.xyz"):
        self.base_url = base_url
        self.auth = HyperLiquidAuth()
        
    async def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict] = None,
        data: Optional[Dict] = None
    ) -> Any:
        """
        APIリクエストの実行
        
        Args:
            method (str): HTTPメソッド
            endpoint (str): APIエンドポイント
            params (Optional[Dict]): クエリパラメータ
            data (Optional[Dict]): リクエストボディ
            
        Returns:
            Any: APIレスポンス
        """
        url = f"{self.base_url}{endpoint}"
        try:
            headers = self.auth.get_auth_headers(method, endpoint, data)
            headers['Content-Type'] = 'application/json'
        except Exception as e:
            logger.warning(f"Authentication error: {e}. Proceeding without authentication.")
            headers = {'Content-Type': 'application/json'}
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.request(
                    method=method,
                    url=url,
                    headers=headers,
                    params=params,
                    json=data
                ) as response:
                    response_data = await response.text()
                    
                    if response.status >= 400:
                        logger.error(f"API error: {response.status} - {response_data}")
                        return {"error": f"API error: {response.status}", "message": response_data}
                        
                    return json.loads(response_data)
        except aiohttp.ClientError as e:
            logger.error(f"HTTP request error: {e}")
            return {"error": "Connection error", "message": str(e)}
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            return {"error": "Invalid response format", "message": str(e)}
        except Exception as e:
            logger.error(f"Request error: {e}")
            return {"error": "Unknown error", "message": str(e)}
            
    # アカウント情報
    async def get_balance(self) -> Dict:
        """アカウント残高の取得"""
        return await self._request('GET', '/api/balance')
        
    async def get_positions(self) -> Dict:
        """ポジション情報の取得"""
        return await self._request('GET', '/api/positions')
        
    async def get_order_history(self) -> Dict:
        """取引履歴の取得"""
        return await self._request('GET', '/api/history')
        
    # 注文管理
    async def place_order(self, order_data: Dict) -> Dict:
        """
        注文の発注
        
        Args:
            order_data (Dict): 注文データ
                {
                    "symbol": "BTC-USD",
                    "side": "buy" or "sell",
                    "type": "limit" or "market",
                    "quantity": float,
                    "price": float (optional for market orders)
                }
        """
        return await self._request('POST', '/api/place_order', data=order_data)
        
    async def cancel_order(self, order_id: str) -> Dict:
        """注文のキャンセル"""
        return await self._request('POST', '/api/cancel_order', data={'order_id': order_id})
        
    async def get_order_status(self, order_id: str) -> Dict:
        """注文ステータスの取得"""
        return await self._request('GET', '/api/order_status', params={'order_id': order_id})
        
    # 市場データ
    async def get_market_prices(self) -> Dict:
        """全取引ペアの現在価格を取得"""
        return await self._request('GET', '/api/market_prices')
        
    async def get_order_book(self, symbol: str) -> Dict:
        """
        特定の取引ペアのオーダーブックを取得
        
        Args:
            symbol (str): 取引ペア（例: "BTC-USD"）
        """
        return await self._request('GET', '/api/order_book', params={'symbol': symbol})
        
    # 資金管理
    async def withdraw(self, amount: float, address: str) -> Dict:
        """
        資金の出金
        
        Args:
            amount (float): 出金額
            address (str): 出金先アドレス
        """
        data = {
            'amount': amount,
            'address': address
        }
        return await self._request('POST', '/api/withdraw', data=data)
