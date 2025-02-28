#!/usr/bin/env python
"""
Hyperliquid API Test Script

This script tests the connectivity to the Hyperliquid API and verifies that
authentication is working correctly. It performs the following tests:
1. Public API endpoints (info, meta, etc.)
2. Authenticated API endpoints (user info, positions, etc.)
3. WebSocket connectivity

Usage:
    python api_test.py [--testnet]
"""

import os
import sys
import json
import time
import asyncio
import logging
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from dotenv import load_dotenv

import aiohttp
import websockets
from core.auth import HyperLiquidAuth
from core.config import settings

# ロギングの設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class HyperliquidAPITester:
    """Hyperliquid APIテスト用クラス"""
    
    def __init__(self):
        """初期化"""
        # 環境変数の読み込み
        load_dotenv(project_root / '.env')
        
        # 認証情報の取得
        self.auth = HyperLiquidAuth()
        self.api_key = os.getenv('HYPERLIQUID_API_KEY', '')
        self.api_secret = os.getenv('HYPERLIQUID_API_SECRET', '')
        
        # APIエンドポイント
        self.rest_url = settings.HYPERLIQUID_API_URL
        self.ws_url = settings.HYPERLIQUID_WS_URL
        
        logger.info(f"APIテスターを初期化しました: REST URL={self.rest_url}, WS URL={self.ws_url}")
        logger.info(f"API Key: {self.api_key[:10]}...{self.api_key[-4:] if len(self.api_key) > 14 else ''}")
        
        if not self.api_key or not self.api_secret:
            logger.warning("API認証情報が設定されていません。.envファイルを確認してください。")
    
    async def test_rest_api(self):
        """REST APIの基本機能をテスト"""
        logger.info("REST APIテストを開始します...")
        
        async with aiohttp.ClientSession() as session:
            # 1. 取引所の情報を取得
            logger.info("取引所情報を取得中...")
            async with session.get(f"{self.rest_url}/info") as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info(f"取引所情報: {json.dumps(data, indent=2)}")
                else:
                    logger.error(f"取引所情報の取得に失敗しました: {response.status}")
                    logger.error(await response.text())
            
            # 2. 市場データ（ティッカー）を取得
            logger.info("市場データ（ティッカー）を取得中...")
            async with session.get(f"{self.rest_url}/ticker") as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info(f"ティッカー情報: {json.dumps(data[:2], indent=2)}... (最初の2件のみ表示)")
                else:
                    logger.error(f"ティッカー情報の取得に失敗しました: {response.status}")
                    logger.error(await response.text())
            
            # 3. 特定の銘柄（ETH-PERP）の情報を取得
            symbol = "ETH-PERP"
            logger.info(f"{symbol}の情報を取得中...")
            async with session.get(f"{self.rest_url}/ticker?symbol={symbol}") as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info(f"{symbol}情報: {json.dumps(data, indent=2)}")
                else:
                    logger.error(f"{symbol}情報の取得に失敗しました: {response.status}")
                    logger.error(await response.text())
            
            # 4. アカウント情報の取得（認証が必要）
            if self.api_key and self.api_secret:
                logger.info("アカウント情報を取得中...")
                timestamp = int(time.time() * 1000)
                signature = self.auth.generate_signature(timestamp, "GET", "/user/info", None)
                
                headers = {
                    "HL-API-KEY": self.api_key,
                    "HL-SIGNATURE": signature,
                    "HL-TIMESTAMP": str(timestamp)
                }
                
                async with session.get(f"{self.rest_url}/user/info", headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        logger.info(f"アカウント情報: {json.dumps(data, indent=2)}")
                    else:
                        logger.error(f"アカウント情報の取得に失敗しました: {response.status}")
                        logger.error(await response.text())
            else:
                logger.warning("API認証情報が設定されていないため、アカウント情報の取得をスキップします。")
    
    async def test_websocket(self, duration=30):
        """WebSocketを使用したリアルタイムデータの取得をテスト"""
        logger.info(f"WebSocketテストを開始します... ({duration}秒間実行)")
        
        try:
            async with websockets.connect(self.ws_url) as websocket:
                # 1. ティッカーのサブスクリプション
                subscribe_msg = {
                    "method": "subscribe",
                    "params": ["ticker"]
                }
                await websocket.send(json.dumps(subscribe_msg))
                logger.info(f"サブスクリプションリクエスト送信: {json.dumps(subscribe_msg)}")
                
                # 指定された時間だけデータを受信
                start_time = time.time()
                message_count = 0
                
                while time.time() - start_time < duration:
                    try:
                        message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                        data = json.loads(message)
                        message_count += 1
                        
                        if message_count <= 3:  # 最初の3メッセージのみ詳細表示
                            logger.info(f"受信データ #{message_count}: {json.dumps(data)}")
                        else:
                            # それ以降はカウントのみ
                            if message_count % 10 == 0:
                                logger.info(f"{message_count}件のメッセージを受信しました")
                    
                    except asyncio.TimeoutError:
                        # タイムアウトは正常、継続
                        continue
                    except Exception as e:
                        logger.error(f"WebSocketデータ受信中にエラーが発生しました: {e}")
                        break
                
                logger.info(f"WebSocketテスト完了: {message_count}件のメッセージを受信しました")
        
        except Exception as e:
            logger.error(f"WebSocket接続中にエラーが発生しました: {e}")
    
    async def run_tests(self):
        """すべてのテストを実行"""
        logger.info("Hyperliquid APIテストを開始します...")
        
        try:
            # REST APIテスト
            await self.test_rest_api()
            
            # WebSocketテスト
            await self.test_websocket(duration=30)
            
            logger.info("すべてのテストが完了しました")
        
        except Exception as e:
            logger.error(f"テスト実行中にエラーが発生しました: {e}", exc_info=True)


async def main():
    """メイン関数"""
    tester = HyperliquidAPITester()
    await tester.run_tests()


if __name__ == "__main__":
    asyncio.run(main())
