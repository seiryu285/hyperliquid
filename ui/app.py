import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import asyncio
import sys
import os
from pathlib import Path

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from core.rest_client import HyperLiquidRestClient
from trading.order_manager import OrderManager
from trading.strategies.simple_market_maker import SimpleMarketMaker

# ページ設定
st.set_page_config(
    page_title="HyperLiquid Trading Dashboard",
    page_icon="📈",
    layout="wide"
)

# セッション状態の初期化
if 'client' not in st.session_state:
    st.session_state.client = HyperLiquidRestClient()
if 'order_manager' not in st.session_state:
    st.session_state.order_manager = OrderManager()
if 'market_maker' not in st.session_state:
    st.session_state.market_maker = None

# サイドバー
with st.sidebar:
    st.title("⚙️ 設定")
    
    # API設定
    st.subheader("API設定")
    api_key = st.text_input("API Key", type="password")
    api_secret = st.text_input("API Secret", type="password")
    
    if st.button("API設定を保存"):
        with open(".env", "w") as f:
            f.write(f"HYPERLIQUID_API_KEY={api_key}\n")
            f.write(f"HYPERLIQUID_API_SECRET={api_secret}\n")
        st.success("API設定を保存しました")

    # 取引設定
    st.subheader("取引設定")
    symbol = st.selectbox(
        "取引ペア",
        ["BTC-USD", "ETH-USD", "SOL-USD"]
    )
    
    order_size = st.number_input(
        "注文サイズ",
        min_value=0.001,
        value=0.01,
        step=0.001,
        format="%.3f"
    )
    
    spread = st.slider(
        "スプレッド (%)",
        min_value=0.1,
        max_value=1.0,
        value=0.2,
        step=0.1
    )
    
    max_position = st.number_input(
        "最大ポジション",
        min_value=0.01,
        value=0.05,
        step=0.01
    )

# メインコンテンツ
st.title("📊 HyperLiquid Trading Dashboard")

# マーケットデータ表示
col1, col2 = st.columns(2)

async def get_market_data():
    try:
        prices = await st.session_state.client.get_market_prices()
        return prices
    except Exception as e:
        st.error(f"市場データの取得に失敗しました: {e}")
        return None

async def get_order_book(symbol):
    try:
        order_book = await st.session_state.client.get_order_book(symbol)
        return order_book
    except Exception as e:
        st.error(f"オーダーブックの取得に失敗しました: {e}")
        return None

# 価格チャート
with col1:
    st.subheader("価格チャート")
    prices = asyncio.run(get_market_data())
    if prices:
        price_data = pd.DataFrame(prices).T
        fig = go.Figure(data=[
            go.Scatter(
                x=price_data.index,
                y=price_data['last'],
                mode='lines',
                name='Last Price'
            )
        ])
        fig.update_layout(
            height=400,
            margin=dict(l=0, r=0, t=0, b=0)
        )
        st.plotly_chart(fig, use_container_width=True)

# オーダーブック
with col2:
    st.subheader("オーダーブック")
    order_book = asyncio.run(get_order_book(symbol))
    if order_book:
        bids_df = pd.DataFrame(order_book['bids'], columns=['価格', '数量'])
        asks_df = pd.DataFrame(order_book['asks'], columns=['価格', '数量'])
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=bids_df['価格'],
            y=bids_df['数量'],
            name='買い注文',
            marker_color='rgba(0, 255, 0, 0.5)'
        ))
        fig.add_trace(go.Bar(
            x=asks_df['価格'],
            y=asks_df['数量'],
            name='売り注文',
            marker_color='rgba(255, 0, 0, 0.5)'
        ))
        fig.update_layout(
            height=400,
            margin=dict(l=0, r=0, t=0, b=0)
        )
        st.plotly_chart(fig, use_container_width=True)

# ポジション・注文管理
col3, col4 = st.columns(2)

# ポジション情報
with col3:
    st.subheader("ポジション情報")
    async def get_positions():
        try:
            positions = await st.session_state.order_manager.get_position(symbol)
            return positions
        except Exception as e:
            st.error(f"ポジション情報の取得に失敗しました: {e}")
            return None
    
    positions = asyncio.run(get_positions())
    if positions:
        st.metric(
            "現在のポジション",
            f"{positions['size']} {symbol.split('-')[0]}",
            f"評価損益: {positions.get('unrealized_pnl', 0):.2f} USD"
        )

# アクティブな注文
with col4:
    st.subheader("アクティブな注文")
    active_orders = st.session_state.order_manager.get_active_orders(symbol)
    if active_orders:
        orders_df = pd.DataFrame(active_orders)
        st.dataframe(orders_df)

# 取引操作
st.subheader("取引操作")
col5, col6, col7 = st.columns(3)

with col5:
    if st.button("マーケットメイカーを開始"):
        if not st.session_state.market_maker:
            st.session_state.market_maker = SimpleMarketMaker(
                symbol=symbol,
                order_size=order_size,
                spread_percentage=spread/100,
                max_position=max_position
            )
            asyncio.run(st.session_state.market_maker.initialize())
            st.success("マーケットメイカーを開始しました")

with col6:
    if st.button("マーケットメイカーを停止"):
        if st.session_state.market_maker:
            st.session_state.market_maker.stop()
            st.session_state.market_maker = None
            st.success("マーケットメイカーを停止しました")

with col7:
    if st.button("全注文をキャンセル"):
        async def cancel_all_orders():
            orders = st.session_state.order_manager.get_active_orders()
            for order in orders:
                await st.session_state.order_manager.cancel_order(order['order_id'])
        asyncio.run(cancel_all_orders())
        st.success("全ての注文をキャンセルしました")

# 手動注文フォーム
st.subheader("手動注文")
col8, col9, col10, col11 = st.columns(4)

with col8:
    order_type = st.selectbox("注文タイプ", ["指値", "成行"])

with col9:
    side = st.selectbox("売買", ["買い", "売り"])

with col10:
    quantity = st.number_input("数量", min_value=0.001, value=0.01, step=0.001)

with col11:
    price = st.number_input("価格 (指値のみ)", min_value=0.01, value=None, disabled=order_type=="成行")

if st.button("注文を送信"):
    async def place_manual_order():
        try:
            if order_type == "成行":
                await st.session_state.order_manager.place_market_order(
                    symbol=symbol,
                    side="buy" if side == "買い" else "sell",
                    quantity=quantity
                )
            else:
                await st.session_state.order_manager.place_limit_order(
                    symbol=symbol,
                    side="buy" if side == "買い" else "sell",
                    quantity=quantity,
                    price=price
                )
            st.success("注文を送信しました")
        except Exception as e:
            st.error(f"注文の送信に失敗しました: {e}")
    
    asyncio.run(place_manual_order())
