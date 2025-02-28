import axios from 'axios';

// API基本設定
const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || 'http://localhost:5000/api';

// APIクライアントの設定
const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

/**
 * 特定の銘柄の市場データを取得する
 * @param symbol 銘柄シンボル (例: 'ETH-PERP')
 * @returns 市場データ (オーダーブック、ポジション、最近の取引など)
 */
export const fetchMarketData = async (symbol: string) => {
  try {
    const response = await apiClient.get(`/market-data/${symbol}`);
    return response.data;
  } catch (error) {
    console.error('Error fetching market data:', error);
    throw error;
  }
};

/**
 * 特定の銘柄のOHLCVデータを取得する
 * @param symbol 銘柄シンボル (例: 'ETH-PERP')
 * @param timeframe タイムフレーム (例: '1m', '5m', '1h', '4h', '1d')
 * @param limit 取得するデータポイントの数
 * @returns OHLCVデータの配列
 */
export const fetchOHLCVData = async (symbol: string, timeframe: string, limit: number = 100) => {
  try {
    const response = await apiClient.get(`/ohlcv/${symbol}`, {
      params: { timeframe, limit },
    });
    return response.data;
  } catch (error) {
    console.error('Error fetching OHLCV data:', error);
    throw error;
  }
};

/**
 * 特定の銘柄のティッカーデータを取得する
 * @param symbol 銘柄シンボル (例: 'ETH-PERP')
 * @returns ティッカーデータ
 */
export const fetchTickerData = async (symbol: string) => {
  try {
    const response = await apiClient.get(`/ticker/${symbol}`);
    return response.data;
  } catch (error) {
    console.error('Error fetching ticker data:', error);
    throw error;
  }
};

/**
 * アカウント情報を取得する
 * @returns アカウント情報 (残高、ポジションなど)
 */
export const fetchAccountInfo = async () => {
  try {
    const response = await apiClient.get('/account');
    return response.data;
  } catch (error) {
    console.error('Error fetching account info:', error);
    throw error;
  }
};

/**
 * 注文を作成する
 * @param orderData 注文データ
 * @returns 作成された注文の情報
 */
export const createOrder = async (orderData: any) => {
  try {
    const response = await apiClient.post('/orders', orderData);
    return response.data;
  } catch (error) {
    console.error('Error creating order:', error);
    throw error;
  }
};

/**
 * 注文をキャンセルする
 * @param orderId 注文ID
 * @returns キャンセル結果
 */
export const cancelOrder = async (orderId: string) => {
  try {
    const response = await apiClient.delete(`/orders/${orderId}`);
    return response.data;
  } catch (error) {
    console.error('Error canceling order:', error);
    throw error;
  }
};

/**
 * アクティブな注文のリストを取得する
 * @param symbol オプションの銘柄フィルター
 * @returns アクティブな注文のリスト
 */
export const fetchActiveOrders = async (symbol?: string) => {
  try {
    const response = await apiClient.get('/orders/active', {
      params: symbol ? { symbol } : undefined,
    });
    return response.data;
  } catch (error) {
    console.error('Error fetching active orders:', error);
    throw error;
  }
};

/**
 * 注文履歴を取得する
 * @param symbol オプションの銘柄フィルター
 * @param limit 取得する注文の数
 * @returns 注文履歴
 */
export const fetchOrderHistory = async (symbol?: string, limit: number = 50) => {
  try {
    const response = await apiClient.get('/orders/history', {
      params: {
        ...(symbol ? { symbol } : {}),
        limit,
      },
    });
    return response.data;
  } catch (error) {
    console.error('Error fetching order history:', error);
    throw error;
  }
};

/**
 * トレード履歴を取得する
 * @param symbol オプションの銘柄フィルター
 * @param limit 取得するトレードの数
 * @returns トレード履歴
 */
export const fetchTradeHistory = async (symbol?: string, limit: number = 50) => {
  try {
    const response = await apiClient.get('/trades/history', {
      params: {
        ...(symbol ? { symbol } : {}),
        limit,
      },
    });
    return response.data;
  } catch (error) {
    console.error('Error fetching trade history:', error);
    throw error;
  }
};

export default {
  fetchMarketData,
  fetchOHLCVData,
  fetchTickerData,
  fetchAccountInfo,
  createOrder,
  cancelOrder,
  fetchActiveOrders,
  fetchOrderHistory,
  fetchTradeHistory,
};
