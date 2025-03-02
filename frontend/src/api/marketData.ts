// API関数：市場データとOHLCVデータの取得
import { HyperLiquidApiClient, ApiResponse, createApiClientFromEnv } from './apiClient';
import { HyperLiquidWebSocketClient, WebSocketMessageHandler, createWebSocketClientFromEnv } from './webSocketClient';
import { Order, OrderParams, OrderResponse, OrderSide, OrderStatus, OrderType, TimeInForce } from '../types/order';

// シングルトンインスタンス
let apiClient: HyperLiquidApiClient | null = null;
let wsClient: HyperLiquidWebSocketClient | null = null;

/**
 * APIクライアントを取得する
 * @returns HyperLiquidApiClientインスタンス
 */
export const getApiClient = (): HyperLiquidApiClient => {
  if (!apiClient) {
    apiClient = createApiClientFromEnv();
  }
  return apiClient;
};

/**
 * WebSocketクライアントを取得する
 * @returns HyperLiquidWebSocketClientインスタンス
 */
export const getWsClient = (): HyperLiquidWebSocketClient => {
  if (!wsClient) {
    wsClient = createWebSocketClientFromEnv({
      reconnectOnClose: true,
      reconnectInterval: 2000,
      maxReconnectAttempts: 5,
      pingInterval: 30000
    });
  }
  return wsClient;
};

// モックデータの生成
const generateMockData = () => {
  // 現在の価格（ランダム）
  const currentPrice = 3000 + Math.random() * 500;
  
  // モックのオーダーブック
  const orderBook = {
    bids: Array(20).fill(0).map((_, i) => [currentPrice - (i + 1) * 5, 10 + Math.random() * 20]),
    asks: Array(20).fill(0).map((_, i) => [currentPrice + (i + 1) * 5, 10 + Math.random() * 20])
  };
  
  // モックのポジション
  const positions = [
    {
      symbol: 'ETH-PERP',
      size: 1.5,
      entryPrice: currentPrice - 50,
      markPrice: currentPrice,
      liquidationPrice: currentPrice - 300,
      margin: 1000,
      leverage: 3,
      unrealizedPnl: 75,
      unrealizedPnlPercent: 7.5
    }
  ];
  
  // モックの最近の取引
  const recentTrades = Array(20).fill(0).map((_, i) => ({
    id: `trade-${i}`,
    price: currentPrice - 50 + Math.random() * 100,
    size: 0.1 + Math.random() * 2,
    side: Math.random() > 0.5 ? 'buy' : 'sell',
    timestamp: new Date(Date.now() - i * 60 * 1000).toISOString()
  }));
  
  // 前回の価格（少し低め）
  const previousPrice = currentPrice - 10 + Math.random() * 20;
  
  return {
    price: currentPrice,
    previousPrice,
    priceChange: currentPrice - previousPrice,
    orderBook,
    positions,
    recentTrades
  };
};

// モックのOHLCVデータ生成
const generateMockOHLCVData = (timeframe: string, count: number = 100) => {
  const now = new Date();
  const data = [];
  
  // タイムフレームに応じた時間間隔を設定
  let interval: number;
  switch (timeframe) {
    case '1h':
      interval = 60 * 60 * 1000; // 1時間
      break;
    case '4h':
      interval = 4 * 60 * 60 * 1000; // 4時間
      break;
    case '1d':
      interval = 24 * 60 * 60 * 1000; // 1日
      break;
    case '1w':
      interval = 7 * 24 * 60 * 60 * 1000; // 1週間
      break;
    default:
      interval = 60 * 60 * 1000; // デフォルトは1時間
  }
  
  // 基準価格
  let basePrice = 3000;
  
  // 過去のデータから生成
  for (let i = count - 1; i >= 0; i--) {
    const time = new Date(now.getTime() - i * interval);
    
    // 価格の変動をシミュレート
    const volatility = 0.02; // 2%の変動
    const changePercent = (Math.random() * 2 - 1) * volatility;
    basePrice = basePrice * (1 + changePercent);
    
    const open = basePrice;
    const close = basePrice * (1 + (Math.random() * 0.01 - 0.005)); // ±0.5%
    const high = Math.max(open, close) * (1 + Math.random() * 0.01); // 最大1%高く
    const low = Math.min(open, close) * (1 - Math.random() * 0.01); // 最大1%低く
    const volume = 100 + Math.random() * 900; // 100-1000の範囲
    
    data.push({
      time: time.toISOString(),
      open,
      high,
      low,
      close,
      volume
    });
  }
  
  return data;
};

/**
 * オーダーブックデータの型
 */
export interface OrderBook {
  bids: [number, number][]; // [価格, 数量]のペアの配列
  asks: [number, number][]; // [価格, 数量]のペアの配列
}

/**
 * ポジションデータの型
 */
export interface Position {
  symbol: string;
  size: number;
  entryPrice: number;
  markPrice: number;
  liquidationPrice: number;
  margin: number;
  leverage: number;
  unrealizedPnl: number;
  unrealizedPnlPercent: number;
}

/**
 * 取引データの型
 */
export interface Trade {
  id: string;
  price: number;
  size: number;
  side: 'buy' | 'sell';
  timestamp: string;
}

/**
 * 市場データの型
 */
export interface MarketData {
  price: number;
  previousPrice: number;
  priceChange: number;
  orderBook: OrderBook;
  positions: Position[];
  recentTrades: Trade[];
}

/**
 * OHLCVデータの型
 */
export interface OHLCVData {
  time: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

/**
 * 注文タイプ
 */
export enum OrderType {
  MARKET = 'market',
  LIMIT = 'limit',
  STOP_MARKET = 'stopMarket',
  STOP_LIMIT = 'stopLimit',
  TAKE_PROFIT_MARKET = 'takeProfitMarket',
  TAKE_PROFIT_LIMIT = 'takeProfitLimit'
}

/**
 * 注文サイド
 */
export enum OrderSide {
  BUY = 'buy',
  SELL = 'sell'
}

/**
 * 注文の有効期限
 */
export enum TimeInForce {
  GTC = 'gtc', // Good Till Cancel
  IOC = 'ioc', // Immediate Or Cancel
  FOK = 'fok'  // Fill Or Kill
}

/**
 * 注文パラメータの型
 */
export interface OrderParams {
  symbol: string;
  side: OrderSide;
  type: OrderType;
  size: number;
  price?: number;
  stopPrice?: number;
  reduceOnly?: boolean;
  postOnly?: boolean;
  timeInForce?: TimeInForce;
  clientOrderId?: string;
}

/**
 * 注文レスポンスの型
 */
export interface OrderResponse {
  orderId: string;
  clientOrderId?: string;
  symbol: string;
  side: OrderSide;
  type: OrderType;
  size: number;
  price?: number;
  stopPrice?: number;
  reduceOnly: boolean;
  postOnly: boolean;
  timeInForce: TimeInForce;
  status: string;
  timestamp: string;
}

/**
 * 銘柄情報の型
 */
export interface SymbolInfo {
  symbol: string;
  baseCurrency: string;
  quoteCurrency: string;
  pricePrecision: number;
  quantityPrecision: number;
  minOrderSize: number;
  maxLeverage: number;
  tradingFee: number;
  fundingRate: number;
  isActive: boolean;
}

/**
 * 市場データを取得する
 * @param symbol 銘柄シンボル（デフォルトはETH-PERP）
 * @param useMock モックデータを使用するかどうか
 * @returns 市場データ（オーダーブック、ポジション、最近の取引など）
 */
export const fetchMarketData = async (symbol: string = 'ETH-PERP', useMock: boolean = false): Promise<MarketData> => {
  try {
    if (useMock) {
      return generateMockData();
    }

    // APIクライアントを取得
    const client = getApiClient();
    
    // 並列でリクエストを実行
    const [orderBookResponse, positionsResponse, tradesResponse] = await Promise.all([
      client.get<OrderBook>(`/api/orderbook/${symbol}`),
      client.get<Position[]>('/api/positions'),
      client.get<Trade[]>(`/api/trades/${symbol}?limit=20`)
    ]);
    
    // エラーチェック
    if (!orderBookResponse.success) {
      throw new Error(`Failed to fetch order book: ${orderBookResponse.error}`);
    }
    
    if (!positionsResponse.success) {
      throw new Error(`Failed to fetch positions: ${positionsResponse.error}`);
    }
    
    if (!tradesResponse.success) {
      throw new Error(`Failed to fetch trades: ${tradesResponse.error}`);
    }
    
    // 最新の価格を取得（最新の取引から）
    const latestTrade = tradesResponse.data![0];
    const currentPrice = latestTrade ? latestTrade.price : 0;
    
    // 前回の価格（1時間前の価格）を取得
    const oneHourAgo = new Date(Date.now() - 60 * 60 * 1000).toISOString();
    const previousTradeResponse = await client.get<Trade[]>(`/api/trades/${symbol}?endTime=${oneHourAgo}&limit=1`);
    
    let previousPrice = currentPrice;
    if (previousTradeResponse.success && previousTradeResponse.data && previousTradeResponse.data.length > 0) {
      previousPrice = previousTradeResponse.data[0].price;
    }
    
    return {
      price: currentPrice,
      previousPrice,
      priceChange: currentPrice - previousPrice,
      orderBook: orderBookResponse.data!,
      positions: positionsResponse.data!.filter(pos => pos.symbol === symbol),
      recentTrades: tradesResponse.data!
    };
  } catch (error) {
    console.error('Error fetching market data:', error);
    
    // エラー時はモックデータを返す
    return generateMockData();
  }
};

/**
 * OHLCV（ローソク足）データを取得する
 * @param timeframe タイムフレーム（1h, 4h, 1d, 1wなど）
 * @param symbol 銘柄シンボル（デフォルトはETH-PERP）
 * @param limit 取得するデータポイントの数
 * @param useMock モックデータを使用するかどうか
 * @returns OHLCVデータの配列
 */
export const fetchOHLCVData = async (
  timeframe: string = '1h',
  symbol: string = 'ETH-PERP',
  limit: number = 100,
  useMock: boolean = false
): Promise<OHLCVData[]> => {
  try {
    if (useMock) {
      return generateMockOHLCVData(timeframe, limit);
    }
    
    // APIクライアントを取得
    const client = getApiClient();
    
    // OHLCVデータを取得
    const response = await client.get<OHLCVData[]>(`/api/klines/${symbol}?interval=${timeframe}&limit=${limit}`);
    
    // エラーチェック
    if (!response.success) {
      throw new Error(`Failed to fetch OHLCV data: ${response.error}`);
    }
    
    return response.data!;
  } catch (error) {
    console.error('Error fetching OHLCV data:', error);
    
    // エラー時はモックデータを返す
    return generateMockOHLCVData(timeframe, limit);
  }
};

/**
 * WebSocketでオーダーブックデータをサブスクライブする
 * @param symbol 銘柄シンボル
 * @param handler データハンドラ
 * @returns サブスクリプションが成功したかどうか
 */
export const subscribeOrderBook = async (
  symbol: string,
  handler: WebSocketMessageHandler
): Promise<boolean> => {
  const ws = getWsClient();
  
  // WebSocketが接続されていない場合は接続
  if (ws.getState() !== 'open' && ws.getState() !== 'authenticated') {
    await ws.connect();
  }
  
  // オーダーブックチャンネルにハンドラを登録
  ws.addMessageHandler(`orderbook:${symbol}`, handler);
  
  // オーダーブックチャンネルをサブスクライブ
  return ws.subscribe(`orderbook:${symbol}`);
};

/**
 * WebSocketでオーダーブックデータのサブスクリプションを解除する
 * @param symbol 銘柄シンボル
 * @param handler データハンドラ
 * @returns アンサブスクリプションが成功したかどうか
 */
export const unsubscribeOrderBook = (
  symbol: string,
  handler: WebSocketMessageHandler
): boolean => {
  const ws = getWsClient();
  
  // オーダーブックチャンネルからハンドラを削除
  ws.removeMessageHandler(`orderbook:${symbol}`, handler);
  
  // オーダーブックチャンネルをアンサブスクライブ
  return ws.unsubscribe(`orderbook:${symbol}`);
};

/**
 * WebSocketで取引データをサブスクライブする
 * @param symbol 銘柄シンボル
 * @param handler データハンドラ
 * @returns サブスクリプションが成功したかどうか
 */
export const subscribeTrades = async (
  symbol: string,
  handler: WebSocketMessageHandler
): Promise<boolean> => {
  const ws = getWsClient();
  
  // WebSocketが接続されていない場合は接続
  if (ws.getState() !== 'open' && ws.getState() !== 'authenticated') {
    await ws.connect();
  }
  
  // 取引チャンネルにハンドラを登録
  ws.addMessageHandler(`trades:${symbol}`, handler);
  
  // 取引チャンネルをサブスクライブ
  return ws.subscribe(`trades:${symbol}`);
};

/**
 * WebSocketで取引データのサブスクリプションを解除する
 * @param symbol 銘柄シンボル
 * @param handler データハンドラ
 * @returns アンサブスクリプションが成功したかどうか
 */
export const unsubscribeTrades = (
  symbol: string,
  handler: WebSocketMessageHandler
): boolean => {
  const ws = getWsClient();
  
  // 取引チャンネルからハンドラを削除
  ws.removeMessageHandler(`trades:${symbol}`, handler);
  
  // 取引チャンネルをアンサブスクライブ
  return ws.unsubscribe(`trades:${symbol}`);
};

/**
 * 注文を送信する
 * @param params 注文パラメータ
 * @returns 注文レスポンス
 */
export const placeOrder = async (params: OrderParams): Promise<ApiResponse<OrderResponse>> => {
  try {
    // APIクライアントを取得
    const client = getApiClient();
    
    // 注文を送信
    return await client.post<OrderResponse>('/api/orders', params);
  } catch (error) {
    console.error('Error placing order:', error);
    return {
      success: false,
      error: error instanceof Error ? error.message : 'Unknown error'
    };
  }
};

/**
 * 注文をキャンセルする
 * @param orderId 注文ID
 * @returns キャンセルが成功したかどうか
 */
export const cancelOrder = async (orderId: string): Promise<ApiResponse<boolean>> => {
  try {
    // APIクライアントを取得
    const client = getApiClient();
    
    // 注文をキャンセル
    return await client.delete<boolean>(`/api/orders/${orderId}`);
  } catch (error) {
    console.error('Error canceling order:', error);
    return {
      success: false,
      error: error instanceof Error ? error.message : 'Unknown error'
    };
  }
};

/**
 * すべての注文をキャンセルする
 * @param symbol 銘柄シンボル（指定した場合はその銘柄の注文のみキャンセル）
 * @returns キャンセルが成功したかどうか
 */
export const cancelAllOrders = async (symbol?: string): Promise<ApiResponse<boolean>> => {
  try {
    // APIクライアントを取得
    const client = getApiClient();
    
    // すべての注文をキャンセル
    const endpoint = symbol ? `/api/orders?symbol=${symbol}` : '/api/orders';
    return await client.delete<boolean>(endpoint);
  } catch (error) {
    console.error('Error canceling all orders:', error);
    return {
      success: false,
      error: error instanceof Error ? error.message : 'Unknown error'
    };
  }
};

/**
 * 利用可能な銘柄一覧を取得する
 * @param useMock モックデータを使用するかどうか
 * @returns 銘柄情報の配列
 */
export const fetchAvailableSymbols = async (useMock: boolean = false): Promise<ApiResponse<SymbolInfo[]>> => {
  if (useMock) {
    // モックデータを返す
    const mockSymbols: SymbolInfo[] = [
      {
        symbol: 'ETH-PERP',
        baseCurrency: 'ETH',
        quoteCurrency: 'USD',
        pricePrecision: 2,
        quantityPrecision: 3,
        minOrderSize: 0.001,
        maxLeverage: 50,
        tradingFee: 0.0005,
        fundingRate: 0.0001,
        isActive: true
      },
      {
        symbol: 'BTC-PERP',
        baseCurrency: 'BTC',
        quoteCurrency: 'USD',
        pricePrecision: 1,
        quantityPrecision: 4,
        minOrderSize: 0.0001,
        maxLeverage: 50,
        tradingFee: 0.0005,
        fundingRate: 0.0001,
        isActive: true
      },
      {
        symbol: 'SOL-PERP',
        baseCurrency: 'SOL',
        quoteCurrency: 'USD',
        pricePrecision: 3,
        quantityPrecision: 1,
        minOrderSize: 0.1,
        maxLeverage: 20,
        tradingFee: 0.0005,
        fundingRate: 0.0002,
        isActive: true
      },
      {
        symbol: 'AVAX-PERP',
        baseCurrency: 'AVAX',
        quoteCurrency: 'USD',
        pricePrecision: 3,
        quantityPrecision: 1,
        minOrderSize: 0.1,
        maxLeverage: 20,
        tradingFee: 0.0005,
        fundingRate: 0.0002,
        isActive: true
      }
    ];
    
    return {
      success: true,
      data: mockSymbols,
      error: null
    };
  }
  
  try {
    const client = getApiClient();
    const response = await client.get('/info/markets');
    
    if (response.success && response.data) {
      // APIレスポンスを適切な形式に変換
      const symbols: SymbolInfo[] = response.data.map((item: any) => ({
        symbol: item.symbol,
        baseCurrency: item.baseCurrency,
        quoteCurrency: item.quoteCurrency,
        pricePrecision: item.pricePrecision,
        quantityPrecision: item.quantityPrecision,
        minOrderSize: item.minOrderSize,
        maxLeverage: item.maxLeverage,
        tradingFee: item.tradingFee,
        fundingRate: item.fundingRate,
        isActive: item.isActive
      }));
      
      return {
        success: true,
        data: symbols,
        error: null
      };
    }
    
    return {
      success: false,
      data: null,
      error: response.error || 'Failed to fetch available symbols'
    };
  } catch (error) {
    console.error('Error fetching available symbols:', error);
    return {
      success: false,
      data: null,
      error: error instanceof Error ? error.message : 'Unknown error'
    };
  }
};

/**
 * 注文履歴を取得する
 * @param symbol 銘柄シンボル（指定しない場合はすべての銘柄）
 * @param limit 取得する注文数
 * @param useMock モックデータを使用するかどうか
 * @returns 注文履歴
 */
export const fetchOrderHistory = async (
  symbol?: string,
  limit: number = 50,
  useMock: boolean = false
): Promise<ApiResponse<Order[]>> => {
  if (useMock) {
    // モックデータを返す
    const mockOrders: Order[] = Array(limit).fill(0).map((_, i) => ({
      id: `order-${i}`,
      symbol: symbol || (Math.random() > 0.5 ? 'ETH-PERP' : 'BTC-PERP'),
      side: Math.random() > 0.5 ? 'buy' : 'sell',
      type: Math.random() > 0.3 ? 'limit' : 'market',
      price: Math.random() > 0.3 ? 3000 + Math.random() * 500 : 0,
      quantity: 0.1 + Math.random() * 2,
      filledQuantity: Math.random() > 0.3 ? 0 : (0.1 + Math.random() * 2),
      status: ['open', 'filled', 'partially_filled', 'canceled'][Math.floor(Math.random() * 4)],
      timestamp: Date.now() - Math.floor(Math.random() * 1000000),
      timeInForce: ['GTC', 'IOC', 'FOK'][Math.floor(Math.random() * 3)]
    }));
    
    return {
      success: true,
      data: mockOrders,
      error: null
    };
  }
  
  try {
    const client = getApiClient();
    const params: Record<string, any> = { limit };
    if (symbol) {
      params.symbol = symbol;
    }
    
    const response = await client.get('/user/orders', params);
    
    if (response.success && response.data) {
      // APIレスポンスを適切な形式に変換
      const orders: Order[] = response.data.map((item: any) => ({
        id: item.orderId,
        symbol: item.symbol,
        side: item.side.toLowerCase(),
        type: item.type.toLowerCase(),
        price: item.price || 0,
        quantity: item.quantity,
        filledQuantity: item.filledQuantity,
        status: item.status.toLowerCase(),
        timestamp: new Date(item.timestamp).getTime(),
        timeInForce: item.timeInForce
      }));
      
      return {
        success: true,
        data: orders,
        error: null
      };
    }
    
    return {
      success: false,
      data: null,
      error: response.error || 'Failed to fetch order history'
    };
  } catch (error) {
    console.error('Error fetching order history:', error);
    return {
      success: false,
      data: null,
      error: error instanceof Error ? error.message : 'Unknown error'
    };
  }
};

/**
 * 注文板データを取得する
 * @param symbol 銘柄シンボル
 * @returns 注文板データ
 */
export const getOrderBook = async (symbol: string): Promise<{ bids: [number, number][]; asks: [number, number][] }> => {
  try {
    const response = await fetchMarketData(symbol, true);
    if (response.success && response.data) {
      return response.data.orderBook;
    }
    return { bids: [], asks: [] };
  } catch (error) {
    console.error('Error fetching order book:', error);
    return { bids: [], asks: [] };
  }
};

/**
 * 取引履歴を取得する
 * @param symbol 銘柄シンボル
 * @returns 取引履歴
 */
export const getTradeHistory = async (symbol: string): Promise<{ price: number; size: number; side: string; timestamp: number }[]> => {
  try {
    const response = await fetchMarketData(symbol, true);
    if (response.success && response.data) {
      return response.data.recentTrades.map(trade => ({
        price: trade.price,
        size: trade.size,
        side: trade.side,
        timestamp: new Date(trade.timestamp).getTime()
      }));
    }
    return [];
  } catch (error) {
    console.error('Error fetching trade history:', error);
    return [];
  }
};

/**
 * OHLCV（ローソク足）データを取得する
 * @param symbol 銘柄シンボル
 * @param timeframe タイムフレーム
 * @param limit 取得するデータポイントの数
 * @returns OHLCVデータ
 */
export const getOHLCV = async (
  symbol: string,
  timeframe: string = '1h',
  limit: number = 100
): Promise<{ time: number; open: number; high: number; low: number; close: number; volume: number }[]> => {
  try {
    const response = await fetchOHLCVData(timeframe, symbol, limit, true);
    if (response.success && response.data) {
      return response.data.map(item => ({
        time: new Date(item.time).getTime(),
        open: item.open,
        high: item.high,
        low: item.low,
        close: item.close,
        volume: item.volume
      }));
    }
    return [];
  } catch (error) {
    console.error('Error fetching OHLCV data:', error);
    return [];
  }
};

/**
 * WebSocketでオーダーブックデータをサブスクライブする（エイリアス）
 */
export const subscribeToOrderBook = subscribeOrderBook;

/**
 * WebSocketでオーダーブックデータのサブスクリプションを解除する（エイリアス）
 */
export const unsubscribeFromOrderBook = unsubscribeOrderBook;

/**
 * WebSocketで取引データをサブスクライブする（エイリアス）
 */
export const subscribeToTrades = subscribeTrades;

/**
 * WebSocketで取引データのサブスクリプションを解除する（エイリアス）
 */
export const unsubscribeFromTrades = unsubscribeTrades;
