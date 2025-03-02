/**
 * 注文サイド（買い/売り）
 */
export enum OrderSide {
  BUY = 'buy',
  SELL = 'sell'
}

/**
 * 注文タイプ（成行/指値）
 */
export enum OrderType {
  MARKET = 'market',
  LIMIT = 'limit'
}

/**
 * 注文の有効期限
 */
export enum TimeInForce {
  GTC = 'GTC', // Good Till Cancel - キャンセルするまで有効
  IOC = 'IOC', // Immediate or Cancel - 即時執行されなかった部分はキャンセル
  FOK = 'FOK'  // Fill or Kill - 全数量即時執行されるか、全てキャンセル
}

/**
 * 注文ステータス
 */
export enum OrderStatus {
  OPEN = 'open',
  FILLED = 'filled',
  PARTIALLY_FILLED = 'partially_filled',
  CANCELED = 'canceled',
  REJECTED = 'rejected'
}

/**
 * 注文オブジェクト
 */
export interface Order {
  id: string;
  symbol: string;
  side: OrderSide;
  type: OrderType;
  price: number;
  quantity: number;
  filledQuantity: number;
  status: OrderStatus;
  timestamp: number;
  timeInForce: TimeInForce;
}

/**
 * 注文パラメータ
 */
export interface OrderParams {
  symbol: string;
  side: OrderSide;
  type: OrderType;
  price?: number;
  quantity: number;
  timeInForce?: TimeInForce;
}

/**
 * 注文レスポンス
 */
export interface OrderResponse {
  orderId: string;
  symbol: string;
  status: OrderStatus;
  message?: string;
}
