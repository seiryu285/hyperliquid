/**
 * 市場データの共通型定義
 */

/**
 * オーダーブックデータの型
 */
export interface OrderBook {
  bids: [number, number][];
  asks: [number, number][];
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
  timestamp: number;
  price: number;
  previousPrice?: number;
  priceChange?: number;
  currentPrice?: number;
  bidPrice?: number;
  askPrice?: number;
  volume24h?: number;
  fundingRate?: number;
  orderbook?: OrderBook;
  orderBook?: OrderBook;
  positions?: Position[];
  recentTrades?: Trade[];
}

/**
 * OHLCVデータの型
 */
export interface OHLCVData {
  time: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
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
