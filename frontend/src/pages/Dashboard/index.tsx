import React, { useState, useEffect, useCallback } from 'react';
import {
  Box,
  Grid,
  Paper,
  Typography,
  Button,
  CircularProgress,
  Divider,
  FormControlLabel,
  Switch,
  Tabs,
  Tab,
  Alert,
  Snackbar
} from '@mui/material';
import RefreshIcon from '@mui/icons-material/Refresh';
import View3dIcon from '@mui/icons-material/ViewInAr';
import View2dIcon from '@mui/icons-material/Splitscreen';
import OrderBook from '../../components/OrderBook';
import TradeHistory from '../../components/TradeHistory';
import PriceChart from '../../components/PriceChart';
import OrderForm from '../../components/OrderForm';
import WebSocketStatus from '../../components/WebSocketStatus';
import OrderHistory from '../../components/OrderHistory';
import SymbolSelector from '../../components/SymbolSelector';
import {
  getOrderBook,
  getTradeHistory,
  getOHLCV,
  subscribeToOrderBook,
  subscribeToTrades,
  unsubscribeFromOrderBook,
  unsubscribeFromTrades,
  placeOrder,
  cancelOrder
} from '../../api/marketData';
import { OrderSide, OrderType, TimeInForce } from '../../types/order';
import { getWebSocketClient, WebSocketState } from '../../api/webSocketClient';

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`simple-tabpanel-${index}`}
      aria-labelledby={`simple-tab-${index}`}
      {...other}
      style={{ height: '100%' }}
    >
      {value === index && (
        <Box sx={{ p: 0, height: '100%' }}>
          {children}
        </Box>
      )}
    </div>
  );
}

const Dashboard: React.FC = () => {
  // 選択された銘柄
  const [selectedSymbol, setSelectedSymbol] = useState<string>('ETH-PERP');
  
  // マーケットデータ
  const [orderBook, setOrderBook] = useState<{ bids: [number, number][]; asks: [number, number][] }>({ bids: [], asks: [] });
  const [trades, setTrades] = useState<{ price: number; size: number; side: string; timestamp: number }[]>([]);
  const [ohlcv, setOHLCV] = useState<{ time: number; open: number; high: number; low: number; close: number; volume: number }[]>([]);
  
  // UI状態
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [is3DView, setIs3DView] = useState<boolean>(false);
  const [autoRefresh, setAutoRefresh] = useState<boolean>(true);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);
  const [tabValue, setTabValue] = useState<number>(0);
  
  // WebSocket状態
  const [wsState, setWsState] = useState<WebSocketState>(WebSocketState.CLOSED);
  const [wsLatency, setWsLatency] = useState<number | null>(null);
  
  // 注文状態
  const [orderStatus, setOrderStatus] = useState<{ message: string; type: 'success' | 'error' | 'info' | 'warning' } | null>(null);
  const [isSubmitting, setIsSubmitting] = useState<boolean>(false);
  
  // WebSocketクライアントの取得
  const wsClient = getWebSocketClient();
  
  // WebSocketの状態監視
  useEffect(() => {
    const checkWsState = () => {
      const state = wsClient.getState();
      setWsState(state);
      
      const stats = wsClient.getStats();
      setWsLatency(stats.latency);
    };
    
    // 初期状態を設定
    checkWsState();
    
    // 1秒ごとに状態をチェック
    const interval = setInterval(checkWsState, 1000);
    
    return () => {
      clearInterval(interval);
    };
  }, [wsClient]);
  
  // WebSocketの接続
  useEffect(() => {
    const connectWebSocket = async () => {
      if (wsClient.getState() === WebSocketState.CLOSED) {
        await wsClient.connect();
      }
    };
    
    connectWebSocket();
    
    return () => {
      // コンポーネントのアンマウント時に切断しない
      // WebSocketクライアントはシングルトンなので、アプリケーション全体で共有される
    };
  }, [wsClient]);
  
  // データの初期読み込み
  useEffect(() => {
    const fetchInitialData = async () => {
      setLoading(true);
      setError(null);
      
      try {
        // 注文板データの取得
        const orderBookData = await getOrderBook(selectedSymbol);
        setOrderBook(orderBookData);
        
        // 取引履歴の取得
        const tradeHistoryData = await getTradeHistory(selectedSymbol);
        setTrades(tradeHistoryData);
        
        // OHLCV（ローソク足）データの取得
        const ohlcvData = await getOHLCV(selectedSymbol, '1h', 100);
        setOHLCV(ohlcvData);
        
        setLastUpdated(new Date());
      } catch (err) {
        console.error('Failed to fetch initial data:', err);
        setError('データの取得に失敗しました。再試行してください。');
      } finally {
        setLoading(false);
      }
    };
    
    fetchInitialData();
  }, [selectedSymbol]);
  
  // WebSocketサブスクリプション
  useEffect(() => {
    if (!autoRefresh) return;
    
    // 注文板のサブスクリプション
    const orderBookHandler = (data: any) => {
      if (data.symbol === selectedSymbol) {
        setOrderBook(prevOrderBook => ({
          bids: data.bids || prevOrderBook.bids,
          asks: data.asks || prevOrderBook.asks
        }));
        setLastUpdated(new Date());
      }
    };
    
    // 取引履歴のサブスクリプション
    const tradeHandler = (data: any) => {
      if (data.symbol === selectedSymbol) {
        setTrades(prevTrades => {
          const newTrades = [...prevTrades, ...data.trades];
          // 最新の100件のみを保持
          return newTrades.slice(-100);
        });
        setLastUpdated(new Date());
      }
    };
    
    // サブスクリプションの登録
    subscribeToOrderBook(selectedSymbol, orderBookHandler);
    subscribeToTrades(selectedSymbol, tradeHandler);
    
    return () => {
      // サブスクリプションの解除
      unsubscribeFromOrderBook(selectedSymbol, orderBookHandler);
      unsubscribeFromTrades(selectedSymbol, tradeHandler);
    };
  }, [selectedSymbol, autoRefresh]);
  
  // データの手動更新
  const handleRefresh = async () => {
    setLoading(true);
    setError(null);
    
    try {
      // 注文板データの取得
      const orderBookData = await getOrderBook(selectedSymbol);
      setOrderBook(orderBookData);
      
      // 取引履歴の取得
      const tradeHistoryData = await getTradeHistory(selectedSymbol);
      setTrades(tradeHistoryData);
      
      // OHLCV（ローソク足）データの取得
      const ohlcvData = await getOHLCV(selectedSymbol, '1h', 100);
      setOHLCV(ohlcvData);
      
      setLastUpdated(new Date());
    } catch (err) {
      console.error('Failed to refresh data:', err);
      setError('データの更新に失敗しました。再試行してください。');
    } finally {
      setLoading(false);
    }
  };
  
  // 注文の送信
  const handleSubmitOrder = async (order: {
    side: OrderSide;
    type: OrderType;
    price?: number;
    quantity: number;
    timeInForce?: TimeInForce;
  }) => {
    setIsSubmitting(true);
    setOrderStatus(null);
    
    try {
      const result = await placeOrder({
        symbol: selectedSymbol,
        side: order.side,
        type: order.type,
        price: order.price,
        quantity: order.quantity,
        timeInForce: order.timeInForce || TimeInForce.GTC
      });
      
      if (result.success && result.data) {
        setOrderStatus({
          message: `注文が正常に送信されました。注文ID: ${result.data.orderId}`,
          type: 'success'
        });
      } else {
        setOrderStatus({
          message: `注文の送信に失敗しました: ${result.error || '不明なエラー'}`,
          type: 'error'
        });
      }
      
      // 注文板の更新
      handleRefresh();
    } catch (err) {
      console.error('Failed to place order:', err);
      setOrderStatus({
        message: `注文の送信に失敗しました: ${err instanceof Error ? err.message : '不明なエラー'}`,
        type: 'error'
      });
    } finally {
      setIsSubmitting(false);
    }
  };
  
  // 注文のキャンセル
  const handleCancelOrder = async (orderId: string) => {
    setIsSubmitting(true);
    setOrderStatus(null);
    
    try {
      const result = await cancelOrder(orderId);
      
      if (result.success && result.data) {
        setOrderStatus({
          message: `注文 ${orderId} が正常にキャンセルされました。`,
          type: 'success'
        });
      } else {
        setOrderStatus({
          message: `注文のキャンセルに失敗しました: ${result.error || '不明なエラー'}`,
          type: 'error'
        });
      }
      
      // 注文板の更新
      handleRefresh();
    } catch (err) {
      console.error('Failed to cancel order:', err);
      setOrderStatus({
        message: `注文のキャンセルに失敗しました: ${err instanceof Error ? err.message : '不明なエラー'}`,
        type: 'error'
      });
    } finally {
      setIsSubmitting(false);
    }
  };
  
  // タブの切り替え
  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };
  
  // 銘柄の変更
  const handleSymbolChange = (symbol: string) => {
    setSelectedSymbol(symbol);
  };
  
  // 通知の閉じる
  const handleCloseSnackbar = () => {
    setOrderStatus(null);
  };
  
  return (
    <Box sx={{ flexGrow: 1, p: 3, height: 'calc(100vh - 64px)', overflow: 'hidden' }}>
      <Box sx={{ mb: 2, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Typography variant="h4" component="h1" gutterBottom>
          トレーディングダッシュボード
        </Typography>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
          <WebSocketStatus state={wsState} latency={wsLatency} />
          <SymbolSelector selectedSymbol={selectedSymbol} onSymbolChange={handleSymbolChange} />
        </Box>
      </Box>
      
      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}
      
      <Grid container spacing={2} sx={{ height: 'calc(100% - 60px)' }}>
        {/* 左側: チャートと取引履歴 */}
        <Grid item xs={12} md={8} sx={{ height: '100%' }}>
          <Grid container spacing={2} sx={{ height: '100%' }}>
            {/* チャート */}
            <Grid item xs={12} sx={{ height: '60%' }}>
              <Paper sx={{ p: 2, height: '100%', display: 'flex', flexDirection: 'column' }}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                  <Typography variant="h6" component="h2">
                    価格チャート
                  </Typography>
                  <Box sx={{ display: 'flex', alignItems: 'center' }}>
                    <Button
                      size="small"
                      startIcon={is3DView ? <View2dIcon /> : <View3dIcon />}
                      onClick={() => setIs3DView(!is3DView)}
                      sx={{ mr: 1 }}
                    >
                      {is3DView ? '2D表示' : '3D表示'}
                    </Button>
                    <FormControlLabel
                      control={
                        <Switch
                          size="small"
                          checked={autoRefresh}
                          onChange={(e) => setAutoRefresh(e.target.checked)}
                        />
                      }
                      label="自動更新"
                    />
                    <Button
                      size="small"
                      startIcon={loading ? <CircularProgress size={20} /> : <RefreshIcon />}
                      onClick={handleRefresh}
                      disabled={loading || autoRefresh}
                    >
                      更新
                    </Button>
                  </Box>
                </Box>
                <Divider sx={{ mb: 2 }} />
                <Box sx={{ flexGrow: 1, minHeight: 0 }}>
                  <PriceChart data={ohlcv} is3D={is3DView} symbol={selectedSymbol} />
                </Box>
                {lastUpdated && (
                  <Typography variant="caption" color="text.secondary" align="right" sx={{ mt: 1 }}>
                    最終更新: {lastUpdated.toLocaleTimeString()}
                  </Typography>
                )}
              </Paper>
            </Grid>
            
            {/* 取引履歴とオーダー履歴のタブ */}
            <Grid item xs={12} sx={{ height: '40%' }}>
              <Paper sx={{ height: '100%' }}>
                <Tabs value={tabValue} onChange={handleTabChange} aria-label="trading tabs">
                  <Tab label="取引履歴" id="tab-0" aria-controls="tabpanel-0" />
                  <Tab label="注文履歴" id="tab-1" aria-controls="tabpanel-1" />
                </Tabs>
                <TabPanel value={tabValue} index={0}>
                  <TradeHistory trades={trades} />
                </TabPanel>
                <TabPanel value={tabValue} index={1}>
                  <OrderHistory symbol={selectedSymbol} />
                </TabPanel>
              </Paper>
            </Grid>
          </Grid>
        </Grid>
        
        {/* 右側: 注文板と注文フォーム */}
        <Grid item xs={12} md={4} sx={{ height: '100%' }}>
          <Grid container spacing={2} sx={{ height: '100%' }}>
            {/* 注文板 */}
            <Grid item xs={12} sx={{ height: '60%' }}>
              <Paper sx={{ p: 2, height: '100%' }}>
                <Typography variant="h6" component="h2" gutterBottom>
                  注文板
                </Typography>
                <OrderBook orderBook={orderBook} />
              </Paper>
            </Grid>
            
            {/* 注文フォーム */}
            <Grid item xs={12} sx={{ height: '40%' }}>
              <Paper sx={{ p: 2, height: '100%' }}>
                <Typography variant="h6" component="h2" gutterBottom>
                  注文
                </Typography>
                <OrderForm
                  symbol={selectedSymbol}
                  currentPrice={ohlcv.length > 0 ? ohlcv[ohlcv.length - 1].close : 0}
                  onOrderPlaced={(success, message) => {
                    setOrderStatus({
                      message,
                      type: success ? 'success' : 'error'
                    });
                    handleRefresh();
                  }}
                  onOrderCancelled={(success, message) => {
                    setOrderStatus({
                      message,
                      type: success ? 'success' : 'error'
                    });
                    handleRefresh();
                  }}
                />
              </Paper>
            </Grid>
          </Grid>
        </Grid>
      </Grid>
      
      {/* 注文ステータス通知 */}
      {orderStatus && (
        <Snackbar
          open={true}
          autoHideDuration={6000}
          onClose={handleCloseSnackbar}
          anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
        >
          <Alert onClose={handleCloseSnackbar} severity={orderStatus.type}>
            {orderStatus.message}
          </Alert>
        </Snackbar>
      )}
    </Box>
  );
};

export default Dashboard;
