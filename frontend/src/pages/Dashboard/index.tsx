import React, { useEffect, useState } from 'react';
import { Container, Grid, Typography, Paper, Box, CircularProgress } from '@mui/material';
import { fetchMarketData, fetchOHLCVData } from '../../services/api';
import PriceChart from './components/PriceChart';
import OrderBook from './components/OrderBook';
import TradesTable from './components/TradesTable';
import TechnicalIndicators from './components/TechnicalIndicators';
import PositionSummary from './components/PositionSummary';

const Dashboard: React.FC = () => {
  const [loading, setLoading] = useState<boolean>(true);
  const [marketData, setMarketData] = useState<any>(null);
  const [ohlcvData, setOhlcvData] = useState<any[]>([]);
  const [error, setError] = useState<string | null>(null);
  
  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        
        // Fetch market data for ETH-PERP
        const marketDataResponse = await fetchMarketData('ETH-PERP');
        setMarketData(marketDataResponse);
        
        // Fetch OHLCV data for ETH-PERP
        const ohlcvResponse = await fetchOHLCVData('ETH-PERP', '1h', 100);
        setOhlcvData(ohlcvResponse);
        
        setLoading(false);
      } catch (err) {
        console.error('Error fetching data:', err);
        setError('データの取得中にエラーが発生しました。');
        setLoading(false);
      }
    };
    
    fetchData();
    
    // Set up polling interval to refresh data every 30 seconds
    const intervalId = setInterval(() => {
      fetchData();
    }, 30000);
    
    // Clean up interval on component unmount
    return () => clearInterval(intervalId);
  }, []);
  
  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="100vh">
        <CircularProgress />
        <Typography variant="h6" sx={{ ml: 2 }}>データを読み込み中...</Typography>
      </Box>
    );
  }
  
  if (error) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="100vh">
        <Typography variant="h6" color="error">{error}</Typography>
      </Box>
    );
  }
  
  return (
    <Container maxWidth="xl">
      <Typography variant="h4" component="h1" gutterBottom sx={{ mt: 3 }}>
        HyperLiquid ETH-PERP ダッシュボード
      </Typography>
      
      <Grid container spacing={3}>
        {/* Price Chart */}
        <Grid item xs={12} lg={8}>
          <Paper elevation={2} sx={{ p: 2, height: '400px' }}>
            <Typography variant="h6" gutterBottom>価格チャート</Typography>
            {ohlcvData.length > 0 ? (
              <PriceChart data={ohlcvData} />
            ) : (
              <Box display="flex" justifyContent="center" alignItems="center" height="100%">
                <Typography>データがありません</Typography>
              </Box>
            )}
          </Paper>
        </Grid>
        
        {/* Order Book */}
        <Grid item xs={12} lg={4}>
          <Paper elevation={2} sx={{ p: 2, height: '400px' }}>
            <Typography variant="h6" gutterBottom>オーダーブック</Typography>
            {marketData?.orderBook ? (
              <OrderBook data={marketData.orderBook} />
            ) : (
              <Box display="flex" justifyContent="center" alignItems="center" height="100%">
                <Typography>データがありません</Typography>
              </Box>
            )}
          </Paper>
        </Grid>
        
        {/* Technical Indicators */}
        <Grid item xs={12} md={6}>
          <Paper elevation={2} sx={{ p: 2, minHeight: '300px' }}>
            <Typography variant="h6" gutterBottom>テクニカル指標</Typography>
            {ohlcvData.length > 0 ? (
              <TechnicalIndicators data={ohlcvData} />
            ) : (
              <Box display="flex" justifyContent="center" alignItems="center" height="100%">
                <Typography>データがありません</Typography>
              </Box>
            )}
          </Paper>
        </Grid>
        
        {/* Position Summary */}
        <Grid item xs={12} md={6}>
          <Paper elevation={2} sx={{ p: 2, minHeight: '300px' }}>
            <Typography variant="h6" gutterBottom>ポジション概要</Typography>
            {marketData?.positions ? (
              <PositionSummary data={marketData.positions} />
            ) : (
              <Box display="flex" justifyContent="center" alignItems="center" height="100%">
                <Typography>データがありません</Typography>
              </Box>
            )}
          </Paper>
        </Grid>
        
        {/* Recent Trades */}
        <Grid item xs={12}>
          <Paper elevation={2} sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>最近の取引</Typography>
            {marketData?.trades ? (
              <TradesTable data={marketData.trades} />
            ) : (
              <Box display="flex" justifyContent="center" alignItems="center" height="100px">
                <Typography>データがありません</Typography>
              </Box>
            )}
          </Paper>
        </Grid>
      </Grid>
    </Container>
  );
};

export default Dashboard;
