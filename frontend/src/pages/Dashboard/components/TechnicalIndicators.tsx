import React, { useMemo } from 'react';
import { Grid, Paper, Typography, Box, LinearProgress } from '@mui/material';

interface TechnicalIndicatorsProps {
  data: any[];
}

const TechnicalIndicators: React.FC<TechnicalIndicatorsProps> = ({ data }) => {
  // Calculate technical indicators from the most recent data
  const indicators = useMemo(() => {
    if (!data || data.length === 0) return null;
    
    // Get the most recent data point
    const latestData = data[data.length - 1];
    
    // Calculate RSI
    const rsi = latestData.rsi_14 || 50; // Default to 50 if not available
    
    // Calculate MACD
    const macdLine = latestData.macd_line || 0;
    const macdSignal = latestData.macd_signal || 0;
    const macdHistogram = latestData.macd_histogram || 0;
    
    // Calculate Bollinger Bands
    const bbMiddle = latestData.bb_middle || latestData.close;
    const bbUpper = latestData.bb_upper || latestData.close * 1.02;
    const bbLower = latestData.bb_lower || latestData.close * 0.98;
    
    // Calculate price position within Bollinger Bands (0-100%)
    const bbRange = bbUpper - bbLower;
    const pricePosition = bbRange !== 0 
      ? ((latestData.close - bbLower) / bbRange) * 100 
      : 50;
    
    // Calculate moving averages
    const sma7 = latestData.sma_7 || latestData.close;
    const sma25 = latestData.sma_25 || latestData.close;
    const ema9 = latestData.ema_9 || latestData.close;
    const ema21 = latestData.ema_21 || latestData.close;
    
    // Calculate ATR
    const atr = latestData.atr_14 || 0;
    
    // Calculate ATR as percentage of price
    const atrPercent = (atr / latestData.close) * 100;
    
    return {
      rsi,
      macdLine,
      macdSignal,
      macdHistogram,
      bbMiddle,
      bbUpper,
      bbLower,
      pricePosition,
      sma7,
      sma25,
      ema9,
      ema21,
      atr,
      atrPercent,
      close: latestData.close,
    };
  }, [data]);
  
  if (!indicators) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" height="100%">
        <Typography>データがありません</Typography>
      </Box>
    );
  }
  
  return (
    <Grid container spacing={2}>
      {/* RSI */}
      <Grid item xs={12} sm={6}>
        <Paper elevation={1} sx={{ p: 1.5 }}>
          <Typography variant="subtitle2" gutterBottom>
            RSI (14)
          </Typography>
          <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
            <Typography variant="h6" sx={{ mr: 1 }}>
              {indicators.rsi.toFixed(2)}
            </Typography>
            <Typography 
              variant="body2" 
              sx={{ 
                color: indicators.rsi > 70 
                  ? '#ef5350' 
                  : indicators.rsi < 30 
                    ? '#26a69a' 
                    : 'text.secondary' 
              }}
            >
              {indicators.rsi > 70 
                ? '買われすぎ' 
                : indicators.rsi < 30 
                  ? '売られすぎ' 
                  : '中立'}
            </Typography>
          </Box>
          <LinearProgress 
            variant="determinate" 
            value={indicators.rsi} 
            sx={{ 
              height: 8, 
              borderRadius: 1,
              '& .MuiLinearProgress-bar': {
                backgroundColor: indicators.rsi > 70 
                  ? '#ef5350' 
                  : indicators.rsi < 30 
                    ? '#26a69a' 
                    : '#1976d2'
              }
            }} 
          />
          <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 0.5 }}>
            <Typography variant="caption">0</Typography>
            <Typography variant="caption">50</Typography>
            <Typography variant="caption">100</Typography>
          </Box>
        </Paper>
      </Grid>
      
      {/* MACD */}
      <Grid item xs={12} sm={6}>
        <Paper elevation={1} sx={{ p: 1.5 }}>
          <Typography variant="subtitle2" gutterBottom>
            MACD
          </Typography>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
            <Typography variant="body2">
              MACD: <span style={{ fontWeight: 'bold' }}>{indicators.macdLine.toFixed(2)}</span>
            </Typography>
            <Typography variant="body2">
              Signal: <span style={{ fontWeight: 'bold' }}>{indicators.macdSignal.toFixed(2)}</span>
            </Typography>
          </Box>
          <Box sx={{ display: 'flex', alignItems: 'center' }}>
            <Typography variant="h6" sx={{ mr: 1 }}>
              {indicators.macdHistogram.toFixed(2)}
            </Typography>
            <Typography 
              variant="body2" 
              sx={{ 
                color: indicators.macdHistogram > 0 
                  ? '#26a69a' 
                  : '#ef5350'
              }}
            >
              {indicators.macdHistogram > 0 
                ? (indicators.macdHistogram > indicators.macdHistogram * 1.1 ? '強い上昇トレンド' : '上昇トレンド')
                : (indicators.macdHistogram < indicators.macdHistogram * 1.1 ? '強い下降トレンド' : '下降トレンド')}
            </Typography>
          </Box>
        </Paper>
      </Grid>
      
      {/* Bollinger Bands */}
      <Grid item xs={12} sm={6}>
        <Paper elevation={1} sx={{ p: 1.5 }}>
          <Typography variant="subtitle2" gutterBottom>
            ボリンジャーバンド
          </Typography>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
            <Typography variant="body2">
              上限: <span style={{ fontWeight: 'bold' }}>{indicators.bbUpper.toFixed(2)}</span>
            </Typography>
            <Typography variant="body2">
              中央: <span style={{ fontWeight: 'bold' }}>{indicators.bbMiddle.toFixed(2)}</span>
            </Typography>
            <Typography variant="body2">
              下限: <span style={{ fontWeight: 'bold' }}>{indicators.bbLower.toFixed(2)}</span>
            </Typography>
          </Box>
          <Box sx={{ position: 'relative', height: 24, bgcolor: '#f5f5f5', borderRadius: 1, mb: 0.5 }}>
            <Box 
              sx={{ 
                position: 'absolute', 
                left: `${indicators.pricePosition}%`, 
                top: '50%', 
                transform: 'translate(-50%, -50%)',
                width: 12,
                height: 12,
                borderRadius: '50%',
                bgcolor: indicators.pricePosition > 80 
                  ? '#ef5350' 
                  : indicators.pricePosition < 20 
                    ? '#26a69a' 
                    : '#1976d2',
                border: '2px solid white',
                zIndex: 1,
              }} 
            />
          </Box>
          <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
            <Typography variant="caption">下限</Typography>
            <Typography variant="caption">中央</Typography>
            <Typography variant="caption">上限</Typography>
          </Box>
        </Paper>
      </Grid>
      
      {/* ATR */}
      <Grid item xs={12} sm={6}>
        <Paper elevation={1} sx={{ p: 1.5 }}>
          <Typography variant="subtitle2" gutterBottom>
            ATR (14)
          </Typography>
          <Typography variant="h6">
            {indicators.atr.toFixed(2)} ({indicators.atrPercent.toFixed(2)}%)
          </Typography>
          <Typography variant="body2" sx={{ color: 'text.secondary' }}>
            {indicators.atrPercent > 3 
              ? '高いボラティリティ' 
              : indicators.atrPercent < 1 
                ? '低いボラティリティ' 
                : '中程度のボラティリティ'}
          </Typography>
        </Paper>
      </Grid>
      
      {/* Moving Averages */}
      <Grid item xs={12}>
        <Paper elevation={1} sx={{ p: 1.5 }}>
          <Typography variant="subtitle2" gutterBottom>
            移動平均線
          </Typography>
          <Grid container spacing={1}>
            <Grid item xs={6} sm={3}>
              <Typography variant="body2">
                SMA (7): <span style={{ fontWeight: 'bold' }}>{indicators.sma7.toFixed(2)}</span>
              </Typography>
              <Typography 
                variant="caption" 
                sx={{ 
                  color: indicators.close > indicators.sma7 
                    ? '#26a69a' 
                    : '#ef5350'
                }}
              >
                {((indicators.close / indicators.sma7 - 1) * 100).toFixed(2)}%
              </Typography>
            </Grid>
            <Grid item xs={6} sm={3}>
              <Typography variant="body2">
                SMA (25): <span style={{ fontWeight: 'bold' }}>{indicators.sma25.toFixed(2)}</span>
              </Typography>
              <Typography 
                variant="caption" 
                sx={{ 
                  color: indicators.close > indicators.sma25 
                    ? '#26a69a' 
                    : '#ef5350'
                }}
              >
                {((indicators.close / indicators.sma25 - 1) * 100).toFixed(2)}%
              </Typography>
            </Grid>
            <Grid item xs={6} sm={3}>
              <Typography variant="body2">
                EMA (9): <span style={{ fontWeight: 'bold' }}>{indicators.ema9.toFixed(2)}</span>
              </Typography>
              <Typography 
                variant="caption" 
                sx={{ 
                  color: indicators.close > indicators.ema9 
                    ? '#26a69a' 
                    : '#ef5350'
                }}
              >
                {((indicators.close / indicators.ema9 - 1) * 100).toFixed(2)}%
              </Typography>
            </Grid>
            <Grid item xs={6} sm={3}>
              <Typography variant="body2">
                EMA (21): <span style={{ fontWeight: 'bold' }}>{indicators.ema21.toFixed(2)}</span>
              </Typography>
              <Typography 
                variant="caption" 
                sx={{ 
                  color: indicators.close > indicators.ema21 
                    ? '#26a69a' 
                    : '#ef5350'
                }}
              >
                {((indicators.close / indicators.ema21 - 1) * 100).toFixed(2)}%
              </Typography>
            </Grid>
          </Grid>
        </Paper>
      </Grid>
    </Grid>
  );
};

export default TechnicalIndicators;
