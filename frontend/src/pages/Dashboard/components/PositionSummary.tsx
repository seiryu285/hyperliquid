import React from 'react';
import { Box, Card, CardContent, Divider, Grid, Typography } from '@mui/material';

interface Position {
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

interface PositionSummaryProps {
  data: Position[];
}

const PositionSummary: React.FC<PositionSummaryProps> = ({ data }) => {
  // Find ETH-PERP position
  const ethPosition = data.find(position => position.symbol === 'ETH-PERP');
  
  if (!ethPosition) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" height="100%">
        <Typography>アクティブなポジションがありません</Typography>
      </Box>
    );
  }
  
  const {
    size,
    entryPrice,
    markPrice,
    liquidationPrice,
    margin,
    leverage,
    unrealizedPnl,
    unrealizedPnlPercent
  } = ethPosition;
  
  const isLong = size > 0;
  
  return (
    <Card variant="outlined">
      <CardContent>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <Typography variant="h6">ETH-PERP</Typography>
          <Typography 
            variant="subtitle1" 
            sx={{ 
              px: 1.5, 
              py: 0.5, 
              borderRadius: 1, 
              backgroundColor: isLong ? 'rgba(38, 166, 154, 0.1)' : 'rgba(239, 83, 80, 0.1)',
              color: isLong ? '#26a69a' : '#ef5350',
              fontWeight: 'bold'
            }}
          >
            {isLong ? 'ロング' : 'ショート'} {Math.abs(size).toFixed(4)}
          </Typography>
        </Box>
        
        <Divider sx={{ my: 1 }} />
        
        <Grid container spacing={2}>
          <Grid item xs={6}>
            <Typography variant="body2" color="text.secondary">
              エントリー価格
            </Typography>
            <Typography variant="body1" fontWeight="medium">
              {entryPrice.toFixed(2)}
            </Typography>
          </Grid>
          
          <Grid item xs={6}>
            <Typography variant="body2" color="text.secondary">
              マーク価格
            </Typography>
            <Typography variant="body1" fontWeight="medium">
              {markPrice.toFixed(2)}
            </Typography>
          </Grid>
          
          <Grid item xs={6}>
            <Typography variant="body2" color="text.secondary">
              清算価格
            </Typography>
            <Typography variant="body1" fontWeight="medium" color={
              isLong 
                ? (liquidationPrice / markPrice > 0.9 ? 'error.main' : 'text.primary')
                : (liquidationPrice / markPrice < 1.1 ? 'error.main' : 'text.primary')
            }>
              {liquidationPrice.toFixed(2)}
            </Typography>
          </Grid>
          
          <Grid item xs={6}>
            <Typography variant="body2" color="text.secondary">
              レバレッジ
            </Typography>
            <Typography variant="body1" fontWeight="medium">
              {leverage.toFixed(1)}x
            </Typography>
          </Grid>
          
          <Grid item xs={6}>
            <Typography variant="body2" color="text.secondary">
              証拠金
            </Typography>
            <Typography variant="body1" fontWeight="medium">
              {margin.toFixed(2)} USD
            </Typography>
          </Grid>
          
          <Grid item xs={6}>
            <Typography variant="body2" color="text.secondary">
              未実現損益
            </Typography>
            <Box sx={{ display: 'flex', alignItems: 'baseline' }}>
              <Typography 
                variant="body1" 
                fontWeight="medium" 
                sx={{ color: unrealizedPnl >= 0 ? '#26a69a' : '#ef5350' }}
              >
                {unrealizedPnl.toFixed(2)} USD
              </Typography>
              <Typography 
                variant="caption" 
                sx={{ 
                  ml: 0.5, 
                  color: unrealizedPnl >= 0 ? '#26a69a' : '#ef5350' 
                }}
              >
                ({unrealizedPnl >= 0 ? '+' : ''}{unrealizedPnlPercent.toFixed(2)}%)
              </Typography>
            </Box>
          </Grid>
        </Grid>
        
        <Divider sx={{ my: 1 }} />
        
        <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 1 }}>
          <Typography variant="body2" color="text.secondary">
            リスク評価
          </Typography>
          <Typography 
            variant="body2" 
            sx={{ 
              fontWeight: 'medium',
              color: 
                Math.abs(liquidationPrice - markPrice) / markPrice < 0.1 
                  ? 'error.main' 
                  : Math.abs(liquidationPrice - markPrice) / markPrice < 0.2 
                    ? 'warning.main' 
                    : 'success.main'
            }}
          >
            {Math.abs(liquidationPrice - markPrice) / markPrice < 0.1 
              ? '高リスク' 
              : Math.abs(liquidationPrice - markPrice) / markPrice < 0.2 
                ? '中リスク' 
                : '低リスク'}
          </Typography>
        </Box>
      </CardContent>
    </Card>
  );
};

export default PositionSummary;
