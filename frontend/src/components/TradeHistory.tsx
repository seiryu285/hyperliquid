import React from 'react';
import { 
  Table, 
  TableBody, 
  TableCell, 
  TableContainer, 
  TableHead, 
  TableRow, 
  Paper, 
  Box,
  Typography
} from '@mui/material';

interface Trade {
  price: number;
  size: number;
  side: string;
  timestamp: number;
}

interface TradeHistoryProps {
  trades: Trade[];
}

const TradeHistory: React.FC<TradeHistoryProps> = ({ trades }) => {
  // 価格の表示を整形
  const formatPrice = (price: number): string => {
    return price.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  };
  
  // 数量の表示を整形
  const formatSize = (size: number): string => {
    return size.toLocaleString(undefined, { minimumFractionDigits: 4, maximumFractionDigits: 4 });
  };
  
  // タイムスタンプをフォーマット
  const formatTimestamp = (timestamp: number): string => {
    const date = new Date(timestamp);
    return date.toLocaleTimeString();
  };

  return (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      {trades.length === 0 ? (
        <Typography variant="body2" sx={{ textAlign: 'center', py: 2 }}>
          取引データがありません
        </Typography>
      ) : (
        <TableContainer component={Paper} sx={{ maxHeight: '100%', overflow: 'auto', flex: 1 }}>
          <Table size="small" stickyHeader>
            <TableHead>
              <TableRow>
                <TableCell>時間</TableCell>
                <TableCell align="right">価格</TableCell>
                <TableCell align="right">数量</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {trades.map((trade, index) => (
                <TableRow key={`trade-${index}`}>
                  <TableCell component="th" scope="row">
                    {formatTimestamp(trade.timestamp)}
                  </TableCell>
                  <TableCell 
                    align="right"
                    sx={{ 
                      color: trade.side.toLowerCase() === 'buy' ? 'success.main' : 'error.main',
                      fontWeight: 'medium'
                    }}
                  >
                    {formatPrice(trade.price)}
                  </TableCell>
                  <TableCell align="right">{formatSize(trade.size)}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      )}
    </Box>
  );
};

export default TradeHistory;
