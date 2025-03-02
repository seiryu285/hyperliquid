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
  Typography,
  Divider
} from '@mui/material';

interface OrderBookProps {
  orderBook: {
    bids: [number, number][];
    asks: [number, number][];
  };
}

const OrderBook: React.FC<OrderBookProps> = ({ orderBook }) => {
  const { bids, asks } = orderBook;
  
  // 注文板の深さを計算
  const calculateDepth = (orders: [number, number][], index: number): number => {
    let depth = 0;
    for (let i = 0; i <= index; i++) {
      depth += orders[i][1];
    }
    return depth;
  };

  // 最大の深さを取得して、バーの幅を正規化
  const getMaxDepth = (): number => {
    if (bids.length === 0 && asks.length === 0) return 1;
    
    let maxBidDepth = 0;
    let maxAskDepth = 0;
    
    if (bids.length > 0) {
      maxBidDepth = calculateDepth(bids, bids.length - 1);
    }
    
    if (asks.length > 0) {
      maxAskDepth = calculateDepth(asks, asks.length - 1);
    }
    
    return Math.max(maxBidDepth, maxAskDepth);
  };
  
  const maxDepth = getMaxDepth();
  
  // 価格の表示を整形
  const formatPrice = (price: number): string => {
    return price.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  };
  
  // 数量の表示を整形
  const formatSize = (size: number): string => {
    return size.toLocaleString(undefined, { minimumFractionDigits: 4, maximumFractionDigits: 4 });
  };

  return (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <TableContainer component={Paper} sx={{ maxHeight: '100%', overflow: 'auto', flex: 1 }}>
        <Table size="small" stickyHeader>
          <TableHead>
            <TableRow>
              <TableCell align="right">価格</TableCell>
              <TableCell align="right">数量</TableCell>
              <TableCell align="right">合計</TableCell>
              <TableCell align="left" sx={{ width: '40%' }}>深さ</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {/* 売り注文（降順） */}
            {asks.slice().reverse().map((ask, index) => {
              const depth = calculateDepth(asks, asks.length - 1 - index);
              const depthPercentage = (depth / maxDepth) * 100;
              
              return (
                <TableRow key={`ask-${index}`}>
                  <TableCell align="right" sx={{ color: 'error.main' }}>
                    {formatPrice(ask[0])}
                  </TableCell>
                  <TableCell align="right">{formatSize(ask[1])}</TableCell>
                  <TableCell align="right">{formatSize(depth)}</TableCell>
                  <TableCell align="left">
                    <Box
                      sx={{
                        width: `${depthPercentage}%`,
                        height: '80%',
                        backgroundColor: 'rgba(244, 67, 54, 0.2)',
                        borderRadius: 1
                      }}
                    />
                  </TableCell>
                </TableRow>
              );
            })}
            
            {/* スプレッド表示 */}
            {bids.length > 0 && asks.length > 0 && (
              <TableRow>
                <TableCell colSpan={4} sx={{ py: 0.5 }}>
                  <Divider />
                  <Typography variant="caption" sx={{ display: 'block', textAlign: 'center' }}>
                    スプレッド: {formatPrice(asks[0][0] - bids[0][0])} ({((asks[0][0] / bids[0][0] - 1) * 100).toFixed(2)}%)
                  </Typography>
                  <Divider />
                </TableCell>
              </TableRow>
            )}
            
            {/* 買い注文（降順） */}
            {bids.map((bid, index) => {
              const depth = calculateDepth(bids, index);
              const depthPercentage = (depth / maxDepth) * 100;
              
              return (
                <TableRow key={`bid-${index}`}>
                  <TableCell align="right" sx={{ color: 'success.main' }}>
                    {formatPrice(bid[0])}
                  </TableCell>
                  <TableCell align="right">{formatSize(bid[1])}</TableCell>
                  <TableCell align="right">{formatSize(depth)}</TableCell>
                  <TableCell align="left">
                    <Box
                      sx={{
                        width: `${depthPercentage}%`,
                        height: '80%',
                        backgroundColor: 'rgba(76, 175, 80, 0.2)',
                        borderRadius: 1
                      }}
                    />
                  </TableCell>
                </TableRow>
              );
            })}
          </TableBody>
        </Table>
      </TableContainer>
    </Box>
  );
};

export default OrderBook;
