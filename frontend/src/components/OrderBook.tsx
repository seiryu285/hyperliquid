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
  Divider,
  useTheme,
  alpha
} from '@mui/material';

interface OrderBookProps {
  orderBook: {
    bids: [number, number][];
    asks: [number, number][];
  };
}

const OrderBook: React.FC<OrderBookProps> = ({ orderBook }) => {
  const { bids, asks } = orderBook;
  const theme = useTheme();
  
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

  // 売り注文と買い注文の色
  const askColor = theme.palette.error.main;
  const bidColor = theme.palette.success.main;

  return (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <TableContainer 
        component={Paper} 
        sx={{ 
          maxHeight: '100%', 
          overflow: 'auto', 
          flex: 1,
          bgcolor: 'background.paper',
          borderRadius: 1,
          boxShadow: 'none',
          '&::-webkit-scrollbar': {
            width: '8px',
            height: '8px',
          },
          '&::-webkit-scrollbar-track': {
            backgroundColor: alpha(theme.palette.text.primary, 0.05),
          },
          '&::-webkit-scrollbar-thumb': {
            backgroundColor: alpha(theme.palette.text.primary, 0.2),
            borderRadius: '4px',
            '&:hover': {
              backgroundColor: alpha(theme.palette.text.primary, 0.3),
            },
          },
        }}
      >
        <Table size="small" stickyHeader>
          <TableHead>
            <TableRow>
              <TableCell 
                align="right" 
                sx={{ 
                  fontWeight: 'bold', 
                  py: 1.5,
                  bgcolor: alpha(theme.palette.background.paper, 0.9),
                }}
              >
                価格
              </TableCell>
              <TableCell 
                align="right"
                sx={{ 
                  fontWeight: 'bold', 
                  py: 1.5,
                  bgcolor: alpha(theme.palette.background.paper, 0.9),
                }}
              >
                数量
              </TableCell>
              <TableCell 
                align="right"
                sx={{ 
                  fontWeight: 'bold', 
                  py: 1.5,
                  bgcolor: alpha(theme.palette.background.paper, 0.9),
                }}
              >
                合計
              </TableCell>
              <TableCell 
                align="left" 
                sx={{ 
                  width: '40%', 
                  fontWeight: 'bold', 
                  py: 1.5,
                  bgcolor: alpha(theme.palette.background.paper, 0.9),
                }}
              >
                深さ
              </TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {/* 売り注文（降順） */}
            {asks.slice().reverse().map((ask, index) => {
              const depth = calculateDepth(asks, asks.length - 1 - index);
              const depthPercentage = (depth / maxDepth) * 100;
              
              return (
                <TableRow 
                  key={`ask-${index}`}
                  hover
                  sx={{ 
                    '&:hover': { 
                      bgcolor: alpha(askColor, 0.05),
                    },
                  }}
                >
                  <TableCell 
                    align="right" 
                    sx={{ 
                      color: askColor,
                      fontWeight: 500,
                      py: 1,
                    }}
                  >
                    {formatPrice(ask[0])}
                  </TableCell>
                  <TableCell align="right" sx={{ py: 1 }}>{formatSize(ask[1])}</TableCell>
                  <TableCell align="right" sx={{ py: 1 }}>{formatSize(depth)}</TableCell>
                  <TableCell align="left" sx={{ py: 1 }}>
                    <Box
                      sx={{
                        width: `${depthPercentage}%`,
                        height: '80%',
                        backgroundColor: alpha(askColor, 0.2),
                        borderRadius: 1
                      }}
                    />
                  </TableCell>
                </TableRow>
              );
            })}
            
            {/* スプレッド表示 */}
            {bids.length > 0 && asks.length > 0 && (
              <TableRow sx={{ bgcolor: alpha(theme.palette.divider, 0.05) }}>
                <TableCell colSpan={4} align="center" sx={{ py: 1 }}>
                  <Typography variant="body2" color="text.secondary">
                    スプレッド: {formatPrice(asks[0][0] - bids[0][0])} ({((asks[0][0] / bids[0][0] - 1) * 100).toFixed(2)}%)
                  </Typography>
                </TableCell>
              </TableRow>
            )}
            
            {/* 買い注文（降順） */}
            {bids.map((bid, index) => {
              const depth = calculateDepth(bids, index);
              const depthPercentage = (depth / maxDepth) * 100;
              
              return (
                <TableRow 
                  key={`bid-${index}`}
                  hover
                  sx={{ 
                    '&:hover': { 
                      bgcolor: alpha(bidColor, 0.05),
                    },
                  }}
                >
                  <TableCell 
                    align="right" 
                    sx={{ 
                      color: bidColor,
                      fontWeight: 500,
                      py: 1,
                    }}
                  >
                    {formatPrice(bid[0])}
                  </TableCell>
                  <TableCell align="right" sx={{ py: 1 }}>{formatSize(bid[1])}</TableCell>
                  <TableCell align="right" sx={{ py: 1 }}>{formatSize(depth)}</TableCell>
                  <TableCell align="left" sx={{ py: 1 }}>
                    <Box
                      sx={{
                        width: `${depthPercentage}%`,
                        height: '80%',
                        backgroundColor: alpha(bidColor, 0.2),
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
