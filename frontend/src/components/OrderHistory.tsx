import React, { useState, useEffect } from 'react';
import {
  Box,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Typography,
  Chip,
  IconButton,
  Tooltip,
  CircularProgress,
  TablePagination
} from '@mui/material';
import RefreshIcon from '@mui/icons-material/Refresh';
import { Order, OrderStatus, OrderSide } from '../types/order';
import { getOrderHistory } from '../api/marketData';

interface OrderHistoryProps {
  symbol: string;
}

const OrderHistory: React.FC<OrderHistoryProps> = ({ symbol }) => {
  const [orders, setOrders] = useState<Order[]>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const [page, setPage] = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(5);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);

  const fetchOrderHistory = async () => {
    setLoading(true);
    try {
      const history = await getOrderHistory(symbol);
      setOrders(history);
      setLastUpdated(new Date());
    } catch (error) {
      console.error('Failed to fetch order history:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchOrderHistory();
    // 1分ごとに自動更新
    const interval = setInterval(fetchOrderHistory, 60000);
    return () => clearInterval(interval);
  }, [symbol]);

  const handleChangePage = (event: unknown, newPage: number) => {
    setPage(newPage);
  };

  const handleChangeRowsPerPage = (event: React.ChangeEvent<HTMLInputElement>) => {
    setRowsPerPage(parseInt(event.target.value, 10));
    setPage(0);
  };

  const getStatusChip = (status: OrderStatus) => {
    let color: 'success' | 'error' | 'warning' | 'default' = 'default';
    let label = status;

    switch (status) {
      case OrderStatus.FILLED:
        color = 'success';
        label = '約定済み';
        break;
      case OrderStatus.PARTIALLY_FILLED:
        color = 'warning';
        label = '一部約定';
        break;
      case OrderStatus.OPEN:
        color = 'default';
        label = '注文中';
        break;
      case OrderStatus.CANCELED:
        color = 'error';
        label = 'キャンセル';
        break;
      case OrderStatus.REJECTED:
        color = 'error';
        label = '拒否';
        break;
    }

    return <Chip size="small" color={color} label={label} />;
  };

  const getSideChip = (side: OrderSide) => {
    return (
      <Chip
        size="small"
        color={side === OrderSide.BUY ? 'success' : 'error'}
        label={side === OrderSide.BUY ? '買い' : '売り'}
      />
    );
  };

  const formatDate = (timestamp: number) => {
    const date = new Date(timestamp);
    return date.toLocaleString('ja-JP');
  };

  return (
    <Paper sx={{ p: 2, height: '100%', display: 'flex', flexDirection: 'column' }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Typography variant="h6" component="h2">
          注文履歴
        </Typography>
        <Box sx={{ display: 'flex', alignItems: 'center' }}>
          {lastUpdated && (
            <Typography variant="caption" color="text.secondary" sx={{ mr: 1 }}>
              最終更新: {lastUpdated.toLocaleTimeString()}
            </Typography>
          )}
          <Tooltip title="更新">
            <IconButton size="small" onClick={fetchOrderHistory} disabled={loading}>
              {loading ? <CircularProgress size={20} /> : <RefreshIcon />}
            </IconButton>
          </Tooltip>
        </Box>
      </Box>

      <TableContainer sx={{ flexGrow: 1 }}>
        <Table size="small" stickyHeader>
          <TableHead>
            <TableRow>
              <TableCell>日時</TableCell>
              <TableCell>銘柄</TableCell>
              <TableCell>タイプ</TableCell>
              <TableCell>価格</TableCell>
              <TableCell>数量</TableCell>
              <TableCell>状態</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {orders.length === 0 ? (
              <TableRow>
                <TableCell colSpan={6} align="center">
                  {loading ? '読み込み中...' : '注文履歴がありません'}
                </TableCell>
              </TableRow>
            ) : (
              orders
                .slice(page * rowsPerPage, page * rowsPerPage + rowsPerPage)
                .map((order) => (
                  <TableRow key={order.id} hover>
                    <TableCell>{formatDate(order.timestamp)}</TableCell>
                    <TableCell>{order.symbol}</TableCell>
                    <TableCell>{getSideChip(order.side)}</TableCell>
                    <TableCell>{order.price.toFixed(2)}</TableCell>
                    <TableCell>{order.quantity.toFixed(4)}</TableCell>
                    <TableCell>{getStatusChip(order.status)}</TableCell>
                  </TableRow>
                ))
            )}
          </TableBody>
        </Table>
      </TableContainer>

      <TablePagination
        rowsPerPageOptions={[5, 10, 25]}
        component="div"
        count={orders.length}
        rowsPerPage={rowsPerPage}
        page={page}
        onPageChange={handleChangePage}
        onRowsPerPageChange={handleChangeRowsPerPage}
        labelRowsPerPage="表示件数:"
      />
    </Paper>
  );
};

export default OrderHistory;
