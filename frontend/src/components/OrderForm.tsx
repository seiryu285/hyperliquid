import React, { useState, useEffect } from 'react';
import { 
  Box, 
  Button, 
  FormControl, 
  FormControlLabel, 
  Grid, 
  InputAdornment, 
  InputLabel, 
  MenuItem, 
  Radio, 
  RadioGroup, 
  Select, 
  TextField, 
  Typography,
  Paper,
  Divider,
  Snackbar,
  Alert
} from '@mui/material';
import { placeOrder, cancelOrder } from '../api/marketData';
import { OrderSide, OrderType, TimeInForce, OrderParams } from '../types/order';

interface OrderFormProps {
  symbol: string;
  currentPrice: number;
  onOrderPlaced?: (success: boolean, message: string) => void;
  onOrderCancelled?: (success: boolean, message: string) => void;
}

const OrderForm: React.FC<OrderFormProps> = ({ 
  symbol, 
  currentPrice, 
  onOrderPlaced, 
  onOrderCancelled 
}) => {
  // 状態管理
  const [side, setSide] = useState<OrderSide>(OrderSide.BUY);
  const [type, setType] = useState<OrderType>(OrderType.LIMIT);
  const [price, setPrice] = useState<number | ''>(currentPrice);
  const [quantity, setQuantity] = useState<number | ''>('');
  const [timeInForce, setTimeInForce] = useState<TimeInForce>(TimeInForce.GTC);
  const [orderId, setOrderId] = useState<string>('');
  const [isSubmitting, setIsSubmitting] = useState<boolean>(false);
  const [isCancelling, setIsCancelling] = useState<boolean>(false);
  const [notification, setNotification] = useState<{
    open: boolean;
    message: string;
    severity: 'success' | 'error' | 'info' | 'warning';
  }>({
    open: false,
    message: '',
    severity: 'info'
  });

  // 現在価格が変更されたら、指値注文の価格を更新
  useEffect(() => {
    if (type === OrderType.LIMIT && price === '') {
      setPrice(currentPrice);
    }
  }, [currentPrice, type, price]);

  // シンボルが変更されたら、フォームをリセット
  useEffect(() => {
    resetForm();
  }, [symbol]);

  // フォームリセット
  const resetForm = () => {
    setSide(OrderSide.BUY);
    setType(OrderType.LIMIT);
    setPrice(currentPrice);
    setQuantity('');
    setTimeInForce(TimeInForce.GTC);
    setOrderId('');
  };

  // 注文送信
  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    
    if (quantity === '' || (type === OrderType.LIMIT && price === '')) {
      showNotification('数量と価格を入力してください', 'error');
      return;
    }

    setIsSubmitting(true);

    try {
      const orderParams: OrderParams = {
        symbol,
        side,
        type,
        quantity: Number(quantity),
        timeInForce
      };

      if (type === OrderType.LIMIT && price !== '') {
        orderParams.price = Number(price);
      }

      const response = await placeOrder(orderParams);

      if (response.success && response.data) {
        setOrderId(response.data.orderId);
        showNotification(`注文が送信されました: ${response.data.orderId}`, 'success');
        if (onOrderPlaced) {
          onOrderPlaced(true, `${side === OrderSide.BUY ? '買い' : '売り'}注文が送信されました`);
        }
      } else {
        showNotification(`注文送信エラー: ${response.error}`, 'error');
        if (onOrderPlaced) {
          onOrderPlaced(false, `注文送信エラー: ${response.error}`);
        }
      }
    } catch (error) {
      console.error('注文送信エラー:', error);
      showNotification('注文送信中にエラーが発生しました', 'error');
      if (onOrderPlaced) {
        onOrderPlaced(false, '注文送信中にエラーが発生しました');
      }
    } finally {
      setIsSubmitting(false);
    }
  };

  // 注文キャンセル
  const handleCancel = async () => {
    if (!orderId) {
      showNotification('キャンセルする注文IDがありません', 'error');
      return;
    }

    setIsCancelling(true);

    try {
      const response = await cancelOrder(orderId);

      if (response.success && response.data) {
        showNotification(`注文がキャンセルされました: ${orderId}`, 'success');
        setOrderId('');
        if (onOrderCancelled) {
          onOrderCancelled(true, `注文がキャンセルされました`);
        }
      } else {
        showNotification(`注文キャンセルエラー: ${response.error}`, 'error');
        if (onOrderCancelled) {
          onOrderCancelled(false, `注文キャンセルエラー: ${response.error}`);
        }
      }
    } catch (error) {
      console.error('注文キャンセルエラー:', error);
      showNotification('注文キャンセル中にエラーが発生しました', 'error');
      if (onOrderCancelled) {
        onOrderCancelled(false, '注文キャンセル中にエラーが発生しました');
      }
    } finally {
      setIsCancelling(false);
    }
  };

  // 通知表示
  const showNotification = (message: string, severity: 'success' | 'error' | 'info' | 'warning') => {
    setNotification({
      open: true,
      message,
      severity
    });
  };

  // 通知閉じる
  const handleCloseNotification = () => {
    setNotification({
      ...notification,
      open: false
    });
  };

  return (
    <Paper elevation={3} sx={{ p: 2, mb: 2 }}>
      <Typography variant="h6" gutterBottom>
        注文フォーム ({symbol})
      </Typography>
      <Divider sx={{ mb: 2 }} />
      
      <Box component="form" onSubmit={handleSubmit}>
        <Grid container spacing={2}>
          {/* 注文サイド選択 */}
          <Grid item xs={12}>
            <FormControl component="fieldset">
              <RadioGroup
                row
                value={side}
                onChange={(e) => setSide(e.target.value as OrderSide)}
              >
                <FormControlLabel
                  value={OrderSide.BUY}
                  control={<Radio color="success" />}
                  label="買い"
                  sx={{ 
                    '& .MuiFormControlLabel-label': { 
                      color: 'success.main',
                      fontWeight: 'bold'
                    } 
                  }}
                />
                <FormControlLabel
                  value={OrderSide.SELL}
                  control={<Radio color="error" />}
                  label="売り"
                  sx={{ 
                    '& .MuiFormControlLabel-label': { 
                      color: 'error.main',
                      fontWeight: 'bold'
                    } 
                  }}
                />
              </RadioGroup>
            </FormControl>
          </Grid>

          {/* 注文タイプ選択 */}
          <Grid item xs={12} sm={6}>
            <FormControl fullWidth>
              <InputLabel>注文タイプ</InputLabel>
              <Select
                value={type}
                label="注文タイプ"
                onChange={(e) => {
                  const newType = e.target.value as OrderType;
                  setType(newType);
                  if (newType === OrderType.MARKET) {
                    setPrice('');
                  } else {
                    setPrice(currentPrice);
                  }
                }}
              >
                <MenuItem value={OrderType.LIMIT}>指値</MenuItem>
                <MenuItem value={OrderType.MARKET}>成行</MenuItem>
              </Select>
            </FormControl>
          </Grid>

          {/* 有効期限選択 */}
          <Grid item xs={12} sm={6}>
            <FormControl fullWidth>
              <InputLabel>有効期限</InputLabel>
              <Select
                value={timeInForce}
                label="有効期限"
                onChange={(e) => setTimeInForce(e.target.value as TimeInForce)}
                disabled={type === OrderType.MARKET}
              >
                <MenuItem value={TimeInForce.GTC}>GTC (キャンセルまで有効)</MenuItem>
                <MenuItem value={TimeInForce.IOC}>IOC (即時執行または取消)</MenuItem>
                <MenuItem value={TimeInForce.FOK}>FOK (全数量執行または取消)</MenuItem>
              </Select>
            </FormControl>
          </Grid>

          {/* 価格入力 */}
          {type === OrderType.LIMIT && (
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                label="価格"
                type="number"
                value={price}
                onChange={(e) => setPrice(e.target.value === '' ? '' : Number(e.target.value))}
                InputProps={{
                  endAdornment: <InputAdornment position="end">USD</InputAdornment>,
                }}
                inputProps={{
                  step: 0.1,
                  min: 0.1,
                }}
              />
            </Grid>
          )}

          {/* 数量入力 */}
          <Grid item xs={12} sm={type === OrderType.LIMIT ? 6 : 12}>
            <TextField
              fullWidth
              label="数量"
              type="number"
              value={quantity}
              onChange={(e) => setQuantity(e.target.value === '' ? '' : Number(e.target.value))}
              InputProps={{
                endAdornment: <InputAdornment position="end">{symbol.split('-')[0]}</InputAdornment>,
              }}
              inputProps={{
                step: 0.01,
                min: 0.01,
              }}
            />
          </Grid>

          {/* 注文ボタン */}
          <Grid item xs={12} sm={6}>
            <Button
              fullWidth
              type="submit"
              variant="contained"
              color={side === OrderSide.BUY ? "success" : "error"}
              disabled={isSubmitting}
              sx={{ py: 1 }}
            >
              {isSubmitting ? '送信中...' : `${side === OrderSide.BUY ? '買い' : '売り'}${type === OrderType.LIMIT ? '指値' : '成行'}`}
            </Button>
          </Grid>

          {/* キャンセルボタン */}
          <Grid item xs={12} sm={6}>
            <Button
              fullWidth
              variant="outlined"
              color="warning"
              onClick={handleCancel}
              disabled={!orderId || isCancelling}
              sx={{ py: 1 }}
            >
              {isCancelling ? 'キャンセル中...' : '注文キャンセル'}
            </Button>
          </Grid>
        </Grid>
      </Box>

      {/* 通知 */}
      <Snackbar
        open={notification.open}
        autoHideDuration={6000}
        onClose={handleCloseNotification}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
      >
        <Alert onClose={handleCloseNotification} severity={notification.severity}>
          {notification.message}
        </Alert>
      </Snackbar>
    </Paper>
  );
};

export default OrderForm;
