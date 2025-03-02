import React, { useRef, useMemo } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Text, useHelper } from '@react-three/drei';
import * as THREE from 'three';
import { Box, Typography, useTheme } from '@mui/material';

interface OrderBookEntry {
  price: number;
  size: number;
  total: number;
  sizePercent: number;
}

interface OrderBookVisualizationProps {
  bids: OrderBookEntry[];
  asks: OrderBookEntry[];
  spread: number;
  maxSize: number;
}

// 注文バーコンポーネント
const OrderBar: React.FC<{
  entry: OrderBookEntry;
  position: [number, number, number];
  color: string;
  maxSize: number;
  type: 'bid' | 'ask';
}> = ({ entry, position, color, maxSize, type }) => {
  const mesh = useRef<THREE.Mesh>(null);
  const height = (entry.size / maxSize) * 5; // 高さをサイズに比例させる
  
  // バーの位置を調整
  const barPosition: [number, number, number] = [
    position[0],
    position[1] + height / 2, // 高さの半分だけY軸方向に移動
    position[2]
  ];
  
  // 価格ラベルの位置を調整
  const labelPosition: [number, number, number] = [
    position[0] + (type === 'bid' ? -2 : 2),
    position[1],
    position[2]
  ];
  
  // サイズラベルの位置を調整
  const sizePosition: [number, number, number] = [
    position[0],
    position[1] + height + 0.2,
    position[2]
  ];
  
  return (
    <group>
      {/* 注文バー */}
      <mesh ref={mesh} position={barPosition}>
        <boxGeometry args={[1, height, 1]} />
        <meshStandardMaterial color={color} transparent opacity={0.7} />
      </mesh>
      
      {/* 価格ラベル */}
      <Text
        position={labelPosition}
        fontSize={0.3}
        color="#ffffff"
        anchorX={type === 'bid' ? 'right' : 'left'}
        anchorY="middle"
      >
        {entry.price.toFixed(2)}
      </Text>
      
      {/* サイズラベル */}
      <Text
        position={sizePosition}
        fontSize={0.25}
        color="#ffffff"
        anchorX="center"
        anchorY="bottom"
      >
        {entry.size.toFixed(2)}
      </Text>
    </group>
  );
};

// スプレッド表示コンポーネント
const SpreadIndicator: React.FC<{ spread: number; position: [number, number, number] }> = ({ spread, position }) => {
  return (
    <Text
      position={position}
      fontSize={0.4}
      color="#ffff00"
      anchorX="center"
      anchorY="middle"
    >
      Spread: {spread.toFixed(2)}
    </Text>
  );
};

// メインの3Dシーン
const OrderBookScene: React.FC<OrderBookVisualizationProps> = ({ bids, asks, spread, maxSize }) => {
  const directionalLightRef = useRef<THREE.DirectionalLight>(null);
  
  // 光源のヘルパーを表示（開発時のみ）
  // useHelper(directionalLightRef, THREE.DirectionalLightHelper, 1, 'red');
  
  return (
    <>
      {/* 環境光 */}
      <ambientLight intensity={0.5} />
      
      {/* 指向性光源 */}
      <directionalLight
        ref={directionalLightRef}
        position={[10, 10, 10]}
        intensity={1}
        castShadow
      />
      
      {/* 床のグリッド */}
      <gridHelper args={[20, 20, '#444444', '#222222']} position={[0, -0.5, 0]} />
      
      {/* 座標軸 */}
      <axesHelper args={[5]} />
      
      {/* スプレッド表示 */}
      <SpreadIndicator spread={spread} position={[0, 0, 0]} />
      
      {/* 買い注文（Bids） */}
      {bids.map((bid, index) => (
        <OrderBar
          key={`bid-${index}`}
          entry={bid}
          position={[-index - 1, 0, 0]} // X軸の負の方向に配置
          color="#4CAF50" // 緑色
          maxSize={maxSize}
          type="bid"
        />
      ))}
      
      {/* 売り注文（Asks） */}
      {asks.map((ask, index) => (
        <OrderBar
          key={`ask-${index}`}
          entry={ask}
          position={[index + 1, 0, 0]} // X軸の正の方向に配置
          color="#F44336" // 赤色
          maxSize={maxSize}
          type="ask"
        />
      ))}
      
      {/* カメラコントロール */}
      <OrbitControls
        enablePan={true}
        enableZoom={true}
        enableRotate={true}
        minDistance={5}
        maxDistance={20}
      />
    </>
  );
};

// メインコンポーネント
const OrderBookVisualization: React.FC<{
  data?: any;
  height?: number | string;
}> = ({ data, height = 400 }) => {
  const theme = useTheme();
  
  // データがない場合のメッセージ
  if (!data || !data.bids || !data.asks) {
    return (
      <Box
        display="flex"
        justifyContent="center"
        alignItems="center"
        height={height}
        bgcolor={theme.palette.background.paper}
        borderRadius={1}
      >
        <Typography>オーダーブックデータがありません</Typography>
      </Box>
    );
  }
  
  // 注文データの処理
  const { bids, asks, maxSize, spread } = useMemo(() => {
    // 買い注文（Bids）の処理
    let bidTotal = 0;
    const processedBids: OrderBookEntry[] = data.bids
      .slice(0, 10) // 上位10件のみ表示
      .map((bid: [number, number]) => {
        bidTotal += bid[1];
        return {
          price: bid[0],
          size: bid[1],
          total: bidTotal,
          sizePercent: 0 // 後で計算
        };
      });
    
    // 売り注文（Asks）の処理
    let askTotal = 0;
    const processedAsks: OrderBookEntry[] = data.asks
      .slice(0, 10) // 上位10件のみ表示
      .map((ask: [number, number]) => {
        askTotal += ask[1];
        return {
          price: ask[0],
          size: ask[1],
          total: askTotal,
          sizePercent: 0 // 後で計算
        };
      });
    
    // スプレッド（最安売り価格 - 最高買い価格）の計算
    const topBidPrice = processedBids.length > 0 ? processedBids[0].price : 0;
    const topAskPrice = processedAsks.length > 0 ? processedAsks[0].price : 0;
    const spread = topAskPrice - topBidPrice;
    
    // 最大サイズを計算（3Dバーのスケーリングに使用）
    const maxBidSize = Math.max(...processedBids.map(bid => bid.size));
    const maxAskSize = Math.max(...processedAsks.map(ask => ask.size));
    const maxSize = Math.max(maxBidSize, maxAskSize);
    
    // パーセンテージの計算
    processedBids.forEach(bid => {
      bid.sizePercent = (bid.size / maxSize) * 100;
    });
    
    processedAsks.forEach(ask => {
      ask.sizePercent = (ask.size / maxSize) * 100;
    });
    
    return { bids: processedBids, asks: processedAsks, maxSize, spread };
  }, [data]);
  
  return (
    <Box
      sx={{
        height,
        width: '100%',
        bgcolor: theme.palette.background.paper,
        borderRadius: 1,
        overflow: 'hidden'
      }}
    >
      <Canvas
        camera={{ position: [0, 5, 10], fov: 60 }}
        shadows
        gl={{ antialias: true }}
      >
        <OrderBookScene
          bids={bids}
          asks={asks}
          spread={spread}
          maxSize={maxSize}
        />
      </Canvas>
    </Box>
  );
};

export default OrderBookVisualization;
