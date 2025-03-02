import React, { useRef, useMemo } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Text, Line, Html } from '@react-three/drei';
import * as THREE from 'three';
import { Box, Typography, useTheme } from '@mui/material';

interface PricePoint {
  time: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

interface PriceChartVisualizationProps {
  data: PricePoint[];
  timeframe: string;
  height?: number | string;
}

// 3D価格チャートのメインコンポーネント
const PriceChart3D: React.FC<{
  data: PricePoint[];
  timeframe: string;
}> = ({ data, timeframe }) => {
  const lineRef = useRef<THREE.Line>(null);
  const volumeGroupRef = useRef<THREE.Group>(null);
  
  // 価格データを3D座標に変換
  const { points, volumeData, minPrice, maxPrice, priceRange, timeLabels } = useMemo(() => {
    if (!data || data.length === 0) {
      return { 
        points: [], 
        volumeData: [], 
        minPrice: 0, 
        maxPrice: 0, 
        priceRange: 0,
        timeLabels: []
      };
    }
    
    // 最小・最大価格を検出
    const minPrice = Math.min(...data.map(d => d.low));
    const maxPrice = Math.max(...data.map(d => d.high));
    const priceRange = maxPrice - minPrice;
    
    // 最大ボリュームを検出
    const maxVolume = Math.max(...data.map(d => d.volume));
    
    // 価格線のポイント
    const points = data.map((point, index) => {
      // X座標は時間（インデックス）に基づく
      const x = index - data.length / 2;
      // Y座標は価格に基づく（正規化）
      const y = ((point.close - minPrice) / priceRange) * 5;
      
      return new THREE.Vector3(x, y, 0);
    });
    
    // ボリュームデータ
    const volumeData = data.map((point, index) => {
      const x = index - data.length / 2;
      const volumeHeight = (point.volume / maxVolume) * 2; // 最大高さを2に設定
      const color = point.close >= point.open ? '#4CAF50' : '#F44336'; // 陽線は緑、陰線は赤
      
      return {
        position: [x, 0, 0] as [number, number, number],
        height: volumeHeight,
        color
      };
    });
    
    // 時間ラベル（データ量に応じて間引く）
    const labelInterval = Math.max(1, Math.floor(data.length / 10));
    const timeLabels = data
      .filter((_, index) => index % labelInterval === 0)
      .map((point, index) => {
        const x = (index * labelInterval) - data.length / 2;
        const date = new Date(point.time);
        let label = '';
        
        // タイムフレームに応じたラベル形式
        switch (timeframe) {
          case '1h':
            label = `${date.getHours()}:00`;
            break;
          case '4h':
            label = `${date.getMonth()+1}/${date.getDate()} ${date.getHours()}:00`;
            break;
          case '1d':
            label = `${date.getMonth()+1}/${date.getDate()}`;
            break;
          default:
            label = `${date.getHours()}:${date.getMinutes().toString().padStart(2, '0')}`;
        }
        
        return {
          position: [x, -0.5, 0] as [number, number, number],
          text: label
        };
      });
    
    return { points, volumeData, minPrice, maxPrice, priceRange, timeLabels };
  }, [data, timeframe]);
  
  // アニメーション効果
  useFrame((state) => {
    if (lineRef.current) {
      lineRef.current.rotation.z = Math.sin(state.clock.getElapsedTime() * 0.1) * 0.02;
    }
    
    if (volumeGroupRef.current) {
      volumeGroupRef.current.rotation.x = Math.sin(state.clock.getElapsedTime() * 0.1) * 0.02;
    }
  });
  
  // 価格ラベル（Y軸）
  const priceLabels = useMemo(() => {
    const labels = [];
    const labelCount = 6;
    
    for (let i = 0; i < labelCount; i++) {
      const ratio = i / (labelCount - 1);
      const price = minPrice + priceRange * ratio;
      const y = ratio * 5;
      
      labels.push({
        position: [-data.length / 2 - 1, y, 0] as [number, number, number],
        text: price.toFixed(2)
      });
    }
    
    return labels;
  }, [minPrice, maxPrice, priceRange, data.length]);
  
  return (
    <>
      {/* 環境光 */}
      <ambientLight intensity={0.5} />
      
      {/* 指向性光源 */}
      <directionalLight position={[10, 10, 10]} intensity={1} castShadow />
      
      {/* 背景グリッド */}
      <gridHelper args={[30, 30, '#444444', '#222222']} position={[0, 0, -0.5]} rotation={[Math.PI / 2, 0, 0]} />
      
      {/* 価格線 */}
      {points.length > 0 && (
        <Line
          ref={lineRef}
          points={points}
          color="#2196F3"
          lineWidth={2}
        />
      )}
      
      {/* ボリューム表示 */}
      <group ref={volumeGroupRef}>
        {volumeData.map((volume, index) => (
          <mesh key={`volume-${index}`} position={[volume.position[0], volume.position[1] - 2, volume.position[2]]}>
            <boxGeometry args={[0.8, volume.height, 0.2]} />
            <meshStandardMaterial color={volume.color} transparent opacity={0.7} />
          </mesh>
        ))}
      </group>
      
      {/* 時間ラベル（X軸） */}
      {timeLabels.map((label, index) => (
        <Text
          key={`time-${index}`}
          position={label.position}
          fontSize={0.3}
          color="#ffffff"
          anchorX="center"
          anchorY="top"
        >
          {label.text}
        </Text>
      ))}
      
      {/* 価格ラベル（Y軸） */}
      {priceLabels.map((label, index) => (
        <Text
          key={`price-${index}`}
          position={label.position}
          fontSize={0.3}
          color="#ffffff"
          anchorX="right"
          anchorY="middle"
        >
          {label.text}
        </Text>
      ))}
      
      {/* タイムフレームラベル */}
      <Text
        position={[data.length / 2 - 2, 5.5, 0]}
        fontSize={0.4}
        color="#ffff00"
        anchorX="right"
        anchorY="top"
      >
        {timeframe} Chart
      </Text>
      
      {/* カメラコントロール */}
      <OrbitControls
        enablePan={true}
        enableZoom={true}
        enableRotate={true}
        minDistance={5}
        maxDistance={30}
        maxPolarAngle={Math.PI / 2}
      />
    </>
  );
};

// メインコンポーネント
const PriceChartVisualization: React.FC<PriceChartVisualizationProps> = ({ 
  data, 
  timeframe,
  height = 400
}) => {
  const theme = useTheme();
  
  // データがない場合のメッセージ
  if (!data || data.length === 0) {
    return (
      <Box
        display="flex"
        justifyContent="center"
        alignItems="center"
        height={height}
        bgcolor={theme.palette.background.paper}
        borderRadius={1}
      >
        <Typography>価格データがありません</Typography>
      </Box>
    );
  }
  
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
        camera={{ position: [0, 0, 15], fov: 60 }}
        shadows
        gl={{ antialias: true }}
      >
        <PriceChart3D
          data={data}
          timeframe={timeframe}
        />
      </Canvas>
    </Box>
  );
};

export default PriceChartVisualization;
