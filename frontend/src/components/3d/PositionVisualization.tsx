import React, { useRef, useMemo } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Text, Html } from '@react-three/drei';
import * as THREE from 'three';
import { Box, Typography, useTheme, Chip } from '@mui/material';

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

interface PositionVisualizationProps {
  data?: Position[] | null;
  height?: number | string;
}

// 3Dポジションビジュアライゼーションのメインコンポーネント
const Position3D: React.FC<{
  position: Position;
}> = ({ position }) => {
  const groupRef = useRef<THREE.Group>(null);
  const sphereRef = useRef<THREE.Mesh>(null);
  
  // ポジションの利益/損失に基づく色
  const color = position.unrealizedPnl >= 0 ? '#4CAF50' : '#F44336';
  
  // ポジションサイズに基づく球体の大きさ
  const sphereSize = Math.max(0.5, Math.min(3, Math.abs(position.size) * 0.5));
  
  // レバレッジに基づくリング（トーラス）の大きさ
  const torusSize = Math.max(1, Math.min(5, position.leverage * 0.5));
  
  // アニメーション効果
  useFrame((state) => {
    if (groupRef.current) {
      groupRef.current.rotation.y = state.clock.getElapsedTime() * 0.2;
    }
    
    if (sphereRef.current) {
      sphereRef.current.scale.x = 1 + Math.sin(state.clock.getElapsedTime() * 2) * 0.05;
      sphereRef.current.scale.y = 1 + Math.sin(state.clock.getElapsedTime() * 2) * 0.05;
      sphereRef.current.scale.z = 1 + Math.sin(state.clock.getElapsedTime() * 2) * 0.05;
    }
  });
  
  // 利益/損失に基づく浮遊効果
  const yOffset = position.unrealizedPnl >= 0 ? 0.5 : -0.5;
  
  return (
    <group ref={groupRef} position={[0, yOffset, 0]}>
      {/* 中央の球体（ポジション） */}
      <mesh ref={sphereRef}>
        <sphereGeometry args={[sphereSize, 32, 32]} />
        <meshStandardMaterial
          color={color}
          metalness={0.8}
          roughness={0.2}
          emissive={color}
          emissiveIntensity={0.5}
        />
        
        {/* シンボル表示 */}
        <Html position={[0, 0, 0]} center>
          <div style={{ 
            color: 'white', 
            fontWeight: 'bold',
            textShadow: '0 0 5px black',
            fontSize: '16px'
          }}>
            {position.symbol}
          </div>
        </Html>
      </mesh>
      
      {/* レバレッジを表すリング */}
      <mesh rotation={[Math.PI / 2, 0, 0]}>
        <torusGeometry args={[torusSize, 0.1, 16, 100]} />
        <meshStandardMaterial color="#FFEB3B" metalness={0.5} roughness={0.5} />
      </mesh>
      
      {/* エントリー価格 */}
      <Text
        position={[0, sphereSize + 1, 0]}
        fontSize={0.4}
        color="#ffffff"
        anchorX="center"
        anchorY="bottom"
      >
        Entry: ${position.entryPrice.toFixed(2)}
      </Text>
      
      {/* 現在価格 */}
      <Text
        position={[0, sphereSize + 1.5, 0]}
        fontSize={0.4}
        color="#ffffff"
        anchorX="center"
        anchorY="bottom"
      >
        Mark: ${position.markPrice.toFixed(2)}
      </Text>
      
      {/* 清算価格 */}
      <Text
        position={[0, sphereSize + 2, 0]}
        fontSize={0.4}
        color="#FF5722"
        anchorX="center"
        anchorY="bottom"
      >
        Liquidation: ${position.liquidationPrice.toFixed(2)}
      </Text>
      
      {/* 未実現損益 */}
      <Text
        position={[0, -sphereSize - 1, 0]}
        fontSize={0.5}
        color={position.unrealizedPnl >= 0 ? '#4CAF50' : '#F44336'}
        anchorX="center"
        anchorY="top"
      >
        PnL: ${position.unrealizedPnl.toFixed(2)} ({position.unrealizedPnlPercent.toFixed(2)}%)
      </Text>
      
      {/* サイズとレバレッジ */}
      <Text
        position={[0, -sphereSize - 1.8, 0]}
        fontSize={0.4}
        color="#BBBBBB"
        anchorX="center"
        anchorY="top"
      >
        Size: {position.size.toFixed(2)} | Leverage: {position.leverage}x
      </Text>
      
      {/* 方向を示す矢印 */}
      <mesh position={[0, 0, position.size >= 0 ? sphereSize + 0.5 : -sphereSize - 0.5]} rotation={[0, 0, position.size >= 0 ? 0 : Math.PI]}>
        <coneGeometry args={[0.5, 1, 32]} />
        <meshStandardMaterial color={position.size >= 0 ? '#4CAF50' : '#F44336'} />
      </mesh>
    </group>
  );
};

// 背景効果
const Background: React.FC = () => {
  const starsRef = useRef<THREE.Points>(null);
  
  // 星の生成
  const { positions, colors } = useMemo(() => {
    const positions = [];
    const colors = [];
    const color = new THREE.Color();
    
    for (let i = 0; i < 1000; i++) {
      positions.push((Math.random() - 0.5) * 50);
      positions.push((Math.random() - 0.5) * 50);
      positions.push((Math.random() - 0.5) * 50);
      
      const c = color.setHSL(Math.random(), 0.7, 0.8);
      colors.push(c.r, c.g, c.b);
    }
    
    return { positions, colors };
  }, []);
  
  // 星のアニメーション
  useFrame((state) => {
    if (starsRef.current) {
      starsRef.current.rotation.y = state.clock.getElapsedTime() * 0.05;
    }
  });
  
  return (
    <points ref={starsRef}>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          count={positions.length / 3}
          array={new Float32Array(positions)}
          itemSize={3}
        />
        <bufferAttribute
          attach="attributes-color"
          count={colors.length / 3}
          array={new Float32Array(colors)}
          itemSize={3}
        />
      </bufferGeometry>
      <pointsMaterial
        size={0.1}
        vertexColors
        transparent
        opacity={0.8}
      />
    </points>
  );
};

// メインコンポーネント
const PositionVisualization: React.FC<PositionVisualizationProps> = ({ 
  data,
  height = 400
}) => {
  const theme = useTheme();
  
  // ポジションがない場合のメッセージ
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
        <Typography>アクティブなポジションがありません</Typography>
      </Box>
    );
  }
  
  // ETH-PERPのポジションを検索
  const ethPosition = data.find(position => position.symbol === 'ETH-PERP');
  
  if (!ethPosition) {
    return (
      <Box
        display="flex"
        justifyContent="center"
        alignItems="center"
        height={height}
        bgcolor={theme.palette.background.paper}
        borderRadius={1}
      >
        <Typography>ETH-PERPのポジションがありません</Typography>
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
        <Background />
        <Position3D position={ethPosition} />
        
        {/* カメラコントロール */}
        <OrbitControls
          enablePan={true}
          enableZoom={true}
          enableRotate={true}
          minDistance={5}
          maxDistance={20}
        />
      </Canvas>
    </Box>
  );
};

export default PositionVisualization;
