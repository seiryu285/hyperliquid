#!/usr/bin/env python3
"""
Custom Trading Environment implementing OpenAI Gym interface
"""

import gym
import numpy as np
from gym import spaces
from typing import Tuple, Dict
from dataclasses import dataclass
from collections import deque

@dataclass
class MarketState:
    price: float
    orderbook: dict
    volume: float
    funding_rate: float
    volatility: float

class TradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, 
                 initial_balance: float = 100000.0,
                 max_steps: int = 1000,
                 lookback_window: int = 50,
                 risk_free_rate: float = 0.0):
        super(TradingEnv, self).__init__()
        
        # Initialize state parameters
        self.initial_balance = initial_balance
        self.max_steps = max_steps
        self.lookback_window = lookback_window
        self.risk_free_rate = risk_free_rate
        
        # Define action space
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0]),  # [position_direction, leverage]
            high=np.array([1.0, 10.0]), 
            dtype=np.float32
        )
        
        # Define observation space
        self.observation_space = spaces.Dict({
            'market_data': spaces.Box(low=-np.inf, high=np.inf, shape=(5,)),
            'portfolio': spaces.Box(low=0, high=np.inf, shape=(3,)),
            'history': spaces.Box(low=-np.inf, high=np.inf, 
                                shape=(lookback_window, 5))
        })
        
        # Initialize state tracking
        self.reset()

    def reset(self):
        self.balance = self.initial_balance
        self.position = 0.0
        self.portfolio_value = self.initial_balance
        self.current_step = 0
        self.trade_history = deque(maxlen=self.lookback_window)
        self.cluster_labels = []
        self.returns = []
        
        # Initialize market state
        self.market_state = MarketState(
            price=0.0,
            orderbook={'bids': [], 'asks': []},
            volume=0.0,
            funding_rate=0.0,
            volatility=0.0
        )
        
        return self._get_obs()

    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, Dict]:
        # Execute trade
        direction, leverage = action
        done = False
        info = {}
        
        # Calculate transaction
        new_position = self._calculate_new_position(direction, leverage)
        
        # Update portfolio
        self._update_portfolio(new_position)
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Update step
        self.current_step += 1
        if self.current_step >= self.max_steps:
            done = True
        
        return self._get_obs(), reward, done, info

    def _calculate_new_position(self, direction: float, leverage: float) -> float:
        max_position = self.balance * leverage / self.market_state.price
        return max_position * direction

    def _update_portfolio(self, new_position: float):
        # Calculate transaction costs
        delta_position = new_position - self.position
        transaction_cost = abs(delta_position) * self.market_state.price * 0.0002
        
        # Update position and balance
        self.position = new_position
        self.balance -= transaction_cost
        
        # Calculate funding costs
        funding_cost = abs(self.position) * self.market_state.funding_rate
        self.balance -= funding_cost
        
        # Update portfolio value
        self.portfolio_value = self.balance + \
            self.position * self.market_state.price

    def _calculate_reward(self) -> float:
        # Calculate raw returns
        returns = np.diff(self.returns[-self.lookback_window:]) if self.returns else 0
        
        # Risk-adjusted return (Sharpe Ratio)
        sharpe = (np.mean(returns) - self.risk_free_rate) / np.std(returns) \
            if len(returns) > 1 else 0
        
        # Position-based penalties
        leverage_penalty = 0.01 * (abs(self.position) ** 2)
        liquidation_risk = 0.1 * max(0, abs(self.position) - 5)
        
        # Cluster-based adjustment
        cluster_bonus = self._get_cluster_adjustment()
        
        return sharpe - leverage_penalty - liquidation_risk + cluster_bonus

    def _get_cluster_adjustment(self) -> float:
        if not self.cluster_labels:
            return 0
        
        # Get latest cluster metrics
        current_cluster = self.cluster_labels[-1]
        cluster_perf = self.cluster_metrics.get(current_cluster, 0)
        return 0.1 * cluster_perf

    def _get_obs(self) -> Dict:
        return {
            'market_data': np.array([
                self.market_state.price,
                self.market_state.volatility,
                self.market_state.volume,
                self.market_state.funding_rate,
                len(self.market_state.orderbook['bids'])
            ]),
            'portfolio': np.array([
                self.balance,
                self.position,
                self.portfolio_value
            ]),
            'history': np.array(self.trade_history)
        }

    def update_market_state(self, market_data: MarketState):
        self.market_state = market_data
        self.trade_history.append([
            market_data.price,
            market_data.volatility,
            market_data.volume,
            market_data.funding_rate,
            len(market_data.orderbook['bids'])
        ])

    def update_clusters(self, cluster_labels: list, cluster_metrics: dict):
        self.cluster_labels = cluster_labels
        self.cluster_metrics = cluster_metrics

    def render(self, mode='human'):
        print(f"Step: {self.current_step}")
        print(f"Portfolio Value: {self.portfolio_value:.2f}")
        print(f"Position: {self.position:.4f}")
        print(f"Current Price: {self.market_state.price:.2f}")

if __name__ == '__main__':
    env = TradingEnv()
    obs = env.reset()
    for _ in range(3):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            break
