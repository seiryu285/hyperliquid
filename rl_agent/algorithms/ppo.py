#!/usr/bin/env python3
"""
Advanced PPO (Proximal Policy Optimization) implementation with market state clustering
and group-based penalties for robust trading strategies.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from sklearn.cluster import KMeans
import yaml
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple, Optional

from ..models.actor import Actor
from ..models.critic import Critic
from ..evaluations.metrics import calculate_sharpe_ratio
from ..utils.logger import setup_logger

logger = setup_logger(__name__)

class AdvancedPPO:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config_path: str = "config/hyperparameters.yaml",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the Advanced PPO algorithm with clustering capabilities.
        
        Args:
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            config_path: Path to hyperparameters configuration
            device: Device to run computations on (cuda/cpu)
        """
        self.device = device
        self.load_config(config_path)
        
        # Initialize networks
        self.actor = Actor(state_dim, action_dim).to(device)
        self.critic = Critic(state_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.config['learning_rate'])
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.config['learning_rate'])
        
        # Initialize clustering
        self.kmeans = KMeans(n_clusters=self.config['n_clusters'])
        
        # Initialize parallel processing
        self.executor = ThreadPoolExecutor(max_workers=self.config['n_workers'])

    def load_config(self, config_path: str) -> None:
        """Load hyperparameters from yaml config file."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)['ppo']
            
        # Extract key parameters
        self.clip_param = self.config['clip_param']
        self.max_grad_norm = self.config['max_grad_norm']
        self.ppo_epochs = self.config['ppo_epochs']
        self.batch_size = self.config['batch_size']
        self.gamma = self.config['gamma']
        self.gae_lambda = self.config['gae_lambda']
        self.group_penalty_lambda = self.config['group_penalty_lambda']
        self.target_sharpe = self.config['target_sharpe']

    def select_action(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Select action using the current policy.
        
        Args:
            state: Current state tensor
            
        Returns:
            action: Selected action
            log_prob: Log probability of the action
        """
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            mean, std = self.actor(state)
            dist = Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
        return action.cpu().numpy(), log_prob.cpu().numpy()

    def compute_gae(
        self,
        rewards: List[float],
        values: List[float],
        next_value: float,
        dones: List[bool]
    ) -> Tuple[List[float], List[float]]:
        """
        Compute Generalized Advantage Estimation (GAE).
        
        Args:
            rewards: List of rewards
            values: List of value estimates
            next_value: Value estimate of the next state
            dones: List of done flags
            
        Returns:
            advantages: Computed advantages
            returns: Computed returns
        """
        gae = 0
        returns = []
        advantages = []
        
        for step in reversed(range(len(rewards))):
            if step == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[step]
                next_value = next_value
            else:
                next_non_terminal = 1.0 - dones[step + 1]
                next_value = values[step + 1]
                
            delta = rewards[step] + self.gamma * next_value * next_non_terminal - values[step]
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            returns.insert(0, gae + values[step])
            advantages.insert(0, gae)
            
        return advantages, returns

    def cluster_market_states(self, states: np.ndarray, returns: np.ndarray) -> Dict:
        """
        Cluster market states and compute group-wise metrics.
        
        Args:
            states: Array of market states
            returns: Array of corresponding returns
            
        Returns:
            Dictionary containing clustering results and metrics
        """
        # Extract relevant features for clustering
        clustering_features = self._extract_clustering_features(states)
        
        # Perform clustering
        cluster_labels = self.kmeans.fit_predict(clustering_features)
        
        # Compute metrics for each cluster
        cluster_metrics = {}
        for i in range(self.config['n_clusters']):
            cluster_mask = cluster_labels == i
            cluster_returns = returns[cluster_mask]
            
            if len(cluster_returns) > 0:
                sharpe = calculate_sharpe_ratio(cluster_returns)
                cluster_metrics[i] = {
                    'sharpe_ratio': sharpe,
                    'size': np.sum(cluster_mask),
                    'mean_return': np.mean(cluster_returns)
                }
        
        return {
            'labels': cluster_labels,
            'metrics': cluster_metrics
        }

    def _extract_clustering_features(self, states: np.ndarray) -> np.ndarray:
        """
        Extract features for clustering from market states.
        
        Args:
            states: Array of market states
            
        Returns:
            Array of features for clustering
        """
        # Example feature extraction (modify based on your state representation)
        volatility = np.std(states, axis=1)
        trend = np.mean(states, axis=1)
        volume = states[:, -1]  # Assuming volume is the last feature
        
        return np.column_stack([volatility, trend, volume])

    def compute_group_penalty(self, cluster_metrics: Dict) -> float:
        """
        Compute penalty based on group performance.
        
        Args:
            cluster_metrics: Dictionary containing cluster metrics
            
        Returns:
            Computed penalty value
        """
        worst_group_sharpe = float('inf')
        for metrics in cluster_metrics.values():
            if metrics['sharpe_ratio'] < worst_group_sharpe:
                worst_group_sharpe = metrics['sharpe_ratio']
        
        penalty = max(0, self.target_sharpe - worst_group_sharpe)
        return self.group_penalty_lambda * penalty

    @torch.no_grad()
    def evaluate_actions(
        self,
        states: torch.Tensor,
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions using current policy and value function.
        
        Args:
            states: Batch of states
            actions: Batch of actions
            
        Returns:
            values: Value estimates
            log_probs: Log probabilities of actions
            entropy: Policy entropy
        """
        mean, std = self.actor(states)
        dist = Normal(mean, std)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()
        values = self.critic(states)
        
        return values, log_probs, entropy

    def update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        cluster_results: Dict
    ) -> Dict[str, float]:
        """
        Update policy and value function using PPO algorithm.
        
        Args:
            states: Batch of states
            actions: Batch of actions
            old_log_probs: Log probabilities of actions under old policy
            advantages: Computed advantages
            returns: Computed returns
            cluster_results: Results from market state clustering
            
        Returns:
            Dictionary containing various loss metrics
        """
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Compute group penalty
        group_penalty = self.compute_group_penalty(cluster_results['metrics'])
        
        for _ in range(self.ppo_epochs):
            # Get minibatch
            for idx in range(0, len(states), self.batch_size):
                batch_states = states[idx:idx + self.batch_size]
                batch_actions = actions[idx:idx + self.batch_size]
                batch_old_log_probs = old_log_probs[idx:idx + self.batch_size]
                batch_advantages = advantages[idx:idx + self.batch_size]
                batch_returns = returns[idx:idx + self.batch_size]
                
                # Evaluate actions under current policy
                values, log_probs, entropy = self.evaluate_actions(
                    batch_states,
                    batch_actions
                )
                
                # Compute ratio between new and old policy
                ratio = torch.exp(log_probs - batch_old_log_probs)
                
                # Compute surrogate losses
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * batch_advantages
                
                # Compute actor loss
                actor_loss = -torch.min(surr1, surr2).mean()
                actor_loss += group_penalty  # Add group penalty
                
                # Compute critic loss
                value_loss = F.mse_loss(values.squeeze(), batch_returns)
                
                # Compute entropy loss for exploration
                entropy_loss = -self.config['entropy_coef'] * entropy
                
                # Total loss
                total_loss = actor_loss + self.config['value_loss_coef'] * value_loss + entropy_loss
                
                # Update networks
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                total_loss.backward()
                
                # Clip gradients
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                
                self.actor_optimizer.step()
                self.critic_optimizer.step()
        
        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': value_loss.item(),
            'entropy': entropy.item(),
            'group_penalty': group_penalty
        }

    def save(self, path: str) -> None:
        """Save model parameters."""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
        }, path)

    def load(self, path: str) -> None:
        """Load model parameters."""
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])

if __name__ == '__main__':
    print('Advanced PPO Algorithm')