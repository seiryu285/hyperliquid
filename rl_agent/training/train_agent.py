#!/usr/bin/env python3
"""
Main training script for the RL agent, implementing the training loop with
environment interaction, logging, and checkpointing functionality.
"""

import os
import time
import yaml
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from ..algorithms.ppo import AdvancedPPO
from ..algorithms.custom_loss import CustomLoss
from ..utils.logger import setup_logger
from ..utils.checkpoint import CheckpointManager
from ..evaluations.metrics import calculate_metrics
from ..evaluations.visualizations import plot_training_curves

logger = setup_logger(__name__)

class TrainingManager:
    def __init__(
        self,
        env,
        config_path: str = "config/hyperparameters.yaml",
        checkpoint_dir: str = "checkpoints",
        log_dir: str = "logs"
    ):
        """
        Initialize the training manager.
        
        Args:
            env: Trading environment instance
            config_path: Path to configuration file
            checkpoint_dir: Directory for saving checkpoints
            log_dir: Directory for saving logs
        """
        self.env = env
        self.load_config(config_path)
        
        # Initialize directories
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        # Setup components
        self.setup_agent()
        self.setup_checkpoint_manager(checkpoint_dir)
        
        # Initialize training metrics
        self.episode_rewards: List[float] = []
        self.sharpe_ratios: List[float] = []
        self.group_metrics: List[Dict] = []
        
        # Training state
        self.current_episode = 0
        self.best_sharpe = float('-inf')
        
    def load_config(self, config_path: str) -> None:
        """Load training configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.max_episodes = self.config['training']['max_episodes']
        self.eval_frequency = self.config['training']['eval_frequency']
        self.save_frequency = self.config['training']['save_frequency']
        self.log_frequency = self.config['training']['log_frequency']
        
    def setup_agent(self) -> None:
        """Initialize the PPO agent and custom loss function."""
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        
        self.agent = AdvancedPPO(
            state_dim=state_dim,
            action_dim=action_dim,
            config_path=self.config_path
        )
        
        self.custom_loss = CustomLoss(self.config['custom_loss'])
        
    def setup_checkpoint_manager(self, checkpoint_dir: str) -> None:
        """Initialize checkpoint management."""
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=checkpoint_dir,
            max_checkpoints=self.config['training']['max_checkpoints']
        )
        
    def collect_episode_data(self) -> Tuple[List, List, List, List, List, Dict]:
        """
        Collect data from a single episode.
        
        Returns:
            states: List of states
            actions: List of actions
            rewards: List of rewards
            log_probs: List of log probabilities
            values: List of value estimates
            market_data: Dictionary of market-related data
        """
        states = []
        actions = []
        rewards = []
        log_probs = []
        values = []
        market_data = {
            'market_impact': [],
            'funding_rates': [],
            'margin_ratios': [],
            'volatility': []
        }
        
        state = self.env.reset()
        done = False
        
        while not done:
            # Get action from policy
            action, log_prob = self.agent.select_action(state)
            value = self.agent.critic(torch.FloatTensor(state).to(self.agent.device))
            
            # Execute action
            next_state, reward, done, info = self.env.step(action)
            
            # Store transition
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            log_probs.append(log_prob)
            values.append(value.cpu().numpy())
            
            # Store market data
            for key in market_data.keys():
                market_data[key].append(info[key])
            
            state = next_state
            
        # Convert lists to arrays
        episode_data = {
            'states': np.array(states),
            'actions': np.array(actions),
            'rewards': np.array(rewards),
            'log_probs': np.array(log_probs),
            'values': np.array(values),
            'market_data': {k: np.array(v) for k, v in market_data.items()}
        }
        
        return episode_data
        
    def update_agent(self, episode_data: Dict) -> Dict[str, float]:
        """
        Update the agent using collected episode data.
        
        Args:
            episode_data: Dictionary containing episode data
            
        Returns:
            Dictionary of training metrics
        """
        # Compute advantages and returns
        advantages, returns = self.agent.compute_gae(
            rewards=episode_data['rewards'],
            values=episode_data['values'],
            next_value=episode_data['values'][-1],
            dones=[False] * len(episode_data['rewards'])
        )
        
        # Convert to tensors
        states = torch.FloatTensor(episode_data['states']).to(self.agent.device)
        actions = torch.FloatTensor(episode_data['actions']).to(self.agent.device)
        old_log_probs = torch.FloatTensor(episode_data['log_probs']).to(self.agent.device)
        advantages = torch.FloatTensor(advantages).to(self.agent.device)
        returns = torch.FloatTensor(returns).to(self.agent.device)
        
        # Cluster market states
        cluster_results = self.agent.cluster_market_states(
            states=episode_data['states'],
            returns=episode_data['rewards']
        )
        
        # Update policy and value function
        update_metrics = self.agent.update(
            states=states,
            actions=actions,
            old_log_probs=old_log_probs,
            advantages=advantages,
            returns=returns,
            cluster_results=cluster_results
        )
        
        return update_metrics, cluster_results
        
    def log_training_progress(
        self,
        episode: int,
        metrics: Dict[str, float],
        cluster_results: Dict
    ) -> None:
        """Log training progress and metrics."""
        logger.info(f"Episode {episode}/{self.max_episodes}")
        logger.info(f"Average Reward: {np.mean(self.episode_rewards[-100:]):.2f}")
        logger.info(f"Sharpe Ratio: {self.sharpe_ratios[-1]:.2f}")
        
        # Log loss components
        for key, value in metrics.items():
            logger.info(f"{key}: {value:.4f}")
        
        # Log cluster metrics
        for cluster_id, cluster_metrics in cluster_results['metrics'].items():
            logger.info(f"Cluster {cluster_id} - Size: {cluster_metrics['size']}, "
                      f"Sharpe: {cluster_metrics['sharpe_ratio']:.2f}")
        
    def save_checkpoint(self, episode: int, metrics: Dict[str, float]) -> None:
        """Save training checkpoint."""
        checkpoint_data = {
            'episode': episode,
            'agent_state': self.agent.state_dict(),
            'optimizer_state': self.agent.optimizer.state_dict(),
            'best_sharpe': self.best_sharpe,
            'metrics': metrics
        }
        
        self.checkpoint_manager.save(
            checkpoint_data,
            f"checkpoint_episode_{episode}.pt",
            metrics['sharpe_ratio']
        )
        
    def train(self, resume_from: Optional[str] = None) -> None:
        """
        Main training loop.
        
        Args:
            resume_from: Optional checkpoint path to resume training from
        """
        if resume_from:
            self.load_checkpoint(resume_from)
            
        start_time = time.time()
        
        while self.current_episode < self.max_episodes:
            # Collect episode data
            episode_data = self.collect_episode_data()
            
            # Update agent
            update_metrics, cluster_results = self.update_agent(episode_data)
            
            # Calculate episode metrics
            episode_metrics = calculate_metrics(
                returns=episode_data['rewards'],
                positions=episode_data['actions']
            )
            
            # Store metrics
            self.episode_rewards.append(np.sum(episode_data['rewards']))
            self.sharpe_ratios.append(episode_metrics['sharpe_ratio'])
            self.group_metrics.append(cluster_results['metrics'])
            
            # Log progress
            if self.current_episode % self.log_frequency == 0:
                self.log_training_progress(
                    self.current_episode,
                    update_metrics,
                    cluster_results
                )
                
            # Save checkpoint
            if self.current_episode % self.save_frequency == 0:
                self.save_checkpoint(self.current_episode, episode_metrics)
                
            # Update best model if necessary
            if episode_metrics['sharpe_ratio'] > self.best_sharpe:
                self.best_sharpe = episode_metrics['sharpe_ratio']
                self.save_checkpoint(self.current_episode, episode_metrics)
                
            # Plot training curves
            if self.current_episode % self.eval_frequency == 0:
                plot_training_curves(
                    rewards=self.episode_rewards,
                    sharpe_ratios=self.sharpe_ratios,
                    group_metrics=self.group_metrics,
                    save_path=f"logs/training_curves_episode_{self.current_episode}.png"
                )
            
            self.current_episode += 1
            
        # Training finished
        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time:.2f} seconds")
        logger.info(f"Best Sharpe Ratio achieved: {self.best_sharpe:.2f}")
        
if __name__ == '__main__':
    # Example usage
    import gym
    env = gym.make('YourTradingEnv-v0')
    
    trainer = TrainingManager(env)
    trainer.train()