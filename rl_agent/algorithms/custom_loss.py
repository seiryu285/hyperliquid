#!/usr/bin/env python3
"""
Custom loss functions for the PPO algorithm, incorporating trading costs,
leverage costs, and liquidation risk penalties.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple

class CustomLoss:
    def __init__(self, config: Dict):
        """
        Initialize custom loss function with configuration parameters.
        
        Args:
            config: Dictionary containing loss function parameters
        """
        self.config = config
        
        # Extract coefficients from config
        self.trading_cost_coef = config['trading_cost_coef']
        self.leverage_cost_coef = config['leverage_cost_coef']
        self.liquidation_risk_coef = config['liquidation_risk_coef']
        self.group_penalty_coef = config['group_penalty_coef']
        
    def compute_trading_costs(
        self,
        actions: torch.Tensor,
        prev_actions: torch.Tensor,
        market_impact: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute trading costs based on action changes and market impact.
        
        Trading Cost = Σ |a_t - a_{t-1}| * market_impact * trading_cost_coef
        
        Args:
            actions: Current actions tensor
            prev_actions: Previous actions tensor
            market_impact: Estimated market impact of trades
            
        Returns:
            Trading cost tensor
        """
        action_changes = torch.abs(actions - prev_actions)
        trading_costs = action_changes * market_impact * self.trading_cost_coef
        return trading_costs.mean()

    def compute_leverage_costs(
        self,
        positions: torch.Tensor,
        funding_rates: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute leverage costs based on position sizes and funding rates.
        
        Leverage Cost = Σ |position| * funding_rate * leverage_cost_coef
        
        Args:
            positions: Position sizes tensor
            funding_rates: Funding rates tensor
            
        Returns:
            Leverage cost tensor
        """
        leverage_costs = torch.abs(positions) * funding_rates * self.leverage_cost_coef
        return leverage_costs.mean()

    def compute_liquidation_risk(
        self,
        positions: torch.Tensor,
        margin_ratios: torch.Tensor,
        volatility: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute liquidation risk penalty based on position size, margin, and volatility.
        
        Liquidation Risk = Σ |position| * (1/margin_ratio) * volatility * liquidation_risk_coef
        
        Args:
            positions: Position sizes tensor
            margin_ratios: Margin ratios tensor
            volatility: Market volatility tensor
            
        Returns:
            Liquidation risk penalty tensor
        """
        # Avoid division by zero in margin ratios
        safe_margin_ratios = torch.clamp(margin_ratios, min=1e-6)
        
        liquidation_risk = (
            torch.abs(positions) *
            (1.0 / safe_margin_ratios) *
            volatility *
            self.liquidation_risk_coef
        )
        return liquidation_risk.mean()

    def compute_group_loss(
        self,
        cluster_results: Dict,
        target_sharpe: float
    ) -> torch.Tensor:
        """
        Compute group-based loss using clustering results.
        
        Group Loss = Σ max(0, target_sharpe - group_sharpe) * group_penalty_coef
        
        Args:
            cluster_results: Dictionary containing clustering metrics
            target_sharpe: Target Sharpe ratio
            
        Returns:
            Group-based loss tensor
        """
        group_losses = []
        for metrics in cluster_results['metrics'].values():
            sharpe_gap = max(0, target_sharpe - metrics['sharpe_ratio'])
            group_size = metrics['size']
            # Weight the loss by group size
            weighted_loss = sharpe_gap * (group_size / cluster_results['labels'].shape[0])
            group_losses.append(weighted_loss)
            
        return torch.tensor(group_losses).mean() * self.group_penalty_coef

    def __call__(
        self,
        policy_loss: torch.Tensor,
        value_loss: torch.Tensor,
        entropy_loss: torch.Tensor,
        actions: torch.Tensor,
        prev_actions: torch.Tensor,
        positions: torch.Tensor,
        market_data: Dict[str, torch.Tensor],
        cluster_results: Dict
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total loss combining PPO losses with custom penalties.
        
        Args:
            policy_loss: Standard PPO policy loss
            value_loss: Value function loss
            entropy_loss: Entropy bonus loss
            actions: Current actions
            prev_actions: Previous actions
            positions: Current positions
            market_data: Dictionary containing market-related data
            cluster_results: Results from market state clustering
            
        Returns:
            total_loss: Combined loss value
            metrics: Dictionary of individual loss components
        """
        # Compute trading costs
        trading_costs = self.compute_trading_costs(
            actions,
            prev_actions,
            market_data['market_impact']
        )
        
        # Compute leverage costs
        leverage_costs = self.compute_leverage_costs(
            positions,
            market_data['funding_rates']
        )
        
        # Compute liquidation risk
        liquidation_risk = self.compute_liquidation_risk(
            positions,
            market_data['margin_ratios'],
            market_data['volatility']
        )
        
        # Compute group-based loss
        group_loss = self.compute_group_loss(
            cluster_results,
            self.config['target_sharpe']
        )
        
        # Combine all losses
        total_loss = (
            policy_loss +
            self.config['value_loss_coef'] * value_loss +
            entropy_loss +
            trading_costs +
            leverage_costs +
            liquidation_risk +
            group_loss
        )
        
        # Return total loss and individual components for logging
        metrics = {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'trading_costs': trading_costs.item(),
            'leverage_costs': leverage_costs.item(),
            'liquidation_risk': liquidation_risk.item(),
            'group_loss': group_loss.item(),
            'total_loss': total_loss.item()
        }
        
        return total_loss, metrics