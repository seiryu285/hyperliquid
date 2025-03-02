"""
Performance metrics for evaluating prediction models.

This module provides metrics for evaluating the performance of price prediction models,
including MAE, MSE, direction accuracy, and more.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Union, Tuple


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Error (MAE).
    
    Args:
        y_true: Array of true values
        y_pred: Array of predicted values
        
    Returns:
        MAE value
    """
    return np.mean(np.abs(y_true - y_pred))


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Squared Error (MSE).
    
    Args:
        y_true: Array of true values
        y_pred: Array of predicted values
        
    Returns:
        MSE value
    """
    return np.mean(np.square(y_true - y_pred))


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Root Mean Squared Error (RMSE).
    
    Args:
        y_true: Array of true values
        y_pred: Array of predicted values
        
    Returns:
        RMSE value
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Percentage Error (MAPE).
    
    Args:
        y_true: Array of true values
        y_pred: Array of predicted values
        
    Returns:
        MAPE value as a percentage
    """
    return 100.0 * np.mean(np.abs((y_true - y_pred) / (y_true + np.finfo(float).eps)))


def direction_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate directional accuracy (percentage of correct price movement predictions).
    
    Args:
        y_true: Array of true values
        y_pred: Array of predicted values
        
    Returns:
        Direction accuracy as a percentage
    """
    # Calculate the direction of movement for true values
    true_direction = np.sign(np.diff(y_true))
    
    # Calculate the direction of movement for predicted values
    pred_direction = np.sign(np.diff(y_pred))
    
    # Calculate accuracy
    correct_direction = np.sum(true_direction == pred_direction)
    total_directions = len(true_direction)
    
    return 100.0 * correct_direction / total_directions if total_directions > 0 else 0.0


def calculate_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate all available metrics for model evaluation.
    
    Args:
        y_true: Array of true values
        y_pred: Array of predicted values
        
    Returns:
        Dictionary containing all metrics
    """
    metrics = {
        'mae': mean_absolute_error(y_true, y_pred),
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': root_mean_squared_error(y_true, y_pred),
        'mape': mean_absolute_percentage_error(y_true, y_pred),
    }
    
    # Only calculate direction accuracy if we have more than one point
    if len(y_true) > 1:
        metrics['direction_accuracy'] = direction_accuracy(y_true, y_pred)
    
    return metrics
