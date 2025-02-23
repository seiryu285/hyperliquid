"""
Machine learning based anomaly detection for security and performance metrics.
"""

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
import logging
from datetime import datetime, timedelta
import joblib
from pathlib import Path

from core.config import settings

logger = logging.getLogger(__name__)

class AnomalyDetector:
    """Anomaly detection using Isolation Forest algorithm."""
    
    def __init__(self, db, model_path: Optional[Path] = None):
        """Initialize anomaly detector."""
        self.db = db
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.model_path = model_path or settings.MODEL_DIR / "anomaly_detector.joblib"
    
    def prepare_features(
        self,
        data: List[Dict[str, Any]]
    ) -> Tuple[np.ndarray, List[str]]:
        """Prepare feature matrix from raw data."""
        df = pd.DataFrame(data)
        
        # Extract relevant features
        features = []
        
        # Market data features
        if 'price' in df:
            features.append('price')
        if 'volume' in df:
            features.append('volume')
        
        # Extract nested metrics if present
        if 'metrics' in df:
            metrics_df = pd.json_normalize(df['metrics'])
            for col in metrics_df.columns:
                df[f'metric_{col}'] = metrics_df[col]
                features.append(f'metric_{col}')
        
        # Time-based features
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
        features.extend(['hour', 'day_of_week'])
        
        # Select features and handle missing values
        X = df[features].fillna(0).values
        
        return X, features
    
    def train(
        self,
        training_data: List[Dict[str, Any]],
        contamination: float = 0.1
    ) -> None:
        """Train the anomaly detection model."""
        try:
            # Prepare features
            X, feature_names = self.prepare_features(training_data)
            self.feature_names = feature_names
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train model
            self.model = IsolationForest(
                contamination=contamination,
                random_state=42,
                n_jobs=-1
            )
            self.model.fit(X_scaled)
            
            logger.info(
                f"Trained anomaly detection model with {len(training_data)} samples"
            )
            
        except Exception as e:
            logger.error(f"Error training anomaly detection model: {e}")
            raise
    
    def save_model(self) -> None:
        """Save model to disk."""
        try:
            # Create directory if it doesn't exist
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save model and scaler
            joblib.dump(
                {
                    'model': self.model,
                    'scaler': self.scaler,
                    'feature_names': self.feature_names
                },
                self.model_path
            )
            logger.info(f"Saved model to {self.model_path}")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise
    
    def load_model(self) -> None:
        """Load model from disk."""
        try:
            if not self.model_path.exists():
                raise FileNotFoundError(
                    f"Model file not found at {self.model_path}"
                )
            
            # Load model and scaler
            saved_data = joblib.load(self.model_path)
            self.model = saved_data['model']
            self.scaler = saved_data['scaler']
            self.feature_names = saved_data['feature_names']
            
            logger.info(f"Loaded model from {self.model_path}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def detect_anomalies(
        self,
        data: List[Dict[str, Any]],
        threshold: float = -0.5
    ) -> List[Dict[str, Any]]:
        """Detect anomalies in the input data."""
        try:
            if not self.model:
                raise ValueError("Model not trained or loaded")
            
            # Prepare features
            X, _ = self.prepare_features(data)
            if not len(self.feature_names) == X.shape[1]:
                raise ValueError(
                    f"Input data has {X.shape[1]} features, "
                    f"but model expects {len(self.feature_names)}"
                )
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Get anomaly scores
            scores = self.model.score_samples(X_scaled)
            
            # Prepare results
            results = []
            for i, (score, item) in enumerate(zip(scores, data)):
                is_anomaly = score < threshold
                result = {
                    'timestamp': item['timestamp'],
                    'score': float(score),
                    'is_anomaly': is_anomaly,
                    'data': item
                }
                if is_anomaly:
                    logger.warning(
                        f"Detected anomaly: score={score:.3f}, "
                        f"data={item}"
                    )
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error detecting anomalies: {e}")
            raise
