"""
Script to prepare historical data for backtesting.
Downloads data from S3, processes it, and saves to local directory.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import boto3
import logging
from typing import List, Dict
import json
from datetime import datetime, timedelta
import pytz

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BacktestDataPreparer:
    def __init__(self,
                 s3_bucket: str,
                 s3_prefix: str,
                 local_data_dir: Path,
                 start_date: datetime,
                 end_date: datetime):
        """
        Initialize data preparer.
        
        Args:
            s3_bucket: S3 bucket name
            s3_prefix: Prefix for S3 objects
            local_data_dir: Local directory to save processed data
            start_date: Start date for historical data
            end_date: End date for historical data
        """
        self.s3_bucket = s3_bucket
        self.s3_prefix = s3_prefix
        self.local_data_dir = local_data_dir
        self.start_date = start_date
        self.end_date = end_date
        
        # Initialize S3 client
        self.s3 = boto3.client('s3')
        
        # Create local directory if it doesn't exist
        self.local_data_dir.mkdir(parents=True, exist_ok=True)
        
    def download_from_s3(self) -> List[Path]:
        """Download raw data files from S3."""
        downloaded_files = []
        
        try:
            # List objects in S3
            paginator = self.s3.get_paginator('list_objects_v2')
            
            for page in paginator.paginate(
                Bucket=self.s3_bucket,
                Prefix=self.s3_prefix
            ):
                for obj in page.get('Contents', []):
                    # Check if file is within date range
                    file_date = self._extract_date_from_key(obj['Key'])
                    if not self._is_date_in_range(file_date):
                        continue
                    
                    # Download file
                    local_path = self.local_data_dir / Path(obj['Key']).name
                    logger.info(f"Downloading {obj['Key']} to {local_path}")
                    
                    self.s3.download_file(
                        self.s3_bucket,
                        obj['Key'],
                        str(local_path)
                    )
                    downloaded_files.append(local_path)
            
        except Exception as e:
            logger.error(f"Error downloading from S3: {e}")
            raise
        
        return downloaded_files
    
    def process_data(self, raw_files: List[Path]) -> pd.DataFrame:
        """Process raw data files into clean format for backtesting."""
        dfs = []
        
        for file in raw_files:
            try:
                # Read raw data
                df = pd.read_csv(file)
                
                # Clean and process
                df = self._clean_data(df)
                
                # Calculate additional features
                df = self._calculate_features(df)
                
                dfs.append(df)
                
            except Exception as e:
                logger.error(f"Error processing {file}: {e}")
                continue
        
        # Combine all data
        if not dfs:
            raise ValueError("No data processed successfully")
            
        combined_df = pd.concat(dfs, ignore_index=True)
        
        # Sort by timestamp
        combined_df.sort_values('timestamp', inplace=True)
        
        return combined_df
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean raw data."""
        # Remove duplicates
        df = df.drop_duplicates(subset=['timestamp'])
        
        # Handle missing values
        df = df.ffill()  # Forward fill
        
        # Remove outliers
        for col in ['price', 'volume']:
            if col in df.columns:
                mean = df[col].mean()
                std = df[col].std()
                df = df[abs(df[col] - mean) <= 3 * std]  # Remove 3 sigma outliers
        
        return df
    
    def _calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate additional features for backtesting."""
        # Calculate returns
        df['returns'] = df['price'].pct_change()
        
        # Calculate volatility
        df['volatility'] = df['returns'].rolling(window=30).std() * np.sqrt(252)
        
        # Calculate volume indicators
        df['volume_ma'] = df['volume'].rolling(window=24).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # Calculate price momentum
        df['momentum'] = df['price'].pct_change(periods=12)
        
        return df
    
    def save_processed_data(self, df: pd.DataFrame):
        """Save processed data to local directory."""
        # Save main dataset
        output_path = self.local_data_dir / 'processed_data.parquet'
        df.to_parquet(output_path, index=False)
        logger.info(f"Saved processed data to {output_path}")
        
        # Save metadata
        metadata = {
            'start_date': self.start_date.isoformat(),
            'end_date': self.end_date.isoformat(),
            'num_records': len(df),
            'columns': list(df.columns),
            'data_stats': {
                col: {
                    'mean': float(df[col].mean()),
                    'std': float(df[col].std()),
                    'min': float(df[col].min()),
                    'max': float(df[col].max())
                }
                for col in df.select_dtypes(include=[np.number]).columns
            }
        }
        
        metadata_path = self.local_data_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved metadata to {metadata_path}")
    
    def _extract_date_from_key(self, key: str) -> datetime:
        """Extract date from S3 object key."""
        # Implement based on your S3 key format
        pass
    
    def _is_date_in_range(self, date: datetime) -> bool:
        """Check if date is within specified range."""
        return self.start_date <= date <= self.end_date

def main():
    # Configuration
    s3_bucket = "your-bucket-name"
    s3_prefix = "market-data/"
    local_data_dir = Path("data/processed/historical")
    
    # Date range (example: last 6 months)
    end_date = datetime.now(pytz.UTC)
    start_date = end_date - timedelta(days=180)
    
    # Initialize preparer
    preparer = BacktestDataPreparer(
        s3_bucket=s3_bucket,
        s3_prefix=s3_prefix,
        local_data_dir=local_data_dir,
        start_date=start_date,
        end_date=end_date
    )
    
    try:
        # Download data
        logger.info("Downloading data from S3...")
        raw_files = preparer.download_from_s3()
        
        # Process data
        logger.info("Processing downloaded data...")
        processed_df = preparer.process_data(raw_files)
        
        # Save results
        logger.info("Saving processed data...")
        preparer.save_processed_data(processed_df)
        
        logger.info("Data preparation completed successfully")
        
    except Exception as e:
        logger.error(f"Data preparation failed: {e}")
        raise

if __name__ == '__main__':
    main()
