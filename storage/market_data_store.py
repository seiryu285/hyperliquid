import logging
import asyncio
from typing import Dict, List, Optional
from datetime import datetime
import yaml
import motor.motor_asyncio
from pymongo import ASCENDING, DESCENDING
import json
import gzip
from pathlib import Path

logger = logging.getLogger(__name__)

class MarketDataStore:
    def __init__(self, config_path: str = "config/market_data.yaml"):
        self.config = self._load_config(config_path)
        self.client = self._initialize_mongodb_client()
        self.db = self.client[self.config['storage']['mongodb']['database']]
        self.batch_data: Dict[str, List[Dict]] = {}
        self.batch_size = self.config['processing']['batch_size']
        self._ensure_indexes()

    def _load_config(self, config_path: str) -> dict:
        """Load market data configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Successfully loaded market data configuration from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load market data configuration: {e}")
            raise

    def _initialize_mongodb_client(self) -> motor.motor_asyncio.AsyncIOMotorClient:
        """Initialize MongoDB client."""
        try:
            uri = self.config['storage']['mongodb']['uri']
            return motor.motor_asyncio.AsyncIOMotorClient(uri)
        except Exception as e:
            logger.error(f"Failed to initialize MongoDB client: {e}")
            raise

    async def _ensure_indexes(self) -> None:
        """Ensure required indexes exist in MongoDB collections."""
        try:
            collections_config = self.config['storage']['mongodb']['collections']
            
            # Create indexes for each collection
            for collection_name in collections_config.values():
                collection = self.db[collection_name]
                
                # Compound index on symbol and timestamp
                await collection.create_index(
                    [('symbol', ASCENDING), ('timestamp', DESCENDING)],
                    background=True
                )
                
                # Index for timestamp alone
                await collection.create_index(
                    [('timestamp', DESCENDING)],
                    background=True
                )
                
            logger.info("Successfully ensured MongoDB indexes")
            
        except Exception as e:
            logger.error(f"Error ensuring indexes: {e}")
            raise

    async def store_data(self, data: Dict) -> None:
        """Store market data in batches."""
        try:
            message_type = data.get('type')
            if not message_type:
                logger.warning("Message type not found in data")
                return

            collection_name = self.config['storage']['mongodb']['collections'].get(message_type)
            if not collection_name:
                logger.warning(f"Collection not found for message type: {message_type}")
                return

            if collection_name not in self.batch_data:
                self.batch_data[collection_name] = []

            self.batch_data[collection_name].append(data)

            # If batch size is reached, flush the data
            if len(self.batch_data[collection_name]) >= self.batch_size:
                await self._flush_collection(collection_name)

        except Exception as e:
            logger.error(f"Error storing data: {e}")
            raise

    async def _flush_collection(self, collection_name: str) -> None:
        """Flush batched data for a specific collection to MongoDB."""
        try:
            if not self.batch_data.get(collection_name):
                return

            collection = self.db[collection_name]
            data_to_insert = self.batch_data[collection_name]
            
            if self.config['storage']['compression_enabled']:
                data_to_insert = self._compress_data(data_to_insert)

            if data_to_insert:
                await collection.insert_many(data_to_insert)
                
                # Backup data if enabled
                if self.config['storage']['backup_enabled']:
                    await self._backup_data(collection_name, data_to_insert)

            self.batch_data[collection_name] = []
            logger.debug(f"Flushed {len(data_to_insert)} records to {collection_name}")

        except Exception as e:
            logger.error(f"Error flushing collection {collection_name}: {e}")
            raise

    async def flush(self) -> None:
        """Flush all batched data to MongoDB."""
        try:
            for collection_name in self.batch_data.keys():
                await self._flush_collection(collection_name)
        except Exception as e:
            logger.error(f"Error flushing all data: {e}")
            raise

    def _compress_data(self, data: List[Dict]) -> List[Dict]:
        """Compress data before storing."""
        try:
            compressed_data = []
            for item in data:
                if isinstance(item.get('data'), (dict, list)):
                    item['data'] = gzip.compress(json.dumps(item['data']).encode())
                compressed_data.append(item)
            return compressed_data
        except Exception as e:
            logger.error(f"Error compressing data: {e}")
            return data

    async def _backup_data(self, collection_name: str, data: List[Dict]) -> None:
        """Backup data to files."""
        try:
            backup_dir = Path('data/backup') / collection_name / datetime.now().strftime('%Y-%m-%d')
            backup_dir.mkdir(parents=True, exist_ok=True)

            backup_file = backup_dir / f"{datetime.now().strftime('%H-%M-%S')}.json.gz"
            
            compressed_data = gzip.compress(json.dumps(data).encode())
            backup_file.write_bytes(compressed_data)
            
            logger.debug(f"Backed up data to {backup_file}")

        except Exception as e:
            logger.error(f"Error backing up data: {e}")

    async def get_latest_data(self, symbol: str, channel: str) -> Optional[Dict]:
        """Get the latest market data for a symbol and channel."""
        try:
            collection_name = self.config['storage']['mongodb']['collections'].get(channel)
            if not collection_name:
                return None

            collection = self.db[collection_name]
            latest_data = await collection.find_one(
                {'symbol': symbol},
                sort=[('timestamp', DESCENDING)]
            )

            if latest_data and self.config['storage']['compression_enabled']:
                latest_data = self._decompress_data(latest_data)

            return latest_data

        except Exception as e:
            logger.error(f"Error getting latest data: {e}")
            return None

    async def get_historical_data(
        self,
        symbol: str,
        channel: str,
        start_time: datetime,
        end_time: datetime
    ) -> List[Dict]:
        """Get historical market data for a symbol and channel."""
        try:
            collection_name = self.config['storage']['mongodb']['collections'].get(channel)
            if not collection_name:
                return []

            collection = self.db[collection_name]
            cursor = collection.find(
                {
                    'symbol': symbol,
                    'timestamp': {
                        '$gte': start_time,
                        '$lte': end_time
                    }
                },
                sort=[('timestamp', ASCENDING)]
            )

            historical_data = await cursor.to_list(length=None)

            if self.config['storage']['compression_enabled']:
                historical_data = [self._decompress_data(data) for data in historical_data]

            return historical_data

        except Exception as e:
            logger.error(f"Error getting historical data: {e}")
            return []

    def _decompress_data(self, data: Dict) -> Dict:
        """Decompress data after retrieval."""
        try:
            if isinstance(data.get('data'), bytes):
                data['data'] = json.loads(gzip.decompress(data['data']).decode())
            return data
        except Exception as e:
            logger.error(f"Error decompressing data: {e}")
            return data

    async def cleanup_old_data(self) -> None:
        """Clean up data older than retention period."""
        try:
            retention_days = self.config['storage']['retention_days']
            cutoff_date = datetime.now() - timedelta(days=retention_days)

            for collection_name in self.config['storage']['mongodb']['collections'].values():
                collection = self.db[collection_name]
                result = await collection.delete_many({'timestamp': {'$lt': cutoff_date}})
                logger.info(f"Deleted {result.deleted_count} old records from {collection_name}")

        except Exception as e:
            logger.error(f"Error cleaning up old data: {e}")
            raise

    async def get_collection_stats(self) -> Dict[str, Dict]:
        """Get statistics about the collections."""
        try:
            stats = {}
            for channel, collection_name in self.config['storage']['mongodb']['collections'].items():
                collection = self.db[collection_name]
                stats[channel] = {
                    'total_documents': await collection.count_documents({}),
                    'size_bytes': (await self.db.command('collstats', collection_name))['size'],
                    'batched_documents': len(self.batch_data.get(collection_name, []))
                }
            return stats
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {}
