"""
Database optimization script for creating and managing indexes.
"""

import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
import logging
from typing import List, Dict, Any
import time

from core.config import settings

logger = logging.getLogger(__name__)

async def create_indexes(db: AsyncIOMotorClient) -> None:
    """Create optimized indexes for frequently accessed collections."""
    try:
        # User collection indexes
        await db.users.create_index("email", unique=True)
        await db.users.create_index("username", unique=True)
        await db.users.create_index([
            ("last_login", -1),
            ("is_active", 1)
        ])

        # Security events collection indexes
        await db.security_events.create_index([
            ("timestamp", -1),
            ("event_type", 1)
        ])
        await db.security_events.create_index([
            ("user_id", 1),
            ("timestamp", -1)
        ])

        # Market data collection indexes
        await db.market_data.create_index([
            ("symbol", 1),
            ("timestamp", -1)
        ])
        await db.market_data.create_index([
            ("timestamp", -1),
            ("type", 1)
        ])

        # Positions collection indexes
        await db.positions.create_index([
            ("user_id", 1),
            ("status", 1),
            ("timestamp", -1)
        ])
        await db.positions.create_index([
            ("symbol", 1),
            ("status", 1),
            ("timestamp", -1)
        ])

        # Alerts collection indexes
        await db.alerts.create_index([
            ("user_id", 1),
            ("status", 1),
            ("timestamp", -1)
        ])
        await db.alerts.create_index([
            ("type", 1),
            ("severity", 1),
            ("timestamp", -1)
        ])

        logger.info("Successfully created database indexes")
    except Exception as e:
        logger.error(f"Error creating indexes: {e}")
        raise

async def analyze_query_performance(
    db: AsyncIOMotorClient,
    collection_name: str,
    query: Dict[str, Any]
) -> Dict[str, Any]:
    """Analyze query performance using explain plan."""
    try:
        collection = db[collection_name]
        explain_result = await collection.find(query).explain()
        
        # Extract relevant metrics
        execution_stats = explain_result.get("executionStats", {})
        
        return {
            "collection": collection_name,
            "query": query,
            "execution_time_ms": execution_stats.get("executionTimeMillis"),
            "docs_examined": execution_stats.get("totalDocsExamined"),
            "docs_returned": execution_stats.get("nReturned"),
            "index_used": execution_stats.get("indexesUsed", []),
            "in_memory_sort": execution_stats.get("sortInMemory", False)
        }
    except Exception as e:
        logger.error(
            f"Error analyzing query performance for {collection_name}: {e}"
        )
        return {}

async def monitor_index_usage(
    db: AsyncIOMotorClient
) -> List[Dict[str, Any]]:
    """Monitor index usage statistics."""
    try:
        index_stats = []
        collections = await db.list_collection_names()
        
        for collection in collections:
            stats = await db.command({
                "aggregate": collection,
                "pipeline": [{"$indexStats": {}}],
                "cursor": {}
            })
            
            for stat in stats["cursor"]["firstBatch"]:
                index_stats.append({
                    "collection": collection,
                    "index_name": stat["name"],
                    "accesses": stat.get("accesses", {}).get("ops", 0),
                    "since": stat.get("accesses", {}).get("since")
                })
        
        return index_stats
    except Exception as e:
        logger.error(f"Error monitoring index usage: {e}")
        return []

async def optimize_indexes(db: AsyncIOMotorClient) -> None:
    """Optimize database indexes based on usage statistics."""
    try:
        # Get current index usage
        index_stats = await monitor_index_usage(db)
        
        # Identify unused indexes
        unused_indexes = [
            stat for stat in index_stats
            if stat["accesses"] == 0
        ]
        
        # Log unused indexes for review
        if unused_indexes:
            logger.warning(
                f"Found {len(unused_indexes)} unused indexes: "
                f"{unused_indexes}"
            )
        
        # Analyze slow queries
        slow_queries = await db.system.profile.find({
            "millis": {"$gt": 100}
        }).to_list(None)
        
        if slow_queries:
            logger.warning(
                f"Found {len(slow_queries)} slow queries: {slow_queries}"
            )
        
        # Create new indexes for slow queries if needed
        # This should be done carefully and during off-peak hours
        
    except Exception as e:
        logger.error(f"Error optimizing indexes: {e}")
        raise

async def main():
    """Main function to run database optimization."""
    client = AsyncIOMotorClient(settings.MONGODB_URL)
    db = client[settings.MONGO_DB]
    
    try:
        # Create initial indexes
        await create_indexes(db)
        
        # Monitor and optimize
        while True:
            # Analyze index usage
            index_stats = await monitor_index_usage(db)
            logger.info(f"Current index usage: {index_stats}")
            
            # Optimize indexes if needed
            await optimize_indexes(db)
            
            # Wait before next analysis
            await asyncio.sleep(3600)  # Run every hour
            
    except KeyboardInterrupt:
        logger.info("Stopping database optimization")
    finally:
        client.close()

if __name__ == "__main__":
    asyncio.run(main())
