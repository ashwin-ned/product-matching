"""
mongodb_logger.py
-----------------
A simple MongoDB logger for storing logs, errors, and execution results.
"""

import os
from dotenv import load_dotenv
import logging
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional
from pymongo import MongoClient

# Configure basic logging as fallback
logging.basicConfig(level=logging.INFO)

load_dotenv()
# Environment variables with defaults
MONGO_URI = os.getenv("LOGGER_MONGO_URI")
DB_NAME = os.getenv("LOGGER_DB_NAME")
COLLECTION_NAME = os.getenv("LOGGER_MONGO_COLLECTION")

print(f"MongoDB URI: {MONGO_URI}, DB: {DB_NAME}, Collection: {COLLECTION_NAME}")

# Initialize MongoDB connection
try:
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]
except Exception as e:
    logging.error(f"Failed to connect to MongoDB: {e}")
    collection = None  # Will fall back to standard logging

def log_event(level: str, message: str, extra: dict = None) -> Optional[str]:
    """
    Synchronously log an event to MongoDB with timestamp
    """
    log_entry = {
        "timestamp": datetime.now(timezone.utc),
        "level": level.upper(),
        "message": message,
        "extra": extra or {}
    }

    try:
        if collection is not None:
            result = collection.insert_one(log_entry)
            return str(result.inserted_id)
        else:
            # Fallback to standard logging with proper level conversion
            level_num = getattr(logging, level.upper(), logging.INFO)
            logging.log(level_num, f"{message} - {extra}")
            return None
    except Exception as e:
        logging.error(f"Logging failed: {e}")
        return None

def get_logs(limit: int = 10) -> List[Dict]:
    """
    Retrieve recent logs from MongoDB
    """
    try:
        if collection is None:
            logging.warning("No MongoDB connection available")
            return []
            
        logs = list(collection.find().sort("timestamp", -1).limit(limit))
        # Convert ObjectId to string for serialization
        for log in logs:
            log["_id"] = str(log["_id"])
        return logs
    except Exception as e:
        logging.error(f"Failed to retrieve logs: {e}")
        return []

if __name__ == "__main__":
    # Example logging
    print("\nLogging test messages:")
    log_event("info", "System initialization started", {"module": "boot"})
    log_event("warning", "Low disk space warning", {"module": "storage", "free_space": "5GB"})
    log_event("error", "Database connection timeout", {"module": "database", "attempts": 3})
    log_event("info", "Service startup completed", {"module": "services"})


    # Retrieve and display logs
    print("\nRetrieving recent logs:")
    recent_logs = get_logs(limit=5)
    
    for log in recent_logs:
        timestamp = log["timestamp"].strftime("%Y-%m-%d %H:%M:%S%z")
        print(f"[{timestamp}] {log['level']}: {log['message']}")
        print(f"    Extra: {log.get('extra', {})}")
        print("-" * 60)