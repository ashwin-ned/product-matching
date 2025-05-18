"""
MongoDB metadata store for product information.
Uses a products collection keyed by product_id.
"""
import os, random
import json
from pymongo import MongoClient
from pymongo import server_api
from pymongo.collection import Collection
from pymongo.errors import PyMongoError
from dotenv import load_dotenv
from mongodb_logger import log_event

load_dotenv()
# Configuration via environment variables
MONGO_URI        = os.getenv("MONGO_URI")
MONGO_DB_NAME    = os.getenv("MONGO_DB", "productdb")
MONGO_COLLECTION = os.getenv("MONGO_COLLECTION", "products")

class MongoDB:
    def __init__(self,
                 uri: str = MONGO_URI,
                 db_name: str = MONGO_DB_NAME,
                 collection_name: str = MONGO_COLLECTION):
        """
        Initialize MongoClient, database, and collection.
        """
        try:
            self.client: MongoClient = MongoClient(uri, server_api=server_api.ServerApi('1'))
            # Trigger connection on init
            self.client.server_info()
            self.db = self.client[db_name]
            self.collection: Collection = self.db[collection_name]
        except PyMongoError as e:
            log_event("error", f"MongoDB connection error: {e}")
            raise RuntimeError(f"Failed to connect to MongoDB: {e}")
        
        
    def add_product(self, product_id: str, metadata: dict) -> None:
        """
        Insert or update product metadata.
        Upsert ensures existing docs are updated.
        """
        try:
            self.collection.update_one(
                {"_id": product_id},
                {"$set": metadata},
                upsert=True
            )
        except PyMongoError as e:
            log_event("error", f"Failed to add/update product {product_id}: {e}")
            raise RuntimeError(f"Failed to add/update product {product_id}: {e}")

    def get_product(self, product_id: str) -> dict:
        """
        Retrieve metadata dict for a given product_id.
        Returns {} if not found.
        """
        try:
            doc = self.collection.find_one({"_id": product_id}, {"_id": 0})
            return doc or {}
        except PyMongoError as e:
            raise RuntimeError(f"Failed to retrieve product {product_id}: {e}")

    def list_products(self, limit: int = 100) -> list:
        """
        (Optional) List up to `limit` products for inspection.
        """
        try:
            cursor = self.collection.find({}, {"_id": 1}).limit(limit)
            return [str(doc["_id"]) for doc in cursor]
        except PyMongoError as e:
            log_event("error", f"Failed to list products: {e}")
            raise RuntimeError(f"Failed to list products: {e}")
        
if __name__ == "__main__":
    # Example usage
    db = MongoDB()
    print(f"Connected to MongoDB database '{MONGO_DB_NAME}'")
    
    # Load metadata from products.json
    with open("../metadata/products.json", "r", encoding="utf-8") as f:
        products = json.load(f)
    print(f"Loaded {len(products)} products from products.json")

    # Ingest all products
    for product in products:
        pid = product["id"]
        db.add_product(pid, product)
    print(f"Ingested {len(products)} products into MongoDB.")

    # Verify by retrieving random 3 entries
    random_ids = random.sample([p["id"] for p in products], k=min(3, len(products)))
    print("Random sample retrieval:")
    for pid in random_ids:
        result = db.get_product(pid)
        print(f"  ID {pid} → {result.get('name', 'Not found')}")
