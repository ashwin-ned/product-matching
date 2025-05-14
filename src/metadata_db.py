# src/metadata_db.py
"""
Load product metadata and provide simple query interface.
"""
import os
import json

# Path to metadata file
METADATA_PATH = os.path.join(
    os.path.dirname(__file__),
    "..", "metadata", "products.json"
)

class MetadataDB:
    def __init__(self, metadata_path: str = METADATA_PATH):
        """
        Initialize MetadataDB by loading JSON metadata into a dict.
        """
        self.metadata_path = metadata_path
        self.products = self._load_metadata()

    def _load_metadata(self) -> dict:
        """
        Load metadata JSON file and return a dict keyed by product ID.
        """
        try:
            with open(self.metadata_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except (IOError, json.JSONDecodeError) as e:
            raise RuntimeError(f"Failed to load metadata: {e}")

        # Expecting list of product dicts with 'id' field
        prod_dict = {}
        for item in data:
            pid = str(item.get('id'))
            prod_dict[pid] = item
        return prod_dict

    def get_product(self, product_id: str) -> dict:
        """
        Retrieve metadata for a given product ID.
        Returns metadata dict or None if not found.
        """
        return self.products.get(str(product_id))

    def list_products(self) -> list:
        """
        Return list of all product metadata dicts.
        """
        return list(self.products.values())

if __name__ == '__main__':
    # Example usage
    db = MetadataDB()
    print(f"Loaded {len(db.products)} products.")
    sample_id = list(db.products.keys())[0]
    print(f"Sample product ({sample_id}):", db.get_product(sample_id))