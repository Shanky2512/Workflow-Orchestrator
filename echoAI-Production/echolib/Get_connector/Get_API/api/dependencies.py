"""
FastAPI dependencies for dependency injection.
"""

from functools import lru_cache
from storage import FilesystemStorage, StorageBase


@lru_cache()
def get_storage() -> StorageBase:
    """
    Get storage instance (singleton pattern).
    
    Returns:
        StorageBase instance
    """
    return FilesystemStorage(storage_dir="connectors_data")
