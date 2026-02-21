"""
Storage package for connector persistence.
"""

from .base import StorageBase
from .filesystem import FilesystemStorage

__all__ = [
    "StorageBase",
    "FilesystemStorage",
]
