"""
Filesystem-based storage implementation.
Stores connectors as JSON files in a directory.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime
from .base import StorageBase


class FilesystemStorage(StorageBase):
    """
    Filesystem-based storage for connectors.
    
    Each connector is stored as a separate JSON file.
    Includes metadata like creation and update timestamps.
    Thread-safe for single-process usage.
    """
    
    def __init__(self, storage_dir: str = "connectors_data") -> None:
        """
        Initialize filesystem storage.
        
        Args:
            storage_dir: Directory to store connector files
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_filepath(self, connector_id: str) -> Path:
        """
        Get filepath for a connector.
        
        Args:
            connector_id: Connector identifier
            
        Returns:
            Path to connector file
        """
        safe_id = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in connector_id)
        return self.storage_dir / f"{safe_id}.json"
    
    def _add_metadata(self, data: Dict[str, Any], is_new: bool = True) -> Dict[str, Any]:
        """
        Add metadata to connector data.
        
        Args:
            data: Connector data
            is_new: Whether this is a new connector
            
        Returns:
            Data with metadata added
        """
        now = datetime.utcnow().isoformat()
        
        if is_new:
            data["created_at"] = now
        
        data["updated_at"] = now
        return data
    
    def save(self, connector_id: str, data: Dict[str, Any]) -> None:
        """
        Save connector data to filesystem.
        
        Args:
            connector_id: Unique connector identifier
            data: Connector data to save
            
        Raises:
            ValueError: If save fails
        """
        try:
            filepath = self._get_filepath(connector_id)
            is_new = not filepath.exists()
            
            data_with_metadata = self._add_metadata(data.copy(), is_new=is_new)
            
            with open(filepath, 'w') as f:
                json.dump(data_with_metadata, f, indent=2)
        
        except Exception as e:
            raise ValueError(f"Failed to save connector {connector_id}: {str(e)}")
    
    def load(self, connector_id: str) -> Dict[str, Any]:
        """
        Load connector data from filesystem.
        
        Args:
            connector_id: Unique connector identifier
            
        Returns:
            Connector data
            
        Raises:
            KeyError: If connector not found
            ValueError: If load fails
        """
        filepath = self._get_filepath(connector_id)
        
        if not filepath.exists():
            raise KeyError(f"Connector {connector_id} not found")
        
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        
        except Exception as e:
            raise ValueError(f"Failed to load connector {connector_id}: {str(e)}")
    
    def delete(self, connector_id: str) -> None:
        """
        Delete connector data from filesystem.
        
        Args:
            connector_id: Unique connector identifier
            
        Raises:
            KeyError: If connector not found
            ValueError: If delete fails
        """
        filepath = self._get_filepath(connector_id)
        
        if not filepath.exists():
            raise KeyError(f"Connector {connector_id} not found")
        
        try:
            filepath.unlink()
        
        except Exception as e:
            raise ValueError(f"Failed to delete connector {connector_id}: {str(e)}")
    
    def exists(self, connector_id: str) -> bool:
        """
        Check if connector exists.
        
        Args:
            connector_id: Unique connector identifier
            
        Returns:
            True if connector exists, False otherwise
        """
        filepath = self._get_filepath(connector_id)
        return filepath.exists()
    
    def list_all(self) -> List[str]:
        """
        List all connector IDs.
        
        Returns:
            List of connector IDs
        """
        connector_files = list(self.storage_dir.glob("*.json"))
        return [f.stem for f in connector_files]
    
    def update(self, connector_id: str, data: Dict[str, Any]) -> None:
        """
        Update existing connector data.
        
        Args:
            connector_id: Unique connector identifier
            data: Updated connector data
            
        Raises:
            KeyError: If connector not found
            ValueError: If update fails
        """
        if not self.exists(connector_id):
            raise KeyError(f"Connector {connector_id} not found")
        
        try:
            # Load existing data to preserve created_at
            existing_data = self.load(connector_id)
            data["created_at"] = existing_data.get("created_at")
            
            self.save(connector_id, data)
        
        except KeyError:
            raise
        except Exception as e:
            raise ValueError(f"Failed to update connector {connector_id}: {str(e)}")
    
    def load_all(self) -> Dict[str, Dict[str, Any]]:
        """
        Load all connectors.
        
        Returns:
            Dictionary mapping connector IDs to their data
        """
        result = {}
        for connector_id in self.list_all():
            try:
                result[connector_id] = self.load(connector_id)
            except Exception:
                continue
        
        return result
