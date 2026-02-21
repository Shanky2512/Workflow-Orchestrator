"""
Abstract base class for connector storage.
Defines the contract for persistence implementations.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from models.config import ConnectorConfig


class StorageBase(ABC):
    """
    Abstract base class for connector storage.
    
    All storage implementations must inherit from this class.
    This allows swapping storage backends (filesystem, database, etc.)
    without changing application logic.
    """
    
    @abstractmethod
    def save(self, connector_id: str, data: Dict[str, Any]) -> None:
        """
        Save connector data.
        
        Args:
            connector_id: Unique connector identifier
            data: Connector data to save
            
        Raises:
            ValueError: If save fails
        """
        pass
    
    @abstractmethod
    def load(self, connector_id: str) -> Dict[str, Any]:
        """
        Load connector data.
        
        Args:
            connector_id: Unique connector identifier
            
        Returns:
            Connector data
            
        Raises:
            KeyError: If connector not found
            ValueError: If load fails
        """
        pass
    
    @abstractmethod
    def delete(self, connector_id: str) -> None:
        """
        Delete connector data.
        
        Args:
            connector_id: Unique connector identifier
            
        Raises:
            KeyError: If connector not found
            ValueError: If delete fails
        """
        pass
    
    @abstractmethod
    def exists(self, connector_id: str) -> bool:
        """
        Check if connector exists.
        
        Args:
            connector_id: Unique connector identifier
            
        Returns:
            True if connector exists, False otherwise
        """
        pass
    
    @abstractmethod
    def list_all(self) -> List[str]:
        """
        List all connector IDs.
        
        Returns:
            List of connector IDs
        """
        pass
    
    @abstractmethod
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
        pass
