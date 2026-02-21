"""
Factory for creating connector instances from configuration.
"""

from typing import Dict, Any
from echolib.Get_connector.Get_API.connectors.base import BaseConnector
from echolib.Get_connector.Get_API.connectors.http import HTTPConnector
from echolib.Get_connector.Get_API.models.config import ConnectorConfig


class ConnectorFactory:
    """
    Factory for creating connector instances.
    
    Handles instantiation of the appropriate connector type based on configuration.
    Currently supports HTTP connectors, but designed to be extensible.
    """
    
    @staticmethod
    def create(config: ConnectorConfig) -> BaseConnector:
        """
        Create a connector instance from configuration.
        
        Args:
            config: Connector configuration
            
        Returns:
            BaseConnector instance (HTTPConnector currently)
            
        Raises:
            ValueError: If connector type is unsupported
        """
        # Currently only HTTP connectors are supported
        # Future: Could add WebSocket, GraphQL, gRPC connectors
        return HTTPConnector(config=config)
    
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> BaseConnector:
        """
        Create a connector from dictionary representation.
        
        Args:
            data: Dictionary containing connector data
            
        Returns:
            BaseConnector instance
            
        Raises:
            ValueError: If data is invalid
        """
        # Currently assumes HTTP connector
        # Future: Could inspect data to determine connector type
        return HTTPConnector.from_dict(data)
