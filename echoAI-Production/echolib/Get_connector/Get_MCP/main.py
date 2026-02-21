"""
Unified entry point for MCP Connector Generator.
Supports both CLI and API modes.
"""

import argparse
import asyncio
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def run_cli():
    """Run interactive CLI mode"""
    from cli import main as cli_main
    asyncio.run(cli_main())


def run_api(host: str = "0.0.0.0", port: int = 8000):
    """Run FastAPI server"""
    import uvicorn
    from api import app
    
    logger.info(f"Starting MCP Connector API on {host}:{port}")
    uvicorn.run(app, host=host, port=port)


def run_test_suite():
    """Run test suite for all connectors"""
    import asyncio
    from storage import get_storage
    from tester import ConnectorTester, create_default_test_payload
    from http_script import HTTPMCPConnector
    from sse import SSEMCPConnector
    from stdio import STDIOMCPConnector
    from echolib.Get_connector.Get_MCP.base import TransportType
    
    async def test_all():
        storage = get_storage()
        connectors_data = storage.list_all()
        
        if not connectors_data:
            print("No connectors found to test.")
            return
        
        print(f"\nTesting {len(connectors_data)} connectors...\n")
        
        for conn_meta in connectors_data:
            connector_id = conn_meta["connector_id"]
            data = storage.load(connector_id)
            
            # Recreate connector
            transport = TransportType(data["transport_type"])
            if transport == TransportType.HTTP:
                connector = HTTPMCPConnector.from_dict(data)
            elif transport == TransportType.SSE:
                connector = SSEMCPConnector.from_dict(data)
            elif transport == TransportType.STDIO:
                connector = STDIOMCPConnector.from_dict(data)
            
            # Test
            test_payload = create_default_test_payload(transport.value)
            result = await ConnectorTester.test_connector(connector, test_payload)
            
            status = "✓" if result.success else "✗"
            print(f"{status} {connector.name} ({result.duration_ms}ms)")
            if not result.success:
                print(f"  Error: {result.error}")
            
            # Update storage
            storage.update(connector.serialize())
        
        print("\nTest suite complete.")
    
    asyncio.run(test_all())


def export_connectors(output_path: str):
    """Export all connectors to JSON file"""
    from storage import get_storage
    
    storage = get_storage()
    success = storage.export_all(output_path)
    
    if success:
        print(f"✓ Exported connectors to {output_path}")
    else:
        print(f"✗ Failed to export connectors")


def import_connectors(input_path: str, overwrite: bool = False):
    """Import connectors from JSON file"""
    from storage import get_storage
    
    storage = get_storage()
    count = storage.import_all(input_path, overwrite)
    
    print(f"✓ Imported {count} connectors")


def list_connectors(transport_type: str = None):
    """List all connectors"""
    from storage import get_storage
    
    storage = get_storage()
    connectors = storage.list_all(transport_type)
    
    if not connectors:
        print("No connectors found.")
        return
    
    print(f"\nFound {len(connectors)} connector(s):\n")
    
    for conn in connectors:
        print(f"ID: {conn['connector_id']}")
        print(f"Name: {conn['name']}")
        print(f"Type: {conn['transport_type']}")
        print(f"Status: {conn['status']}")
        print(f"Updated: {conn['updated_at']}")
        print("-" * 60)


def delete_connector(connector_id: str):
    """Delete a connector"""
    from storage import get_storage
    
    storage = get_storage()
    success = storage.delete(connector_id)
    
    if success:
        print(f"✓ Deleted connector {connector_id}")
    else:
        print(f"✗ Connector not found: {connector_id}")


def main():
    """Main entry point with argument parsing"""
    parser = argparse.ArgumentParser(
        description="MCP Connector Generator - Production-grade connector creation and management",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive CLI mode
  python main.py
  
  # Start API server
  python main.py --api
  
  # List all connectors
  python main.py --list
  
  # Test all connectors
  python main.py --test-all
  
  # Export connectors
  python main.py --export connectors.json
  
  # Import connectors
  python main.py --import connectors.json --overwrite
        """
    )
    
    parser.add_argument(
        "--api",
        action="store_true",
        help="Run in API server mode"
    )
    
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="API server host (default: 0.0.0.0)"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="API server port (default: 8000)"
    )
    
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all connectors"
    )
    
    parser.add_argument(
        "--transport",
        choices=["http", "sse", "stdio"],
        help="Filter connectors by transport type"
    )
    
    parser.add_argument(
        "--test-all",
        action="store_true",
        help="Test all connectors"
    )
    
    parser.add_argument(
        "--export",
        metavar="PATH",
        help="Export all connectors to JSON file"
    )
    
    parser.add_argument(
        "--import",
        metavar="PATH",
        dest="import_path",
        help="Import connectors from JSON file"
    )
    
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing connectors during import"
    )
    
    parser.add_argument(
        "--delete",
        metavar="ID",
        help="Delete connector by ID"
    )
    
    parser.add_argument(
        "--storage-dir",
        default="./connectors",
        help="Storage directory for connectors (default: ./connectors)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Set storage directory
    import os
    os.environ["MCP_STORAGE_DIR"] = args.storage_dir
    
    # Route to appropriate mode
    if args.api:
        run_api(args.host, args.port)
    elif args.list:
        list_connectors(args.transport)
    elif args.test_all:
        run_test_suite()
    elif args.export:
        export_connectors(args.export)
    elif args.import_path:
        import_connectors(args.import_path, args.overwrite)
    elif args.delete:
        delete_connector(args.delete)
    else:
        # Default: interactive CLI
        run_cli()


if __name__ == "__main__":
    main()