#!/usr/bin/env python3
"""
Test script for the API Connector Framework.
Demonstrates creating, testing, and executing connectors programmatically.
"""

import json
from models.config import (
    ConnectorConfig,
    NoAuthConfig,
    BearerTokenAuthConfig,
    ApiKeyAuthConfig,
    ApiKeyLocation
)
from connectors import ConnectorFactory
from storage import FilesystemStorage


def print_section(title: str) -> None:
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def test_httpbin_connector() -> None:
    """Test with HTTPBin - public API with no auth."""
    print_section("Test 1: HTTPBin (No Authentication)")
    
    # Create configuration
    config = ConnectorConfig(
        id="test-httpbin",
        name="HTTPBin Test",
        description="Test connector for HTTPBin API",
        base_url="https://httpbin.org",
        auth=NoAuthConfig(),
        timeout=30.0
    )
    
    print(f"Creating connector: {config.name}")
    print(f"Base URL: {config.base_url}")
    print(f"Auth Type: {config.auth.type.value}")
    
    # Create connector
    connector = ConnectorFactory.create(config)
    
    # Test connection
    print("\nTesting connection...")
    result = connector.test_connection()
    print(f"✓ Connection test: {'SUCCESS' if result.success else 'FAILED'}")
    print(f"  Status Code: {result.status_code}")
    print(f"  Elapsed: {result.elapsed_seconds:.3f}s")
    
    # Execute GET request
    print("\nExecuting GET /get...")
    result = connector.execute(
        method="GET",
        endpoint="/get",
        query_params={"test": "value", "foo": "bar"}
    )
    print(f"✓ Request: {'SUCCESS' if result.success else 'FAILED'}")
    print(f"  Status Code: {result.status_code}")
    print(f"  Elapsed: {result.elapsed_seconds:.3f}s")
    if result.body:
        print(f"  Response (partial): {json.dumps(result.body, indent=2)[:200]}...")
    
    # Execute POST request
    print("\nExecuting POST /post...")
    result = connector.execute(
        method="POST",
        endpoint="/post",
        body={"message": "Hello from API Connector Framework!", "test": True},
        headers={"X-Custom-Header": "test-value"}
    )
    print(f"✓ Request: {'SUCCESS' if result.success else 'FAILED'}")
    print(f"  Status Code: {result.status_code}")
    print(f"  Elapsed: {result.elapsed_seconds:.3f}s")
    
    # Save connector
    storage = FilesystemStorage()
    storage.save(config.id, connector.to_dict())
    print(f"\n✓ Connector saved to storage")


def test_storage_operations() -> None:
    """Test storage operations."""
    print_section("Test 2: Storage Operations")
    
    storage = FilesystemStorage()
    
    # List all connectors
    connectors = storage.list_all()
    print(f"Total connectors in storage: {len(connectors)}")
    
    for connector_id in connectors:
        print(f"  - {connector_id}")
    
    # Load a connector
    if "test-httpbin" in connectors:
        print("\nLoading test-httpbin connector...")
        data = storage.load("test-httpbin")
        print(f"✓ Loaded connector")
        print(f"  Created: {data.get('created_at', 'N/A')}")
        print(f"  Updated: {data.get('updated_at', 'N/A')}")
        
        # Reload connector and test
        connector = ConnectorFactory.from_dict(data)
        result = connector.execute(method="GET", endpoint="/uuid")
        print(f"\n✓ Quick test: {'SUCCESS' if result.success else 'FAILED'}")


def test_serialization() -> None:
    """Test connector serialization and deserialization."""
    print_section("Test 3: Serialization")
    
    # Create a complex configuration
    config = ConnectorConfig(
        id="test-serialization",
        name="Serialization Test",
        description="Test connector serialization",
        base_url="https://api.example.com",
        auth=ApiKeyAuthConfig(
            key="test-api-key-12345",
            location=ApiKeyLocation.HEADER,
            param_name="X-API-Key"
        ),
        default_headers={"User-Agent": "TestAgent/1.0"},
        timeout=45.0
    )
    
    print(f"Original config ID: {config.id}")
    print(f"Original auth type: {config.auth.type.value}")
    
    # Create connector
    connector = ConnectorFactory.create(config)
    
    # Serialize
    serialized = connector.to_dict()
    print(f"\n✓ Serialized to dictionary")
    print(f"  Keys: {list(serialized.keys())}")
    
    # Deserialize
    restored_connector = ConnectorFactory.from_dict(serialized)
    restored_config = restored_connector.get_config()
    
    print(f"\n✓ Deserialized from dictionary")
    print(f"  Restored ID: {restored_config.id}")
    print(f"  Restored auth type: {restored_config.auth.type.value}")
    print(f"  Restored base URL: {restored_config.base_url}")
    
    # Verify
    assert restored_config.id == config.id
    assert restored_config.name == config.name
    assert restored_config.base_url == config.base_url
    print(f"\n✓ Serialization integrity verified")


def test_error_handling() -> None:
    """Test error handling."""
    print_section("Test 4: Error Handling")
    
    config = ConnectorConfig(
        id="test-errors",
        name="Error Test",
        base_url="https://httpbin.org",
        auth=NoAuthConfig()
    )
    
    connector = ConnectorFactory.create(config)
    
    # Test 404
    print("Testing 404 error...")
    result = connector.execute(method="GET", endpoint="/status/404")
    print(f"  Status: {result.status_code}")
    print(f"  Success: {result.success}")
    print(f"  Error: {result.error}")
    
    # Test 500
    print("\nTesting 500 error...")
    result = connector.execute(method="GET", endpoint="/status/500")
    print(f"  Status: {result.status_code}")
    print(f"  Success: {result.success}")
    print(f"  Error: {result.error}")
    
    # Test timeout (will fail quickly if httpbin delays)
    print("\nTesting timeout...")
    result = connector.execute(
        method="GET",
        endpoint="/delay/1",
        timeout_override=0.5
    )
    print(f"  Status: {result.status_code}")
    print(f"  Success: {result.success}")
    if result.error:
        print(f"  Error handled: {result.error[:50]}...")


def test_ai_agent_pattern() -> None:
    """Demonstrate AI agent usage pattern."""
    print_section("Test 5: AI Agent Pattern")
    
    print("Simulating AI agent workflow...")
    
    # Agent loads connector from storage
    storage = FilesystemStorage()
    
    if not storage.exists("test-httpbin"):
        print("Creating test connector first...")
        config = ConnectorConfig(
            id="test-httpbin",
            name="HTTPBin Test",
            base_url="https://httpbin.org",
            auth=NoAuthConfig()
        )
        connector = ConnectorFactory.create(config)
        storage.save(config.id, connector.to_dict())
    
    print("1. Agent loads connector from storage")
    data = storage.load("test-httpbin")
    connector = ConnectorFactory.from_dict(data)
    print("   ✓ Loaded")
    
    print("\n2. Agent determines API call needed")
    print("   Task: Get current IP address")
    print("   Endpoint: /ip")
    
    print("\n3. Agent executes stateless call")
    result = connector.execute(method="GET", endpoint="/ip")
    print(f"   ✓ Executed")
    
    print("\n4. Agent parses response")
    if result.success:
        print(f"   Status: {result.status_code}")
        print(f"   Body: {result.body}")
        print("   ✓ Success - agent can proceed with result")
    else:
        print(f"   Error: {result.error}")
        print("   ✓ Failure - agent handles error gracefully")


def main() -> None:
    """Run all tests."""
    print("""
╔════════════════════════════════════════════════════════════════╗
║        API Connector Framework - Test Suite                   ║
╚════════════════════════════════════════════════════════════════╝
    """)
    
    try:
        test_httpbin_connector()
        test_storage_operations()
        test_serialization()
        test_error_handling()
        test_ai_agent_pattern()
        
        print("\n" + "=" * 70)
        print("  All Tests Completed Successfully!")
        print("=" * 70)
        print("""
Next steps:
  1. Run the CLI:        python main.py
  2. Run the API server: python main.py --api
  3. Check the docs:     http://localhost:8000/docs
        """)
    
    except Exception as e:
        print(f"\n✗ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
