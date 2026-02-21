"""
Interactive CLI for MCP connector creation and management.
Provides user-friendly interface for building connectors.
"""

import asyncio
import sys
import json
from typing import Dict, Any, Optional
import logging

from .base import TransportType, AuthType
from .http_script import HTTPMCPConnector
from .sse import SSEMCPConnector
from .stdio import STDIOMCPConnector
from .validator import validate_and_normalize, ValidationError
from .storage import get_storage
from .tester import ConnectorTester, create_default_test_payload

logger = logging.getLogger(__name__)


class CLI:
    """Interactive command-line interface for connector management"""
    
    def __init__(self):
        self.storage = get_storage()
        
    def print_header(self, text: str):
        """Print section header"""
        print(f"\n{'='*60}")
        print(f"  {text}")
        print(f"{'='*60}\n")
        
    def print_success(self, text: str):
        """Print success message"""
        print(f"✓ {text}")
        
    def print_error(self, text: str):
        """Print error message"""
        print(f"✗ {text}")
        
    def print_info(self, text: str):
        """Print info message"""
        print(f"ℹ {text}")
        
    def get_input(self, prompt: str, default: Optional[str] = None) -> str:
        """Get user input with optional default"""
        if default:
            prompt = f"{prompt} [{default}]"
        prompt = f"{prompt}: "
        
        value = input(prompt).strip()
        return value if value else (default or "")
    
    def get_choice(self, prompt: str, options: list) -> str:
        """Get user choice from list of options"""
        print(f"\n{prompt}")
        for i, option in enumerate(options, 1):
            print(f"  {i}. {option}")
        
        while True:
            choice = input("\nSelect option (number): ").strip()
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(options):
                    return options[idx]
                else:
                    print("Invalid choice. Try again.")
            except ValueError:
                print("Please enter a number.")
    
    def get_yes_no(self, prompt: str, default: bool = False) -> bool:
        """Get yes/no confirmation"""
        default_str = "Y/n" if default else "y/N"
        response = input(f"{prompt} ({default_str}): ").strip().lower()
        
        if not response:
            return default
        return response in ['y', 'yes']
    
    def collect_base_fields(self) -> Dict[str, Any]:
        """Collect fields common to all connectors"""
        self.print_header("Basic Connector Information")
        
        name = self.get_input("Connector name (alphanumeric, -, _)")
        description = self.get_input("Description")
        
        # Select transport type
        transport = self.get_choice(
            "Select transport type",
            [t.value for t in TransportType]
        )
        
        return {
            "name": name,
            "description": description,
            "transport_type": transport
        }
    
    def collect_auth_config(self) -> Dict[str, Any]:
        """Collect authentication configuration"""
        self.print_header("Authentication Configuration")
        
        use_auth = self.get_yes_no("Does this connector require authentication?", False)
        
        if not use_auth:
            return {"type": "none"}
        
        auth_type = self.get_choice(
            "Select authentication type",
            [t.value for t in AuthType if t != AuthType.NONE]
        )
        
        auth_config = {"type": auth_type}
        
        if auth_type == "api_key":
            auth_config["api_key"] = self.get_input("API Key")
            auth_config["key_name"] = self.get_input("Key name (e.g., 'X-API-Key')")
            location = self.get_choice("Key location", ["header", "query"])
            auth_config["location"] = location
            
        elif auth_type == "bearer":
            auth_config["token"] = self.get_input("Bearer token")
            
        elif auth_type == "custom_header":
            auth_config["header_name"] = self.get_input("Header name")
            auth_config["header_value"] = self.get_input("Header value")
            
        elif auth_type == "basic":
            auth_config["username"] = self.get_input("Username")
            auth_config["password"] = self.get_input("Password")
            
        elif auth_type == "oauth2":
            auth_config["access_token"] = self.get_input("Access token")
            
        return auth_config
    
    def collect_schemas(self) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """Collect input and output schemas"""
        self.print_header("Input/Output Schemas")
        
        self.print_info("Enter schemas as JSON. Example: {\"type\": \"object\", \"properties\": {...}}")
        
        # Input schema
        while True:
            input_str = self.get_input("Input schema (JSON)", '{"type": "object"}')
            try:
                input_schema = json.loads(input_str)
                break
            except json.JSONDecodeError as e:
                self.print_error(f"Invalid JSON: {e}")
        
        # Output schema
        while True:
            output_str = self.get_input("Output schema (JSON)", '{"type": "object"}')
            try:
                output_schema = json.loads(output_str)
                break
            except json.JSONDecodeError as e:
                self.print_error(f"Invalid JSON: {e}")
        
        return input_schema, output_schema
    
    def collect_http_config(self) -> Dict[str, Any]:
        """Collect HTTP-specific configuration"""
        self.print_header("HTTP Configuration")
        
        endpoint_url = self.get_input("Endpoint URL (https://...)")
        method = self.get_choice(
            "HTTP method",
            ["GET", "POST", "PUT", "PATCH", "DELETE"]
        )
        
        # Optional headers
        add_headers = self.get_yes_no("Add custom headers?", False)
        headers = {}
        if add_headers:
            while True:
                key = self.get_input("Header name (or empty to finish)")
                if not key:
                    break
                value = self.get_input(f"Value for {key}")
                headers[key] = value
        
        timeout = int(self.get_input("Timeout (seconds)", "30"))
        
        return {
            "endpoint_url": endpoint_url,
            "method": method,
            "headers": headers,
            "timeout": timeout
        }
    
    def collect_sse_config(self) -> Dict[str, Any]:
        """ Collect SSE-specific configuration """

        self.print_header("SSE Configuration")
        
        endpoint_url = self.get_input("SSE endpoint URL (https://...)")
        
        # Optional headers
        add_headers = self.get_yes_no("Add custom headers?", False)
        headers = {}
        if add_headers:
            while True:
                key = self.get_input("Header name (or empty to finish)")
                if not key:
                    break
                value = self.get_input(f"Value for {key}")
                headers[key] = value
        
        reconnect = self.get_yes_no("Enable auto-reconnect?", True)
        max_attempts = 5
        if reconnect:
            max_attempts = int(self.get_input("Max reconnect attempts", "5"))
        
        return {
            "endpoint_url": endpoint_url,
            "headers": headers,
            "reconnect": reconnect,
            "max_reconnect_attempts": max_attempts
        }

    def collect_stdio_config(self) -> Dict[str, Any]:
        """Collect STDIO-specific configuration"""
        self.print_header("STDIO Configuration")
        
        command = self.get_input("Command to execute (path to executable)")
        
        # Args
        add_args = self.get_yes_no("Add command-line arguments?", False)
        args = []
        if add_args:
            while True:
                arg = self.get_input("Argument (or empty to finish)")
                if not arg:
                    break
                args.append(arg)
        
        # Environment variables
        add_env = self.get_yes_no("Add environment variables?", False)
        env_vars = {}
        if add_env:
            while True:
                key = self.get_input("Variable name (or empty to finish)")
                if not key:
                    break
                value = self.get_input(f"Value for {key}")
                env_vars[key] = value
        
        timeout = int(self.get_input("Timeout (seconds)", "30"))
        
        return {
            "command": command,
            "args": args,
            "env_vars": env_vars,
            "timeout": timeout
        }
    
    async def create_connector_interactive(self) -> Optional[str]:
        """Create connector through interactive prompts"""
        try:
            # Collect base fields
            data = self.collect_base_fields()
            
            # Collect auth config
            data["auth_config"] = self.collect_auth_config()
            
            # Collect schemas
            input_schema, output_schema = self.collect_schemas()
            data["input_schema"] = input_schema
            data["output_schema"] = output_schema
            
            # Collect transport-specific config
            transport = data["transport_type"]
            if transport == "http":
                data.update(self.collect_http_config())
            elif transport == "sse":
                data.update(self.collect_sse_config())
            elif transport == "stdio":
                data.update(self.collect_stdio_config())
            
            # Validate and create
            self.print_header("Creating Connector")
            
            normalized_data = validate_and_normalize(data)
            
            # Create connector based on type
            if transport == "http":
                connector = HTTPMCPConnector(**{
                    k: v for k, v in normalized_data.items()
                    if k in HTTPMCPConnector.__init__.__code__.co_varnames
                })
            elif transport == "sse":
                connector = SSEMCPConnector(**{
                    k: v for k, v in normalized_data.items()
                    if k in SSEMCPConnector.__init__.__code__.co_varnames
                })
            elif transport == "stdio":
                connector = STDIOMCPConnector(**{
                    k: v for k, v in normalized_data.items()
                    if k in STDIOMCPConnector.__init__.__code__.co_varnames
                })
            
            # Validate
            is_valid, errors = connector.validate_config()
            if not is_valid:
                self.print_error("Validation failed:")
                for error in errors:
                    print(f"  - {error}")
                return None
            
            self.print_success("Connector created successfully")
            self.print_info(f"ID: {connector.connector_id}")
            self.print_info(f"Name: {connector.name}")
            
            # Save to storage immediately
            self.storage.save(connector.serialize())
            
            return connector.connector_id
            
        except ValidationError as e:
            self.print_error("Validation failed:")
            for error in e.errors:
                print(f"  - {error}")
            return None
        except Exception as e:
            self.print_error(f"Failed to create connector: {e}")
            logger.error("Connector creation failed", exc_info=True)
            return None
    
    async def test_connector_interactive(self, connector_id: str):
        """Test connector interactively"""
        try:
            self.print_header("Testing Connector")
            
            # Load connector
            data = self.storage.load(connector_id)
            if not data:
                self.print_error(f"Connector not found: {connector_id}")
                return
            
            transport = TransportType(data["transport_type"])
            
            # Recreate connector
            if transport == TransportType.HTTP:
                connector = HTTPMCPConnector.from_dict(data)
            elif transport == TransportType.SSE:
                connector = SSEMCPConnector.from_dict(data)
            elif transport == TransportType.STDIO:
                connector = STDIOMCPConnector.from_dict(data)
            
            # Get test payload
            use_custom = self.get_yes_no("Use custom test payload?", False)
            
            if use_custom:
                payload_str = self.get_input("Test payload (JSON)")
                test_payload = json.loads(payload_str)
            else:
                test_payload = create_default_test_payload(transport.value)
                self.print_info(f"Using default test payload: {json.dumps(test_payload)}")
            
            # Run test
            self.print_info("Executing test...")
            result = await ConnectorTester.test_connector(connector, test_payload)
            
            # Display result
            print()
            if result.success:
                self.print_success(f"Test passed ({result.duration_ms}ms)")
                print(f"\nOutput:")
                print(json.dumps(result.output, indent=2))
            else:
                self.print_error(f"Test failed ({result.duration_ms}ms)")
                print(f"\nError: {result.error}")
            
        except Exception as e:
            self.print_error(f"Test failed: {e}")
            logger.error("Test execution failed", exc_info=True)
    
    async def save_connector_interactive(self, connector_id: str):
        """Save connector to storage"""
        try:
            data = self.storage.load(connector_id)
            if not data:
                self.print_error(f"Connector not found: {connector_id}")
                return
            
            # Already saved, just confirm
            self.print_success(f"Connector {connector_id} is saved")
            self.print_info(f"Location: {self.storage._get_connector_path(connector_id)}")
            
        except Exception as e:
            self.print_error(f"Failed to save: {e}")
    
    async def delete_connector_interactive(self, connector_id: str) -> bool:
        """Delete connector from storage"""
        try:
            # Check if connector exists
            data = self.storage.load(connector_id)
            if not data:
                self.print_error(f"Connector not found: {connector_id}")
                return False
            
            # Show connector info and ask for confirmation
            self.print_info(f"Connector: {data['name']}")
            self.print_info(f"ID: {connector_id}")
            self.print_info(f"Type: {data['transport_type']}")
            
            confirm = self.get_yes_no(f"\nAre you sure you want to delete this connector?", False)
            
            if not confirm:
                self.print_info("Deletion cancelled")
                return False
            
            # Delete
            success = self.storage.delete(connector_id)
            
            if success:
                self.print_success(f"Connector {connector_id} deleted successfully")
                logger.info(f"Deleted connector: {connector_id}")
                return True
            else:
                self.print_error(f"Failed to delete connector {connector_id}")
                return False
                
        except Exception as e:
            self.print_error(f"Failed to delete connector: {e}")
            logger.error(f"Deletion failed: {e}", exc_info=True)
            return False
    
    async def run(self):
        """Main CLI loop"""
        self.print_header("MCP Connector Generator CLI")
        
        # Main flow
        connector_id = await self.create_connector_interactive()
        
        if not connector_id:
            print("\nExiting due to creation failure.")
            return
        
        # Menu loop
        while True:
            print("\n" + "="*60)
            print("What would you like to do next?")
            print("="*60)
            
            choice = self.get_choice(
                "",
                ["Test connector", "Save connector", "Configure connector", "Delete connector", "Exit"]
            )
            
            if choice == "Test connector":
                await self.test_connector_interactive(connector_id)
                
            elif choice == "Save connector":
                await self.save_connector_interactive(connector_id)
                
            elif choice == "Configure connector":
                self.print_info("Configuration editing not yet implemented")
                
            elif choice == "Delete connector":
                deleted = await self.delete_connector_interactive(connector_id)
                if deleted:
                    self.print_success("\nDone!")
                    return
                    
            elif choice == "Exit":
                self.print_success("\nDone!")
                return


async def main():
    """CLI entry point"""
    # Configure logging
    logging.basicConfig(
        level=logging.WARNING,  # Less verbose for CLI
        format="%(message)s"
    )
    
    cli = CLI()
    await cli.run()


if __name__ == "__main__":
    asyncio.run(main())