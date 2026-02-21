"""
Command-line interface for connector management.
Provides an interactive terminal interface for creating and testing connectors.
"""

import json
import sys
from typing import Dict, Any, Optional
from models.config import (
    ConnectorConfig,
    AuthType,
    ApiKeyLocation,
    OAuth2GrantType,
    HTTPMethod,
    NoAuthConfig,
    ApiKeyAuthConfig,
    BearerTokenAuthConfig,
    JWTAuthConfig,
    OAuth2AuthConfig,
    MTLSAuthConfig,
    CustomHeaderAuthConfig,
    RetryConfig
)
from connectors import ConnectorFactory
from storage import FilesystemStorage
from auth import OAuth2Auth

class ConnectorCLI:
    """
    Interactive CLI for connector management.
    
    Guides users through creating, testing, and managing connectors
    via terminal prompts.
    """
    
    def __init__(self) -> None:
        """Initialize CLI with storage."""
        self.storage = FilesystemStorage(storage_dir="connectors_data")
    
    def print_header(self, text: str) -> None:
        """Print a formatted header."""
        print("\n" + "=" * 70)
        print(text.center(70))
        print("=" * 70 + "\n")
    
    def print_section(self, text: str) -> None:
        """Print a formatted section header."""
        print("\n" + "-" * 70)
        print(text)
        print("-" * 70)
    
    def get_input(self, prompt: str, default: Optional[str] = None, required: bool = True) -> str:
        """
        Get user input with optional default.
        
        Args:
            prompt: Prompt text
            default: Default value
            required: Whether input is required
            
        Returns:
            User input
        """
        if default:
            prompt = f"{prompt} [{default}]: "
        else:
            prompt = f"{prompt}: "
        
        while True:
            value = input(prompt).strip()
            
            if not value and default:
                return default
            
            if not value and not required:
                return ""
            
            if value:
                return value
            
            if required:
                print("This field is required. Please enter a value.")
    
    def get_choice(self, prompt: str, choices: Dict[str, str]) -> str:
        """
        Get user choice from a menu.
        
        Args:
            prompt: Prompt text
            choices: Dictionary of choice key to description
            
        Returns:
            Selected choice key
        """
        print(f"\n{prompt}")
        for key, desc in choices.items():
            print(f"  {key}. {desc}")
        
        while True:
            choice = input("\nEnter choice: ").strip()
            if choice in choices:
                return choice
            print(f"Invalid choice. Please enter one of: {', '.join(choices.keys())}")
    
    def configure_basic_info(self) -> Dict[str, Any]:
        """
        Configure basic connector information.
        
        Returns:
            Dictionary with basic config
        """
        self.print_section("Basic Information")
        
        return {
            "id": self.get_input("Connector ID (unique identifier)"),
            "name": self.get_input("Connector Name"),
            "description": self.get_input("Description", required=False),
            "base_url": self.get_input("Base URL (e.g., https://api.example.com)"),
        }
    
    def configure_no_auth(self) -> NoAuthConfig:
        """Configure no authentication."""
        print("\nNo authentication selected.")
        return NoAuthConfig()
    
    def configure_api_key_auth(self) -> ApiKeyAuthConfig:
        """Configure API key authentication."""
        self.print_section("API Key Authentication")
        
        key = self.get_input("API Key")
        
        location_choice = self.get_choice(
            "Where should the API key be placed?",
            {
                "1": "Header",
                "2": "Query Parameter",
                "3": "Cookie"
            }
        )
        
        location_map = {
            "1": ApiKeyLocation.HEADER,
            "2": ApiKeyLocation.QUERY,
            "3": ApiKeyLocation.COOKIE
        }
        location = location_map[location_choice]
        
        param_name = self.get_input(
            f"{'Header' if location == ApiKeyLocation.HEADER else 'Parameter'} name",
            default="X-API-Key" if location == ApiKeyLocation.HEADER else "api_key"
        )
        
        return ApiKeyAuthConfig(
            key=key,
            location=location,
            param_name=param_name
        )
    
    def configure_bearer_auth(self) -> BearerTokenAuthConfig:
        """Configure Bearer token authentication."""
        self.print_section("Bearer Token Authentication")
        
        token = self.get_input("Bearer Token")
        
        return BearerTokenAuthConfig(token=token)
    
    def configure_jwt_auth(self) -> JWTAuthConfig:
        """Configure JWT authentication."""
        self.print_section("JWT Authentication")
        
        use_existing = self.get_choice(
            "Do you have an existing JWT token?",
            {"1": "Yes, I have a token", "2": "No, generate from secret + payload"}
        )
        
        if use_existing == "1":
            token = self.get_input("JWT Token")
            return JWTAuthConfig(token=token)
        else:
            secret = self.get_input("JWT Secret")
            print("\nEnter JWT payload as JSON (e.g., {\"sub\": \"user123\"}):")
            payload_str = input().strip()
            
            try:
                payload = json.loads(payload_str)
            except json.JSONDecodeError:
                print("Invalid JSON. Using empty payload.")
                payload = {}
            
            algorithm = self.get_input("Algorithm", default="HS256")
            
            return JWTAuthConfig(
                secret=secret,
                payload=payload,
                algorithm=algorithm
            )
    
    def configure_oauth2_auth(self) -> OAuth2AuthConfig:
        """Configure OAuth2 authentication."""
        self.print_section("OAuth2 Authentication")
        
        grant_choice = self.get_choice(
            "Select OAuth2 grant type:",
            {
                "1": "Client Credentials",
                "2": "Authorization Code",
                "3": "Refresh Token",
                "4": "Device Code"
            }
        )
        
        grant_map = {
            "1": OAuth2GrantType.CLIENT_CREDENTIALS,
            "2": OAuth2GrantType.AUTHORIZATION_CODE,
            "3": OAuth2GrantType.REFRESH_TOKEN,
            "4": OAuth2GrantType.DEVICE_CODE
        }
        grant_type = grant_map[grant_choice]
        
        # Common fields
        token_url = self.get_input("Token URL")
        
        # GitHub guardrails
        if "github.com" in token_url.lower():
            if grant_type in [OAuth2GrantType.CLIENT_CREDENTIALS, OAuth2GrantType.REFRESH_TOKEN]:
                print(f"\n✗ GitHub does not support {grant_type.value} grant type.")
                print("Supported grants: authorization_code, device_code")
                return self.configure_oauth2_auth()
        
        client_id = self.get_input("Client ID")
        
        # Device Code does NOT use client_secret
        if grant_type == OAuth2GrantType.DEVICE_CODE:
            client_secret = ""
        else:
            client_secret = self.get_input("Client Secret")
        
        scope = self.get_input("Scope (optional)", required=False)
        
        config_dict: Dict[str, Any] = {
            "grant_type": grant_type,
            "token_url": token_url,
            "client_id": client_id,
            "client_secret": client_secret,
            "scope": scope if scope else None
        }
        
        if grant_type == OAuth2GrantType.AUTHORIZATION_CODE:
            config_dict["authorization_url"] = self.get_input("Authorization URL")
            config_dict["redirect_uri"] = self.get_input("Redirect URI")
            config_dict["code"] = self.get_input("Authorization Code")
        
        elif grant_type == OAuth2GrantType.REFRESH_TOKEN:
            config_dict["refresh_token"] = self.get_input("Refresh Token")
        
        elif grant_type == OAuth2GrantType.DEVICE_CODE:
            config_dict["device_code_url"] = self.get_input("Device Code URL")
            
            # Initiate device code flow
            print("\nInitiating device code flow...")
            try:
                temp_auth = OAuth2Auth(
                    grant_type=grant_type,
                    token_url=token_url,
                    client_id=client_id,
                    client_secret="",  # Not used for device code
                    scope=scope,
                    device_code_url=config_dict["device_code_url"]
                )
                
                device_response = temp_auth._request_device_code()
                
                print(f"\nVisit:\n{device_response.get('verification_uri')}")
                print(f"\nEnter code:\n{device_response.get('user_code')}")
                print(f"\nWaiting for authorization...")
                
                interval = int(device_response.get("interval", 5))
                device_code = device_response.get("device_code")
                
                token_response = temp_auth._poll_device_token(device_code, interval)
                temp_auth._update_token(token_response)
                
                print(f"\n✓ Authorization successful!")
                
                config_dict["device_code"] = device_code
                config_dict["access_token"] = temp_auth.access_token
                config_dict["token_expiry"] = temp_auth.token_expiry
                
            except Exception as e:
                print(f"\n✗ Device code flow failed: {str(e)}")
                retry = self.get_choice("Retry?", {"1": "Yes", "2": "No"})
                if retry == "1":
                    return self.configure_oauth2_auth()
                raise
        
        return OAuth2AuthConfig(**config_dict)
    
    def configure_mtls_auth(self) -> MTLSAuthConfig:
        """Configure mTLS authentication."""
        self.print_section("Mutual TLS Authentication")
        
        cert_path = self.get_input("Client Certificate Path")
        key_path = self.get_input("Client Key Path")
        ca_bundle = self.get_input("CA Bundle Path (optional)", required=False)
        
        verify_choice = self.get_choice(
            "Verify SSL certificates?",
            {"1": "Yes", "2": "No"}
        )
        verify_ssl = verify_choice == "1"
        
        return MTLSAuthConfig(
            cert_path=cert_path,
            key_path=key_path,
            ca_bundle_path=ca_bundle if ca_bundle else None,
            verify_ssl=verify_ssl
        )
    
    def configure_custom_header_auth(self) -> CustomHeaderAuthConfig:
        """Configure custom header authentication."""
        self.print_section("Custom Header Authentication")
        
        headers = {}
        print("\nEnter custom headers (empty header name to finish):")
        
        while True:
            name = self.get_input("Header Name", required=False)
            if not name:
                break
            
            value = self.get_input(f"Value for '{name}'")
            headers[name] = value
        
        if not headers:
            print("No headers provided. Adding default header.")
            headers["X-Custom-Auth"] = self.get_input("Header Value")
        
        return CustomHeaderAuthConfig(headers=headers)
    
    def configure_auth(self) -> Any:
        """
        Configure authentication.
        
        Returns:
            Auth configuration object
        """
        self.print_section("Authentication Configuration")
        
        auth_choice = self.get_choice(
            "Select authentication type:",
            {
                "1": "No Authentication",
                "2": "API Key",
                "3": "Bearer Token",
                "4": "JWT",
                "5": "OAuth2",
                "6": "Mutual TLS (mTLS)",
                "7": "Custom Headers"
            }
        )
        
        auth_configs = {
            "1": self.configure_no_auth,
            "2": self.configure_api_key_auth,
            "3": self.configure_bearer_auth,
            "4": self.configure_jwt_auth,
            "5": self.configure_oauth2_auth,
            "6": self.configure_mtls_auth,
            "7": self.configure_custom_header_auth
        }
        
        return auth_configs[auth_choice]()
    
    def configure_advanced_options(self) -> Dict[str, Any]:
        """
        Configure advanced connector options.
        
        Returns:
            Dictionary with advanced options
        """
        self.print_section("Advanced Options (Optional)")
        
        configure_advanced = self.get_choice(
            "Configure advanced options?",
            {"1": "Yes", "2": "No, use defaults"}
        )
        
        if configure_advanced == "2":
            return {}
        
        options = {}
        
        timeout = self.get_input("Timeout (seconds)", default="30.0")
        options["timeout"] = float(timeout)
        
        max_retries = self.get_input("Max Retries", default="3")
        options["retry_config"] = RetryConfig(max_retries=int(max_retries))
        
        verify_ssl = self.get_choice(
            "Verify SSL certificates?",
            {"1": "Yes", "2": "No"}
        )
        options["verify_ssl"] = verify_ssl == "1"
        
        return options
    
    def create_connector(self) -> None:
        """Create a new connector interactively."""
        self.print_header("Create New Connector")
        
        try:
            # Gather configuration
            basic_info = self.configure_basic_info()
            auth_config = self.configure_auth()
            advanced_options = self.configure_advanced_options()
            
            # Build config
            config_dict = {
                **basic_info,
                "auth": auth_config,
                **advanced_options
            }
            
            # Validate and create
            config = ConnectorConfig(**config_dict)
            connector = ConnectorFactory.create(config)
            
            # Save
            self.storage.save(config.id, connector.to_dict())
            
            print(f"\n✓ Connector '{config.name}' created successfully!")
            print(f"  ID: {config.id}")
            print(f"  Base URL: {config.base_url}")
            print(f"  Auth Type: {config.auth.type.value}")
            
            # Offer to test
            test_now = self.get_choice(
                "\nWould you like to test the connector now?",
                {"1": "Yes", "2": "No"}
            )
            
            if test_now == "1":
                self.test_connector_by_id(config.id)
        
        except KeyboardInterrupt:
            print("\n\nCancelled.")
            return
        except Exception as e:
            print(f"\n✗ Error creating connector: {str(e)}")
    
    def configure_connector(self) -> None:
        """Configure an existing connector."""
        self.print_header("Configure Existing Connector")
        
        # List connectors
        connector_ids = self.storage.list_all()
        
        if not connector_ids:
            print("No connectors found. Create one first.")
            return
        
        print("Available connectors:\n")
        for i, connector_id in enumerate(connector_ids, 1):
            try:
                data = self.storage.load(connector_id)
                config_data = data.get("config", {})
                print(f"{i}. {config_data.get('name', 'Unknown')} ({connector_id})")
                print(f"   URL: {config_data.get('base_url', 'N/A')}")
                print(f"   Auth: {config_data.get('auth', {}).get('type', 'N/A')}\n")
            except Exception:
                print(f"{i}. {connector_id} (Error loading)\n")
        
        # Select connector
        connector_id = self.get_input("\nEnter Connector ID to configure")
        
        try:
            # Load existing connector
            data = self.storage.load(connector_id)
            config_data = data.get("config", {})
            
            print(f"\nConfiguring: {config_data.get('name', connector_id)}")
            
            # Basic info
            self.print_section("Basic Information")
            
            current_name = config_data.get("name", "")
            print(f"Current name: {current_name}")
            new_name = self.get_input("New name (press Enter to keep)", default=current_name, required=False)
            name = new_name if new_name else current_name
            
            current_base_url = config_data.get("base_url", "")
            print(f"\nCurrent base URL: {current_base_url}")
            new_base_url = self.get_input("New base URL (press Enter to keep)", default=current_base_url, required=False)
            base_url = new_base_url if new_base_url else current_base_url
            
            current_description = config_data.get("description", "")
            if current_description:
                print(f"\nCurrent description: {current_description}")
            new_description = self.get_input("New description (press Enter to keep)", default=current_description, required=False)
            description = new_description if new_description else current_description
            
            # Authentication
            self.print_section("Authentication")
            
            current_auth_type = config_data.get("auth", {}).get("type", "none")
            print(f"Current auth type: {current_auth_type}")
            
            change_auth = self.get_choice(
                "Authentication:",
                {
                    "1": f"Keep existing ({current_auth_type})",
                    "2": "Reconfigure authentication"
                }
            )
            
            if change_auth == "1":
                auth_config = config_data.get("auth")
            else:
                auth_config = self.configure_auth()
            
            # Advanced options
            self.print_section("Advanced Options")
            
            configure_advanced = self.get_choice(
                "Configure advanced options?",
                {"1": "Yes", "2": "No, keep existing"}
            )
            
            if configure_advanced == "1":
                current_timeout = config_data.get("timeout", 30.0)
                print(f"\nCurrent timeout: {current_timeout}s")
                new_timeout = self.get_input("New timeout (seconds, press Enter to keep)", default=str(current_timeout), required=False)
                timeout = float(new_timeout) if new_timeout else current_timeout
                
                current_retries = config_data.get("retry_config", {}).get("max_retries", 3)
                print(f"\nCurrent max retries: {current_retries}")
                new_retries = self.get_input("New max retries (press Enter to keep)", default=str(current_retries), required=False)
                max_retries = int(new_retries) if new_retries else current_retries
                
                current_verify_ssl = config_data.get("verify_ssl", True)
                print(f"\nCurrent SSL verification: {current_verify_ssl}")
                verify_ssl_choice = self.get_choice(
                    "Verify SSL certificates?",
                    {"1": "Yes", "2": "No"}
                )
                verify_ssl = verify_ssl_choice == "1"
                
                retry_config = RetryConfig(max_retries=max_retries)
            else:
                timeout = config_data.get("timeout", 30.0)
                retry_config = config_data.get("retry_config", RetryConfig())
                verify_ssl = config_data.get("verify_ssl", True)
            
            # Build updated config
            updated_config_dict = {
                "id": connector_id,  # Keep same ID
                "name": name,
                "description": description if description else None,
                "base_url": base_url,
                "auth": auth_config,
                "timeout": timeout,
                "retry_config": retry_config,
                "verify_ssl": verify_ssl
            }
            
            # Validate and update
            config = ConnectorConfig(**updated_config_dict)
            connector = ConnectorFactory.create(config)
            
            # Update in storage
            self.storage.update(config.id, connector.to_dict())
            
            print(f"\n✓ Connector '{config.name}' updated successfully!")
            print(f"  ID: {config.id}")
            print(f"  Base URL: {config.base_url}")
            print(f"  Auth Type: {config.auth.type.value}")
            
            # Offer to test
            test_now = self.get_choice(
                "\nWould you like to test the updated connector?",
                {"1": "Yes", "2": "No"}
            )
            
            if test_now == "1":
                self.test_connector_by_id(config.id)
        
        except KeyError:
            print(f"\n✗ Connector '{connector_id}' not found.")
        except KeyboardInterrupt:
            print("\n\nCancelled.")
        except Exception as e:
            print(f"\n✗ Error configuring connector: {str(e)}")
        
    def list_connectors(self) -> None:
        """List all connectors."""
        self.print_header("Saved Connectors")
            
        connector_ids = self.storage.list_all()
            
        if not connector_ids:
            print("No connectors found.")
            return
            
        print(f"Found {len(connector_ids)} connector(s):\n")
            
        for i, connector_id in enumerate(connector_ids, 1):
            try:
                data = self.storage.load(connector_id)
                config_data = data.get("config", {})
                    
                print(f"{i}. {config_data.get('name', 'Unknown')} ({connector_id})")
                print(f"   URL: {config_data.get('base_url', 'N/A')}")
                print(f"   Auth: {config_data.get('auth', {}).get('type', 'N/A')}")
                print(f"   Updated: {data.get('updated_at', 'N/A')}\n")
            except Exception as e:
                print(f"{i}. {connector_id} (Error loading: {str(e)})\n")
    
    def test_connector_by_id(self, connector_id: Optional[str] = None) -> None:
        """
        Test a connector.
        
        Args:
            connector_id: Connector ID (optional, will prompt if not provided)
        """
        if not connector_id:
            self.print_header("Test Connector")
            connector_id = self.get_input("Connector ID")
        
        try:
            # Load connector
            data = self.storage.load(connector_id)
            connector = ConnectorFactory.from_dict(data)
            
            print(f"\nTesting connector '{connector_id}'...")
            
            # Ask if user wants to provide custom payload
            use_custom = self.get_choice(
                "Test mode:",
                {
                    "1": "Simple connection test (GET /)",
                    "2": "Custom request with payload"
                }
            )
            
            if use_custom == "1":
                # Simple test
                result = connector.test_connection()
            else:
                # Custom payload test
                print("\nEnter request payload as JSON (multi-line supported).")
                print("Press Enter on empty line when done.")
                print("Example:")
                print('{\n  "method": "GET",\n  "endpoint": "/weather",\n  "query_params": {"q": "Bangalore"}\n}\n')
                
                lines = []
                print("Payload:")
                while True:
                    line = input()
                    if not line and lines:
                        break
                    lines.append(line)
                
                payload_str = '\n'.join(lines).strip()
                
                if not payload_str:
                    print("No payload provided. Using simple test.")
                    result = connector.test_connection()
                else:
                    try:
                        payload = json.loads(payload_str)
                    except json.JSONDecodeError as e:
                        print(f"\n✗ Invalid JSON: {str(e)}")
                        return
                    
                    # Extract parameters
                    method = payload.get("method", "GET")
                    endpoint = payload.get("endpoint", "/")
                    query_params = payload.get("query_params")
                    headers = payload.get("headers")
                    body = payload.get("body")
                    
                    print(f"\nExecuting {method} {endpoint}...")
                    result = connector.execute(
                        method=method,
                        endpoint=endpoint,
                        query_params=query_params,
                        headers=headers,
                        body=body
                    )
            
            print("\nTest Results:")
            print(f"  Success: {result.success}")
            print(f"  Status Code: {result.status_code}")
            print(f"  Elapsed: {result.elapsed_seconds:.3f}s")
            
            if result.error:
                print(f"  Error: {result.error}")
            
            if result.success:
                print("\n✓ Connector is working!")
                
                # Show response body on success
                if result.body:
                    print("\nResponse:")
                    if isinstance(result.body, dict):
                        print(json.dumps(result.body, indent=2))
                    else:
                        print(result.body)
            else:
                print("\n✗ Connector test failed.")
        
        except KeyError:
            print(f"\n✗ Connector '{connector_id}' not found.")
        except Exception as e:
            print(f"\n✗ Test failed: {str(e)}")
    
    def execute_connector(self) -> None:
        """Execute a connector with custom request."""
        self.print_header("Execute Connector")
        
        try:
            connector_id = self.get_input("Connector ID")
            
            # Load connector
            data = self.storage.load(connector_id)
            connector = ConnectorFactory.from_dict(data)
            
            # Get request details
            method = self.get_choice(
                "Select HTTP method:",
                {
                    "1": "GET",
                    "2": "POST",
                    "3": "PUT",
                    "4": "PATCH",
                    "5": "DELETE"
                }
            )
            method_map = {"1": "GET", "2": "POST", "3": "PUT", "4": "PATCH", "5": "DELETE"}
            http_method = method_map[method]
            
            endpoint = self.get_input("Endpoint path (e.g., /users)")
            
            # Optional body for POST/PUT/PATCH
            body = None
            if http_method in ["POST", "PUT", "PATCH"]:
                has_body = self.get_choice(
                    "Include request body?",
                    {"1": "Yes", "2": "No"}
                )
                
                if has_body == "1":
                    print("\nEnter request body as JSON:")
                    body_str = input().strip()
                    try:
                        body = json.loads(body_str)
                    except json.JSONDecodeError:
                        print("Invalid JSON. Skipping body.")
            
            # Execute
            print(f"\nExecuting {http_method} {endpoint}...")
            result = connector.execute(
                method=http_method,
                endpoint=endpoint,
                body=body
            )
            
            # Display results
            print("\nResponse:")
            print(f"  Success: {result.success}")
            print(f"  Status Code: {result.status_code}")
            print(f"  Elapsed: {result.elapsed_seconds:.3f}s")
            
            if result.body:
                print("\n  Body:")
                if isinstance(result.body, dict):
                    print(json.dumps(result.body, indent=2))
                else:
                    print(f"  {result.body}")
            
            if result.error:
                print(f"\n  Error: {result.error}")
        
        except KeyError:
            print(f"\n✗ Connector not found.")
        except KeyboardInterrupt:
            print("\n\nCancelled.")
        except Exception as e:
            print(f"\n✗ Execution failed: {str(e)}")
    
    def delete_connector(self) -> None:
        """Delete a connector."""
        self.print_header("Delete Connector")
        
        try:
            connector_id = self.get_input("Connector ID to delete")
            
            # Confirm
            confirm = self.get_choice(
                f"Are you sure you want to delete '{connector_id}'?",
                {"1": "Yes", "2": "No"}
            )
            
            if confirm == "2":
                print("Cancelled.")
                return
            
            self.storage.delete(connector_id)
            print(f"\n✓ Connector '{connector_id}' deleted successfully.")
        
        except KeyError:
            print(f"\n✗ Connector not found.")
        except Exception as e:
            print(f"\n✗ Delete failed: {str(e)}")
    
    def run(self) -> None:
        """Run the interactive CLI."""
        self.print_header("API Connector Framework - CLI")
        
        while True:
            try:
                choice = self.get_choice(
                    "\nMain Menu",
                    {
                        "1": "Create New Connector",
                        "2": "List Connectors",
                        "3": "Configure Existing Connector",
                        "4": "Test Connector",
                        "5": "Execute Connector",
                        "6": "Delete Connector",
                        "7": "Exit"
                    }
                )

                if choice == "1":
                    self.create_connector()
                elif choice == "2":
                    self.list_connectors()
                elif choice == "3":
                    self.configure_connector()
                elif choice == "4":
                    self.test_connector_by_id()
                elif choice == "5":
                    self.execute_connector()
                elif choice == "6":
                    self.delete_connector()
                elif choice == "7":
                    print("\nGoodbye!")
                    sys.exit(0)
            
            except KeyboardInterrupt:
                print("\n\nUse option 6 to exit.")
            except Exception as e:
                print(f"\n✗ Unexpected error: {str(e)}")


def main() -> None:
    """Entry point for CLI."""
    cli = ConnectorCLI()
    cli.run()


if __name__ == "__main__":
    main()
