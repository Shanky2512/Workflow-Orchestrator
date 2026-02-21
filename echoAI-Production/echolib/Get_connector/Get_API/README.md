# API Connector Framework

A production-grade Python framework for generating AI-agent-compatible API connectors with comprehensive authentication support and dual interface (CLI + FastAPI).

## 🎯 Features

### Core Capabilities
- **Universal API Support**: Works with any RESTful API
- **AI-Agent Compatible**: Stateless, serializable connectors for LLM orchestration
- **Dual Interface**: FastAPI REST API + Interactive CLI
- **Production-Ready**: Comprehensive error handling, retry logic, validation

### Authentication Support
- ✅ No Authentication
- ✅ API Key (Header, Query, Cookie)
- ✅ Bearer Token
- ✅ JWT (with generation support)
- ✅ OAuth2 (Client Credentials, Authorization Code, Refresh Token)
- ✅ Mutual TLS (mTLS)
- ✅ Custom Headers

### Advanced Features
- Automatic retry with exponential backoff
- Configurable timeouts
- SSL/TLS verification control
- Response streaming support
- Persistent storage (filesystem-based)
- Full request/response logging
- Comprehensive validation

## 📋 Requirements

- Python 3.10+
- Dependencies listed in `requirements.txt`

## 🚀 Installation

```bash
# Clone or download the framework
cd api_connector_framework

# Install dependencies
pip install -r requirements.txt
```

## 💻 Usage

### Option 1: Interactive CLI

```bash
python main.py
```

The CLI provides an interactive menu for:
- Creating connectors (step-by-step)
- Testing connectivity
- Executing requests
- Managing saved connectors

### Option 2: FastAPI Server

```bash
# Start server on default port 8000
python main.py --api

# Custom host and port
python main.py --api --host 0.0.0.0 --port 8080
```

Access the API documentation at: `http://localhost:8000/docs`

## 📚 API Endpoints

### Create Connector
```http
POST /connectors/create
Content-Type: application/json

{
  "id": "github-api",
  "name": "GitHub API",
  "base_url": "https://api.github.com",
  "auth": {
    "type": "bearer",
    "token": "ghp_xxxxxxxxxxxx"
  }
}
```

### Execute Request
```http
POST /connectors/github-api/execute
Content-Type: application/json

{
  "method": "GET",
  "endpoint": "/user",
  "headers": {
    "Accept": "application/vnd.github.v3+json"
  }
}
```

### List All Connectors
```http
GET /connectors
```

### Get Connector Details
```http
GET /connectors/{connector_id}
```

### Update Connector
```http
PUT /connectors/{connector_id}
Content-Type: application/json

{
  "id": "github-api",
  "name": "GitHub API Updated",
  ...
}
```

### Delete Connector
```http
DELETE /connectors/{connector_id}
```

### Test Connector
```http
POST /connectors/{connector_id}/test
```

## 🔐 Authentication Examples

### API Key (Header)
```python
{
  "type": "api_key",
  "key": "your-api-key",
  "location": "header",
  "param_name": "X-API-Key"
}
```

### Bearer Token
```python
{
  "type": "bearer",
  "token": "your-bearer-token"
}
```

### JWT (Pre-generated)
```python
{
  "type": "jwt",
  "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
}
```

### JWT (Generated)
```python
{
  "type": "jwt",
  "secret": "your-secret-key",
  "algorithm": "HS256",
  "payload": {
    "sub": "user123",
    "exp": 1234567890
  }
}
```

### OAuth2 (Client Credentials)
```python
{
  "type": "oauth2",
  "grant_type": "client_credentials",
  "token_url": "https://auth.example.com/oauth/token",
  "client_id": "your-client-id",
  "client_secret": "your-client-secret",
  "scope": "read write"
}
```

### OAuth2 (Authorization Code)
```python
{
  "type": "oauth2",
  "grant_type": "authorization_code",
  "token_url": "https://auth.example.com/oauth/token",
  "authorization_url": "https://auth.example.com/oauth/authorize",
  "client_id": "your-client-id",
  "client_secret": "your-client-secret",
  "redirect_uri": "https://yourapp.com/callback",
  "code": "authorization-code-from-callback"
}
```

### Mutual TLS
```python
{
  "type": "mtls",
  "cert_path": "/path/to/client.crt",
  "key_path": "/path/to/client.key",
  "ca_bundle_path": "/path/to/ca-bundle.crt",
  "verify_ssl": true
}
```

### Custom Headers
```python
{
  "type": "custom_header",
  "headers": {
    "X-Custom-Auth": "secret-value",
    "X-API-Version": "v2"
  }
}
```

## 🤖 AI Agent Integration

Connectors are designed for seamless AI agent integration:

```python
from connectors import ConnectorFactory
from storage import FilesystemStorage

# Load connector
storage = FilesystemStorage()
data = storage.load("my-connector")
connector = ConnectorFactory.from_dict(data)

# Stateless execution - perfect for AI agents
result = connector.execute(
    method="POST",
    endpoint="/api/resource",
    body={"key": "value"},
    headers={"X-Custom": "header"}
)

# Parse result
if result.success:
    print(f"Status: {result.status_code}")
    print(f"Response: {result.body}")
else:
    print(f"Error: {result.error}")
```

## 🏗️ Architecture

```
api_connector_framework/
├── auth/                   # Authentication strategies
│   ├── base.py            # Abstract auth base class
│   ├── api_key.py         # API key authentication
│   ├── bearer.py          # Bearer token authentication
│   ├── jwt_auth.py        # JWT authentication
│   ├── oauth2.py          # OAuth2 authentication
│   ├── mtls.py            # Mutual TLS authentication
│   ├── custom.py          # Custom header authentication
│   └── no_auth.py         # No authentication
├── connectors/             # Connector implementations
│   ├── base.py            # Abstract connector base
│   ├── http.py            # HTTP connector
│   └── factory.py         # Connector factory
├── models/                 # Pydantic models
│   ├── config.py          # Configuration models
│   └── responses.py       # Response models
├── storage/                # Persistence layer
│   ├── base.py            # Abstract storage base
│   └── filesystem.py      # Filesystem storage
├── api/                    # FastAPI interface
│   ├── routes.py          # API endpoints
│   └── dependencies.py    # FastAPI dependencies
├── cli/                    # CLI interface
│   └── runner.py          # Interactive CLI
├── main.py                 # Application entry point
└── requirements.txt        # Dependencies
```

## 🔧 Advanced Configuration

### Retry Configuration
```python
{
  "retry_config": {
    "max_retries": 3,
    "backoff_factor": 1.0,
    "retry_status_codes": [429, 500, 502, 503, 504]
  }
}
```

### Timeout Configuration
```python
{
  "timeout": 30.0  # seconds
}
```

### SSL Verification
```python
{
  "verify_ssl": true,
  "allow_redirects": true,
  "max_redirects": 10
}
```

### Default Headers
```python
{
  "default_headers": {
    "User-Agent": "MyApp/1.0",
    "Accept": "application/json"
  }
}
```

## 📝 Complete Example

```python
from models.config import ConnectorConfig, BearerTokenAuthConfig
from connectors import ConnectorFactory
from storage import FilesystemStorage

# Create configuration
config = ConnectorConfig(
    id="my-api",
    name="My API",
    description="Production API connector",
    base_url="https://api.example.com",
    auth=BearerTokenAuthConfig(token="my-token"),
    timeout=30.0,
    verify_ssl=True
)

# Create connector
connector = ConnectorFactory.create(config)

# Save for later use
storage = FilesystemStorage()
storage.save(config.id, connector.to_dict())

# Execute request
result = connector.execute(
    method="GET",
    endpoint="/users",
    query_params={"page": 1}
)

print(f"Success: {result.success}")
print(f"Status: {result.status_code}")
print(f"Body: {result.body}")
```

## 🧪 Testing

```bash
# Test with a public API
python main.py

# In CLI:
# 1. Create New Connector
#    ID: httpbin
#    Name: HTTPBin Test
#    Base URL: https://httpbin.org
#    Auth: No Authentication
#
# 2. Test Connector
#    Connector ID: httpbin
#
# 3. Execute Connector
#    Connector ID: httpbin
#    Method: GET
#    Endpoint: /get
```

## 🛡️ Security Considerations

1. **Credentials Storage**: Currently stored in JSON files. For production:
   - Use encrypted storage
   - Integrate with secrets management (AWS Secrets Manager, HashiCorp Vault)
   - Implement access controls

2. **SSL/TLS**: Always verify SSL certificates in production unless specifically required

3. **API Keys**: Never commit API keys to version control

4. **Rate Limiting**: Implement rate limiting for API endpoints if exposed publicly

## 🚧 Extending the Framework

### Adding New Authentication Types

1. Create new auth class in `auth/`:
```python
from auth.base import AuthBase

class MyCustomAuth(AuthBase):
    def apply(self, headers, params, client):
        # Implementation
        pass
    
    # Implement other required methods
```

2. Register in `connectors/http.py`:
```python
elif auth_type == AuthType.MY_CUSTOM:
    return MyCustomAuth(...)
```

### Adding New Connector Types

1. Create new connector in `connectors/`:
```python
from connectors.base import BaseConnector

class WebSocketConnector(BaseConnector):
    # Implementation
```

2. Update factory in `connectors/factory.py`

## 📄 License

This is a production-ready framework provided as-is for educational and commercial use.

## 🤝 Contributing

This framework is designed to be extensible. Key extension points:
- New authentication strategies (`auth/`)
- New connector types (`connectors/`)
- New storage backends (`storage/`)
- Additional API endpoints (`api/routes.py`)

## ⚡ Performance

- Async-ready (FastAPI)
- Reusable HTTP connections
- Efficient JSON serialization
- Minimal memory footprint
- Stateless design for horizontal scaling

## 📞 Support

For issues or questions:
1. Check the documentation
2. Review example configurations
3. Test with public APIs (httpbin.org)

---

Built with ❤️ for production use
