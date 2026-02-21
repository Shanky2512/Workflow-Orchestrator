# Quick Start Guide

Get up and running with the API Connector Framework in 5 minutes.

## Installation

```bash
# Navigate to the framework directory
cd api_connector_framework

# Install dependencies
pip install -r requirements.txt
```

## Test the Framework

Run the test suite to verify everything works:

```bash
python test_framework.py
```

This will:
- Create test connectors
- Make real API calls to httpbin.org
- Demonstrate all core features
- Verify storage and serialization

## Option 1: Interactive CLI

Perfect for exploring and testing:

```bash
python main.py
```

### Quick Example - Create a Test Connector

1. Select `1. Create New Connector`
2. Enter these values:
   - ID: `test-api`
   - Name: `Test API`
   - Base URL: `https://httpbin.org`
   - Select `1. No Authentication`
   - Select `2. No, use defaults` for advanced options

3. Test it immediately or select `2. List Connectors` to see your connector

4. Execute a request with `4. Execute Connector`:
   - Connector ID: `test-api`
   - Method: `1. GET`
   - Endpoint: `/get`

## Option 2: FastAPI Server

Perfect for integration with other services:

```bash
python main.py --api
```

Then visit `http://localhost:8000/docs` for interactive API documentation.

### Quick API Example

```bash
# Create a connector
curl -X POST http://localhost:8000/connectors/create \
  -H "Content-Type: application/json" \
  -d '{
    "id": "httpbin-test",
    "name": "HTTPBin",
    "base_url": "https://httpbin.org",
    "auth": {"type": "none"}
  }'

# Test it
curl -X POST http://localhost:8000/connectors/httpbin-test/test

# Execute a request
curl -X POST http://localhost:8000/connectors/httpbin-test/execute \
  -H "Content-Type: application/json" \
  -d '{
    "method": "GET",
    "endpoint": "/json"
  }'
```

## Common Use Cases

### Use Case 1: API Key Authentication

```python
# Via Python
from models.config import ConnectorConfig, ApiKeyAuthConfig, ApiKeyLocation
from connectors import ConnectorFactory

config = ConnectorConfig(
    id="my-api",
    name="My API",
    base_url="https://api.example.com",
    auth=ApiKeyAuthConfig(
        key="your-api-key-here",
        location=ApiKeyLocation.HEADER,
        param_name="X-API-Key"
    )
)

connector = ConnectorFactory.create(config)
result = connector.execute(method="GET", endpoint="/users")
```

### Use Case 2: Bearer Token (GitHub, OpenAI, etc.)

```python
from models.config import ConnectorConfig, BearerTokenAuthConfig

config = ConnectorConfig(
    id="github",
    name="GitHub API",
    base_url="https://api.github.com",
    auth=BearerTokenAuthConfig(token="ghp_your_token")
)

connector = ConnectorFactory.create(config)
result = connector.execute(method="GET", endpoint="/user")
```

### Use Case 3: OAuth2 Client Credentials

```python
from models.config import ConnectorConfig, OAuth2AuthConfig, OAuth2GrantType

config = ConnectorConfig(
    id="oauth-api",
    name="OAuth API",
    base_url="https://api.example.com",
    auth=OAuth2AuthConfig(
        grant_type=OAuth2GrantType.CLIENT_CREDENTIALS,
        token_url="https://auth.example.com/token",
        client_id="your-client-id",
        client_secret="your-client-secret"
    )
)

connector = ConnectorFactory.create(config)
result = connector.execute(method="GET", endpoint="/protected-resource")
```

## For AI Agents

The framework is optimized for AI agent use:

```python
# Agent workflow
from connectors import ConnectorFactory
from storage import FilesystemStorage

# 1. Load connector (agents can store connector IDs)
storage = FilesystemStorage()
connector_data = storage.load("my-connector-id")
connector = ConnectorFactory.from_dict(connector_data)

# 2. Execute stateless call with dynamic parameters
result = connector.execute(
    method="POST",
    endpoint="/api/action",
    body={"dynamic": "data", "from": "agent"}
)

# 3. Parse and use results
if result.success:
    agent_processes(result.body)
else:
    agent_handles_error(result.error)
```

## Directory Structure

After running, you'll see:

```
api_connector_framework/
├── connectors_data/        # Saved connectors (gitignored)
│   ├── test-api.json
│   └── httpbin-test.json
├── main.py                 # Your entry point
└── ...
```

## Troubleshooting

### Import Errors
```bash
# Make sure you're in the right directory
cd api_connector_framework

# Reinstall dependencies
pip install -r requirements.txt
```

### Network Errors
```bash
# Test with httpbin.org first - it's always available
# If other APIs fail, check:
# 1. API credentials
# 2. Network connectivity
# 3. API endpoint URLs
```

### Storage Errors
```bash
# The framework creates connectors_data/ automatically
# If you get permission errors, check directory permissions
chmod 755 connectors_data/
```

## Next Steps

1. **Read the full README**: `README.md` has complete documentation
2. **Check examples**: `examples/README.md` has real-world configurations
3. **Explore the API**: Run `python main.py --api` and visit `/docs`
4. **Customize**: Extend with your own auth types or connectors

## Production Checklist

Before deploying:

- [ ] Store credentials securely (not in JSON files)
- [ ] Enable SSL verification (`verify_ssl: true`)
- [ ] Configure appropriate timeouts
- [ ] Set up retry logic for your use case
- [ ] Add rate limiting if exposing API publicly
- [ ] Enable CORS only for trusted origins
- [ ] Monitor connector usage and errors
- [ ] Implement proper logging

## Getting Help

- Check `README.md` for detailed documentation
- Review `examples/README.md` for configuration examples
- Run `test_framework.py` to verify your setup
- Check the API docs at `/docs` when running the server

---

**You're ready to go!** Start with the CLI (`python main.py`) to explore features interactively.
