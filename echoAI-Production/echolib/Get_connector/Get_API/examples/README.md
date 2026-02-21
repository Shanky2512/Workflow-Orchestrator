# Example Connector Configurations

This directory contains example configurations for various API connectors.

## HTTPBin (No Auth)

Simple test API with no authentication:

```json
{
  "id": "httpbin",
  "name": "HTTPBin Test API",
  "description": "Public test API for HTTP requests",
  "base_url": "https://httpbin.org",
  "auth": {
    "type": "none"
  },
  "timeout": 30.0
}
```

## GitHub API (Bearer Token)

```json
{
  "id": "github",
  "name": "GitHub API",
  "description": "GitHub REST API v3",
  "base_url": "https://api.github.com",
  "auth": {
    "type": "bearer",
    "token": "ghp_your_token_here"
  },
  "default_headers": {
    "Accept": "application/vnd.github.v3+json",
    "User-Agent": "MyApp/1.0"
  }
}
```

## OpenAI API (Bearer Token)

```json
{
  "id": "openai",
  "name": "OpenAI API",
  "description": "OpenAI GPT API",
  "base_url": "https://api.openai.com/v1",
  "auth": {
    "type": "bearer",
    "token": "sk-your-key-here"
  },
  "default_headers": {
    "Content-Type": "application/json"
  }
}
```

## Stripe API (Bearer Token)

```json
{
  "id": "stripe",
  "name": "Stripe API",
  "description": "Stripe payment processing API",
  "base_url": "https://api.stripe.com/v1",
  "auth": {
    "type": "bearer",
    "token": "sk_test_your_key_here"
  },
  "default_headers": {
    "Stripe-Version": "2023-10-16"
  }
}
```

## SendGrid API (API Key in Header)

```json
{
  "id": "sendgrid",
  "name": "SendGrid Email API",
  "description": "SendGrid email delivery API",
  "base_url": "https://api.sendgrid.com/v3",
  "auth": {
    "type": "api_key",
    "key": "SG.your_api_key_here",
    "location": "header",
    "param_name": "Authorization"
  }
}
```

## Weather API (API Key in Query)

```json
{
  "id": "weather",
  "name": "Weather API",
  "description": "Weather data API",
  "base_url": "https://api.openweathermap.org/data/2.5",
  "auth": {
    "type": "api_key",
    "key": "your_api_key_here",
    "location": "query",
    "param_name": "appid"
  }
}
```

## OAuth2 Example (Client Credentials)

```json
{
  "id": "oauth2-service",
  "name": "OAuth2 Protected API",
  "description": "API using OAuth2 client credentials",
  "base_url": "https://api.example.com/v1",
  "auth": {
    "type": "oauth2",
    "grant_type": "client_credentials",
    "token_url": "https://auth.example.com/oauth/token",
    "client_id": "your_client_id",
    "client_secret": "your_client_secret",
    "scope": "read write"
  }
}
```

## JWT Example

```json
{
  "id": "jwt-api",
  "name": "JWT Protected API",
  "description": "API using JWT authentication",
  "base_url": "https://api.example.com",
  "auth": {
    "type": "jwt",
    "secret": "your-jwt-secret",
    "algorithm": "HS256",
    "payload": {
      "sub": "user123",
      "iss": "myapp"
    }
  }
}
```

## Usage

To use these examples:

1. **Via CLI:**
   ```bash
   python main.py
   # Select "Create New Connector"
   # Fill in values from examples above
   ```

2. **Via API:**
   ```bash
   curl -X POST http://localhost:8000/connectors/create \
     -H "Content-Type: application/json" \
     -d @examples/httpbin.json
   ```

3. **Via Python:**
   ```python
   import json
   from models.config import ConnectorConfig
   from connectors import ConnectorFactory
   
   with open('examples/httpbin.json') as f:
       config_dict = json.load(f)
   
   config = ConnectorConfig(**config_dict)
   connector = ConnectorFactory.create(config)
   ```

## Testing Examples

After creating a connector, test it:

```bash
# Via API
curl -X POST http://localhost:8000/connectors/httpbin/test

# Execute a request
curl -X POST http://localhost:8000/connectors/httpbin/execute \
  -H "Content-Type: application/json" \
  -d '{
    "method": "GET",
    "endpoint": "/get",
    "query_params": {"test": "value"}
  }'
```
