MCP Connector API - Endpoint Reference
1. CREATE Connector
Endpoint: POST /connectors/mcp/create
Description: Creates a new MCP connector, validates the configuration, auto-tests it
with the example payload, and saves to storage. Returns the connector ID and test
result.
Request Payload:
json
{
 "name": "openweather-api",
 "description": "Get weather data for any city",
 "transport_type": "http",
 "auth_config": {
 "type": "bearer",
 "token": "eyJ0eXAiOiJKV1Qi..."
 },
 "endpoint_url": "https://api.openweathermap.org/data/2.5/weather",
 "method": "GET",
 "headers": {
 "Accept": "application/json"
 },
 "query_params": {},
 "query_mapping": {
 "city": "q",
 "unit": "units"
 },
 "input_schema": {
 "type": "object",
 "required": ["city"],
 "properties": {
 "city": {"type": "string"},
 "unit": {"type": "string", "default": "metric"}
 }
 },
 "output_schema": {
 "type": "object"
 },
 "timeout": 15,
 "retry_count": 0,
 "verify_ssl": false,
 "example_payload": {
 "city": "London",
 "unit": "metric"
 }
}
Required Fields:
Field Type Description
name string Unique connector name
transport_type string http, sse, or stdio
auth_config object Authentication config (see auth types below)
input_schema object JSON schema of accepted input
output_schema object JSON schema of expected output
example_payload object Test payload - cannot be null/empty
Optional Fields:
Field Type Default Description
description string "" What connector does
method string GET HTTP method
headers object {} HTTP headers
query_params object {} Static query parameters
query_mapping object {} Maps input fields to API params
timeout integer 30 Request timeout in seconds
retry_count integer 0 Number of retries on failure
verify_ssl boolean true SSL verification
Auth Config Types:
json
// No auth
{"type": "none"}
// Bearer token
{"type": "bearer", "token": "eyJ0eXAi..."}
// API Key in query
{"type": "api_key", "location": "query", "key_name": "appid", "api_key": "abc123"}
// API Key in header
{"type": "api_key", "location": "header", "key_name": "X-API-Key", "api_key": "abc123"}
// Basic auth
{"type": "basic", "username": "user", "password": "pass"}
// OAuth2
{
 "type": "oauth2",
 "tenant_id": "...",
 "client_id": "...",
 "client_secret": "...",
 "token_url": "https://login.microsoftonline.com/.../oauth2/v2.0/token",
 "scope": "https://graph.microsoft.com/.default"
}
// Custom header
{"type": "custom_header", "header_name": "X-Custom", "header_value": "my-value"}
Success Response:
json
{
 "success": true,
 "connector_id": "mcp_f237cb0812c3",
 "validation_status": "validated",
 "message": "Connector created and validated successfully",
 "test_result": {
 "success": true,
 "output": {...},
 "error": null,
 "duration_ms": 450,
 "metadata": {
 "status_code": 200,
 "headers": {...},
 "url": "https://..."
 }
 }
}
Failure Response:
json
{
 "success": false,
 "connector_id": "mcp_f237cb0812c3",
 "validation_status": "failed",
 "message": "Connector created but test failed",
 "test_result": {
 "success": false,
 "output": {...},
 "error": "HTTP 401",
 "duration_ms": 350
 }
}
```
---
## **2. GET Single Connector**
**Endpoint:** `GET /connectors/mcp/{connector_id}`
**Description:** Returns full details of a single connector by ID. Includes all config
EXCEPT sensitive auth fields which are redacted.
**Request Payload:** None (connector_id in URL path)
**Example:**
```
GET /connectors/mcp/mcp_f237cb0812c3
Success Response:
json
{
 "success": true,
 "connector": {
 "connector_id": "mcp_f237cb0812c3",
 "connector_name": "sharepoint-root-site",
 "creation_payload": {
 "name": "sharepoint-root-site",
 "description": "Get SharePoint root site",
 "transport_type": "http",
 "auth_config": {
 "type": "bearer",
 "token": "***REDACTED***"
 },
 "endpoint_url": "https://graph.microsoft.com/v1.0/sites/root",
 "method": "GET",
 "headers": {"Accept": "application/json"},
 "query_mapping": {},
 "input_schema": {...},
 "output_schema": {...},
 "timeout": 15,
 "verify_ssl": false,
 "example_payload": {"$select": "id,displayName"}
 },
 "validation_status": "validated",
 "validation_error": null,
 "tested_at": "2026-02-03T14:34:34",
 "created_at": "2026-02-03T14:34:34",
 "updated_at": "2026-02-03T14:34:34"
 }
}
Failure Response:
json
{
 "success": false,
 "error": "Connector not found: mcp_f237cb0812c3"
}
```
---
## **3. LIST All Connectors**
**Endpoint:** `GET /connectors/mcp`
**Description:** Returns list of all connectors in storage. Shows full details for all
connectors. Sensitive auth fields are redacted.
**Request Payload:** None
**Example:**
```
GET /connectors/mcp
Success Response:
json
{
 "success": true,
 "connectors": [
 {
 "connector_id": "mcp_f237cb0812c3",
 "connector_name": "sharepoint-root-site",
 "creation_payload": {
 "name": "sharepoint-root-site",
 "description": "Get SharePoint root site",
 "transport_type": "http",
 "auth_config": {
 "type": "bearer",
 "token": "***REDACTED***"
 },
 "endpoint_url": "https://graph.microsoft.com/v1.0/sites/root",
 "method": "GET",
 "headers": {...},
 "input_schema": {...},
 "output_schema": {...},
 "example_payload": {...}
 },
 "validation_status": "validated",
 "tested_at": "2026-02-03T14:34:34",
 "created_at": "2026-02-03T14:34:34",
 "updated_at": "2026-02-03T14:34:34"
 },
 {
 "connector_id": "mcp_5dbe90aeffe3",
 "connector_name": "openweather-complete",
 "creation_payload": {...},
 "validation_status": "validated",
 "tested_at": "...",
 "created_at": "...",
 "updated_at": "..."
 }
 ],
 "total": 2
}
4. UPDATE Connector
Endpoint: PUT /connectors/mcp/update
Description: Updates an existing connector using deep merge strategy. Mandatory
tests the updated config before saving. If test fails, changes are NOT saved. Returns list
of changed fields.
Request Payload:
json
{
 "connector_id": "mcp_f237cb0812c3",
 "updates": {
 // Any updatable fields here
 }
}
Updatable Fields:
json
{
 "updates": {
 "name": "new-name",
 "description": "new description",
 "auth_config": {
 "token": "NEW_TOKEN_HERE"
 },
 "endpoint_url": "https://new-endpoint.com",
 "method": "POST",
 "headers": {
 "Accept": "application/json",
 "X-Custom": "value"
 },
 "query_params": {},
 "query_mapping": {"city": "q"},
 "timeout": 30,
 "retry_count": 1,
 "verify_ssl": true,
 "example_payload": {
 "city": "Tokyo"
 }
 }
}
```
**Fields That CANNOT Be Updated:**
```
transport_type → Immutable
input_schema → Immutable
output_schema → Immutable
connector_id → Immutable
auth_config.type → Cannot change auth type
example_payload → Cannot be set to null (can be updated)
```
**Merge Behavior:**
```
Deep merge - updates are merged into existing config:
Existing: {"auth_config": {"type": "bearer", "token": "old"}}
Update: {"auth_config": {"token": "new"}}
Result: {"auth_config": {"type": "bearer", "token": "new"}}
Existing: {"headers": {"Accept": "application/json", "X-Old": "value"}}
Update: {"headers": {"X-New": "value"}}
Result: {"headers": {"Accept": "application/json", "X-Old": "value", "X-New": "value"}}
Success Response:
json
{
 "success": true,
 "connector_id": "mcp_f237cb0812c3",
 "message": "Connector updated and validated successfully",
 "changed_fields": [
 "auth_config.token",
 "timeout",
 "example_payload.city"
 ],
 "test_result": {
 "success": true,
 "output": {...},
 "duration_ms": 450
 }
}
Failure - Test Failed (Changes NOT Saved):
json
{
 "success": false,
 "error": "Update test failed - changes not saved",
 "test_result": {
 "success": false,
 "error": "HTTP 401",
 "output": {...}
 }
}
Failure - Forbidden Field:
json
{
 "success": false,
 "error": "Cannot change the following fields: transport_type"
}
Failure - Auth Type Change:
json
{
 "success": false,
 "error": "Cannot change auth type from 'bearer' to 'api_key'"
}
Failure - No Valid Updates:
json
{
 "success": false,
 "error": "No valid fields to update"
}
```
---
## **5. DELETE Connector**
**Endpoint:** `DELETE /connectors/mcp/{connector_id}`
**Description:** Permanently deletes a connector from storage. Cannot be undone.
**Request Payload:** None (connector_id in URL path)
**Example:**
```
DELETE /connectors/mcp/mcp_f237cb0812c3
Success Response:
json
{
 "success": true,
 "connector_id": "mcp_f237cb0812c3",
 "message": "Connector deleted successfully"
}
Failure Response:
json
{
 "success": false,
 "error": "Connector not found: mcp_f237cb0812c3"
}
6. INVOKE Connector
Endpoint: POST /connectors/mcp/invoke
Description: Executes a connector with given payload. If no payload provided, uses
stored example_payload (must be validated). Loads fresh from storage every time - no
caching. Returns raw API response.
Request Payload:
json
{
 "connector_id": "mcp_f237cb0812c3",
 "payload": {
 "city": "London",
 "unit": "metric"
 }
}
```
**Fields:**
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `connector_id` | string | Yes | ID of connector to invoke |
| `payload` | object | No | Input data. If empty, uses stored example_payload |
**Payload Behavior:**
```
payload provided → Use it
payload not provided + connector validated → Use stored example_payload
payload not provided + no example_payload → Error
Success Response:
json
{
 "success": true,
 "output": {
 // Raw API response here
 "main": {
 "temp": 15.2,
 "humidity": 82
 },
 "weather": [
 {"description": "overcast clouds"}
 ]
 },
 "error": null,
 "duration_ms": 450,
 "metadata": {
 "status_code": 200,
 "headers": {...},
 "url": "https://api.openweathermap.org/..."
 },
 "timestamp": "2026-02-14T10:30:00"
}
Failure - API Error:
json
{
 "success": false,
 "output": {
 "error": {
 "code": "Unauthorized",
 "message": "Invalid token"
 }
 },
 "error": "HTTP 401",
 "duration_ms": 350,
 "metadata": {
 "status_code": 401,
 "headers": {...}
 }
}
Failure - Connector Not Found:
json
{
 "success": false,
 "error": "Connector not found: mcp_f237cb0812c3"
}
Failure - No Payload:
json
{
 "success": false,
 "error": "No payload provided and no validated example payload stored"
}
Quick Reference:
Endpoint Method URL Body
Create POST /connectors/mcp/create Full connector config
Get One GET /connectors/mcp/{id} None
List All GET /connectors/mcp None
Update PUT /connectors/mcp/update {connector_id, updates}
Delete DELETE /connectors/mcp/{id} None
Invoke POST /connectors/mcp/invoke {connector_id, payload}