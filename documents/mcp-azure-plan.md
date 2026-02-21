# Implementation Plan: Azure MCP Integration

> **Last Audited:** 2026-02-18
> **Source:** https://learn.microsoft.com/en-us/azure/developer/azure-mcp-server/get-started
> **Audit Result:** 4 Critical | 3 High | 3 Medium issues identified and corrected below.

---

## 1. Goal

Integrate the **Azure MCP Server** into the EchoAI Workflow Builder so that agents can
execute natural language queries against 42+ Azure services directly from an MCP node in
the workflow graph.

**Target**: A user types a natural language prompt (e.g. "List all my resource groups") into
a workflow. An agent node interprets the intent, selects the correct Azure MCP tool, and an
MCP node executes it — returning live Azure data back into the workflow state.

---

## 2. How the Azure MCP Server Actually Works (from Official Docs)

> Source: https://learn.microsoft.com/en-us/azure/developer/azure-mcp-server/get-started/languages/python

### 2.1 Official Python Integration Pattern

Microsoft's official approach uses the **`mcp` Python library** (`pip install mcp`).
The library handles the entire JSON-RPC 2.0 protocol internally — including `initialize`,
`notifications/initialized`, `tools/list`, and `tools/call`. **We must not reimplement this
protocol from scratch.**

**The complete official pattern (verbatim from Microsoft docs):**

```python
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

server_params = StdioServerParameters(
    command="npx",
    args=["-y", "@azure/mcp@latest", "server", "start"],
    env=None
)

async with stdio_client(server_params) as (read, write):
    async with ClientSession(read, write) as session:
        await session.initialize()                              # handles full JSON-RPC handshake
        tools = await session.list_tools()                     # tools/list → returns tool registry
        result = await session.call_tool(tool_name, arguments) # tools/call → executes tool
```

### 2.2 How Tool Selection Works (LLM-in-the-Loop)

The `mcp` library exposes discovered tools as OpenAI-compatible `function` definitions.
The LLM (GPT-4o, Claude, etc.) receives the tool list and decides which tool to call based
on the user's natural language prompt. The client then calls `session.call_tool()` with
the name and arguments chosen by the LLM.

```python
# Tools formatted for OpenAI function-calling
available_tools = [{
    "type": "function",
    "function": {
        "name": tool.name,
        "description": tool.description,
        "parameters": tool.inputSchema
    }
} for tool in tools.tools]

# LLM selects the tool → client executes it
result = await session.call_tool(tool_call.function.name, function_args)
```

### 2.3 Authentication

The Azure MCP Server uses **credential chain mode** — it tries (in order):
1. Environment variables (`AZURE_CLIENT_ID`, `AZURE_CLIENT_SECRET`, `AZURE_TENANT_ID`)
2. Visual Studio Code / Visual Studio session
3. Azure CLI (`az login`)
4. Azure PowerShell (`Connect-AzAccount`)
5. Interactive browser

**No API key is required.** Credentials are inherited from the host environment.
The MCP node must pass through the host's full `os.environ` when spawning the server.

### 2.4 Server Start Parameters

The server supports optional flags that control behaviour:

| Flag | Values | Purpose |
|------|--------|---------|
| `--mode` | `namespace` (default), `consolidated`, `all`, `single` | Controls tool grouping |
| `--namespace` | `storage`, `keyvault`, `cosmos`, etc. | Filter which namespaces to expose |
| `--read-only` | flag | Disable all write operations |
| `--tool` | tool name | Expose a single specific tool |

### 2.5 Tool Namespaces (from Official Tools Reference)

Tools are grouped by namespace. The server exposes tools with names following the pattern
`azmcp_<namespace>_<action>`. Tool names are discovered dynamically via `list_tools()`.

| Namespace | Service |
|-----------|---------|
| `group` | Azure Resource Groups |
| `subscription` | Azure Subscriptions |
| `storage` | Azure Storage (blobs, accounts, tables) |
| `keyvault` | Azure Key Vault (secrets, keys, certs) |
| `cosmos` | Azure Cosmos DB |
| `sql` | Azure SQL (servers, databases, firewall rules, elastic pools) |
| `aks` | Azure Kubernetes Service |
| `acr` | Azure Container Registry |
| `appservice` | Azure App Service |
| `functionapp` | Azure Functions |
| `monitor` | Azure Monitor / Log Analytics |
| `eventgrid` | Azure Event Grid |
| `servicebus` | Azure Service Bus |
| `eventhubs` | Azure Event Hubs |
| `appconfig` | Azure App Configuration |
| `communication` | Azure Communication Services (SMS, Email) |
| `search` | Azure AI Search |
| `foundry` | Microsoft AI Foundry (models, agents, threads) |
| `speech` | Azure AI Services Speech |
| `kusto` | Azure Data Explorer |
| `postgres` | Azure Database for PostgreSQL |
| `mysql` | Azure Database for MySQL |
| `redis` | Azure Redis Cache |
| `policy` | Azure Policy |
| `role` | Azure RBAC |
| `quota` | Azure Quotas |
| `resourcehealth` | Azure Resource Health |
| `confidentialledger` | Azure Confidential Ledger |
| `fileshares` | Azure File Shares |
| `storagesync` | Azure File Sync |
| `managedlustre` | Azure Managed Lustre |
| `virtualdesktop` | Azure Virtual Desktop |
| `grafana` | Azure Managed Grafana |
| `workbooks` | Azure Workbooks |
| `loadtesting` | Azure Load Testing |
| `bicepschema` | Azure Bicep |
| `cloudarchitect` | Cloud Architect guidance |
| `marketplace` | Azure Marketplace |
| `deploy` | Azure Deploy |
| `extension` | Azure CLI / Azure Developer CLI |
| `applens` | Azure App Lens |
| `signalr` | Azure SignalR |
| `applicationinsights` | Application Insights |
| `datadog` | Azure Native ISV / Datadog |

### 2.6 Elicitation (User Confirmation for Sensitive Operations)

The Azure MCP Server has a security mechanism called **elicitation** for tools that expose
sensitive data (Key Vault secrets, connection strings, certificate private keys).
Before returning sensitive data, the server sends an elicitation request asking for user
confirmation. The MCP client must handle this response type.

The `mcp` Python library handles elicitation via `ClientSession`. EchoAI's integration
must either:
- Surface the confirmation prompt to the user (preferred for interactive workflows).
- Pass `--insecure-disable-user-confirmation` to suppress elicitation in automated scenarios.

---

## 3. Audit Findings & Corrections

### Issue Table

| ID | Severity | Location | Issue | Correction |
|----|----------|----------|-------|------------|
| C-1 | CRITICAL | Plan | Frontend file `workflow_builder_ide.html` does not exist | Identify correct frontend file before any work |
| C-2 | CRITICAL | Plan | `STDIOMCPConnector` treated as pending — already implemented | Remove from TODO; focus on integration |
| C-3 | CRITICAL | Plan | `ToolExecutor` does not exist | Target `WorkflowCompiler._create_mcp_node` instead |
| C-4 | CRITICAL | `stdio.py` + Plan | Plan proposes reimplementing JSON-RPC 2.0 manually | Use official `mcp` Python library (`pip install mcp`) — it handles the full protocol |
| H-1 | HIGH | `compiler.py:2230` | Ad-hoc block only instantiates `HTTPMCPConnector` | Add STDIO branch using `mcp.ClientSession` + `stdio_client` |
| H-2 | HIGH | `compiler.py:2245` | Default 30s timeout insufficient for Azure | Set 120s minimum for STDIO/Azure path |
| H-3 | HIGH | `stdio.py:101` | `validate_config` requires non-empty schemas; Azure schemas are dynamic | Bypass schema validation for STDIO/Azure path (schemas discovered at runtime) |
| M-1 | MEDIUM | `compiler.py:2251` | `asyncio.run()` crashes inside running event loop | Use async-aware event loop handling |
| M-2 | MEDIUM | `stdio.py:237` | `stderr` truncated at 500 chars — hides Azure CLI auth errors | Increase to 2000 chars |
| M-3 | MEDIUM | `stdio.py:153` | Windows requires `ProactorEventLoop` for subprocess STDIO | Add platform guard |

### C-4 Correction Detail (Most Important)

**Original plan proposed:** Write a `call_tool()` method implementing raw JSON-RPC 2.0
messages (initialize → notifications/initialized → tools/call) manually.

**Correct approach:** Use the official `mcp` Python library.

```
pip install mcp
```

The `mcp.ClientSession` + `mcp.client.stdio.stdio_client` classes already implement the
entire MCP protocol including the handshake, tool discovery, and tool invocation.
`STDIOMCPConnector` does NOT need a `call_tool()` method — the `mcp` library replaces it
for this integration.

The STDIO branch in `compiler.py` should use `ClientSession` directly, NOT `STDIOMCPConnector`.

---

## 4. Corrected Architecture

### Frontend

> **[BLOCKED — C-1]**
> The original file `workflow_builder_ide.html` does not exist. All frontend is Streamlit.
> Identify the correct frontend file before proceeding.

Once the correct file is identified:
- Add "Azure MCP" option to the MCP Server selector.
- Show a `command` field (pre-filled: `npx -y @azure/mcp@latest server start`).
- Show a `timeout` field (default: `120`).
- Pass `"transport_type": "stdio"` in the node config payload.

---

### Backend (`apps/workflow/designer/compiler.py`)

**Target function:** `WorkflowCompiler._create_mcp_node` — line **2154**
**Change target:** Ad-hoc mode `else` block — lines **2230–2251**

> `ToolExecutor` does not exist. Removed from plan. See C-3.

**New STDIO branch logic (pseudocode — no code until commanded):**

```
if live_config has "command" key OR transport_type == "stdio":

    1. Parse command string → command + args list
       "npx -y @azure/mcp@latest server start"
       → command="npx", args=["-y", "@azure/mcp@latest", "server", "start"]

    2. Build StdioServerParameters(command=command, args=args, env=os.environ.copy())

    3. async with stdio_client(server_params) as (read, write):
         async with ClientSession(read, write) as session:
           await session.initialize()

           # Option A: Direct tool call (tool_name + arguments from config)
           result = await session.call_tool(tool_name, arguments)

           # Option B: LLM-driven tool selection (pass tool list to agent LLM)
           tools = await session.list_tools()
           # → pass tools as OpenAI-compatible functions to the EchoAI LLM agent
           # → agent selects tool → call session.call_tool(selected_name, selected_args)

    4. Extract result.content[0].text → push to workflow state

else:
    → existing HTTPMCPConnector path (unchanged)
```

**Architecture decision — Option A vs Option B:**

| Option | Description | When to Use |
|--------|-------------|-------------|
| A — Direct | `tool_name` + `arguments` come from the node config (set by a prior agent node) | When the workflow has an agent node before the MCP node that decides the tool |
| B — LLM-driven | The MCP node itself contains an LLM loop that picks the tool from the discovered list | When the MCP node is standalone and receives only a natural language prompt |

Both options must be supported. Option A is simpler and aligns with EchoAI's existing
agent → MCP → agent pattern. Option B mirrors the Microsoft docs reference implementation.

---

### New Dependency

```
mcp                  # Official MCP Python library (ClientSession, StdioServerParameters)
```

Must be added to `requirements.txt`. No other new dependency is needed for the protocol.

---

## 5. Required Changes (Ordered)

### Change 1 — `requirements.txt`
Add `mcp` to the backend requirements file.

### Change 2 — `apps/workflow/designer/compiler.py` (lines 2230–2251)

Add STDIO branch before the `HTTPMCPConnector` block:
- Detect STDIO by `live_config.get("command")` or `transport_type == "stdio"`.
- Parse command string → `command`, `args` list.
- Use `mcp.StdioServerParameters` + `mcp.client.stdio.stdio_client` + `mcp.ClientSession`.
- Call `session.initialize()` → `session.call_tool(tool_name, arguments)`.
- Use `timeout=live_config.get("timeout", 120)` — not 30s.
- Pass `env=os.environ.copy()` so Azure CLI credentials are inherited.
- Add Windows `ProactorEventLoop` guard before entering the async context.

### Change 3 — `asyncio` handling in `mcp_node` closure
Declare `mcp_node` as `async def` (already supported by LangGraph) and `await` the
`ClientSession` calls directly instead of using `asyncio.run()`.

### Change 4 — `echolib/Get_connector/Get_MCP/stdio.py` (minor fixes only)
- Increase `stderr` capture from 500 to 2000 chars (`stdio.py:237`).
- `STDIOMCPConnector` itself is NOT used for the Azure MCP path — the `mcp` library is
  used instead. No `call_tool()` method needed on the connector class.

---

## 6. Corrected Implementation Order

| Step | Action | Depends On | Status |
|------|--------|------------|--------|
| 1 | Identify correct frontend file path | — | BLOCKED |
| 2 | Add `mcp` to `requirements.txt` | — | **Do first — all backend depends on this** |
| 3 | Add STDIO branch to `compiler.py:2230` using `mcp.ClientSession` (Option A: direct tool call) | 2 | Not started |
| 4 | Extend STDIO branch to support Option B (LLM-driven tool selection) | 3 | Not started |
| 5 | Fix `asyncio.run()` → `async def mcp_node` | 3 | Not started |
| 6 | Increase stderr capture to 2000 chars in `stdio.py` | — | Not started |
| 7 | Add Azure MCP UI to correct frontend file | 1 | Not started |
| 8 | Test: `"List resource groups"` (smoke test) | 3, 5, 7 | Not started |
| 9 | Test: `"List all SQL servers"` (verify SQL namespace) | 8 | Not started |
| 10 | Test: `"List all secrets in my key vault"` (verify elicitation) | 8 | Not started |

---

## 7. Prerequisites

1. **Node.js** — `npx` must be in the system PATH.
2. **Azure CLI** — user must run `az login`. Server inherits credentials via `os.environ`.
3. **`mcp` Python library** — `pip install mcp`.
4. **Timeout** — minimum 120s. First `npx` run downloads the package (can add 30–60s).
5. **Windows** — `asyncio.WindowsProactorEventLoopPolicy` must be set before subprocess spawn.
6. **RBAC** — user account must have appropriate Azure RBAC roles for the services they query.

---

## 8. Detailed Capabilities

### A. Execution Strategy

- **Primary (npx):** `npx -y @azure/mcp@latest server start` — no .NET SDK management,
  always uses latest version, works on any machine with Node.js + Azure CLI.
- **First-run note:** `npx` downloads `@azure/mcp@latest` on first run (30–60s).
  120s timeout accommodates this. Subsequent runs use the cache (~2–5s startup).

### B. Full Query Coverage (Test Cases)

#### Microsoft Foundry
- List Microsoft Foundry models
- Deploy Microsoft Foundry models
- List Microsoft Foundry model deployments
- List knowledge indexes
- Get knowledge index schema configuration
- Create Microsoft Foundry agents
- List Microsoft Foundry agents
- Connect and query Microsoft Foundry agents
- Evaluate Microsoft Foundry agents
- Get SDK samples for interacting with Microsoft Foundry agent
- Create Microsoft Foundry agent threads
- List Microsoft Foundry agent threads
- Get messages of a Microsoft Foundry thread

#### Azure Advisor
- "List my Advisor recommendations"

#### Azure AI Search
- "What indexes do I have in my Azure AI Search service 'mysvc'?"
- "Let's search this index for 'my search query'"

#### Azure AI Services Speech
- "Convert this audio file to text using Azure Speech Services"
- "Recognize speech from my audio file with language detection"
- "Transcribe speech from audio with profanity filtering"
- "Transcribe audio with phrase hints for better accuracy"
- "Convert text to speech and save to output.wav"
- "Synthesize speech from 'Hello, welcome to Azure' with Spanish voice"
- "Generate MP3 audio from text with high quality format"

#### Azure App Configuration
- "List my App Configuration stores"
- "Show my key-value pairs in App Config"

#### Azure App Lens
- "Help me diagnose issues with my app"

#### Azure App Service
- "List the websites in my subscription"
- "Show me the websites in my 'my-resource-group' resource group"
- "Get the details for website 'my-website'"
- "Get the details for app service plan 'my-app-service-plan'"

#### Azure CLI Generate
- Generate Azure CLI commands based on user intent

#### Azure CLI Install
- Get installation instructions for Azure CLI, Azure Developer CLI, and Azure Functions Core Tools CLI

#### Azure Communication Services
- "Send an SMS message to +1234567890"
- "Send SMS with delivery reporting enabled"
- "Send a broadcast SMS to multiple recipients"
- "Send SMS with custom tracking tag"
- "Send an email from 'sender@example.com' to 'recipient@example.com' with subject 'Hello' and message 'Welcome!'"
- "Send an HTML email to multiple recipients with CC and BCC using Azure Communication Services"
- "Send an email with reply-to address 'reply@example.com' and subject 'Support Request'"
- "Send an email from my communication service endpoint with custom sender name and multiple recipients"
- "Send an email to 'user1@example.com' and 'user2@example.com' with subject 'Team Update' and message 'Please review the attached document.'"

#### Azure Compute
- "List all my managed disks in subscription 'my-subscription'"
- "Show me all disks in resource group 'my-resource-group'"
- "Get details of disk 'my-disk' in resource group 'my-resource-group'"
- "List all virtual machines in my subscription"
- "Show me all VMs in resource group 'my-resource-group'"
- "Get details for virtual machine 'my-vm' in resource group 'my-resource-group'"
- "Get virtual machine 'my-vm' with instance view including power state and runtime status"
- "Show me the power state and provisioning status of VM 'my-vm'"
- "What is the current status of my virtual machine 'my-vm'?"

#### Azure Container Apps
- "List the container apps in my subscription"
- "Show me the container apps in my 'my-resource-group' resource group"

#### Azure Confidential Ledger
- "Append entry {\"foo\":\"bar\"} to ledger contoso"
- "Get entry with id 2.40 from ledger contoso"

#### Azure Container Registry (ACR)
- "List all my Azure Container Registries"
- "Show me my container registries in the 'my-resource-group' resource group"
- "List all my Azure Container Registry repositories"

#### Azure Cosmos DB
- "Show me all my Cosmos DB databases"
- "List containers in my Cosmos DB database"

#### Azure Data Explorer
- "Get Azure Data Explorer databases in cluster 'mycluster'"
- "Sample 10 rows from table 'StormEvents' in Azure Data Explorer database 'db1'"

#### Azure Event Grid
- "List all Event Grid topics in subscription 'my-subscription'"
- "Show me the Event Grid topics in my subscription"
- "List all Event Grid topics in resource group 'my-resourcegroup' in my subscription"
- "List Event Grid subscriptions for topic 'my-topic' in resource group 'my-resourcegroup'"
- "List Event Grid subscriptions for topic 'my-topic' in subscription 'my-subscription'"
- "List Event Grid Subscriptions in subscription 'my-subscription'"
- "List Event Grid subscriptions for topic 'my-topic' in location 'my-location'"
- "Publish an event with data '{\"name\": \"test\"}' to topic 'my-topic' using CloudEvents schema"
- "Send custom event data to Event Grid topic 'analytics-events' with EventGrid schema"

#### Azure File Shares
- "Get details about a specific file share in my resource group"
- "Create a new Azure managed file share with NFS protocol"
- "Create a file share with 64 GiB storage, 3000 IOPS, and 125 MiB/s throughput"
- "Update the provisioned storage size of my file share"
- "Update network access settings for my file share"
- "Delete a file share from my resource group"
- "Check if a file share name is available"
- "Get details about a file share snapshot"
- "Create a snapshot of my file share"
- "Update tags on a file share snapshot"
- "Delete a file share snapshot"
- "Get a private endpoint connection for my file share"
- "Update private endpoint connection status to Approved"
- "Delete a private endpoint connection"
- "Get file share limits and quotas for a region"
- "Get provisioning recommendations for my file share workload"
- "Get usage data and metrics for my file share"

#### Azure Key Vault
- "List all secrets in my key vault 'my-vault'"
- "Create a new secret called 'apiKey' with value 'xyz' in key vault 'my-vault'"
- "List all keys in key vault 'my-vault'"
- "Create a new RSA key called 'encryption-key' in key vault 'my-vault'"
- "List all certificates in key vault 'my-vault'"
- "Import a certificate file into key vault 'my-vault' using the name 'tls-cert'"
- "Get the account settings for my key vault 'my-vault'"

#### Azure Kubernetes Service (AKS)
- "List my AKS clusters in my subscription"
- "Show me all my Azure Kubernetes Service clusters"
- "List the node pools for my AKS cluster"
- "Get details for the node pool 'np1' of my AKS cluster 'my-aks-cluster' in the 'my-resource-group' resource group"

#### Azure Managed Lustre
- "List the Azure Managed Lustre clusters in resource group 'my-resource-group'"
- "How many IP Addresses I need to create a 128 TiB cluster of AMLFS 500?"
- "Check if 'my-subnet-id' can host an Azure Managed Lustre with 'my-size' TiB and 'my-sku' in 'my-region'"
- "Create a 4 TiB Azure Managed Lustre filesystem in 'my-region' attaching to 'my-subnet' in virtual network 'my-virtual-network'"

#### Azure Monitor
- "Query my Log Analytics workspace"

#### Azure Resource Management
- "List my resource groups"
- "List my Azure CDN endpoints"
- "Help me build an Azure application using Node.js"

#### Azure SQL Database
- "List all SQL servers in my subscription"
- "List all SQL servers in my resource group 'my-resource-group'"
- "Show me details about my Azure SQL database 'mydb'"
- "List all databases in my Azure SQL server 'myserver'"
- "Update the performance tier of my Azure SQL database 'mydb'"
- "Rename my Azure SQL database 'mydb' to 'newname'"
- "List all firewall rules for my Azure SQL server 'myserver'"
- "Create a firewall rule for my Azure SQL server 'myserver'"
- "Delete a firewall rule from my Azure SQL server 'myserver'"
- "List all elastic pools in my Azure SQL server 'myserver'"
- "List Active Directory administrators for my Azure SQL server 'myserver'"
- "Create a new Azure SQL server in my resource group 'my-resource-group'"
- "Show me details about my Azure SQL server 'myserver'"
- "Delete my Azure SQL server 'myserver'"

#### Azure Storage
- "List my Azure storage accounts"
- "Get details about my storage account 'mystorageaccount'"
- "Create a new storage account in East US with Data Lake support"
- "Get details about my Storage container"
- "Upload my file to the blob container"

#### Azure Migrate
- "Generate a Platform Landing Zone"
- "Turn off DDoS protection in my Platform Landing Zone"
- "Turn off Bastion host in my Platform Landing Zone"

### C. Complete List of Supported Azure Services (42+)

- Microsoft Foundry — AI model management, deployment, knowledge index management (`foundry`)
- Azure Advisor — Recommendations
- Azure AI Search — Search engine / vector database (`search`)
- Azure AI Services Speech — Speech-to-text and text-to-speech (`speech`)
- Azure App Configuration — Configuration management (`appconfig`)
- Azure App Service — Web app hosting (`appservice`)
- Azure Best Practices — Secure, production-grade guidance
- Azure CLI Generate — Natural language to CLI commands (`extension`)
- Azure CLI Install — CLI installation instructions (`extension`)
- Azure Communication Services — SMS and email (`communication`)
- Azure Compute — VM, VMSS, and Disk management
- Azure Confidential Ledger — Tamper-proof ledger (`confidentialledger`)
- Azure Container Apps — Container hosting
- Azure Container Registry (ACR) — Container registry (`acr`)
- Azure Cosmos DB — NoSQL database (`cosmos`)
- Azure Data Explorer — Analytics / KQL (`kusto`)
- Azure Database for MySQL — MySQL management (`mysql`)
- Azure Database for PostgreSQL — PostgreSQL management (`postgres`)
- Azure Event Grid — Event routing and publishing (`eventgrid`)
- Azure File Shares — Managed file share operations (`fileshares`)
- Azure Functions — Function App management (`functionapp`)
- Azure Key Vault — Secrets, keys, certificates (`keyvault`)
- Azure Kubernetes Service (AKS) — Container orchestration (`aks`)
- Azure Load Testing — Performance testing (`loadtesting`)
- Azure Managed Grafana — Monitoring dashboards (`grafana`)
- Azure Managed Lustre — High-performance filesystem (`managedlustre`)
- Azure Marketplace — Product discovery (`marketplace`)
- Azure Migrate — Platform Landing Zone generation
- Azure Monitor — Logging, metrics, health (`monitor`)
- Azure Policy — Organizational standards (`policy`)
- Azure Native ISV Services — Third-party integrations (`datadog`)
- Azure Quick Review CLI — Compliance scanning (`extension`)
- Azure Quota — Resource quota and usage (`quota`)
- Azure RBAC — Access control management (`role`)
- Azure Redis Cache — In-memory data store (`redis`)
- Azure Resource Groups — Resource organization (`group`)
- Azure Service Bus — Message queuing (`servicebus`)
- Azure Service Fabric — Managed cluster node operations
- Azure Service Health — Resource health (`resourcehealth`)
- Azure SQL Database — Relational database management (`sql`)
- Azure SQL Elastic Pool — Database resource sharing (`sql`)
- Azure SQL Server — Server administration (`sql`)
- Azure Storage — Blob storage (`storage`)
- Azure Storage Sync — Azure File Sync management (`storagesync`)
- Azure Subscription — Subscription management (`subscription`)
- Azure Terraform Best Practices — Infrastructure as code (`bicepschema`)
- Azure Virtual Desktop — Virtual desktop infrastructure (`virtualdesktop`)
- Azure Workbooks — Custom visualizations (`workbooks`)
- Bicep — Azure resource templates (`bicepschema`)
- Cloud Architect — Guided architecture design (`cloudarchitect`)
- Application Insights — Monitoring (`applicationinsights`)
- Azure Deploy — Resource deployment (`deploy`)
- Azure Event Hubs — Event streaming (`eventhubs`)
- Azure SignalR — Real-time messaging (`signalr`)

### D. Implementation Strategy Notes

- **Use the `mcp` library, not raw JSON-RPC.** `mcp.ClientSession` handles all protocol
  framing, `initialize`, `notifications/initialized`, `tools/list`, and `tools/call`.
- **Tools are discovered dynamically at runtime.** Do not hardcode tool names.
  Call `session.list_tools()` on each session start to get the current registry.
- **LLM selects the tool.** The agent node (prior in the workflow) or the MCP node itself
  (Option B) maps natural language to a specific tool name + arguments.
- **Inherit the full environment.** Pass `env=os.environ.copy()` so Azure CLI credentials,
  PATH (for `npx`, `az`), and .NET runtime locations are all available to the subprocess.
- **120s minimum timeout.** First `npx` run downloads the package. Subsequent runs are fast.
- **Elicitation.** Key Vault and other sensitive tools trigger a user-confirmation prompt.
  EchoAI must handle `elicitation` responses from `ClientSession` in the integration.
- **Windows.** Set `asyncio.WindowsProactorEventLoopPolicy` before spawning the subprocess.
