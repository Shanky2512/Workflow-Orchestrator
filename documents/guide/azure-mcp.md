Here is the same content rewritten clearly in structured points:

---

## Steps to Install and Configure Azure CLI

### Step 1 — Install Azure CLI

* Go to the official Azure CLI installer page:
  [https://aka.ms/installazurecliwindows](https://aka.ms/installazurecliwindows)
* Download the Windows `.msi` installer.
* Run the installer and complete the installation process.
* After installation finishes, close your terminal.
* Reopen Command Prompt or PowerShell to refresh the environment.

---

### Step 2 — Verify Installation

* Open a new terminal (Command Prompt or PowerShell).
* Run the following command:

az version

* You should see a JSON output showing Azure CLI version information.
* This confirms Azure CLI is installed correctly.

---

### Step 3 — Login to Azure

* Run the following command in the terminal:

az login

* This will open a browser window automatically.
* Sign in using your Azure account credentials.
* After successful login, your Azure subscriptions will be displayed in the terminal.

---

### Step 4 — Get Your Subscription ID

* Run the following command:

az account show --query id -o tsv

* This will output your Subscription ID.
* It will look like this:


xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx

* Copy this ID.

---

### Step 5 — Paste Your Subscription ID

* Paste the copied Subscription ID into the following JSON format:

{
  "subscription": "PASTE-YOUR-ID-HERE"
}

* This will be used to update the workflow arguments JSON so the workflow can function correctly.

---

If needed, I can also provide automated verification steps or a script to validate Azure CLI, login status, and subscription configuration.


=================================================================================================================


## Pre-Run Checklist (Host Machine Requirements)

### Step 1 — Login to Azure CLI

* Open a terminal (Command Prompt, PowerShell, or Terminal).
* Run the following command:

```
az login
```

* A browser window will open automatically.
* Sign in using your Azure account credentials.
* This authenticates Azure CLI on your machine.
* You only need to do this once.

---

### Step 2 — Verify Node.js Installation

* In the terminal, run:

```
node --version
```

* If Node.js is installed, you will see a version number, for example:

```
v20.11.0
```

* If you do not see a version number or get an error, install Node.js from:

  [https://nodejs.org](https://nodejs.org)

---

### Important Notes

* Both `az login` and `node --version` must be run on your host machine.
* These commands are not run inside EchoAI.
* They only need to be done once unless you log out or reinstall Node.js.


Based on the discovered tool list from your Azure MCP server, here are the compiled Azure MCP use cases in the same structured format:

---



============================================================================================================


## Use Cases


## 1. List Subscriptions (No Subscription ID Required)

```
Based on the discovered tool list from your Azure MCP server:

┌───────────────────┬───────────────────────────────────────┐
│ Field             │ Value                                 │
├───────────────────┼───────────────────────────────────────┤
│ Transport         │ STDIO                                 │
├───────────────────┼───────────────────────────────────────┤
│ Command           │ npx -y @azure/mcp@latest server start │
├───────────────────┼───────────────────────────────────────┤
│ Timeout (seconds) │ 120                                   │
├───────────────────┼───────────────────────────────────────┤
│ Tool Name         │ subscription_list                     │
├───────────────────┼───────────────────────────────────────┤
│ Arguments (JSON)  │ {}                                    │
└───────────────────┴───────────────────────────────────────┘
```

Purpose: Retrieves all Azure subscriptions associated with your account.

---

## 2. Get Azure Pricing Information (No Subscription ID Required)

```
Based on the discovered tool list from your Azure MCP server:

┌───────────────────┬───────────────────────────────────────┐
│ Field             │ Value                                 │
├───────────────────┼───────────────────────────────────────┤
│ Transport         │ STDIO                                 │
├───────────────────┼───────────────────────────────────────┤
│ Command           │ npx -y @azure/mcp@latest server start │
├───────────────────┼───────────────────────────────────────┤
│ Timeout (seconds) │ 120                                   │
├───────────────────┼───────────────────────────────────────┤
│ Tool Name         │ pricing                               │
├───────────────────┼───────────────────────────────────────┤
│ Arguments (JSON)  │ {"learn": true}                      │
└───────────────────┴───────────────────────────────────────┘
```

Purpose: Explore Azure pricing tools and supported pricing queries.

---

## 3. Access Azure Documentation (No Subscription ID Required)

```
Based on the discovered tool list from your Azure MCP server:

┌───────────────────┬───────────────────────────────────────┐
│ Field             │ Value                                 │
├───────────────────┼───────────────────────────────────────┤
│ Transport         │ STDIO                                 │
├───────────────────┼───────────────────────────────────────┤
│ Command           │ npx -y @azure/mcp@latest server start │
├───────────────────┼───────────────────────────────────────┤
│ Timeout (seconds) │ 120                                   │
├───────────────────┼───────────────────────────────────────┤
│ Tool Name         │ documentation                         │
├───────────────────┼───────────────────────────────────────┤
│ Arguments (JSON)  │ {"learn": true}                      │
└───────────────────┴───────────────────────────────────────┘
```

Purpose: Discover Azure MCP capabilities and supported operations.

---

## 4. List Resource Groups (Requires Subscription ID)

```
Based on the discovered tool list from your Azure MCP server:

┌───────────────────┬────────────────────────────────────────────┐
│ Field             │ Value                                      │
├───────────────────┼────────────────────────────────────────────┤
│ Transport         │ STDIO                                      │
├───────────────────┼────────────────────────────────────────────┤
│ Command           │ npx -y @azure/mcp@latest server start      │
├───────────────────┼────────────────────────────────────────────┤
│ Timeout (seconds) │ 120                                        │
├───────────────────┼────────────────────────────────────────────┤
│ Tool Name         │ group_list                                 │
├───────────────────┼────────────────────────────────────────────┤
│ Arguments (JSON)  │ {"subscription": "your-subscription-id"}  │
└───────────────────┴────────────────────────────────────────────┘
```

Purpose: Lists all resource groups in a specific subscription.

---

## 5. List Storage Accounts (Requires Subscription ID)

```
Based on the discovered tool list from your Azure MCP server:

┌───────────────────┬────────────────────────────────────────────────────────┐
│ Field             │ Value                                                  │
├───────────────────┼────────────────────────────────────────────────────────┤
│ Transport         │ STDIO                                                  │
├───────────────────┼────────────────────────────────────────────────────────┤
│ Command           │ npx -y @azure/mcp@latest server start                  │
├───────────────────┼────────────────────────────────────────────────────────┤
│ Timeout (seconds) │ 120                                                    │
├───────────────────┼────────────────────────────────────────────────────────┤
│ Tool Name         │ storage                                                │
├───────────────────┼────────────────────────────────────────────────────────┤
│ Arguments (JSON)  │ {"subscription": "your-subscription-id", "learn": true} │
└───────────────────┴────────────────────────────────────────────────────────┘
```

Purpose: Lists storage accounts and explores storage-related operations.

---

If you want, I can also generate **Azure MCP tools for VM listing, deployments, AKS, and Key Vault**, which are the most useful real production scenarios.


## Make sure the node is install.
If Node.js is not installed:
Download and install from https://nodejs.org (LTS version). After installing, restart the backend.

If Node.js is installed but PATH is missing:
Note the path returned by where npx (e.g. C:\Program Files\nodejs\npx.cmd) and use the full path in the Command field:

C:\Program Files\nodejs\npx.cmd -y @azure/mcp@latest server start


rystallizing… (thought for 2s)                                                                                            ⎿  Tip: Working with HTML/CSS? Add the frontend-design plugin:                                                           
     /plugin marketplace add anthropics/claude-code                                                                             /plugin install frontend-design@claude-code-plugins