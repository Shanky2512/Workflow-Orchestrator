# Bug Report: API Node Authentication Configuration Mismatch

## 1. Issue Overview
The Frontend Workflow Builder (`workflow_builder_ide.html`) provides a simplified authentication configuration interface for **API Nodes**. However, the backend (and specifically the Oracle HCM connector example) requires a structured authentication object, particularly for **Basic Auth** which necessitates separate `username` and `password` fields.

The current frontend implementation forces all authentication types to use a single "value" field, making it impossible to correctly configure Basic Auth or other multi-field authentication methods without manual JSON hacking.

## 2. Frontend Implementation (`workflow_builder_ide.html`)

**Location:** `frontend/workflow_builder_ide.html` (approx. line 1840)

**Current Code:**
```html
<section>
    <label class="inspector-label">Authentication</label>
    <select v-model="selected.config.auth.type" class="v-input mb-2">
        <option value="none">None</option>
        <option value="api_key">API Key</option>
        <option value="bearer">Bearer Token</option>
        <option value="basic">Basic Auth</option> <!-- Problematic -->
    </select>
    
    <!-- SINGLE INPUT FIELD FOR ALL TYPES -->
    <input v-if="selected.config.auth.type !== 'none'" 
           v-model="selected.config.auth.value"
           :type="selected.config.auth.type === 'bearer' ? 'password' : 'text'"
           placeholder="Enter credentials..." class="v-input">
</section>
```

**Resulting JSON Data (Frontend State):**
When a user selects "Basic Auth" and enters a credential, the frontend generates:
```json
"auth": {
  "type": "basic",
  "value": "some_credential_string" 
}
```
*Note: It is unclear how the user is expected to enter username/password here (e.g., "user:pass"?).*

## 3. Backend Requirement (Oracle HCM Example)

The target backend connector configuration (as provided in the requirement) expects a structured `auth_config` object.

**Required JSON Structure:**
```json
"auth_config": {
    "type": "basic",
    "username": "KPMG_ABSENCE_BOT",
    "password": "welcome123"
}
```

## 4. Gap Analysis

| Feature | Frontend Implementation | Backend Requirement | Status |
| :--- | :--- | :--- | :--- |
| **Basic Auth** | Single input field (`value`) | Two fields (`username`, `password`) | **BROKEN** |
| **API Key** | Single input field (Value only) | Key Name + Value (often) | **Likely Insufficient** |
| **Bearer Token** | Single input field (`value`) | Single field (`token`) | **Compatible** |

**Critical Failure:**
The frontend does not provide a way to separate the username and password for Basic Auth. The backend connector will likely fail to authenticate because it looks for `username` and `password` keys in the config, which are missing.

## 5. Proposed Solution (Frontend Logic Update)

We need to update `workflow_builder_ide.html` to conditionally render different input fields based on the selected `auth.type`.

**New Logic:**

1.  **If `type === 'basic'`**:
    *   Show "Username" input (optional) -> binds to `selected.config.auth.username`
    *   Show "Password" input (optional) -> binds to `selected.config.auth.password`
    *   Hide generic `value` input.

2.  **If `type === 'api_key'`** (Enhancement):
    *   Show "Key Name" input (default: `X-API-Key`) -> binds to `selected.config.auth.key_name`
    *   Show "Value" input -> binds to `selected.config.auth.value`

3.  **If `type === 'bearer'`**:
    *   Show "Token" input -> binds to `selected.config.auth.token` (or `value` for backward compatibility).

**Mockup of Fix:**
```html
<section>
    <label class="inspector-label">Authentication</label>
    <select v-model="selected.config.auth.type" ...>...</select>

    <!-- Basic Auth -->
    <template v-if="selected.config.auth.type === 'basic'">
        <input v-model="selected.config.auth.username" placeholder="Username (optional)" class="v-input mb-1">
        <input v-model="selected.config.auth.password" type="password" placeholder="Password (optional)" class="v-input">
    </template>

    <!-- Bearer Token -->
    <template v-if="selected.config.auth.type === 'bearer'">
        <input v-model="selected.config.auth.token" type="password" placeholder="Bearer Token" class="v-input">
    </template>
    
    <!-- API Key -->
    <template v-if="selected.config.auth.type === 'api_key'">
        <input v-model="selected.config.auth.key_name" placeholder="Key Name (e.g. X-API-KEY)" class="v-input mb-1">
        <input v-model="selected.config.auth.value" type="password" placeholder="API Key Value" class="v-input">
    </template>
</section>
```
