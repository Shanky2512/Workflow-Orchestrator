## Here's a comprehensive breakdown of the user isolation problem across


## your codebase:

------------------------------------------------------------------------

## Root Cause: 3 Critical Design Flaws

Flaw 1: All users collapse to a single identity - AUTH_ENFORCEMENT
defaults to "optional", so unauthenticated requests get user_id =
"anonymous" - Every \_resolve_user_id() function (in
apps/appmgr/routes.py, apps/session/routes.py) maps "anonymous" to
DEFAULT_USER_ID = "00000000-0000-0000-0000-000000000001" - Result: Every
unauthenticated user is treated as the same person

Flaw 2: Routes hardcode DEFAULT_USER_ID instead of using actual auth
context - apps/workflow/routes.py (lines 665, 770, 873, 888, 1117, 1350)
--- all write user_id=DEFAULT_USER_ID - apps/agent/routes.py (line 136)
--- agent creation uses DEFAULT_USER_ID - Even authenticated users have
their data saved under the shared admin user

Flaw 3: Cache keys and filesystem paths have no user dimension - Cache
keys: session:{session_id} --- no user scoping - Filesystem:
workflows/draft/wf_xxx.json --- flat, no user directories - WebSocket:
/ws/execution/{run_id} --- no auth check, any user can eavesdrop

------------------------------------------------------------------------

## Where Isolation Breaks (By Component) – Compact Format

### 1. Appmgr (apps feature)
- Location: `apps/appmgr/routes.py`
- Problem: `_resolve_user_id()` collapses anonymous to shared ID

### 2. Chat History
- Location: `apps/appmgr/routes.py` (chat endpoints)
- Problem: Sessions tied to `DEFAULT_USER_ID`, so User B sees User A's chats

### 3. Agents
- Location: `apps/agent/routes.py`
- Problem: Creates/queries agents with `DEFAULT_USER_ID`

### 4. Workflows
- Location: `apps/workflow/routes.py`
- Problem: Saves workflows with `DEFAULT_USER_ID` at 6+ locations

### 5. Session Routes
- Location: `apps/session/routes.py`
- Problem: Same `_resolve_user_id()` → shared identity

### 6. Memcached
- Location: `echolib/repositories/session_repo.py`
- Problem: Cache key `session:{id}` has no user component

### 7. Filesystem
- Location: `apps/workflow/storage/filesystem.py`
- Problem: No user-level directory partitioning

### 8. WebSocket
- Location: `apps/gateway/main.py`
- Problem: No user validation on connect

### 9. Lookup APIs
- Location: `apps/appmgr/routes.py` (lookup routes)
- Problem: No auth dependency at all

------------------------------------------------------------------------

## What the Fix Must Cover

1.  Enforce authentication --- AUTH_ENFORCEMENT=required or at minimum
    generate unique per-session anonymous IDs
2.  Replace all DEFAULT_USER_ID hardcodes in workflow routes, agent
    routes, and appmgr routes with the actual user.user_id from the
    authenticated request
3.  Fix \_resolve_user_id() --- reject anonymous users or assign unique
    ephemeral IDs (never collapse to shared)
4.  Add user_id to cache keys --- session:{user_id}:{session_id} format
5.  Add user validation to WebSocket --- extract JWT from connection,
    validate ownership of run_id
6.  Ensure ALL database queries filter by user_id --- not just some
7.  Add auth to lookup endpoints --- prevent unauthenticated enumeration

------------------------------------------------------------------------

## For Appmgr specifically (the apps feature / chat history)

The apps/appmgr/routes.py file: - Creates applications, chat sessions,
messages, documents all under \_resolve_user_id() which returns the
shared DEFAULT_USER_ID - list_sessions(), get_history(), send_message()
--- all use this collapsed identity - Fix: Every endpoint must use the
real authenticated user_id and reject requests without valid auth

------------------------------------------------------------------------

In short: The system has the plumbing for multi-tenancy (user_id columns
exist in all tables, repositories accept user_id parameters) but the
routing layer bypasses all of it by hardcoding a shared user identity.
The fix is entirely in the routing/auth layer --- the database schema is
already correct.

Waiting for your command to proceed with implementation.

------------------------------------------------------------------------

## Authentication: Microsoft Entra ID (Azure AD) for kpmg

Users will authenticate via Microsoft company ID (kpmg Azure AD).
Login flow uses OAuth 2.0 / OIDC with Microsoft Entra ID.

### Login Flow

```
User clicks "Login" in EchoAI frontend
        |
        v
Browser redirects to Microsoft login
(login.microsoftonline.com)
        |
        v
User enters kpmg Microsoft credentials
(company_id@kpmg.com)
        |
        v
Microsoft returns an ID Token (JWT) to frontend
        |
        v
Frontend sends: Authorization: Bearer <microsoft_jwt>
        |
        v
EchoAI backend validates the Microsoft JWT
  - Verifies signature using Microsoft's public keys (JWKS)
  - Checks audience (app's client_id)
  - Checks issuer (kpmg tenant)
  - Extracts: oid (unique user ID), email, name
        |
        v
UserContext(user_id=oid_from_microsoft, email="user@kpmg.com")
```

### What Changes vs Current System

| Aspect | Current | With Microsoft Entra ID |
|--------|---------|------------------------|
| Token creation | security.py creates JWT with HS256 | Not needed -- Microsoft creates the JWT |
| Token validation | decode_token() uses local jwt_secret | Validates Microsoft JWT via JWKS public keys + RS256 |
| User identity | sub from self-signed token | oid (Object ID) from Microsoft token -- globally unique per user |
| Anonymous fallback | Falls back to shared DEFAULT_USER_ID | Rejected with 401 -- no anonymous access |

### Key Claims from Microsoft ID Token

- **oid** -- Object ID (unique per user in kpmg Azure AD) -- becomes user_id
- **preferred_username** -- e.g. shashank.singh@kpmg.com
- **name** -- display name
- **tid** -- Tenant ID (kpmg Azure AD tenant)

------------------------------------------------------------------------

## Exact Files to Modify (10 files, 1 new dependency)

### Layer 1: Authentication (Microsoft JWT validation)

| File | Change |
|------|--------|
| `echolib/security.py` | Rewrite decode_token() to validate Microsoft Entra ID JWTs using JWKS (RS256). Remove self-signed create_token(). Change get_current_user() to always require auth (no anonymous fallback). Extract oid as user_id, preferred_username as email. |
| `echolib/config.py` | Add Azure AD settings: AZURE_AD_TENANT_ID, AZURE_AD_CLIENT_ID, AZURE_AD_AUTHORITY. Remove auth_enforcement "optional" -- auth is now always required. |
| `echolib/types.py` | Extend UserContext to include display_name, tenant_id from Microsoft token. |

### Layer 2: Remove anonymous/DEFAULT_USER_ID collapse

| File | Change |
|------|--------|
| `apps/appmgr/routes.py` | Remove _resolve_user_id(). Replace all calls with user.user_id directly (now always a real Microsoft oid). Fixes: app CRUD, chat sessions, chat history, document uploads, skills listing. |
| `apps/session/routes.py` | Same -- remove _resolve_user_id(), use authenticated user.user_id directly. |
| `apps/workflow/routes.py` | Replace all ~6 locations where DEFAULT_USER_ID is hardcoded with user.user_id from request context. |
| `apps/agent/routes.py` | Replace DEFAULT_USER_ID with actual user from auth context. |
| `echolib/repositories/base.py` | Remove DEFAULT_USER_ID constant (no longer needed). |

### Layer 3: Cache key isolation

| File | Change |
|------|--------|
| `echolib/repositories/session_repo.py` | Change cache key from session:{session_id} to session:{user_id}:{session_id}. Ensures User A's cached session is never served to User B. |

### Layer 4: WebSocket auth

| File | Change |
|------|--------|
| `apps/gateway/main.py` | Add JWT validation to WebSocket /ws/execution/{run_id}. Extract user from query param token, validate ownership of run_id. |

### Layer 5: Gateway enforcement

| File | Change |
|------|--------|
| `apps/gateway/main.py` | Update AuthMiddleware to reject requests without valid Microsoft JWT (except /healthz, /docs). Switch all routes from get_current_user (optional) to require_user (mandatory). |

### Layer 6: Lookup endpoints

| File | Change |
|------|--------|
| `apps/appmgr/routes.py` (lookup_router) | Add Depends(require_user) to all lookup endpoints (/api/llms, /api/personas, /api/tags, etc.). No unauthenticated access. |

### New Dependency

| Package | Purpose |
|---------|---------|
| PyJWT[crypto] | Fetches Microsoft JWKS public keys to verify RS256 JWTs (replaces plain PyJWT) |

### Files Summary

```
MODIFY:
  1. echolib/security.py              <- Microsoft JWT validation (core change)
  2. echolib/config.py                <- Azure AD settings
  3. echolib/types.py                 <- Extended UserContext
  4. echolib/repositories/base.py     <- Remove DEFAULT_USER_ID
  5. echolib/repositories/session_repo.py <- User-scoped cache keys
  6. apps/appmgr/routes.py            <- Remove anonymous collapse, enforce auth
  7. apps/session/routes.py           <- Remove anonymous collapse
  8. apps/workflow/routes.py          <- Replace hardcoded DEFAULT_USER_ID
  9. apps/agent/routes.py             <- Replace hardcoded DEFAULT_USER_ID
  10. apps/gateway/main.py            <- Enforce auth on all routes + WebSocket

NEW DEPENDENCY:
  PyJWT[crypto] (replaces plain PyJWT -- adds RS256/JWKS support)
```

---

