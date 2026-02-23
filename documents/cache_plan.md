# EchoAI Memcached Session Cache Plan

## Document Information
- **Created**: 2026-02-02
- **Status**: Planning Phase
- **Author**: Claude Code Analysis
- **Related DB Plan**: `documents/db_plan.md`
- **Related Progress**: `documents/cache_progress.md`

---

## 1. EXECUTIVE SUMMARY

### 1.1 Objective
Implement a **configurable** Memcached caching layer for chat sessions to provide:
- Fast LLM context retrieval (< 50ms for cache hits)
- Reduced PostgreSQL load for frequently accessed sessions
- Graceful fallback to PostgreSQL when cache unavailable
- Toggle-able via environment variable for dev vs prod

### 1.2 Why Memcached for Sessions?

| Use Case | Without Cache | With Cache |
|----------|---------------|------------|
| Get session for LLM context | ~100-200ms (DB query + JSONB parse) | ~5-20ms (memory read) |
| Send chat message | DB write only | DB write + cache update (parallel) |
| Resume inactive session | Same as above | Cache miss → DB → populate cache |
| Session listing | DB query | DB query (not cached) |

**Key Insight**: Chat sessions with embedded messages (~50-200 messages, ~200KB) are read frequently during active conversations but written less often. This is an ideal caching pattern.

### 1.3 Design Principles

1. **Configurable**: Disabled by default for development simplicity
2. **Fallback**: Application works without Memcached (PostgreSQL-only mode)
3. **Write-Through**: Writes go to both cache and database
4. **TTL-Based Eviction**: Inactive sessions expire from cache
5. **Session-Only**: Only cache session data (not workflows, agents, tools)

---

## 2. ARCHITECTURE

### 2.1 High-Level Flow

```
                    ┌─────────────────────────────────────────────────────┐
                    │                    Application                       │
                    └─────────────────────────┬───────────────────────────┘
                                              │
                    ┌─────────────────────────▼───────────────────────────┐
                    │                   SessionRepository                  │
                    │  ┌─────────────────────────────────────────────────┐│
                    │  │ if MEMCACHED_ENABLED:                           ││
                    │  │   try: cache_client.get/set                     ││
                    │  │   except: fallback to database                  ││
                    │  │ else:                                           ││
                    │  │   database only                                 ││
                    │  └─────────────────────────────────────────────────┘│
                    └─────────────────────────┬───────────────────────────┘
                                              │
                    ┌─────────────┬───────────┴───────────┬───────────────┐
                    │             │                       │               │
                    ▼             ▼                       ▼               ▼
            ┌───────────┐  ┌───────────┐          ┌───────────┐   ┌───────────┐
            │ Memcached │  │ Memcached │    ...   │ Memcached │   │PostgreSQL │
            │  Node 1   │  │  Node 2   │          │  Node N   │   │ (primary) │
            └───────────┘  └───────────┘          └───────────┘   └───────────┘
                    │             │                       │               │
                    └─────────────┴───────────────────────┴───────────────┘
                                              │
                                      TTL: 30 minutes
                                   Eviction: LRU
```

### 2.2 Caching Strategy: Write-Through + Read-Through

```
WRITE PATH (send message):
┌─────────┐                ┌──────────────┐     ┌────────────┐
│ Client  │──▶ Add message │SessionRepo   │────▶│ PostgreSQL │
└─────────┘    to session  └───────┬──────┘     └────────────┘
                                   │
                                   │ parallel
                                   ▼
                           ┌────────────┐
                           │ Memcached  │
                           └────────────┘

READ PATH (get session for LLM):
┌─────────┐                ┌──────────────┐     ┌────────────┐
│ Client  │──▶ Get session │SessionRepo   │────▶│ Memcached  │──▶ HIT: Return
└─────────┘                └───────┬──────┘     └─────┬──────┘
                                   │                  │ MISS
                                   │                  ▼
                                   │          ┌────────────┐
                                   │◀─────────│ PostgreSQL │──▶ Populate cache
                                   │          └────────────┘
                                   ▼
                             Return to client
```

### 2.3 Cache Key Format

```
session:{session_id}
```

**Examples**:
- `session:550e8400-e29b-41d4-a716-446655440000`
- `session:c9bf9e57-1685-4c89-bafb-ff5af830be8a`

### 2.4 Cached Data Structure

The cached value is the **entire session object** serialized as JSON:

```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "user_id": "user_123",
  "title": "RFP Analysis Chat",
  "context_type": "workflow",
  "context_id": "wf_xxx",
  "workflow_mode": "draft",
  "messages": [
    {"id": "msg_1", "role": "user", "content": "...", "timestamp": "..."},
    {"id": "msg_2", "role": "assistant", "content": "...", "timestamp": "..."}
  ],
  "selected_tool_ids": ["tool_1", "tool_2"],
  "variables": {"key": "value"},
  "last_activity": "2026-02-02T16:53:08.084505Z"
}
```

**Size Estimate**:
- 50 messages × 500 chars avg = ~25KB
- 200 messages × 500 chars avg = ~100KB
- With metadata overhead: ~120KB max per session

---

## 3. CONFIGURATION

### 3.1 Environment Variables

```env
# Toggle Memcached on/off (default: false for dev)
MEMCACHED_ENABLED=false

# Comma-separated list of Memcached servers
MEMCACHED_HOSTS=localhost:11211

# Session TTL in seconds (default: 30 minutes)
MEMCACHED_TTL=1800

# Fallback to PostgreSQL on cache failure (default: true)
MEMCACHED_FALLBACK=true

# Connection pool size
MEMCACHED_POOL_SIZE=10

# Connection timeout in seconds
MEMCACHED_TIMEOUT=5
```

### 3.2 Config Model

```python
# echolib/config.py

class Settings(BaseSettings):
    # ... existing settings ...

    # Memcached Configuration
    memcached_enabled: bool = False
    memcached_hosts: str = "localhost:11211"
    memcached_ttl: int = 1800  # 30 minutes
    memcached_fallback: bool = True
    memcached_pool_size: int = 10
    memcached_timeout: int = 5

    @property
    def memcached_host_list(self) -> List[Tuple[str, int]]:
        """Parse hosts string into list of (host, port) tuples."""
        hosts = []
        for host_str in self.memcached_hosts.split(","):
            parts = host_str.strip().split(":")
            host = parts[0]
            port = int(parts[1]) if len(parts) > 1 else 11211
            hosts.append((host, port))
        return hosts
```

---

## 4. IMPLEMENTATION

### 4.1 Cache Client Module

```python
# echolib/cache.py

"""
Memcached Cache Client for EchoAI
Provides session caching with configurable fallback.
"""

import json
import logging
from typing import Optional, Any
from contextlib import asynccontextmanager

import aiomcache
from .config import settings

logger = logging.getLogger(__name__)


class CacheClient:
    """Async Memcached client with fallback support."""

    def __init__(self):
        self._client: Optional[aiomcache.Client] = None
        self._enabled = settings.memcached_enabled
        self._fallback = settings.memcached_fallback
        self._ttl = settings.memcached_ttl

    async def connect(self) -> None:
        """Initialize Memcached connection pool."""
        if not self._enabled:
            logger.info("Memcached disabled by configuration")
            return

        try:
            hosts = settings.memcached_host_list
            # aiomcache uses first host; for multi-node, use consistent hashing wrapper
            host, port = hosts[0]
            self._client = aiomcache.Client(
                host=host,
                port=port,
                pool_size=settings.memcached_pool_size,
                conn_timeout=settings.memcached_timeout
            )
            # Test connection
            await self._client.version()
            logger.info(f"Memcached connected: {host}:{port}")
        except Exception as e:
            logger.warning(f"Memcached connection failed: {e}")
            if not self._fallback:
                raise
            self._client = None

    async def disconnect(self) -> None:
        """Close Memcached connections."""
        if self._client:
            await self._client.close()
            self._client = None

    @property
    def is_available(self) -> bool:
        """Check if cache is available."""
        return self._enabled and self._client is not None

    async def get(self, key: str) -> Optional[dict]:
        """Get value from cache."""
        if not self.is_available:
            return None

        try:
            value = await self._client.get(key.encode())
            if value:
                return json.loads(value.decode())
            return None
        except Exception as e:
            logger.warning(f"Cache get failed for {key}: {e}")
            return None

    async def set(self, key: str, value: dict, ttl: Optional[int] = None) -> bool:
        """Set value in cache."""
        if not self.is_available:
            return False

        try:
            ttl = ttl or self._ttl
            serialized = json.dumps(value).encode()
            await self._client.set(key.encode(), serialized, exptime=ttl)
            return True
        except Exception as e:
            logger.warning(f"Cache set failed for {key}: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """Delete value from cache."""
        if not self.is_available:
            return False

        try:
            await self._client.delete(key.encode())
            return True
        except Exception as e:
            logger.warning(f"Cache delete failed for {key}: {e}")
            return False

    async def touch(self, key: str, ttl: Optional[int] = None) -> bool:
        """Refresh TTL for a key."""
        if not self.is_available:
            return False

        try:
            ttl = ttl or self._ttl
            await self._client.touch(key.encode(), exptime=ttl)
            return True
        except Exception as e:
            logger.warning(f"Cache touch failed for {key}: {e}")
            return False


# Singleton instance
cache_client = CacheClient()


@asynccontextmanager
async def cache_lifespan():
    """Lifespan context manager for FastAPI."""
    await cache_client.connect()
    yield
    await cache_client.disconnect()
```

### 4.2 Session Repository with Caching

```python
# echolib/repositories/session_repo.py

"""
Session Repository with Memcached Caching
"""

from typing import Optional, List
from uuid import UUID
import json

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete
from sqlalchemy.dialects.postgresql import insert

from ..models.session import ChatSession
from ..cache import cache_client
from .base import BaseRepository


class SessionRepository(BaseRepository):
    """Session data access with caching layer."""

    CACHE_PREFIX = "session:"

    def _cache_key(self, session_id: str) -> str:
        return f"{self.CACHE_PREFIX}{session_id}"

    async def get_by_id(
        self,
        session_id: str,
        user_id: str,
        db: AsyncSession
    ) -> Optional[dict]:
        """
        Get session by ID with caching.
        Read-through: check cache first, then DB.
        """
        cache_key = self._cache_key(session_id)

        # 1. Try cache first
        cached = await cache_client.get(cache_key)
        if cached:
            # Verify user ownership
            if cached.get("user_id") == user_id:
                return cached
            # Wrong user - don't return cached data
            return None

        # 2. Cache miss - query database
        stmt = select(ChatSession).where(
            ChatSession.session_id == UUID(session_id),
            ChatSession.user_id == UUID(user_id),
            ChatSession.is_deleted == False
        )
        result = await db.execute(stmt)
        session = result.scalar_one_or_none()

        if not session:
            return None

        # 3. Convert to dict
        session_dict = session.to_dict()

        # 4. Populate cache
        await cache_client.set(cache_key, session_dict)

        return session_dict

    async def create(
        self,
        user_id: str,
        session_data: dict,
        db: AsyncSession
    ) -> dict:
        """Create new session."""
        session = ChatSession(
            user_id=UUID(user_id),
            title=session_data.get("title", "New Chat"),
            context_type=session_data.get("context_type", "general"),
            context_id=UUID(session_data["context_id"]) if session_data.get("context_id") else None,
            workflow_mode=session_data.get("workflow_mode"),
            messages=session_data.get("messages", []),
            selected_tool_ids=session_data.get("selected_tool_ids", []),
            variables=session_data.get("variables", {}),
            state_schema=session_data.get("state_schema", {})
        )

        db.add(session)
        await db.commit()
        await db.refresh(session)

        session_dict = session.to_dict()

        # Populate cache
        cache_key = self._cache_key(str(session.session_id))
        await cache_client.set(cache_key, session_dict)

        return session_dict

    async def add_message(
        self,
        session_id: str,
        user_id: str,
        message: dict,
        db: AsyncSession
    ) -> Optional[dict]:
        """
        Add message to session.
        Write-through: update both DB and cache.
        """
        # Get current session
        session_dict = await self.get_by_id(session_id, user_id, db)
        if not session_dict:
            return None

        # Append message
        messages = session_dict.get("messages", [])
        messages.append(message)

        # Update database
        stmt = (
            update(ChatSession)
            .where(
                ChatSession.session_id == UUID(session_id),
                ChatSession.user_id == UUID(user_id)
            )
            .values(messages=messages)
        )
        await db.execute(stmt)
        await db.commit()

        # Update cache
        session_dict["messages"] = messages
        cache_key = self._cache_key(session_id)
        await cache_client.set(cache_key, session_dict)

        return session_dict

    async def update(
        self,
        session_id: str,
        user_id: str,
        updates: dict,
        db: AsyncSession
    ) -> Optional[dict]:
        """Update session with cache invalidation."""
        stmt = (
            update(ChatSession)
            .where(
                ChatSession.session_id == UUID(session_id),
                ChatSession.user_id == UUID(user_id)
            )
            .values(**updates)
            .returning(ChatSession)
        )
        result = await db.execute(stmt)
        session = result.scalar_one_or_none()

        if not session:
            return None

        await db.commit()
        await db.refresh(session)

        session_dict = session.to_dict()

        # Update cache
        cache_key = self._cache_key(session_id)
        await cache_client.set(cache_key, session_dict)

        return session_dict

    async def delete(
        self,
        session_id: str,
        user_id: str,
        db: AsyncSession
    ) -> bool:
        """Soft delete session with cache invalidation."""
        stmt = (
            update(ChatSession)
            .where(
                ChatSession.session_id == UUID(session_id),
                ChatSession.user_id == UUID(user_id)
            )
            .values(is_deleted=True)
        )
        result = await db.execute(stmt)
        await db.commit()

        # Invalidate cache
        cache_key = self._cache_key(session_id)
        await cache_client.delete(cache_key)

        return result.rowcount > 0

    async def list_by_user(
        self,
        user_id: str,
        db: AsyncSession,
        limit: int = 50,
        offset: int = 0
    ) -> List[dict]:
        """
        List user's sessions.
        Note: List queries hit DB directly (not cached).
        """
        stmt = (
            select(ChatSession)
            .where(
                ChatSession.user_id == UUID(user_id),
                ChatSession.is_deleted == False
            )
            .order_by(ChatSession.last_activity.desc())
            .limit(limit)
            .offset(offset)
        )
        result = await db.execute(stmt)
        sessions = result.scalars().all()

        return [s.to_dict() for s in sessions]

    async def refresh_ttl(self, session_id: str) -> bool:
        """Refresh session TTL in cache (called on activity)."""
        cache_key = self._cache_key(session_id)
        return await cache_client.touch(cache_key)


# Singleton instance
session_repository = SessionRepository()
```

### 4.3 FastAPI Lifespan Integration

```python
# apps/gateway/main.py

from contextlib import asynccontextmanager
from fastapi import FastAPI
from echolib.database import database_lifespan
from echolib.cache import cache_lifespan

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Combined lifespan for database and cache."""
    async with database_lifespan():
        async with cache_lifespan():
            yield

app = FastAPI(
    title="EchoAI Gateway",
    lifespan=lifespan
)
```

---

## 5. DOCKER COMPOSE

```yaml
version: '3.8'

services:
  postgres:
    image: postgres:16-alpine
    container_name: echoai-postgres
    environment:
      POSTGRES_USER: echoai
      POSTGRES_PASSWORD: echoai_dev
      POSTGRES_DB: echoai
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U echoai"]
      interval: 5s
      timeout: 5s
      retries: 5

  memcached:
    image: memcached:1.6-alpine
    container_name: echoai-memcached
    ports:
      - "11211:11211"
    # Allocate 256MB memory, max item size 1MB
    command: memcached -m 256 -I 1m
    healthcheck:
      test: ["CMD", "nc", "-z", "localhost", "11211"]
      interval: 5s
      timeout: 5s
      retries: 5

volumes:
  pgdata:
```

---

## 6. TESTING STRATEGY

### 6.1 Unit Tests

```python
# tests/test_cache.py

import pytest
from unittest.mock import AsyncMock, patch
from echolib.cache import CacheClient

@pytest.fixture
def mock_client():
    return AsyncMock()

class TestCacheClient:
    async def test_get_cache_hit(self, mock_client):
        """Test successful cache hit."""
        mock_client.get.return_value = b'{"session_id": "123"}'
        client = CacheClient()
        client._client = mock_client
        client._enabled = True

        result = await client.get("session:123")

        assert result == {"session_id": "123"}
        mock_client.get.assert_called_once()

    async def test_get_cache_miss(self, mock_client):
        """Test cache miss returns None."""
        mock_client.get.return_value = None
        client = CacheClient()
        client._client = mock_client
        client._enabled = True

        result = await client.get("session:123")

        assert result is None

    async def test_get_disabled(self):
        """Test disabled cache returns None."""
        client = CacheClient()
        client._enabled = False

        result = await client.get("session:123")

        assert result is None

    async def test_set_with_ttl(self, mock_client):
        """Test set with custom TTL."""
        client = CacheClient()
        client._client = mock_client
        client._enabled = True

        await client.set("session:123", {"data": "value"}, ttl=600)

        mock_client.set.assert_called_once()
        call_args = mock_client.set.call_args
        assert call_args[1]["exptime"] == 600

    async def test_fallback_on_error(self, mock_client):
        """Test fallback behavior on error."""
        mock_client.get.side_effect = Exception("Connection failed")
        client = CacheClient()
        client._client = mock_client
        client._enabled = True
        client._fallback = True

        result = await client.get("session:123")

        assert result is None  # Fallback returns None, not exception
```

### 6.2 Integration Tests

```python
# tests/test_session_cache_integration.py

import pytest
from httpx import AsyncClient
from echolib.cache import cache_client

@pytest.mark.integration
class TestSessionCacheIntegration:
    async def test_session_cached_on_create(self, client: AsyncClient, auth_headers):
        """Test that new session is cached."""
        response = await client.post(
            "/sessions",
            json={"title": "Test Session"},
            headers=auth_headers
        )
        session_id = response.json()["session_id"]

        # Verify in cache
        cached = await cache_client.get(f"session:{session_id}")
        assert cached is not None
        assert cached["title"] == "Test Session"

    async def test_session_served_from_cache(self, client: AsyncClient, auth_headers):
        """Test that repeated gets hit cache."""
        # Create session
        response = await client.post("/sessions", json={}, headers=auth_headers)
        session_id = response.json()["session_id"]

        # First get (should populate cache)
        await client.get(f"/sessions/{session_id}", headers=auth_headers)

        # Clear DB session to prove cache is used
        # (In real test, mock DB to verify no call)

        # Second get (should hit cache)
        response = await client.get(f"/sessions/{session_id}", headers=auth_headers)
        assert response.status_code == 200

    async def test_fallback_when_cache_down(self, client: AsyncClient, auth_headers):
        """Test PostgreSQL fallback when Memcached unavailable."""
        # Disconnect cache
        await cache_client.disconnect()

        # Create session (should still work via DB)
        response = await client.post(
            "/sessions",
            json={"title": "Fallback Test"},
            headers=auth_headers
        )
        assert response.status_code == 200

        # Reconnect for cleanup
        await cache_client.connect()
```

### 6.3 Manual Testing Commands

```bash
# Start services
docker-compose up -d

# Verify Memcached is running
echo "stats" | nc localhost 11211

# Test with cache enabled
MEMCACHED_ENABLED=true uvicorn apps.gateway.main:app --reload

# Test with cache disabled
MEMCACHED_ENABLED=false uvicorn apps.gateway.main:app --reload

# Monitor Memcached stats
watch -n 1 'echo "stats" | nc localhost 11211 | grep -E "curr_items|get_hits|get_misses"'
```

---

## 7. MONITORING & OBSERVABILITY

### 7.1 Metrics to Track

| Metric | Description | Alert Threshold |
|--------|-------------|-----------------|
| `cache_hit_rate` | % of gets served from cache | < 80% |
| `cache_miss_rate` | % of gets requiring DB | > 20% |
| `cache_latency_p99` | 99th percentile latency | > 50ms |
| `cache_errors` | Connection/operation failures | > 10/min |
| `cache_memory_used` | Memory consumption | > 80% of allocated |
| `cache_evictions` | Items evicted by LRU | Sudden spike |

### 7.2 Logging

```python
# Cache operations logging (already in cache.py)
logger.info(f"Memcached connected: {host}:{port}")
logger.warning(f"Memcached connection failed: {e}")
logger.warning(f"Cache get failed for {key}: {e}")
logger.warning(f"Cache set failed for {key}: {e}")
```

---

## 8. EDGE CASES & FAILURE MODES

### 8.1 Handled Scenarios

| Scenario | Behavior |
|----------|----------|
| Memcached down at startup | Log warning, continue without cache |
| Memcached goes down mid-operation | Fallback to PostgreSQL, log warning |
| Cache miss | Read from PostgreSQL, populate cache |
| Cache and DB diverge | DB is source of truth; cache refreshed on read |
| Session deleted while cached | Cache invalidated on delete |
| TTL expires | Automatic eviction, next read populates |
| Memory pressure (LRU eviction) | Least-recently-used items evicted |

### 8.2 Not Handled (Out of Scope)

| Scenario | Reason |
|----------|--------|
| Multi-node consistent hashing | Single-node sufficient for MVP |
| Cache warming on startup | Sessions loaded on-demand |
| Distributed locking | Not needed for session data |

---

## 9. DEPENDENCIES

### 9.1 Python Packages

```
# requirements.txt additions
aiomcache>=0.8.1    # Async Memcached client
```

### 9.2 Docker Images

```
memcached:1.6-alpine    # Lightweight Memcached image (~5MB)
```

---

## 10. IMPLEMENTATION PHASES

| Phase | Tasks | Dependencies |
|-------|-------|--------------|
| 1 | Create `echolib/cache.py` | None |
| 2 | Update `echolib/config.py` with Memcached settings | Phase 1 |
| 3 | Update `docker-compose.yml` with Memcached | None |
| 4 | Create `SessionRepository` with caching | Phase 1, 2 |
| 5 | Update `apps/gateway/main.py` lifespan | Phase 1, 4 |
| 6 | Write unit tests | Phase 1-5 |
| 7 | Write integration tests | Phase 1-6 |
| 8 | Manual testing | Phase 1-7 |

---

## 11. SUCCESS CRITERIA

| Criteria | Target |
|----------|--------|
| Cache hit rate | > 90% for active sessions |
| Cache latency | < 20ms p99 |
| Fallback works | App functions when Memcached down |
| No data loss | PostgreSQL always written |
| Toggle works | MEMCACHED_ENABLED=false skips cache |

---

## APPENDIX A: Quick Reference

### Enable Memcached
```bash
# .env
MEMCACHED_ENABLED=true
MEMCACHED_HOSTS=localhost:11211
MEMCACHED_TTL=1800
```

### Disable Memcached
```bash
# .env
MEMCACHED_ENABLED=false
```

### Flush Cache (Development)
```bash
echo "flush_all" | nc localhost 11211
```

### Check Cache Stats
```bash
echo "stats" | nc localhost 11211
```
