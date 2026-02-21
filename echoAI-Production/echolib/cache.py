"""
EchoAI Memcached Cache Client

Provides async caching layer for session data with configurable fallback to database.
Uses python-memcached with asyncio.run_in_executor() for async compatibility.

Usage:
    from echolib.cache import cache_client

    # Get cached value
    data = await cache_client.get("session:123")

    # Set cached value with TTL
    await cache_client.set("session:123", {"user_id": "abc"}, ttl=1800)

Configuration (via environment or .env):
    MEMCACHED_ENABLED=true/false
    MEMCACHED_HOSTS=localhost:11211
    MEMCACHED_TTL=1800
    MEMCACHED_FALLBACK=true
    MEMCACHED_POOL_SIZE=10
    MEMCACHED_TIMEOUT=5
"""

import asyncio
import json
import logging
from typing import Optional, Any, Dict
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor

from .config import settings

logger = logging.getLogger(__name__)

# Import memcache conditionally to allow running without it installed
try:
    import memcache
    MEMCACHE_AVAILABLE = True
except ImportError:
    MEMCACHE_AVAILABLE = False
    logger.warning("python-memcached not installed - Memcached caching disabled")

# Thread pool for running synchronous memcached operations
_executor: Optional[ThreadPoolExecutor] = None


def _get_executor() -> ThreadPoolExecutor:
    """Get or create the thread pool executor for memcached operations."""
    global _executor
    if _executor is None:
        # Use a small pool since memcached operations are fast
        _executor = ThreadPoolExecutor(max_workers=settings.memcached_pool_size)
    return _executor


class CacheClient:
    """
    Async Memcached client with fallback support.

    Provides a simple interface for caching session data with:
    - Configurable enable/disable via settings
    - Automatic fallback behavior on cache failures
    - JSON serialization/deserialization
    - TTL-based expiration
    - Async compatibility via run_in_executor()

    Attributes:
        _client: The underlying memcache.Client instance.
        _enabled: Whether caching is enabled in configuration.
        _fallback: Whether to silently fail on cache errors.
        _ttl: Default TTL in seconds for cached items.
    """

    def __init__(self):
        """Initialize cache client with settings from config."""
        self._client: Optional[Any] = None  # memcache.Client when connected
        self._enabled = settings.memcached_enabled
        self._fallback = settings.memcached_fallback
        self._ttl = settings.memcached_ttl

    async def connect(self) -> None:
        """
        Initialize Memcached connection.

        Establishes connection to Memcached server(s) specified in settings.
        If connection fails and fallback is enabled, logs warning and continues.
        If connection fails and fallback is disabled, raises the exception.
        """
        if not self._enabled:
            logger.info("Memcached disabled by configuration (MEMCACHED_ENABLED=false)")
            return

        if not MEMCACHE_AVAILABLE:
            logger.warning("python-memcached package not available - caching disabled")
            self._enabled = False
            return

        try:
            hosts = settings.memcached_host_list
            if not hosts:
                logger.warning("No Memcached hosts configured")
                return

            # python-memcached expects servers as "host:port" strings
            server_list = [f"{host}:{port}" for host, port in hosts]

            # Create the synchronous memcache client
            self._client = memcache.Client(
                servers=server_list,
                debug=0,
                socket_timeout=settings.memcached_timeout
            )

            # Test connection by getting stats (run in executor)
            loop = asyncio.get_event_loop()
            stats = await loop.run_in_executor(
                _get_executor(),
                self._client.get_stats
            )

            if stats:
                # Extract some useful stats for logging
                # python-memcached returns strings, not bytes
                for server, server_stats in stats:
                    # Handle both string and bytes keys (varies by python-memcached version)
                    def get_stat(key: str) -> str:
                        # Try string key first, then bytes key
                        val = server_stats.get(key) or server_stats.get(key.encode(), '')
                        if isinstance(val, bytes):
                            return val.decode()
                        return str(val) if val else '0'

                    version = get_stat('version') or 'unknown'
                    curr_items = get_stat('curr_items') or '0'
                    max_bytes_str = get_stat('limit_maxbytes') or '0'
                    max_bytes = int(max_bytes_str) // (1024 * 1024) if max_bytes_str.isdigit() else 0

                    # Server name might be string or bytes
                    server_name = server.decode() if isinstance(server, bytes) else str(server)
                    logger.info(f"✅ Memcached connected: {server_name} (version={version}, items={curr_items}, max={max_bytes}MB)")
                logger.info(f"🚀 Cache layer ENABLED - TTL={self._ttl}s, Fallback={self._fallback}")
            else:
                logger.warning("⚠️ Memcached connection test returned no stats")
                if not self._fallback:
                    raise ConnectionError("Memcached connection test failed")
                self._client = None

        except Exception as e:
            logger.warning(f"Memcached connection failed: {e}")
            if not self._fallback:
                raise
            self._client = None

    async def disconnect(self) -> None:
        """
        Close Memcached connections.

        Gracefully closes all connections.
        """
        global _executor
        if self._client is not None:
            try:
                # python-memcached uses disconnect_all() to close connections
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    _get_executor(),
                    self._client.disconnect_all
                )
                logger.info("Memcached connections closed")
            except Exception as e:
                logger.warning(f"Error closing Memcached connections: {e}")
            finally:
                self._client = None

        # Shutdown the executor
        if _executor is not None:
            _executor.shutdown(wait=False)
            _executor = None

    @property
    def is_available(self) -> bool:
        """
        Check if cache is available for use.

        Returns:
            bool: True if caching is enabled and client is connected.
        """
        return self._enabled and self._client is not None

    def _sync_get(self, key: str) -> Optional[str]:
        """Synchronous get operation."""
        return self._client.get(key)

    def _sync_set(self, key: str, value: str, ttl: int) -> bool:
        """Synchronous set operation."""
        return self._client.set(key, value, time=ttl)

    def _sync_delete(self, key: str) -> bool:
        """Synchronous delete operation."""
        return self._client.delete(key)

    def _sync_touch(self, key: str, ttl: int) -> bool:
        """Synchronous touch operation."""
        return self._client.touch(key, ttl)

    def _sync_flush_all(self) -> bool:
        """Synchronous flush_all operation."""
        return self._client.flush_all()

    async def get(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Get value from cache.

        Args:
            key: Cache key to retrieve.

        Returns:
            Optional[dict]: Cached value as dict, or None if not found/error.
        """
        if not self.is_available:
            return None

        try:
            loop = asyncio.get_event_loop()
            value = await loop.run_in_executor(
                _get_executor(),
                self._sync_get,
                key
            )
            if value:
                logger.debug(f"🎯 Cache HIT: {key}")
                return json.loads(value)
            logger.debug(f"❌ Cache MISS: {key}")
            return None
        except Exception as e:
            logger.warning(f"⚠️ Cache get failed for key '{key}': {e}")
            return None

    async def set(
        self,
        key: str,
        value: Dict[str, Any],
        ttl: Optional[int] = None
    ) -> bool:
        """
        Set value in cache with TTL.

        Args:
            key: Cache key to set.
            value: Dict value to cache (will be JSON serialized).
            ttl: Time-to-live in seconds. Uses default TTL if not specified.

        Returns:
            bool: True if successfully cached, False otherwise.
        """
        if not self.is_available:
            return False

        try:
            effective_ttl = ttl if ttl is not None else self._ttl
            serialized = json.dumps(value, default=str)

            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                _get_executor(),
                self._sync_set,
                key,
                serialized,
                effective_ttl
            )
            if result:
                logger.debug(f"💾 Cache SET: {key} (TTL={effective_ttl}s, size={len(serialized)} bytes)")
            return bool(result)
        except Exception as e:
            logger.warning(f"⚠️ Cache set failed for key '{key}': {e}")
            return False

    async def delete(self, key: str) -> bool:
        """
        Delete value from cache.

        Args:
            key: Cache key to delete.

        Returns:
            bool: True if deletion was attempted (regardless of key existence).
        """
        if not self.is_available:
            return False

        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                _get_executor(),
                self._sync_delete,
                key
            )
            if result:
                logger.debug(f"🗑️ Cache DELETE: {key}")
            return bool(result)
        except Exception as e:
            logger.warning(f"⚠️ Cache delete failed for key '{key}': {e}")
            return False

    async def touch(self, key: str, ttl: Optional[int] = None) -> bool:
        """
        Refresh TTL for a key without retrieving value.

        Args:
            key: Cache key to touch.
            ttl: New TTL in seconds. Uses default TTL if not specified.

        Returns:
            bool: True if touch was successful, False otherwise.
        """
        if not self.is_available:
            return False

        try:
            effective_ttl = ttl if ttl is not None else self._ttl

            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                _get_executor(),
                self._sync_touch,
                key,
                effective_ttl
            )
            return bool(result)
        except Exception as e:
            logger.warning(f"Cache touch failed for key '{key}': {e}")
            return False

    async def flush_all(self) -> bool:
        """
        Flush all keys from cache.

        Warning: Use with caution - removes ALL cached data.

        Returns:
            bool: True if flush was successful.
        """
        if not self.is_available:
            return False

        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                _get_executor(),
                self._sync_flush_all
            )
            logger.info("Cache flushed")
            return bool(result)
        except Exception as e:
            logger.warning(f"Cache flush_all failed: {e}")
            return False


# Singleton instance
cache_client = CacheClient()


@asynccontextmanager
async def cache_lifespan():
    """
    FastAPI lifespan context manager for cache.

    Usage in FastAPI:
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            async with cache_lifespan():
                yield

        app = FastAPI(lifespan=lifespan)

    Yields:
        None
    """
    await cache_client.connect()
    try:
        yield
    finally:
        await cache_client.disconnect()
