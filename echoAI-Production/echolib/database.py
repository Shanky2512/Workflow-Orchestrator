"""
EchoAI Database Module

Async SQLAlchemy engine and session management for PostgreSQL.
Uses SQLAlchemy 2.0 patterns with asyncpg driver.

Usage:
    from echolib.database import get_db_session, init_db

    async with get_db_session() as session:
        result = await session.execute(select(User))
        users = result.scalars().all()
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator
import logging

from sqlalchemy.ext.asyncio import (
    create_async_engine,
    AsyncSession,
    async_sessionmaker,
    AsyncEngine
)
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.ext.asyncio import AsyncAttrs

from .config import settings

logger = logging.getLogger(__name__)


class Base(AsyncAttrs, DeclarativeBase):
    """
    Base class for all SQLAlchemy models.

    Uses AsyncAttrs mixin for lazy loading support in async context.
    All models should inherit from this class.
    """
    pass


# Engine instance (initialized lazily)
_engine: AsyncEngine | None = None


def get_engine() -> AsyncEngine:
    """
    Get or create the async database engine.

    Returns:
        AsyncEngine: The SQLAlchemy async engine instance.
    """
    global _engine
    if _engine is None:
        _engine = create_async_engine(
            settings.database_url,
            pool_size=settings.database_pool_size,
            max_overflow=settings.database_max_overflow,
            echo=False,  # Set to True for SQL debugging
            pool_pre_ping=True,  # Verify connections before use
        )
        logger.info(f"Database engine created for: {_mask_connection_string(settings.database_url)}")
    return _engine


def _mask_connection_string(url: str) -> str:
    """Mask password in connection string for logging."""
    try:
        if "@" in url and ":" in url:
            # postgresql+asyncpg://user:password@host:port/db
            prefix, rest = url.split("://", 1)
            if "@" in rest:
                creds, host = rest.split("@", 1)
                if ":" in creds:
                    user, _ = creds.split(":", 1)
                    return f"{prefix}://{user}:****@{host}"
        return url
    except Exception:
        return "****"


# Session factory (initialized lazily)
_async_session_maker: async_sessionmaker[AsyncSession] | None = None


def get_session_maker() -> async_sessionmaker[AsyncSession]:
    """
    Get or create the async session factory.

    Returns:
        async_sessionmaker: Factory for creating AsyncSession instances.
    """
    global _async_session_maker
    if _async_session_maker is None:
        _async_session_maker = async_sessionmaker(
            bind=get_engine(),
            class_=AsyncSession,
            expire_on_commit=False,  # Required for async - prevents lazy loading issues
            autoflush=False,
        )
    return _async_session_maker


@asynccontextmanager
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Get an async database session with automatic transaction management.

    Usage:
        async with get_db_session() as session:
            result = await session.execute(select(Model))
            # ... work with session
            # Commits automatically on successful exit
            # Rolls back on exception

    Yields:
        AsyncSession: An async database session.
    """
    session_maker = get_session_maker()
    async with session_maker() as session:
        try:
            yield session
            await session.commit()
        except Exception as e:
            await session.rollback()
            logger.error(f"Database session error, rolling back: {e}")
            raise
        finally:
            await session.close()


async def init_db() -> None:
    """
    Initialize database by creating all tables.

    Note: In production, use Alembic migrations instead.
    This is primarily for development/testing.
    """
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables initialized")


async def dispose_engine() -> None:
    """
    Dispose of the database engine and close all connections.

    Call this during application shutdown.
    """
    global _engine, _async_session_maker
    if _engine is not None:
        await _engine.dispose()
        logger.info("Database engine disposed")
        _engine = None
        _async_session_maker = None


@asynccontextmanager
async def database_lifespan():
    """
    FastAPI lifespan context manager for database.

    Usage in FastAPI:
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            async with database_lifespan():
                yield

        app = FastAPI(lifespan=lifespan)

    Yields:
        None
    """
    # Initialize engine on startup
    get_engine()
    logger.info("Database connection pool initialized")
    try:
        yield
    finally:
        await dispose_engine()
        logger.info("Database connection pool closed")
