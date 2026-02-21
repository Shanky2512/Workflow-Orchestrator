"""
HITL Checkpoint Persistence via LangGraph PostgresSaver.

Provides a PostgresSaver instance for HITL-enabled workflows.
Non-HITL workflows continue using MemorySaver (unchanged).

This module does NOT modify any existing MemorySaver usage in compiler.py.

Connection String:
    echolib uses: postgresql+asyncpg://user:pass@host:port/db
    PostgresSaver needs: postgresql://user:pass@host:port/db
    This module strips the SQLAlchemy async driver prefix automatically.
"""
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def get_hitl_checkpointer_uri() -> str:
    """
    Derive a plain PostgreSQL URI for PostgresSaver from the existing
    echolib database_url config.

    echolib uses: postgresql+asyncpg://user:pass@host:port/db
    PostgresSaver needs: postgresql://user:pass@host:port/db

    Returns:
        Plain PostgreSQL connection string suitable for psycopg v3.
    """
    from echolib.config import settings

    db_url = settings.database_url
    # Strip SQLAlchemy async driver prefix
    if "+asyncpg" in db_url:
        return db_url.replace("+asyncpg", "")
    if "+psycopg" in db_url:
        return db_url.replace("+psycopg", "")
    return db_url


def get_postgres_checkpointer():
    """
    Create and return a ready-to-use PostgresSaver instance for HITL checkpointing.

    Enters the context manager returned by from_conn_string() so the caller
    receives the actual PostgresSaver object (not the context manager wrapper).
    The connection stays alive for the lifetime of the process.

    Returns:
        PostgresSaver instance connected to the echoAI database.
    """
    from langgraph.checkpoint.postgres import PostgresSaver

    db_uri = get_hitl_checkpointer_uri()
    logger.info("Creating PostgresSaver checkpointer (host derived from echolib config)")
    ctx = PostgresSaver.from_conn_string(db_uri)
    checkpointer = ctx.__enter__()
    # Run setup to ensure checkpoint tables exist (idempotent)
    checkpointer.setup()
    # CRITICAL: Prevent the context manager from being garbage-collected.
    # ctx owns the psycopg connection; if it goes out of scope, Python's
    # refcount GC destroys it immediately, closing the connection before
    # the checkpointer is ever used by invoke().
    checkpointer._conn_ctx = ctx
    return checkpointer


def setup_checkpoint_tables() -> None:
    """
    Create LangGraph checkpoint tables in PostgreSQL if they do not exist.

    This is idempotent and safe to call on every application startup.
    Should be called once during app initialization.
    """
    try:
        with get_postgres_checkpointer() as cp:
            cp.setup()
        logger.info("LangGraph checkpoint tables verified/created successfully")
    except Exception as e:
        logger.warning(
            f"Failed to setup LangGraph checkpoint tables: {e}. "
            "HITL workflows requiring PostgresSaver will not function until "
            "the database is available."
        )
