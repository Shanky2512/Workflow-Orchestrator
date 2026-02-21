#!/usr/bin/env python3
"""
EchoAI Database Connection Test Script

Tests connectivity to the PostgreSQL database using the configured settings.

Usage:
    python scripts/test_db_connection.py

Configuration:
    Set DATABASE_URL in .env file or environment variable.
    Default: postgresql+asyncpg://echoai:echoai_dev@localhost:5432/echoai
"""

import asyncio
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import text


async def test_connection() -> bool:
    """
    Test database connection.

    Returns:
        bool: True if connection successful, False otherwise.
    """
    try:
        # Import here to ensure path is set up
        from echolib.database import get_engine
        from echolib.config import settings

        print("=" * 60)
        print("EchoAI Database Connection Test")
        print("=" * 60)

        # Mask password for display
        db_url = settings.database_url
        masked_url = db_url
        if "@" in db_url:
            prefix, rest = db_url.split("://", 1)
            if "@" in rest:
                creds, host = rest.split("@", 1)
                if ":" in creds:
                    user, _ = creds.split(":", 1)
                    masked_url = f"{prefix}://{user}:****@{host}"

        print(f"\nConnection URL: {masked_url}")
        print(f"Pool Size: {settings.database_pool_size}")
        print(f"Max Overflow: {settings.database_max_overflow}")
        print()

        # Get the engine
        print("Creating database engine...")
        engine = get_engine()

        # Test connection with a simple query
        print("Testing connection...")
        async with engine.connect() as conn:
            result = await conn.execute(text("SELECT version()"))
            version = result.scalar()
            print(f"\nPostgreSQL Version: {version}")

            # Test basic query execution
            result = await conn.execute(text("SELECT current_database(), current_user"))
            row = result.fetchone()
            if row:
                print(f"Database: {row[0]}")
                print(f"User: {row[1]}")

            # Check connection pool status
            print(f"\nConnection Pool Status:")
            print(f"  Pool Size: {engine.pool.size()}")
            print(f"  Checked In: {engine.pool.checkedin()}")
            print(f"  Checked Out: {engine.pool.checkedout()}")

        # Dispose engine
        await engine.dispose()

        print("\n" + "=" * 60)
        print("SUCCESS: Database connection test passed!")
        print("=" * 60)
        return True

    except ImportError as e:
        print(f"\nERROR: Import failed - {e}")
        print("Make sure you have installed all requirements:")
        print("  pip install -r requirements.txt")
        return False

    except Exception as e:
        print(f"\nERROR: Connection failed - {type(e).__name__}: {e}")
        print("\nTroubleshooting:")
        print("  1. Check if PostgreSQL is running")
        print("  2. Verify DATABASE_URL in .env file")
        print("  3. Check if database and user exist")
        print("  4. Verify network connectivity to database host")
        return False


def main():
    """Main entry point."""
    print()
    success = asyncio.run(test_connection())
    print()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
