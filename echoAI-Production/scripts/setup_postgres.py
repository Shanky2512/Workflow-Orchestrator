"""
EchoAI PostgreSQL Setup Script

Automatically initializes PostgreSQL database, user, and permissions.
Similar to what docker-compose.yml would do, but for local PostgreSQL installation.

Usage:
    python scripts/setup_postgres.py

    # With custom postgres password:
    python scripts/setup_postgres.py --postgres-password yourpassword

    # With custom settings:
    python scripts/setup_postgres.py --db-name mydb --db-user myuser --db-password mypass

Requirements:
    - PostgreSQL installed and running locally
    - psycopg2-binary package (pip install psycopg2-binary)
"""

import argparse
import sys
import getpass

try:
    import psycopg2
    from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
except ImportError:
    print("=" * 60)
    print("ERROR: psycopg2 not installed")
    print("=" * 60)
    print("\nRun: pip install psycopg2-binary")
    print("\nThen try again.")
    sys.exit(1)


def setup_database(
    postgres_host: str = "localhost",
    postgres_port: int = 5432,
    postgres_user: str = "postgres",
    postgres_password: str = None,
    db_name: str = "echoai",
    db_user: str = "echoai",
    db_password: str = "echoai_dev"
):
    """
    Set up PostgreSQL database for EchoAI.

    Creates:
    - Database: echoai
    - User: echoai with password echoai_dev
    - Grants all privileges
    """

    print("=" * 60)
    print("EchoAI PostgreSQL Setup")
    print("=" * 60)

    # Prompt for postgres password if not provided
    if postgres_password is None:
        postgres_password = getpass.getpass(f"Enter password for PostgreSQL user '{postgres_user}': ")

    print(f"\n📋 Configuration:")
    print(f"   PostgreSQL Host: {postgres_host}:{postgres_port}")
    print(f"   PostgreSQL User: {postgres_user}")
    print(f"   Database Name:   {db_name}")
    print(f"   App User:        {db_user}")
    print(f"   App Password:    {'*' * len(db_password)}")
    print()

    try:
        # Connect to PostgreSQL as superuser
        print("🔌 Connecting to PostgreSQL...")
        conn = psycopg2.connect(
            host=postgres_host,
            port=postgres_port,
            user=postgres_user,
            password=postgres_password,
            database="postgres"  # Connect to default database first
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
        print("✅ Connected to PostgreSQL")

        # Check if database exists
        print(f"\n📦 Checking if database '{db_name}' exists...")
        cursor.execute(
            "SELECT 1 FROM pg_database WHERE datname = %s",
            (db_name,)
        )
        db_exists = cursor.fetchone()

        if db_exists:
            print(f"ℹ️  Database '{db_name}' already exists")
        else:
            print(f"📦 Creating database '{db_name}'...")
            cursor.execute(f'CREATE DATABASE {db_name}')
            print(f"✅ Database '{db_name}' created")

        # Check if user exists
        print(f"\n👤 Checking if user '{db_user}' exists...")
        cursor.execute(
            "SELECT 1 FROM pg_roles WHERE rolname = %s",
            (db_user,)
        )
        user_exists = cursor.fetchone()

        if user_exists:
            print(f"ℹ️  User '{db_user}' already exists")
            # Update password anyway
            print(f"🔑 Updating password for user '{db_user}'...")
            cursor.execute(
                f"ALTER USER {db_user} WITH PASSWORD %s",
                (db_password,)
            )
            print(f"✅ Password updated")
        else:
            print(f"👤 Creating user '{db_user}'...")
            cursor.execute(
                f"CREATE USER {db_user} WITH PASSWORD %s",
                (db_password,)
            )
            print(f"✅ User '{db_user}' created")

        # Grant privileges
        print(f"\n🔐 Granting privileges...")

        # Grant all privileges on database
        cursor.execute(f'GRANT ALL PRIVILEGES ON DATABASE {db_name} TO {db_user}')
        print(f"   ✅ Granted all privileges on database '{db_name}'")

        # Make user the owner of the database
        cursor.execute(f'ALTER DATABASE {db_name} OWNER TO {db_user}')
        print(f"   ✅ Set '{db_user}' as owner of database '{db_name}'")

        cursor.close()
        conn.close()

        # Now connect to the new database to grant schema privileges
        print(f"\n🔐 Granting schema privileges...")
        conn = psycopg2.connect(
            host=postgres_host,
            port=postgres_port,
            user=postgres_user,
            password=postgres_password,
            database=db_name
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()

        cursor.execute(f'GRANT ALL ON SCHEMA public TO {db_user}')
        print(f"   ✅ Granted all privileges on schema 'public'")

        cursor.execute(f'GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO {db_user}')
        print(f"   ✅ Granted all privileges on all tables")

        cursor.execute(f'GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO {db_user}')
        print(f"   ✅ Granted all privileges on all sequences")

        # Set default privileges for future tables
        cursor.execute(f'ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO {db_user}')
        cursor.execute(f'ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO {db_user}')
        print(f"   ✅ Set default privileges for future objects")

        cursor.close()
        conn.close()

        # ============================================================
        # VERIFICATION STEP - Test the new credentials
        # ============================================================
        print("\n" + "=" * 60)
        print("🔍 VERIFICATION - Testing new credentials...")
        print("=" * 60)

        try:
            # Connect using the NEW user credentials
            verify_conn = psycopg2.connect(
                host=postgres_host,
                port=postgres_port,
                user=db_user,
                password=db_password,
                database=db_name
            )
            verify_cursor = verify_conn.cursor()

            # Test 1: Check connection
            print(f"\n✅ Test 1: Connected to '{db_name}' as user '{db_user}'")

            # Test 2: Check we can create a table
            verify_cursor.execute("""
                CREATE TABLE IF NOT EXISTS _setup_verification_test (
                    id SERIAL PRIMARY KEY,
                    test_value VARCHAR(50),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            verify_conn.commit()
            print(f"✅ Test 2: Can CREATE tables")

            # Test 3: Check we can insert
            verify_cursor.execute("""
                INSERT INTO _setup_verification_test (test_value)
                VALUES ('setup_test') RETURNING id
            """)
            inserted_id = verify_cursor.fetchone()[0]
            verify_conn.commit()
            print(f"✅ Test 3: Can INSERT data (id={inserted_id})")

            # Test 4: Check we can select
            verify_cursor.execute("SELECT COUNT(*) FROM _setup_verification_test")
            count = verify_cursor.fetchone()[0]
            print(f"✅ Test 4: Can SELECT data (count={count})")

            # Test 5: Check we can drop
            verify_cursor.execute("DROP TABLE _setup_verification_test")
            verify_conn.commit()
            print(f"✅ Test 5: Can DROP tables")

            # Get database info
            verify_cursor.execute("SELECT version()")
            pg_version = verify_cursor.fetchone()[0].split(',')[0]
            print(f"\n📊 PostgreSQL Version: {pg_version}")

            verify_cursor.execute(f"""
                SELECT pg_size_pretty(pg_database_size('{db_name}'))
            """)
            db_size = verify_cursor.fetchone()[0]
            print(f"📊 Database Size: {db_size}")

            verify_cursor.close()
            verify_conn.close()

            print("\n" + "=" * 60)
            print("🎉 ALL VERIFICATIONS PASSED!")
            print("=" * 60)
            verification_passed = True

        except Exception as e:
            print(f"\n❌ Verification FAILED: {e}")
            verification_passed = False

        print(f"\n📝 Your DATABASE_URL for .env:")
        print(f"\n   DATABASE_URL=postgresql+asyncpg://{db_user}:{db_password}@{postgres_host}:{postgres_port}/{db_name}")
        print(f"\n📋 Next Steps:")
        print(f"   1. Update your .env file with the DATABASE_URL above")
        print(f"   2. Run migrations: alembic upgrade head")
        print(f"   3. Migrate data: python scripts/migrate_json_to_db.py")
        print(f"   4. Start server: uvicorn apps.gateway.main:app --reload")
        print()

        return verification_passed

    except psycopg2.OperationalError as e:
        print(f"\n❌ Connection Error: {e}")
        print("\nPossible causes:")
        print("  - PostgreSQL is not running")
        print("  - Wrong password for postgres user")
        print("  - PostgreSQL not accepting connections on localhost:5432")
        print("\nTry:")
        print("  - Check if PostgreSQL service is running")
        print("  - Verify your postgres password")
        return False

    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Set up PostgreSQL database for EchoAI"
    )
    parser.add_argument(
        "--postgres-host",
        default="localhost",
        help="PostgreSQL host (default: localhost)"
    )
    parser.add_argument(
        "--postgres-port",
        type=int,
        default=5432,
        help="PostgreSQL port (default: 5432)"
    )
    parser.add_argument(
        "--postgres-user",
        default="postgres",
        help="PostgreSQL superuser (default: postgres)"
    )
    parser.add_argument(
        "--postgres-password",
        default=None,
        help="PostgreSQL superuser password (will prompt if not provided)"
    )
    parser.add_argument(
        "--db-name",
        default="echoai",
        help="Database name to create (default: echoai)"
    )
    parser.add_argument(
        "--db-user",
        default="echoai",
        help="Database user to create (default: echoai)"
    )
    parser.add_argument(
        "--db-password",
        default="echoai_dev",
        help="Database user password (default: echoai_dev)"
    )

    args = parser.parse_args()

    success = setup_database(
        postgres_host=args.postgres_host,
        postgres_port=args.postgres_port,
        postgres_user=args.postgres_user,
        postgres_password=args.postgres_password,
        db_name=args.db_name,
        db_user=args.db_user,
        db_password=args.db_password
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
