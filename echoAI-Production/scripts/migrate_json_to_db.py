"""
EchoAI JSON to PostgreSQL Migration Script

Migrates existing JSON file storage to PostgreSQL database.
Uses SQLAlchemy 2.0 async patterns for data access.

Usage:
    cd echoAI
    python scripts/migrate_json_to_db.py

Prerequisites:
    - PostgreSQL running (docker-compose up -d)
    - Migrations applied (alembic upgrade head)
"""
import asyncio
import json
import hashlib
import sys
from pathlib import Path
from datetime import datetime, timezone
from uuid import UUID
import logging

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from echolib.database import get_db_session
from echolib.models import User, Agent, Workflow, WorkflowVersion, ChatSession, Tool
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Storage paths relative to echoAI directory
BASE_DIR = Path(__file__).parent.parent
AGENTS_DIR = BASE_DIR / "apps" / "storage" / "agents"
WORKFLOWS_DIR = BASE_DIR / "apps" / "workflow" / "storage" / "workflows"
SESSIONS_DIR = BASE_DIR / "apps" / "workflow" / "storage" / "sessions"
TOOLS_DIR = BASE_DIR / "apps" / "storage" / "tools"

# Admin user ID from seed migration (002_seed_admin_user)
ADMIN_USER_ID = UUID("00000000-0000-0000-0000-000000000001")


def parse_uuid(value: str) -> UUID:
    """
    Parse UUID from various string formats.

    Handles prefixed IDs like 'agt_xxx', 'wf_xxx', 'tool_xxx', 'sess_xxx'.
    Falls back to deterministic UUID generation from string hash if parsing fails.

    Args:
        value: String value that may contain a UUID

    Returns:
        UUID object
    """
    if isinstance(value, UUID):
        return value

    if not value:
        raise ValueError("Cannot parse empty value as UUID")

    # Remove common prefixes
    clean = value
    for prefix in ('agt_', 'wf_', 'tool_', 'sess_'):
        if clean.startswith(prefix):
            clean = clean[len(prefix):]
            break

    try:
        return UUID(clean)
    except ValueError:
        # Generate deterministic UUID from string hash
        hash_bytes = hashlib.md5(value.encode()).digest()
        return UUID(bytes=hash_bytes)


def get_workflow_status(file_path: Path, data: dict) -> str:
    """
    Determine workflow status from file path or data.

    Args:
        file_path: Path to the workflow JSON file
        data: Parsed workflow data

    Returns:
        Status string: 'draft', 'final', or 'archived'
    """
    path_str = str(file_path).lower()

    if 'draft' in path_str:
        return 'draft'
    elif 'final' in path_str:
        return 'final'
    elif 'archive' in path_str:
        return 'archived'

    # Fall back to status in data
    status = data.get('status', 'draft')

    # Map 'validated' to 'draft' since validated is a transient state
    if status == 'validated':
        return 'draft'

    return status if status in ('draft', 'final', 'archived') else 'draft'


async def ensure_admin_user() -> UUID:
    """
    Ensure the admin user exists in the database.

    The admin user should already exist from migration 002_seed_admin_user.
    This function verifies and logs the result.

    Returns:
        Admin user UUID
    """
    logger.info("Ensuring admin user exists...")

    async with get_db_session() as db:
        result = await db.execute(
            select(User).where(User.user_id == ADMIN_USER_ID)
        )
        user = result.scalar_one_or_none()

        if user:
            logger.info(f"Admin user ready: {ADMIN_USER_ID}")
            return ADMIN_USER_ID

        # Create admin user if not exists (fallback for direct script runs)
        logger.info("Admin user not found, creating...")
        admin_user = User(
            user_id=ADMIN_USER_ID,
            email="admin@echoai.local",
            display_name="System Admin",
            user_metadata={"role": "admin", "migrated": True, "system_user": True},
            is_active=True
        )
        db.add(admin_user)
        await db.commit()
        logger.info(f"Admin user created: {ADMIN_USER_ID}")

    return ADMIN_USER_ID


async def migrate_agents() -> int:
    """
    Migrate agent JSON files to PostgreSQL agents table.

    Reads all agt_*.json files from the agents storage directory
    and upserts them into the database.

    Returns:
        Count of migrated agents
    """
    logger.info(f"Migrating agents from {AGENTS_DIR}...")

    if not AGENTS_DIR.exists():
        logger.warning(f"Agents directory not found: {AGENTS_DIR}")
        return 0

    migrated_count = 0

    async with get_db_session() as db:
        # Find all agent JSON files (excluding index files)
        agent_files = list(AGENTS_DIR.glob("agt_*.json"))

        for file_path in agent_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError, IOError) as e:
                logger.warning(f"Skipping {file_path}: {e}")
                continue

            # Extract required fields
            agent_id_str = data.get('agent_id')
            if not agent_id_str:
                logger.warning(f"Skipping {file_path}: no agent_id found")
                continue

            try:
                agent_id = parse_uuid(agent_id_str)
            except ValueError as e:
                logger.warning(f"Skipping {file_path}: invalid agent_id - {e}")
                continue

            name = data.get('name', 'Unnamed Agent')

            # Prepare UPSERT statement
            stmt = insert(Agent).values(
                agent_id=agent_id,
                user_id=ADMIN_USER_ID,
                name=name,
                definition=data,
                source_workflow_id=None,
                is_deleted=False
            )

            stmt = stmt.on_conflict_do_update(
                index_elements=['agent_id'],
                set_={
                    'name': stmt.excluded.name,
                    'definition': stmt.excluded.definition,
                    'updated_at': datetime.now(timezone.utc)
                }
            )

            await db.execute(stmt)
            logger.info(f"Migrated agent: {agent_id_str} ({name})")
            migrated_count += 1

        await db.commit()

    logger.info(f"Migrated {migrated_count} agents")
    return migrated_count


async def migrate_workflows() -> int:
    """
    Migrate workflow JSON files to PostgreSQL workflows table.

    Reads workflow files from draft/, final/, and archive/ subdirectories.
    Skips temp/ directory as per the plan (temporary files not stored in DB).
    Also syncs embedded agents to the agents table.

    Returns:
        Count of migrated workflows
    """
    logger.info(f"Migrating workflows from {WORKFLOWS_DIR}...")

    if not WORKFLOWS_DIR.exists():
        logger.warning(f"Workflows directory not found: {WORKFLOWS_DIR}")
        return 0

    migrated_count = 0

    # Directories to process (excluding temp/)
    subdirs = ['draft', 'final', 'archive']

    async with get_db_session() as db:
        for subdir in subdirs:
            subdir_path = WORKFLOWS_DIR / subdir
            if not subdir_path.exists():
                continue

            # Find all JSON files in subdirectory
            workflow_files = list(subdir_path.glob("*.json"))

            for file_path in workflow_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                except (json.JSONDecodeError, FileNotFoundError, IOError) as e:
                    logger.warning(f"Skipping {file_path}: {e}")
                    continue

                # Extract required fields
                workflow_id_str = data.get('workflow_id')
                if not workflow_id_str:
                    logger.warning(f"Skipping {file_path}: no workflow_id found")
                    continue

                try:
                    workflow_id = parse_uuid(workflow_id_str)
                except ValueError as e:
                    logger.warning(f"Skipping {file_path}: invalid workflow_id - {e}")
                    continue

                name = data.get('name', 'Unnamed Workflow')
                status = get_workflow_status(file_path, data)
                version = data.get('version', '0.1')

                # Sync embedded agents first
                embedded_agents = data.get('agents', [])
                for agent_data in embedded_agents:
                    agent_id_str = agent_data.get('agent_id')
                    if not agent_id_str:
                        continue

                    try:
                        agent_id = parse_uuid(agent_id_str)
                    except ValueError:
                        continue

                    agent_name = agent_data.get('name', 'Unnamed Agent')

                    # UPSERT embedded agent
                    agent_stmt = insert(Agent).values(
                        agent_id=agent_id,
                        user_id=ADMIN_USER_ID,
                        name=agent_name,
                        definition=agent_data,
                        source_workflow_id=workflow_id,
                        is_deleted=False
                    )

                    agent_stmt = agent_stmt.on_conflict_do_update(
                        index_elements=['agent_id'],
                        set_={
                            'name': agent_stmt.excluded.name,
                            'definition': agent_stmt.excluded.definition,
                            'source_workflow_id': agent_stmt.excluded.source_workflow_id,
                            'updated_at': datetime.now(timezone.utc)
                        }
                    )

                    await db.execute(agent_stmt)

                # UPSERT workflow
                workflow_stmt = insert(Workflow).values(
                    workflow_id=workflow_id,
                    user_id=ADMIN_USER_ID,
                    name=name,
                    status=status,
                    definition=data,
                    is_deleted=False
                )

                workflow_stmt = workflow_stmt.on_conflict_do_update(
                    index_elements=['workflow_id'],
                    set_={
                        'name': workflow_stmt.excluded.name,
                        'status': workflow_stmt.excluded.status,
                        'definition': workflow_stmt.excluded.definition,
                        'updated_at': datetime.now(timezone.utc)
                    }
                )

                await db.execute(workflow_stmt)

                # Create workflow version record for final workflows
                if status == 'final':
                    # Check if version already exists
                    version_check = await db.execute(
                        select(WorkflowVersion).where(
                            WorkflowVersion.workflow_id == workflow_id,
                            WorkflowVersion.version == version
                        )
                    )
                    existing_version = version_check.scalar_one_or_none()

                    if not existing_version:
                        version_record = WorkflowVersion(
                            workflow_id=workflow_id,
                            version=version,
                            definition=data,
                            status='final',
                            created_by=ADMIN_USER_ID,
                            notes=f"Migrated from {file_path.name}"
                        )
                        db.add(version_record)

                logger.info(f"Migrated workflow: {workflow_id_str} ({name}) [status={status}]")
                migrated_count += 1

        await db.commit()

    logger.info(f"Migrated {migrated_count} workflows")
    return migrated_count


async def migrate_sessions() -> int:
    """
    Migrate session JSON files to PostgreSQL chat_sessions table.

    Reads all *.json files from the sessions storage directory
    and upserts them into the database with embedded messages.

    Returns:
        Count of migrated sessions
    """
    logger.info(f"Migrating sessions from {SESSIONS_DIR}...")

    if not SESSIONS_DIR.exists():
        logger.warning(f"Sessions directory not found: {SESSIONS_DIR}")
        return 0

    migrated_count = 0

    async with get_db_session() as db:
        # Find all session JSON files
        session_files = list(SESSIONS_DIR.glob("*.json"))

        for file_path in session_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError, IOError) as e:
                logger.warning(f"Skipping {file_path}: {e}")
                continue

            # Extract session_id from data or filename
            session_id_str = data.get('session_id')
            if not session_id_str:
                # Try to extract from filename (e.g., "46f5ec0f-a236-45eb-858a-90c7798fa574.json")
                session_id_str = file_path.stem

            try:
                session_id = parse_uuid(session_id_str)
            except ValueError as e:
                logger.warning(f"Skipping {file_path}: invalid session_id - {e}")
                continue

            # Extract workflow_id if present
            workflow_id_str = data.get('workflow_id')
            workflow_id = None
            context_type = 'general'

            if workflow_id_str:
                try:
                    workflow_id = parse_uuid(workflow_id_str)
                    context_type = 'workflow'
                except ValueError:
                    pass

            # Extract messages
            messages = data.get('messages', [])

            # Extract other fields
            title = data.get('context', {}).get('workflow_name', 'Chat Session')
            if not title or title == 'Chat Session':
                title = f"Session {session_id_str[:8]}"

            workflow_mode = data.get('workflow_mode')
            workflow_version = data.get('workflow_version')
            context_data = data.get('context', {})
            run_ids = data.get('run_ids', [])

            # Parse run_ids as UUIDs where possible
            parsed_run_ids = []
            for rid in run_ids:
                try:
                    parsed_run_ids.append(parse_uuid(str(rid)))
                except ValueError:
                    pass

            # Parse last_activity timestamp
            last_activity_str = data.get('last_activity')
            last_activity = datetime.now(timezone.utc)
            if last_activity_str:
                try:
                    # Handle ISO format timestamps
                    last_activity = datetime.fromisoformat(
                        last_activity_str.replace('Z', '+00:00')
                    )
                except ValueError:
                    pass

            # Prepare UPSERT statement
            stmt = insert(ChatSession).values(
                session_id=session_id,
                user_id=ADMIN_USER_ID,
                title=title,
                context_type=context_type,
                context_id=workflow_id,
                workflow_mode=workflow_mode,
                workflow_version=workflow_version,
                messages=messages,
                selected_tool_ids=[],
                context_data=context_data,
                variables={},
                state_schema={},
                run_ids=parsed_run_ids,
                last_activity=last_activity,
                is_deleted=False
            )

            stmt = stmt.on_conflict_do_update(
                index_elements=['session_id'],
                set_={
                    'title': stmt.excluded.title,
                    'messages': stmt.excluded.messages,
                    'context_data': stmt.excluded.context_data,
                    'run_ids': stmt.excluded.run_ids,
                    'last_activity': stmt.excluded.last_activity
                }
            )

            await db.execute(stmt)
            logger.info(f"Migrated session: {session_id_str} ({title})")
            migrated_count += 1

        await db.commit()

    logger.info(f"Migrated {migrated_count} sessions")
    return migrated_count


async def migrate_tools() -> int:
    """
    Migrate tool JSON files to PostgreSQL tools table.

    Reads tool_index.json for the tool list, then reads individual
    tool definition files and upserts them into the database.

    Returns:
        Count of migrated tools
    """
    logger.info(f"Migrating tools from {TOOLS_DIR}...")

    if not TOOLS_DIR.exists():
        logger.warning(f"Tools directory not found: {TOOLS_DIR}")
        return 0

    migrated_count = 0

    # Read tool index if available
    tool_index_path = TOOLS_DIR / "tool_index.json"
    tool_ids = []

    if tool_index_path.exists():
        try:
            with open(tool_index_path, 'r', encoding='utf-8') as f:
                index_data = json.load(f)
                # Get tool IDs from index
                tools_dict = index_data.get('tools', {})
                tool_ids = list(tools_dict.keys())
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Could not read tool index: {e}")

    # Also find any tool_*.json files directly
    tool_files = list(TOOLS_DIR.glob("tool_*.json"))

    async with get_db_session() as db:
        # Process individual tool files
        for file_path in tool_files:
            # Skip the index file
            if file_path.name == "tool_index.json":
                continue

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError, IOError) as e:
                logger.warning(f"Skipping {file_path}: {e}")
                continue

            # Extract required fields
            tool_id_str = data.get('tool_id')
            if not tool_id_str:
                # Use filename as tool_id
                tool_id_str = file_path.stem

            try:
                tool_id = parse_uuid(tool_id_str)
            except ValueError as e:
                logger.warning(f"Skipping {file_path}: invalid tool_id - {e}")
                continue

            name = data.get('name', 'Unnamed Tool')
            description = data.get('description')
            tool_type = data.get('tool_type', 'local')

            # Validate tool_type
            valid_types = ('local', 'mcp', 'api', 'crewai', 'custom')
            if tool_type not in valid_types:
                tool_type = 'local'

            status = data.get('status', 'active')
            valid_statuses = ('active', 'deprecated', 'disabled')
            if status not in valid_statuses:
                status = 'active'

            version = data.get('version', '1.0')
            tags = data.get('tags', [])

            # Ensure tags is a list of strings
            if not isinstance(tags, list):
                tags = []
            tags = [str(t) for t in tags if t]

            # Prepare UPSERT statement
            stmt = insert(Tool).values(
                tool_id=tool_id,
                user_id=ADMIN_USER_ID,
                name=name,
                description=description,
                tool_type=tool_type,
                definition=data,
                status=status,
                version=version,
                tags=tags,
                is_deleted=False
            )

            stmt = stmt.on_conflict_do_update(
                index_elements=['tool_id'],
                set_={
                    'name': stmt.excluded.name,
                    'description': stmt.excluded.description,
                    'tool_type': stmt.excluded.tool_type,
                    'definition': stmt.excluded.definition,
                    'status': stmt.excluded.status,
                    'version': stmt.excluded.version,
                    'tags': stmt.excluded.tags,
                    'updated_at': datetime.now(timezone.utc)
                }
            )

            await db.execute(stmt)
            logger.info(f"Migrated tool: {tool_id_str} ({name})")
            migrated_count += 1

        await db.commit()

    logger.info(f"Migrated {migrated_count} tools")
    return migrated_count


async def main():
    """
    Main migration entry point.

    Runs all migrations in dependency order:
    1. Ensure admin user exists
    2. Migrate agents
    3. Migrate workflows (includes embedded agents)
    4. Migrate sessions
    5. Migrate tools
    """
    logger.info("=" * 60)
    logger.info("EchoAI JSON to PostgreSQL Migration")
    logger.info("=" * 60)

    try:
        # Ensure admin user exists
        await ensure_admin_user()

        # Run migrations in dependency order
        agents_count = await migrate_agents()
        workflows_count = await migrate_workflows()
        sessions_count = await migrate_sessions()
        tools_count = await migrate_tools()

        # Summary
        logger.info("=" * 60)
        logger.info("Migration Complete!")
        logger.info(f"  Agents: {agents_count}")
        logger.info(f"  Workflows: {workflows_count}")
        logger.info(f"  Sessions: {sessions_count}")
        logger.info(f"  Tools: {tools_count}")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Migration failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
