"""
EchoAI Workflow Repository

Provides data access for workflow definitions with dual-write support.
Workflows contain embedded agents that are synced to the agents table
when saved.

Dual-Write Strategy:
    - DRAFT save: Database + Agent sync
    - TEMP save: Filesystem ONLY (not in this repository)
    - FINAL save: Database + Versions table + Agent sync
"""

from typing import Any, Dict, List, Optional
from uuid import UUID

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from echolib.models import Workflow, WorkflowVersion
from .base import BaseRepository
from .agent_repo import AgentRepository
from apps.workflow.visualization.node_mapper import normalize_agent_config


class WorkflowRepository(BaseRepository[Workflow]):
    """
    Repository for Workflow entity operations.

    Provides standard CRUD plus:
        - save_with_agents: Atomic workflow + agent sync
        - Status-based filtering
        - Version management

    Workflows store complete JSON in definition JSONB, including
    embedded agent copies. When saved, agents are extracted and
    synced to the agents table via AgentRepository.
    """

    model_class = Workflow
    id_field = "workflow_id"
    user_id_field = "user_id"
    soft_delete_field = "is_deleted"

    def __init__(self):
        self._agent_repo = AgentRepository()

    async def save_with_agents(
        self,
        db: AsyncSession,
        user_id: str,
        workflow_data: Dict[str, Any],
        db_workflow_id: Optional[str] = None
    ) -> Workflow:
        """
        Save workflow AND upsert all embedded agents atomically.

        This is the primary save method for workflows. It:
        1. Creates or updates the workflow record
        2. Extracts embedded agents from workflow definition
        3. Upserts each agent to the agents table

        Args:
            db: Async database session
            user_id: Owner user ID
            workflow_data: Complete workflow definition dict
            db_workflow_id: Optional UUID string for the DB primary key.
                If provided, this is used for the PK column so that
                workflow_data can keep its original prefixed ID (e.g. wf_...)
                in the definition JSONB.

        Returns:
            The saved Workflow instance
        """
        user_uuid = UUID(user_id) if isinstance(user_id, str) else user_id

        # Determine the UUID for the DB primary key
        if db_workflow_id:
            workflow_uuid = UUID(db_workflow_id) if isinstance(db_workflow_id, str) else db_workflow_id
        else:
            # Backward compat: extract from workflow_data
            workflow_id = workflow_data.get("workflow_id")
            if not workflow_id:
                raise ValueError("workflow_data must include workflow_id")
            workflow_uuid = UUID(workflow_id) if isinstance(workflow_id, str) else workflow_id
        name = workflow_data.get("name", "Unnamed Workflow")
        status = workflow_data.get("status", "draft")

        # Check if workflow exists
        existing = await self.get_by_id(db, str(workflow_uuid), str(user_uuid))

        if existing:
            # Update existing workflow
            stmt = (
                update(Workflow)
                .where(
                    Workflow.workflow_id == workflow_uuid,
                    Workflow.user_id == user_uuid
                )
                .values(
                    name=name,
                    status=status,
                    definition=workflow_data,
                    is_deleted=False
                )
                .returning(Workflow)
            )
            result = await db.execute(stmt)
            workflow = result.scalar_one()
        else:
            # Create new workflow
            workflow = Workflow(
                workflow_id=workflow_uuid,
                user_id=user_uuid,
                name=name,
                status=status,
                definition=workflow_data,
                is_deleted=False
            )
            db.add(workflow)

        await db.flush()
        await db.refresh(workflow)

        # Extract and sync embedded agents
        agents = workflow_data.get("agents", [])
        if agents:
            # Normalize agent configs before DB write
            for agent_entry in agents:
                if isinstance(agent_entry, dict):
                    normalize_agent_config(agent_entry)
            await self._agent_repo.bulk_upsert(
                db,
                str(user_uuid),
                agents,
                source_workflow_id=str(workflow_uuid)
            )

        return workflow

    async def get_by_status(
        self,
        db: AsyncSession,
        user_id: str,
        status: str,
        limit: int = 100,
        offset: int = 0
    ) -> List[Workflow]:
        """
        Get workflows by status for a user.

        Args:
            db: Async database session
            user_id: Owner user ID
            status: Workflow status (draft/validated/final/archived)
            limit: Maximum results to return
            offset: Number of records to skip

        Returns:
            List of Workflow instances with matching status
        """
        user_uuid = UUID(user_id) if isinstance(user_id, str) else user_id

        stmt = (
            select(Workflow)
            .where(
                Workflow.user_id == user_uuid,
                Workflow.status == status,
                Workflow.is_deleted == False
            )
            .order_by(Workflow.updated_at.desc())
            .limit(limit)
            .offset(offset)
        )

        result = await db.execute(stmt)
        return list(result.scalars().all())

    async def create_version(
        self,
        db: AsyncSession,
        workflow_id: str,
        version: str,
        created_by: Optional[str] = None,
        notes: Optional[str] = None
    ) -> WorkflowVersion:
        """
        Create an immutable version snapshot of a workflow.

        Called when saving a workflow to 'final' status.
        Captures the current workflow definition as a version.

        Args:
            db: Async database session
            workflow_id: Workflow to version
            version: Version string (e.g., "1.0", "2.1")
            created_by: User ID who created the version
            notes: Optional release notes

        Returns:
            The created WorkflowVersion instance

        Raises:
            ValueError: If workflow not found
        """
        workflow_uuid = UUID(workflow_id) if isinstance(workflow_id, str) else workflow_id

        # Get current workflow
        stmt = select(Workflow).where(Workflow.workflow_id == workflow_uuid)
        result = await db.execute(stmt)
        workflow = result.scalar_one_or_none()

        if not workflow:
            raise ValueError(f"Workflow {workflow_id} not found")

        # Prepare created_by
        created_by_uuid = None
        if created_by:
            created_by_uuid = UUID(created_by) if isinstance(created_by, str) else created_by

        # Create version snapshot
        workflow_version = WorkflowVersion(
            workflow_id=workflow_uuid,
            version=version,
            definition=workflow.definition,
            status="final",
            created_by=created_by_uuid,
            notes=notes
        )
        db.add(workflow_version)
        await db.flush()
        await db.refresh(workflow_version)

        return workflow_version

    async def get_version(
        self,
        db: AsyncSession,
        workflow_id: str,
        version: str
    ) -> Optional[WorkflowVersion]:
        """
        Get a specific version of a workflow.

        Args:
            db: Async database session
            workflow_id: Workflow ID
            version: Version string to retrieve

        Returns:
            WorkflowVersion instance if found, None otherwise
        """
        workflow_uuid = UUID(workflow_id) if isinstance(workflow_id, str) else workflow_id

        stmt = select(WorkflowVersion).where(
            WorkflowVersion.workflow_id == workflow_uuid,
            WorkflowVersion.version == version
        )

        result = await db.execute(stmt)
        return result.scalar_one_or_none()

    async def get_versions(
        self,
        db: AsyncSession,
        workflow_id: str
    ) -> List[WorkflowVersion]:
        """
        Get all versions of a workflow.

        Args:
            db: Async database session
            workflow_id: Workflow ID

        Returns:
            List of WorkflowVersion instances, ordered by created_at descending
        """
        workflow_uuid = UUID(workflow_id) if isinstance(workflow_id, str) else workflow_id

        stmt = (
            select(WorkflowVersion)
            .where(WorkflowVersion.workflow_id == workflow_uuid)
            .order_by(WorkflowVersion.created_at.desc())
        )

        result = await db.execute(stmt)
        return list(result.scalars().all())

    async def archive_version(
        self,
        db: AsyncSession,
        workflow_id: str,
        version: str
    ) -> Optional[WorkflowVersion]:
        """
        Archive a specific workflow version.

        Sets the version status to 'archived'.

        Args:
            db: Async database session
            workflow_id: Workflow ID
            version: Version string to archive

        Returns:
            Updated WorkflowVersion if found, None otherwise
        """
        workflow_uuid = UUID(workflow_id) if isinstance(workflow_id, str) else workflow_id

        stmt = (
            update(WorkflowVersion)
            .where(
                WorkflowVersion.workflow_id == workflow_uuid,
                WorkflowVersion.version == version
            )
            .values(status="archived")
            .returning(WorkflowVersion)
        )

        result = await db.execute(stmt)
        await db.flush()

        archived = result.scalar_one_or_none()
        if archived:
            await db.refresh(archived)
        return archived

    async def update_status(
        self,
        db: AsyncSession,
        workflow_id: str,
        user_id: str,
        new_status: str
    ) -> Optional[Workflow]:
        """
        Update workflow status.

        Args:
            db: Async database session
            workflow_id: Workflow ID
            user_id: Owner user ID for scoping
            new_status: New status value

        Returns:
            Updated Workflow if found, None otherwise
        """
        return await self.update(
            db,
            workflow_id,
            user_id,
            {"status": new_status}
        )
