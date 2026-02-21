"""
EchoAI Agent Repository

Provides data access for AI agent definitions with UPSERT for workflow sync.
Agents are synced from workflows - when a workflow is saved, its embedded
agents are upserted to the agents table.
"""

from typing import Any, Dict, List, Optional
from uuid import UUID

from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from echolib.models import Agent
from .base import BaseRepository


class AgentRepository(BaseRepository[Agent]):
    """
    Repository for Agent entity operations.

    Provides standard CRUD plus:
        - UPSERT for workflow agent sync
        - Name-based lookup
        - Full-text search on name

    Agents are synced via UPSERT when workflows are saved:
        - If agent_id exists -> UPDATE definition
        - If agent_id is new -> INSERT new agent
    """

    model_class = Agent
    id_field = "agent_id"
    user_id_field = "user_id"
    soft_delete_field = "is_deleted"

    async def upsert(
        self,
        db: AsyncSession,
        user_id: str,
        agent_data: Dict[str, Any],
        source_workflow_id: Optional[str] = None
    ) -> Agent:
        """
        Insert or update an agent by agent_id or (user_id, name).

        This is the primary sync mechanism for workflow agents.
        When a workflow is saved, each embedded agent is upserted.

        Handles two conflict scenarios:
        1. Same agent_id exists → UPDATE that agent
        2. Same (user_id, name) exists but different agent_id → UPDATE existing by name

        Args:
            db: Async database session
            user_id: Owner user ID
            agent_data: Complete agent definition dict (must include agent_id)
            source_workflow_id: Optional workflow ID that triggered the sync

        Returns:
            The inserted or updated Agent instance
        """
        user_uuid = UUID(user_id) if isinstance(user_id, str) else user_id

        # Extract agent_id from data
        agent_id = agent_data.get("agent_id")
        if not agent_id:
            raise ValueError("agent_data must include agent_id")

        agent_uuid = UUID(agent_id) if isinstance(agent_id, str) else agent_id

        # Extract name from definition
        name = agent_data.get("name", "Unnamed Agent")

        # Prepare source_workflow_id
        source_wf_uuid = None
        if source_workflow_id:
            source_wf_uuid = (
                UUID(source_workflow_id)
                if isinstance(source_workflow_id, str)
                else source_workflow_id
            )

        # Check if agent with same name already exists for this user
        # This handles the unique_agent_name_per_user constraint
        existing_by_name = await self.get_by_name(db, str(user_uuid), name)

        if existing_by_name and existing_by_name.agent_id != agent_uuid:
            # Agent with this name exists but has different ID
            # Update the existing agent instead of creating new one
            stmt = (
                update(Agent)
                .where(
                    Agent.agent_id == existing_by_name.agent_id,
                    Agent.user_id == user_uuid
                )
                .values(
                    definition=agent_data,
                    source_workflow_id=source_wf_uuid,
                    is_deleted=False
                )
                .returning(Agent)
            )
            result = await db.execute(stmt)
            await db.flush()
            agent = result.scalar_one()
            await db.refresh(agent)
            return agent

        # No name conflict - proceed with normal upsert by agent_id
        insert_values = {
            "agent_id": agent_uuid,
            "user_id": user_uuid,
            "name": name,
            "definition": agent_data,
            "source_workflow_id": source_wf_uuid,
            "is_deleted": False
        }

        stmt = insert(Agent).values(**insert_values)

        # On conflict by agent_id, update definition and metadata
        stmt = stmt.on_conflict_do_update(
            index_elements=["agent_id"],
            set_={
                "name": name,
                "definition": agent_data,
                "source_workflow_id": source_wf_uuid,
                "is_deleted": False  # Reactivate if was deleted
            }
        ).returning(Agent)

        result = await db.execute(stmt)
        await db.flush()

        agent = result.scalar_one()
        await db.refresh(agent)
        return agent

    async def get_by_name(
        self,
        db: AsyncSession,
        user_id: str,
        name: str
    ) -> Optional[Agent]:
        """
        Get an agent by name for a user.

        Agent names are unique per user (enforced by unique constraint).

        Args:
            db: Async database session
            user_id: Owner user ID
            name: Agent name to look up

        Returns:
            Agent instance if found, None otherwise
        """
        user_uuid = UUID(user_id) if isinstance(user_id, str) else user_id

        stmt = select(Agent).where(
            Agent.user_id == user_uuid,
            Agent.name == name,
            Agent.is_deleted == False
        )

        result = await db.execute(stmt)
        return result.scalar_one_or_none()

    async def search(
        self,
        db: AsyncSession,
        user_id: str,
        query: str,
        limit: int = 20
    ) -> List[Agent]:
        """
        Search agents by name using case-insensitive LIKE.

        Args:
            db: Async database session
            user_id: Owner user ID
            query: Search query string
            limit: Maximum results to return

        Returns:
            List of matching Agent instances
        """
        user_uuid = UUID(user_id) if isinstance(user_id, str) else user_id

        # Use ILIKE for case-insensitive search
        search_pattern = f"%{query}%"

        stmt = (
            select(Agent)
            .where(
                Agent.user_id == user_uuid,
                Agent.name.ilike(search_pattern),
                Agent.is_deleted == False
            )
            .order_by(Agent.name)
            .limit(limit)
        )

        result = await db.execute(stmt)
        return list(result.scalars().all())

    async def list_by_workflow(
        self,
        db: AsyncSession,
        user_id: str,
        workflow_id: str
    ) -> List[Agent]:
        """
        List all agents that were synced from a specific workflow.

        Args:
            db: Async database session
            user_id: Owner user ID
            workflow_id: Source workflow ID

        Returns:
            List of Agent instances from the workflow
        """
        user_uuid = UUID(user_id) if isinstance(user_id, str) else user_id
        workflow_uuid = UUID(workflow_id) if isinstance(workflow_id, str) else workflow_id

        stmt = (
            select(Agent)
            .where(
                Agent.user_id == user_uuid,
                Agent.source_workflow_id == workflow_uuid,
                Agent.is_deleted == False
            )
            .order_by(Agent.name)
        )

        result = await db.execute(stmt)
        return list(result.scalars().all())

    async def bulk_upsert(
        self,
        db: AsyncSession,
        user_id: str,
        agents_data: List[Dict[str, Any]],
        source_workflow_id: Optional[str] = None
    ) -> List[Agent]:
        """
        Upsert multiple agents in a single transaction.

        Used when saving workflows with multiple embedded agents.

        Args:
            db: Async database session
            user_id: Owner user ID
            agents_data: List of agent definition dicts
            source_workflow_id: Optional workflow ID that triggered the sync

        Returns:
            List of upserted Agent instances
        """
        results = []
        for agent_data in agents_data:
            agent = await self.upsert(
                db, user_id, agent_data, source_workflow_id
            )
            results.append(agent)

        return results
