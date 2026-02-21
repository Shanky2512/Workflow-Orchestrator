"""
EchoAI Orchestrator -- Skill Manifest Builder

Queries PostgreSQL to build a structured manifest of all skills
(workflows + agents) linked to an application.  The manifest is
included in the orchestrator LLM system prompt so it can decide
which skills to invoke.

Data Sources:
    - application_skill_links  (which skills are bound to this app)
    - workflows table          (definition JSONB for rich metadata)
    - agents table             (definition JSONB for rich metadata)

Caching:
    Simple in-memory dict per application_id with a TTL.
    Invalidated when the caller signals a setup change.
"""

import logging
import time
from typing import Any, Dict, List, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

# Default TTL for cached manifests (seconds)
_DEFAULT_CACHE_TTL = 300  # 5 minutes


class SkillManifestBuilder:
    """
    Builds a structured skill manifest from PostgreSQL for inclusion
    in the orchestrator's system prompt.
    """

    def __init__(self, cache_ttl: int = _DEFAULT_CACHE_TTL) -> None:
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._cache_timestamps: Dict[str, float] = {}
        self._cache_ttl = cache_ttl

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def build_manifest(
        self,
        db: AsyncSession,
        application_id: str,
    ) -> Dict[str, Any]:
        """
        Build (or return cached) skill manifest for an application.

        Args:
            db: Async database session.
            application_id: Application UUID string.

        Returns:
            Dict with key ``skills`` containing a list of skill metadata dicts.
        """
        # Check cache
        cached = self._get_cached(application_id)
        if cached is not None:
            return cached

        manifest = await self._build_from_db(db, application_id)
        self._set_cached(application_id, manifest)
        return manifest

    def invalidate(self, application_id: str) -> None:
        """
        Invalidate cached manifest for an application.

        Should be called whenever the application's skill links change
        (e.g. after a setup PATCH).
        """
        self._cache.pop(application_id, None)
        self._cache_timestamps.pop(application_id, None)

    def invalidate_all(self) -> None:
        """Clear entire manifest cache."""
        self._cache.clear()
        self._cache_timestamps.clear()

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _get_cached(self, application_id: str) -> Optional[Dict[str, Any]]:
        ts = self._cache_timestamps.get(application_id)
        if ts is None:
            return None
        if (time.monotonic() - ts) > self._cache_ttl:
            self.invalidate(application_id)
            return None
        return self._cache.get(application_id)

    def _set_cached(self, application_id: str, manifest: Dict[str, Any]) -> None:
        self._cache[application_id] = manifest
        self._cache_timestamps[application_id] = time.monotonic()

    # ------------------------------------------------------------------
    # Database query
    # ------------------------------------------------------------------

    async def _build_from_db(
        self,
        db: AsyncSession,
        application_id: str,
    ) -> Dict[str, Any]:
        """
        Query skill links and join with workflows/agents tables for
        rich metadata.

        Deduplication:
            Agents that are already embedded inside a linked workflow's
            definition are excluded from the standalone agent list.  This
            prevents the orchestrator LLM from seeing (and potentially
            invoking) the same agent twice -- once via the workflow and
            once as a standalone skill.
        """
        from echolib.models.application import ApplicationSkillLink
        from echolib.models.workflow import Workflow
        from echolib.models.agent import Agent
        from echolib.repositories.base import safe_uuid

        app_uuid = safe_uuid(application_id)
        if app_uuid is None:
            return {"skills": []}

        # Fetch skill links for this application
        stmt = select(ApplicationSkillLink).where(
            ApplicationSkillLink.application_id == app_uuid
        )
        result = await db.execute(stmt)
        links = list(result.scalars().all())

        if not links:
            return {"skills": []}

        # Partition links by type
        workflow_ids: List[str] = []
        agent_ids: List[str] = []
        link_map: Dict[str, ApplicationSkillLink] = {}

        for link in links:
            link_map[f"{link.skill_type}:{link.skill_id}"] = link
            if link.skill_type == "workflow":
                workflow_ids.append(link.skill_id)
            elif link.skill_type == "agent":
                agent_ids.append(link.skill_id)

        skills: List[Dict[str, Any]] = []

        # ----------------------------------------------------------
        # Phase 1: Fetch workflow metadata and collect embedded agents
        # ----------------------------------------------------------
        # Set of agent IDs that are already embedded inside at least
        # one linked workflow.  These will be excluded from the
        # standalone agent list to prevent double-execution.
        embedded_agent_ids: set = set()

        if workflow_ids:
            wf_uuids = [safe_uuid(wid) for wid in workflow_ids]
            wf_uuids = [u for u in wf_uuids if u is not None]
            if wf_uuids:
                wf_stmt = select(Workflow).where(
                    Workflow.workflow_id.in_(wf_uuids),
                    Workflow.is_deleted == False,  # noqa: E712
                )
                wf_result = await db.execute(wf_stmt)
                workflows = list(wf_result.scalars().all())

                for wf in workflows:
                    link_key = f"workflow:{str(wf.workflow_id)}"
                    link = link_map.get(link_key)
                    defn = wf.definition or {}
                    skills.append(
                        self._build_workflow_entry(wf, defn, link)
                    )

                    # Extract agent IDs embedded in this workflow
                    if isinstance(defn, dict):
                        for embedded_agent in defn.get("agents", []):
                            if isinstance(embedded_agent, dict):
                                aid = embedded_agent.get("agent_id")
                                if aid:
                                    embedded_agent_ids.add(str(aid))

        # ----------------------------------------------------------
        # Phase 2: Fetch agent metadata (excluding embedded agents)
        # ----------------------------------------------------------
        if agent_ids:
            agt_uuids = [safe_uuid(aid) for aid in agent_ids]
            agt_uuids = [u for u in agt_uuids if u is not None]
            if agt_uuids:
                agt_stmt = select(Agent).where(
                    Agent.agent_id.in_(agt_uuids),
                    Agent.is_deleted == False,  # noqa: E712
                )
                agt_result = await db.execute(agt_stmt)
                agents = list(agt_result.scalars().all())

                for agt in agents:
                    agent_id_str = str(agt.agent_id)

                    # Skip agents already embedded in a linked workflow
                    if agent_id_str in embedded_agent_ids:
                        link_key = f"agent:{agent_id_str}"
                        link = link_map.get(link_key)
                        logger.info(
                            "Skipping agent '%s' (id=%s) from manifest -- "
                            "already embedded in a linked workflow",
                            link.skill_name if link else agt.name,
                            agent_id_str,
                        )
                        continue

                    link_key = f"agent:{agent_id_str}"
                    link = link_map.get(link_key)
                    defn = agt.definition or {}
                    skills.append(
                        self._build_agent_entry(agt, defn, link)
                    )

        # Log deduplication summary when agents were removed
        if embedded_agent_ids:
            linked_agent_id_strs = {str(aid) for aid in agent_ids}
            excluded_count = len(embedded_agent_ids & linked_agent_id_strs)
            if excluded_count:
                logger.info(
                    "Agent deduplication: %d agent(s) embedded in workflows, "
                    "%d excluded from standalone manifest",
                    len(embedded_agent_ids),
                    excluded_count,
                )

        # Also add skills from links that had no matching DB record
        # (use denormalized data from the link itself).
        # Exclude agent links whose IDs are embedded in a workflow.
        existing_keys = {f"{s['skill_type']}:{s['skill_id']}" for s in skills}
        for key, link in link_map.items():
            if key not in existing_keys:
                # Skip fallback agent entries that are embedded in workflows
                if (
                    link.skill_type == "agent"
                    and str(link.skill_id) in embedded_agent_ids
                ):
                    continue
                skills.append({
                    "skill_id": link.skill_id,
                    "skill_type": link.skill_type,
                    "name": link.skill_name or link.skill_id,
                    "description": link.skill_description or "",
                    "capabilities": [],
                })

        return {"skills": skills}

    # ------------------------------------------------------------------
    # Entry builders
    # ------------------------------------------------------------------

    @staticmethod
    def _build_workflow_entry(
        wf, defn: Dict[str, Any], link
    ) -> Dict[str, Any]:
        """Build a skill manifest entry for a workflow."""
        name = defn.get("name") or wf.name
        description = defn.get("description", "")
        if link and link.skill_description:
            description = link.skill_description

        execution_model = defn.get("execution_model", "sequential")

        # Extract capabilities from embedded agents
        capabilities: List[str] = []
        for agent_def in defn.get("agents", []):
            if isinstance(agent_def, dict):
                role = agent_def.get("role", "")
                if role:
                    capabilities.append(role)

        return {
            "skill_id": str(wf.workflow_id),
            "skill_type": "workflow",
            "name": name,
            "description": description,
            "capabilities": capabilities,
            "execution_model": execution_model,
        }

    @staticmethod
    def _build_agent_entry(
        agt, defn: Dict[str, Any], link
    ) -> Dict[str, Any]:
        """Build a skill manifest entry for an agent."""
        name = defn.get("name") or agt.name
        description = defn.get("description", "")
        if link and link.skill_description:
            description = link.skill_description

        role = defn.get("role", "")
        goal = defn.get("goal", "")
        tools_list = defn.get("tools", [])
        tool_names = []
        for t in tools_list:
            if isinstance(t, str):
                tool_names.append(t)
            elif isinstance(t, dict):
                tool_names.append(t.get("name", str(t)))

        # Build capabilities from role and goal
        capabilities: List[str] = []
        if role:
            capabilities.append(role)
        if goal and goal != role:
            capabilities.append(goal)

        return {
            "skill_id": str(agt.agent_id),
            "skill_type": "agent",
            "name": name,
            "description": description,
            "capabilities": capabilities,
            "tools": tool_names,
        }
