"""
EchoAI Application Repository

Async data access for the Application Orchestrator module.
Provides CRUD operations for applications and sync helpers for
all association (link) tables.

All queries enforce user_id scoping for multi-tenancy and exclude
soft-deleted records (is_deleted = TRUE).

Pattern follows existing BaseRepository / SessionRepository conventions:
    - All methods accept ``db: AsyncSession`` as the first argument
    - UUID strings are validated via ``safe_uuid()`` before use
    - ``session.flush()`` is called after mutations; the caller's
      context manager (``get_db_session``) handles commit/rollback
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from echolib.models import (
    Application,
    ApplicationBusinessUnitLink,
    ApplicationDataSourceLink,
    ApplicationDesignationLink,
    ApplicationGuardrailLink,
    ApplicationLlmLink,
    ApplicationSkillLink,
    ApplicationTagLink,
)
from .base import safe_uuid


logger = logging.getLogger(__name__)


class ApplicationRepository:
    """
    Repository for Application entity operations.

    Provides:
        - Standard CRUD (create, get, list, update, soft-delete)
        - Publish / unpublish lifecycle transitions
        - Dashboard statistics (total, draft, published, error counts)
        - Link-table synchronisation for LLMs, skills, data sources,
          designations, business units, tags, and guardrail categories
        - Config JSONB dual-write (rebuild consolidated config after link changes)
    """

    # ------------------------------------------------------------------
    # CONFIG JSONB HELPERS
    # ------------------------------------------------------------------

    @staticmethod
    def _build_config_from_links(app: Application) -> dict:
        """
        Build config JSONB dict from existing link table relationships.

        This reads the eagerly-loaded relationship collections on the
        Application instance and produces a single dict suitable for
        storage in the ``config`` JSONB column.

        Args:
            app: Application instance with relationships loaded.

        Returns:
            Dict matching the config JSONB schema.
        """
        config = {
            "llm_bindings": [
                {"llm_id": link.llm_id, "role": link.role, "name": link.name}
                for link in (app.llm_links or [])
            ],
            "skills": [
                {
                    "skill_id": link.skill_id,
                    "skill_type": link.skill_type,
                    "name": link.skill_name,
                    "description": link.skill_description,
                }
                for link in (app.skill_links or [])
            ],
            "data_source_ids": [
                link.data_source_id for link in (app.data_source_links or [])
            ],
            "access_control": {
                "designation_ids": [
                    str(link.designation_id)
                    for link in (app.designation_links or [])
                ],
                "business_unit_ids": [
                    str(link.business_unit_id)
                    for link in (app.business_unit_links or [])
                ],
                "tag_ids": [
                    str(link.tag_id) for link in (app.tag_links or [])
                ],
            },
            "guardrails": {
                "category_ids": [
                    str(link.guardrail_category_id)
                    for link in (app.guardrail_links or [])
                ],
                "custom_rules": None,
            },
            "persona": {
                "persona_id": str(app.persona_id) if app.persona_id else None,
                "custom_text": app.persona_text,
            },
        }
        return config

    async def sync_config_jsonb(
        self,
        db: AsyncSession,
        app_id: str,
        user_id: str,
    ) -> None:
        """
        Rebuild config JSONB from current link table data.

        Call this after any link sync operation to keep the consolidated
        JSONB column in sync with the relational link tables (dual-write).

        Args:
            db: Async database session.
            app_id: Application primary key.
            user_id: Owner user ID for scoping.
        """
        app = await self.get_application(db, app_id, user_id)
        if app is None:
            return
        app.config = self._build_config_from_links(app)
        await db.flush()

    # ------------------------------------------------------------------
    # CREATE
    # ------------------------------------------------------------------

    async def create_application(
        self,
        db: AsyncSession,
        user_id: str,
        name: str,
        **kwargs: Any,
    ) -> Application:
        """
        Create a new application in draft status.

        Args:
            db: Async database session.
            user_id: Owner user ID (FK to users table).
            name: Application name (only required field for draft).
            **kwargs: Optional fields (description, welcome_prompt, etc.).

        Returns:
            The newly created Application instance.

        Raises:
            ValueError: If user_id is not a valid UUID.
        """
        user_uuid = safe_uuid(user_id)
        if user_uuid is None:
            raise ValueError(f"Invalid user_id: {user_id}")

        # Ensure persona_id is converted to UUID if provided
        if "persona_id" in kwargs and kwargs["persona_id"] is not None:
            kwargs["persona_id"] = safe_uuid(kwargs["persona_id"])

        app = Application(
            user_id=user_uuid,
            name=name,
            status="draft",
            config={},
            **kwargs,
        )
        db.add(app)
        await db.flush()
        await db.refresh(app)
        return app

    # ------------------------------------------------------------------
    # GET (single)
    # ------------------------------------------------------------------

    async def get_application(
        self,
        db: AsyncSession,
        application_id: str,
        user_id: str,
    ) -> Optional[Application]:
        """
        Get an application by ID, scoped to user, excluding deleted.

        The Application model's relationships use ``lazy='selectin'`` so all
        link-table data is loaded automatically.

        Args:
            db: Async database session.
            application_id: Application primary key.
            user_id: Owner user ID for scoping.

        Returns:
            Application instance if found and owned by user, None otherwise.
        """
        app_uuid = safe_uuid(application_id)
        user_uuid = safe_uuid(user_id)
        if app_uuid is None or user_uuid is None:
            return None

        stmt = (
            select(Application)
            .where(
                Application.application_id == app_uuid,
                Application.user_id == user_uuid,
                Application.is_deleted == False,  # noqa: E712
            )
        )
        result = await db.execute(stmt)
        return result.scalar_one_or_none()

    # ------------------------------------------------------------------
    # LIST (paginated with filters)
    # ------------------------------------------------------------------

    async def list_applications(
        self,
        db: AsyncSession,
        user_id: str,
        status: Optional[str] = None,
        q: Optional[str] = None,
        sort: Optional[str] = None,
        page: int = 1,
        page_size: int = 20,
    ) -> Tuple[List[Application], int]:
        """
        List applications for a user with optional filters and pagination.

        Args:
            db: Async database session.
            user_id: Owner user ID.
            status: Filter by status (draft / published / error).
            q: Search term (ILIKE on name).
            sort: Sort mode — ``'recently_updated'`` orders by updated_at DESC.
            page: Page number (1-based).
            page_size: Items per page.

        Returns:
            Tuple of (list of Application instances, total count).
        """
        user_uuid = safe_uuid(user_id)
        if user_uuid is None:
            return [], 0

        # Base filter
        base_filter = [
            Application.user_id == user_uuid,
            Application.is_deleted == False,  # noqa: E712
        ]

        if status:
            base_filter.append(Application.status == status)

        if q:
            base_filter.append(Application.name.ilike(f"%{q}%"))

        # Total count
        count_stmt = select(func.count()).select_from(Application).where(*base_filter)
        total = (await db.execute(count_stmt)).scalar() or 0

        # Items query
        items_stmt = select(Application).where(*base_filter)

        if sort == "recently_updated":
            items_stmt = items_stmt.order_by(Application.updated_at.desc())
        else:
            items_stmt = items_stmt.order_by(Application.created_at.desc())

        offset = (page - 1) * page_size
        items_stmt = items_stmt.offset(offset).limit(page_size)

        result = await db.execute(items_stmt)
        items = list(result.scalars().all())

        return items, total

    # ------------------------------------------------------------------
    # UPDATE
    # ------------------------------------------------------------------

    async def update_application(
        self,
        db: AsyncSession,
        application_id: str,
        user_id: str,
        **kwargs: Any,
    ) -> Optional[Application]:
        """
        Update an application's scalar fields.

        Only provided keyword arguments are written; unset fields are
        left untouched.  The ``updated_at`` column is auto-bumped by
        the database trigger.

        Args:
            db: Async database session.
            application_id: Application primary key.
            user_id: Owner user ID for scoping.
            **kwargs: Fields to update (name, description, status, etc.).

        Returns:
            Updated Application instance, or None if not found.
        """
        app_uuid = safe_uuid(application_id)
        user_uuid = safe_uuid(user_id)
        if app_uuid is None or user_uuid is None:
            return None

        if not kwargs:
            return await self.get_application(db, application_id, user_id)

        # Convert persona_id to UUID if present
        if "persona_id" in kwargs and kwargs["persona_id"] is not None:
            kwargs["persona_id"] = safe_uuid(kwargs["persona_id"])

        stmt = (
            update(Application)
            .where(
                Application.application_id == app_uuid,
                Application.user_id == user_uuid,
                Application.is_deleted == False,  # noqa: E712
            )
            .values(**kwargs)
            .returning(Application)
        )

        result = await db.execute(stmt)
        await db.flush()

        updated = result.scalar_one_or_none()
        if updated:
            await db.refresh(updated)
        return updated

    # ------------------------------------------------------------------
    # SOFT DELETE
    # ------------------------------------------------------------------

    async def soft_delete_application(
        self,
        db: AsyncSession,
        application_id: str,
        user_id: str,
    ) -> bool:
        """
        Soft-delete an application (set is_deleted = TRUE).

        Args:
            db: Async database session.
            application_id: Application primary key.
            user_id: Owner user ID for scoping.

        Returns:
            True if a record was marked deleted, False if not found.
        """
        app_uuid = safe_uuid(application_id)
        user_uuid = safe_uuid(user_id)
        if app_uuid is None or user_uuid is None:
            return False

        stmt = (
            update(Application)
            .where(
                Application.application_id == app_uuid,
                Application.user_id == user_uuid,
                Application.is_deleted == False,  # noqa: E712
            )
            .values(is_deleted=True)
        )

        result = await db.execute(stmt)
        await db.flush()
        return result.rowcount > 0

    # ------------------------------------------------------------------
    # PUBLISH / UNPUBLISH
    # ------------------------------------------------------------------

    async def publish_application(
        self,
        db: AsyncSession,
        application_id: str,
        user_id: str,
    ) -> Optional[Application]:
        """
        Set application status to 'published'.

        Validation of required fields (LLMs, skills, welcome_prompt, etc.)
        is performed at the route layer before calling this method.

        Args:
            db: Async database session.
            application_id: Application primary key.
            user_id: Owner user ID for scoping.

        Returns:
            Updated Application, or None if not found.
        """
        return await self.update_application(
            db, application_id, user_id, status="published", error_message=None
        )

    async def unpublish_application(
        self,
        db: AsyncSession,
        application_id: str,
        user_id: str,
    ) -> Optional[Application]:
        """
        Revert application status to 'draft'.

        Args:
            db: Async database session.
            application_id: Application primary key.
            user_id: Owner user ID for scoping.

        Returns:
            Updated Application, or None if not found.
        """
        return await self.update_application(
            db, application_id, user_id, status="draft"
        )

    # ------------------------------------------------------------------
    # STATS
    # ------------------------------------------------------------------

    async def get_application_stats(
        self,
        db: AsyncSession,
        user_id: str,
    ) -> Dict[str, int]:
        """
        Count applications by status for a user.

        Args:
            db: Async database session.
            user_id: Owner user ID.

        Returns:
            Dict with keys: total, published, draft, errors.
        """
        user_uuid = safe_uuid(user_id)
        if user_uuid is None:
            return {"total": 0, "published": 0, "draft": 0, "errors": 0}

        base_filter = [
            Application.user_id == user_uuid,
            Application.is_deleted == False,  # noqa: E712
        ]

        # Single query with conditional aggregation
        stmt = select(
            func.count().label("total"),
            func.count().filter(Application.status == "published").label("published"),
            func.count().filter(Application.status == "draft").label("draft"),
            func.count().filter(Application.status == "error").label("errors"),
        ).where(*base_filter)

        row = (await db.execute(stmt)).one()

        return {
            "total": row.total,
            "published": row.published,
            "draft": row.draft,
            "errors": row.errors,
        }

    # ------------------------------------------------------------------
    # LINK TABLE SYNC HELPERS
    # ------------------------------------------------------------------
    # Pattern: delete all existing links for the application, then insert
    # the new set.  This is idempotent and avoids diff logic.
    # ------------------------------------------------------------------

    async def sync_llm_links(
        self,
        db: AsyncSession,
        application_id: str,
        llm_links: List[Dict[str, Any]],
    ) -> None:
        """
        Replace all LLM links for an application.

        Args:
            db: Async database session.
            application_id: Application primary key.
            llm_links: List of dicts, each with keys 'llm_id' and optional 'role'.
                       Example: [{"llm_id": "...", "role": "orchestrator"}]
        """
        app_uuid = safe_uuid(application_id)
        if app_uuid is None:
            return

        await db.execute(
            delete(ApplicationLlmLink).where(
                ApplicationLlmLink.application_id == app_uuid
            )
        )

        new_links = []
        for entry in llm_links:
            llm_id_str = str(entry.get("llm_id") or entry.get("id") or "").strip()
            if not llm_id_str:
                continue
            new_links.append(
                ApplicationLlmLink(
                    application_id=app_uuid,
                    llm_id=llm_id_str,
                    role=entry.get("role", "general"),
                    name=entry.get("name"),
                )
            )
        if new_links:
            db.add_all(new_links)
            await db.flush()

    async def sync_skill_links(
        self,
        db: AsyncSession,
        application_id: str,
        skill_links: List[Dict[str, Any]],
    ) -> None:
        """
        Replace all skill links for an application.

        Args:
            db: Async database session.
            application_id: Application primary key.
            skill_links: List of dicts with keys:
                         skill_id, skill_type, skill_name (opt), skill_description (opt).
        """
        app_uuid = safe_uuid(application_id)
        if app_uuid is None:
            return

        await db.execute(
            delete(ApplicationSkillLink).where(
                ApplicationSkillLink.application_id == app_uuid
            )
        )

        new_links = []
        for entry in skill_links:
            skill_id = entry.get("skill_id") or entry.get("id")
            skill_type = entry.get("skill_type")
            skill_name = entry.get("skill_name") or entry.get("name") or ""

            # If skill_type not provided, parse from name prefix ("workflow: Name")
            if not skill_type and ": " in skill_name:
                prefix, rest = skill_name.split(": ", 1)
                if prefix.lower() in ("workflow", "agent"):
                    skill_type = prefix.lower()
                    skill_name = rest

            if not skill_id or not skill_type:
                continue
            new_links.append(
                ApplicationSkillLink(
                    application_id=app_uuid,
                    skill_type=skill_type,
                    skill_id=str(skill_id),
                    skill_name=skill_name or None,
                    skill_description=entry.get("skill_description") or entry.get("description"),
                )
            )
        if new_links:
            db.add_all(new_links)
            await db.flush()

    async def sync_data_source_links(
        self,
        db: AsyncSession,
        application_id: str,
        data_source_ids: List[str],
    ) -> None:
        """
        Replace all data source links for an application.

        Args:
            db: Async database session.
            application_id: Application primary key.
            data_source_ids: List of data_source_id strings.
        """
        app_uuid = safe_uuid(application_id)
        if app_uuid is None:
            return

        await db.execute(
            delete(ApplicationDataSourceLink).where(
                ApplicationDataSourceLink.application_id == app_uuid
            )
        )

        new_links = [
            ApplicationDataSourceLink(
                application_id=app_uuid,
                data_source_id=str(ds_id),
            )
            for ds_id in data_source_ids
            if ds_id
        ]
        if new_links:
            db.add_all(new_links)
            await db.flush()

    async def sync_designation_links(
        self,
        db: AsyncSession,
        application_id: str,
        designation_ids: List[str],
    ) -> None:
        """
        Replace all designation links for an application.

        Args:
            db: Async database session.
            application_id: Application primary key.
            designation_ids: List of designation UUID strings.
        """
        app_uuid = safe_uuid(application_id)
        if app_uuid is None:
            return

        await db.execute(
            delete(ApplicationDesignationLink).where(
                ApplicationDesignationLink.application_id == app_uuid
            )
        )

        new_links = []
        for did in designation_ids:
            d_uuid = safe_uuid(did)
            if d_uuid is None:
                continue
            new_links.append(
                ApplicationDesignationLink(
                    application_id=app_uuid,
                    designation_id=d_uuid,
                )
            )
        if new_links:
            db.add_all(new_links)
            await db.flush()

    async def sync_business_unit_links(
        self,
        db: AsyncSession,
        application_id: str,
        business_unit_ids: List[str],
    ) -> None:
        """
        Replace all business unit links for an application.

        Args:
            db: Async database session.
            application_id: Application primary key.
            business_unit_ids: List of business_unit UUID strings.
        """
        app_uuid = safe_uuid(application_id)
        if app_uuid is None:
            return

        await db.execute(
            delete(ApplicationBusinessUnitLink).where(
                ApplicationBusinessUnitLink.application_id == app_uuid
            )
        )

        new_links = []
        for bu_id in business_unit_ids:
            bu_uuid = safe_uuid(bu_id)
            if bu_uuid is None:
                continue
            new_links.append(
                ApplicationBusinessUnitLink(
                    application_id=app_uuid,
                    business_unit_id=bu_uuid,
                )
            )
        if new_links:
            db.add_all(new_links)
            await db.flush()

    async def sync_tag_links(
        self,
        db: AsyncSession,
        application_id: str,
        tag_ids: List[str],
    ) -> None:
        """
        Replace all tag links for an application.

        Args:
            db: Async database session.
            application_id: Application primary key.
            tag_ids: List of tag UUID strings.
        """
        app_uuid = safe_uuid(application_id)
        if app_uuid is None:
            return

        await db.execute(
            delete(ApplicationTagLink).where(
                ApplicationTagLink.application_id == app_uuid
            )
        )

        new_links = []
        for tid in tag_ids:
            t_uuid = safe_uuid(tid)
            if t_uuid is None:
                continue
            new_links.append(
                ApplicationTagLink(
                    application_id=app_uuid,
                    tag_id=t_uuid,
                )
            )
        if new_links:
            db.add_all(new_links)
            await db.flush()

    async def sync_guardrail_links(
        self,
        db: AsyncSession,
        application_id: str,
        guardrail_category_ids: List[str],
    ) -> None:
        """
        Replace all guardrail category links for an application.

        Args:
            db: Async database session.
            application_id: Application primary key.
            guardrail_category_ids: List of guardrail_category UUID strings.
        """
        app_uuid = safe_uuid(application_id)
        if app_uuid is None:
            return

        await db.execute(
            delete(ApplicationGuardrailLink).where(
                ApplicationGuardrailLink.application_id == app_uuid
            )
        )

        new_links = []
        for gc_id in guardrail_category_ids:
            gc_uuid = safe_uuid(gc_id)
            if gc_uuid is None:
                continue
            new_links.append(
                ApplicationGuardrailLink(
                    application_id=app_uuid,
                    guardrail_category_id=gc_uuid,
                )
            )
        if new_links:
            db.add_all(new_links)
            await db.flush()
