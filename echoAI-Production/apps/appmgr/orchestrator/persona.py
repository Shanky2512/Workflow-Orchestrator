"""
EchoAI Orchestrator -- Persona Formatter

Builds the persona portion of the orchestrator system prompt.

Priority order:
    1. If persona_text is set --> use it verbatim
    2. If persona_id is set  --> load persona name from DB, then:
       a. If a domain-specific template exists for that name, build a rich prompt
       b. Otherwise, fall back to a generic one-liner
    3. If neither             --> return empty string (no persona injection)
"""

import logging
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from echolib.persona_templates import (
    get_persona_key_by_name,
    get_persona_template,
    build_rich_prompt,
)

logger = logging.getLogger(__name__)


class PersonaFormatter:
    """
    Builds a persona instruction string for inclusion in the
    orchestrator's system prompt.
    """

    async def build_persona_prompt(
        self,
        db: AsyncSession,
        persona_id: Optional[str] = None,
        persona_text: Optional[str] = None,
    ) -> str:
        """
        Build the persona prompt string.

        Args:
            db: Async database session.
            persona_id: UUID string referencing app_personas table.
            persona_text: Custom persona text (takes precedence over persona_id).

        Returns:
            Persona instruction string, or empty string if no persona is configured.
        """
        # persona_text takes precedence
        if persona_text:
            return persona_text.strip()

        if persona_id:
            return await self._load_persona_from_db(db, persona_id)

        return ""

    @staticmethod
    async def _load_persona_from_db(
        db: AsyncSession, persona_id: str
    ) -> str:
        """
        Load persona name from the app_personas table and generate
        a default persona instruction.

        Args:
            db: Async database session.
            persona_id: UUID string for the persona record.

        Returns:
            Generated persona instruction, or empty string if not found.
        """
        from echolib.models.application_lookups import AppPersona
        from echolib.repositories.base import safe_uuid

        pid = safe_uuid(persona_id)
        if pid is None:
            logger.warning("Invalid persona_id UUID: %s", persona_id)
            return ""

        stmt = select(AppPersona).where(AppPersona.persona_id == pid)
        result = await db.execute(stmt)
        persona = result.scalar_one_or_none()

        if persona is None:
            logger.warning("Persona not found for id: %s", persona_id)
            return ""

        name = persona.name

        # Try to load a rich domain-specific template for this persona
        persona_key = get_persona_key_by_name(name)
        if persona_key:
            template = get_persona_template(persona_key)
            if template:
                logger.info(
                    "Using rich persona template '%s' for persona '%s'",
                    persona_key, name,
                )
                return build_rich_prompt(template)

        # Fallback: generic one-liner if no template matches
        logger.debug(
            "No persona template found for '%s'; using generic prompt", name
        )
        return (
            f"You are a {name}. "
            f"Respond in the style and tone of a {name}. "
            f"Maintain this persona consistently throughout the conversation."
        )
