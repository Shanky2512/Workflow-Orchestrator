"""
EchoAI Persona Template Loader

Loads domain-specific persona templates from JSON files in this directory,
caches them in memory, and provides lookup utilities for the PersonaFormatter.

Templates are JSON objects keyed by ``persona_key`` (e.g. "hr_assistant")
and contain rich prompt context: system prompt, domain expertise, tone/style
guidelines, response format instructions, and guardrail hints.

Public API:
    get_persona_template(persona_key)   -> dict | None
    get_all_persona_keys()              -> list[str]
    get_persona_key_by_name(name)       -> str | None
    build_rich_prompt(template)         -> str
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level cache
# ---------------------------------------------------------------------------

_TEMPLATE_DIR = Path(__file__).resolve().parent

# Lazy-loaded caches -- populated on first access
_templates_cache: Optional[Dict[str, dict]] = None
_display_name_to_key: Optional[Dict[str, str]] = None


def _load_all_templates() -> None:
    """
    Scan the template directory for JSON files, parse them, and populate
    the module-level caches.

    This is called lazily on the first public API call and never re-runs
    unless ``reload_templates()`` is invoked.
    """
    global _templates_cache, _display_name_to_key

    _templates_cache = {}
    _display_name_to_key = {}

    for json_path in _TEMPLATE_DIR.glob("*.json"):
        try:
            with open(json_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)

            persona_key = data.get("persona_key")
            if not persona_key:
                logger.warning(
                    "Persona template %s missing 'persona_key' -- skipped",
                    json_path.name,
                )
                continue

            _templates_cache[persona_key] = data

            display_name = data.get("display_name", "")
            if display_name:
                # Store both the original display name and a normalized form
                _display_name_to_key[display_name] = persona_key
                _display_name_to_key[_normalize_name(display_name)] = persona_key

            logger.debug(
                "Loaded persona template: %s (display_name=%s)",
                persona_key,
                display_name,
            )

        except json.JSONDecodeError as exc:
            logger.error(
                "Invalid JSON in persona template %s: %s",
                json_path.name,
                exc,
            )
        except OSError as exc:
            logger.error(
                "Failed to read persona template %s: %s",
                json_path.name,
                exc,
            )

    logger.info(
        "Persona templates loaded: %d template(s) from %s",
        len(_templates_cache),
        _TEMPLATE_DIR,
    )


def _ensure_loaded() -> None:
    """Trigger lazy loading if caches are not yet populated."""
    if _templates_cache is None:
        _load_all_templates()


def _normalize_name(name: str) -> str:
    """
    Normalize a display name to a persona_key-style string.

    "HR Assistant"  -> "hr_assistant"
    "Sales Assistant" -> "sales_assistant"
    "Support Agent"   -> "support_agent"
    """
    # Lowercase, replace non-alphanumeric runs with underscores, strip edges
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_persona_template(persona_key: str) -> Optional[dict]:
    """
    Retrieve a persona template by its persona_key.

    Args:
        persona_key: The template identifier (e.g. "hr_assistant").

    Returns:
        The full template dict if found, or None.
    """
    _ensure_loaded()
    assert _templates_cache is not None  # satisfy type checker post-load
    return _templates_cache.get(persona_key)


def get_all_persona_keys() -> List[str]:
    """
    Return a sorted list of all available persona template keys.

    Returns:
        List of persona_key strings.
    """
    _ensure_loaded()
    assert _templates_cache is not None
    return sorted(_templates_cache.keys())


def get_persona_key_by_name(name: str) -> Optional[str]:
    """
    Map a display name to its corresponding persona_key.

    Performs an exact match first, then tries a normalized match.
    For example, "HR Assistant" matches "hr_assistant".

    Args:
        name: The persona display name (e.g. "HR Assistant").

    Returns:
        The persona_key string if found, or None.
    """
    _ensure_loaded()
    assert _display_name_to_key is not None

    # Exact display name match
    key = _display_name_to_key.get(name)
    if key:
        return key

    # Normalized match
    normalized = _normalize_name(name)
    return _display_name_to_key.get(normalized)


def build_rich_prompt(template: dict) -> str:
    """
    Construct a rich system prompt string from a persona template dict.

    The generated prompt includes:
        - Core system prompt
        - Domain expertise and knowledge base
        - Communication style guidelines (do/don't)
        - Response format instructions
        - Safety guardrail hints

    Args:
        template: A persona template dict (as loaded from JSON).

    Returns:
        Formatted multi-section prompt string ready for LLM injection.
    """
    sections: List[str] = []

    # 1. Core system prompt
    system_prompt = template.get("system_prompt", "")
    if system_prompt:
        sections.append(system_prompt)

    # 2. Domain expertise
    domain = template.get("domain_context", {})
    knowledge_base = domain.get("knowledge_base", "")
    if knowledge_base:
        sections.append(f"## Domain Expertise\n{knowledge_base}")

    # 3. Communication style
    tone = template.get("tone_and_style", {})
    style = tone.get("communication_style", "")
    do_list = tone.get("do", [])
    dont_list = tone.get("dont", [])

    style_parts: List[str] = []
    if style:
        style_parts.append(style)
    if do_list:
        do_formatted = "\n".join(f"- {item}" for item in do_list)
        style_parts.append(f"Do:\n{do_formatted}")
    if dont_list:
        dont_formatted = "\n".join(f"- {item}" for item in dont_list)
        style_parts.append(f"Don't:\n{dont_formatted}")

    if style_parts:
        sections.append("## Communication Style\n" + "\n\n".join(style_parts))

    # 4. Response format
    response_tmpl = template.get("response_template", {})
    format_instruction = response_tmpl.get("output_format_instruction", "")
    if format_instruction:
        sections.append(f"## Response Format\n{format_instruction}")

    # 5. Safety guidelines
    guardrails = template.get("guardrail_hints", [])
    if guardrails:
        guardrail_lines = "\n".join(f"- {hint}" for hint in guardrails)
        sections.append(f"## Safety Guidelines\n{guardrail_lines}")

    return "\n\n".join(sections)


def reload_templates() -> None:
    """
    Force a reload of all persona templates from disk.

    Useful for development or hot-reloading scenarios.
    """
    global _templates_cache, _display_name_to_key
    _templates_cache = None
    _display_name_to_key = None
    _ensure_loaded()
