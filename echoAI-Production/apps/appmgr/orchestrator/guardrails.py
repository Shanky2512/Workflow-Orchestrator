"""
EchoAI Orchestrator -- Guardrails Engine

Pre-processing and post-processing safety checks applied to user input
and LLM output respectively.  All checks are regex-based (no LLM cost)
and configurable per application through guardrail categories and
custom guardrail text.

Pre-processing (on user input):
    - PII detection: SSN, credit card, email, phone
    - Safety keyword blocklist
    - Custom compliance rules parsed from guardrail_text

Post-processing (on LLM output):
    - Same PII regex scan, redacts matches with [REDACTED]
    - Custom compliance rules

Output:
    GuardrailResult dataclass with is_safe, violations, sanitized_text,
    and categories_triggered.
"""

import re
import logging
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PII Detection Patterns
# ---------------------------------------------------------------------------

_PII_PATTERNS = {
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "credit_card": re.compile(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b"),
    "email": re.compile(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
    ),
    "phone": re.compile(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b"),
}

# ---------------------------------------------------------------------------
# Default Safety Keyword Blocklist
# ---------------------------------------------------------------------------

_DEFAULT_SAFETY_KEYWORDS: List[str] = [
    "bomb",
    "exploit",
    "hack into",
    "bypass security",
    "inject sql",
    "drop table",
    "rm -rf",
    "suicide",
    "self-harm",
    "kill yourself",
]


# ---------------------------------------------------------------------------
# GuardrailResult Dataclass
# ---------------------------------------------------------------------------

@dataclass
class GuardrailResult:
    """Result of a guardrail check pass."""

    is_safe: bool = True
    violations: List[str] = field(default_factory=list)
    sanitized_text: str = ""
    categories_triggered: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# GuardrailsEngine
# ---------------------------------------------------------------------------

class GuardrailsEngine:
    """
    Stateless guardrails engine.

    Performs regex-based PII detection, safety keyword checking, and
    custom compliance rule enforcement.  No LLM calls are made.
    """

    def pre_process(
        self,
        text: str,
        guardrail_categories: Optional[List[str]] = None,
        guardrail_text: Optional[str] = None,
    ) -> GuardrailResult:
        """
        Pre-process user input before sending to the orchestrator LLM.

        Detects PII, checks safety keywords, and evaluates custom compliance
        rules.  Does NOT redact -- returns the original text with violation
        metadata so the caller can decide whether to block.

        Args:
            text: Raw user input.
            guardrail_categories: List of active guardrail category names
                                  (e.g. ["PII", "Safety", "Compliance"]).
            guardrail_text: Custom compliance rules written by the app creator.

        Returns:
            GuardrailResult with is_safe=False if any violations found.
        """
        if not text:
            return GuardrailResult(is_safe=True, sanitized_text=text)

        categories = {c.lower() for c in (guardrail_categories or [])}
        violations: List[str] = []
        cats_triggered: List[str] = []

        # --- PII checks ---
        if not categories or "pii" in categories:
            pii_hits = self._detect_pii(text)
            if pii_hits:
                violations.extend(pii_hits)
                if "pii" not in cats_triggered:
                    cats_triggered.append("pii")

        # --- Safety keyword checks ---
        if not categories or "safety" in categories:
            safety_hits = self._check_safety_keywords(text)
            if safety_hits:
                violations.extend(safety_hits)
                if "safety" not in cats_triggered:
                    cats_triggered.append("safety")

        # --- Custom compliance rules ---
        if not categories or "compliance" in categories:
            compliance_hits = self._check_custom_rules(text, guardrail_text)
            if compliance_hits:
                violations.extend(compliance_hits)
                if "compliance" not in cats_triggered:
                    cats_triggered.append("compliance")

        return GuardrailResult(
            is_safe=len(violations) == 0,
            violations=violations,
            sanitized_text=text,
            categories_triggered=cats_triggered,
        )

    def post_process(
        self,
        text: str,
        guardrail_categories: Optional[List[str]] = None,
        guardrail_text: Optional[str] = None,
    ) -> GuardrailResult:
        """
        Post-process LLM output before returning to the user.

        Scans for PII and redacts any matches with ``[REDACTED]``.
        Also applies custom compliance rules.

        Args:
            text: LLM-generated output text.
            guardrail_categories: List of active guardrail category names.
            guardrail_text: Custom compliance rules.

        Returns:
            GuardrailResult with sanitized_text containing redactions.
        """
        if not text:
            return GuardrailResult(is_safe=True, sanitized_text=text)

        categories = {c.lower() for c in (guardrail_categories or [])}
        violations: List[str] = []
        cats_triggered: List[str] = []
        sanitized = text

        # --- PII redaction ---
        if not categories or "pii" in categories:
            sanitized, pii_hits = self._redact_pii(sanitized)
            if pii_hits:
                violations.extend(pii_hits)
                if "pii" not in cats_triggered:
                    cats_triggered.append("pii")

        # --- Custom compliance rules ---
        if not categories or "compliance" in categories:
            compliance_hits = self._check_custom_rules(sanitized, guardrail_text)
            if compliance_hits:
                violations.extend(compliance_hits)
                if "compliance" not in cats_triggered:
                    cats_triggered.append("compliance")

        return GuardrailResult(
            is_safe=len(violations) == 0,
            violations=violations,
            sanitized_text=sanitized,
            categories_triggered=cats_triggered,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_pii(text: str) -> List[str]:
        """Return list of PII violation strings found in text."""
        hits: List[str] = []
        for pii_type, pattern in _PII_PATTERNS.items():
            if pattern.search(text):
                hits.append(f"PII_DETECTED: {pii_type}")
        return hits

    @staticmethod
    def _redact_pii(text: str) -> tuple:
        """Redact PII in text and return (sanitized_text, violation_list)."""
        hits: List[str] = []
        sanitized = text
        for pii_type, pattern in _PII_PATTERNS.items():
            if pattern.search(sanitized):
                hits.append(f"PII_REDACTED: {pii_type}")
                sanitized = pattern.sub("[REDACTED]", sanitized)
        return sanitized, hits

    @staticmethod
    def _check_safety_keywords(text: str) -> List[str]:
        """Check text against the safety keyword blocklist."""
        hits: List[str] = []
        text_lower = text.lower()
        for keyword in _DEFAULT_SAFETY_KEYWORDS:
            if keyword in text_lower:
                hits.append(f"SAFETY: blocked keyword '{keyword}'")
        return hits

    @staticmethod
    def _check_custom_rules(
        text: str, guardrail_text: Optional[str]
    ) -> List[str]:
        """
        Parse custom compliance rules from guardrail_text and check text.

        Custom rules are plain-text lines.  Lines that start with
        "BLOCK:" are treated as substring blocklist entries.
        Lines that start with "REGEX:" are treated as regex patterns.
        All other non-empty lines are treated as substring checks.
        """
        if not guardrail_text:
            return []

        hits: List[str] = []
        text_lower = text.lower()

        for line in guardrail_text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            if line.upper().startswith("BLOCK:"):
                keyword = line[6:].strip().lower()
                if keyword and keyword in text_lower:
                    hits.append(f"COMPLIANCE: blocked term '{keyword}'")

            elif line.upper().startswith("REGEX:"):
                pattern_str = line[6:].strip()
                try:
                    if re.search(pattern_str, text, re.IGNORECASE):
                        hits.append(
                            f"COMPLIANCE: matched custom regex '{pattern_str}'"
                        )
                except re.error:
                    logger.warning(
                        "Invalid regex in guardrail_text: %s", pattern_str
                    )

            else:
                # Treat as a plain-text substring check
                if line.lower() in text_lower:
                    hits.append(f"COMPLIANCE: matched rule '{line}'")

        return hits
