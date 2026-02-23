"""
EchoAI Observability Utilities

Provides utilities for the 5 monitoring requirements:
1. Visualize: Covered by Langfuse + OTel auto-instrumentation
2. Time Travel Debugging: Session-based trace grouping
3. Debug and Audit: Prompt injection detection, audit trail
4. LLM Cost Tracking: Token/cost tracking utilities
5. Fine-Tuning: Dataset creation from production completions

All functions are safe to call regardless of whether Langfuse is
initialized. If Langfuse is not available, functions degrade gracefully
to standard Python logging or return None.
"""
import logging
import re
from typing import Any, Dict, List, Optional

from echolib.config import settings

logger = logging.getLogger(__name__)


def _get_langfuse():
    """
    Get the Langfuse singleton client.

    Returns None if Langfuse is not initialized or not configured.
    """
    if not settings.LANGFUSE_TRACING_ENABLED or not settings.LANGFUSE_PUBLIC_KEY:
        return None
    try:
        from langfuse import get_client
        return get_client()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 2. Time Travel Debugging -- Session-based trace grouping
# ---------------------------------------------------------------------------

def create_trace_for_run(
    run_id: str,
    workflow_id: str,
    user_id: str = "system",
    execution_mode: str = "draft",
    tags: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
):
    """
    Create a top-level Langfuse trace for a workflow run.

    Groups all spans under a single trace with session_id for
    time-travel debugging (rewind/replay by session).

    Uses the real Langfuse SDK v3 trace creation API.

    Args:
        run_id: Unique run identifier (used as session_id).
        workflow_id: Workflow identifier.
        user_id: User who triggered the run.
        execution_mode: "draft", "test", or "final".
        tags: Additional tags for filtering.
        metadata: Extra metadata to attach to the trace.

    Returns:
        Langfuse trace object, or None if Langfuse is unavailable.
    """
    langfuse = _get_langfuse()
    if not langfuse:
        return None

    trace_tags = [f"workflow:{workflow_id}", f"mode:{execution_mode}"]
    if tags:
        trace_tags.extend(tags)

    try:
        trace = langfuse.trace(
            name=f"workflow:{workflow_id}",
            session_id=run_id,
            user_id=user_id,
            tags=trace_tags,
            metadata={
                "workflow_id": workflow_id,
                "execution_mode": execution_mode,
                **(metadata or {}),
            },
        )
        return trace
    except Exception as e:
        logger.debug(f"Failed to create Langfuse trace: {e}")
        return None


# ---------------------------------------------------------------------------
# 3. Debug and Audit -- Prompt injection detection
# ---------------------------------------------------------------------------

# Common prompt injection patterns
_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"ignore\s+(all\s+)?above\s+instructions",
    r"disregard\s+(all\s+)?previous",
    r"forget\s+(all\s+)?previous",
    r"you\s+are\s+now\s+(a|an)\s+",
    r"new\s+instruction[s]?\s*:",
    r"system\s*:\s*you\s+are",
    r"<\s*system\s*>",
    r"\[INST\]",
    r"###\s*(system|instruction)",
    r"act\s+as\s+if\s+you\s+have\s+no\s+restrictions",
    r"pretend\s+you\s+(are|have)\s+no\s+(rules|restrictions|limitations)",
    r"jailbreak",
    r"DAN\s+mode",
]

_COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in _INJECTION_PATTERNS]


def detect_prompt_injection(text: str) -> Dict[str, Any]:
    """
    Detect potential prompt injection attacks in user input.

    Args:
        text: User-provided text to scan.

    Returns:
        Dict with keys:
            - is_suspicious: bool
            - matched_patterns: list of matched pattern strings
            - risk_score: float 0.0 - 1.0
    """
    if not text:
        return {"is_suspicious": False, "matched_patterns": [], "risk_score": 0.0}

    matches = []
    for i, pattern in enumerate(_COMPILED_PATTERNS):
        if pattern.search(text):
            matches.append(_INJECTION_PATTERNS[i])

    risk_score = min(len(matches) / 3.0, 1.0)  # 3+ matches = max risk

    return {
        "is_suspicious": len(matches) > 0,
        "matched_patterns": matches,
        "risk_score": risk_score,
    }


def audit_log_input(
    run_id: str,
    user_input: str,
    user_id: str = "system",
    workflow_id: str = "",
) -> Dict[str, Any]:
    """
    Audit log a user input, including prompt injection detection.

    Creates a Langfuse score on the trace if injection is detected.
    Always returns the detection result regardless of Langfuse availability.

    Args:
        run_id: Run identifier (used as trace_id for scoring).
        user_input: The raw user input text.
        user_id: User identifier.
        workflow_id: Workflow identifier.

    Returns:
        Detection result dict from detect_prompt_injection().
    """
    detection = detect_prompt_injection(user_input)

    if detection["is_suspicious"]:
        logger.warning(
            f"[AUDIT] Potential prompt injection detected in run {run_id}: "
            f"risk={detection['risk_score']}, "
            f"patterns={detection['matched_patterns']}"
        )

    langfuse = _get_langfuse()
    if not langfuse:
        return detection

    if detection["is_suspicious"]:
        try:
            langfuse.create_score(
                name="prompt_injection_risk",
                value=detection["risk_score"],
                comment=f"Matched patterns: {detection['matched_patterns']}",
                trace_id=run_id,
            )
        except Exception as e:
            logger.debug(f"Failed to create Langfuse score: {e}")

    return detection


# ---------------------------------------------------------------------------
# 4. LLM Cost Tracking -- Token and cost utilities
# ---------------------------------------------------------------------------

def track_llm_usage(
    trace_id: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    total_tokens: Optional[int] = None,
    input_cost: Optional[float] = None,
    output_cost: Optional[float] = None,
    total_cost: Optional[float] = None,
):
    """
    Explicitly track LLM token usage and cost on a Langfuse trace.

    This is a supplement to auto-tracked usage from CallbackHandler.
    Use for LLM calls that bypass LangChain (e.g., direct OpenAI calls).

    Uses the real Langfuse SDK v3 generation API.

    Args:
        trace_id: Langfuse trace identifier.
        model: Model name string (e.g., "gpt-4o").
        input_tokens: Number of prompt tokens.
        output_tokens: Number of completion tokens.
        total_tokens: Total tokens (auto-calculated if omitted).
        input_cost: Cost of prompt tokens in USD.
        output_cost: Cost of completion tokens in USD.
        total_cost: Total cost in USD.
    """
    langfuse = _get_langfuse()
    if not langfuse:
        return

    usage = {
        "input": input_tokens,
        "output": output_tokens,
        "total": total_tokens or (input_tokens + output_tokens),
    }
    if input_cost is not None:
        usage["input_cost"] = input_cost
    if output_cost is not None:
        usage["output_cost"] = output_cost
    if total_cost is not None:
        usage["total_cost"] = total_cost

    try:
        langfuse.generation(
            trace_id=trace_id,
            name=f"llm_call:{model}",
            model=model,
            usage=usage,
        )
    except Exception as e:
        logger.debug(f"Failed to track LLM usage in Langfuse: {e}")


# ---------------------------------------------------------------------------
# 5. Fine-Tuning -- Dataset creation from production completions
# ---------------------------------------------------------------------------

def save_completion_for_finetuning(
    dataset_name: str,
    input_text: str,
    output_text: str,
    metadata: Optional[Dict[str, Any]] = None,
    source_trace_id: Optional[str] = None,
):
    """
    Save a completion (input/output pair) to a Langfuse Dataset
    for later fine-tuning export.

    Creates the dataset if it does not exist (idempotent).

    Args:
        dataset_name: Name of the Langfuse dataset (e.g., "finetune-v1").
        input_text: The prompt/input text.
        output_text: The completion/output text.
        metadata: Optional metadata (model, workflow_id, etc.).
        source_trace_id: Optional trace ID to link the dataset item to.

    Returns:
        The created DatasetItem object, or None if Langfuse is unavailable.
    """
    langfuse = _get_langfuse()
    if not langfuse:
        logger.warning("Langfuse not initialized -- cannot save fine-tuning data")
        return None

    try:
        # Create dataset if it does not exist (idempotent)
        langfuse.create_dataset(name=dataset_name)
    except Exception:
        pass  # Dataset already exists

    try:
        item = langfuse.create_dataset_item(
            dataset_name=dataset_name,
            input={"messages": [{"role": "user", "content": input_text}]},
            expected_output=output_text,
            metadata={
                **(metadata or {}),
                "source_trace_id": source_trace_id,
            },
        )
        logger.info(
            f"Saved completion to dataset '{dataset_name}' (item_id={item.id})"
        )
        return item
    except Exception as e:
        logger.warning(f"Failed to save completion to Langfuse dataset: {e}")
        return None
