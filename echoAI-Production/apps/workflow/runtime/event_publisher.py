"""
Event Publisher for Execution Transparency

Creates and broadcasts execution events via WebSocket.
All functions are async and designed for fire-and-forget usage.

CRITICAL: Node functions in LangGraph are synchronous but run inside an async context.
Use fire_and_forget() to properly schedule async events from sync code.
"""

from datetime import datetime
from typing import Dict, Any, Optional
from .transparency import StepType, EventType, ExecutionEvent
from .ws_manager import ws_manager
import asyncio
import logging

logger = logging.getLogger(__name__)


def fire_and_forget(coro):
    """
    Schedule a coroutine for fire-and-forget execution.
    Works from both sync and async contexts.
    Non-blocking - does not wait for result.

    CRITICAL: When called from sync code inside an async context (common in LangGraph),
    we must use run_coroutine_threadsafe to properly schedule the task.

    Usage:
        fire_and_forget(publish_step_started(run_id, step_id, ...))
    """
    try:
        # First, try to get the running loop (only works from async context)
        try:
            loop = asyncio.get_running_loop()
            # We're in an async context - create_task works directly
            loop.create_task(coro)
            logger.debug("Event scheduled via create_task (async context)")
            return
        except RuntimeError:
            # No running loop in current coroutine context - we're in sync code
            pass

        # We're in sync code - try to get the event loop
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            logger.warning("No event loop available for transparency event")
            return

        if loop.is_running():
            # Loop is running (in another call frame) but we're in sync code
            # Use run_coroutine_threadsafe which is designed for this exact case
            future = asyncio.run_coroutine_threadsafe(coro, loop)
            # Don't wait for result - fire and forget
            logger.debug("Event scheduled via run_coroutine_threadsafe (sync context)")
        else:
            # Loop exists but not running - run synchronously
            loop.run_until_complete(coro)
            logger.debug("Event executed via run_until_complete")
    except Exception as e:
        # Log but don't fail - transparency is optional enhancement
        logger.warning(f"Failed to schedule transparency event: {e}")


async def publish_run_started(run_id: str, workflow_id: str) -> None:
    """
    Publish run_started event.

    Args:
        run_id: Run identifier
        workflow_id: Workflow being executed
    """
    event = ExecutionEvent(
        event_type=EventType.RUN_STARTED,
        run_id=run_id,
        step_id=None,
        timestamp=datetime.utcnow().isoformat(),
        payload={"workflow_id": workflow_id}
    )
    await ws_manager.broadcast(run_id, event)
    logger.debug(f"Published run_started for {run_id}")


async def publish_step_started(
    run_id: str,
    step_id: str,
    step_name: str,
    step_type: StepType,
    input_summary: Optional[Dict[str, Any]] = None
) -> None:
    """
    Publish step_started event.

    Args:
        run_id: Run identifier
        step_id: Step identifier
        step_name: Human-readable step name
        step_type: Type of step
        input_summary: Sanitized input data (no CoT)
    """
    event = ExecutionEvent(
        event_type=EventType.STEP_STARTED,
        run_id=run_id,
        step_id=step_id,
        timestamp=datetime.utcnow().isoformat(),
        payload={
            "step_name": step_name,
            "step_type": step_type.value,
            "input_summary": input_summary or {}
        }
    )
    await ws_manager.broadcast(run_id, event)
    logger.debug(f"Published step_started for {step_id}")


async def publish_step_output(
    run_id: str,
    step_id: str,
    intermediate_output: Dict[str, Any]
) -> None:
    """
    Publish step_output event for intermediate results.

    Args:
        run_id: Run identifier
        step_id: Step identifier
        intermediate_output: Partial output data
    """
    event = ExecutionEvent(
        event_type=EventType.STEP_OUTPUT,
        run_id=run_id,
        step_id=step_id,
        timestamp=datetime.utcnow().isoformat(),
        payload={"intermediate_output": intermediate_output}
    )
    await ws_manager.broadcast(run_id, event)
    logger.debug(f"Published step_output for {step_id}")


async def publish_step_completed(
    run_id: str,
    step_id: str,
    output_summary: Optional[Dict[str, Any]] = None
) -> None:
    """
    Publish step_completed event.

    Args:
        run_id: Run identifier
        step_id: Step identifier
        output_summary: Sanitized output data (no CoT)
    """
    event = ExecutionEvent(
        event_type=EventType.STEP_COMPLETED,
        run_id=run_id,
        step_id=step_id,
        timestamp=datetime.utcnow().isoformat(),
        payload={"output_summary": output_summary or {}}
    )
    await ws_manager.broadcast(run_id, event)
    logger.debug(f"Published step_completed for {step_id}")


async def publish_step_failed(
    run_id: str,
    step_id: str,
    error: str
) -> None:
    """
    Publish step_failed event.

    Args:
        run_id: Run identifier
        step_id: Step identifier
        error: Error message
    """
    event = ExecutionEvent(
        event_type=EventType.STEP_FAILED,
        run_id=run_id,
        step_id=step_id,
        timestamp=datetime.utcnow().isoformat(),
        payload={"error": error}
    )
    await ws_manager.broadcast(run_id, event)
    logger.debug(f"Published step_failed for {step_id}")


async def publish_run_completed(
    run_id: str,
    final_state: Dict[str, Any]
) -> None:
    """
    Publish run_completed event with final response.

    Extracts the final response from the workflow output state
    using the same logic as routes.py.

    Args:
        run_id: Run identifier
        final_state: Final workflow state dictionary
    """
    # CRITICAL: Flush all buffered events to clients BEFORE sending run_completed
    # This ensures step events (buffered from sync code) are delivered first
    await ws_manager.flush_to_clients(run_id)

    # Check if this is a HITL interrupt — pass through interrupt data directly
    if final_state.get("status") == "interrupted" and "interrupt" in final_state:
        event = ExecutionEvent(
            event_type=EventType.RUN_COMPLETED,
            run_id=run_id,
            step_id=None,
            timestamp=datetime.utcnow().isoformat(),
            payload={
                "status": "interrupted",
                "interrupt": final_state["interrupt"],
            }
        )
        await ws_manager.broadcast(run_id, event)
        # Do NOT clean up WebSocket — frontend needs it until decision is made
        return

    # Extract final response (same logic as routes.py)
    output = final_state
    if "crew_result" in output and output["crew_result"]:
        final_response = str(output["crew_result"])
    elif "result" in output:
        final_response = str(output["result"])
    elif "hierarchical_output" in output and output["hierarchical_output"]:
        final_response = str(output["hierarchical_output"])
    elif "parallel_output" in output and output["parallel_output"]:
        final_response = str(output["parallel_output"])
    else:
        # Fallback
        final_response = str(output.get("messages", ""))

    # Truncate only extremely large responses (50KB limit for WebSocket payload)
    if len(final_response) > 50000:
        final_response = final_response[:50000] + "\n\n[Response truncated due to size limit]"

    event = ExecutionEvent(
        event_type=EventType.RUN_COMPLETED,
        run_id=run_id,
        step_id=None,
        timestamp=datetime.utcnow().isoformat(),
        payload={
            "final_response": final_response,
            "status": "completed"
        }
    )
    await ws_manager.broadcast(run_id, event)

    # Clean up WebSocket connections for this run
    await ws_manager.async_cleanup_run(run_id)
    logger.debug(f"Published run_completed for {run_id}")


async def publish_run_failed(run_id: str, error: str) -> None:
    """
    Publish run_failed event.

    Args:
        run_id: Run identifier
        error: Error message
    """
    event = ExecutionEvent(
        event_type=EventType.RUN_FAILED,
        run_id=run_id,
        step_id=None,
        timestamp=datetime.utcnow().isoformat(),
        payload={
            "error": error,
            "status": "failed"
        }
    )
    await ws_manager.broadcast(run_id, event)

    # Clean up WebSocket connections for this run
    await ws_manager.async_cleanup_run(run_id)
    logger.debug(f"Published run_failed for {run_id}")


# ==============================================================================
# SYNCHRONOUS PUBLISH FUNCTIONS
# ==============================================================================
# Use these from synchronous code (LangGraph node functions, CrewAI adapter)
# They buffer events immediately without async, ensuring correct event ordering.
# ==============================================================================

def publish_step_started_sync(
    run_id: str,
    step_id: str,
    step_name: str,
    step_type: StepType,
    input_summary: Optional[Dict[str, Any]] = None
) -> None:
    """
    Synchronously buffer step_started event.

    Use this from sync code (LangGraph node functions) to ensure events
    are buffered immediately in the correct order.

    Args:
        run_id: Run identifier
        step_id: Step identifier
        step_name: Human-readable step name
        step_type: Type of step
        input_summary: Sanitized input data (no CoT)
    """
    event = ExecutionEvent(
        event_type=EventType.STEP_STARTED,
        run_id=run_id,
        step_id=step_id,
        timestamp=datetime.utcnow().isoformat(),
        payload={
            "step_name": step_name,
            "step_type": step_type.value,
            "input_summary": input_summary or {}
        }
    )
    ws_manager.buffer_event_sync(run_id, event)
    logger.debug(f"Buffered step_started for {step_id}")


def publish_step_completed_sync(
    run_id: str,
    step_id: str,
    output_summary: Optional[Dict[str, Any]] = None
) -> None:
    """
    Synchronously buffer step_completed event.

    Use this from sync code (LangGraph node functions) to ensure events
    are buffered immediately in the correct order.

    Args:
        run_id: Run identifier
        step_id: Step identifier
        output_summary: Sanitized output data (should include output_preview)
    """
    # Extract output_preview for direct access in frontend
    output_preview = ""
    if output_summary:
        output_preview = output_summary.get("output_preview", "")
        # Truncate to 300 chars for WebSocket payload
        if len(output_preview) > 300:
            output_preview = output_preview[:300] + "..."

    event = ExecutionEvent(
        event_type=EventType.STEP_COMPLETED,
        run_id=run_id,
        step_id=step_id,
        timestamp=datetime.utcnow().isoformat(),
        payload={
            "output_summary": output_summary or {},
            "output_preview": output_preview  # Direct access for frontend
        }
    )
    ws_manager.buffer_event_sync(run_id, event)
    logger.debug(f"Buffered step_completed for {step_id} (preview: {len(output_preview)} chars)")


def publish_step_failed_sync(
    run_id: str,
    step_id: str,
    error: str
) -> None:
    """
    Synchronously buffer step_failed event.

    Use this from sync code (LangGraph node functions) to ensure events
    are buffered immediately in the correct order.

    Args:
        run_id: Run identifier
        step_id: Step identifier
        error: Error message
    """
    event = ExecutionEvent(
        event_type=EventType.STEP_FAILED,
        run_id=run_id,
        step_id=step_id,
        timestamp=datetime.utcnow().isoformat(),
        payload={"error": error}
    )
    ws_manager.buffer_event_sync(run_id, event)
    logger.debug(f"Buffered step_failed for {step_id}")
