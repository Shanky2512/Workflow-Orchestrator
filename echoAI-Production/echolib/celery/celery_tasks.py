"""
Celery tasks for asynchronous workflow execution.

These tasks wrap the WorkflowExecutor methods for background execution,
particularly for HITL workflows that require pause/resume semantics.

Tasks:
    - execute_workflow_task: Execute a workflow (supports HITL interrupt detection).
    - resume_workflow_task: Resume a workflow paused at a HITL interrupt.
"""
import logging
from typing import Dict, Any, Optional

from echolib.celery.celery_app import app

logger = logging.getLogger(__name__)


@app.task(
    name='echolib.celery.celery_tasks.execute_workflow_task',
    bind=True,
    max_retries=2,
    default_retry_delay=30,
)
def execute_workflow_task(
    self,
    run_id: str,
    workflow_id: str,
    input_data: Optional[Dict[str, Any]] = None,
    execution_mode: str = "draft",
    version: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Celery task to execute a workflow asynchronously.

    Delegates to WorkflowExecutor.execute_workflow_hitl() which handles
    HITL interrupt detection. If the workflow pauses at an interrupt,
    the task completes with status="interrupted" and the Celery worker
    is released. State is preserved in PostgresSaver.

    Args:
        run_id: Execution run identifier.
        workflow_id: Workflow identifier.
        input_data: Input payload for the workflow.
        execution_mode: "draft", "test", or "final".
        version: Version string (for final mode).

    Returns:
        Execution result dict with status, output, and interrupt info.
    """
    from echolib.di import container

    logger.info(
        f"[CELERY] Starting workflow execution: "
        f"run_id={run_id}, workflow_id={workflow_id}, mode={execution_mode}"
    )

    try:
        executor = container.resolve('workflow.executor')

        # Merge run_id into input_data
        payload = dict(input_data or {})
        payload["run_id"] = run_id

        result = executor.execute_workflow_hitl(
            workflow_id=workflow_id,
            execution_mode=execution_mode,
            version=version,
            input_payload=payload,
        )

        status = result.get("status", "unknown")
        logger.info(
            f"[CELERY] Workflow execution completed: "
            f"run_id={run_id}, status={status}"
        )

        if status == "interrupted":
            logger.info(
                f"[CELERY] Workflow paused at HITL interrupt: run_id={run_id}. "
                f"Worker released. Awaiting human decision."
            )

        return result

    except Exception as exc:
        logger.error(
            f"[CELERY] Workflow execution failed: "
            f"run_id={run_id}, error={exc}",
            exc_info=True,
        )
        # Return error result instead of retrying for non-transient failures
        return {
            "run_id": run_id,
            "workflow_id": workflow_id,
            "status": "failed",
            "error": str(exc),
            "output": {},
        }


@app.task(
    name='echolib.celery.celery_tasks.resume_workflow_task',
    bind=True,
    max_retries=1,
    default_retry_delay=15,
)
def resume_workflow_task(
    self,
    run_id: str,
    action: str,
    payload: Optional[Dict[str, Any]] = None,
    workflow_id: Optional[str] = None,
    execution_mode: str = "draft",
    version: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Celery task to resume a workflow paused at a HITL interrupt.

    Uses LangGraph's Command(resume=...) to continue execution from
    the exact interrupt point. The resume value becomes the return
    value of interrupt() inside the HITL node.

    Args:
        run_id: The original run_id (used as thread_id for checkpoint).
        action: Human decision ("approve", "reject", "edit", "defer").
        payload: Additional data (edit content, rationale, etc.).
        workflow_id: Workflow identifier (needed to re-compile graph).
        execution_mode: "draft", "test", or "final".
        version: Version string (for final mode).

    Returns:
        Execution result dict (may be another interrupt or final completion).
    """
    from echolib.di import container

    logger.info(
        f"[CELERY] Resuming workflow: "
        f"run_id={run_id}, action={action}, workflow_id={workflow_id}"
    )

    if not workflow_id:
        # Try to resolve workflow_id from HITL manager context
        try:
            hitl = container.resolve('workflow.hitl')
            context = hitl.get_context(run_id)
            if context:
                workflow_id = context.workflow_id
        except Exception:
            pass

    if not workflow_id:
        error_msg = (
            f"Cannot resume run '{run_id}': workflow_id not provided "
            f"and could not be resolved from HITL context."
        )
        logger.error(f"[CELERY] {error_msg}")
        return {
            "run_id": run_id,
            "status": "failed",
            "error": error_msg,
            "output": {},
        }

    try:
        executor = container.resolve('workflow.executor')

        result = executor.resume_workflow(
            workflow_id=workflow_id,
            run_id=run_id,
            action=action,
            payload=payload or {},
            execution_mode=execution_mode,
            version=version,
        )

        status = result.get("status", "unknown")
        logger.info(
            f"[CELERY] Workflow resume completed: "
            f"run_id={run_id}, status={status}"
        )

        return result

    except Exception as exc:
        logger.error(
            f"[CELERY] Workflow resume failed: "
            f"run_id={run_id}, error={exc}",
            exc_info=True,
        )
        return {
            "run_id": run_id,
            "workflow_id": workflow_id,
            "status": "failed",
            "error": str(exc),
            "output": {},
        }
