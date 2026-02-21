"""
Celery Application Configuration for EchoAI Workflow Orchestration.

Broker: Redis (redis://localhost:6379/0)
Backend: Database via SQLAlchemy (uses existing echoAI PostgreSQL)

This module initializes the Celery app used for asynchronous workflow
execution, particularly for HITL (Human-in-the-Loop) workflows that
require pause/resume semantics.

Usage:
    Start worker:
        celery -A echolib.celery.celery_app worker --loglevel=info --queues=default,workflows

    Start beat (optional, for scheduled tasks):
        celery -A echolib.celery.celery_app beat --loglevel=info
"""
import os
from celery import Celery

# Derive SQLAlchemy database URL for Celery result backend.
# Celery's db+postgresql backend requires a synchronous driver.
# echolib uses postgresql+asyncpg, so we derive the sync version.
_db_url = os.getenv(
    'DATABASE_URL',
    'postgresql+asyncpg://echoai:echoai_dev@localhost:5432/echoai'
)
# Convert async URL to sync for Celery backend
_sync_db_url = _db_url.replace('+asyncpg', '').replace('+psycopg', '')
_celery_backend = f"db+{_sync_db_url}"

# Redis broker URL
_redis_broker = os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/0')

# Create Celery application
app = Celery(
    'echoai',
    broker=_redis_broker,
    backend=_celery_backend,
)

# Celery configuration
app.conf.update(
    # Task serialization
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',

    # Timezone
    timezone='UTC',
    enable_utc=True,

    # Task routing — workflow tasks go to dedicated queue
    task_routes={
        'echolib.celery.celery_tasks.execute_workflow_task': {'queue': 'workflows'},
        'echolib.celery.celery_tasks.resume_workflow_task': {'queue': 'workflows'},
    },

    # Default queue for unrouted tasks
    task_default_queue='default',

    # Task execution settings
    task_acks_late=True,           # Ack after task completes (safer for HITL)
    worker_prefetch_multiplier=1,  # One task at a time per worker (HITL needs this)

    # Result expiration (7 days)
    result_expires=604800,

    # Task soft/hard time limits (workflow execution can be long)
    task_soft_time_limit=3600,   # 1 hour soft limit
    task_time_limit=3900,        # 1 hour 5 min hard limit

    # Retry settings
    task_default_retry_delay=60,   # 1 minute between retries
    task_max_retries=3,
)

# Auto-discover tasks from echolib.celery package
app.autodiscover_tasks(['echolib.celery'])
