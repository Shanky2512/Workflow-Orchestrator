"""
Workbench Mode -- Pydantic Schemas

Request/response models for all Workbench APIs.
Completely independent of appmgr/orchestrator schemas.
"""

from pydantic import BaseModel
from typing import Optional
from datetime import datetime


# ── Chat ───────────────────────────────────────────────────────────────────

class WorkbenchChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


class WorkbenchChatResponse(BaseModel):
    response: str
    session_id: str
    mode: str = "workbench"
    timestamp: datetime


# ── Status ─────────────────────────────────────────────────────────────────

class WorkbenchStatusResponse(BaseModel):
    enabled: bool
    mode: str = "workbench"
    available_services: list[str]
