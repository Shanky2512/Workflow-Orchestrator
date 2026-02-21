"""
A2UI v0.8 Protocol Models

JSONL-based streaming protocol for agent-to-UI communication.
Messages are newline-delimited JSON objects streamed server->client via SSE.

Message types (server -> client):
  surfaceUpdate    -- define/update UI component tree (flat adjacency list)
  dataModelUpdate  -- update data bindings / application state
  beginRendering   -- signal client to start rendering the surface
  deleteSurface    -- remove a surface

Components use flat adjacency list -- never nested JSON.
BoundValue is used wherever a value can be static or data-bound.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, model_serializer


class BoundValue(BaseModel):
    """A value that can be a literal string or bound to a data model path."""
    literalString: Optional[str] = None
    path: Optional[str] = None

    model_config = {"populate_by_name": True}


class TemplateSpec(BaseModel):
    """Specifies a dynamic list template bound to a data path."""
    dataPath: str
    componentType: str
    childIds: List[str]


class ChildrenSpec(BaseModel):
    """Children of a component -- either explicit list or dynamic template."""
    explicitList: Optional[List[str]] = None
    template: Optional[TemplateSpec] = None


class ActionDef(BaseModel):
    """Action triggered by user interaction."""
    actionId: str
    context: Optional[Dict[str, BoundValue]] = None


class Component(BaseModel):
    """A single UI component node in the flat adjacency list."""
    id: str
    type: str
    children: Optional[ChildrenSpec] = None
    content: Optional[BoundValue] = None
    hint: Optional[str] = None           # h1-h5, caption, body
    name: Optional[BoundValue] = None    # for Icon components
    label: Optional[BoundValue] = None   # for Button/TextField/CheckBox
    src: Optional[BoundValue] = None     # for Image components
    checked: Optional[BoundValue] = None # for CheckBox
    action: Optional[ActionDef] = None
    primary: Optional[bool] = None       # for Button
    weight: Optional[int] = None         # flex weight in Row/Column
    distribution: Optional[str] = None   # for Row/Column layout
    inputType: Optional[str] = None      # for TextField
    elevation: Optional[int] = None      # for Card

    model_config = {"populate_by_name": True}

    def model_dump_json_compact(self) -> str:
        return self.model_dump_json(exclude_none=True)


class SurfaceUpdateMessage(BaseModel):
    """Define or update UI components for a surface."""
    type: Literal["surfaceUpdate"] = "surfaceUpdate"
    surfaceId: str
    catalog: Optional[str] = None
    components: List[Component]

    def to_jsonl(self) -> str:
        return self.model_dump_json(exclude_none=True)


class DataModelUpdateMessage(BaseModel):
    """Populate or update application state / data bindings."""
    type: Literal["dataModelUpdate"] = "dataModelUpdate"
    surfaceId: str
    data: Dict[str, Any]

    def to_jsonl(self) -> str:
        return self.model_dump_json(exclude_none=True)


class BeginRenderingMessage(BaseModel):
    """Signal client to begin rendering. Must be last message for a surface."""
    type: Literal["beginRendering"] = "beginRendering"
    surfaceId: str
    rootComponentId: str

    def to_jsonl(self) -> str:
        return self.model_dump_json(exclude_none=True)


class DeleteSurfaceMessage(BaseModel):
    """Remove a surface from the client."""
    type: Literal["deleteSurface"] = "deleteSurface"
    surfaceId: str

    def to_jsonl(self) -> str:
        return self.model_dump_json(exclude_none=True)
