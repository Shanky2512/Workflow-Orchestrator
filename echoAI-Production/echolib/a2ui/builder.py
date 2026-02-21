"""
A2UIStreamBuilder

Helper class that produces A2UI v0.8 JSONL strings for the EchoAI execution
step-list surface. All methods return strings or lists of strings, ready to be
put into an asyncio.Queue for SSE streaming.

Surface layout:
  Column
    +-- header (Text h3, data-bound to /title)
    +-- steps  (List, template bound to /steps, each step is a Card)
    |     Card: [step-icon (Icon), step-label (Text h5), step-detail (Text caption), step-status (Text caption)]
    +-- output (Card -> output-text (Text body, data-bound to /final_output))

Step dict shape:
  {
    "id":     str,   # unique step identifier
    "icon":   str,   # Material Icons name
    "label":  str,   # human-readable stage name
    "detail": str,   # extra context (may be empty)
    "status": str,   # pending | running | completed | failed | interrupted | skipped
  }
"""

from __future__ import annotations

import json
from typing import Any, Dict, List


class A2UIStreamBuilder:
    """
    Builds A2UI v0.8 JSONL messages for the EchoAI execution step-list surface.

    Usage:
        builder = A2UIStreamBuilder()
        # On stream start:
        for line in builder.build_initial_surface(run_id, "My App"):
            await queue.put(line)
        # On each stage change:
        await queue.put(builder.step_update(run_id, steps))
        # On completion:
        await queue.put(builder.final_output_update(run_id, result_text))
    """

    @staticmethod
    def surface_id(run_id: str) -> str:
        return f"exec_{run_id}"

    def build_initial_surface(self, run_id: str, title: str) -> List[str]:
        """
        Returns a list of 3 JSONL lines:
          1. surfaceUpdate   -- component tree definition
          2. dataModelUpdate -- initial empty data model
          3. beginRendering  -- signal client to render
        """
        sid = self.surface_id(run_id)

        surface_update = {
            "type": "surfaceUpdate",
            "surfaceId": sid,
            "catalog": "std:v1",
            "components": [
                {
                    "id": "root",
                    "type": "Column",
                    "children": {"explicitList": ["header", "steps", "output"]},
                },
                {
                    "id": "header",
                    "type": "Text",
                    "content": {"path": "/title"},
                    "hint": "h3",
                },
                {
                    "id": "steps",
                    "type": "List",
                    "children": {
                        "template": {
                            "dataPath": "/steps",
                            "componentType": "Card",
                            "childIds": [
                                "step-icon",
                                "step-label",
                                "step-detail",
                                "step-status",
                            ],
                        }
                    },
                },
                {
                    "id": "step-icon",
                    "type": "Icon",
                    "name": {"path": "/steps[*]/icon"},
                },
                {
                    "id": "step-label",
                    "type": "Text",
                    "content": {"path": "/steps[*]/label"},
                    "hint": "h5",
                },
                {
                    "id": "step-detail",
                    "type": "Text",
                    "content": {"path": "/steps[*]/detail"},
                    "hint": "caption",
                },
                {
                    "id": "step-status",
                    "type": "Text",
                    "content": {"path": "/steps[*]/status"},
                    "hint": "caption",
                },
                {
                    "id": "output",
                    "type": "Card",
                    "children": {"explicitList": ["output-text"]},
                },
                {
                    "id": "output-text",
                    "type": "Text",
                    "content": {"path": "/final_output"},
                    "hint": "body",
                },
            ],
        }

        data_model_update = {
            "type": "dataModelUpdate",
            "surfaceId": sid,
            "data": {
                "title": title,
                "steps": [],
                "final_output": "",
            },
        }

        begin_rendering = {
            "type": "beginRendering",
            "surfaceId": sid,
            "rootComponentId": "root",
        }

        return [
            json.dumps(surface_update, separators=(",", ":")),
            json.dumps(data_model_update, separators=(",", ":")),
            json.dumps(begin_rendering, separators=(",", ":")),
        ]

    def step_update(self, run_id: str, steps: List[Dict[str, str]]) -> str:
        """
        Returns a single JSONL line: dataModelUpdate with the full steps array.

        Each step dict must have: id, icon, label, detail, status.
        The full array is always sent (client replaces /steps entirely).
        """
        sid = self.surface_id(run_id)
        msg = {
            "type": "dataModelUpdate",
            "surfaceId": sid,
            "data": {"steps": steps},
        }
        return json.dumps(msg, separators=(",", ":"))

    def final_output_update(self, run_id: str, text: str) -> str:
        """
        Returns a single JSONL line: dataModelUpdate that sets /final_output.
        Call this after all steps are complete.
        """
        sid = self.surface_id(run_id)
        msg = {
            "type": "dataModelUpdate",
            "surfaceId": sid,
            "data": {"final_output": text},
        }
        return json.dumps(msg, separators=(",", ":"))

    def title_update(self, run_id: str, title: str) -> str:
        """Returns a JSONL line that updates just the /title field."""
        sid = self.surface_id(run_id)
        msg = {
            "type": "dataModelUpdate",
            "surfaceId": sid,
            "data": {"title": title},
        }
        return json.dumps(msg, separators=(",", ":"))

    def delete_surface(self, run_id: str) -> str:
        """Returns a JSONL line to remove the surface from the client."""
        sid = self.surface_id(run_id)
        msg = {"type": "deleteSurface", "surfaceId": sid}
        return json.dumps(msg, separators=(",", ":"))

    @staticmethod
    def make_step(
        step_id: str,
        icon: str,
        label: str,
        status: str,
        detail: str = "",
    ) -> Dict[str, str]:
        """Convenience factory for a step dict."""
        return {
            "id": step_id,
            "icon": icon,
            "label": label,
            "detail": detail,
            "status": status,
        }
