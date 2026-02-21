"""
Filesystem-based workflow storage.
Provides atomic writes, versioning, and lifecycle management.
"""
import json
import tempfile
import os
import copy
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime


class WorkflowStorage:
    """
    Workflow filesystem storage service.
    Manages draft, temp, final, and archived workflows.
    """

    def __init__(self, base_dir: str = None):
        """
        Initialize storage.

        Args:
            base_dir: Base directory for workflow storage
        """
        if base_dir is None:
            base_dir = Path(__file__).parent / "workflows"

        self.base_dir = Path(base_dir)
        self.draft_dir = self.base_dir / "draft"
        self.temp_dir = self.base_dir / "temp"
        self.final_dir = self.base_dir / "final"
        self.archive_dir = self.base_dir / "archive"

        # Ensure directories exist
        for directory in [self.draft_dir, self.temp_dir, self.final_dir, self.archive_dir]:
            directory.mkdir(parents=True, exist_ok=True)

    def _workflow_filename(self, workflow_id: str, suffix: str) -> str:
        """Generate workflow filename."""
        return f"{workflow_id}.{suffix}.json"

    def _atomic_write_json(self, path: Path, data: Dict[str, Any]) -> None:
        """
        Atomically write JSON to file.

        Args:
            path: Target file path
            data: Data to write
        """
        # Write to temp file first
        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=path.parent,
            delete=False,
            suffix=".tmp"
        ) as tmp:
            json.dump(data, tmp, indent=2)
            tmp.flush()
            os.fsync(tmp.fileno())
            tmp_path = tmp.name

        # Atomic replace
        os.replace(tmp_path, path)

    def save_workflow(
        self,
        workflow: Dict[str, Any],
        state: str  # "draft" | "temp" | "final"
    ) -> Dict[str, str]:
        """
        Save workflow to specified state.

        Args:
            workflow: Workflow definition
            state: Lifecycle state

        Returns:
            dict with workflow_id and path
        """
        workflow_id = workflow["workflow_id"]

        if state == "draft":
            directory = self.draft_dir
            suffix = "draft"
        elif state == "temp":
            directory = self.temp_dir
            suffix = "temp"
            workflow["status"] = "testing"
            workflow.setdefault("metadata", {})["is_temp"] = True
        elif state == "final":
            directory = self.final_dir
            version = workflow.get("version", "0.1")
            suffix = f"v{version}"
            workflow["status"] = "final"
            workflow.setdefault("metadata", {})["immutable"] = True
        else:
            raise ValueError(f"Invalid workflow state: {state}")

        path = directory / self._workflow_filename(workflow_id, suffix)
        self._atomic_write_json(path, workflow)

        return {
            "workflow_id": workflow_id,
            "path": str(path),
            "state": state
        }

    def load_workflow(
        self,
        workflow_id: str,
        state: str,
        version: str = None
    ) -> Dict[str, Any]:
        """
        Load workflow from specified state.

        Args:
            workflow_id: Workflow identifier
            state: Lifecycle state
            version: Version (required for final state)

        Returns:
            Workflow definition

        Raises:
            FileNotFoundError: If workflow not found
        """
        if state == "draft":
            path = self.draft_dir / self._workflow_filename(workflow_id, "draft")

        elif state == "temp":
            path = self.temp_dir / self._workflow_filename(workflow_id, "temp")

        elif state == "final":
            if not version:
                raise ValueError("Final workflow requires version")
            path = self.final_dir / self._workflow_filename(workflow_id, f"v{version}")

        else:
            raise ValueError(f"Invalid workflow state: {state}")

        if not path.exists():
            raise FileNotFoundError(f"Workflow not found: {path}")

        with open(path) as f:
            return json.load(f)

    def save_final_workflow(self, workflow: Dict[str, Any]) -> Dict[str, str]:
        """
        Save workflow as final (versioned, immutable).

        Args:
            workflow: Workflow definition

        Returns:
            dict with workflow_id, version, and path
        """
        if workflow.get("status") != "validated":
            raise ValueError("Workflow must be validated before final save")

        workflow["status"] = "final"
        workflow.setdefault("metadata", {})["immutable"] = True

        result = self.save_workflow(workflow, state="final")
        result["version"] = workflow["version"]
        return result

    def list_versions(self, workflow_id: str) -> List[str]:
        """
        List all final versions of a workflow.

        Args:
            workflow_id: Workflow identifier

        Returns:
            List of version strings
        """
        versions = []

        for file in self.final_dir.glob(f"{workflow_id}.v*.json"):
            # Extract version from filename
            version = file.stem.split(".v")[-1]
            versions.append(version)

        return sorted(versions)

    def clone_final_to_draft(
        self,
        workflow_id: str,
        from_version: str
    ) -> Dict[str, Any]:
        """
        Clone a final workflow to draft for editing.

        Args:
            workflow_id: Workflow identifier
            from_version: Version to clone from

        Returns:
            Cloned workflow definition
        """
        # Load final workflow
        final_workflow = self.load_workflow(
            workflow_id=workflow_id,
            state="final",
            version=from_version
        )

        # Clone safely
        cloned = copy.deepcopy(final_workflow)

        # Reset lifecycle fields
        cloned["status"] = "draft"
        cloned["version"] = from_version  # Keep base version
        cloned.setdefault("metadata", {})["cloned_from"] = from_version
        cloned["metadata"]["cloned_at"] = datetime.utcnow().isoformat()
        cloned["metadata"]["immutable"] = False

        # Remove validation lock
        cloned.pop("validation", None)

        # Save as draft
        self.save_workflow(cloned, state="draft")

        return cloned

    def delete_workflow(self, workflow_id: str, state: str) -> None:
        """
        Delete workflow from specified state.

        Args:
            workflow_id: Workflow identifier
            state: Lifecycle state
        """
        if state == "draft":
            path = self.draft_dir / self._workflow_filename(workflow_id, "draft")
        elif state == "temp":
            path = self.temp_dir / self._workflow_filename(workflow_id, "temp")
        else:
            raise ValueError("Can only delete draft or temp workflows")

        if path.exists():
            os.remove(path)

    def archive_workflow(self, workflow_id: str, version: str) -> None:
        """
        Archive a final workflow.

        Args:
            workflow_id: Workflow identifier
            version: Version to archive
        """
        src = self.final_dir / self._workflow_filename(workflow_id, f"v{version}")
        dst = self.archive_dir / self._workflow_filename(workflow_id, f"v{version}")

        if src.exists():
            os.replace(src, dst)

    def bump_version(self, version: str, level: str = "minor") -> str:
        """
        Bump semantic version.

        Args:
            version: Current version (e.g., "1.0")
            level: "major" or "minor"

        Returns:
            New version string
        """
        parts = version.split(".")
        major = int(parts[0])
        minor = int(parts[1]) if len(parts) > 1 else 0

        if level == "major":
            return f"{major + 1}.0"
        elif level == "minor":
            return f"{major}.{minor + 1}"
        else:
            raise ValueError("Invalid version level")

    def list_draft_workflows(self) -> List[Dict]:
        """
        Return list of all workflows in draft folder with metadata.

        Returns:
            List of workflow metadata dictionaries sorted by created_at (newest first)
        """
        workflows = []
        for file in self.draft_dir.glob("*.draft.json"):
            try:
                with open(file) as f:
                    workflow = json.load(f)
                workflows.append({
                    "workflow_id": workflow.get("workflow_id"),
                    "name": workflow.get("name", "Untitled"),
                    "execution_model": workflow.get("execution_model", "sequential"),
                    "agent_count": len(workflow.get("agents", [])),
                    "created_at": workflow.get("metadata", {}).get("created_at"),
                    "status": workflow.get("status", "draft")
                })
            except Exception:
                continue  # Skip corrupted files
        return sorted(workflows, key=lambda x: x.get("created_at") or "", reverse=True)
