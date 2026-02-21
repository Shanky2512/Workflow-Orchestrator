"""
Checkpoint manager.
Handles workflow state persistence for resumability.
"""
from typing import Dict, Any, Optional
import json
from pathlib import Path


class CheckpointManager:
    """
    Manages workflow execution checkpoints.
    Enables pause/resume functionality.
    """

    def __init__(self, checkpoint_dir: str = None):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory for checkpoint storage
        """
        if checkpoint_dir is None:
            checkpoint_dir = Path(__file__).parent / "checkpoints"

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(
        self,
        run_id: str,
        workflow_id: str,
        current_state: Dict[str, Any]
    ) -> None:
        """
        Save workflow execution checkpoint.

        Args:
            run_id: Execution run identifier
            workflow_id: Workflow identifier
            current_state: Current execution state
        """
        checkpoint_path = self.checkpoint_dir / f"{run_id}.json"

        checkpoint_data = {
            "run_id": run_id,
            "workflow_id": workflow_id,
            "state": current_state
        }

        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint_data, f, indent=2)

    def load_checkpoint(self, run_id: str) -> Optional[Dict[str, Any]]:
        """
        Load workflow execution checkpoint.

        Args:
            run_id: Execution run identifier

        Returns:
            Checkpoint data or None if not found
        """
        checkpoint_path = self.checkpoint_dir / f"{run_id}.json"

        if not checkpoint_path.exists():
            return None

        with open(checkpoint_path) as f:
            return json.load(f)

    def delete_checkpoint(self, run_id: str) -> None:
        """
        Delete checkpoint after successful completion.

        Args:
            run_id: Execution run identifier
        """
        checkpoint_path = self.checkpoint_dir / f"{run_id}.json"

        if checkpoint_path.exists():
            checkpoint_path.unlink()
