"""
Tool Storage Module

This module provides the ToolStorage class for persisting and retrieving
tool definitions from the filesystem. It manages the storage structure,
indexing, and CRUD operations for tool metadata.

Tool definitions are stored as individual JSON files with a centralized
index for fast lookups and querying.
"""

import json
import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from echolib.types import ToolDef

logger = logging.getLogger(__name__)


class ToolStorage:
    """
    Storage manager for tool definitions.

    The ToolStorage class is responsible for:
    - Persisting tool definitions to filesystem as JSON
    - Loading tool definitions on demand or in bulk
    - Maintaining a centralized index of all tools
    - Managing storage directory structure
    - Atomic file writing for data integrity

    Storage structure:
        tools/
            tool_index.json     # Centralized tool index
            tool_xxx.json       # Individual tool definition files

    Attributes:
        storage_dir (Path): Root directory for tool storage
        index_file (Path): Path to the tool index file
    """

    def __init__(self, storage_dir: Path):
        """
        Initialize the ToolStorage.

        Args:
            storage_dir: Base directory for tool storage.
                        Creates directory if it doesn't exist.
        """
        self.storage_dir = Path(storage_dir)
        self.index_file = self.storage_dir / "tool_index.json"

        # Create storage directory if it doesn't exist
        try:
            self.storage_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"ToolStorage initialized at {self.storage_dir}")
        except OSError as e:
            logger.error(f"Failed to create storage directory: {e}")
            raise

    def save_tool(self, tool: ToolDef) -> str:
        """
        Save a tool definition to storage.

        Args:
            tool: ToolDef instance to persist

        Returns:
            str: Path to the saved tool file

        Raises:
            IOError: If file write fails
            ValueError: If tool is invalid
        """
        if not tool.tool_id:
            raise ValueError("Tool must have a tool_id")

        file_path = self.storage_dir / f"{tool.tool_id}.json"

        try:
            # Serialize tool to dict
            tool_data = tool.model_dump()
            tool_data['_saved_at'] = datetime.utcnow().isoformat()

            # Write to file atomically
            self._atomic_write(file_path, tool_data)

            # Update the index
            self._update_index(tool)

            logger.info(f"Tool '{tool.tool_id}' saved to {file_path}")
            return str(file_path)

        except Exception as e:
            logger.error(f"Failed to save tool '{tool.tool_id}': {e}")
            raise IOError(f"Failed to save tool: {e}")

    def load_tool(self, tool_id: str) -> Optional[ToolDef]:
        """
        Load a tool definition from storage.

        Args:
            tool_id: Unique identifier of the tool to load

        Returns:
            ToolDef instance if found, None otherwise
        """
        file_path = self.storage_dir / f"{tool_id}.json"

        if not file_path.exists():
            logger.debug(f"Tool file not found: {file_path}")
            return None

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Remove internal fields before creating ToolDef
            data.pop('_saved_at', None)

            return ToolDef(**data)

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in tool file {file_path}: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to load tool '{tool_id}': {e}")
            return None

    def load_all(self) -> List[ToolDef]:
        """
        Load all tool definitions from storage.

        Returns:
            List of all ToolDef instances in storage
        """
        tools = []

        # Look for all tool_*.json files (but not tool_index.json)
        for file_path in self.storage_dir.glob("*.json"):
            # Skip index file
            if file_path.name == "tool_index.json":
                continue

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Remove internal fields
                data.pop('_saved_at', None)

                tool = ToolDef(**data)
                tools.append(tool)

            except Exception as e:
                logger.warning(f"Skipping invalid tool file {file_path}: {e}")
                continue

        logger.info(f"Loaded {len(tools)} tools from storage")
        return tools

    def delete_tool(self, tool_id: str) -> bool:
        """
        Delete a tool definition from storage.

        Args:
            tool_id: Unique identifier of the tool to delete

        Returns:
            True if deletion successful, False if tool not found
        """
        file_path = self.storage_dir / f"{tool_id}.json"

        if not file_path.exists():
            logger.warning(f"Tool '{tool_id}' not found for deletion")
            return False

        try:
            # Delete the file
            file_path.unlink()

            # Remove from index
            self._remove_from_index(tool_id)

            logger.info(f"Tool '{tool_id}' deleted")
            return True

        except Exception as e:
            logger.error(f"Failed to delete tool '{tool_id}': {e}")
            return False

    def _update_index(self, tool: ToolDef) -> None:
        """
        Update the tool index with tool metadata.

        Args:
            tool: ToolDef instance to add/update in index
        """
        index = self._load_index()

        # Ensure tools dict exists
        if "tools" not in index:
            index["tools"] = {}

        # Add/update tool metadata in index
        index["tools"][tool.tool_id] = {
            "name": tool.name,
            "tool_type": tool.tool_type.value if hasattr(tool.tool_type, 'value') else str(tool.tool_type),
            "status": tool.status,
            "version": tool.version,
            "tags": tool.tags,
            "updated_at": datetime.utcnow().isoformat()
        }

        index["last_updated"] = datetime.utcnow().isoformat()
        index["tool_count"] = len(index["tools"])

        self._save_index(index)

    def _remove_from_index(self, tool_id: str) -> None:
        """
        Remove a tool from the index.

        Args:
            tool_id: Tool identifier to remove
        """
        index = self._load_index()

        if "tools" in index and tool_id in index["tools"]:
            del index["tools"][tool_id]
            index["last_updated"] = datetime.utcnow().isoformat()
            index["tool_count"] = len(index["tools"])
            self._save_index(index)

    def _load_index(self) -> Dict[str, Any]:
        """
        Load the tool index from disk.

        Returns:
            Index dictionary, empty structure if file doesn't exist
        """
        if not self.index_file.exists():
            return {
                "tools": {},
                "created_at": datetime.utcnow().isoformat(),
                "last_updated": datetime.utcnow().isoformat(),
                "tool_count": 0
            }

        try:
            with open(self.index_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to load index, creating new: {e}")
            return {
                "tools": {},
                "created_at": datetime.utcnow().isoformat(),
                "last_updated": datetime.utcnow().isoformat(),
                "tool_count": 0
            }

    def _save_index(self, index: Dict[str, Any]) -> None:
        """
        Save the tool index to disk atomically.

        Args:
            index: Index dictionary to persist
        """
        self._atomic_write(self.index_file, index)

    def _atomic_write(self, file_path: Path, data: Dict[str, Any]) -> None:
        """
        Write data to file atomically using temp file + rename.

        This ensures file integrity even if the process is interrupted.

        Args:
            file_path: Target file path
            data: Dictionary data to write as JSON
        """
        # Create temp file in same directory for atomic rename
        fd, temp_path = tempfile.mkstemp(
            dir=self.storage_dir,
            suffix='.tmp',
            prefix='tool_'
        )

        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            # Atomic rename (works on same filesystem)
            temp_path_obj = Path(temp_path)
            temp_path_obj.replace(file_path)

        except Exception as e:
            # Clean up temp file on failure
            try:
                os.unlink(temp_path)
            except OSError:
                pass
            raise IOError(f"Atomic write failed: {e}")

    def get_index(self) -> Dict[str, Any]:
        """
        Get the current tool index.

        Returns:
            Copy of the tool index dictionary
        """
        return self._load_index()

    def exists(self, tool_id: str) -> bool:
        """
        Check if a tool exists in storage.

        Args:
            tool_id: Tool identifier to check

        Returns:
            True if tool exists, False otherwise
        """
        file_path = self.storage_dir / f"{tool_id}.json"
        return file_path.exists()
