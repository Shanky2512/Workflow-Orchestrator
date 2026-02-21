"""
Storage layer for persisting MCP connectors.
Supports JSON file storage and future extensibility to databases.
"""

from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import json
import logging
from datetime import datetime
from .base import TransportType  # kept as-is if used elsewhere

logger = logging.getLogger(__name__)


class StorageError(Exception):
    """Raised when storage operations fail"""
    pass


# storage.py
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import os
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

ECHOLIB_ANCHOR = "echolib"               # folder anchor (case-insensitive)
REL_CONNECTORS = ["Get_connector", "Get_MCP", "connectors"]  # relative under echolib


def _find_echolib_root(start: Path) -> Optional[Path]:
    """
    Walk up from 'start' until we find a folder named 'echolib' (case-insensitive).
    Return that path or None if not found.
    """
    current = start
    target_lower = ECHOLIB_ANCHOR.lower()
    for parent in [current] + list(current.parents):
        if parent.name.lower() == target_lower:
            return parent
    return None


class ConnectorStorage:
    def __init__(self, storage_dir: Optional[Union[str, Path]] = None):
        """
        Resolve storage directory in this order:
        1) MCP_STORAGE_DIR environment variable (if set)
        2) explicit storage_dir param (if provided)
        3) <echolib>/Get_connector/Get_MCP/connectors (discovered by walking up from this file)
        4) fallback: <this_file_dir>/Get_connector/Get_MCP/connectors
        """
        # 1) ENV override
        env_dir = os.getenv("MCP_STORAGE_DIR")
        if env_dir:
            self.storage_dir = Path(env_dir).expanduser().resolve()
        elif storage_dir:
            # 2) explicit param
            self.storage_dir = Path(storage_dir).expanduser().resolve()
        else:
            # 3) discover echolib root and build the relative path under it
            here = Path(__file__).resolve().parent
            echolib_root = _find_echolib_root(here)
            if echolib_root:
                self.storage_dir = (echolib_root / Path(*REL_CONNECTORS)).resolve()
            else:
                # 4) fallback if echolib not found in parents (shouldn’t happen normally)
                self.storage_dir = (here / Path(*REL_CONNECTORS)).resolve()

        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._index_file = self.storage_dir / "_index.json"
        self._ensure_index()

        logger.info(f"[ConnectorStorage] storage_dir = {self.storage_dir}")
        logger.info(f"[ConnectorStorage] index_file  = {self._index_file}")

    def _ensure_index(self):
        """Ensure index file exists"""
        if not self._index_file.exists():
            self._write_index({})

    def _read_index(self) -> Dict[str, Any]:
        """Read connector index"""
        try:
            with open(self._index_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to read index: {e}")
            return {}

    def _write_index(self, index: Dict[str, Any]):
        """Write connector index"""
        try:
            with open(self._index_file, 'w', encoding='utf-8') as f:
                json.dump(index, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to write index: {e}")
            raise StorageError(f"Failed to write index: {e}")

    def _get_connector_path(self, connector_id: str) -> Path:
        """Get file path for connector"""
        return self.storage_dir / f"{connector_id}.json"

    
    def save(
        self,
        connector_data: Dict[str, Any],
        creation_payload: Optional[Dict[str, Any]] = None,
        validation_status: str = "not_tested",
        validation_error: Optional[str] = None,
        tested_at: Optional[str] = None
    ) -> bool:
        """
        Save connector with metadata.

        Behavior:
        - If connector_data already matches the minimal, API-style schema
            (has 'connector_name' and 'creation_payload' at top level, and
            does NOT have 'connector_config'), we write it as-is (adding/merging
            timestamps & validation fields).
        - Otherwise, we transform the incoming data to the minimal schema
            to avoid duplication and 'connector_config' bloat.
        """
        try:
            connector_id = connector_data.get("connector_id")
            if not connector_id:
                raise ValueError("connector_id required")

            now = datetime.utcnow().isoformat()

            # Preserve created_at if already present on disk
            existing = self.load(connector_id)
            created_at = existing.get("created_at", now) if existing else now

            # Detect if data is already in the minimal shape (API-like):
            # - top-level has connector_name and creation_payload
            # - and does NOT have connector_config
            is_minimal_shape = (
                "connector_name" in connector_data and
                "creation_payload" in connector_data and
                "connector_config" not in connector_data
            )

            if is_minimal_shape:
                # Use the provided minimal dict as the base, then merge/overwrite standard fields
                storage_data = dict(connector_data)  # shallow copy
                # Allow caller to pass validation params as call-time args; we normalize/overwrite here
                if validation_status is not None:
                    storage_data["validation_status"] = validation_status
                if "validation_status" not in storage_data:
                    storage_data["validation_status"] = "not_tested"

                storage_data["validation_error"] = validation_error
                storage_data["tested_at"] = tested_at
                storage_data["created_at"] = created_at
                storage_data["updated_at"] = now

            else:
                # Legacy / full shape -> convert to minimal to avoid duplication.
                # Try to derive a connector_name and get a proper creation_payload.
                connector_name = (
                    connector_data.get("connector_name") or
                    connector_data.get("name") or
                    connector_data.get("connector_config", {}).get("name")
                )

                # Prefer explicit creation_payload arg, else derive from input
                # (for MCP this is your normalized config, including example_payload)
                final_creation_payload = creation_payload if creation_payload is not None else (
                    connector_data.get("creation_payload") or connector_data
                )

                storage_data = {
                    "connector_id": connector_id,
                    "connector_name": connector_name,
                    "creation_payload": final_creation_payload,
                    "validation_status": validation_status,
                    "validation_error": validation_error,
                    "tested_at": tested_at,
                    "created_at": created_at,
                    "updated_at": now
                }

            # Write file
            filepath = self._get_connector_path(connector_id)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(storage_data, f, indent=2, ensure_ascii=False)

            return True

        except Exception as e:
            logger.error(f"Failed to save connector: {e}")
            return False


    def load(self, connector_id: str) -> Optional[Dict[str, Any]]:
        """Load connector with all metadata (new/minimal format friendly)."""
        try:
            filepath = self._get_connector_path(connector_id)
            if not filepath.exists():
                return None

            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # --- Normalize transport_type where it may live in new formats ---
            # 1) creation_payload.transport_type (your primary case)
            cp = data.get("creation_payload")
            if isinstance(cp, dict):
                tt = cp.get("transport_type")
                if isinstance(tt, dict):
                    # Try common keys; fallback to any first value
                    cp["transport_type"] = (
                        tt.get("value")
                        or tt.get("name")
                        or next(iter(tt.values()), None)
                    )

            # 2) connector_config.transport_type (some files may carry it here)
            cc = data.get("connector_config")
            if isinstance(cc, dict):
                tt = cc.get("transport_type")
                if isinstance(tt, dict):
                    cc["transport_type"] = (
                        tt.get("value")
                        or tt.get("name")
                        or next(iter(tt.values()), None)
                    )

            # --- Ensure some metadata keys exist (nice-to-have defaults) ---
            data.setdefault("validation_status", "not_tested")
            data.setdefault("validation_error", None)
            data.setdefault("tested_at", None)

            # Return as-is for new format
            return data

        except Exception as e:
            # You can log here if needed
            return None

    def delete(self, connector_id: str) -> bool:
        """
        Delete connector from storage.

        Args:
            connector_id: ID of connector to delete

        Returns:
            True if successful
        """
        try:
            connector_path = self._get_connector_path(connector_id)

            if not connector_path.exists():
                logger.warning(f"Connector not found for deletion: {connector_id}")
                return False

            # Delete file
            connector_path.unlink()

            # Update index
            index = self._read_index()
            if connector_id in index:
                del index[connector_id]
                self._write_index(index)

            logger.info(f"Deleted connector {connector_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete connector: {e}", exc_info=True)
            raise StorageError(f"Failed to delete connector: {e}")

    def list_all(self) -> List[Dict[str, Any]]:
        """List all stored connectors with metadata."""
        connectors: List[Dict[str, Any]] = []

        try:
            for filepath in self.storage_dir.glob("*.json"):
                if filepath.name == "_index.json":
                    continue
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    # Handle both old and new formats
                    if "connector_config" in data:
                        # New format - extract key info
                        connector_info = {
                            "connector_id": data.get("connector_id"),
                            "name": data.get("connector_config", {}).get("name"),
                            "transport_type": data.get("connector_config", {}).get("transport_type"),
                            "example_payload" : data.get("example_payload"),
                            "creation_payload" : data.get("creation_payload"),
                            "validation_status": data.get("validation_status", "not_tested"),
                            "created_at": data.get("created_at"),
                            "updated_at": data.get("updated_at")
                        }
                    else:
                        # Old format
                        transport = data.get("transport_type")
                        if isinstance(transport, dict):
                            transport = transport.get("value", transport)

                        connector_info = {
                            "connector_id": data.get("connector_id"),
                            "name": data.get("creation_payload", {}).get("name"),
                            "validation_status": data.get("validation_status"),
                            "example_payload" : data.get("creation_payload", {}).get("example_payload"),
                            "creation_payload" : data.get("creation_payload"),
                            "created_at": data.get("created_at"),
                            "updated_at": data.get("updated_at")
                        }

                    connectors.append(connector_info)

                except Exception as e:
                    logger.error(f"Failed to load connector from {filepath}: {e}")
                    continue

            return connectors

        except Exception as e:
            logger.error(f"Failed to list connectors: {e}")
            return []

    def exists(self, connector_id: str) -> bool:
        """Check if connector exists"""
        return self._get_connector_path(connector_id).exists()

    def get_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Find connector by name.

        Args:
            name: Connector name

        Returns:
            Connector data or None
        """
        try:
            index = self._read_index()

            for connector_id, metadata in index.items():
                if metadata.get("name") == name:
                    return self.load(connector_id)

            return None

        except Exception as e:
            logger.error(f"Failed to get connector by name: {e}", exc_info=True)
            return None

    def update(
        self,
        connector_id: str,
        validation_status: Optional[str] = None,
        validation_error: Optional[str] = None,
        tested_at: Optional[str] = None
    ) -> bool:
        """Update connector metadata without changing config."""
        try:
            data = self.load(connector_id)
            if not data:
                return False

            now = datetime.utcnow().isoformat()

            # Update only provided fields
            if validation_status is not None:
                data["validation_status"] = validation_status
            if validation_error is not None:
                data["validation_error"] = validation_error
            if tested_at is not None:
                data["tested_at"] = tested_at

            data["updated_at"] = now

            # Save back
            filepath = self._get_connector_path(connector_id)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            return True

        except Exception as e:
            logger.error(f"Failed to update connector {connector_id}: {e}")
            return False

    def export_all(self, export_path: str) -> bool:
        """
        Export all connectors to a single JSON file.

        Useful for backups or migration.
        """
        try:
            connectors: List[Dict[str, Any]] = []
            for metadata in self.list_all():
                connector_id = metadata["connector_id"]
                connector_data = self.load(connector_id)
                if connector_data:
                    connectors.append(connector_data)

            export_data = {
                "export_date": datetime.utcnow().isoformat(),
                "connector_count": len(connectors),
                "connectors": connectors
            }

            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)

            logger.info(f"Exported {len(connectors)} connectors to {export_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to export connectors: {e}", exc_info=True)
            raise StorageError(f"Failed to export connectors: {e}")

    def import_all(self, import_path: str, overwrite: bool = False) -> int:
        """
        Import connectors from exported JSON file.

        Args:
            import_path: Path to export file
            overwrite: Whether to overwrite existing connectors

        Returns:
            Number of connectors imported
        """
        try:
            with open(import_path, 'r', encoding='utf-8') as f:
                import_data = json.load(f)

            connectors = import_data.get("connectors", [])
            imported_count = 0

            for connector_data in connectors:
                connector_id = connector_data["connector_id"]

                if self.exists(connector_id) and not overwrite:
                    logger.warning(f"Skipping existing connector: {connector_id}")
                    continue

                self.save(connector_data)
                imported_count += 1

            logger.info(f"Imported {imported_count} connectors from {import_path}")
            return imported_count

        except Exception as e:
            logger.error(f"Failed to import connectors: {e}", exc_info=True)
            raise StorageError(f"Failed to import connectors: {e}")


# ---------- Singleton helpers (module-level) ----------
_storage_instance: Optional[ConnectorStorage] = None
_storage_path: Optional[Path] = None

def get_storage(storage_dir: Optional[Union[str, Path]] = None) -> ConnectorStorage:
    """
    Singleton getter. If a different path is passed than before, recreate.
    If storage_dir is None, ConnectorStorage defaults to:
    <this file dir>/connectors  (i.e., echolib/Get_connector/Get_MCP/connectors)
    """
    global _storage_instance, _storage_path
    desired = Path(storage_dir) if storage_dir else None

    if _storage_instance is None or (desired and desired != _storage_path):
        _storage_path = desired
        _storage_instance = ConnectorStorage(storage_dir=desired)

    return _storage_instance

def reset_storage(storage_dir: Optional[Union[str, Path]] = None) -> ConnectorStorage:
    """
    Force a fresh instance at the desired path (use from MCPConnector.__init__).
    """
    global _storage_instance, _storage_path
    _storage_instance = None
    _storage_path = None
    return get_storage(storage_dir)