"""
Chat Session Manager for Workflow Testing
Manages conversational workflow execution with session continuity
"""
import uuid
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass, field
import json
from pathlib import Path


@dataclass
class ChatMessage:
    """Single message in a chat session"""
    role: str  # "user" | "assistant" | "system"
    content: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    agent_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "agent_id": self.agent_id,
            "metadata": self.metadata
        }


@dataclass
class ChatSession:
    """Chat session for workflow testing"""
    session_id: str
    workflow_id: str
    workflow_mode: str  # "draft" | "test" | "final"
    workflow_version: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_activity: datetime = field(default_factory=datetime.utcnow)
    messages: List[ChatMessage] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    run_ids: List[str] = field(default_factory=list)  # Track all runs in this session
    state_schema: Dict[str, str] = field(default_factory=dict)  # Workflow variables schema
    variables: Dict[str, Any] = field(default_factory=dict)  # Current variable values

    def add_message(self, message: ChatMessage):
        """Add message to session history"""
        self.messages.append(message)
        self.last_activity = datetime.utcnow()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "workflow_id": self.workflow_id,
            "workflow_mode": self.workflow_mode,
            "workflow_version": self.workflow_version,
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "messages": [m.to_dict() for m in self.messages],
            "context": self.context,
            "run_ids": self.run_ids,
            "message_count": len(self.messages),
            "state_schema": self.state_schema,
            "variables": self.variables
        }


class ChatSessionManager:
    """
    Manages chat sessions for workflow testing.

    Features:
    - Session creation and retrieval
    - Message history management
    - Session persistence
    - Context tracking across executions
    """

    def __init__(self, storage_dir: str = None):
        """
        Initialize session manager.

        Args:
            storage_dir: Directory for session persistence
        """
        if storage_dir is None:
            storage_dir = Path(__file__).parent.parent / "storage" / "sessions"

        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # In-memory session cache
        self._sessions: Dict[str, ChatSession] = {}

        # Load existing sessions
        self._load_sessions()

    def create_session(
        self,
        workflow_id: str,
        workflow_mode: str = "test",
        workflow_version: Optional[str] = None,
        initial_context: Optional[Dict[str, Any]] = None,
        state_schema: Optional[Dict[str, str]] = None,
        initial_variables: Optional[Dict[str, Any]] = None
    ) -> ChatSession:
        """
        Create a new chat session.

        Args:
            workflow_id: Workflow to test
            workflow_mode: "test" or "final"
            workflow_version: Version (required if mode is "final")
            initial_context: Optional initial context
            state_schema: Workflow state schema (variable definitions)
            initial_variables: Initial values for workflow variables

        Returns:
            ChatSession instance
        """
        session_id = str(uuid.uuid4())

        # Initialize variables with defaults from state_schema
        variables = {}
        if state_schema:
            for var_name, var_type in state_schema.items():
                if var_type == "number":
                    variables[var_name] = 0
                elif var_type == "boolean":
                    variables[var_name] = False
                elif var_type == "array":
                    variables[var_name] = []
                elif var_type == "object":
                    variables[var_name] = {}
                else:
                    variables[var_name] = ""

        # Override with initial_variables if provided
        if initial_variables:
            variables.update(initial_variables)

        session = ChatSession(
            session_id=session_id,
            workflow_id=workflow_id,
            workflow_mode=workflow_mode,
            workflow_version=workflow_version,
            context=initial_context or {},
            state_schema=state_schema or {},
            variables=variables
        )

        # Add system message
        session.add_message(ChatMessage(
            role="system",
            content=f"Started testing workflow: {workflow_id}"
        ))

        self._sessions[session_id] = session
        self._save_session(session)

        return session

    def get_session(self, session_id: str) -> Optional[ChatSession]:
        """
        Get session by ID.

        Args:
            session_id: Session identifier

        Returns:
            ChatSession if found, None otherwise
        """
        if session_id in self._sessions:
            return self._sessions[session_id]

        # Try loading from disk
        session = self._load_session_from_disk(session_id)
        if session:
            self._sessions[session_id] = session

        return session

    def add_user_message(
        self,
        session_id: str,
        message: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ChatMessage:
        """
        Add user message to session.

        Args:
            session_id: Session identifier
            message: User message content
            metadata: Optional message metadata

        Returns:
            ChatMessage instance
        """
        session = self.get_session(session_id)
        if not session:
            raise ValueError(f"Session not found: {session_id}")

        chat_message = ChatMessage(
            role="user",
            content=message,
            metadata=metadata or {}
        )

        session.add_message(chat_message)
        self._save_session(session)

        return chat_message

    def add_assistant_message(
        self,
        session_id: str,
        message: str,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ChatMessage:
        """
        Add assistant message to session.

        Args:
            session_id: Session identifier
            message: Assistant response
            agent_id: Agent that generated response
            run_id: Execution run ID
            metadata: Optional message metadata

        Returns:
            ChatMessage instance
        """
        session = self.get_session(session_id)
        if not session:
            raise ValueError(f"Session not found: {session_id}")

        chat_message = ChatMessage(
            role="assistant",
            content=message,
            agent_id=agent_id,
            metadata=metadata or {}
        )

        session.add_message(chat_message)

        # Track run ID
        if run_id and run_id not in session.run_ids:
            session.run_ids.append(run_id)

        self._save_session(session)

        return chat_message

    def update_context(
        self,
        session_id: str,
        context_updates: Dict[str, Any]
    ):
        """
        Update session context.

        Args:
            session_id: Session identifier
            context_updates: Context updates to merge
        """
        session = self.get_session(session_id)
        if not session:
            raise ValueError(f"Session not found: {session_id}")

        session.context.update(context_updates)
        session.last_activity = datetime.utcnow()
        self._save_session(session)

    def update_variables(
        self,
        session_id: str,
        variable_updates: Dict[str, Any]
    ):
        """
        Update session workflow variables.

        Args:
            session_id: Session identifier
            variable_updates: Variable updates to merge
        """
        session = self.get_session(session_id)
        if not session:
            raise ValueError(f"Session not found: {session_id}")

        session.variables.update(variable_updates)
        session.last_activity = datetime.utcnow()
        self._save_session(session)

    def get_variables(self, session_id: str) -> Dict[str, Any]:
        """
        Get current workflow variables for a session.

        Args:
            session_id: Session identifier

        Returns:
            Dict of variable names to current values
        """
        session = self.get_session(session_id)
        if not session:
            raise ValueError(f"Session not found: {session_id}")

        return session.variables.copy()

    def get_execution_context(self, session_id: str) -> Dict[str, Any]:
        """
        Get full execution context for workflow execution.

        Combines session context, variables, and message history
        for passing to the executor.

        Args:
            session_id: Session identifier

        Returns:
            Dict with all context needed for execution
        """
        session = self.get_session(session_id)
        if not session:
            raise ValueError(f"Session not found: {session_id}")

        return {
            "session_id": session.session_id,
            "workflow_id": session.workflow_id,
            "workflow_mode": session.workflow_mode,
            "workflow_version": session.workflow_version,
            "context": session.context,
            "variables": session.variables,
            "state_schema": session.state_schema,
            "message_history": [m.to_dict() for m in session.messages[-10:]],  # Last 10 messages
            "run_count": len(session.run_ids)
        }

    def list_sessions(
        self,
        workflow_id: Optional[str] = None,
        limit: int = 50
    ) -> List[ChatSession]:
        """
        List chat sessions.

        Args:
            workflow_id: Filter by workflow ID
            limit: Maximum number of sessions to return

        Returns:
            List of ChatSession instances
        """
        sessions = list(self._sessions.values())

        if workflow_id:
            sessions = [s for s in sessions if s.workflow_id == workflow_id]

        # Sort by last activity
        sessions.sort(key=lambda s: s.last_activity, reverse=True)

        return sessions[:limit]

    def delete_session(self, session_id: str):
        """
        Delete a session.

        Args:
            session_id: Session identifier
        """
        if session_id in self._sessions:
            del self._sessions[session_id]

        # Delete from disk
        session_file = self.storage_dir / f"{session_id}.json"
        if session_file.exists():
            session_file.unlink()

    def clear_old_sessions(self, days: int = 7):
        """
        Clear sessions older than specified days.

        Args:
            days: Age threshold in days
        """
        from datetime import timedelta
        threshold = datetime.utcnow() - timedelta(days=days)

        to_delete = []
        for session_id, session in self._sessions.items():
            if session.last_activity < threshold:
                to_delete.append(session_id)

        for session_id in to_delete:
            self.delete_session(session_id)

    # ==================== PERSISTENCE ====================

    def _save_session(self, session: ChatSession):
        """Save session to disk."""
        session_file = self.storage_dir / f"{session.session_id}.json"

        with open(session_file, 'w') as f:
            json.dump(session.to_dict(), f, indent=2)

    def _load_session_from_disk(self, session_id: str) -> Optional[ChatSession]:
        """Load session from disk."""
        session_file = self.storage_dir / f"{session_id}.json"

        if not session_file.exists():
            return None

        try:
            with open(session_file) as f:
                data = json.load(f)

            # Reconstruct session
            session = ChatSession(
                session_id=data["session_id"],
                workflow_id=data["workflow_id"],
                workflow_mode=data["workflow_mode"],
                workflow_version=data.get("workflow_version"),
                created_at=datetime.fromisoformat(data["created_at"]),
                last_activity=datetime.fromisoformat(data["last_activity"]),
                context=data.get("context", {}),
                run_ids=data.get("run_ids", []),
                state_schema=data.get("state_schema", {}),
                variables=data.get("variables", {})
            )

            # Reconstruct messages
            for msg_data in data.get("messages", []):
                message = ChatMessage(
                    role=msg_data["role"],
                    content=msg_data["content"],
                    timestamp=datetime.fromisoformat(msg_data["timestamp"]),
                    agent_id=msg_data.get("agent_id"),
                    metadata=msg_data.get("metadata", {})
                )
                session.messages.append(message)

            return session

        except Exception as e:
            print(f"Error loading session {session_id}: {e}")
            return None

    def _load_sessions(self):
        """Load all sessions from disk on startup."""
        if not self.storage_dir.exists():
            return

        for session_file in self.storage_dir.glob("*.json"):
            session_id = session_file.stem
            session = self._load_session_from_disk(session_id)
            if session:
                self._sessions[session_id] = session
