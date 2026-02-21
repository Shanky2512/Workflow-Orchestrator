"""
WebSocket Manager for Execution Transparency

Manages WebSocket connections for real-time event streaming.
Supports multiple clients per run_id and handles dead connection cleanup.

Features:
- Event buffering: Events are stored per run_id when published
- Event replay: Late-joining clients receive all buffered events
- Real-time streaming: Connected clients receive events as they occur
- Auto-cleanup: Buffers are cleaned up 60 seconds after run completion
"""

from collections import defaultdict
from fastapi import WebSocket
from typing import Dict, Set, List, Any
import asyncio
import concurrent.futures
import json
import logging

logger = logging.getLogger(__name__)


class WebSocketManager:
    """
    Manages WebSocket connections for execution transparency.

    Features:
    - Multiple clients can subscribe to the same run_id
    - Async-safe with lock for connection management
    - Automatic dead connection cleanup on broadcast
    - Event buffering for late-joining clients
    - Event replay on client connection
    - Run cleanup after execution completes (60-second delay for buffers)
    """

    def __init__(self):
        # Map of run_id -> set of connected WebSockets
        self._connections: Dict[str, Set[WebSocket]] = {}
        # Map of run_id -> list of buffered events (stored as dicts for serialization)
        self._event_buffers: Dict[str, List[Dict[str, Any]]] = {}
        # Track how many events per run_id were already sent in real-time
        # so flush_to_clients can skip them and avoid duplicates
        self._sent_count: Dict[str, int] = {}
        # Lock for thread-safe connection and buffer management
        self._lock = asyncio.Lock()
        # Reference to the main event loop (captured from async context)
        self._main_loop: asyncio.AbstractEventLoop = None
        # Map of run_id -> set of asyncio.Queue subscribers for SSE streaming
        self._queue_subscribers: Dict[str, Set[asyncio.Queue]] = defaultdict(set)

    async def connect(self, run_id: str, websocket: WebSocket) -> None:
        """
        Subscribe a WebSocket to execution events for a run.

        Accepts the connection, registers it, and replays all buffered
        events to ensure the client receives complete execution history.

        Args:
            run_id: Run identifier to subscribe to
            websocket: WebSocket connection to register
        """
        await websocket.accept()
        async with self._lock:
            if run_id not in self._connections:
                self._connections[run_id] = set()
            self._connections[run_id].add(websocket)

        logger.info(f"WebSocket connected for run {run_id}")

        # Replay buffered events to the newly connected client
        await self._replay_events(run_id, websocket)

    async def _replay_events(self, run_id: str, websocket: WebSocket) -> None:
        """
        Replay all buffered events to a newly connected client.

        This ensures late-joining clients receive complete execution history.
        Events are sent in the order they were originally published.

        Args:
            run_id: Run identifier
            websocket: WebSocket connection to replay events to
        """
        # Get a copy of buffered events while holding the lock
        async with self._lock:
            events = self._event_buffers.get(run_id, []).copy()

        if not events:
            logger.debug(f"No buffered events to replay for run {run_id}")
            return

        logger.info(f"Replaying {len(events)} buffered events for run {run_id}")

        for event in events:
            try:
                message = json.dumps(event)
                await websocket.send_text(message)
            except Exception as e:
                logger.warning(f"Failed to replay event to WebSocket: {e}")
                # Stop replay if connection is broken
                break

    def subscribe_queue(self, run_id: str, queue: asyncio.Queue) -> None:
        """
        Subscribe an asyncio.Queue to receive events for a run_id.

        Events are put via put_nowait (non-blocking). If the queue is
        full the event is silently dropped to avoid blocking the
        LangGraph execution thread.

        Args:
            run_id: Run identifier to subscribe to.
            queue: asyncio.Queue to receive events.
        """
        self._queue_subscribers[run_id].add(queue)
        logger.debug(
            "Queue subscribed for run %s (total: %d)",
            run_id, len(self._queue_subscribers[run_id]),
        )

    def unsubscribe_queue(self, run_id: str, queue: asyncio.Queue) -> None:
        """
        Unsubscribe an asyncio.Queue from a run_id.

        Safe to call even if the queue was never subscribed.

        Args:
            run_id: Run identifier.
            queue: asyncio.Queue to remove.
        """
        if run_id in self._queue_subscribers:
            self._queue_subscribers[run_id].discard(queue)
            if not self._queue_subscribers[run_id]:
                del self._queue_subscribers[run_id]
            logger.debug("Queue unsubscribed for run %s", run_id)

    async def disconnect(self, run_id: str, websocket: WebSocket) -> None:
        """
        Unsubscribe a WebSocket from execution events.

        Note: This does NOT clean up the event buffer. Buffers are cleaned
        up separately via schedule_buffer_cleanup() after run completion.

        Args:
            run_id: Run identifier
            websocket: WebSocket connection to remove
        """
        async with self._lock:
            if run_id in self._connections:
                self._connections[run_id].discard(websocket)
                if not self._connections[run_id]:
                    del self._connections[run_id]
        logger.info(f"WebSocket disconnected for run {run_id}")

    async def broadcast(self, run_id: str, event) -> None:
        """
        Broadcast event to all subscribers of a run.

        Events are buffered first to support late-joining clients,
        then broadcast to all currently connected clients.

        Args:
            run_id: Run identifier
            event: ExecutionEvent to broadcast (must have to_dict() method)
        """
        # Capture the main event loop for use by sync code
        if self._main_loop is None:
            self._main_loop = asyncio.get_running_loop()
            logger.info("Captured main event loop for sync-to-async communication")

        # Serialize event to dict for buffering and sending
        event_dict = event.to_dict() if hasattr(event, 'to_dict') else event

        # Buffer the event and get current connections
        async with self._lock:
            # Initialize buffer if needed
            if run_id not in self._event_buffers:
                self._event_buffers[run_id] = []
            # Store event in buffer
            self._event_buffers[run_id].append(event_dict)
            # Get copy of connections
            connections = self._connections.get(run_id, set()).copy()

        if not connections:
            logger.debug(f"No WebSocket connections for run {run_id}, event buffered")
            return

        # Serialize event to JSON for sending
        message = json.dumps(event_dict)
        dead_connections = []

        # Send to all connected clients
        for ws in connections:
            try:
                await ws.send_text(message)
            except Exception as e:
                logger.warning(f"Failed to send to WebSocket: {e}")
                dead_connections.append(ws)

        # Clean up dead connections
        if dead_connections:
            async with self._lock:
                for ws in dead_connections:
                    if run_id in self._connections:
                        self._connections[run_id].discard(ws)

        # Forward event to queue subscribers (SSE streaming)
        for q in list(self._queue_subscribers.get(run_id, set())):
            try:
                q.put_nowait(event_dict)
            except asyncio.QueueFull:
                pass  # slow consumer -- drop silently, never block

    def buffer_and_send_sync(self, run_id: str, event) -> None:
        """
        Synchronously buffer an event AND send it to connected clients immediately.

        This method BLOCKS until the event is sent, ensuring real-time streaming
        of step events during workflow execution.

        Args:
            run_id: Run identifier
            event: ExecutionEvent to send (must have to_dict() method or be a dict)
        """
        # Serialize event to dict
        event_dict = event.to_dict() if hasattr(event, 'to_dict') else event

        # Buffer the event for late-joining clients
        if run_id not in self._event_buffers:
            self._event_buffers[run_id] = []
        self._event_buffers[run_id].append(event_dict)

        # Get connected clients
        connections = self._connections.get(run_id, set()).copy()
        if not connections:
            logger.debug(f"No connections for run {run_id}, event buffered only")
            return

        # Send to all connected clients using blocking approach
        message = json.dumps(event_dict)
        logger.info(f"Sending event {event_dict.get('event', 'unknown')} to {len(connections)} client(s) for run {run_id}")

        # Use the captured main event loop
        if self._main_loop is None or not self._main_loop.is_running():
            logger.warning("Main event loop not available, cannot send event in real-time")
            return

        # Schedule the async send and WAIT for it (blocking)
        sent = False
        for ws in connections:
            try:
                future = asyncio.run_coroutine_threadsafe(
                    self._send_event_to_client(ws, message),
                    self._main_loop
                )
                # Wait for the send to complete (with timeout)
                future.result(timeout=5.0)
                logger.debug(f"Successfully sent event to client")
                sent = True
            except concurrent.futures.TimeoutError:
                logger.warning(f"Timeout sending event to WebSocket for run {run_id}")
            except Exception as e:
                logger.warning(f"Error sending event to WebSocket for run {run_id}: {e}")

        # Track that this event was sent in real-time so flush skips it
        if sent:
            self._sent_count[run_id] = self._sent_count.get(run_id, 0) + 1

        # Forward event to queue subscribers (SSE streaming).
        # put_nowait is sync-safe -- no await needed from LangGraph thread.
        for q in list(self._queue_subscribers.get(run_id, set())):
            try:
                q.put_nowait(event_dict)
            except asyncio.QueueFull:
                pass  # slow consumer -- drop silently, never block LangGraph thread

    async def _send_event_to_client(self, websocket, message: str) -> None:
        """Helper coroutine to send a message to a WebSocket client."""
        await websocket.send_text(message)

    # Keep old method name as alias for compatibility
    def buffer_event_sync(self, run_id: str, event) -> None:
        """Alias for buffer_and_send_sync for backward compatibility."""
        self.buffer_and_send_sync(run_id, event)

    async def flush_to_clients(self, run_id: str) -> None:
        """
        Send all buffered step events to connected clients.

        Call this before sending run_completed to ensure all step events
        are delivered before the final event.

        Note: Skips run_started event since it's already sent via broadcast().

        Args:
            run_id: Run identifier
        """
        async with self._lock:
            events = self._event_buffers.get(run_id, []).copy()
            connections = self._connections.get(run_id, set()).copy()

        if not connections:
            logger.debug(f"No connections to flush events to for run {run_id}")
            return

        if not events:
            logger.debug(f"No events to flush for run {run_id}")
            return

        # Filter to only step events (skip run_started which was already broadcast)
        step_events = [e for e in events if e.get('event') in ('step_started', 'step_completed', 'step_failed', 'step_output')]

        if not step_events:
            logger.debug(f"No step events to flush for run {run_id}")
            return

        # Skip events that were already sent in real-time by buffer_and_send_sync
        # to prevent the frontend from receiving duplicate step messages
        already_sent = self._sent_count.get(run_id, 0)
        unsent_events = step_events[already_sent:]

        if not unsent_events:
            logger.debug(f"All {len(step_events)} step events already sent in real-time for run {run_id}, nothing to flush")
            return

        logger.info(f"Flushing {len(unsent_events)} unsent step events (skipped {already_sent} already-sent) to {len(connections)} client(s) for run {run_id}")

        for event_dict in unsent_events:
            message = json.dumps(event_dict)
            for ws in connections:
                try:
                    await ws.send_text(message)
                except Exception as e:
                    logger.warning(f"Failed to flush event to WebSocket for run {run_id}: {e}")

    def cleanup_run(self, run_id: str) -> None:
        """
        Clean up all connections and buffers for a completed run.

        Note: This is synchronous for use in cleanup callbacks.
        Connections are simply removed from tracking; they'll be
        closed by the client or timeout.

        Args:
            run_id: Run identifier to clean up
        """
        # Use blocking approach for cleanup callback
        if run_id in self._connections:
            del self._connections[run_id]
            logger.debug(f"Cleaned up WebSocket connections for run {run_id}")
        if run_id in self._event_buffers:
            del self._event_buffers[run_id]
            logger.debug(f"Cleaned up event buffer for run {run_id}")
        self._sent_count.pop(run_id, None)

    async def async_cleanup_run(self, run_id: str) -> None:
        """
        Async version of cleanup_run for use in async contexts.

        Cleans up connections immediately but schedules buffer cleanup
        with a 60-second delay to allow late-joining clients to receive
        buffered events.

        Args:
            run_id: Run identifier to clean up
        """
        async with self._lock:
            if run_id in self._connections:
                del self._connections[run_id]
                logger.debug(f"Cleaned up WebSocket connections for run {run_id}")

        # Schedule buffer cleanup with 60-second delay
        self._schedule_buffer_cleanup(run_id)

    def _schedule_buffer_cleanup(self, run_id: str) -> None:
        """
        Schedule cleanup of event buffer after 60 seconds.

        This delay allows late-joining clients to still receive
        buffered events even after the run has completed.

        Args:
            run_id: Run identifier to clean up
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.call_later(60, lambda: self._cleanup_buffer(run_id))
                logger.debug(f"Scheduled buffer cleanup for run {run_id} in 60 seconds")
            else:
                # Fallback: create task for cleanup
                asyncio.create_task(self._async_cleanup_buffer(run_id))
        except RuntimeError:
            # No event loop - cleanup immediately or skip
            logger.warning(f"No event loop for buffer cleanup scheduling of run {run_id}")
            self._cleanup_buffer(run_id)

    async def _async_cleanup_buffer(self, run_id: str) -> None:
        """Async cleanup of buffer after delay."""
        await asyncio.sleep(60)
        self._cleanup_buffer(run_id)

    def _cleanup_buffer(self, run_id: str) -> None:
        """
        Remove event buffer and sent count from memory.

        Args:
            run_id: Run identifier to remove buffer for
        """
        if run_id in self._event_buffers:
            del self._event_buffers[run_id]
            logger.debug(f"Cleaned up event buffer for run {run_id}")
        self._sent_count.pop(run_id, None)

    def get_connection_count(self, run_id: str) -> int:
        """
        Get number of connected clients for a run.

        Args:
            run_id: Run identifier

        Returns:
            Number of connected WebSocket clients
        """
        return len(self._connections.get(run_id, set()))

    def has_connections(self, run_id: str) -> bool:
        """
        Check if a run has any connected clients.

        Args:
            run_id: Run identifier

        Returns:
            True if there are connected clients
        """
        return run_id in self._connections and len(self._connections[run_id]) > 0

    def has_buffered_events(self, run_id: str) -> bool:
        """
        Check if a run has any buffered events.

        Args:
            run_id: Run identifier

        Returns:
            True if there are buffered events
        """
        return run_id in self._event_buffers and len(self._event_buffers[run_id]) > 0

    def get_buffer_size(self, run_id: str) -> int:
        """
        Get number of buffered events for a run.

        Args:
            run_id: Run identifier

        Returns:
            Number of buffered events
        """
        return len(self._event_buffers.get(run_id, []))


# Singleton instance for global access
ws_manager = WebSocketManager()
