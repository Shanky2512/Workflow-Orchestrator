"""
Session-Scoped RAG Manager

Manages per-session FAISS indexes for document retrieval in application chat.

Design principles:
    - Service-level component -- lives in echolib/service/rag/, reusable by any feature
    - Shared embeddings -- one AzureOpenAIEmbeddings instance across all sessions
    - Shared text splitter -- same config as TraditionalRAGService (chunk_size=800, chunk_overlap=200)
    - Multi-format loaders -- reuses the enhanced _load_file() pattern from TraditionalRAGService
    - No LLM -- retrieval only (the orchestrator/agents handle answering)
    - LRU cache -- in-memory dict of {session_id: FAISS} with eviction
    - Disk persistence -- save/load from {UPLOADS_DIR}/rag_indexes/{session_id}/
"""

import logging
import os
from collections import OrderedDict
from typing import Dict, List, Optional

from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


class SessionRAGManager:
    """Manages session-scoped FAISS indexes for document retrieval."""

    def __init__(self, uploads_dir: str = None, max_cached_sessions: int = 50):
        """
        Initialize with shared embeddings and text splitter.

        Credentials are read from env vars: AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY.

        Args:
            uploads_dir: Base directory for file uploads. Defaults to
                         APP_UPLOADS_DIR env var or 'apps/appmgr/uploads'.
            max_cached_sessions: Maximum number of session FAISS indexes to
                                 keep in memory before LRU eviction.
        """
        self._uploads_dir = uploads_dir or os.getenv(
            "APP_UPLOADS_DIR", os.path.join("apps", "appmgr", "uploads")
        )
        self._max_cached_sessions = max_cached_sessions

        # Shared embeddings instance (one for all sessions)
        self._embeddings = AzureOpenAIEmbeddings(
            deployment="text-embedding-ada-002",
            model="text-embedding-ada-002",
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
            api_key=os.getenv("AZURE_OPENAI_API_KEY", ""),
            chunk_size=1,
        )

        # Shared text splitter (same config as TraditionalRAGService)
        self._text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=200,
        )

        # LRU cache: OrderedDict of {session_id: FAISS}
        self._cache: OrderedDict[str, FAISS] = OrderedDict()

        logger.info("SessionRAGManager initialized (max_cached_sessions=%d)", max_cached_sessions)

    def process_document(self, session_id: str, file_path: str, filename: str) -> dict:
        """
        Chunk + embed + add to session's FAISS index.

        Uses FAISS.from_documents() for the first document in a session,
        and FAISS.merge_from() for subsequent documents. Persists the index
        to disk after each addition.

        Args:
            session_id: Chat session UUID string.
            file_path: Absolute path to the uploaded file on disk.
            filename: Original filename (used as metadata source).

        Returns:
            Dict with 'chunk_count' (int) and 'status' ('ready').

        Raises:
            ValueError: If the file type is unsupported.
            Exception: If embedding or FAISS operations fail.
        """
        logger.info(
            "Processing document for session %s: %s (%s)",
            session_id, filename, file_path,
        )

        # Load file using multi-format loader
        documents = self._load_file(file_path)

        # Set source metadata to the original filename
        for doc in documents:
            doc.metadata["source"] = filename

        # Split into chunks
        chunks = self._text_splitter.split_documents(documents)
        chunk_count = len(chunks)
        logger.info("Created %d chunks from %s", chunk_count, filename)

        if chunk_count == 0:
            logger.warning("No chunks created from %s -- file may be empty", filename)
            return {"chunk_count": 0, "status": "ready"}

        # Create new FAISS index from chunks
        new_db = FAISS.from_documents(documents=chunks, embedding=self._embeddings)

        # Get or create session index
        existing_db = self._get_or_load_index(session_id)
        if existing_db is None:
            # First document for this session
            self._cache[session_id] = new_db
        else:
            # Merge into existing index
            existing_db.merge_from(new_db)
            self._cache[session_id] = existing_db

        # Move to end (most recently used)
        self._cache.move_to_end(session_id)

        # Persist to disk
        self._save_index(session_id)

        # Evict LRU if needed
        self._evict_lru()

        logger.info(
            "Document processed: session=%s, file=%s, chunks=%d",
            session_id, filename, chunk_count,
        )
        return {"chunk_count": chunk_count, "status": "ready"}

    def retrieve(self, session_id: str, query: str, top_k: int = 10) -> str:
        """
        Similarity search against session's FAISS index.

        Args:
            session_id: Chat session UUID string.
            query: The user's query text.
            top_k: Number of top chunks to retrieve.

        Returns:
            Formatted context string with source filenames and content.
            Returns empty string if no index exists for the session.
        """
        db = self._get_or_load_index(session_id)
        if db is None:
            return ""

        # Move to end (most recently used)
        if session_id in self._cache:
            self._cache.move_to_end(session_id)

        docs = db.similarity_search(query, k=top_k)
        if not docs:
            return ""

        # Format: same as TraditionalRAGService._format_docs()
        formatted_docs = []
        for doc in docs:
            source_name = os.path.basename(doc.metadata.get("source", "Unknown"))
            content = doc.page_content
            formatted_docs.append(f"{source_name}\n{content}")

        return "\n\n".join(formatted_docs)

    def has_documents(self, session_id: str) -> bool:
        """
        Check if session has any indexed documents (in cache or on disk).

        Args:
            session_id: Chat session UUID string.

        Returns:
            True if the session has an index, False otherwise.
        """
        # Check in-memory cache first
        if session_id in self._cache:
            return True

        # Check on disk
        index_dir = self._index_dir(session_id)
        index_file = os.path.join(index_dir, "index.faiss")
        return os.path.isfile(index_file)

    def _get_or_load_index(self, session_id: str) -> Optional[FAISS]:
        """
        Load from in-memory cache or disk.

        Args:
            session_id: Chat session UUID string.

        Returns:
            FAISS index if found, None if no index exists for this session.
        """
        # Check in-memory cache
        if session_id in self._cache:
            return self._cache[session_id]

        # Try loading from disk
        index_dir = self._index_dir(session_id)
        index_file = os.path.join(index_dir, "index.faiss")
        if not os.path.isfile(index_file):
            return None

        try:
            db = FAISS.load_local(
                index_dir,
                self._embeddings,
                allow_dangerous_deserialization=True,
            )
            self._cache[session_id] = db
            self._cache.move_to_end(session_id)
            logger.info("Loaded FAISS index from disk for session %s", session_id)
            return db
        except Exception:
            logger.exception("Failed to load FAISS index from disk for session %s", session_id)
            return None

    def _save_index(self, session_id: str) -> None:
        """
        Persist session's FAISS index to disk at {uploads_dir}/rag_indexes/{session_id}/.

        Args:
            session_id: Chat session UUID string.
        """
        if session_id not in self._cache:
            return

        index_dir = self._index_dir(session_id)
        os.makedirs(index_dir, exist_ok=True)

        try:
            self._cache[session_id].save_local(index_dir)
            logger.debug("Saved FAISS index to disk for session %s", session_id)
        except Exception:
            logger.exception("Failed to save FAISS index for session %s", session_id)

    def _load_file(self, file_path: str) -> List:
        """
        Multi-format document loader. Detects type by extension.

        Supported formats:
            - .pdf: PyMuPDFLoader
            - .docx: Docx2txtLoader
            - .txt, .md, .csv: TextLoader

        Args:
            file_path: Path to the file to load.

        Returns:
            List of loaded LangChain Document objects.

        Raises:
            ValueError: If the file extension is not supported.
        """
        from langchain_community.document_loaders import PyMuPDFLoader

        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".pdf":
            loader = PyMuPDFLoader(file_path)
        elif ext == ".docx":
            from langchain_community.document_loaders import Docx2txtLoader
            loader = Docx2txtLoader(file_path)
        elif ext in (".txt", ".md", ".csv"):
            from langchain_community.document_loaders import TextLoader
            loader = TextLoader(file_path, encoding="utf-8")
        else:
            raise ValueError(f"Unsupported file type: {ext}")
        return loader.load()

    def _evict_lru(self) -> None:
        """Evict least-recently-used session from cache when max_cached_sessions exceeded."""
        while len(self._cache) > self._max_cached_sessions:
            evicted_session_id, _ = self._cache.popitem(last=False)
            logger.info("Evicted LRU session from cache: %s", evicted_session_id)

    def _index_dir(self, session_id: str) -> str:
        """Return the disk path for a session's FAISS index."""
        return os.path.join(self._uploads_dir, "rag_indexes", session_id)
