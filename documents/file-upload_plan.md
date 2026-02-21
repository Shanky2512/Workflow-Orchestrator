# RAG File Upload Integration into App Chat — Implementation Plan (v2)

## Context

The apps feature has a working orchestration pipeline (16-step flow in `pipeline.py`) that runs workflows and agents. Users can upload files via `POST /{id}/chat/upload`, but the upload **only saves the file to disk and creates a DB record with `status=pending`** — no processing happens. There is no chunking, no embedding, no vector storage, and no retrieval at query time.

**Goal**: When a user uploads a file in the app chat (e.g., a resume PDF), it should be immediately chunked, embedded, and stored in a session-scoped FAISS index. On every subsequent chat message, relevant chunks are retrieved via similarity search and injected as context into the orchestration pipeline so workflows/agents can use the document content.

**Decisions confirmed by user**:
- Embeddings: Azure OpenAI (`text-embedding-ada-002`)
- Scope: Session-scoped (each chat session gets its own FAISS index, like ChatGPT)
- RAG is a **service-level concern** — lives in `echolib/service/rag/`, not tied to the apps feature
- Currently Traditional RAG only; architecture supports future Hybrid/Graph RAG

---

## Existing RAG Infrastructure (already in codebase)

| File | What it provides |
|------|-----------------|
| `echolib/service/rag/trad_rag.py` | `TraditionalRAGService` — chunking, FAISS embedding, save/load index, query answering |
| `echolib/service/rag/vector_store.py` | `VectorDocumentStore` — low-level FAISS + OpenAI embeddings |
| `echolib/service/rag/graph_rag_service.py` | `GraphRAGService` — entity/relationship extraction (future use) |
| `echolib/service/rag/hybrid_rag_service.py` | `HybridRAGService` — combines vector + graph (future use) |
| `echolib/services.py` | `RAGService` class — already exists (line 84) |
| `apps/rag/container.py` | Registers `rag.traditional_service` in global DI container |
| `apps/rag/routes.py` | Standalone RAG API endpoints (`/rag/traditional/load`, `/query`, `/stats`) |

---

## Files to Create

### 1. `echolib/service/rag/session_rag_manager.py` (NEW)

**Session-scoped RAG manager** — a service-level component that manages per-session FAISS indexes.

Design principles:
- **Service-level** — lives in `echolib/service/rag/`, reusable by any feature
- **Shared embeddings** — one `AzureOpenAIEmbeddings` instance shared across all sessions (not per-session)
- **Shared text splitter** — same config as `TraditionalRAGService` (chunk_size=800, chunk_overlap=200)
- **Multi-format loaders** — reuses the enhanced `_load_file()` from `TraditionalRAGService`
- **No LLM** — retrieval only (the orchestrator/agents handle answering)
- **LRU cache** — in-memory dict of `{session_id: FAISS}` with eviction
- **Disk persistence** — save/load from `{UPLOADS_DIR}/rag_indexes/{session_id}/`

```python
class SessionRAGManager:
    """Manages session-scoped FAISS indexes for document retrieval."""

    def __init__(self, uploads_dir: str = None, max_cached_sessions: int = 50):
        """
        Initialize with shared embeddings and text splitter.
        Credentials from env vars: AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY.
        """

    def process_document(self, session_id: str, file_path: str, filename: str) -> dict:
        """
        Chunk + embed + add to session's FAISS index.
        Uses FAISS.from_documents() for first doc, FAISS.merge_from() for subsequent.
        Persists index to disk after each addition.
        Returns {"chunk_count": N, "status": "ready"}.
        """

    def retrieve(self, session_id: str, query: str, top_k: int = 10) -> str:
        """
        Similarity search against session's FAISS index.
        Returns formatted context string with source filenames and content.
        """

    def has_documents(self, session_id: str) -> bool:
        """Check if session has any indexed documents (in cache or on disk)."""

    def _get_or_load_index(self, session_id: str) -> Optional[FAISS]:
        """Load from in-memory cache or disk. Returns None if no index exists."""

    def _save_index(self, session_id: str) -> None:
        """Persist session's FAISS index to disk at {uploads_dir}/rag_indexes/{session_id}/."""

    def _load_file(self, file_path: str) -> List:
        """
        Multi-format document loader. Detects type by extension:
        - .pdf → PyMuPDFLoader
        - .docx → Docx2txtLoader
        - .txt, .md, .csv → TextLoader
        """

    def _evict_lru(self) -> None:
        """Evict least-recently-used session from cache when max_cached_sessions exceeded."""
```

Key implementation details:
- Uses `RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)`
- Uses `AzureOpenAIEmbeddings(deployment="text-embedding-ada-002")` with env var credentials
- Index path: `{UPLOADS_DIR}/rag_indexes/{session_id}/index.faiss` + `index.pkl`
- `FAISS.load_local(..., allow_dangerous_deserialization=True)` for loading persisted indexes
- Formatted retrieval output: `"{filename}\n{chunk_content}"` joined by `"\n\n"` (same format as `TraditionalRAGService._format_docs()`)

---

## Files to Modify

### 2. `echolib/service/rag/trad_rag.py` — Add Multi-Format Loader Support + Merge

**Current**: `_load_documents()` only supports PDF via `PyMuPDFLoader`.

**Changes**:

a) **Add multi-format `_load_file()` method** — detects file type by extension:
```python
def _load_file(self, file_path: str) -> List:
    """Load a single file with format-appropriate loader."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.pdf':
        loader = PyMuPDFLoader(file_path)
    elif ext == '.docx':
        from langchain_community.document_loaders import Docx2txtLoader
        loader = Docx2txtLoader(file_path)
    elif ext in ('.txt', '.md', '.csv'):
        from langchain_community.document_loaders import TextLoader
        loader = TextLoader(file_path, encoding='utf-8')
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    return loader.load()
```

b) **Update `_load_documents()`** — for single files, use `_load_file()` instead of hardcoded `PyMuPDFLoader`:
```python
def _load_documents(self, path: str) -> List:
    if os.path.isfile(path):
        return self._load_file(path)
    elif os.path.isdir(path):
        # DirectoryLoader with glob for supported extensions
        ...
```

c) **Add `add_documents()` method** — merges new documents into an existing FAISS index:
```python
def add_documents(self, path: str) -> Dict[str, Any]:
    """Load and merge additional documents into existing index using FAISS.merge_from()."""
    documents = self._load_documents(path)
    chunks = self._split_documents(documents)
    new_db = FAISS.from_documents(documents=chunks, embedding=self.embeddings)
    if self.db is None:
        self.db = new_db
    else:
        self.db.merge_from(new_db)
    self.retriever = self.db.as_retriever(search_kwargs={"k": self.retrieval_k})
    # Update stats...
    return {"chunks_added": len(chunks), ...}
```

### 3. `echolib/types.py` — Add Traditional RAG Types

**Add** the following types (required by `apps/rag/routes.py` which already imports them):

```python
# ==================== TRADITIONAL RAG TYPES ====================

class TraditionalRAGLoadRequest(BaseModel):
    path: str = Field(..., description="File path or directory path to load documents from")

class TraditionalRAGLoadResponse(BaseModel):
    status: str = Field(..., description="Load status (success/error)")
    documents_loaded: int = Field(default=0)
    chunks_created: int = Field(default=0)
    sources: List[str] = Field(default_factory=list)
    path: str = Field(..., description="Upload summary or path that was loaded")

class TraditionalRAGQueryRequest(BaseModel):
    query: str = Field(..., description="User query/question")

class TraditionalRAGQueryResponse(BaseModel):
    query: str = Field(..., description="Original query")
    answer: str = Field(..., description="Generated answer")
    sources: List[str] = Field(default_factory=list)
    chunks_used: int = Field(default=0)

class TraditionalRAGStats(BaseModel):
    initialized: bool = Field(default=False)
    documents_loaded: int = Field(default=0)
    chunks_indexed: int = Field(default=0)
    sources: List[str] = Field(default_factory=list)
    config: Dict[str, Any] = Field(default_factory=dict)
```

### 4. `apps/rag/container.py` — Register SessionRAGManager

**Add** registration of `SessionRAGManager` alongside existing RAG services:

```python
from echolib.service.rag.session_rag_manager import SessionRAGManager

# Session-scoped RAG (for app chat file uploads)
_session_rag_manager = SessionRAGManager()
container.register('rag.session_manager', lambda: _session_rag_manager)
```

### 5. `apps/appmgr/routes.py` — Upload Endpoint (lines 907-979)

**Current**: Saves file to disk, creates DB record with `status=pending`, returns immediately.

**Change**: After saving the file, **trigger RAG processing** (chunk + embed + index). Update the DB record with `status=ready` and `chunk_count`.

```
Current flow:
  save file → create DB record (pending) → return

New flow:
  save file → create DB record (processing)
  → rag_manager.process_document(session_id, file_path, filename)
  → update DB record (ready, chunk_count=N)
  → return (with chunk_count in response)
```

Resolve RAG manager from global container:
```python
from echolib.di import container
rag_manager = container.resolve('rag.session_manager')
```

- If `session_id` is not provided, auto-create a new session
- If processing fails → set `status=failed`, return error detail

### 6. `apps/appmgr/schemas.py` — Extend Response Schema

**Modify `DocumentUploadResponse`**: Add `chunk_count: int = 0` field.

### 7. `echolib/repositories/application_chat_repo.py` — Add Document Query Methods

**Add two new methods** to `ApplicationChatRepository`:

```python
async def get_session_documents(self, db, chat_session_id, status=None) -> List[ApplicationDocument]:
    """Get all documents for a session, optionally filtered by status."""

async def update_document_status(self, db, document_id, status, chunk_count=None):
    """Update document processing_status and chunk_count."""
```

### 8. `apps/appmgr/orchestrator/orchestrator.py` — Add Document Context to System Prompt

**Modify `plan()` signature**: Add `document_context: Optional[str] = None` parameter.

**Modify `_build_system_prompt()`**: Add a new section to the system prompt template:

```python
# Add between guardrail_section and skill_manifest_section:
{document_context_section}
```

Where `document_context_section`:
```
## Uploaded Document Context

The user has uploaded documents in this conversation. The following relevant
excerpts were retrieved based on their current question. Use this context
when selecting skills and understand that the skills will also receive this
context as input.

{document_context}
```

This tells the orchestrator LLM that document content exists, so it can make better skill selection decisions.

### 9. `apps/appmgr/orchestrator/skill_executor.py` — Inject Document Context into Step Inputs

**Modify `execute_plan()` signature**: Add `document_context: Optional[str] = None`.

**Modify `_execute_step()`**: If `document_context` is provided, prepend it to the `step_input`:

```python
# In _execute_step, before calling _execute_workflow or _execute_agent:
if document_context:
    step_input = f"{step_input}\n\nUPLOADED DOCUMENT CONTEXT:\n{document_context}"
```

This ensures every workflow and agent in the execution plan receives the relevant document chunks as part of their input. The workflow executor treats it as part of `input_payload["user_input"]`, and the agent executor includes it in the task description.

**Why inject at step_input level** (not deeper):
- Workflows receive `input_payload = {"user_input": step_input, "message": step_input}` — the document context flows naturally into the workflow's agent prompts
- Standalone agents build their task description from `user_input` — the document context becomes part of "USER REQUEST"
- No modifications needed to `crewai_adapter.py` or `executor.py` (the DO NOT TOUCH files)

### 10. `apps/appmgr/orchestrator/pipeline.py` — Add RAG Retrieval Steps

**Add new step between steps 6 and 7** (after prompt enhancement, before skill manifest):

```
Step 6a — DOCUMENT RETRIEVAL:
  rag_manager = container.resolve('rag.session_manager')
  if rag_manager.has_documents(chat_session_id):
      document_context = rag_manager.retrieve(chat_session_id, enhanced_prompt, top_k=10)
  else:
      document_context = None
```

**Modify step 9** — Pass `document_context` to orchestrator:
```python
orchestrator_output = await self._orchestrator.plan(
    enhanced_prompt=enhanced_prompt,
    skill_manifest=skill_manifest,
    conversation_history=conversation_history,
    persona_prompt=persona_prompt,
    guardrail_rules=guardrail_rules_text,
    document_context=document_context,  # NEW
)
```

**Modify step 11** — Pass `document_context` to skill executor:
```python
execution_result = await self._skill_executor.execute_plan(
    db=db,
    execution_plan=plan,
    execution_strategy=strategy,
    user_input=enhanced_prompt,
    document_context=document_context,  # NEW
)
```

**Add to trace_data**: `trace_data["document_context_used"] = document_context is not None`

---

## Files NOT Modified

| File | Reason |
|------|--------|
| `apps/appmgr/container.py` | Resolves `rag.session_manager` from global DI container — no app-specific registration needed |
| `echolib/services.py` | `RAGService` class already exists (line 84) |
| `apps/workflow/crewai_adapter.py` | DO NOT TOUCH — document context flows via `input_payload["user_input"]` |
| `apps/workflow/runtime/executor.py` | DO NOT MODIFY — black box, receives augmented input naturally |
| `apps/workflow/designer/compiler.py` | DO NOT MODIFY |
| `llm_manager.py` | DO NOT MODIFY |
| `echolib/models/application_chat.py` | `ApplicationDocument` model already has all needed fields |

---

## Implementation Order

### Phase A: Service Layer (RAG as a reusable service)
1. Add Traditional RAG types to `echolib/types.py`
2. Modify `echolib/service/rag/trad_rag.py` — multi-format loaders + `add_documents()`
3. Create `echolib/service/rag/session_rag_manager.py` — `SessionRAGManager`
4. Register in `apps/rag/container.py` as `rag.session_manager`

### Phase B: Upload Processing
5. Add `get_session_documents()` and `update_document_status()` to `application_chat_repo.py`
6. Modify upload endpoint in `apps/appmgr/routes.py` to trigger processing
7. Extend `DocumentUploadResponse` in `apps/appmgr/schemas.py`

### Phase C: Pipeline Integration
8. Modify `orchestrator.py` — add `document_context` to system prompt
9. Modify `skill_executor.py` — inject document context into step inputs
10. Modify `pipeline.py` — add retrieval step, pass context through pipeline

---

## Verification Plan

1. **Unit test — RAG processing**: Upload a PDF, verify FAISS index is created at the expected path, verify chunk count > 0
2. **Unit test — Retrieval**: After indexing, call `retrieve()` with a query, verify relevant chunks are returned
3. **Integration test — Upload endpoint**: `POST /chat/upload` with a PDF, verify response has `processing_status=ready` and `chunk_count > 0`
4. **Integration test — Chat with document**: Upload a resume, send a chat message asking about the candidate, verify the response contains information from the resume (not generic)
5. **Isolation test**: Upload doc in session A, query in session B, verify session B returns no document context
6. **Multi-doc test**: Upload 2 files to same session, verify both are searchable
7. **Persistence test**: Upload and index, restart server, verify index loads from disk

---

## Dependencies

Libraries already in `requirements.txt` (based on existing RAG code):
- `langchain-community` (FAISS, document loaders)
- `langchain-openai` (AzureOpenAIEmbeddings)
- `langchain-text-splitters` (RecursiveCharacterTextSplitter)
- `faiss-cpu`
- `pymupdf` (PyMuPDFLoader)

May need to add:
- `docx2txt` — for DOCX file support (check if already present)
