# RAG File Upload Integration into App Chat — Implementation Plan

> **HOW TO RESUME**: In a new chat, say:
> *"Read `documents/file-upload_plan.md` then continue implementation."*

> **Status**: PENDING APPROVAL
> **Embeddings**: Azure OpenAI (`text-embedding-ada-002`)
> **Scope**: Session-scoped (each chat session gets its own FAISS index, like ChatGPT)
> **CRITICAL RULE**: `apps/workflow/crewai_adapter.py` is UNTOUCHED — zero modifications

---

## Context

The apps feature has a working orchestration pipeline (16-step flow in `pipeline.py`) that runs workflows and agents. Users can upload files via `POST /{id}/chat/upload`, but the upload **only saves the file to disk and creates a DB record with `status=pending`** — no processing happens. There is no chunking, no embedding, no vector storage, and no retrieval at query time.

**Goal**: When a user uploads a file in the app chat (e.g., a resume PDF), it should be immediately chunked, embedded, and stored in a session-scoped FAISS index. On every subsequent chat message, relevant chunks are retrieved via similarity search and injected as context into the orchestration pipeline so workflows/agents can use the document content.

**Decisions confirmed by user**:
- Embeddings: Azure OpenAI (`text-embedding-ada-002`)
- Scope: Session-scoped (each chat session gets its own FAISS index, like ChatGPT)

---

## Files to Create

### 1. `apps/appmgr/orchestrator/rag_service.py` (NEW)
**Session-scoped RAG service** adapted from user's `TraditionalRAGService`.

Key changes from the original:
- **No singleton** — instance per session, managed by a `SessionRAGManager` class
- **Credentials from env vars** — `os.getenv("AZURE_OPENAI_ENDPOINT")`, `os.getenv("AZURE_OPENAI_API_KEY")`
- **Multi-format support** — PDF (PyMuPDFLoader), DOCX (Docx2txtLoader), TXT/CSV/MD (TextLoader)
- **FAISS persistence** — save/load from `{UPLOADS_DIR}/rag_indexes/{session_id}/`
- **No LLM chain** — only chunking + embedding + retrieval (the orchestrator/agents handle the answering)
- **In-memory cache** — `SessionRAGManager` keeps a dict of `{session_id: FAISS retriever}` for active sessions, with LRU eviction

```python
class SessionRAGManager:
    """Manages session-scoped FAISS indexes."""

    def process_document(session_id, file_path, filename) -> dict:
        """Chunk + embed + add to session's FAISS index. Returns chunk_count."""

    def retrieve(session_id, query, top_k=10) -> str:
        """Similarity search against session's index. Returns formatted context string."""

    def has_documents(session_id) -> bool:
        """Check if session has any indexed documents."""

    def _get_or_load_index(session_id) -> FAISS:
        """Load from disk cache or create new."""

    def _save_index(session_id):
        """Persist FAISS index to disk."""
```

Key implementation details:
- Uses `RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)` — same as user's existing code
- Uses `AzureOpenAIEmbeddings(deployment="text-embedding-ada-002")` — same as user's existing code
- Uses `FAISS.from_documents()` for initial creation, `FAISS.merge_from()` for adding more docs to existing index
- File type detection via extension + mime_type
- Loaders: `PyMuPDFLoader` (PDF), `Docx2txtLoader` (DOCX), `TextLoader` (TXT/MD/CSV)
- Index path: `{UPLOADS_DIR}/rag_indexes/{session_id}/index.faiss` + `index.pkl`

---

## Files to Modify

### 2. `apps/appmgr/routes.py` — Upload Endpoint (lines 907-979)

**Current**: Saves file to disk, creates DB record with `status=pending`, returns immediately.

**Change**: After saving the file, **trigger RAG processing** (chunk + embed + index). Update the DB record with `status=ready` and `chunk_count`.

```
Current flow:
  save file → create DB record (pending) → return

New flow:
  save file → create DB record (processing)
  → SessionRAGManager.process_document(session_id, file_path, filename)
  → update DB record (ready, chunk_count=N)
  → return (with chunk_count in response)
```

- If `session_id` is not provided, auto-create a new session (file must be scoped to a session)
- If processing fails → set `status=failed`, return error detail
- Extend `DocumentUploadResponse` schema to include `chunk_count`

### 3. `apps/appmgr/orchestrator/pipeline.py` — Add RAG Retrieval Steps

**Add new step between steps 6 and 7** (after prompt enhancement, before skill manifest):

```
Step 6a — DOCUMENT RETRIEVAL:
  if SessionRAGManager.has_documents(chat_session_id):
      document_context = SessionRAGManager.retrieve(chat_session_id, enhanced_prompt, top_k=10)
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

### 4. `apps/appmgr/orchestrator/orchestrator.py` — Add Document Context to System Prompt

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

### 5. `apps/appmgr/orchestrator/skill_executor.py` — Inject Document Context into Step Inputs

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

### 8. `apps/appmgr/container.py` — Register RAG Manager

```python
from apps.appmgr.orchestrator.rag_service import SessionRAGManager

_session_rag_manager = SessionRAGManager()
container.register('app.rag_manager', lambda: _session_rag_manager)
```

---

## Files NOT Modified

| File | Reason |
|------|--------|
| `apps/workflow/crewai_adapter.py` | DO NOT TOUCH — document context flows via `input_payload["user_input"]` |
| `apps/workflow/runtime/executor.py` | DO NOT MODIFY — black box, receives augmented input naturally |
| `apps/workflow/designer/compiler.py` | DO NOT MODIFY |
| `llm_manager.py` | DO NOT MODIFY |
| `echolib/service/rag/*` | Existing RAG services — untouched, we build our own in `appmgr/orchestrator/` |
| `echolib/models/application_chat.py` | `ApplicationDocument` model already has all needed fields |

---

## Implementation Order

### Phase A: Core RAG Service
1. Create `apps/appmgr/orchestrator/rag_service.py` — `SessionRAGManager` with process/retrieve/persistence
2. Register in `apps/appmgr/container.py`

### Phase B: Upload Processing
3. Add `get_session_documents()` and `update_document_status()` to `application_chat_repo.py`
4. Modify upload endpoint in `routes.py` to trigger processing
5. Extend `DocumentUploadResponse` in `schemas.py`

### Phase C: Pipeline Integration
6. Modify `orchestrator.py` — add `document_context` to system prompt
7. Modify `skill_executor.py` — inject document context into step inputs
8. Modify `pipeline.py` — add retrieval step, pass context through pipeline

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
