# RAG File Upload Integration -- Progress Tracker

## Overall Status: COMPLETE (All 3 Phases Done)

---

## Phase A: Service Layer -- COMPLETE

### 1. echolib/types.py -- MODIFIED
- Added 5 Traditional RAG Pydantic models: `TraditionalRAGLoadRequest`, `TraditionalRAGLoadResponse`, `TraditionalRAGQueryRequest`, `TraditionalRAGQueryResponse`, `TraditionalRAGStats`
- Inserted before `CardResponse` class, after the existing Graph RAG and Tool System types

### 2. echolib/service/rag/trad_rag.py -- MODIFIED
- Added `_load_file()` method: multi-format document loader supporting .pdf, .docx, .txt, .md, .csv
- Updated `_load_documents()`: now delegates single-file loading to `_load_file()` instead of hardcoded PyMuPDFLoader
- Added `add_documents()` method: loads new documents, creates a new FAISS index, merges into existing index via `FAISS.merge_from()`, updates stats

### 3. echolib/service/rag/session_rag_manager.py -- CREATED (NEW FILE)
- `SessionRAGManager` class with full implementation:
  - `process_document()`: chunk + embed + add to session FAISS index, persist to disk
  - `retrieve()`: similarity search against session index, returns formatted context string
  - `has_documents()`: checks in-memory cache and disk for session index
  - `_get_or_load_index()`: LRU cache lookup with disk fallback
  - `_save_index()`: persist FAISS index to `{uploads_dir}/rag_indexes/{session_id}/`
  - `_load_file()`: multi-format loader (PDF, DOCX, TXT, MD, CSV)
  - `_evict_lru()`: evicts oldest session when cache exceeds max_cached_sessions
- Shared `AzureOpenAIEmbeddings` instance (one for all sessions)
- Shared `RecursiveCharacterTextSplitter` (chunk_size=800, chunk_overlap=200)
- OrderedDict-based LRU cache with configurable max size (default 50)

### 4. apps/rag/container.py -- MODIFIED
- Added import of `SessionRAGManager`
- Registered singleton instance as `rag.session_manager` in global DI container

---

## Phase B: Upload Processing -- COMPLETE

### 5. echolib/repositories/application_chat_repo.py -- MODIFIED
- Added `get_session_documents()`: queries `ApplicationDocument` by `chat_session_id`, optional `status` filter
- Added `update_document_status()`: updates `processing_status` and optionally `chunk_count` via SQLAlchemy `update()` statement

### 6. apps/appmgr/routes.py -- MODIFIED (upload_chat_document endpoint)
- Auto-creates chat session if `session_id` not provided
- Sets initial document status to `processing`
- Resolves `rag.session_manager` from DI container
- Calls `rag_manager.process_document()` to chunk, embed, and index the file
- Updates document record with final status (`ready` or `failed`) and `chunk_count`
- Returns `chunk_count` in response
- Wraps RAG processing in try/except -- sets status to `failed` on error, does not crash endpoint

### 7. apps/appmgr/schemas.py -- MODIFIED
- Added `chunk_count: int = 0` field to `DocumentUploadResponse`

---

## Phase C: Pipeline Integration -- COMPLETE

### 8. apps/appmgr/orchestrator/orchestrator.py -- MODIFIED
- Added `{document_context_section}` placeholder to `_ORCHESTRATOR_SYSTEM_PROMPT` template (between guardrail_section and skill_manifest_section)
- Added `document_context: Optional[str] = None` parameter to `plan()` method
- Added `document_context: Optional[str] = None` parameter to `_build_system_prompt()` method
- Build `document_context_section` text block when document_context is provided (includes "Uploaded Document Context" header)
- Passed `document_context_section` to the `.format()` call

### 9. apps/appmgr/orchestrator/skill_executor.py -- MODIFIED
- Added `document_context: Optional[str] = None` parameter to `execute_plan()` method
- Stores `document_context` on instance as `self._document_context` for use in `_execute_step()`
- In `_execute_step()`: if `document_context` is set, prepends it to `step_input` as `"\n\nUPLOADED DOCUMENT CONTEXT:\n{document_context}"`
- This ensures every workflow and agent receives document chunks in their input without modifying crewai_adapter.py or executor.py

### 10. apps/appmgr/orchestrator/pipeline.py -- MODIFIED
- Added step 6a (DOCUMENT RETRIEVAL) between prompt enhancement (step 6) and skill manifest (step 7):
  - Resolves `rag.session_manager` from DI container
  - Calls `rag_manager.has_documents(chat_session_id)` to check for indexed docs
  - Calls `rag_manager.retrieve(chat_session_id, enhanced_prompt, top_k=10)` for context
  - Gracefully handles KeyError (RAG module not loaded) and other exceptions
- Passed `document_context=document_context` to `self._orchestrator.plan()` (step 9)
- Passed `document_context=document_context` to `self._skill_executor.execute_plan()` (step 11)
- Added `trace_data["document_context_used"] = document_context is not None`

---

## Files NOT Modified (per plan)
- apps/appmgr/container.py
- echolib/services.py
- apps/workflow/crewai_adapter.py
- apps/workflow/runtime/executor.py
- apps/workflow/designer/compiler.py
- llm_manager.py
- echolib/models/application_chat.py

---

## Summary of All Files

| File | Action |
|------|--------|
| `echolib/types.py` | Modified -- added 5 Traditional RAG types |
| `echolib/service/rag/trad_rag.py` | Modified -- multi-format loaders + add_documents() |
| `echolib/service/rag/session_rag_manager.py` | **Created** -- SessionRAGManager |
| `apps/rag/container.py` | Modified -- registered rag.session_manager |
| `echolib/repositories/application_chat_repo.py` | Modified -- added 2 document query methods |
| `apps/appmgr/routes.py` | Modified -- upload endpoint triggers RAG processing |
| `apps/appmgr/schemas.py` | Modified -- added chunk_count to DocumentUploadResponse |
| `apps/appmgr/orchestrator/orchestrator.py` | Modified -- document_context in system prompt |
| `apps/appmgr/orchestrator/skill_executor.py` | Modified -- inject document context into step inputs |
| `apps/appmgr/orchestrator/pipeline.py` | Modified -- RAG retrieval step + context passthrough |
