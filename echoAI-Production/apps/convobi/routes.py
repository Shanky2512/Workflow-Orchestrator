
# nlp_db/routes.py
# API Routes (APIRouter so gateway can include_router and Convobi shows in main Swagger)
from fastapi import APIRouter, HTTPException, Query, UploadFile, File, Form, Body
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
from sqlalchemy import text, inspect, create_engine
import pandas as pd
import os
import time
import re
import json

# Define router first so convobi.main can "from .routes import router" without circular import
router = APIRouter(tags=["Convobi / NLP DB"])


class ConnectionIdBody(BaseModel):
    """Optional body for POST schema/schema doc; provide either connection_id (from POST /connectors/plugin/load) or file_id (from POST /convobi/files/upload)."""
    connection_id: Optional[str] = Field(None, alias="connectionId", description="ID of the DB connection from POST /connectors/plugin/load to recognize which DB is used")
    file_id: Optional[str] = Field(None, alias="fileId", description="ID of an uploaded file from POST /convobi/files/upload; use either connection_id or file_id")

    class Config:
        populate_by_name = True  # accept both connection_id and connectionId


import uuid
from .main import (
    app_state, CURRENT_DIALECT, CURRENT_CONNECTOR, SUPPORTED_DIALECTS,
    QueryRequest, QueryResponse, SchemaResponse, SummarizeRequest,
    QueryResult, ConnectorResult, ExecutionResult,
    ColumnMeaning, ColumnMeaningCache, SchemaAnalyzer, NLPQueryGenerator,
    LLMSummarizer, SqlValidator, PluginManager, SchemaDocumenter,
    SchemaDocCsvStore, SchemaDoc, FixMemoryStore, DataFixMemoryStore,
    ensure_schema_loaded, get_saved_connection_string, get_connection_string_for_file_id, activate_plugin,
    _apply_fix_map, _apply_data_fix_map, _propose_column_replacements,
    infer_visualization, to_vegalite_spec, compute_db_id, SAMPLE_ROWS_PER_TABLE, COLUMN_SAMPLE_SIZE,
    TOP_K_COMMON_VALUES, _quote_ident, _quote_full_ident, _first_unique,
    _column_semantics_from_samples_generic, UPLOADS_DIR, LOG,
    _is_summary_only_question,
    get_schema_summary_purpose, _infer_table_purpose,
)


def _build_visualization_html(vega_lite_spec: Dict[str, Any], width: int = 600, height: int = 400) -> str:
    """Build self-contained HTML that renders the Vega-Lite spec; safe for iframe srcdoc. Chart uses 100% width/height to fill container."""
    spec_json = json.dumps(vega_lite_spec)
    # Escape for use inside script tag: avoid </script> in JSON
    spec_json_escaped = spec_json.replace("</", "<\\/")
    return (
        "<!DOCTYPE html><html><head><meta charset='UTF-8'>"
        "<style>html, body { width: 100%; height: 100%; margin: 0; padding: 0; } "
        "#chart-div { width: 100%; height: 100%; min-height: 200px; }</style>"
        "<script src='https://cdn.jsdelivr.net/npm/vega@5'></script>"
        "<script src='https://cdn.jsdelivr.net/npm/vega-lite@5'></script>"
        "<script src='https://cdn.jsdelivr.net/npm/vega-embed@6'></script>"
        "</head><body>"
        "<div id='chart-div'></div>"
        "<script>"
        "(function(){"
        "var spec = " + spec_json_escaped + ";"
        "if (spec && document.getElementById('chart-div')) {"
        "vegaEmbed('#chart-div', spec, { actions: false }).catch(function(e){ document.getElementById('chart-div').innerHTML = '<p style=\"color:red\">Chart error: ' + e.message + '</p>'; });"
        "}"
        "})();"
        "</script></body></html>"
    )


@router.get("/")
async def root():
    active_plugin, active_cs = (app_state["plugin_manager"].active() if app_state.get("plugin_manager") else (None, None))
    qg = app_state.get("query_generator")
    llm_display = (getattr(qg, "_resolved_model_id", None) if qg else None) or "LLM"
    return {
        "status": "healthy",
        "service": "NLP Database Interface (LLM API, SQL-only, Domain-neutral)",
        "version": "1.7.0",
        "llm_model": llm_display,
        "tables_loaded": len(app_state["schema_metadata"]) if app_state["schema_metadata"] else 0,
        "dialect": CURRENT_DIALECT,
        "connector": CURRENT_CONNECTOR,
        "plugin": active_plugin.name() if active_plugin else None,
        "connection_string": active_cs,
        "connection_id": app_state.get("active_connection_id"),
        "file_id": app_state.get("active_file_id"),
    }


def _get_schema_response() -> SchemaResponse:
    """Build SchemaResponse from current app_state (used by POST /schema)."""
    tables_dict: Dict[str, Any] = {}
    for table_name, table_meta in app_state["schema_metadata"].items():
        tables_dict[table_name] = {
            "name": table_meta.name,
            "row_count": table_meta.row_count,
            "columns": [
                {
                    "name": col.name,
                    "type": col.type,
                    "nullable": col.nullable,
                    "primary_key": col.primary_key,
                    "foreign_key": col.foreign_key,
                    "meaning": (app_state["meaning_cache"].get(table_name, col.name).meaning if app_state["meaning_cache"].get(table_name, col.name) else None),
                    "description": (app_state["meaning_cache"].get(table_name, col.name).description if app_state["meaning_cache"].get(table_name, col.name) else None),
                    "confidence": (app_state["meaning_cache"].get(table_name, col.name).confidence if app_state["meaning_cache"].get(table_name, col.name) else None),
                    "source": (app_state["meaning_cache"].get(table_name, col.name).source if app_state["meaning_cache"].get(table_name, col.name) else None),
                    "examples": (app_state["meaning_cache"].get(table_name, col.name).examples if app_state["meaning_cache"].get(table_name, col.name) else [])
                }
                for col in table_meta.columns
            ],
            "relationships": table_meta.relationships
        }
    doc: SchemaDoc = app_state.get("schema_doc")
    schema_metadata = app_state.get("schema_metadata") or {}
    purpose_paragraph = get_schema_summary_purpose(schema_metadata, doc)
    table_names = sorted(tables_dict.keys())
    total_tables = len(table_names)
    lines = ["# Schema Summary", ""]
    lines.append(f"**Total Tables:** {total_tables}")
    lines.append("")
    lines.append("## Schema Summary")
    lines.append("")
    lines.append(purpose_paragraph)
    lines.append("")
    lines.append("## Tables")
    lines.append("")
    if table_names:
        lines.append("| ID | Table | Rows | Columns |")
        lines.append("|----|-------|------|---------|")
        for idx, tn in enumerate(table_names):
            t = tables_dict[tn]
            rows = t.get("row_count") if t.get("row_count") is not None else "—"
            cols = len(t.get("columns") or [])
            lines.append(f"| {idx} | {tn} | {rows} | {cols} |")
    else:
        lines.append("No tables loaded.")
    schema_summary = "\n".join(lines)
    return SchemaResponse(tables=tables_dict, total_tables=total_tables, summary=schema_summary)


def _ensure_connection_and_schema(connection_id: Optional[str] = None, file_id: Optional[str] = None) -> None:
    """If connection_id or file_id is given, resolve and activate that DB then load schema. Otherwise just ensure schema loaded for current active DB. Use either connection_id or file_id, not both."""
    if connection_id and file_id:
        raise HTTPException(status_code=400, detail="Provide either connection_id or file_id, not both.")
    if file_id:
        cs = get_connection_string_for_file_id(file_id)
        if not cs:
            raise HTTPException(status_code=404, detail=f"Uploaded file '{file_id}' not found. Use POST /convobi/files/upload to upload a CSV or Excel file first.")
        active_id = app_state.get("active_file_id")
        if active_id != file_id or not app_state.get("schema_analyzer"):
            activate_plugin(cs, connector_hint="sql", connection_id=None)
            app_state["active_file_id"] = file_id
    elif connection_id:
        active_id = app_state.get("active_connection_id")
        if active_id != connection_id or not app_state.get("schema_analyzer"):
            cs = get_saved_connection_string(connection_id)
            if not cs:
                raise HTTPException(status_code=404, detail=f"Saved connection '{connection_id}' not found. Use POST /connectors/plugin/list to see saved_connections.")
            activate_plugin(cs, connector_hint="sql", connection_id=connection_id)
        app_state["active_file_id"] = None
    else:
        app_state["active_file_id"] = None
    ensure_schema_loaded()
    if app_state.get("schema_metadata") is None and app_state.get("schema_analyzer"):
        # ensure_schema_loaded should have populated it; treat as not loaded
        raise HTTPException(status_code=500, detail="Schema not loaded")


@router.post(
    "/schema",
    response_model=SchemaResponse,
    summary="Get schema structure and short summary",
    description="Returns the current database **structure**: tables, columns (with types, meanings, examples), relationships, plus a **short markdown summary** (purpose of the DB + table names with row counts). Use this for dropdowns, column pickers, and a brief overview.",
)
async def get_schema(request: Optional[ConnectionIdBody] = Body(None, description="Optional body with connection_id or file_id")):
    cid = request.connection_id if request else None
    fid = request.file_id if request else None
    _ensure_connection_and_schema(connection_id=cid, file_id=fid)
    if app_state["schema_metadata"] is None:
        raise HTTPException(status_code=500, detail="Schema not loaded")
    return _get_schema_response()
# MAIN: /query
@router.post("/query", response_model=QueryResponse)
async def execute_query(request: QueryRequest):
    """
    Pipeline:
    - Gather learned fixes (column & data) and merge with request maps
    - Generate SQL via LLM (apply_fix/apply_data_fix true => prompt includes corrections)
    - Validate; if invalid or warnings and auto_proceed=False => handshake with proposed fixes
    - Execute; optionally remember_fix / remember_data_fix when execution succeeds
    """
    try:
        _ensure_connection_and_schema(connection_id=request.connector_id, file_id=request.file_id)
        if not request.question or not request.question.strip():
            raise HTTPException(status_code=422, detail="Question must not be empty.")
        if request.max_results is not None and request.max_results <= 0:
            raise HTTPException(status_code=422, detail="max_results must be positive.")
        if request.language != "sql":
            raise HTTPException(status_code=422, detail="Only 'sql' is supported.")
        if not app_state.get("sql_connector"):
            raise HTTPException(
                status_code=400,
                detail=(
                    "No database connected. Connect first: POST /convobi/plugins/load with body "
                    '{"connection_string": "sqlite:///path/to/your.db", "connector": "sql"}.'
                ),
            )
        qg = app_state.get("query_generator")
        schema_meta = app_state.get("schema_metadata") or {}
        has_schema = (qg and getattr(qg, "schema_context", None)) or schema_meta
        if not has_schema:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Schema context not set (database has no tables or was not loaded). "
                    "Connect to a database that has tables via POST /convobi/plugins/load, "
                    'e.g. {"connection_string": "sqlite:///path/to/your.db", "connector": "sql"}.'
                ),
            )
        # CSV/Excel file chat: for summary/brief/overview questions, return only narrative summary (no SQL, no charts)
        if request.file_id and _is_summary_only_question(request.question):
            table_names = sorted(schema_meta.keys())
            if not table_names:
                raise HTTPException(status_code=400, detail="No tables in the uploaded file.")
            summary_text: Optional[str] = None
            try:
                # Sample data from the first table (or all tables if single-sheet CSV)
                for tname in table_names:
                    safe_name = tname.replace('"', '""')
                    exec_res = app_state["sql_connector"].execute(
                        f'SELECT * FROM "{safe_name}" LIMIT 100'
                    )
                    if exec_res.success and exec_res.columns and (exec_res.data or []):
                        summarizer = LLMSummarizer(app_state["query_generator"]._ollama_generate)
                        summary_text = summarizer.summarize_file_data(
                            request.question, tname, exec_res.columns, exec_res.data or []
                        )
                        break
                if not summary_text:
                    summary_text = (
                        "The file has no data rows to summarize, or the tables are empty. "
                        "Upload a CSV or Excel file with data and ask again."
                    )
            except Exception as sum_err:
                LOG.warning("File summary generation failed: %s", sum_err)
                summary_text = f"Could not generate summary: {sum_err}"
            return QueryResponse(
                success=True,
                sql=None,
                explanation="Summary of file data (no SQL or chart generated).",
                data=None,
                row_count=None,
                columns=None,
                visualization_hints=None,
                execution_time_ms=None,
                confidence=1.0,
                optimizations=None,
                warnings=None,
                error=None,
                summary=summary_text,
                visualization_spec=None,
                visualization_html=None,
            )
        schema_fp = str(len(schema_meta))
        cache_key = f"{CURRENT_CONNECTOR}:{CURRENT_DIALECT}:{schema_fp}:sql:{request.question}:{request.max_results}"
        # Learned column fixes (no fix_map in request – payload simplified)
        learned_map = app_state["fix_memory"].find_for(request.question) if app_state.get("fix_memory") else {}
        merged_fix_map: Dict[str, str] = dict(learned_map) if learned_map else {}
        # Learned DATA fixes
        learned_data_map = app_state["data_fix_memory"].find_for(request.question) if app_state.get("data_fix_memory") else {}
        merged_data_fix_map: Dict[str, str] = dict(learned_data_map) if learned_data_map else {}
        apply_regen_cols = bool(merged_fix_map)
        apply_regen_data = bool(merged_data_fix_map)
        fix_map_for_gen = merged_fix_map if apply_regen_cols else None
        data_map_for_gen = merged_data_fix_map if apply_regen_data else None
        app_state["data_fix_map_for_prompt"] = data_map_for_gen or {}
        # Generate SQL via LLM
        try:
            query_result: QueryResult = app_state["query_generator"].generate_query(
                request.question, language="sql", max_results=request.max_results, fix_map=fix_map_for_gen
            )
        except Exception as gen_err:
            LOG.exception("Query generation failed: %s", gen_err)
            err_msg = str(gen_err)
            if "schema context not set" in err_msg.lower():
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Schema context not set. Connect a database with tables first: "
                        "POST /convobi/plugins/load with body "
                        '{"connection_string": "sqlite:///path/to/your.db", "connector": "sql"}.'
                    ),
                )
            if "timed out" in err_msg.lower() or "timeout" in err_msg.lower():
                detail = (
                    "generator_error: LLM request timed out. "
                    "Check that the LLM service (e.g. Ollama at LLM base URL) is running and reachable. "
                    "You can set LLM_REQUEST_TIMEOUT (seconds) to fail faster."
                )
            else:
                detail = f"generator_error: {gen_err}"
            raise HTTPException(status_code=502, detail=detail)
        finally:
            app_state["data_fix_map_for_prompt"] = None
        query_text = (query_result.sql or "").strip()
        if not query_text:
            raise HTTPException(status_code=502, detail="generator_error: empty query")
        # Post-process DATA fixes regardless of regen (safe)
        if merged_data_fix_map:
            query_text = _apply_data_fix_map(query_text, merged_data_fix_map)
        LOG.info("Generated SQL: %s", query_text)
        validation = SqlValidator.validate(query_text)
        if not validation["valid"]:
            meta = app_state.get("schema_metadata") or {}
            cache = app_state.get("meaning_cache")
            proposed = _propose_column_replacements(query_text, meta, cache) if (meta and cache) else []
            return QueryResponse(
                success=False,
                sql=query_text,
                explanation=query_result.explanation,
                confidence=query_result.confidence,
                warnings=(query_result.warnings or []) + (validation.get("issues") or []),
                optimizations=query_result.optimizations,
                error="validation_error",
            )
        warnings = query_result.warnings or []
        if False:  # auto_proceed always True: execute even when there are warnings
            meta = app_state.get("schema_metadata") or {}
            cache = app_state.get("meaning_cache")
            proposed = _propose_column_replacements(query_text, meta, cache) if (meta and cache) else []
            return QueryResponse(
                success=False,
                sql=query_text,
                explanation=query_result.explanation,
                confidence=query_result.confidence,
                warnings=warnings,
                optimizations=query_result.optimizations,
            )
        # Execute
        try:
            exec_res: ConnectorResult = app_state["sql_connector"].execute(query_text)
        except Exception as exec_err:
            LOG.exception("Query execution failed: %s", exec_err)
            raise HTTPException(status_code=500, detail=f"executor_error: {exec_err}")
        # Learn/persist fixes not used (payload simplified – no fix_map/data_fix_map in request)
        # Cache successful results
        if exec_res.success and request.use_cache:
            app_state["result_cache"].put(cache_key, ExecutionResult(**exec_res.dict()))
        app_state["conversation_history"].append({
            "timestamp": datetime.now().isoformat(),
            "question": request.question,
            "sql": query_text,
            "success": exec_res.success,
            "row_count": exec_res.row_count
        })
        if len(app_state["conversation_history"]) > 50:
            app_state["conversation_history"] = app_state["conversation_history"][-50:]
        # Include summary, visualization_spec, and visualization_html in response so frontend gets all in one call
        summary_text: Optional[str] = None
        viz_spec: Optional[Dict[str, Any]] = None
        viz_html: Optional[str] = None
        if exec_res.success and exec_res.data is not None and exec_res.columns:
            cols = exec_res.columns
            dat = exec_res.data
            hints = exec_res.visualization_hints or {}
            # Summary (LLM): cap data to avoid huge payload
            try:
                summarizer = LLMSummarizer(app_state["query_generator"]._ollama_generate)
                summary_text = summarizer.summarize(
                    request.question, cols, dat[:50], hints or {}
                )
            except Exception as sum_err:
                LOG.warning("Summary generation failed: %s", sum_err)
            # Vega-Lite spec and HTML for visualization
            try:
                if not hints or not hints.get("suggested"):
                    hints = infer_visualization(cols, dat)
                viz_spec = to_vegalite_spec(cols, dat, hints)
                if viz_spec:
                    viz_html = _build_visualization_html(viz_spec)
            except Exception as viz_err:
                LOG.warning("Visualization spec generation failed: %s", viz_err)
        return QueryResponse(
            success=exec_res.success, sql=query_text, explanation=query_result.explanation,
            data=exec_res.data, row_count=exec_res.row_count, columns=exec_res.columns,
            visualization_hints=exec_res.visualization_hints, execution_time_ms=exec_res.execution_time_ms,
            confidence=query_result.confidence, optimizations=query_result.optimizations,
            warnings=query_result.warnings, error=exec_res.error,
            summary=summary_text,
            visualization_spec=viz_spec,
            visualization_html=viz_html,
        )
    except HTTPException:
        raise
    except Exception as e:
        LOG.exception("Query pipeline failed")
        return QueryResponse(success=False, error=str(e))
# Summarize
@router.post("/summarize")
async def summarize(req: SummarizeRequest):
    s = LLMSummarizer(app_state["query_generator"]._ollama_generate)
    try:
        summary = s.summarize(req.question, req.columns, req.data, req.visualization_hints or {})
        return {"summary": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"summarizer_error: {e}")
# Column meaning endpoints
@router.get("/columns/meaning")
async def get_column_meaning(
    table: str = Query(...),
    column: str = Query(...),
    connection_id: Optional[str] = Query(None, description="ID of the DB connection from POST /connectors/plugin/load"),
    file_id: Optional[str] = Query(None, description="ID of an uploaded file from POST /convobi/files/upload")
):
    _ensure_connection_and_schema(connection_id=connection_id, file_id=file_id)
    cm = app_state["meaning_cache"].get(table, column)
    if not cm: return {"found": False}
    return {"found": True, "meaning": cm.meaning, "description": cm.description, "confidence": cm.confidence, "source": cm.source, "examples": cm.examples}
# Dialect / Schema / Cache
@router.post("/config/dialect")
async def set_dialect(
    dialect: str,
    connection_id: Optional[str] = Query(None, description="ID of the DB connection from POST /connectors/plugin/load"),
    file_id: Optional[str] = Query(None, description="ID of an uploaded file from POST /convobi/files/upload")
):
    _ensure_connection_and_schema(connection_id=connection_id, file_id=file_id)
    d = (dialect or "").strip().lower()
    global CURRENT_DIALECT
    if d not in SUPPORTED_DIALECTS:
        raise HTTPException(status_code=400, detail=f"Unsupported dialect '{d}'")
    CURRENT_DIALECT = d
    pm = app_state.get("plugin_manager")
    active, _ = pm.active() if pm else (None, None)
    if active:
        app_state["llm_dialect_rules"] = active.llm_rules_for_dialect(CURRENT_DIALECT)
    app_state["result_cache"].clear()
    return {"ok": True, "dialect": CURRENT_DIALECT}
@router.post("/schema/refresh")
async def refresh_schema(
    connection_id: Optional[str] = Query(None, description="ID of the DB connection from POST /connectors/plugin/load"),
    file_id: Optional[str] = Query(None, description="ID of an uploaded file from POST /convobi/files/upload")
):
    _ensure_connection_and_schema(connection_id=connection_id, file_id=file_id)
    if not app_state["schema_analyzer"]:
        raise HTTPException(status_code=500, detail="schema_analyzer_not_initialized")
    schema_metadata = app_state["schema_analyzer"].analyze_schema()
    app_state["schema_metadata"] = schema_metadata
    schema_context = app_state["schema_analyzer"].generate_llm_context(schema_metadata, verbose=True)
    app_state["query_generator"].set_schema_context(schema_context)
    app_state["result_cache"].clear()
    try:
        documenter = SchemaDocumenter(app_state["meaning_cache"], app_state["query_generator"]._ollama_generate, CURRENT_DIALECT)
        doc = documenter.build(schema_metadata)
        fingerprint = SchemaDocumenter._schema_fingerprint(schema_metadata)
        app_state["schema_doc"] = doc
        store: SchemaDocCsvStore = app_state.get("schema_doc_store")
        db_id = app_state.get("db_id")
        if store and db_id:
            store.write(db_id, fingerprint, doc)
    except Exception as e:
        LOG.warning(f"SchemaDoc refresh failed: {e}")
    return {"ok": True, "tables_loaded": len(schema_metadata)}
@router.post("/cache/clear")
async def clear_caches(
    connection_id: Optional[str] = Query(None, description="ID of the DB connection from POST /connectors/plugin/load"),
    file_id: Optional[str] = Query(None, description="ID of an uploaded file from POST /convobi/files/upload")
):
    _ensure_connection_and_schema(connection_id=connection_id, file_id=file_id)
    app_state["result_cache"].clear()
    app_state["meaning_cache"].clear()
    store: SchemaDocCsvStore = app_state.get("schema_doc_store")
    if store:
        store.clear()
    app_state["schema_doc"] = None
    return {"ok": True}
@router.post("/visualize")
async def visualize(
    payload: Dict[str, Any],
    connection_id: Optional[str] = Query(None, description="ID of the DB connection from POST /connectors/plugin/load"),
    file_id: Optional[str] = Query(None, description="ID of an uploaded file from POST /convobi/files/upload")
):
    _ensure_connection_and_schema(connection_id=connection_id, file_id=file_id)
    cols = payload.get("columns") or []
    dat = payload.get("data") or []
    hints = payload.get("visualization_hints") or infer_visualization(cols, dat)
    spec = to_vegalite_spec(cols, dat, hints)
    return {"spec": spec, "hints": hints}


@router.post("/visualize/html")
async def visualize_html(
    payload: Dict[str, Any],
    connection_id: Optional[str] = Query(None, description="ID of the DB connection from POST /connectors/plugin/load"),
    file_id: Optional[str] = Query(None, description="ID of an uploaded file from POST /convobi/files/upload")
):
    """
    Same input as POST /visualize (columns, data, visualization_hints).
    Returns JSON { "html": "<!DOCTYPE html>..." } for use as iframe srcdoc.
    """
    _ensure_connection_and_schema(connection_id=connection_id, file_id=file_id)
    cols = payload.get("columns") or []
    dat = payload.get("data") or []
    hints = payload.get("visualization_hints") or infer_visualization(cols, dat)
    spec = to_vegalite_spec(cols, dat, hints)
    html = _build_visualization_html(spec)
    return {"html": html}


@router.get("/tables/describe")
async def describe_table(
    table: str = Query(...),
    connection_id: Optional[str] = Query(None, description="ID of the DB connection from POST /connectors/plugin/load"),
    file_id: Optional[str] = Query(None, description="ID of an uploaded file from POST /convobi/files/upload")
):
    _ensure_connection_and_schema(connection_id=connection_id, file_id=file_id)
    meta = app_state.get("schema_metadata") or {}
    cache: ColumnMeaningCache = app_state.get("meaning_cache")
    if table not in meta:
        raise HTTPException(status_code=404, detail=f"table_not_found: {table}")
    tmeta = meta[table]
    try:
        missing_cols = [c for c in tmeta.columns if not (cache and cache.get(table, c.name))]
        if missing_cols:
            analyzer: SchemaAnalyzer = app_state.get("schema_analyzer")
            with analyzer.engine.connect() as conn:
                q_cols = ", ".join(_quote_ident(c.name) for c in tmeta.columns)
                pk_cols = [c.name for c in tmeta.columns if c.primary_key]
                order_clause = ""
                if pk_cols:
                    q_pk = _quote_ident(pk_cols[0])
                    order_clause = f" ORDER BY {q_pk} ASC"
                q_table = _quote_full_ident(table)
                if CURRENT_DIALECT == "mssql":
                    select_sql = f"SELECT TOP {SAMPLE_ROWS_PER_TABLE} {q_cols} FROM {q_table}{order_clause}"
                else:
                    select_sql = f"SELECT {q_cols} FROM {q_table}{order_clause} LIMIT {SAMPLE_ROWS_PER_TABLE}"
                sample_rows = conn.execute(text(select_sql)).mappings().all()
                col_samples: Dict[str, List[Any]] = {c.name: [] for c in tmeta.columns}
                for r in sample_rows:
                    for c in tmeta.columns:
                        col_samples[c.name].append(r.get(c.name))
                for c in tmeta.columns:
                    samples_raw = [v for v in col_samples.get(c.name, []) if v is not None]
                    samples20 = _first_unique(samples_raw, COLUMN_SAMPLE_SIZE)
                    inferred = _column_semantics_from_samples_generic(table, c.name, samples20, c.foreign_key) if samples20 else None
                    if inferred and len(samples20) < COLUMN_SAMPLE_SIZE:
                        inferred.confidence = min(inferred.confidence, 0.7)
                    if inferred:
                        cache.put(inferred)
        out = []
        for c in tmeta.columns:
            cm = cache.get(table, c.name)
            out.append({
                "column": c.name,
                "type": c.type,
                "nullable": c.nullable,
                "primary_key": c.primary_key,
                "foreign_key": c.foreign_key,
                "meaning": cm.meaning if cm else None,
                "description": cm.description if cm else None,
                "confidence": cm.confidence if cm else None,
                "examples": cm.examples[:TOP_K_COMMON_VALUES] if cm else []
            })
        return {"table": table, "row_count": tmeta.row_count, "columns": out}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"describe_error: {e}")
# Tabular file uploads -> SQLite
def _sanitize_table_name(filename_or_sheet: str) -> str:
    base = os.path.splitext(os.path.basename(filename_or_sheet))[0]
    base = re.sub(r"[^A-Za-z0-9_]", "_", base).lower()
    return base or "uploaded_table"
def _uploads_db_path_for_file(filename: str) -> str:
    base = os.path.splitext(os.path.basename(filename))[0]
    safe = re.sub(r"[^A-Za-z0-9_]", "_", base).lower() or "uploaded_db"
    return os.path.join(UPLOADS_DIR, f"{safe}.db")
def _ensure_engine_for_path(db_path: str):
    url = f"sqlite:///{db_path}"
    return create_engine(url, pool_pre_ping=True)
@router.post("/files/upload")
async def upload_tabular_file(
    file: UploadFile = File(...),
    activate_source: bool = Form(True),
    sheet: Optional[str] = Form(None),
    table_name: Optional[str] = Form(None)
):
    if not file or not file.filename:
        raise HTTPException(status_code=422, detail="No file provided")
    ext = os.path.splitext(file.filename)[1].lower()
    tmp_path = os.path.join(UPLOADS_DIR, f"tmp_{int(time.time())}_{file.filename}")
    db_path = _uploads_db_path_for_file(file.filename)
    try:
        with open(tmp_path, "wb") as out:
            content = await file.read()
            out.write(content)
        tables_loaded = []
        eng = _ensure_engine_for_path(db_path)
        if ext == ".csv":
            df = pd.read_csv(tmp_path)
            tname = table_name or _sanitize_table_name(file.filename)
            with eng.begin() as conn:
                df.to_sql(tname, con=conn, if_exists="replace", index=False)
            tables_loaded.append({"name": tname, "rows": int(df.shape[0]), "columns": list(df.columns)})
        elif ext == ".xlsx":
            xls = pd.ExcelFile(tmp_path, engine="openpyxl")
            sheet_names = [sheet] if sheet else xls.sheet_names
            for sh in sheet_names:
                df = pd.read_excel(tmp_path, sheet_name=sh, engine="openpyxl")
                tname = (table_name or _sanitize_table_name(sh)) if len(sheet_names) == 1 else _sanitize_table_name(sh)
                with eng.begin() as conn:
                    df.to_sql(tname, con=conn, if_exists="replace", index=False)
                tables_loaded.append({"name": tname, "rows": int(df.shape[0]), "columns": list(df.columns)})
        else:
            raise HTTPException(status_code=415, detail=f"Unsupported file type: {ext}. Use CSV or XLSX")
        activated = False
        if activate_source:
            pm: PluginManager = app_state.get("plugin_manager")
            plugin = pm.choose(f"sqlite:///{db_path}", connector_hint="sql")
            pm.activate(plugin, f"sqlite:///{db_path}")
            app_state["schema_analyzer"] = plugin.create_analyzer(f"sqlite:///{db_path}", app_state["meaning_cache"])
            app_state["sql_connector"] = plugin.create_connector(f"sqlite:///{db_path}")
            schema_metadata = app_state["schema_analyzer"].analyze_schema()
            app_state["schema_metadata"] = schema_metadata
            schema_context = app_state["schema_analyzer"].generate_llm_context(schema_metadata, verbose=True)
            app_state["query_generator"].set_schema_context(schema_context)
            app_state["llm_dialect_rules"] = plugin.llm_rules_for_dialect("sqlite")
            app_state["result_cache"].clear()
            activated = True
            try:
                documenter = SchemaDocumenter(app_state["meaning_cache"], app_state["query_generator"]._ollama_generate, "sqlite")
                doc = documenter.build(schema_metadata)
                fingerprint = SchemaDocumenter._schema_fingerprint(schema_metadata)
                db_id = compute_db_id(f"sqlite:///{db_path}", "sqlite")
                app_state["db_id"] = db_id
                app_state["schema_doc"] = doc
                store: SchemaDocCsvStore = app_state.get("schema_doc_store")
                if store:
                    store.write(db_id, fingerprint, doc)
            except Exception as e:
                LOG.warning(f"SchemaDoc generation for uploads DB failed: {e}")
        file_id = str(uuid.uuid4())
        if "file_id_registry" not in app_state:
            app_state["file_id_registry"] = {}
        app_state["file_id_registry"][file_id] = db_path
        return {
            "ok": True,
            "file_id": file_id,
            "filename": file.filename,
            "database": db_path,
            "dialect": "sqlite",
            "tables": tables_loaded,
            "activated": activated
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"upload_error: {e}")
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
@router.get("/files/list")
async def list_uploaded_tables(
    db: Optional[str] = None,
    connection_id: Optional[str] = Query(None, description="ID of the DB connection from POST /connectors/plugin/load"),
    file_id: Optional[str] = Query(None, description="ID of an uploaded file from POST /convobi/files/upload")
):
    try:
        _ensure_connection_and_schema(connection_id=connection_id, file_id=file_id)
        if not db:
            db_files = [f for f in os.listdir(UPLOADS_DIR) if f.endswith(".db")]
            return {"databases": [os.path.join(UPLOADS_DIR, f) for f in db_files]}
        db_path = db if db.endswith(".db") else os.path.join(UPLOADS_DIR, f"{db}.db")
        eng = _ensure_engine_for_path(db_path)
        insp = inspect(eng)
        tables = insp.get_table_names()
        sizes = {}
        with eng.connect() as conn:
            for t in tables:
                try:
                    rc = conn.execute(text(f'SELECT COUNT(*) AS c FROM "{t}"')).scalar()
                    sizes[t] = int(rc or 0)
                except Exception:
                    sizes[t] = None
        return {"database": db_path, "tables": [{"name": t, "rows": sizes.get(t)} for t in tables]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"list_error: {e}")
# Plugin list/load are served by connector at GET/POST /convobi/plugins/list and /convobi/plugins/load.
# Convobi uses activate_plugin() from main when connector calls it after testing connection.


def _schema_doc_table_to_markdown(table_name: str, table_data: Dict[str, Any]) -> str:
    """Render a single table's schema doc as markdown."""
    overview = (table_data.get("overview") or "").strip()
    row_count = table_data.get("row_count")
    relationships = table_data.get("relationships") or []
    columns = table_data.get("columns") or []
    lines = [f"# Table: {table_name}", ""]
    if overview:
        lines.append("## Overview")
        lines.append(overview)
        lines.append("")
    lines.append(f"- **Rows:** {row_count if row_count is not None else 'N/A'}")
    if relationships:
        lines.append(f"- **Relationships:** {', '.join(relationships)}")
    lines.append("")
    lines.append("## Columns")
    lines.append("")
    lines.append("| Column | Type | Nullable | PK | FK | Meaning | Description | Examples |")
    lines.append("|--------|------|----------|-----|-----|---------|--------------|----------|")
    for c in columns:
        name = c.get("name", "")
        typ = c.get("type", "")
        nullable = "Yes" if c.get("nullable") else "No"
        pk = "Yes" if c.get("primary_key") else ""
        fk = (c.get("foreign_key") or "")[:20]
        meaning = (c.get("meaning") or "")[:30]
        desc = ((c.get("description") or "")[:50]).replace("|", "\\|")
        ex = ", ".join((c.get("examples") or [])[:5]).replace("|", "\\|")[:40]
        lines.append(f"| {name} | {typ} | {nullable} | {pk} | {fk} | {meaning} | {desc} | {ex} |")
    lines.append("")
    return "\n".join(lines)


def _schema_doc_to_markdown(doc: Any, table_name: Optional[str] = None) -> str:
    """
    Convert SchemaDoc to markdown. If table_name is given, return only that table's section.
    Otherwise return full schema doc (overview + all tables).
    """
    doc_dict = doc.dict() if hasattr(doc, "dict") else doc
    tables = doc_dict.get("tables") or {}
    if table_name:
        if table_name not in tables:
            return f"*Table '{table_name}' not found in schema documentation.*"
        return _schema_doc_table_to_markdown(table_name, tables[table_name])
    lines = ["# Schema Documentation", ""]
    lines.append(f"- **Generated:** {doc_dict.get('generated_at', 'N/A')}")
    lines.append(f"- **Dialect:** {doc_dict.get('dialect', 'N/A')}")
    lines.append(f"- **Total tables:** {doc_dict.get('total_tables', 0)}")
    narrative = (doc_dict.get("narrative") or "").strip()
    if narrative:
        lines.append("")
        lines.append("## Database Overview")
        lines.append("")
        lines.append(narrative)
        lines.append("")
    lines.append("---")
    lines.append("")
    for tn in sorted(tables.keys()):
        lines.append(_schema_doc_table_to_markdown(tn, tables[tn]))
        lines.append("")
    return "\n".join(lines).strip()


# Schema Documentation Endpoints
@router.post(
    "/schema/doc",
    summary="Get schema documentation (full)",
    description="Returns **full schema documentation**: narrative overview, per-table overviews and column tables, and markdown (full and per-table for dropdown). Use this for the 'Schema documentation' view and table-by-table docs. Differs from POST /schema which returns structure + short summary only.",
)
async def get_schema_doc(
    request: Optional[ConnectionIdBody] = Body(None, description="Optional body with connection_id or file_id"),
    table: Optional[str] = Query(None, description="If set, return only this table's markdown (for dropdown selection)"),
):
    _ensure_connection_and_schema(connection_id=request.connection_id if request else None, file_id=request.file_id if request else None)
    doc: SchemaDoc = app_state.get("schema_doc")
    if not doc:
        meta = app_state.get("schema_metadata")
        if not meta:
            raise HTTPException(status_code=500, detail="Schema not loaded")
        fingerprint = SchemaDocumenter._schema_fingerprint(meta)
        db_id = app_state.get("db_id")
        store: SchemaDocCsvStore = app_state.get("schema_doc_store")
        loaded = store.read(db_id, fingerprint) if (store and db_id) else None
        if not loaded:
            raise HTTPException(status_code=500, detail="SchemaDoc not available")
        app_state["schema_doc"] = loaded
        doc = loaded
    doc_dict = doc.dict()
    tables_list = sorted(doc.tables.keys())
    markdown_full = _schema_doc_to_markdown(doc, table_name=None)
    markdown_by_table: Dict[str, str] = {tn: _schema_doc_table_to_markdown(tn, doc_dict["tables"][tn]) for tn in tables_list}
    response = {
        "doc": doc_dict,
        "tables": tables_list,
        "markdown": markdown_full,
        "markdown_by_table": markdown_by_table,
    }
    if table:
        response["markdown_selected"] = markdown_by_table.get(table) or _schema_doc_to_markdown(doc, table_name=table)
    return response
@router.post("/schema/doc/refresh")
async def refresh_schema_doc(
    connection_id: Optional[str] = Query(None, description="ID of the DB connection from POST /connectors/plugin/load"),
    file_id: Optional[str] = Query(None, description="ID of an uploaded file from POST /convobi/files/upload")
):
    _ensure_connection_and_schema(connection_id=connection_id, file_id=file_id)
    meta = app_state.get("schema_metadata")
    if not meta:
        raise HTTPException(status_code=500, detail="Schema not loaded")
    try:
        documenter = SchemaDocumenter(app_state["meaning_cache"], app_state["query_generator"]._ollama_generate, CURRENT_DIALECT)
        doc = documenter.build(meta, skip_llm_narrative=False)
        app_state["schema_doc"] = doc
        fingerprint = SchemaDocumenter._schema_fingerprint(meta)
        db_id = app_state.get("db_id")
        store: SchemaDocCsvStore = app_state.get("schema_doc_store")
        if store and db_id:
            store.write(db_id, fingerprint, doc)
        return {"ok": True, "generated_at": doc.generated_at, "total_tables": doc.total_tables}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"schema_doc_error: {e}")
@router.get("/schema/doc/cache")
async def schema_doc_cache_status(
    connection_id: Optional[str] = Query(None, description="ID of the DB connection from POST /connectors/plugin/load"),
    file_id: Optional[str] = Query(None, description="ID of an uploaded file from POST /convobi/files/upload")
):
    _ensure_connection_and_schema(connection_id=connection_id, file_id=file_id)
    meta = app_state.get("schema_metadata")
    if not meta:
        raise HTTPException(status_code=500, detail="Schema not loaded")
    fingerprint = SchemaDocumenter._schema_fingerprint(meta)
    db_id = app_state.get("db_id")
    store: SchemaDocCsvStore = app_state.get("schema_doc_store")
    if not store or not db_id:
        return {"cached": False, "error": "store_not_initialized"}
    doc = store.read(db_id, fingerprint)
    ddir = store._db_dir(db_id)
    return {
        "cached": bool(doc),
        "db_id": db_id,
        "fingerprint": fingerprint,
        "directory": ddir,
        "files": {
            "meta": os.path.join(ddir, "meta.csv"),
            "tables": os.path.join(ddir, "tables.csv"),
            "columns": os.path.join(ddir, "columns.csv")
        }
    }
@router.post("/schema/doc/cache/clear")
async def schema_doc_cache_clear(
    connection_id: Optional[str] = Query(None, description="ID of the DB connection from POST /connectors/plugin/load"),
    file_id: Optional[str] = Query(None, description="ID of an uploaded file from POST /convobi/files/upload")
):
    _ensure_connection_and_schema(connection_id=connection_id, file_id=file_id)
    store: SchemaDocCsvStore = app_state.get("schema_doc_store")
    if not store:
        raise HTTPException(status_code=500, detail="store_not_initialized")
    db_id = app_state.get("db_id")
    if not db_id:
        raise HTTPException(status_code=500, detail="db_id_not_set")
    store.clear(db_id)
    app_state["schema_doc"] = None
    return {"ok": True}
