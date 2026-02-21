
# nlp_db/api_server.py
# NLP Database Interface API (On-Prem Ollama Edition) — v1.7.0
# Adds: DataFixMemoryStore to remember per-question DATA replacements, "apply changes" convenience,
# and teaches the LLM to inspect first 20 UNIQUE values per column before choosing fields.

import os
import re
import csv
import json
import time
import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from urllib.parse import urlparse

import requests
from fastapi import FastAPI, HTTPException, Query, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager

# SQLAlchemy
from sqlalchemy import create_engine, inspect, text, MetaData, Table, func, select
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

# Data ingestion
import pandas as pd

# Optional column glossary for CRM/opportunity schemas (accurate column selection for SQL and charts)
try:
    from .column_glossary import get_column_glossary_description
except ImportError:
    def get_column_glossary_description(column_name: str) -> str:
        return ""

LOG = logging.getLogger("nlpdb")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

DEFAULT_DB_URL = os.getenv("DATABASE_URL")  # may be None

# ------------------------------------------------------------------
# Module-level LLM instance -- fill in before running (used for Convobi NL→SQL)
# ------------------------------------------------------------------
# try:
#     from langchain_openai import ChatOpenAI
#     from langchain_core.messages import SystemMessage, HumanMessage
#     CONVOBI_LLM = ChatOpenAI(
#         model="openai/gpt-oss-20b",          # user will fill
#         base_url="http://10.188.100.131:8004/v1",           # user will fill
#         api_key="ollama",  # user will fill
#         temperature=0.2,
#     )
# except Exception:
#     CONVOBI_LLM = None

try:
    from langchain_openai import ChatOpenAI, AzureChatOpenAI
    from langchain_core.messages import SystemMessage, HumanMessage
#     CONVOBI_LLM = AzureChatOpenAI(
#     azure_endpoint=""I will put it later"",
#     api_key=""I will put it later"",
#     api_version="I will put it later
#     azure_deployment="gpt4o",
#     temperature=0.1,
# )

    CONVOBI_LLM = ChatOpenAI(
        model="liquid/lfm-2.5-1.2b-instruct:free",          # user will fill
        base_url="https://openrouter.ai/api/v1",       # user will fill
        api_key="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",        # user will fill
        temperature=0.2,
        )

except Exception:
    CONVOBI_LLM = None

SUPPORTED_DIALECTS = {"sqlite", "postgres", "mysql", "mssql", "oracle", "bigquery", "snowflake"}
CURRENT_DIALECT = os.getenv("DB_DIALECT", "sqlite").strip().lower()
if CURRENT_DIALECT not in SUPPORTED_DIALECTS:
    CURRENT_DIALECT = "sqlite"
CURRENT_CONNECTOR = None

SCHEMA_DOC_DIR = os.getenv("SCHEMA_DOC_DIR", "schema_docs")
SCHEMA_DOC_TTL = int(os.getenv("SCHEMA_DOC_TTL", "3600"))
# Schema analysis speed vs coverage: lower = faster load, higher = better column semantics for NL→SQL
SAMPLE_ROWS_PER_TABLE = int(os.getenv("SAMPLE_ROWS_PER_TABLE", "200"))  # rows per table for sampling (e.g. 200 fast, 500 thorough)
COLUMN_SAMPLE_SIZE = int(os.getenv("COLUMN_SAMPLE_SIZE", "25"))  # first-K UNIQUE values per column for semantics
TOP_K_COMMON_VALUES = int(os.getenv("TOP_K_COMMON_VALUES", "10"))
# How many unique value examples to show per column in schema context (so LLM picks correct columns)
SCHEMA_EXAMPLES_DISPLAY = int(os.getenv("SCHEMA_EXAMPLES_DISPLAY", "20"))
UPLOADS_DIR = os.getenv("UPLOADS_DIR", "uploads")

LEARNING_FIXES_PATH = os.getenv("LEARNING_FIXES_PATH", "learning_fixes.json")
# NEW: persistent data (value) fixes
LEARNING_DATA_FIXES_PATH = os.getenv("LEARNING_DATA_FIXES_PATH", "learning_data_fixes.json")
# First-K UNIQUE non-null values per column (used in column stats and cache)
FIRST_K_UNIQUE_VALUES = int(os.getenv("FIRST_K_UNIQUE_VALUES", "25"))

os.makedirs(UPLOADS_DIR, exist_ok=True)

# ────────────────────────────────────────────────────────────────────────────────
# API Models
# ────────────────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(..., description="Natural language question")
    language: str = Field("sql", description="Query language: only 'sql' is supported")
    max_results: Optional[int] = Field(None, description="Maximum number of rows")
    use_cache: bool = Field(True, description="Return cached result when available")
    auto_proceed: bool = Field(True, description="If True, execute even when warnings are present (default True)")
    connector_id: Optional[str] = Field(None, description="ID of the DB connection from POST /connectors/plugin/load to recognize which DB is used")
    file_id: Optional[str] = Field(None, description="ID of an uploaded file from POST /convobi/files/upload; use either connector_id or file_id")


class QueryResponse(BaseModel):
    success: bool
    sql: Optional[str] = None
    explanation: Optional[str] = None
    data: Optional[List[Dict[str, Any]]] = None
    row_count: Optional[int] = None
    columns: Optional[List[str]] = None
    visualization_hints: Optional[Dict[str, Any]] = None
    execution_time_ms: Optional[float] = None
    confidence: Optional[float] = None
    optimizations: Optional[List[str]] = None
    warnings: Optional[List[str]] = None
    error: Optional[str] = None
    summary: Optional[str] = None
    visualization_spec: Optional[Dict[str, Any]] = None
    visualization_html: Optional[str] = None


class SchemaResponse(BaseModel):
    tables: Dict[str, Any]
    total_tables: int
    summary: str


class SummarizeRequest(BaseModel):
    question: str
    columns: List[str]
    data: List[Dict[str, Any]]
    visualization_hints: Optional[Dict[str, Any]] = None


class ColumnMeaningRequest(BaseModel):
    table: str
    column: str
    meaning: str
    description: Optional[str] = None
    confidence: Optional[float] = None
    source: Optional[str] = None
    examples: Optional[List[str]] = None


class ColumnMeaningRefineRequest(BaseModel):
    table: str
    column: str
    tone: Optional[str] = Field("concise")
    max_words: Optional[int] = Field(40)


class SuggestQuestionsRequest(BaseModel):
    max_examples: int = Field(30)
    include_tables: Optional[List[str]] = None


class SuggestQuestionsResponse(BaseModel):
    examples: List[Dict[str, Any]]


class RefineTableColumnsRequest(BaseModel):
    table: str
    tone: Optional[str] = Field("business")
    max_words: Optional[int] = Field(40)
    only_missing: bool = Field(True)

# ────────────────────────────────────────────────────────────────────────────────
# Internal Models
# ────────────────────────────────────────────────────────────────────────────────

class TableColumn(BaseModel):
    name: str
    type: str
    nullable: bool
    primary_key: bool = False
    foreign_key: Optional[str] = None


class TableMetadata(BaseModel):
    name: str
    row_count: Optional[int]
    columns: List[TableColumn]
    relationships: List[str] = Field(default_factory=list)


class QueryResult(BaseModel):
    sql: Optional[str] = None
    explanation: Optional[str] = None
    confidence: Optional[float] = 0.7
    optimizations: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class ExecutionResult(BaseModel):
    success: bool
    data: Optional[List[Dict[str, Any]]] = None
    row_count: Optional[int] = None
    columns: Optional[List[str]] = None
    visualization_hints: Optional[Dict[str, Any]] = None
    execution_time_ms: Optional[float] = None
    error: Optional[str] = None
    needs_confirmation: Optional[bool] = None
    proposed_fixes: Optional[List[Dict[str, Any]]] = None


class ConversationMessage(BaseModel):
    timestamp: str
    question: str
    sql: Optional[str]
    success: bool
    row_count: Optional[int]

# ────────────────────────────────────────────────────────────────────────────────
# Result Cache
# ────────────────────────────────────────────────────────────────────────────────

class ResultCache:
    def __init__(self, ttl_seconds: Optional[int] = 600):
        self._cache: Dict[str, Tuple[float, ExecutionResult]] = {}
        self.ttl_seconds = ttl_seconds

    def get(self, key: str) -> Optional[ExecutionResult]:
        item = self._cache.get(key)
        if not item: return None
        ts, value = item
        if self.ttl_seconds and (time.time() - ts) > self.ttl_seconds:
            self._cache.pop(key, None)
            return None
        return value

    def put(self, key: str, value: ExecutionResult):
        self._cache[key] = (time.time(), value)

    def clear(self): self._cache.clear()

# ────────────────────────────────────────────────────────────────────────────────
# Column Meaning Cache
# ────────────────────────────────────────────────────────────────────────────────

class ColumnMeaning(BaseModel):
    table: str
    column: str
    meaning: str
    description: Optional[str] = None
    confidence: float = 0.7
    source: str = "heuristic"
    examples: List[str] = Field(default_factory=list)


class ColumnMeaningCache:
    def __init__(self, ttl_seconds: Optional[int] = 3600):
        self._cache: Dict[str, Tuple[float, ColumnMeaning]] = {}
        self.ttl_seconds = ttl_seconds

    def _key(self, table: str, column: str) -> str:
        return f"{table}.{column}".lower()

    def get(self, table: str, column: str) -> Optional[ColumnMeaning]:
        key = self._key(table, column)
        item = self._cache.get(key)
        if not item: return None
        ts, meaning = item
        if self.ttl_seconds and (time.time() - ts) > self.ttl_seconds:
            self._cache.pop(key, None)
            return None
        return meaning

    def put(self, meaning: ColumnMeaning):
        key = self._key(meaning.table, meaning.column)
        self._cache[key] = (time.time(), meaning)

    def clear(self): self._cache.clear()

# ────────────────────────────────────────────────────────────────────────────────
# NEW: Fix Memory Store (column) + Data Fix Memory Store (values)
# ────────────────────────────────────────────────────────────────────────────────

_STOPWORDS = {
    "the","a","an","of","for","to","in","and","or","by","on","at","from","with","over","last","past",
    "within","between","during","this","that","these","those","all","any","show","list","count","average","avg",
    "sum","total","number","how","many","much"
}

def _normalize_tokens(s: str) -> List[str]:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9_\s]", " ", s)
    toks = [t for t in s.split() if t and t not in _STOPWORDS]
    return toks

def _jaccard(a: set, b: set) -> float:
    if not a and not b: return 1.0
    if not a or not b: return 0.0
    inter = len(a & b); union = len(a | b)
    return inter / max(1, union)

class FixMemoryStore:
    """
    Persist per-question fix maps and reuse them for similar questions.
    File format:
    {
      "entries": [
        {"question":"...", "tokens":["..."], "fix_map":{"bad":"good"}, "ts":"..."}
      ]
    }
    """
    def __init__(self, path: str = LEARNING_FIXES_PATH, sim_threshold: float = 0.6):
        self.path = path
        self.sim_threshold = sim_threshold
        self._store: Dict[str, Any] = {"entries": []}
        self._load()

    def _load(self):
        try:
            if os.path.exists(self.path):
                with open(self.path, "r", encoding="utf-8") as f:
                    self._store = json.load(f) or {"entries": []}
        except Exception as e:
            LOG.warning(f"FixMemoryStore load failed: {e}")
            self._store = {"entries": []}

    def _save(self):
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self._store, f, indent=2)
        except Exception as e:
            LOG.warning(f"FixMemoryStore save failed: {e}")

    def add(self, question: str, fix_map: Dict[str, str]):
        if not question or not fix_map: return
        toks = _normalize_tokens(question)
        best_idx, best_sim = None, 0.0
        for i, e in enumerate(self._store["entries"]):
            sim = _jaccard(set(toks), set(e.get("tokens") or []))
            if sim > best_sim:
                best_idx, best_sim = i, sim
        if best_idx is not None and best_sim >= self.sim_threshold:
            entry = self._store["entries"][best_idx]
            merged = dict(entry.get("fix_map") or {})
            merged.update(fix_map)  # new overrides old
            entry["fix_map"] = merged
            entry["ts"] = datetime.now().isoformat()
        else:
            self._store["entries"].append({
                "question": question,
                "tokens": toks,
                "fix_map": fix_map,
                "ts": datetime.now().isoformat()
            })
        self._save()
        LOG.info(f"✓ Learned fix for question (sim={best_sim:.2f}): {fix_map}")

    def find_for(self, question: str) -> Dict[str, str]:
        """Return best-matching fix_map (if similarity >= threshold)."""
        if not question: return {}
        toks = _normalize_tokens(question)
        best_entry, best_sim = None, 0.0
        for e in self._store["entries"]:
            sim = _jaccard(set(toks), set(e.get("tokens") or []))
            if sim > best_sim:
                best_entry, best_sim = e, sim
        if best_entry and best_sim >= self.sim_threshold:
            LOG.info(f"✓ Applying learned fix (sim={best_sim:.2f})")
            return dict(best_entry.get("fix_map") or {})
        return {}

    def list(self) -> List[Dict[str, Any]]:
        return list(self._store.get("entries", []))

    def clear(self):
        self._store = {"entries": []}
        self._save()


class DataFixMemoryStore:
    """
    Persist per-question DATA (literal value) fix maps and reuse them for similar questions.
    File format:
    { "entries": [ {"question":"...", "tokens":["..."], "data_fix_map":{"old":"new"}, "ts":"..."} ] }
    """
    def __init__(self, path: str = LEARNING_DATA_FIXES_PATH, sim_threshold: float = 0.6):
        self.path = path
        self.sim_threshold = sim_threshold
        self._store: Dict[str, Any] = {"entries": []}
        self._load()

    def _load(self):
        try:
            if os.path.exists(self.path):
                with open(self.path, "r", encoding="utf-8") as f:
                    self._store = json.load(f) or {"entries": []}
        except Exception as e:
            LOG.warning(f"DataFixMemoryStore load failed: {e}")
            self._store = {"entries": []}

    def _save(self):
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self._store, f, indent=2)
        except Exception as e:
            LOG.warning(f"DataFixMemoryStore save failed: {e}")

    def add(self, question: str, data_fix_map: Dict[str, str]):
        if not question or not data_fix_map: return
        toks = _normalize_tokens(question)
        best_idx, best_sim = None, 0.0
        for i, e in enumerate(self._store["entries"]):
            sim = _jaccard(set(toks), set(e.get("tokens") or []))
            if sim > best_sim:
                best_idx, best_sim = i, sim
        if best_idx is not None and best_sim >= self.sim_threshold:
            entry = self._store["entries"][best_idx]
            merged = dict(entry.get("data_fix_map") or {})
            merged.update(data_fix_map)
            entry["data_fix_map"] = merged
            entry["ts"] = datetime.now().isoformat()
        else:
            self._store["entries"].append({
                "question": question,
                "tokens": toks,
                "data_fix_map": data_fix_map,
                "ts": datetime.now().isoformat()
            })
        self._save()
        LOG.info(f"✓ Learned DATA fixes for question (sim={best_sim:.2f}): {data_fix_map}")

    def find_for(self, question: str) -> Dict[str, str]:
        if not question: return {}
        toks = _normalize_tokens(question)
        best_entry, best_sim = None, 0.0
        for e in self._store["entries"]:
            sim = _jaccard(set(toks), set(e.get("tokens") or []))
            if sim > best_sim:
                best_entry, best_sim = e, sim
        if best_entry and best_sim >= self.sim_threshold:
            LOG.info(f"✓ Applying learned DATA fixes (sim={best_sim:.2f})")
            return dict(best_entry.get("data_fix_map") or {})
        return {}

    def list(self) -> List[Dict[str, Any]]:
        return list(self._store.get("entries", []))

    def clear(self):
        self._store = {"entries": []}
        self._save()

# ────────────────────────────────────────────────────────────────────────────────
# Helpers & Detectors
# ────────────────────────────────────────────────────────────────────────────────

def _tokens(s: str) -> List[str]:
    s = s.lower().replace("-", "_")
    return [t for t in re.split(r"[_\s]+", s) if t]

def _singularize(s: str) -> str:
    s = s.lower()
    if s.endswith("ies"): return s[:-3] + "y"
    if s.endswith("s") and not s.endswith("ss"): return s[:-1]
    return s

def _table_singular(table: str) -> str: return _singularize(table)

def _fk_to_text(fk: Optional[str]) -> Optional[str]:
    if not fk: return None
    try:
        table, col = fk.split(".", 1)
        return f"{table}.{col}"
    except Exception:
        return fk

def _value_is_numeric(v: Any) -> bool:
    if isinstance(v, (int, float)) and not isinstance(v, bool): return True
    if isinstance(v, str):
        return bool(re.match(r"^-?\d+(\.\d+)?$", v.strip()))
    return False

def _value_to_float(v: Any) -> Optional[float]:
    if isinstance(v, (int, float)) and not isinstance(v, bool): return float(v)
    if isinstance(v, str):
        s = v.strip()
        try:
            return float(s) if bool(re.match(r"^-?\d+(\.\d+)?$", s)) else None
        except Exception:
            return None
    return None

def _value_is_date_like(v: Any) -> bool:
    if isinstance(v, datetime): return True
    if isinstance(v, str):
        s = v.strip()
        patterns = [
            r"^\d{4}-\d{2}-\d{2}$",
            r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(:\d{2})?$",
            r"^\d{2}/\d{2}/\d{4}$",
            r"^\d{2}-\d{2}-\d{4}$",
            r"^\d{1,2}/\d{1,2}/\d{4}\s+\d{1,2}:\d{2}:\d{2}\s+(AM|PM)$",
        ]
        return any(re.match(p, s) for p in patterns)
    return False

def _value_is_email(v: Any) -> bool:
    return isinstance(v, str) and re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", (v or "").strip()) is not None

def _value_is_phone(v: Any) -> bool:
    return isinstance(v, str) and re.match(r"^\+?\d[\d\-\s]{7,}$", (v or "").strip()) is not None

def _value_is_uuid(v: Any) -> bool:
    return isinstance(v, str) and re.match(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}$", (v or "").strip()) is not None

def _value_is_url(v: Any) -> bool:
    return isinstance(v, str) and re.match(r"^https?://", (v or "").strip(), flags=re.IGNORECASE) is not None

def _value_is_ip(v: Any) -> bool:
    return isinstance(v, str) and re.match(r"^(?:\d{1,3}\.){3}\d{1,3}$", (v or "").strip()) is not None

def _value_is_epoch(v: Any) -> bool:
    s = str(v).strip()
    return bool(re.match(r"^\d{10}$", s) or re.match(r"^\d{13}$", s))

def _is_boolean_like(values: List[Any]) -> bool:
    truthy = {"y", "yes", "true", "t", "1", "met"}
    falsy = {"n", "no", "false", "f", "0", "not met"}
    seen = set()
    for v in values:
        if v is None: continue
        s = str(v).strip().lower()
        if s in truthy or s in falsy:
            seen.add(s)
        else:
            return False
    return len(seen) > 0

def _most_common(values: List[Any]) -> List[str]:
    freq: Dict[str, int] = {}
    for v in values:
        if v is None: continue
        key = str(v)
        freq[key] = freq.get(key, 0) + 1
    items = sorted(freq.items(), key=lambda x: (-x[1], x[0]))
    return [k for k, _ in items[:TOP_K_COMMON_VALUES]]

# NEW: first-K unique (order-preserving)
def _first_unique(values: List[Any], k: int) -> List[str]:
    seen = set()
    out: List[str] = []
    for v in values:
        if v is None: continue
        sv = str(v)
        if sv in seen: continue
        seen.add(sv)
        out.append(sv)
        if len(out) >= k: break
    return out

def _safe_range(nums: List[float]) -> Optional[Tuple[float, float]]:
    if not nums: return None
    return (min(nums), max(nums))

def _format_range(r: Optional[Tuple[float, float]]) -> Optional[str]:
    if not r: return None
    lo, hi = r
    return f"{lo:g}–{hi:g}" if lo != hi else f"{lo:g}"

def _quote_ident(name: str) -> str:
    if not name: return name
    if CURRENT_DIALECT == "bigquery": return f"`{name}`"
    if CURRENT_DIALECT == "mssql":
        safe = name.replace("]", "]]")
        return f"[{safe}]"
    return f'"{name}"'

def _quote_full_ident(name: str) -> str:
    parts = [p for p in (name or "").split(".") if p != ""]
    if not parts: return _quote_ident(name)
    return ".".join(_quote_ident(p) for p in parts)

# ────────────────────────────────────────────────────────────────────────────────
# Column statistics & semantics
# ────────────────────────────────────────────────────────────────────────────────

class ColumnStats(BaseModel):
    non_null_count: int
    null_ratio: float
    distinct_count: int
    unique_ratio: float
    avg_length: Optional[float] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    numeric_min: Optional[float] = None
    numeric_max: Optional[float] = None
    seen_examples: List[str] = Field(default_factory=list)

def _compute_column_stats(values: List[Any]) -> ColumnStats:
    non_null = [v for v in values if v is not None]
    n = len(values); nn = len(non_null)
    null_ratio = 0.0 if n == 0 else (n - nn) / max(1, n)
    distinct_values = list({str(v) for v in non_null}); distinct_count = len(distinct_values)
    unique_ratio = 0.0 if nn == 0 else (distinct_count / nn)
    str_vals = [str(v) for v in non_null]
    lengths = [len(s) for s in str_vals] if str_vals else []
    avg_len = (sum(lengths) / len(lengths)) if lengths else None
    min_len = (min(lengths) if lengths else None); max_len = (max(lengths) if lengths else None)
    nums = [_value_to_float(v) for v in non_null if _value_is_numeric(v)]
    nums = [x for x in nums if x is not None]
    numeric_min = min(nums) if nums else None; numeric_max = max(nums) if nums else None
    # NEW: show first-K UNIQUE values to LLM
    examples = _first_unique(non_null, FIRST_K_UNIQUE_VALUES)
    return ColumnStats(
        non_null_count=nn, null_ratio=null_ratio, distinct_count=distinct_count, unique_ratio=unique_ratio,
        avg_length=avg_len, min_length=min_len, max_length=max_len,
        numeric_min=numeric_min, numeric_max=numeric_max, seen_examples=examples
    )

def _craft_meaning_description_generic(table: str, col_name: str, meaning: str, fk: Optional[str], stats: ColumnStats) -> str:
    role = _table_singular(table) or "record"
    fk_text = _fk_to_text(fk)
    ex_txt = f" Common values: {', '.join(stats.seen_examples)}." if stats.seen_examples else ""
    rng_txt = None
    if stats.numeric_min is not None and stats.numeric_max is not None:
        rng_txt = _format_range((stats.numeric_min, stats.numeric_max))
    range_txt = f" Range observed: {rng_txt}." if rng_txt else ""
    tail_fk = f" References {fk_text}." if fk_text else ""
    name = col_name.replace("_", " ")
    meaning_map = {
        "id": f"Unique identifier for the {role} used for tracking and joins.{tail_fk}",
        "uuid": f"Globally unique identifier (UUID) for the {role}.{ex_txt}",
        "timestamp": f"Date/time related to the {role} (e.g., creation, update, registered).",
        "email": f"Email address related to the {role}.{ex_txt}",
        "phone": f"Phone/contact number related to the {role}.{ex_txt}",
        "url": f"URL/link related to the {role}.{ex_txt}",
        "ip": f"IP address related to the {role}.{ex_txt}",
        "geo_lat": "Latitude component of a geographic coordinate.",
        "geo_lon": "Longitude component of a geographic coordinate.",
        "boolean": f"Yes/No indicator for {name}.{ex_txt}",
        "numeric": f"Numeric measure related to the {role}.{range_txt}".strip(),
        "duration": f"Elapsed time/interval related to the {role}.{range_txt}".strip(),
        "category": f"Categorical label used to group or classify the {role}.{ex_txt}",
        "text": "Free-text/details related to the record.",
        "code": f"Code-like field (compact categorical identifier) related to the {role}.{ex_txt}",
    }
    return meaning_map.get(meaning, f"{name.capitalize()} related to the {role}.{ex_txt}")

def _column_semantics_from_samples_generic(table: str, col_name: str, samples: List[Any], fk: Optional[str]) -> Optional[ColumnMeaning]:
    if not samples: return None
    stats = _compute_column_stats(samples)
    name = col_name.lower()
    non_null = [v for v in samples if v is not None]
    n = len(non_null)

    def ratio(pred) -> float:
        return sum(1 for v in non_null if pred(v)) / max(1, n)

    email_ratio = ratio(_value_is_email)
    phone_ratio = ratio(_value_is_phone)
    uuid_ratio = ratio(_value_is_uuid)
    url_ratio = ratio(_value_is_url)
    ip_ratio = ratio(_value_is_ip)
    date_ratio = ratio(_value_is_date_like)
    epoch_ratio = ratio(_value_is_epoch)
    num_ratio = sum(1 for v in non_null if _value_is_numeric(v)) / max(1, n)
    bool_like = _is_boolean_like(non_null)

    is_id_like = (name == "id" or name.endswith("_id") or re.search(r"\bid\b", name))
    is_time_hint = any(k in name for k in [
        "date","datetime","datestamp","timestamp","ts","time","timepoint","timecode",
        "created_at", "create_time", "inserted_at", "ingested_at",
        "loaded_at", "ingest_time", "load_time",
        "updated_at", "update_time", "modified_at",
        "last_modified", "last_modified_at", "deleted", "deleted_at",
        "opened_at", "closed_at", "started_at", "ended_at", "completed_at", "resolved_at",
        "approved_at","submitted_at",
        "effective_date", "effective_from", "effective_to", "valid_from", "valid_to",
        "activation_date", "deactivation_date", "expires", "expiry", "expiration", "expiration_date",
        "due", "due_date", "sla_due",
        "publish_date", "published_at", "unpublished_at",
        "birthdate", "dob", "anniversary", "join_date", "hire_date", "retire_date",
        "invoice_date", "posting_date", "posting_time", "posting_datetime", "order_date", "ship_date", "delivery_date",
        "trans_date", "txn_date", "event_time", "event_timestamp", "log_time", "log_timestamp", "run_date", "run_time", "batch_date", "batch_time",
        "month","month_num","month_name","yearmonth","fiscal_month",
        "quarter","qtr","fiscal_quarter","week","week_start","week_end",
        "year","fiscal_year","yearweek",
        "_at","_on","_time","_date","_dt","_ts","_month","month_",
        "registered"
    ])
    is_duration_name = any(k in name for k in ["elapsed","duration","mins","minutes","seconds","response_time","resolution_time"])
    numeric_vals = [_value_to_float(v) for v in non_null if _value_is_numeric(v)]
    numeric_vals = [x for x in numeric_vals if x is not None]

    if uuid_ratio > 0.5 or ("uuid" in name or "guid" in name):
        desc = _craft_meaning_description_generic(table, col_name, "uuid", fk, stats)
        return ColumnMeaning(table=table, column=col_name, meaning="uuid", description=desc, confidence=max(0.9, uuid_ratio), source="heuristic", examples=stats.seen_examples)

    if is_id_like or (num_ratio > 0.7 and stats.unique_ratio > 0.7):
        desc = _craft_meaning_description_generic(table, col_name, "id", fk, stats)
        return ColumnMeaning(table=table, column=col_name, meaning="id", description=desc, confidence=0.88, source="heuristic", examples=stats.seen_examples)

    if email_ratio > 0.5 or "email" in name:
        desc = _craft_meaning_description_generic(table, col_name, "email", fk, stats)
        return ColumnMeaning(table=table, column=col_name, meaning="email", description=desc, confidence=max(0.9, email_ratio), source="heuristic", examples=stats.seen_examples)

    def _name_implies_phone(n: str) -> bool:
        if any(tok in n for tok in ["phone","mobile","tel","telephone","cell"]): return True
        if "contact" in n and not (n.endswith("_name") or re.search(r"\bname\b", n)): return True
        return False

    is_phoney_name = _name_implies_phone(name)
    if phone_ratio > 0.5 or (is_phoney_name and phone_ratio > 0.1):
        desc = _craft_meaning_description_generic(table, col_name, "phone", fk, stats)
        return ColumnMeaning(table=table, column=col_name, meaning="phone", description=desc, confidence=max(0.85, phone_ratio), source="heuristic", examples=stats.seen_examples)

    if url_ratio > 0.5 or "url" in name or "link" in name:
        desc = _craft_meaning_description_generic(table, col_name, "url", fk, stats)
        return ColumnMeaning(table=table, column=col_name, meaning="url", description=desc, confidence=max(0.9, url_ratio), source="heuristic", examples=stats.seen_examples)

    if ip_ratio > 0.5 or "ip" in name:
        desc = _craft_meaning_description_generic(table, col_name, "ip", fk, stats)
        return ColumnMeaning(table=table, column=col_name, meaning="ip", description=desc, confidence=max(0.9, ip_ratio), source="heuristic", examples=stats.seen_examples)

    is_geo_lat = (("lat" in name or name.endswith("_latitude")) and ratio(lambda v: _value_to_float(v) is not None and -90.0 <= float(_value_to_float(v)) <= 90.0) > 0.5)
    is_geo_lon = (("lon" in name or "lng" in name or name.endswith("_longitude")) and ratio(lambda v: _value_to_float(v) is not None and -180.0 <= float(_value_to_float(v)) <= 180.0) > 0.5)
    if is_geo_lat:
        desc = _craft_meaning_description_generic(table, col_name, "geo_lat", fk, stats)
        return ColumnMeaning(table=table, column=col_name, meaning="geo_lat", description=desc, confidence=0.9, source="heuristic", examples=stats.seen_examples)
    if is_geo_lon:
        desc = _craft_meaning_description_generic(table, col_name, "geo_lon", fk, stats)
        return ColumnMeaning(table=table, column=col_name, meaning="geo_lon", description=desc, confidence=0.9, source="heuristic", examples=stats.seen_examples)

    if date_ratio > 0.5 or epoch_ratio > 0.5 or is_time_hint:
        desc = _craft_meaning_description_generic(table, col_name, "timestamp", fk, stats)
        return ColumnMeaning(table=table, column=col_name, meaning="timestamp", description=desc, confidence=max(0.85, date_ratio, epoch_ratio), source="heuristic", examples=stats.seen_examples)

    if bool_like:
        desc = _craft_meaning_description_generic(table, col_name, "boolean", fk, stats)
        return ColumnMeaning(table=table, column=col_name, meaning="boolean", description=desc, confidence=0.9, source="heuristic", examples=stats.seen_examples)

    if is_duration_name and num_ratio > 0.5:
        desc = _craft_meaning_description_generic(table, col_name, "duration", fk, stats)
        return ColumnMeaning(table=table, column=col_name, meaning="duration", description=desc, confidence=0.85, source="heuristic", examples=stats.seen_examples)

    if num_ratio > 0.7:
        desc = _craft_meaning_description_generic(table, col_name, "numeric", fk, stats)
        return ColumnMeaning(table=table, column=col_name, meaning="numeric", description=desc, confidence=max(0.8, num_ratio), source="heuristic", examples=stats.seen_examples)

    def _is_code_like(col_name: str) -> bool:
        nm = col_name.lower()
        return any(k in nm for k in ["code","status_code","err_code","zip","postal","country_code"])
    if _is_code_like(col_name):
        desc = _craft_meaning_description_generic(table, col_name, "code", fk, stats)
        return ColumnMeaning(table=table, column=col_name, meaning="code", description=desc, confidence=0.8, source="heuristic", examples=stats.seen_examples)

    few_distinct_threshold = 3 if stats.non_null_count >= 10 else max(5, stats.non_null_count // 2)
    name_hint_category = any(k in name for k in ["status","state","type","category","severity","priority","source","code"])
    if stats.distinct_count <= few_distinct_threshold or name_hint_category:
        desc = _craft_meaning_description_generic(table, col_name, "category", fk, stats)
        return ColumnMeaning(table=table, column=col_name, meaning="category", description=desc, confidence=0.75, source="heuristic", examples=stats.seen_examples)

    desc = _craft_meaning_description_generic(table, col_name, "text", fk, stats)
    return ColumnMeaning(table=table, column=col_name, meaning="text", description=desc, confidence=0.65, source="heuristic", examples=stats.seen_examples)

def infer_probable_fks(engine: Engine, tables: Dict[str, TableMetadata], sample_n: int = 500, overlap_threshold: float = 0.6) -> Dict[Tuple[str, str], str]:
    result: Dict[Tuple[str, str], str] = {}
    pk_samples: Dict[str, set] = {}
    with engine.connect() as conn:
        for tn, tm in tables.items():
            pk_cols = [c.name for c in tm.columns if c.primary_key]
            if not pk_cols: continue
            pk = pk_cols[0]
            try:
                q_table = _quote_full_ident(tn)
                q_pk = _quote_ident(pk)
                if CURRENT_DIALECT == "mssql":
                    rows = conn.execute(text(f"SELECT TOP {sample_n} {q_pk} FROM {q_table} WHERE {q_pk} IS NOT NULL")).mappings().all()
                else:
                    rows = conn.execute(text(f"SELECT {q_pk} FROM {q_table} WHERE {q_pk} IS NOT NULL LIMIT {sample_n}")).mappings().all()
                pk_samples[tn] = {r.get(pk) for r in rows if r.get(pk) is not None}
            except Exception:
                pk_samples[tn] = set()
        for src_tn, tm in tables.items():
            q_src_table = _quote_full_ident(src_tn)
            for c in tm.columns:
                if c.primary_key or c.foreign_key: continue
                try:
                    q_col = _quote_ident(c.name)
                    if CURRENT_DIALECT == "mssql":
                        rows = conn.execute(text(f"SELECT TOP {sample_n} {q_col} FROM {q_src_table} WHERE {q_col} IS NOT NULL")).mappings().all()
                    else:
                        rows = conn.execute(text(f"SELECT {q_col} FROM {q_src_table} WHERE {q_col} IS NOT NULL LIMIT {sample_n}")).mappings().all()
                    vals = [r.get(c.name) for r in rows if r.get(c.name) is not None]
                except Exception:
                    vals = []
                if not vals: continue
                stats = _compute_column_stats(vals)
                if stats.unique_ratio < 0.6: continue
                candidate_set = set(vals)
                for tgt_tn, tgt_set in pk_samples.items():
                    if not tgt_set: continue
                    overlap = len(candidate_set & tgt_set)
                    ratio = overlap / max(1, len(candidate_set))
                    if ratio >= overlap_threshold:
                        tgt_pk = next((tc.name for tc in tables[tgt_tn].columns if tc.primary_key), None)
                        if tgt_pk:
                            result[(src_tn, c.name)] = f"{tgt_tn}.{tgt_pk}"
    return result

# ────────────────────────────────────────────────────────────────────────────────
# Schema Analyzer
# ────────────────────────────────────────────────────────────────────────────────

class SchemaAnalyzer:
    def __init__(self, connection_string: str, meaning_cache: ColumnMeaningCache):
        self.connection_string = connection_string
        self.engine: Engine = create_engine(connection_string, pool_pre_ping=True)
        self.meaning_cache = meaning_cache

    def analyze_schema(self) -> Dict[str, TableMetadata]:
        LOG.info("Analyzing SQL schema...")
        metadata: Dict[str, TableMetadata] = {}
        inspector = inspect(self.engine)

        for table_name in inspector.get_table_names():
            columns: List[TableColumn] = []
            pk_cols = inspector.get_pk_constraint(table_name).get("constrained_columns", []) or []
            for col in inspector.get_columns(table_name):
                columns.append(TableColumn(
                    name=col.get("name"),
                    type=str(col.get("type")),
                    nullable=bool(col.get("nullable", True)),
                    primary_key=col.get("name") in pk_cols,
                    foreign_key=None
                ))
            fk_constraints = inspector.get_foreign_keys(table_name)
            relationships: List[str] = []
            fk_map: Dict[str, str] = {}
            for fk in fk_constraints:
                referred_table = fk.get("referred_table")
                constrained_cols = fk.get("constrained_columns", [])
                referred_cols = fk.get("referred_columns", [])
                rel = f"{table_name}({', '.join(constrained_cols)}) -> {referred_table}({', '.join(referred_cols)})"
                relationships.append(rel)
                for c in constrained_cols:
                    fk_map[c] = f"{referred_table}.{referred_cols[0] if referred_cols else 'id'}"
            for c in columns:
                if c.name in fk_map:
                    c.foreign_key = fk_map[c.name]

            row_count = None
            try:
                t = Table(table_name, MetaData(), autoload_with=self.engine)
                with self.engine.connect() as conn:
                    rc = conn.execute(select(func.count()).select_from(t)).scalar()
                    row_count = int(rc) if rc is not None else None
            except Exception as e:
                LOG.warning(f"Row count failed for {table_name}: {e}")

            metadata[table_name] = TableMetadata(
                name=table_name, row_count=row_count, columns=columns, relationships=relationships
            )

        # Sampling for column semantics
        try:
            with self.engine.connect() as conn:
                for table_name, tmeta in metadata.items():
                    q_cols = ", ".join(_quote_ident(c.name) for c in tmeta.columns)
                    q_table = _quote_full_ident(table_name)
                    pk_cols = [c.name for c in tmeta.columns if c.primary_key]
                    order_clause = ""
                    if pk_cols:
                        q_pk = _quote_ident(pk_cols[0])
                        order_clause = f" ORDER BY {q_pk} ASC"
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
                        # First-K UNIQUE non-null values per column for semantics and LLM column selection
                        samples_raw = [v for v in col_samples.get(c.name, []) if v is not None]
                        samples_k = _first_unique(samples_raw, COLUMN_SAMPLE_SIZE)
                        inferred = _column_semantics_from_samples_generic(table_name, c.name, samples_k, c.foreign_key) if samples_k else None
                        if inferred and len(samples_k) < COLUMN_SAMPLE_SIZE:
                            inferred.confidence = min(inferred.confidence, 0.7)
                        if inferred:
                            gloss_desc = get_column_glossary_description(c.name)
                            if gloss_desc:
                                inferred = ColumnMeaning(
                                    table=inferred.table, column=inferred.column,
                                    meaning=inferred.meaning, description=gloss_desc,
                                    confidence=max(inferred.confidence, 0.9), source="glossary+heuristic",
                                    examples=inferred.examples
                                )
                            self.meaning_cache.put(inferred)
        except Exception as e:
            LOG.debug(f"Sampling for meanings failed: {e}")

        # Optional: infer probable FKs
        try:
            probable = infer_probable_fks(self.engine, metadata)
            for (tn, cn), target in probable.items():
                tm = metadata.get(tn)
                if not tm: continue
                for c in tm.columns:
                    if c.name == cn and not c.foreign_key:
                        c.foreign_key = target
                tm.relationships.append(f"{tn}({cn}) -> {target}")
        except Exception as e:
            LOG.debug(f"Probable FK inference skipped: {e}")

        LOG.info(f"Schema loaded: {len(metadata)} tables")
        return metadata

    
    def generate_llm_context(self, schema_metadata: Dict[str, TableMetadata], verbose: bool = False) -> str:
        lines: List[str] = []
        lines.append(f"Database Schema Context (dialect: {CURRENT_DIALECT}):")

        for tname, tmeta in schema_metadata.items():
        # Improved: infer a readable, table-specific purpose
            try:
                purpose = _infer_table_purpose(tmeta)
            except Exception:
                purpose = "Stores domain records with identifiers and attributes."

            rc = f"{tmeta.row_count} rows" if tmeta.row_count is not None else "rows: unknown"
            rel = "; ".join(tmeta.relationships) if tmeta.relationships else "None"

            lines.append("")
            lines.append(f"Table: {tname}")
            lines.append(f"Purpose: {purpose}")
            lines.append(f"Rows: {rc}")
            lines.append(f"Relationships: {rel}")

            if verbose:
                lines.append("Columns:")
                for c in tmeta.columns:
                    cm = self.meaning_cache.get(tname, c.name)
                    gloss_desc = get_column_glossary_description(c.name)
                    desc = gloss_desc or (cm.description if cm else None) or ""
                    meaning = (cm.meaning if cm else None) or ""
                    pk = " PK" if c.primary_key else ""
                    fk = f" FK->{c.foreign_key}" if c.foreign_key else ""
                    meaning_str = f" [{meaning}]" if meaning else ""
                    desc_str = f" — {desc}" if desc else ""
                    ex_list = (cm.examples if cm else [])[:SCHEMA_EXAMPLES_DISPLAY]
                    ex_str = f" Examples (unique non-null): {', '.join(ex_list)}" if ex_list else ""
                    lines.append(f"- {c.name} ({c.type}{pk}{fk}){meaning_str}{desc_str}{ex_str}")
            else:
                col_str = ", ".join([
                    f"{c.name}:{c.type}{' PK' if c.primary_key else ''}{' FK->'+c.foreign_key if c.foreign_key else ''}"
                    for c in tmeta.columns
                ])
                lines.append(f"Columns: {col_str}")

        lines.append("")
        lines.append(
            f"Use the meanings and examples (up to {SCHEMA_EXAMPLES_DISPLAY} unique non-null values per column) above to select the correct columns "
            "for the user question. Prefer glossary descriptions when present. Generate accurate SQL and choose appropriate columns for visualizations."
        )
        return "\n".join(lines)

    def close(self):
        try:
            self.engine.dispose()
        except Exception:
            pass

# ────────────────────────────────────────────────────────────────────────────────
# Connector (SQL)
# ────────────────────────────────────────────────────────────────────────────────

class ConnectorResult(BaseModel):
    success: bool
    data: Optional[List[Dict[str, Any]]] = None
    row_count: Optional[int] = None
    columns: Optional[List[str]] = None
    visualization_hints: Optional[Dict[str, Any]] = None
    execution_time_ms: Optional[float] = None
    error: Optional[str] = None
    needs_confirmation: Optional[bool] = None
    proposed_fixes: Optional[List[Dict[str, Any]]] = None

class DataConnector:
    def analyze_schema(self) -> Dict[str, TableMetadata]:
        raise NotImplementedError
    def execute(self, query: str) -> ConnectorResult:
        raise NotImplementedError
    def format_for_json(self, res: ConnectorResult) -> Dict[str, Any]:
        return res.dict()

class SqlConnector(DataConnector):
    def __init__(self, connection_string: str):
        self.engine: Engine = create_engine(connection_string, pool_pre_ping=True)

    def analyze_schema(self) -> Dict[str, TableMetadata]:
        inspector = inspect(self.engine)
        metadata: Dict[str, TableMetadata] = {}
        for table_name in inspector.get_table_names():
            columns: List[TableColumn] = []
            pk_cols = inspector.get_pk_constraint(table_name).get("constrained_columns", []) or []
            for col in inspector.get_columns(table_name):
                columns.append(TableColumn(
                    name=col.get("name"), type=str(col.get("type")),
                    nullable=bool(col.get("nullable", True)),
                    primary_key=col.get("name") in pk_cols, foreign_key=None
                ))
            fk_constraints = inspector.get_foreign_keys(table_name)
            relationships: List[str] = []
            for fk in fk_constraints:
                referred_table = fk.get("referred_table")
                constrained_cols = fk.get("constrained_columns", [])
                referred_cols = fk.get("referred_columns", [])
                rel = f"{table_name}({', '.join(constrained_cols)}) -> {referred_table}({', '.join(referred_cols)})"
                relationships.append(rel)
                for c in columns:
                    if c.name in constrained_cols:
                        c.foreign_key = f"{referred_table}.{referred_cols[0] if referred_cols else 'id'}"
            row_count = None
            try:
                t = Table(table_name, MetaData(), autoload_with=self.engine)
                with self.engine.connect() as conn:
                    rc = conn.execute(select(func.count()).select_from(t)).scalar()
                    row_count = int(rc) if rc is not None else None
            except Exception:
                pass
            metadata[table_name] = TableMetadata(
                name=table_name, row_count=row_count, columns=columns, relationships=relationships
            )
        return metadata

    def execute(self, sql: str) -> ConnectorResult:
        start = time.time()
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(sql))
                rows = result.mappings().all()
                columns = list(result.keys()) if hasattr(result, "keys") else (list(rows[0].keys()) if rows else [])
                data: List[Dict[str, Any]] = []
                for r in rows:
                    rec: Dict[str, Any] = {}
                    for k, v in dict(r).items():
                        if isinstance(v, datetime):
                            rec[k] = v.isoformat()
                        else:
                            rec[k] = v
                    data.append(rec)
                viz_hints = infer_visualization(columns, data)
                exec_ms = (time.time() - start) * 1000.0
                return ConnectorResult(success=True, data=data, row_count=len(data), columns=columns,
                                       visualization_hints=viz_hints, execution_time_ms=exec_ms, error=None)
        except SQLAlchemyError as e:
            exec_ms = (time.time() - start) * 1000.0
            return ConnectorResult(success=False, error=str(getattr(e, "__cause__", e)), execution_time_ms=exec_ms)
        except Exception as e:
            exec_ms = (time.time() - start) * 1000.0
            return ConnectorResult(success=False, error=str(e), execution_time_ms=exec_ms)

    def close(self):
        try:
            self.engine.dispose()
        except Exception:
            pass

# ────────────────────────────────────────────────────────────────────────────────
# Visualization Hints
# ────────────────────────────────────────────────────────────────────────────────

iso_full = re.compile(r"^\d{4}-\d{2}-\d{2}([ T]\d{2}:\d{2}(:\d{2})?(Z|[+\-]\d{2}:\d{2})?)?$")
ymd = re.compile(r"^\d{4}-\d{2}-\d{2}$")
ymd_slash = re.compile(r"^\d{4}/\d{2}/\d{2}$")
dmy_slash = re.compile(r"^\d{2}/\d{2}/\d{4}$")
dmy_dash = re.compile(r"^\d{2}-\d{2}-\d{4}$")
ym = re.compile(r"^\d{4}-\d{2}$")
ym01 = re.compile(r"^\d{4}-\d{2}-01$")
epoch_s = re.compile(r"^\d{10}$")
epoch_ms = re.compile(r"^\d{13}$")
numeric_regex = re.compile(r"^-?\d+(\.\d+)?$")

def infer_visualization(columns: List[str], data: List[Dict[str, Any]]) -> Dict[str, Any]:
    hints: Dict[str, Any] = {"suggested": None}
    if not columns or not data:
        hints["suggested"] = {"type": "table"}
        return hints

    def is_numeric(v: Any) -> bool:
        if isinstance(v, (int, float)) and not isinstance(v, bool): return True
        if isinstance(v, str): return bool(numeric_regex.match(v.strip()))
        return False

    def value_time_info(v: Any) -> Optional[Dict[str, Any]]:
        if isinstance(v, datetime):
            return {"is_temporal": True, "unit": None, "epoch": None, "kind": "datetime"}
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            s = str(int(v))
            if epoch_ms.match(s):
                return {"is_temporal": True, "unit": None, "epoch": "millis", "kind": "epoch"}
            if epoch_s.match(s):
                return {"is_temporal": True, "unit": None, "epoch": "seconds", "kind": "epoch"}
            return None
        if isinstance(v, str):
            s = v.strip()
            if iso_full.match(s) or ymd.match(s) or ymd_slash.match(s) or dmy_slash.match(s) or dmy_dash.match(s):
                return {"is_temporal": True, "unit": None, "epoch": None, "kind": "string_date"}
            if ym.match(s) or ym01.match(s):
                return {"is_temporal": True, "unit": "yearmonth", "epoch": None, "kind": "string_month"}
            return None
        return None

    def col_name_is_timey(name: str) -> bool:
        n = name.lower()
        return any(k in n for k in ["date","time","timestamp","created","updated","closed","resolved","registered","month"])

    sample = data[:200]
    time_cols_info: List[Tuple[str, Dict[str, Any]]] = []
    numeric_cols: List[str] = []
    categorical_cols: List[str] = []

    for col in columns:
        vals = [r.get(col) for r in sample if r.get(col) is not None]
        if not vals: continue
        infos = [value_time_info(v) for v in vals]
        temporal_votes = sum(1 for inf in infos if inf and inf.get("is_temporal"))
        majority_temporal = temporal_votes > (len(vals) * 0.5)
        if not majority_temporal and col_name_is_timey(col):
            majority_temporal = True
        if majority_temporal:
            units = [inf.get("unit") for inf in infos if inf and inf.get("is_temporal")]
            epochs = [inf.get("epoch") for inf in infos if inf and inf.get("is_temporal")]
            unit = "yearmonth" if "yearmonth" in units else None
            epoch = None
            sec_count = epochs.count("seconds"); ms_count = epochs.count("millis")
            if ms_count or sec_count:
                epoch = "millis" if ms_count >= sec_count else "seconds"
            time_cols_info.append((col, {"unit": unit, "epoch": epoch}))
        else:
            num_votes = sum(1 for v in vals if is_numeric(v))
            if num_votes > (len(vals) * 0.6): numeric_cols.append(col)
            else: categorical_cols.append(col)

    if time_cols_info:
        x_col, x_meta = time_cols_info[0]
        if numeric_cols:
            hints["suggested"] = {"type": "line", "x": x_col, "y": numeric_cols[0], "x_timeUnit": x_meta.get("unit"), "x_epoch": x_meta.get("epoch")}
        else:
            hints["suggested"] = {"type": "line_count", "x": x_col, "x_timeUnit": x_meta.get("unit"), "x_epoch": x_meta.get("epoch")}
    elif categorical_cols:
        if numeric_cols:
            hints["suggested"] = {"type": "bar", "x": categorical_cols[0], "y": numeric_cols[0]}
        else:
            hints["suggested"] = {"type": "bar_count", "x": categorical_cols[0]}
    elif numeric_cols:
        hints["suggested"] = {"type": "hist", "x": numeric_cols[0]}
    else:
        hints["suggested"] = {"type": "table"}

    return hints

def to_vegalite_spec(columns: List[str], data: List[Dict[str, Any]], hints: Dict[str, Any]) -> Dict[str, Any]:
    s = hints.get("suggested", {})
    t = s.get("type", "table")
    spec: Dict[str, Any] = {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "data": {"values": data[:200]}
    }
    x_field = s.get("x")
    transforms: List[Dict[str, Any]] = []
    time_unit = s.get("x_timeUnit")
    x_epoch = s.get("x_epoch")

    if x_field and x_epoch in ("seconds","millis"):
        if x_epoch == "seconds":
            transforms.append({"calculate": f"toDate(datum['{x_field}'] * 1000)", "as": "_x"})
        else:
            transforms.append({"calculate": f"toDate(datum['{x_field}'])", "as": "_x"})
        x_encoded_field = "_x"
    else:
        x_encoded_field = x_field if x_field else (columns[0] if columns else "value")

    if t == "line":
        if transforms: spec["transform"] = transforms
        spec.update({
            "mark": {"type": "line", "point": True},
            "encoding": {
                "x": {"field": x_encoded_field, "type": "temporal", **({"timeUnit": time_unit} if time_unit else {}), "sort": "ascending"},
                "y": {"field": s["y"], "type": "quantitative"},
                "tooltip": [{"field": x_encoded_field}, {"field": s["y"]}]
            }
        })
    elif t == "line_count":
        groupby_field = x_encoded_field
        tx = transforms[:]
        tx.append({"aggregate": [{"op": "count", "as": "count"}], "groupby": [groupby_field]})
        spec.update({
            "transform": tx,
            "mark": {"type": "line", "point": True},
            "encoding": {
                "x": {"field": groupby_field, "type": "temporal", **({"timeUnit": time_unit} if time_unit else {}), "sort": "ascending"},
                "y": {"field": "count", "type": "quantitative"},
                "tooltip": [{"field": groupby_field}, {"field": "count"}]
            }
        })
    elif t == "bar":
        spec.update({
            "mark": "bar",
            "encoding": {
                "x": {"field": s["x"], "type": "nominal"},
                "y": {"field": s["y"], "type": "quantitative"},
                "tooltip": [{"field": s["x"]}, {"field": s["y"]}]
            }
        })
    elif t == "bar_count":
        spec.update({
            "transform": [
                {"aggregate": [{"op": "count", "as": "count"}], "groupby": [s["x"]]},
                {"sort": [{"field": "count", "order": "descending"}]}
            ],
            "mark": "bar",
            "encoding": {
                "x": {"field": s["x"], "type": "nominal"},
                "y": {"field": "count", "type": "quantitative"},
                "tooltip": [{"field": s["x"]}, {"field": "count"}]
            }
        })
    elif t == "hist":
        spec.update({
            "mark": "bar",
            "encoding": {
                "x": {"field": s["x"], "type": "quantitative", "bin": {"maxbins": 20}},
                "y": {"aggregate": "count", "type": "quantitative"},
                "tooltip": [{"field": s["x"]}]
            }
        })
    else:
        spec.update({
            "mark": "point",
            "encoding": {
                "x": {"field": columns[0], "type": "nominal"} if columns else {"field": "value", "type": "nominal"},
                "tooltip": [{"field": c} for c in columns[:5]]
            }
        })
    return spec

# ────────────────────────────────────────────────────────────────────────────────
# Validators
# ────────────────────────────────────────────────────────────────────────────────

class SqlValidator:
    BLOCKED = ["insert","update","delete","drop","alter","truncate","create","grant","revoke","merge"]
    DATETIME_TYPES = {"datetime","smalldatetime","date","time","timestamp"}

    @staticmethod
    def _avg_over_datetime_issues(sql: str, meta: Dict[str, TableMetadata]) -> List[str]:
        issues = []
        for tname, tmeta in (meta or {}).items():
            for col in tmeta.columns:
                pattern_br = rf"(?is)\bavg\s*\(\s*\[?{re.escape(col.name)}\]?\s*\)"
                pattern_qt = rf'(?is)\bavg\s*\(\s*"{re.escape(col.name)}"\s*\)'
                pattern_plain = rf"(?is)\bavg\s*\(\s*{re.escape(col.name)}\s*\)"
                if re.search(pattern_br, sql) or re.search(pattern_qt, sql) or re.search(pattern_plain, sql):
                    coltype = (col.type or "").lower()
                    if any(dt in coltype for dt in SqlValidator.DATETIME_TYPES):
                        issues.append(
                            f"AVG() used on date/time column '{col.name}' ({col.type}). "
                            f"Use DATEDIFF to convert to numeric duration, or CAST to numeric if it stores a duration."
                        )
        return issues

    @staticmethod
    def validate(sql: str) -> Dict[str, Any]:
        issues: List[str] = []
        if not sql or not sql.strip():
            issues.append("Empty SQL.")
            return {"valid": False, "issues": issues}
        original = sql
        normalized = original.strip().rstrip(";")
        lowered = normalized.lower()

        if "--" in lowered or "/*" in lowered:
            issues.append("SQL comments are not allowed.")
        if ";" in original.strip():
            issues.append("Multiple statements detected; semicolons are not allowed.")
        starts_with_select = lowered.startswith("select")
        starts_with_with = lowered.startswith("with")
        if not (starts_with_select or starts_with_with):
            issues.append("Only SELECT statements are allowed (CTEs with WITH ... are supported).")
        if starts_with_with:
            after_with = lowered[4:].strip()
            if "select" not in after_with:
                issues.append("WITH query must end with a SELECT statement.")
        for kw in SqlValidator.BLOCKED:
            if re.search(rf"\b{kw}\b", lowered):
                issues.append(f"Forbidden keyword detected: {kw}")

        try:
            meta = app_state.get("schema_metadata") or {}
            issues.extend(SqlValidator._avg_over_datetime_issues(sql, meta))
        except Exception:
            pass

        return {"valid": len(issues) == 0, "issues": issues}

# ────────────────────────────────────────────────────────────────────────────────
# AVG helpers, propose column replacements, fix mappers
# ────────────────────────────────────────────────────────────────────────────────

def _extract_avg_target_columns(sql: str) -> list:
    cols = []
    patterns = [
        r"(?is)\bavg\s*\(\s*\[?([A-Za-z0-9_]+)\]?\s*\)",
        r'(?is)\bavg\s*\(\s*"([^"]+)"\s*\)',
        r"(?is)\bavg\s*\(\s*'([^']+)'\s*\)",
        r"(?is)\bavg\s*\(\s*([A-Za-z0-9_.]+)\s*\)",
    ]
    for p in patterns:
        for m in re.finditer(p, sql):
            cols.append(m.group(1).split(".")[-1])
    return sorted(set(cols))

def _propose_column_replacements(sql: str, meta: dict, cache) -> list:
    suggestions = []
    avg_cols = _extract_avg_target_columns(sql)
    meanings = {}
    for tname, tmeta in (meta or {}).items():
        meanings[tname] = {}
        for c in tmeta.columns:
            cm = cache.get(tname, c.name)
            m = (cm.meaning if cm else "") or ""
            meanings[tname][c.name] = m.lower()
    for tname, tmeta in (meta or {}).items():
        table_cols = {c.name for c in tmeta.columns}
        intersect = [col for col in avg_cols if col in table_cols]
        if not intersect: continue
        candidates = []
        for c in tmeta.columns:
            m = meanings.get(tname, {}).get(c.name, "")
            if m in ("duration","numeric"):
                candidates.append(c.name)
        for bad_col in intersect:
            for cand in candidates:
                suggestions.append({
                    "table": tname,
                    "problem": f"AVG() on '{bad_col}' may be datetime/time-like",
                    "suggest": cand,
                    "confidence": 0.9,
                    "reason": "Prefer aggregating numeric/duration fields for averages",
                    "bad_col": bad_col
                })
    return suggestions

def _apply_fix_map(sql: str, fix_map: dict) -> str:
    fixed = sql
    if not fix_map: return fixed
    for src, dst in fix_map.items():
        fixed = fixed.replace(f'"{src}"', f'"{dst}"')
        fixed = fixed.replace(f'[{src}]', f'[{dst}]')
        fixed = re.sub(rf"(?<![A-Za-z0-9_]){re.escape(src)}(?![A-Za-z0-9_])", dst, fixed)
        fixed = re.sub(rf"\b([A-Za-z0-9_]+)\.{re.escape(src)}\b", rf"\g<1>.{dst}", fixed)
    return fixed

# NEW: apply DATA value fix map safely inside SQL
_re_str_lit = re.compile(r"'([^']*)'")
_re_num_token = re.compile(r"\b(-?\d+(?:\.\d+)?)\b")
def _apply_data_fix_map(sql: str, data_fix_map: dict) -> str:
    if not data_fix_map: return sql
    fixed = sql
    # replace string literals
    def _repl_str(m: re.Match) -> str:
        val = m.group(1)
        new = data_fix_map.get(val)
        if new is None: return m.group(0)
        return "'" + new.replace("'", "''") + "'"
    fixed = _re_str_lit.sub(_repl_str, fixed)
    # replace numeric tokens (exact match)
    def _repl_num(m: re.Match) -> str:
        val = m.group(1)
        return data_fix_map.get(val, val)
    fixed = _re_num_token.sub(_repl_num, fixed)
    return fixed

# ────────────────────────────────────────────────────────────────────────────────
# LLM Generator (Ollama) for SQL and Summarization
# ────────────────────────────────────────────────────────────────────────────────

def _default_model_id_from_catalog(catalog_path: str = "llm_provider_onprem.json") -> Optional[str]:
    """Read default model id from catalog file. Returns None if unavailable."""
    path = os.path.abspath(catalog_path) if not os.path.isabs(catalog_path) else catalog_path
    if not os.path.isfile(path):
        path = os.path.join(os.getcwd(), catalog_path)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        models = data.get("models") if isinstance(data, dict) else None
        if not models or not isinstance(models, list):
            return None
        for m in models:
            if isinstance(m, dict) and m.get("is_default"):
                mid = m.get("id")
                if mid:
                    return mid
        first = models[0]
        return first.get("id") if isinstance(first, dict) else (first if isinstance(first, str) else None)
    except Exception:
        return None


def _catalog_id_to_factory_id(catalog_model_id: str, catalog_path: str = "llm_provider_onprem.json") -> Optional[str]:
    """Convert catalog model id to factory id (provider:model_name). Use when in-process factory uses different id format."""
    path = os.path.abspath(catalog_path) if not os.path.isabs(catalog_path) else catalog_path
    if not os.path.isfile(path):
        path = os.path.join(os.getcwd(), catalog_path)
    if not os.path.isfile(path):
        _root = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
        path = os.path.join(_root, os.path.basename(catalog_path))
    if not path or not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for m in (data.get("models") or []):
            if isinstance(m, dict) and m.get("id") == catalog_model_id:
                provider = (m.get("provider") or "ollama").strip()
                model_name = (m.get("model_name") or "").strip()
                if model_name:
                    return f"{provider}:{model_name}"
                return None
        return None
    except Exception:
        return None


def _normalize_identifier(name: str) -> str:
    if name is None: return ""
    s = name.strip().strip("'\"[]`").replace("-", "_")
    s = re.sub(r"\s+", "_", s)
    return s.lower()

def _build_schema_whitelist(meta: Dict[str, TableMetadata]) -> Dict[str, set]:
    whitelist: Dict[str, set] = {}
    for tname, tmeta in (meta or {}).items():
        tkey = _normalize_identifier(tname)
        whitelist[tkey] = set(_normalize_identifier(c.name) for c in tmeta.columns)
    return whitelist

def _example_is_valid(ex: Dict[str, Any], whitelist: Dict[str, set]) -> bool:
    tables = [_normalize_identifier(t) for t in (ex.get("tables") or [])]
    columns = [_normalize_identifier(c) for c in (ex.get("columns") or [])]
    if not tables or not columns: return False
    if not all(t in whitelist for t in tables): return False
    for col in columns:
        if not any(col in whitelist[t] for t in tables): return False
    return True

def _filter_valid_examples(examples: List[Dict[str, Any]], whitelist: Dict[str, set]) -> List[Dict[str, Any]]:
    return [ex for ex in (examples or []) if _example_is_valid(ex, whitelist)]

def _chart_type_from_columns(tables: List[str], columns: List[str], meta: Dict[str, TableMetadata], cache: ColumnMeaningCache) -> Optional[str]:
    meanings: List[str] = []
    for t in (tables or []):
        tm = meta.get(t)
        if not tm: continue
        for c in tm.columns:
            if c.name in (columns or []):
                cm = cache.get(t, c.name)
                m = (cm.meaning if cm else "").lower()
                n = c.name.lower()
                if not m:
                    if any(k in n for k in ["date","time","timestamp","month","created","updated"]):
                        m = "timestamp"
                    elif any(k in n for k in ["status","type","category","segment","region"]):
                        m = "category"
                    elif re.search(r"\b(id|uuid)\b", n):
                        m = "id"
                    else:
                        m = "numeric" if ("int" in c.type.lower() or "num" in c.type.lower() or "float" in c.type.lower() or "decimal" in c.type.lower()) else "text"
                meanings.append(m)
    has_time = any(m in ("timestamp","datetime","date","string_month") for m in meanings)
    has_numeric = any(m in ("numeric","duration") for m in meanings)
    has_category = any(m in ("category","code") for m in meanings)
    if has_time and has_numeric: return "line"
    if has_category: return "bar"
    if has_numeric and not has_time and not has_category: return "hist"
    return None

class NLPQueryGenerator:
    def __init__(
        self,
        llm_api_base_url: str = "",
        llm_model_id: str = "",
        timeout_seconds: int = 120,
        llm_service: Any = None,
        llm: Any = None,
    ):
        self.schema_context: Optional[str] = None
        self.llm_api_base_url = llm_api_base_url.rstrip("/")
        self.llm_model_id = llm_model_id
        self.timeout_seconds = timeout_seconds
        self._resolved_model_id: Optional[str] = None  # resolved from catalog or /LLMs/models
        self.llm_service = llm_service  # when set (gateway), use .get_model().chat(); else HTTP /LLMs/chat
        self.llm = llm  # when set (e.g. ChatOpenAI/OpenRouter), use .invoke(); takes precedence over llm_service/HTTP

    def set_schema_context(self, ctx: str):
        self.schema_context = ctx

    def _build_compact_schema_context(self, question: str, max_tables: int = 6, max_cols_per_table: int = 35) -> str:
        q_toks = set(_tokens(question))
        meta: Dict[str, TableMetadata] = app_state.get("schema_metadata") or {}
        cache: ColumnMeaningCache = app_state.get("meaning_cache")
        if not meta or not cache:
            return self.schema_context or ""

        def score_table(tname: str, tmeta: TableMetadata) -> int:
            score = len(set(_tokens(tname)) & q_toks)
            for c in tmeta.columns:
                score += len(set(_tokens(c.name)) & q_toks)
                cm = cache.get(tname, c.name)
                if cm:
                    score += len(set(_tokens(cm.meaning or "")) & q_toks)
                    score += len(set(_tokens(cm.description or "")) & q_toks)
            score += len(tmeta.relationships)
            return score

        ranked = sorted(meta.items(), key=lambda kv: score_table(kv[0], kv[1]), reverse=True)
        selected = ranked[:max_tables] if ranked else []
        lines: List[str] = []
        lines.append(f"(dialect: {CURRENT_DIALECT}) Relevant schema excerpt:")
        for tname, tmeta in selected:
            lines.append(f"\nTable: {tname}")
            rel = "; ".join(tmeta.relationships) if tmeta.relationships else "None"
            lines.append(f"Relationships: {rel}")
            lines.append("Columns:")
            cols_scored: List[Tuple[int, TableColumn, Optional[ColumnMeaning]]] = []
            for c in tmeta.columns:
                base = len(q_toks & set(_tokens(c.name)))
                cm = cache.get(tname, c.name)
                if cm:
                    base += len(q_toks & set(_tokens(cm.description or "")))
                    base += len(q_toks & set(_tokens(cm.meaning or "")))
                if c.primary_key or c.foreign_key:
                    base += 1
                cols_scored.append((base, c, cm))
            cols_scored.sort(key=lambda x: x[0], reverse=True)
            for _, c, cm in cols_scored[:max_cols_per_table]:
                gloss_desc = get_column_glossary_description(c.name)
                desc = gloss_desc or (cm.description if cm else None) or ""
                meaning = (cm.meaning if cm else "") or ""
                ex_list = (cm.examples if cm else [])[:SCHEMA_EXAMPLES_DISPLAY]
                ex_txt = f" — {desc}" if desc else ""
                ex_txt += f" e.g., {', '.join(ex_list)}" if ex_list else ""
                lines.append(f"- {c.name} [{meaning}]{ex_txt}")
        return "\n".join(lines)

    def _build_prompt(self, question: str, max_results: Optional[int], language: str, fix_map: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        sqlite_rules = (
            "You are an expert data analyst and SQL generator.\n"
            "Rules:\n"
            "- SQLite: Do NOT use INTERVAL; use date('now','-N units')/datetime('now','-N units').\n"
            "- Use ASCII operators only: >=, <=, <>, =.\n"
            "- For month bucketing, use strftime('%Y-%m-01', column).\n"
            "- Use CURRENT_TIMESTAMP instead of NOW().\n"
            "- Avoid DATE_TRUNC; use strftime instead.\n"
            "- Prefer single SELECT statements; avoid unnecessary CTEs unless they improve clarity.\n"
        )
        plugin_rules = app_state.get("llm_dialect_rules") or sqlite_rules

        fix_rules_block = ""
        if fix_map:
            fix_lines = []
            for src, dst in fix_map.items():
                fix_lines.append(f"- Column correction: replace any use of `{src}` with `{dst}`.")
                fix_lines.append(f"- STRICT: Do NOT use `{src}`; prefer `{dst}`.")
            fix_rules_block = "\nColumn corrections:\n" + "\n".join(fix_lines) + "\n"

        # NEW: Data corrections passed via app_state during build
        data_fix_map: Dict[str, str] = app_state.get("data_fix_map_for_prompt") or {}
        data_rules_block = ""
        if data_fix_map:
            dlines = []
            for src, dst in data_fix_map.items():
                dlines.append(f"- Data correction: if a literal equals '{src}', use '{dst}' in filters and comparisons.")
            data_rules_block = "\nData corrections:\n" + "\n".join(dlines) + "\n"

        system = plugin_rules + (
            "\nValidation:\n"
            "- Ensure the query references only tables/columns present in the schema excerpt.\n"
            "- Do not use comments; do not end with a semicolon.\n"
            "- If ambiguous, choose the most central table and state assumptions in 'warnings'.\n"
        ) + fix_rules_block + data_rules_block

        compact_ctx = self._build_compact_schema_context(question)
        header_ctx = f"{compact_ctx}\n\n" if compact_ctx else (f"{self.schema_context or ''}\n\n")
        user = (
            f"{header_ctx}"
            f"Question: {question}\n"
            f"Language: {language}\n"
            f"Max results: {max_results if max_results is not None else 'None'}\n\n"
            f"When choosing columns, use the descriptions and examples (up to {SCHEMA_EXAMPLES_DISPLAY} unique non-null values per column) to select the correct column. "
            "Match the user question to the right column (e.g. SALESSTAGE vs SALESSTAGECODE, PRIMARY_REGION vs SEC_REGION). Generate SQL and suggest visualization columns accordingly.\n\n"
            "Return JSON only, like:\n"
            "{\n"
            '  "sql": "SELECT ...",\n'
            '  "explanation": "why this query answers the question",\n'
            '  "confidence": 0.76,\n'
            '  "warnings": ["assumption X"],\n'
            '  "optimizations": ["index suggestion or filter tweak"]\n'
            "}\n"
        )
        return {"system": system, "user": user}

    def _resolve_model_id(self) -> str:
        """Resolve model_id: instance llm_model_id > catalog default > /LLMs/models."""
        if self._resolved_model_id:
            return self._resolved_model_id
        if self.llm_model_id:
            self._resolved_model_id = self.llm_model_id
            return self._resolved_model_id
        # Use default from llm_provider_onprem.json (Convobi onprem catalog)
        from_catalog = _default_model_id_from_catalog()
        if from_catalog:
            self._resolved_model_id = from_catalog
            LOG.info("Resolved LLM model_id from catalog: %s", from_catalog)
            return self._resolved_model_id
        url = f"{self.llm_api_base_url}/LLMs/models"
        try:
            resp = requests.get(url, timeout=min(30, self.timeout_seconds))
            if resp.status_code != 200:
                raise RuntimeError(f"LLM API error {resp.status_code}: {resp.text}")
            data = resp.json()
            models = data if isinstance(data, list) else data.get("models", data) or []
            if not models:
                raise RuntimeError("No models returned from /LLMs/models")
            first = models[0] if isinstance(models[0], dict) else {"id": models[0]}
            model_id = first.get("id") or first.get("model_id") or (first if isinstance(first, str) else None)
            if not model_id:
                raise RuntimeError("Could not get model_id from /LLMs/models")
            self._resolved_model_id = model_id
            LOG.info("Resolved LLM model_id: %s", model_id)
            return self._resolved_model_id
        except (requests.ConnectionError, requests.Timeout) as e:
            LOG.warning("LLM API unreachable at %s (connection refused or timeout).", self.llm_api_base_url)
            raise RuntimeError(
                f"LLM API unreachable at {self.llm_api_base_url}. Configure CONVOBI_LLM or a fallback LLM."
            ) from e
        except Exception as e:
            LOG.exception("Failed to resolve model_id from LLM API")
            raise RuntimeError(f"LLM model resolution failed: {e}") from e

    def _ollama_generate(self, prompt: Dict[str, str]) -> str:
        """Call LLM: module-level llm (e.g. ChatOpenAI/OpenRouter) > in-process llm_service > POST /LLMs/chat."""
        model_id = self._resolve_model_id()
        messages = [
            {"role": "system", "content": prompt.get("system", "")},
            {"role": "user", "content": prompt.get("user", "")},
        ]
        if self.llm is not None:
            try:
                from langchain_core.messages import SystemMessage, HumanMessage
                lc_messages = [
                    SystemMessage(content=prompt.get("system", "")),
                    HumanMessage(content=prompt.get("user", "")),
                ]
                response = self.llm.invoke(lc_messages)
                return (getattr(response, "content", None) or str(response) or "").strip()
            except Exception as e:
                LOG.warning("Module-level LLM invoke failed: %s; falling back to llm_service/HTTP", e)
        if self.llm_service is not None:
            tried_ids = [model_id]
            factory_id = _catalog_id_to_factory_id(model_id)
            if factory_id and factory_id not in tried_ids:
                tried_ids.append(factory_id)
            for try_id in tried_ids:
                try:
                    model = self.llm_service.get_model(try_id, temperature=0.2, max_tokens=4096)
                    response = model.chat(messages)
                    if response:
                        if try_id != model_id:
                            LOG.info("In-process LLM using factory id: %s (catalog: %s)", try_id, model_id)
                        return response.strip()
                except Exception as e:
                    LOG.debug("In-process get_model(%s) failed: %s", try_id, e)
                    continue
            try:
                available = self.llm_service.list_models()
                models = available.get("models") if isinstance(available, dict) else (available if isinstance(available, list) else [])
                for m in (models or []):
                    aid = m.get("id") if isinstance(m, dict) else m
                    if not aid or "embed" in str(aid).lower():
                        continue
                    try:
                        model = self.llm_service.get_model(aid, temperature=0.2, max_tokens=4096)
                        response = model.chat(messages)
                        if response:
                            LOG.info("In-process LLM using first available model: %s (requested: %s)", aid, model_id)
                            return response.strip()
                    except Exception:
                        continue
            except Exception as e:
                LOG.debug("list_models fallback failed: %s", e)
            LOG.warning("In-process LLM call failed for %s (tried %s), falling back to HTTP", model_id, tried_ids)
        url = f"{self.llm_api_base_url}/LLMs/chat"
        payload = {
            "model_id": model_id,
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": 4096,
        }
        LOG.info("Calling LLM API model_id=%s at %s", model_id, url)
        resp = requests.post(url, json=payload, timeout=self.timeout_seconds)
        if resp.status_code != 200:
            raise RuntimeError(f"LLM API error {resp.status_code}: {resp.text}")
        data = resp.json()
        return (data.get("response") or "").strip()

    def _extract_json(self, raw: str) -> Dict[str, Any]:
        try:
            return json.loads(raw)
        except Exception:
            pass
        m = re.search(r"`{3}json\s*(\{.*?\})\s*`{3}", raw, re.DOTALL | re.IGNORECASE)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                pass
        m2 = re.search(r"\{.*\}", raw, re.DOTALL)
        if m2:
            try:
                return json.loads(m2.group(0))
            except Exception:
                pass
        return {"sql": "", "explanation": "Model returned non-JSON output.", "confidence": 0.4,
                "warnings": ["Non-JSON output; fallback used"], "optimizations": []}

    def _extract_sql_fallback(self, raw: str) -> str:
        m = re.search(r"`{3}sql\s*(.+?)\r?\n`{3}", raw, re.DOTALL | re.IGNORECASE)
        if m: return m.group(1)
        m2 = re.search(r"(?is)\bSELECT\b.*?(?=$\n```?\n\s*\{)", raw)
        if m2: return m2.group(0).strip()
        return raw

    def _inject_last_n_days_filter(self, sql: str, question: str, time_col_candidates: List[str]) -> str:
        q = question.lower()
        m = re.search(r"last\s+(\d+)\s+days", q)
        if not m: return sql
        n = int(m.group(1))
        if re.search(r"(?is)\bdateadd\b|\binterval\b|date\('now'|"
                     r"\bCURRENT_DATE\b|\bGETDATE\(\)", sql):
            return sql
        meta = app_state.get("schema_metadata") or {}
        time_cols = set()
        for tname, tmeta in meta.items():
            for c in tmeta.columns:
                name = c.name.lower()
                if any(k in name for k in ["date","time","timestamp","created","updated","closed","resolved","registered"]):
                    time_cols.add(c.name)
        chosen = None
        for cand in time_col_candidates or []:
            if cand in time_cols:
                chosen = cand; break
        if not chosen and len(time_cols) == 1:
            chosen = list(time_cols)[0]
        if not chosen: return sql

        q_chosen = _quote_ident(chosen)
        if CURRENT_DIALECT == "mssql":
            pred = f"{q_chosen} >= DATEADD(DAY, -{n}, CAST(GETDATE() AS DATE))"
        elif CURRENT_DIALECT == "sqlite":
            pred = f"date({q_chosen}) >= date('now', '-{n} days')"
        elif CURRENT_DIALECT == "postgres":
            pred = f"{q_chosen} >= CURRENT_DATE - INTERVAL '{n} days'"
        elif CURRENT_DIALECT == "mysql":
            pred = f"{q_chosen} >= DATE_SUB(CURDATE(), INTERVAL {n} DAY)"
        elif CURRENT_DIALECT == "snowflake":
            pred = f"{q_chosen} >= DATEADD(day, -{n}, CURRENT_DATE)"
        elif CURRENT_DIALECT == "bigquery":
            pred = f"{q_chosen} >= DATE_SUB(CURRENT_DATE(), INTERVAL {n} DAY)"
        else:
            pred = f"{q_chosen} >= CURRENT_DATE - INTERVAL '{n} days'"

        if re.search(r"(?is)\bwhere\b", sql):
            return re.sub(r"(?is)\bwhere\b", f"WHERE {pred} AND ", sql, count=1)
        mpos = re.search(r"(?is)\bgroup\s+by\b|\border\s+by\b", sql)
        if mpos:
            idx = mpos.start()
            return f"{sql[:idx]} WHERE {pred} {sql[idx:]}"
        return f"{sql} WHERE {pred}"

    def _to_ascii_operators(self, sql: str) -> str:
        replacements = {"≥": ">=", "≤": "<=", "≠": "<>", "–": "-", "—": "-", "’": "'", "‘": "'", "“": '"', "”": '"'}
        for k, v in replacements.items():
            sql = sql.replace(k, v)
        return sql

    def _dedupe_limit(self, sql: str) -> str:
        matches = re.findall(r"(?is)\bLIMIT\s+\d+(?:\s+OFFSET\s+\d+)?", sql)
        if len(matches) <= 1: return sql
        sql_wo = re.sub(r"(?is)\bLIMIT\s+\d+(?:\s+OFFSET\s+\d+)?", " ", sql).strip()
        return f"{sql_wo} {matches[-1]}".strip()

    @staticmethod
    def _normalize_sql_for_sqlite(sql: str) -> str:
        sql = re.sub(r"\bNOW\s*\(\s*\)\b", "CURRENT_TIMESTAMP", sql, flags=re.IGNORECASE)
        return sql

    @staticmethod
    def _normalize_sql_for_mssql(sql: str) -> str:
        lowered = sql.lower()
        m_lo = re.search(r"(?is)\blimit\s+(?P<n>\d+)\s+offset\s+(?P<m>\d+)\b", lowered)
        if m_lo:
            n = int(m_lo.group("n")); m = int(m_lo.group("m"))
            sql_wo = re.sub(r"(?is)\blimit\s+\d+\s+offset\s+\d+\b", "", sql).strip()
            if re.search(r"(?is)\border\s+by\b", sql_wo) is None:
                sql_wo = f"{sql_wo} ORDER BY (SELECT NULL)"
            sql = f"{sql_wo} OFFSET {m} ROWS FETCH NEXT {n} ROWS ONLY"
            return sql
        m_l = re.search(r"(?is)\blimit\s+(?P<n>\d+)\b", lowered)
        if m_l:
            n = int(m_l.group("n"))
            def insert_top(match: re.Match) -> str:
                select_kw = match.group(0)
                return f"{select_kw} TOP {n} "
            sql = re.sub(r"(?is)\bselect\s+(distinct\s+)?", insert_top, sql, count=1)
            sql = re.sub(r"(?is)\blimit\s+\d+\b", "", sql).strip()
            return sql
        return sql

    def _normalize_sql(self, sql: str) -> str:
        if not sql: return sql
        sql = self._to_ascii_operators(sql)
        sql = re.sub(r";\s*", " ", sql).strip().rstrip(";").strip()
        sql = self._dedupe_limit(sql)
        if CURRENT_DIALECT == "sqlite":
            sql = self._normalize_sql_for_sqlite(sql)
        elif CURRENT_DIALECT == "mssql":
            sql = self._normalize_sql_for_mssql(sql)
        return sql

    def generate_query(self, question: str, language: str = "sql", max_results: Optional[int] = None,
                       fix_map: Optional[Dict[str, str]] = None) -> QueryResult:
        if not self.schema_context and not app_state.get("schema_metadata"):
            raise RuntimeError("Schema context not set for NLPQueryGenerator.")
        if language != "sql":
            raise RuntimeError("Only SQL is supported")
        prompt = self._build_prompt(question, max_results, language, fix_map=fix_map)
        raw = self._ollama_generate(prompt)
        parsed = self._extract_json(raw)
        sql = (parsed.get("sql") or "").strip()
        explanation = parsed.get("explanation")
        confidence = parsed.get("confidence", 0.7)
        warnings = parsed.get("warnings", []) or []
        optimizations = parsed.get("optimizations", []) or []

        if not sql and raw:
            sql = self._extract_sql_fallback(raw).strip()

        if max_results is not None:
            if sql and " limit " not in sql.lower() and CURRENT_DIALECT != "mssql":
                sql = f"{sql} LIMIT {max_results}"

        sql = self._normalize_sql(sql)
        sql = self._inject_last_n_days_filter(sql, question, time_col_candidates=[])

        return QueryResult(sql=sql, explanation=explanation, confidence=confidence, warnings=warnings, optimizations=optimizations)

    def suggest_questions(self, max_examples: int = 30, include_tables: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        meta: Dict[str, TableMetadata] = app_state.get("schema_metadata") or {}
        cache: ColumnMeaningCache = app_state.get("meaning_cache")
        tables = [t for t in meta.keys() if (not include_tables or t in include_tables)]
        examples: List[Dict[str, Any]] = []
        for t in tables:
            tm = meta.get(t); 
            if not tm: continue
            time_cols = []; num_cols = []; cat_cols = []
            for c in tm.columns:
                cm = cache.get(t, c.name) if cache else None
                m = (cm.meaning if cm else "") or ""
                lname = c.name.lower()
                if m in ("timestamp","datetime","date") or any(k in lname for k in ["date","time","timestamp","created","updated","month"]):
                    time_cols.append(c.name)
                elif m in ("numeric","duration") or any(k in c.type.lower() for k in ["int","float","num","dec"]):
                    num_cols.append(c.name)
                elif m in ("category","code"):
                    cat_cols.append(c.name)
            if time_cols and num_cols:
                examples.append({"question": f"Trend of {num_cols[0]} over time ({t})", "category": "Time Series", "tables": [t], "columns": [time_cols[0], num_cols[0]], "viz": "line"})
            if cat_cols and num_cols:
                examples.append({"question": f"{num_cols[0]} by {cat_cols[0]} ({t})", "category": "Comparison", "tables": [t], "columns": [cat_cols[0], num_cols[0]], "viz": "bar"})
            if num_cols:
                examples.append({"question": f"Distribution of {num_cols[0]} ({t})", "category": "Distribution", "tables": [t], "columns": [num_cols[0]], "viz": "hist"})
            if len(examples) >= max_examples:
                break
        return examples[:max_examples]

# ────────────────────────────────────────────────────────────────────────────────
# Summary-only intent (for CSV/Excel file chat: no SQL, no charts)
# ────────────────────────────────────────────────────────────────────────────────

def _is_summary_only_question(question: str) -> bool:
    """
    Return True if the user is asking only for a summary/brief/overview of the data.
    For such questions on uploaded CSV/Excel files we skip SQL generation and charts.
    """
    if not question or not question.strip():
        return False
    q = question.strip().lower()
    summary_phrases = (
        "summary", "summarize", "summarise", "brief", "overview", "describe the data",
        "what's in this file", "what is in this file", "whats in this file",
        "give me a summary", "give me a brief", "high level", "high-level",
        "quick summary", "short summary", "brief overview", "data overview",
        "tell me about", "explain the data", "what does this data", "what do the data"
    )
    if q in ("summary", "brief", "overview", "summarize", "summarise"):
        return True
    for phrase in summary_phrases:
        if phrase in q and "sql" not in q and "query" not in q and "chart" not in q and "graph" not in q and "plot" not in q:
            return True
    return False


# ────────────────────────────────────────────────────────────────────────────────
# LLM Summarizer
# ────────────────────────────────────────────────────────────────────────────────

class LLMSummarizer:
    def __init__(self, ollama_generate_fn):
        self._ollama_generate = ollama_generate_fn

    def summarize_file_data(self, question: str, table_name: str, columns: List[str], data: List[Dict[str, Any]]) -> str:
        """
        Produce a narrative summary of file data only. No SQL, no charts.
        Used when user asks for summary/brief of CSV/Excel uploads.
        """
        system = (
            "You are a helpful analyst. Summarize the given tabular data in plain language. "
            "Respond only with a clear narrative summary in Markdown. Do NOT generate SQL, "
            "do NOT suggest charts or visualizations. Just describe what the data is about, "
            "key columns, and notable patterns or facts from the sample."
        )
        sample = data[:50]
        user = (
            f"Question: {question}\n\n"
            f"Table: {table_name}\n"
            f"Columns: {json.dumps(columns)}\n"
            f"Sample rows (up to 50): {json.dumps(sample)}\n\n"
            "Write a concise summary (2–4 short paragraphs or bullet points) covering: "
            "what this data represents, main columns, and any obvious patterns or insights. "
            "Use Markdown (## for headings, - for bullets). Do not include SQL or chart suggestions."
        )
        try:
            return self._ollama_generate({"system": system, "user": user}).strip()
        except Exception as e:
            return f"Failed to summarize: {e}"

    def summarize(self, question: str, columns: list, data: list, viz_hint: dict) -> str:
        system = "You are a helpful analyst. Summarize tabular data clearly and concisely. Always respond in valid Markdown."
        sample = data[:20]
        user = (
            "Summarize the following query result for a non-technical stakeholder. Use Markdown only.\n"
            f"Question: {question}\n"
            f"Columns: {json.dumps(columns)}\n"
            f"First rows (sample): {json.dumps(sample)}\n"
            f"Visualization hints: {json.dumps(viz_hint or {})}\n\n"
            "Format your response in Markdown with these sections:\n"
            "- ## Executive summary — one sentence\n"
            "- ## Key observations — bullet list\n"
            "- ## Notable trends / anomalies — short bullets or paragraph\n"
            "- ## Suggested next steps — bullet list\n"
            "Use ** for bold, - or * for bullets, ## for section headers. Keep it concise."
        )
        try:
            return self._ollama_generate({"system": system, "user": user}).strip()
        except Exception as e:
            return f"Failed to summarize: {e}"

# ────────────────────────────────────────────────────────────────────────────────
# Plugin System
# ────────────────────────────────────────────────────────────────────────────────

class DBPlugin:
    def name(self) -> str: raise NotImplementedError
    def connector_type(self) -> str: return "sql"
    def supports(self, connection_string: str) -> bool: raise NotImplementedError
    def dialect(self, connection_string: str) -> str: raise NotImplementedError
    def create_connector(self, connection_string: str) -> 'DataConnector': raise NotImplementedError
    def create_analyzer(self, connection_string: str, meaning_cache: ColumnMeaningCache) -> 'SchemaAnalyzer': raise NotImplementedError
    def llm_rules_for_dialect(self, dialect: str) -> str:
        return (
            "You are an expert data analyst and SQL generator.\n"
            "Rules:\n"
            "- Use ANSI SQL compatible with this dialect.\n"
            "- Use ASCII operators only: >=, <=, <>, =.\n"
        )

class SQLAlchemyRDBPlugin(DBPlugin):
    def name(self) -> str: return "sqlalchemy-rdb"
    def connector_type(self) -> str: return "sql"
    def supports(self, connection_string: str) -> bool:
        scheme = urlparse(connection_string).scheme.lower()
        return scheme in {"sqlite","postgresql","postgres","mysql","mariadb","mssql","mssql+pyodbc","oracle","oracle+cx_oracle","bigquery","snowflake"}

    def dialect(self, connection_string: str) -> str:
        scheme = urlparse(connection_string).scheme.lower()
        if scheme in {"postgresql","postgres"}: return "postgres"
        if scheme in {"mysql","mariadb"}: return "mysql"
        if scheme.startswith("mssql"): return "mssql"
        if scheme.startswith("oracle"): return "oracle"
        if scheme == "bigquery": return "bigquery"
        if scheme == "snowflake": return "snowflake"
        return "sqlite"

    def create_connector(self, connection_string: str) -> DataConnector:
        return SqlConnector(connection_string)

    def create_analyzer(self, connection_string: str, meaning_cache: ColumnMeaningCache) -> SchemaAnalyzer:
        return SchemaAnalyzer(connection_string, meaning_cache)

    def llm_rules_for_dialect(self, dialect: str) -> str:
        if dialect == "sqlite":
            return (
                "You are an expert data analyst and SQL generator.\nRules:\n"
                "- SQLite: Do NOT use INTERVAL; use date('now','-N units')/datetime('now','-N units').\n"
                "- Use ASCII operators only: >=, <=, <>, =.\n"
                "- For month bucketing, use strftime('%Y-%m-01', column).\n"
                "- Use CURRENT_TIMESTAMP instead of NOW().\n"
                "- Avoid DATE_TRUNC; use strftime instead.\n"
                "- Prefer single SELECT statements; avoid unnecessary CTEs unless they improve clarity.\n"
            )
        if dialect == "postgres":
            return (
                "You are an expert Postgres SQL generator.\nRules:\n"
                "- Use DATE_TRUNC('month', col); INTERVAL 'N unit' supported.\n"
                "- Prefer explicit casts when mixing types.\n"
                "- Use LIMIT N.\n"
                "- Use ASCII operators only: >=, <=, <>, =.\n"
            )
        if dialect == "mysql":
            return (
                "You are an expert MySQL SQL generator.\nRules:\n"
                "- Use DATE_FORMAT(col, '%Y-%m-01') for month bucketing.\n"
                "- Use NOW(); INTERVAL N UNIT supported.\n"
                "- Use LIMIT N.\n"
                "- Use ASCII operators only: >=, <=, <>, =.\n"
            )
        if dialect == "mssql":
            return (
                "You are an expert SQL Server generator.\nRules:\n"
                "- Use DATEADD and DATETRUNC/FORMAT/CONVERT as applicable.\n"
                "- Use TOP(N) instead of LIMIT.\n"
                "- For pagination, use ORDER BY ... OFFSET M ROWS FETCH NEXT N ROWS ONLY.\n"
                "- Always implement time filters when the question specifies ranges like 'last 30 days'.\n"
                "- Use ASCII operators only: >=, <=, <>, =.\n"
            )
        if dialect == "oracle":
            return (
                "You are an expert Oracle SQL generator.\nRules:\n"
                "- Use TRUNC(DATE), ADD_MONTHS/NUMTODSINTERVAL.\n"
                "- Use FETCH FIRST N ROWS ONLY.\n"
                "- Use ASCII operators only: >=, <=, <>, =.\n"
            )
        if dialect == "bigquery":
            return (
                "You are an expert BigQuery SQL generator.\nRules:\n"
                "- Use Standard SQL; DATE_TRUNC(), TIMESTAMP_TRUNC(), EXTRACT(), INTERVAL supported.\n"
                "- Use CURRENT_DATE() / CURRENT_TIMESTAMP(); DATE_SUB for day ranges.\n"
                "- Use LIMIT N.\n"
                "- Prefer SAFE_CAST / SAFE_DIVIDE to avoid runtime errors.\n"
                "- Use ASCII operators only: >=, <=, <>, =.\n"
            )
        if dialect == "snowflake":
            return (
                "You are an expert Snowflake SQL generator.\nRules:\n"
                "- Use DATE_TRUNC('month', col), DATEADD(unit, n, col).\n"
                "- Use LIMIT N; QUALIFY is available for window filtering.\n"
                "- Prefer explicit casts (TO_DATE/TO_TIMESTAMP) when needed.\n"
                "- Use ASCII operators only: >=, <=, <>, =.\n"
            )
        return super().llm_rules_for_dialect(dialect)

class PluginManager:
    def __init__(self):
        self._plugins: List[DBPlugin] = [SQLAlchemyRDBPlugin()]
        self._active: Optional[DBPlugin] = None
        self._active_connection_string: Optional[str] = None

    def list_plugins(self) -> List[Dict[str, Any]]:
        return [{"name": p.name(), "type": p.connector_type()} for p in self._plugins]

    def choose(self, connection_string: str, connector_hint: Optional[str] = None) -> DBPlugin:
        cs = (connection_string or "").strip()
        for p in self._plugins:
            if connector_hint and p.connector_type() != connector_hint:
                continue
            if p.supports(cs):
                return p
        scheme = urlparse(cs).scheme.lower()
        if scheme in {"sqlite","postgresql","postgres","mysql","mariadb","mssql","mssql+pyodbc","oracle","oracle+cx_oracle","bigquery","snowflake"}:
            return SQLAlchemyRDBPlugin()
        raise RuntimeError(f"No plugin found that supports connection string scheme '{scheme}'")

    def activate(self, plugin: DBPlugin, connection_string: str):
        self._active = plugin
        self._active_connection_string = connection_string

    def active(self) -> Tuple[Optional[DBPlugin], Optional[str]]:
        return self._active, self._active_connection_string


def get_connection_string_for_file_id(file_id: str) -> Optional[str]:
    """
    Return the SQLite connection_string for an uploaded file by file_id (from POST /convobi/files/upload).
    Returns None if file_id not found in registry.
    """
    if not file_id or not file_id.strip():
        return None
    db_path = app_state.get("file_id_registry") and app_state["file_id_registry"].get(file_id)
    if not db_path:
        return None
    abs_path = os.path.abspath(db_path)
    return "sqlite:///" + abs_path.replace("\\", "/")


def get_saved_connection_string(connection_id: str) -> Optional[str]:
    """
    Return the connection_string for a saved DB connection by id (from connector's saved_db_connections).
    Returns None if not found or on read error.
    """
    if not connection_id or not connection_id.strip():
        return None
    try:
        # Same file the connector uses: apps/connector/saved_db_connections.json
        connector_dir = Path(__file__).resolve().parent.parent / "connector"
        path = connector_dir / "saved_db_connections.json"
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        connections = data.get("connections") or []
        for c in connections:
            if c.get("id") == connection_id:
                return c.get("connection_string")
        return None
    except Exception:
        return None


def activate_plugin(connection_string: str, connector_hint: str = "sql", connection_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Activate a DB plugin for Convobi. Only wires the connection; does NOT run schema analysis
    or build schema doc (so POST /connectors/plugin/load returns quickly). Schema is loaded
    lazily on first use (e.g. GET/POST /convobi/schema or POST /convobi/query).
    """
    global CURRENT_DIALECT, CURRENT_CONNECTOR
    pm = app_state.get("plugin_manager")
    if not pm:
        raise RuntimeError("PluginManager not initialized")
    plugin = pm.choose(connection_string, connector_hint=connector_hint)
    pm.activate(plugin, connection_string)
    CURRENT_DIALECT = plugin.dialect(connection_string)
    CURRENT_CONNECTOR = "sql"
    meaning_cache: ColumnMeaningCache = app_state["meaning_cache"] or ColumnMeaningCache(ttl_seconds=3600)
    analyzer = plugin.create_analyzer(connection_string, meaning_cache)
    connector = plugin.create_connector(connection_string)
    app_state["schema_analyzer"] = analyzer
    app_state["sql_connector"] = connector
    # Defer schema analysis and schema doc: do not run here so plugin load is fast
    app_state["schema_metadata"] = None
    app_state["schema_doc"] = None
    app_state["query_generator"].set_schema_context("")
    app_state["llm_dialect_rules"] = plugin.llm_rules_for_dialect(CURRENT_DIALECT)
    app_state["result_cache"].clear()
    db_id = compute_db_id(connection_string, CURRENT_DIALECT)
    app_state["db_id"] = db_id
    app_state["active_connection_id"] = connection_id
    if not app_state.get("schema_doc_store"):
        app_state["schema_doc_store"] = SchemaDocCsvStore(base_dir=SCHEMA_DOC_DIR, ttl_seconds=SCHEMA_DOC_TTL)
    out = {"ok": True, "plugin": plugin.name(), "connector": CURRENT_CONNECTOR, "dialect": CURRENT_DIALECT, "tables_loaded": 0}
    if connection_id is not None:
        out["connection_id"] = connection_id
    return out


def ensure_schema_loaded() -> None:
    """
    If schema has not been loaded yet for the active connection, run schema analysis
    and build schema doc (skip LLM narrative). Idempotent.
    """
    if app_state.get("schema_metadata") and len(app_state["schema_metadata"]) > 0:
        return
    analyzer = app_state.get("schema_analyzer")
    if not analyzer:
        return
    schema_metadata = analyzer.analyze_schema()
    app_state["schema_metadata"] = schema_metadata
    schema_context = analyzer.generate_llm_context(schema_metadata, verbose=True) if schema_metadata else ""
    app_state["query_generator"].set_schema_context(schema_context)
    db_id = app_state.get("db_id")
    if not app_state.get("schema_doc_store"):
        app_state["schema_doc_store"] = SchemaDocCsvStore(base_dir=SCHEMA_DOC_DIR, ttl_seconds=SCHEMA_DOC_TTL)
    try:
        documenter = SchemaDocumenter(app_state["meaning_cache"], app_state["query_generator"]._ollama_generate, CURRENT_DIALECT)
        doc = documenter.build(schema_metadata, skip_llm_narrative=True)
        fingerprint = SchemaDocumenter._schema_fingerprint(schema_metadata)
        app_state["schema_doc"] = doc
        if db_id:
            app_state["schema_doc_store"].write(db_id, fingerprint, doc)
    except Exception as e:
        LOG.warning("Schema doc build during lazy load failed: %s", e)


# ────────────────────────────────────────────────────────────────────────────────
# Schema Documentation Models & Store
# ────────────────────────────────────────────────────────────────────────────────

class ColumnDoc(BaseModel):
    name: str
    type: str
    nullable: bool
    primary_key: bool = False
    foreign_key: Optional[str] = None
    meaning: Optional[str] = None
    description: Optional[str] = None
    confidence: Optional[float] = None
    source: Optional[str] = None
    examples: List[str] = Field(default_factory=list)

class TableDoc(BaseModel):
    name: str
    row_count: Optional[int]
    relationships: List[str] = Field(default_factory=list)
    columns: List[ColumnDoc] = Field(default_factory=list)
    overview: Optional[str] = None

class SchemaDoc(BaseModel):
    generated_at: str
    dialect: str
    total_tables: int
    tables: Dict[str, TableDoc] = Field(default_factory=dict)
    narrative: Optional[str] = None

def compute_db_id(connection_string: str, dialect: str) -> str:
    cs = (connection_string or "").strip()
    p = urlparse(cs)
    scheme = (p.scheme or dialect or "db").lower()
    if scheme == "sqlite":
        name = os.path.basename(p.path or "") or "memory"
    else:
        host = (p.hostname or "localhost").lower()
        dbname = (p.path or "").strip("/").split("/")[-1] or "default"
        name = f"{host}_{dbname}"
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", name)
    h = hashlib.sha1(cs.encode("utf-8")).hexdigest()[:8]
    return f"{scheme}_{safe}_{h}"

class SchemaDocCsvStore:
    def __init__(self, base_dir: str = SCHEMA_DOC_DIR, ttl_seconds: Optional[int] = SCHEMA_DOC_TTL):
        self.base_dir = base_dir
        self.ttl_seconds = ttl_seconds
        self._cache: Dict[Tuple[str, str], Tuple[float, SchemaDoc]] = {}
        os.makedirs(self.base_dir, exist_ok=True)

    def _db_dir(self, db_id: str) -> str:
        return os.path.join(self.base_dir, db_id)

    def write(self, db_id: str, fingerprint: str, doc: SchemaDoc):
        ddir = self._db_dir(db_id); os.makedirs(ddir, exist_ok=True)
        meta_path = os.path.join(ddir, "meta.csv")
        with open(meta_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["key", "value"])
            w.writerow(["dialect", doc.dialect])
            w.writerow(["generated_at", doc.generated_at])
            w.writerow(["total_tables", str(doc.total_tables)])
            w.writerow(["fingerprint", fingerprint])
            w.writerow(["narrative", doc.narrative or ""])
        tables_path = os.path.join(ddir, "tables.csv")
        with open(tables_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["name", "row_count", "overview", "relationships"])
            for tn, td in doc.tables.items():
                rels = " \n ".join(td.relationships or [])
                w.writerow([tn, td.row_count if td.row_count is not None else "", td.overview or "", rels])
        cols_path = os.path.join(ddir, "columns.csv")
        with open(cols_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["table_name", "name", "type", "nullable", "primary_key", "foreign_key", "meaning", "description", "confidence", "source", "examples"])
            for tn, td in doc.tables.items():
                for c in td.columns:
                    ex = " \n ".join(c.examples or [])
                    w.writerow([tn, c.name, c.type, str(bool(c.nullable)), str(bool(c.primary_key)), c.foreign_key or "", c.meaning or "", c.description or "", ("" if c.confidence is None else str(c.confidence)), c.source or "", ex])
        self._cache[(db_id, fingerprint)] = (time.time(), doc)

    def read(self, db_id: str, fingerprint: str) -> Optional[SchemaDoc]:
        key = (db_id, fingerprint)
        item = self._cache.get(key)
        if item:
            ts, doc = item
            if not self.ttl_seconds or (time.time() - ts) <= self.ttl_seconds:
                return doc
            self._cache.pop(key, None)

        ddir = self._db_dir(db_id)
        meta_path = os.path.join(ddir, "meta.csv")
        tables_path = os.path.join(ddir, "tables.csv")
        cols_path = os.path.join(ddir, "columns.csv")
        if not (os.path.exists(meta_path) and os.path.exists(tables_path) and os.path.exists(cols_path)):
            return None

        meta: Dict[str, str] = {}
        try:
            with open(meta_path, "r", newline="", encoding="utf-8") as f:
                r = csv.reader(f); header = next(r, None)
                for row in r:
                    if len(row) >= 2:
                        meta[row[0]] = row[1]
        except Exception:
            return None

        if meta.get("fingerprint") != fingerprint:
            return None
        dialect = meta.get("dialect", "sqlite")
        generated_at = meta.get("generated_at", datetime.now().isoformat())
        try:
            total_tables = int(meta.get("total_tables") or "0")
        except Exception:
            total_tables = 0
        narrative = meta.get("narrative") or None

        tables: Dict[str, TableDoc] = {}
        try:
            with open(tables_path, "r", newline="", encoding="utf-8") as f:
                r = csv.DictReader(f)
                for row in r:
                    tn = row.get("name") or ""
                    rc_val = row.get("row_count")
                    try: rc = int(rc_val) if rc_val not in (None, "") else None
                    except Exception: rc = None
                    rels_raw = row.get("relationships") or ""
                    rels = [s.strip() for s in rels_raw.split("\n") if s.strip()] if rels_raw else []
                    overview = row.get("overview") or None
                    tables[tn] = TableDoc(name=tn, row_count=rc, relationships=rels, columns=[], overview=overview)
        except Exception:
            return None

        try:
            with open(cols_path, "r", newline="", encoding="utf-8") as f:
                r = csv.DictReader(f)
                for row in r:
                    tn = row.get("table_name") or ""
                    if tn not in tables:
                        tables[tn] = TableDoc(name=tn, row_count=None, relationships=[], columns=[], overview=None)
                    ex_raw = row.get("examples") or ""
                    ex_list = [s.strip() for s in ex_raw.split("\n") if s.strip()] if ex_raw else []
                    conf_val = row.get("confidence")
                    try: conf = float(conf_val) if conf_val not in (None, "") else None
                    except Exception: conf = None
                    col = ColumnDoc(
                        name=row.get("name") or "",
                        type=row.get("type") or "",
                        nullable=(row.get("nullable") or "False").lower() == "true",
                        primary_key=(row.get("primary_key") or "False").lower() == "true",
                        foreign_key=row.get("foreign_key") or None,
                        meaning=row.get("meaning") or None,
                        description=row.get("description") or None,
                        confidence=conf,
                        source=row.get("source") or None,
                        examples=ex_list
                    )
                    tables[tn].columns.append(col)
        except Exception:
            return None

        doc = SchemaDoc(generated_at=generated_at, dialect=dialect, total_tables=len(tables), tables=tables, narrative=narrative)
        self._cache[key] = (time.time(), doc)
        return doc

    def clear(self, db_id: Optional[str] = None):
        if db_id:
            ddir = self._db_dir(db_id)
            if os.path.isdir(ddir):
                try:
                    for fn in ["meta.csv","tables.csv","columns.csv"]:
                        fp = os.path.join(ddir, fn)
                        if os.path.exists(fp): os.remove(fp)
                    try: os.rmdir(ddir)
                    except Exception: pass
                except Exception as e:
                    LOG.warning(f"Failed to clear SchemaDoc CSVs for {db_id}: {e}")
            for k in list(self._cache.keys()):
                if k[0] == db_id: self._cache.pop(k, None)
        else:
            for entry in os.listdir(self.base_dir):
                ddir = os.path.join(self.base_dir, entry)
                if os.path.isdir(ddir):
                    try:
                        for fn in ["meta.csv","tables.csv","columns.csv"]:
                            fp = os.path.join(ddir, fn)
                            if os.path.exists(fp): os.remove(fp)
                        try: os.rmdir(ddir)
                        except Exception: pass
                    except Exception as e:
                        LOG.warning(f"Failed to clear dir {ddir}: {e}")
            self._cache.clear()

# ────────────────────────────────────────────────────────────────────────────────
# Schema Documenter (compact)
# ────────────────────────────────────────────────────────────────────────────────

def _infer_table_purpose(tmeta: TableMetadata) -> str:
    """
    Heuristically infer a table's business purpose using:
      - table name patterns,
      - column names/semantics,
      - foreign-key structure (link/bridge tables),
      - presence of lifecycle/status, timestamps, numeric measures, text fields.
    """
    name = (tmeta.name or "").lower()
    colnames = [c.name.lower() for c in (tmeta.columns or [])]
    fk_cols = [c for c in tmeta.columns if c.foreign_key]

    # Strong domain cues from table names
    if any(k in name for k in ["user", "account", "profile"]):
        return "Stores user/account master records with identifying attributes and metadata."
    if any(k in name for k in ["order", "invoice", "payment", "transaction"]):
        return "Stores transactional financial/order records and related attributes."
    if any(k in name for k in ["product", "item", "sku", "inventory", "catalog"]):
        return "Stores product/catalog records including attributes and categorization."
    if any(k in name for k in ["event", "log", "activity", "audit"]):
        return "Captures time-stamped events or activity logs with contextual metadata."
    if any(k in name for k in ["ticket", "case", "incident", "issue", "request"]):
        return "Tracks cases/tickets with lifecycle state, ownership, and history."

    # Link/bridge tables: many-to-many
    if len(fk_cols) >= 2:
        return "Relationship mapping table linking two or more entities (many-to-many)."

    # Column-level signals
    has_status = any("status" in c for c in colnames)
    has_user   = any(c in ("user_id", "owner_id", "created_by") or "user" in c for c in colnames)
    has_time   = any(k in c for c in colnames for k in ["created", "updated", "timestamp", "date"])
    has_amount = any(k in c for c in colnames for k in ["amount", "price", "cost", "total"])
    has_text   = any(k in c for c in colnames for k in ["description", "comment", "message", "notes", "details"])
    has_metric = any(k in c for c in colnames for k in ["score", "metric", "value", "count", "rate"])

    parts: List[str] = []
    if has_status: parts.append("tracks lifecycle/status")
    if has_user:   parts.append("associated with users/ownership")
    if has_time:   parts.append("maintains timestamped history")
    if has_amount: parts.append("stores numeric/financial measures")
    if has_text:   parts.append("contains descriptive text")
    if has_metric: parts.append("holds quantitative metrics")

    if parts:
        return "Stores records that " + ", ".join(parts) + "."
    return "Stores domain records with identifiers and attributes."


def _infer_database_domain(table_names: List[str]) -> str:
    """Infer high-level domain/purpose from table names for schema summary."""
    if not table_names:
        return "business data"
    names_lower = " ".join((tn or "").lower() for tn in table_names)
    if any(k in names_lower for k in ["opportunity", "opp_", "opp ", "pursuit", "client", "sales", "crm", "proposal", "ifi_account"]):
        return "Sales and CRM (opportunities, clients, proposals)"
    if any(k in names_lower for k in ["ticket", "incident", "case", "severity", "workgroup", "resolution", "sla", "caller", "closure_code"]):
        return "Support and ticketing (incidents, cases, SLAs)"
    if any(k in names_lower for k in ["order", "product", "invoice", "customer", "cart", "payment", "line_item"]):
        return "Commerce (orders, products, customers)"
    if any(k in names_lower for k in ["user", "account", "profile", "member"]):
        return "User and account management"
    if any(k in names_lower for k in ["event", "log", "audit", "activity", "history"]):
        return "Events and auditing"
    return "Business data and analytics"


def get_schema_summary_purpose(
    schema_metadata: Optional[Dict[str, TableMetadata]],
    doc: Optional["SchemaDoc"],
) -> str:
    """
    Return a markdown paragraph explaining the purpose of the plugged-in database.
    Used in schema summary response so users see what the DB is for.
    """
    if doc and getattr(doc, "narrative", None) and str(doc.narrative).strip():
        return str(doc.narrative).strip()
    meta = schema_metadata or {}
    table_names = sorted(meta.keys())
    if not table_names:
        return "No schema loaded."
    domain = _infer_database_domain(table_names)
    lines: List[str] = []
    lines.append(f"This database supports **{domain}**.")
    if len(table_names) == 1:
        tn = table_names[0]
        tmeta = meta.get(tn)
        purpose = _infer_table_purpose(tmeta) if tmeta else "Stores domain records."
        lines.append(f"It contains one table, **{tn}**, which {purpose.lower().rstrip('.')}.")
    else:
        lines.append(f"It contains **{len(table_names)} tables**:")
        for tn in table_names[:15]:
            tmeta = meta.get(tn)
            if tmeta:
                purpose = _infer_table_purpose(tmeta)
                short = purpose.split(".")[0].strip() + "." if "." in purpose else purpose
                rc = f" ({tmeta.row_count} rows)" if tmeta.row_count is not None else ""
                lines.append(f"- **{tn}**{rc}: {short}")
            elif doc and getattr(doc, "tables", None) and tn in doc.tables:
                td = doc.tables[tn]
                overview = (td.overview or "").split("\n")[0].strip()
                rc = f" ({td.row_count} rows)" if td.row_count is not None else ""
                lines.append(f"- **{tn}**{rc}: {overview[:120]}{'...' if len(overview) > 120 else ''}")
            else:
                lines.append(f"- **{tn}**")
        if len(table_names) > 15:
            lines.append(f"- _... and {len(table_names) - 15} more tables._")
    return "\n\n".join(lines)


class SchemaDocumenter:
    def __init__(self, meaning_cache: ColumnMeaningCache, ollama_generate_fn, dialect: str):
        self.meaning_cache = meaning_cache
        self._ollama_generate = ollama_generate_fn
        self.dialect = dialect

    @staticmethod
    def _schema_fingerprint(schema_metadata: Dict[str, TableMetadata]) -> str:
        parts = []
        for tname in sorted(schema_metadata.keys()):
            t = schema_metadata[tname]
            parts.append(f"T:{tname}:{t.row_count or 'NA'}")
            for c in sorted(t.columns, key=lambda x: x.name):
                parts.append(f"C:{c.name}:{c.type}:{int(c.nullable)}:{int(c.primary_key)}:{c.foreign_key or ''}")
            for r in t.relationships:
                parts.append(f"R:{r}")
        return " \n ".join(parts)

    def _table_overview_with_purpose(self, t: TableMetadata) -> str:
        purpose = _infer_table_purpose(t)
        pk_cols = [c.name for c in t.columns if c.primary_key]
        fk_cols = [c.name for c in t.columns if c.foreign_key]

        rel_targets = []
        for r in (t.relationships or []):
            m = re.search(r"\->\s*([A-Za-z0-9_]+)\(", r)
            if m:
                rel_targets.append(m.group(1))
        rel_targets = sorted(set(rel_targets))

        rc = f"{t.row_count} rows" if t.row_count is not None else "rows: unknown"
        lines = [
            f"Purpose: {purpose}",
            f"Rows: {rc}",
            f"Primary keys: {', '.join(pk_cols) if pk_cols else '(none declared)'}",
            f"Foreign keys: {', '.join(fk_cols) if fk_cols else '(none)'}",
            f"Links to: {', '.join(rel_targets) if rel_targets else '(none)'}"
        ]
        return "\n".join(lines)

    def build(self, schema_metadata: Dict[str, TableMetadata], skip_llm_narrative: bool = False) -> SchemaDoc:
        tables: Dict[str, TableDoc] = {}
        for tname, tmeta in schema_metadata.items():
            cols: List[ColumnDoc] = []
            for c in tmeta.columns:
                cm = self.meaning_cache.get(tname, c.name)
                desc = cm.description if cm and cm.description else None
                if c.foreign_key and desc and ("foreign key" not in desc.lower() and "references" not in desc.lower()):
                    desc = desc.rstrip(".") + f" (foreign key to `{c.foreign_key}`)."
                cols.append(ColumnDoc(
                    name=c.name, type=c.type, nullable=c.nullable, primary_key=c.primary_key,
                    foreign_key=c.foreign_key, meaning=(cm.meaning if cm else None),
                    description=desc, confidence=(cm.confidence if cm else None),
                    source=(cm.source if cm else None), examples=(cm.examples if cm else [])
                ))
            tables[tname] = TableDoc(
                name=tmeta.name, row_count=tmeta.row_count,
                relationships=tmeta.relationships, columns=cols,
                overview=self._table_overview_with_purpose(tmeta)
            )
        doc = SchemaDoc(generated_at=datetime.now().isoformat(), dialect=self.dialect, total_tables=len(tables), tables=tables, narrative=None)
        # LLM narrative optional (skip during startup so we don't call LLM API before server is listening)
        if skip_llm_narrative:
            return doc
        try:
            compact_lines = [f"{td.name}: {td.overview}" for _, td in tables.items()]
            schema_outline = "\n".join(compact_lines)
            system = (
                "You are a technical writer generating a human-readable database overview. "
                "Explain the purpose of the dataset in plain language, summarizing what each table represents "
                "and how they relate. Avoid generic phrases. Be concise, factual, and domain-neutral."
            )
            user = (
                "Write a short overall summary (2–3 sentences) of the database based on these table purposes:\n\n"


                f"{schema_outline}\n\n"

                "Focus on: entities involved, how they link, the kind of information stored, and typical analytics it enables."
            )
            resp = app_state["query_generator"]._ollama_generate({"system": system, "user": user}).strip()
            if resp:
                doc.narrative = " ".join([ln.strip() for ln in resp.split("\n") if ln.strip()])[:800]
        except Exception as e:
            LOG.warning(f"LLM narrative generation failed; heuristics kept. Error: {e}")
        return doc

# ────────────────────────────────────────────────────────────────────────────────
# Global App State / Lifespan
# ────────────────────────────────────────────────────────────────────────────────

app_state: Dict[str, Any] = {
    "schema_analyzer": None,
    "query_generator": None,
    "sql_connector": None,
    "result_cache": None,
    "meaning_cache": None,
    "schema_metadata": None,
    "schema_doc_store": None,
    "schema_doc": None,
    "db_id": None,
    "conversation_history": [],
    "plugin_manager": None,
    "llm_dialect_rules": None,
    "fix_memory": None,  # columns
    "data_fix_memory": None,  # NEW: data (values)
    "data_fix_map_for_prompt": None,  # transient during prompt build
    "file_id_registry": {},  # file_id -> db_path for uploaded tabular files (POST /files/upload)
    "active_file_id": None,  # set when DB is activated via file_id (from routes)
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    global CURRENT_DIALECT, CURRENT_CONNECTOR
    connection_string = os.getenv("DATABASE_URL")

    meaning_cache = ColumnMeaningCache(ttl_seconds=3600)
    app_state["meaning_cache"] = meaning_cache
    app_state["plugin_manager"] = PluginManager()
    app_state["fix_memory"] = FixMemoryStore(path=LEARNING_FIXES_PATH, sim_threshold=0.6)
    app_state["data_fix_memory"] = DataFixMemoryStore(path=LEARNING_DATA_FIXES_PATH, sim_threshold=0.6)

    llm_service = None
    try:
        from echolib.di import container
        llm_service = container.resolve("llm.service")
        LOG.info("Convobi using in-process LLM service (get_model().chat())")
    except Exception:
        LOG.info("Convobi using module-level CONVOBI_LLM only (no HTTP fallback).")

    qg = NLPQueryGenerator(
        llm_api_base_url="",
        llm_model_id="",
        timeout_seconds=120,
        llm_service=llm_service,
        llm=CONVOBI_LLM,
    )
    app_state["query_generator"] = qg
    app_state["result_cache"] = ResultCache(ttl_seconds=600)
    app_state["schema_doc_store"] = SchemaDocCsvStore(base_dir=SCHEMA_DOC_DIR, ttl_seconds=SCHEMA_DOC_TTL)
    app_state["db_id"] = None

    if not connection_string:
        CURRENT_DIALECT = "sqlite"; CURRENT_CONNECTOR = None
        app_state["schema_analyzer"] = None
        app_state["sql_connector"] = None
        app_state["schema_metadata"] = {}
        app_state["llm_dialect_rules"] = SQLAlchemyRDBPlugin().llm_rules_for_dialect(CURRENT_DIALECT)
        LOG.info("✓ Application initialized without datasource.")
        yield
        LOG.info("✓ Application shutdown complete")
        return

    try:
        plugin = app_state["plugin_manager"].choose(connection_string, connector_hint="sql")
    except Exception as e:
        LOG.error(f"Plugin selection failed: {e}. Falling back to SQLAlchemyRDBPlugin.")
        plugin = SQLAlchemyRDBPlugin()

    app_state["plugin_manager"].activate(plugin, connection_string)
    CURRENT_DIALECT = plugin.dialect(connection_string)
    CURRENT_CONNECTOR = "sql"

    analyzer = plugin.create_analyzer(connection_string, meaning_cache)
    connector = plugin.create_connector(connection_string)

    app_state["schema_analyzer"] = analyzer
    app_state["sql_connector"] = connector

    schema_metadata = analyzer.analyze_schema()
    app_state["schema_metadata"] = schema_metadata

    schema_context = analyzer.generate_llm_context(schema_metadata, verbose=True) if schema_metadata else ""
    qg.set_schema_context(schema_context)
    app_state["llm_dialect_rules"] = plugin.llm_rules_for_dialect(CURRENT_DIALECT)

    active_plugin, active_cs = app_state["plugin_manager"].active()
    db_id = compute_db_id(active_cs or connection_string, CURRENT_DIALECT)
    app_state["db_id"] = db_id

    try:
        fingerprint = SchemaDocumenter._schema_fingerprint(schema_metadata)
        doc = app_state["schema_doc_store"].read(db_id, fingerprint)
        if doc:
            app_state["schema_doc"] = doc
            LOG.info("✓ Loaded SchemaDoc from CSV cache (db_id=%s)", db_id)
        else:
            documenter = SchemaDocumenter(app_state["meaning_cache"], qg._ollama_generate, CURRENT_DIALECT)
            doc = documenter.build(schema_metadata, skip_llm_narrative=True)  # skip during startup so LLM API is not called before server is listening
            app_state["schema_doc"] = doc
            app_state["schema_doc_store"].write(db_id, fingerprint, doc)
            LOG.info("✓ Generated and cached SchemaDoc to CSV (db_id=%s)", db_id)
    except Exception as e:
        LOG.warning(f"SchemaDoc generation failed: {e}")

    LOG.info("✓ Application initialized (plugin=%s, connector=%s, dialect=%s)", plugin.name(), CURRENT_CONNECTOR, CURRENT_DIALECT)
    LOG.info(f"✓ Loaded schema with {len(schema_metadata)} tables")

    yield

    if app_state.get("schema_analyzer"): app_state["schema_analyzer"].close()
    if app_state.get("sql_connector"): app_state["sql_connector"].close()
    LOG.info("✓ Application shutdown complete")

# ────────────────────────────────────────────────────────────────────────────────
# FastAPI App
# ────────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="NLP Database Interface API (On-Prem Ollama, SQL-only, Domain-neutral)",
    description="Natural language to SQL with learned per-question fixes.",
    version="1.7.0",
    lifespan=lifespan
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# Import routes
from .routes import router
app.include_router(router)
