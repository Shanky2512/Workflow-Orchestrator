"""
Universal Database Connector Service
=====================================
Supports connecting to any SQL database — both **public** (direct) and
**private** (SSH tunnel through a bastion/jump host).

Supported databases:
  PostgreSQL · MySQL · MariaDB · Microsoft SQL Server · Oracle
  SQLite · Google BigQuery · Snowflake · ClickHouse · Amazon Redshift

Frontend form fields (matches UI):
  Database Type  ·  Host  ·  Port  ·  Database Name
  Username  ·  Password  ·  [Server IP shown for firewall rules]
"""

import json
import logging
import os
import re
import socket
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote_plus, urlparse

import httpx
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
_SAVED_CONNECTIONS_FILE = (
    Path(__file__).resolve().parent / "saved_db_connections.json"
)

# ---------------------------------------------------------------------------
# Supported database types  (drives the "Database Type" dropdown)
# ---------------------------------------------------------------------------
DB_TYPES: Dict[str, Dict[str, Any]] = {
    "postgresql": {
        "label": "PostgreSQL",
        "default_port": 5432,
        "driver": "postgresql+psycopg2",
        "drivers_available": [
            "postgresql+psycopg2",
            "postgresql+asyncpg",
            "postgresql+pg8000",
        ],
        "placeholder_host": "db.example.com",
    },
    "mysql": {
        "label": "MySQL",
        "default_port": 3306,
        "driver": "mysql+pymysql",
        "drivers_available": ["mysql+pymysql", "mysql+mysqlconnector"],
        "placeholder_host": "db.example.com",
    },
    "mariadb": {
        "label": "MariaDB",
        "default_port": 3306,
        "driver": "mariadb+pymysql",
        "drivers_available": ["mariadb+pymysql"],
        "placeholder_host": "db.example.com",
    },
    "mssql": {
        "label": "Microsoft SQL Server",
        "default_port": 1433,
        "driver": "mssql+pymssql",
        "drivers_available": ["mssql+pymssql", "mssql+pyodbc"],
        "placeholder_host": "sqlserver.example.com",
    },
    "oracle": {
        "label": "Oracle",
        "default_port": 1521,
        "driver": "oracle+oracledb",
        "drivers_available": ["oracle+oracledb", "oracle+cx_oracle"],
        "placeholder_host": "oracle.example.com",
    },
    "sqlite": {
        "label": "SQLite",
        "default_port": None,
        "driver": "sqlite",
        "drivers_available": ["sqlite"],
        "placeholder_host": "(file path)",
    },
    "bigquery": {
        "label": "Google BigQuery",
        "default_port": None,
        "driver": "bigquery",
        "drivers_available": ["bigquery"],
        "placeholder_host": "project-id",
    },
    "snowflake": {
        "label": "Snowflake",
        "default_port": 443,
        "driver": "snowflake",
        "drivers_available": ["snowflake"],
        "placeholder_host": "account.snowflakecomputing.com",
    },
    "clickhouse": {
        "label": "ClickHouse",
        "default_port": 9000,
        "driver": "clickhouse+native",
        "drivers_available": ["clickhouse+native", "clickhouse+http"],
        "placeholder_host": "localhost",
    },
    "redshift": {
        "label": "Amazon Redshift",
        "default_port": 5439,
        "driver": "redshift+psycopg2",
        "drivers_available": ["redshift+psycopg2"],
        "placeholder_host": "cluster.region.redshift.amazonaws.com",
    },
    "hive": {
        "label": "Apache Hive",
        "default_port": 10000,
        "driver": "hive",
        "drivers_available": ["hive"],
        "placeholder_host": "localhost",
    },
    "duckdb": {
        "label": "DuckDB",
        "default_port": None,
        "driver": "duckdb",
        "drivers_available": ["duckdb"],
        "placeholder_host": "(file path or :memory:)",
    },
}

# ---------------------------------------------------------------------------
# Pydantic request / config models
# ---------------------------------------------------------------------------


class SSHTunnelConfig(BaseModel):
    """SSH tunnel configuration for accessing private / firewalled databases."""

    ssh_host: str = Field(..., description="SSH bastion / jump host address")
    ssh_port: int = Field(22, description="SSH port")
    ssh_username: str = Field(..., description="SSH username")
    ssh_password: Optional[str] = Field(
        None, description="SSH password (if not using key)"
    )
    ssh_private_key: Optional[str] = Field(
        None, description="SSH private key content (PEM format)"
    )
    ssh_private_key_passphrase: Optional[str] = Field(
        None, description="Passphrase for the private key"
    )


class DBConnectionConfig(BaseModel):
    """
    Database connection configuration — matches the frontend form exactly.

    **Mode 1 – Structured parameters (preferred, matches UI form):**
        db_type, host, port, database_name, username, password

    **Mode 2 – Raw connection string:**
        connection_string  (full SQLAlchemy DSN)

    For **private databases** behind firewalls add ``ssh_tunnel``.
    """

    # ---- Mode 1: form fields (matches UI) ----
    db_type: Optional[str] = Field(
        None,
        description=(
            "Database type key from the dropdown: "
            "postgresql, mysql, mariadb, mssql, oracle, sqlite, "
            "bigquery, snowflake, clickhouse, redshift, hive, duckdb"
        ),
    )
    host: Optional[str] = Field(None, description="Database host / IP address")
    port: Optional[int] = Field(
        None, description="Database port (auto-filled per db_type if omitted)"
    )
    database_name: Optional[str] = Field(
        None,
        description="Database name (or file path for SQLite, project-id for BigQuery)",
    )
    username: Optional[str] = Field(None, description="Database username")
    password: Optional[str] = Field(None, description="Database password")

    # ---- Snowflake-specific ----
    account_url: Optional[str] = Field(
        None,
        description=(
            "Snowflake account URL, e.g. 'your-account.snowflakecomputing.com'. "
            "Used instead of host/port for Snowflake connections."
        ),
    )

    # ---- SQL Server ODBC-specific ----
    odbc_connection_string: Optional[str] = Field(
        None,
        description=(
            "Raw ODBC connection string for SQL Server, e.g. "
            "'Driver={ODBC Driver 17 for SQL Server};Server=server;Database=database;UID=user;PWD=password;'"
        ),
    )

    # ---- File-based databases (SQLite, DuckDB) ----
    database_file_path: Optional[str] = Field(
        None,
        description=(
            "File path or URL for file-based databases. "
            "For SQLite: path to .sqlite, .db, or .sqlite3 file. "
            "For DuckDB: file:///path/to/database.db or :memory:"
        ),
    )

    # ---- Mode 2: raw DSN ----
    connection_string: Optional[str] = Field(
        None,
        description=(
            "Full SQLAlchemy connection string, e.g. "
            "postgresql://user:pass@host:5432/mydb. "
            "When provided, the form fields above are ignored."
        ),
    )

    # ---- Advanced ----
    driver: Optional[str] = Field(
        None,
        description=(
            "Database driver. Can be a SQLAlchemy dialect (e.g. 'mssql+pyodbc') "
            "OR an ODBC driver name (e.g. 'ODBC Driver 18 for SQL Server'). "
            "ODBC names are automatically placed into the query string. "
            "Auto-selected if omitted."
        ),
    )
    extra_params: Optional[Dict[str, str]] = Field(
        None,
        description=(
            "Additional query-string parameters, e.g. "
            "{'charset': 'utf8mb4', 'driver': 'ODBC Driver 18 for SQL Server'}"
        ),
    )
    connect_args: Optional[Dict[str, Any]] = Field(
        None,
        description="Dict passed to create_engine(connect_args=...). Useful for SSL certificates etc.",
    )
    pool_size: int = Field(5, description="Connection pool size")
    pool_timeout: int = Field(30, description="Pool timeout in seconds")

    # ---- SSH tunnel for private databases ----
    ssh_tunnel: Optional[SSHTunnelConfig] = Field(
        None,
        description="SSH tunnel config for private databases behind firewalls",
    )

    # ---- Display / save ----
    name: Optional[str] = Field(
        None, description="Display name for the saved connection"
    )


class DBQueryRequest(BaseModel):
    """Execute a query on a saved connection."""

    query: str = Field(..., description="SQL query to execute")
    limit: int = Field(100, description="Max rows to return")
    timeout: int = Field(30, description="Query timeout in seconds")


# ---------------------------------------------------------------------------
# SSH Tunnel helper
# ---------------------------------------------------------------------------


class _SSHTunnelWrapper:
    """Manages an SSH tunnel lifecycle."""

    def __init__(
        self, config: SSHTunnelConfig, remote_host: str, remote_port: int
    ):
        self.config = config
        self.remote_host = remote_host
        self.remote_port = remote_port
        self._tunnel = None

    def start(self) -> Tuple[str, int]:
        """Start the tunnel. Returns (local_host, local_port)."""
        try:
            from sshtunnel import SSHTunnelForwarder
        except ImportError:
            raise ImportError(
                "The 'sshtunnel' package is required for SSH tunneling. "
                "Install it with:  pip install sshtunnel"
            )

        kwargs: Dict[str, Any] = {
            "ssh_address_or_host": (self.config.ssh_host, self.config.ssh_port),
            "ssh_username": self.config.ssh_username,
            "remote_bind_address": (self.remote_host, self.remote_port),
        }

        if self.config.ssh_private_key:
            import io
            import paramiko

            pkey = paramiko.RSAKey.from_private_key(
                io.StringIO(self.config.ssh_private_key),
                password=self.config.ssh_private_key_passphrase,
            )
            kwargs["ssh_pkey"] = pkey
        elif self.config.ssh_password:
            kwargs["ssh_password"] = self.config.ssh_password
        else:
            raise ValueError(
                "SSH tunnel requires either ssh_password or ssh_private_key"
            )

        self._tunnel = SSHTunnelForwarder(**kwargs)
        self._tunnel.start()
        local_host = "127.0.0.1"
        local_port = self._tunnel.local_bind_port
        logger.info(
            "SSH tunnel established: %s:%s -> %s:%s",
            local_host,
            local_port,
            self.remote_host,
            self.remote_port,
        )
        return local_host, local_port

    def stop(self):
        if self._tunnel:
            self._tunnel.stop()
            self._tunnel = None
            logger.info("SSH tunnel closed")


# ---------------------------------------------------------------------------
# Connection-string builder
# ---------------------------------------------------------------------------


def build_connection_string(config: DBConnectionConfig) -> str:
    """
    Build a SQLAlchemy DSN from the structured form fields.
    If ``config.connection_string`` is already provided, return it as-is.
    """
    if config.connection_string:
        return config.connection_string.strip()

    if not config.db_type:
        raise ValueError("Either 'connection_string' or 'db_type' must be provided")

    db_type = config.db_type.lower()
    if db_type not in DB_TYPES:
        raise ValueError(
            f"Unsupported db_type '{db_type}'. "
            f"Supported: {', '.join(sorted(DB_TYPES.keys()))}"
        )

    info = DB_TYPES[db_type]

    # --- SQLite (file-based) ---
    if db_type == "sqlite":
        path = config.database_file_path or config.database_name or ":memory:"
        return f"sqlite:///{path}"

    # --- DuckDB (file-based) ---
    if db_type == "duckdb":
        path = config.database_file_path or config.database_name or ":memory:"
        # Strip file:// prefix if present – DuckDB SQLAlchemy driver expects raw path
        if path.startswith("file:///"):
            path = path[len("file:///"):]
        elif path.startswith("file://"):
            path = path[len("file://"):]
        return f"duckdb:///{path}"

    # --- BigQuery ---
    if db_type == "bigquery":
        project = config.database_name or ""
        cs = f"bigquery://{project}"
        if config.extra_params:
            qs = "&".join(f"{k}={v}" for k, v in config.extra_params.items())
            cs += f"?{qs}"
        return cs

    # --- SQL Server with raw ODBC connection string ---
    if db_type == "mssql" and config.odbc_connection_string:
        odbc_cs = config.odbc_connection_string.strip()
        encoded = quote_plus(odbc_cs)
        return f"mssql+pyodbc:///?odbc_connect={encoded}"

    # --- Snowflake (uses account_url instead of host:port) ---
    if db_type == "snowflake":
        account = config.account_url or config.host or ""
        # Remove trailing .snowflakecomputing.com if present
        if account.endswith(".snowflakecomputing.com"):
            account = account[: -len(".snowflakecomputing.com")]
        user = quote_plus(config.username or "")
        pwd = quote_plus(config.password or "")
        db = config.database_name or ""
        extra = dict(config.extra_params or {})
        if user and pwd:
            cs = f"snowflake://{user}:{pwd}@{account}/{db}"
        elif user:
            cs = f"snowflake://{user}@{account}/{db}"
        else:
            cs = f"snowflake://{account}/{db}"
        if extra:
            qs = "&".join(
                f"{k}={quote_plus(str(v))}" for k, v in extra.items()
            )
            cs += f"?{qs}"
        return cs

    # --- Resolve the driver ---
    # The `driver` field can be either:
    #   (a) a SQLAlchemy dialect+driver like "mssql+pyodbc"   →  used as URL scheme
    #   (b) an ODBC driver name like "ODBC Driver 18 for SQL Server"
    #       →  use the appropriate SQLAlchemy scheme and move the ODBC name into ?driver=
    #   (c) omitted  →  use the default for this db_type
    extra = dict(config.extra_params or {})
    raw_driver = (config.driver or "").strip()

    if raw_driver and ("+" not in raw_driver and "://" not in raw_driver):
        # Looks like an ODBC / native driver name, not a SQLAlchemy dialect.
        # e.g. "ODBC Driver 18 for SQL Server", "FreeTDS", "MySQL ODBC 8.0"
        # → push it into the query-string as ?driver=... and pick the right
        #   SQLAlchemy dialect automatically.
        extra["driver"] = raw_driver
        if db_type == "mssql":
            sa_driver = "mssql+pyodbc"
        elif db_type in ("mysql", "mariadb"):
            sa_driver = info["driver"]  # mysql+pymysql, etc.
        else:
            sa_driver = info["driver"]
    elif raw_driver:
        sa_driver = raw_driver          # already a proper SQLAlchemy driver
    else:
        sa_driver = info["driver"]      # default for this db_type

    host = config.host or "localhost"
    port = config.port or info["default_port"]
    user = quote_plus(config.username or "")
    pwd = quote_plus(config.password or "")
    db = config.database_name or ""

    if user and pwd:
        cs = f"{sa_driver}://{user}:{pwd}@{host}:{port}/{db}"
    elif user:
        cs = f"{sa_driver}://{user}@{host}:{port}/{db}"
    else:
        cs = f"{sa_driver}://{host}:{port}/{db}"

    if extra:
        qs = "&".join(
            f"{k}={quote_plus(str(v))}" for k, v in extra.items()
        )
        cs += f"?{qs}"

    return cs


def _infer_db_type(connection_string: str) -> str:
    """Infer db_type from a DSN scheme."""
    scheme = urlparse(connection_string).scheme.lower().split("+")[0]
    mapping = {
        "postgresql": "postgresql",
        "postgres": "postgresql",
        "mysql": "mysql",
        "mariadb": "mariadb",
        "mssql": "mssql",
        "oracle": "oracle",
        "sqlite": "sqlite",
        "bigquery": "bigquery",
        "snowflake": "snowflake",
        "clickhouse": "clickhouse",
        "redshift": "redshift",
        "hive": "hive",
        "duckdb": "duckdb",
    }
    return mapping.get(scheme, scheme)


# ---------------------------------------------------------------------------
# Mask password
# ---------------------------------------------------------------------------


def mask_connection_string(cs: str) -> str:
    """Mask password in a DSN for safe display."""
    if not cs or "://" not in cs:
        return cs or ""
    try:
        scheme, rest = cs.split("://", 1)
        if "@" in rest:
            creds, host_part = rest.split("@", 1)
            if ":" in creds:
                user, _ = creds.split(":", 1)
                return f"{scheme}://{user}:****@{host_part}"
        return f"{scheme}://****"
    except Exception:
        return "***"


# ---------------------------------------------------------------------------
# Server public IP helper  (shown on the form so users can whitelist it)
# ---------------------------------------------------------------------------

_cached_server_ip: Optional[str] = None


def get_server_public_ip() -> str:
    """
    Return the server's public IP address.
    The frontend shows this so the user can add it to their DB firewall rules.
    """
    global _cached_server_ip
    if _cached_server_ip:
        return _cached_server_ip

    # Try multiple services for reliability
    services = [
        "https://api.ipify.org",
        "https://ifconfig.me/ip",
        "https://icanhazip.com",
        "https://checkip.amazonaws.com",
    ]
    for url in services:
        try:
            resp = httpx.get(url, timeout=5.0)
            if resp.status_code == 200:
                ip = resp.text.strip()
                if ip:
                    _cached_server_ip = ip
                    return ip
        except Exception:
            continue

    # Fallback: local hostname resolution
    try:
        hostname = socket.gethostname()
        ip = socket.gethostbyname(hostname)
        _cached_server_ip = ip
        return ip
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Natural language → SQL (for /db/invoke with question)
# ---------------------------------------------------------------------------

# Env: DB_INVOKE_LLM_URL = base URL for LLM API (e.g. http://localhost:8000), used as {url}/LLMs/chat
# Env: DB_INVOKE_LLM_MODEL = optional model_id; if not set, first model from /LLMs/models is used


def _build_schema_summary_for_nl(schema_result: Dict[str, Any]) -> str:
    """Build a compact schema summary string for the LLM prompt from get_full_schema output."""
    lines: List[str] = []
    for table in schema_result.get("tables", [])[:30]:
        tname = table.get("table_name", "?")
        cols = table.get("columns", [])
        col_strs = [f"{c.get('name', '?')} ({str(c.get('type', '')).split('(')[0]})" for c in cols[:25]]
        lines.append(f"Table: {tname}\n  Columns: {', '.join(col_strs)}")
    for view in schema_result.get("views", [])[:10]:
        vname = view.get("view_name", "?")
        cols = view.get("columns", [])
        col_strs = [c.get("name", "?") for c in cols[:15]]
        lines.append(f"View: {vname}\n  Columns: {', '.join(col_strs)}")
    return "\n\n".join(lines) if lines else "No tables or views found."


def _extract_sql_from_llm_response(raw: str) -> Tuple[str, str]:
    """Parse LLM response for JSON with 'sql' and optional 'explanation'. Returns (sql, explanation)."""
    explanation = ""
    raw_clean = raw.strip()
    # Strip common LLM prefixes that break json.loads (e.g. "json\n{...}" or "```json\n{...}")
    if raw_clean.lower().startswith("json"):
        raw_clean = raw_clean[4:].lstrip()
    if raw_clean.startswith("```"):
        raw_clean = re.sub(r"^```(?:json)?\s*", "", raw_clean, flags=re.IGNORECASE)
        raw_clean = re.sub(r"\s*```\s*$", "", raw_clean)
    try:
        data = json.loads(raw_clean)
        sql = (data.get("sql") or "").strip()
        explanation = (data.get("explanation") or "").strip()
        if sql:
            return _normalize_generated_sql(sql), explanation
    except Exception:
        pass
    # Fallback: look for ```sql ... ``` or SELECT... block in original raw
    m = re.search(r"```(?:sql)?\s*([\s\S]+?)```", raw, re.IGNORECASE)
    if m:
        sql = m.group(1).strip().strip(";")
        return _normalize_generated_sql(sql), explanation
    m2 = re.search(r"(\bSELECT\b[\s\S]+?)(?=\n\n|\n```|\Z)", raw, re.IGNORECASE | re.DOTALL)
    if m2:
        sql = m2.group(1).strip().strip(";")
        return _normalize_generated_sql(sql), explanation
    return "", explanation


def _normalize_generated_sql(sql: str) -> str:
    """Remove trailing semicolon and extra whitespace."""
    if not sql:
        return sql
    sql = re.sub(r";\s*$", "", sql.strip())
    return sql.strip()


def _json_safe_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert row values to JSON-serializable form (e.g. datetime/decimal to str)."""
    out = []
    for row in rows:
        safe = {}
        for k, v in row.items():
            if v is None or isinstance(v, (str, int, float, bool)):
                safe[k] = v
            elif hasattr(v, "isoformat"):
                safe[k] = v.isoformat()
            else:
                safe[k] = str(v)
        out.append(safe)
    return out


def _summarize_result_for_question(
    question: str, columns: List[str], rows: List[Dict[str, Any]]
) -> Optional[str]:
    """
    Generate a Convobi-style narrative summary of the query result using the same LLM.
    Returns Markdown summary or None if LLM unavailable.
    """
    system = (
        "You are a helpful analyst. Summarize tabular data clearly and concisely. "
        "Always respond in valid Markdown."
    )
    sample = _json_safe_rows(rows[:20])
    user = (
        "Summarize the following query result for a non-technical stakeholder. Use Markdown only.\n"
        f"Question: {question}\n"
        f"Columns: {json.dumps(columns)}\n"
        f"First rows (sample): {json.dumps(sample)}\n\n"
        "Format your response in Markdown with these sections:\n"
        "- ## Executive summary — one sentence\n"
        "- ## Key observations — bullet list\n"
        "- ## Notable trends / anomalies — short bullets or paragraph\n"
        "- ## Suggested next steps — bullet list (optional)\n"
        "Use ** for bold, - or * for bullets, ## for section headers. Keep it concise."
    )
    return _get_convobi_llm_response(system, user)


def _get_convobi_llm_response(system: str, user: str) -> Optional[str]:
    """
    Use the same LLM as Convobi (apps.convobi.main.CONVOBI_LLM) to generate a response.
    Returns the response text, or None if Convobi LLM is not available.
    """
    try:
        from apps.convobi.main import CONVOBI_LLM
        if CONVOBI_LLM is None:
            return None
        from langchain_core.messages import SystemMessage, HumanMessage
        messages = [
            SystemMessage(content=system),
            HumanMessage(content=user),
        ]
        response = CONVOBI_LLM.invoke(messages)
        return (getattr(response, "content", None) or str(response) or "").strip()
    except Exception as e:
        logger.debug("Convobi LLM not available for db/invoke: %s", e)
        return None


def _generate_sql_via_llm(
    question: str,
    schema_summary: str,
    db_type: str,
    limit: Optional[int] = None,
) -> Tuple[str, str]:
    """
    Generate SQL from natural language using Convobi's LLM (apps.convobi.main.CONVOBI_LLM)
    if available; otherwise fall back to DB_INVOKE_LLM_URL HTTP API.
    Returns (sql, explanation). Raises ValueError if no LLM available or call fails.
    """
    dialect_rules = ""
    if db_type == "sqlite":
        dialect_rules = (
            "SQLite: do not use INTERVAL; use date('now','-N days'). Use strftime for dates. "
            "No semicolon at end of statement."
        )
    elif db_type == "mssql":
        dialect_rules = (
            "Microsoft SQL Server: use TOP N for limit. Use GETDATE() for current date. "
            "Use [brackets] for identifiers if needed. No semicolon at end."
        )
    else:
        dialect_rules = (
            f"Database: {db_type}. Use standard SQL. Use LIMIT for row limit where supported. "
            "Do not end with semicolon."
        )

    system = (
        "You are an expert SQL analyst. Generate only valid SQL from the user question using the given schema. "
        "Return JSON with exactly: {\"sql\": \"SELECT ...\", \"explanation\": \"brief explanation\"}. "
        "Use only tables and columns from the schema. "
    ) + dialect_rules

    user = (
        f"Schema:\n{schema_summary}\n\n"
        f"Question: {question}\n\n"
        f"Max rows to return: {limit if limit is not None else 'default'}\n\n"
        "Return JSON only: {\"sql\": \"...\", \"explanation\": \"...\"}"
    )

    # 1) Prefer Convobi's LLM (same as in convobi/main.py)
    raw = _get_convobi_llm_response(system, user)
    if raw:
        sql, explanation = _extract_sql_from_llm_response(raw)
        if sql:
            return sql, explanation
        raise ValueError("Could not extract SQL from LLM response. Try rephrasing your question.")

    # 2) Fallback: HTTP LLM API (DB_INVOKE_LLM_URL)
    base_url = (os.environ.get("DB_INVOKE_LLM_URL") or "").strip().rstrip("/")
    if not base_url:
        raise ValueError(
            "Natural language invoke requires an LLM. Use the same LLM as Convobi (CONVOBI_LLM in apps/convobi/main.py) "
            "or set environment variable DB_INVOKE_LLM_URL (e.g. http://localhost:8000) to the base URL of your LLM API (POST /LLMs/chat)."
        )

    chat_url = f"{base_url}/LLMs/chat"
    model_id = (os.environ.get("DB_INVOKE_LLM_MODEL") or "").strip()

    if not model_id:
        try:
            models_url = f"{base_url}/LLMs/models"
            resp = httpx.get(models_url, timeout=15.0)
            if resp.status_code == 200:
                data = resp.json()
                models = data if isinstance(data, list) else data.get("models", data) or []
                if models:
                    first = models[0] if isinstance(models[0], dict) else {"id": models[0]}
                    model_id = first.get("id") or first.get("model_id") or (first if isinstance(first, str) else None)
        except Exception as e:
            logger.warning("Could not resolve LLM model_id: %s", e)
        if not model_id:
            raise ValueError(
                "DB_INVOKE_LLM_MODEL not set and could not get model from LLM API. "
                "Set DB_INVOKE_LLM_MODEL to your model id."
            )

    payload = {
        "model_id": model_id,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.2,
        "max_tokens": 2048,
    }

    try:
        resp = httpx.post(chat_url, json=payload, timeout=60.0)
        if resp.status_code != 200:
            raise ValueError(f"LLM API error {resp.status_code}: {resp.text}")
        data = resp.json()
        raw = (data.get("response") or "").strip()
        if not raw:
            raise ValueError("LLM returned empty response")
        sql, explanation = _extract_sql_from_llm_response(raw)
        if not sql:
            raise ValueError("Could not extract SQL from LLM response. Try rephrasing your question.")
        return sql, explanation
    except httpx.HTTPError as e:
        raise ValueError(f"LLM request failed: {e}") from e


# ---------------------------------------------------------------------------
# Core service
# ---------------------------------------------------------------------------


class DatabaseConnectorService:
    """
    Universal database connector supporting public *and* private databases.
    """

    def __init__(self):
        self._connections: Dict[str, Dict[str, Any]] = {}
        self._engines: Dict[str, Engine] = {}
        self._tunnels: Dict[str, _SSHTunnelWrapper] = {}
        self._load_persisted()

    # ---- persistence ----

    def _load_persisted(self):
        try:
            if _SAVED_CONNECTIONS_FILE.exists():
                with open(_SAVED_CONNECTIONS_FILE, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                    for c in data.get("connections", []):
                        self._connections[c["id"]] = c
        except Exception as exc:
            logger.warning("Could not load saved connections: %s", exc)

    def _persist(self):
        try:
            _SAVED_CONNECTIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(_SAVED_CONNECTIONS_FILE, "w", encoding="utf-8") as fh:
                json.dump(
                    {"connections": list(self._connections.values())},
                    fh,
                    indent=2,
                )
        except Exception as exc:
            logger.warning("Could not persist connections: %s", exc)

    # ---- engine creation (with optional SSH tunnel) ----

    def _create_engine(
        self, connection_string: str, config: DBConnectionConfig
    ) -> Tuple[Engine, Optional[_SSHTunnelWrapper]]:
        tunnel: Optional[_SSHTunnelWrapper] = None

        if config.ssh_tunnel:
            parsed = urlparse(connection_string)
            remote_host = parsed.hostname or "localhost"
            db_info = DB_TYPES.get(
                config.db_type or _infer_db_type(connection_string), {}
            )
            remote_port = parsed.port or db_info.get("default_port", 5432)

            tunnel = _SSHTunnelWrapper(config.ssh_tunnel, remote_host, remote_port)
            local_host, local_port = tunnel.start()

            # Rewrite DSN to go through the local tunnel
            connection_string = connection_string.replace(
                f"{remote_host}:{remote_port}",
                f"{local_host}:{local_port}",
            )
            # Fallback if port wasn't explicit in the DSN
            if remote_host in connection_string:
                connection_string = connection_string.replace(
                    remote_host, f"{local_host}:{local_port}"
                )

        engine_kw: Dict[str, Any] = {
            "pool_pre_ping": True,
            "pool_size": config.pool_size,
            "pool_timeout": config.pool_timeout,
        }
        if config.connect_args:
            engine_kw["connect_args"] = config.connect_args

        # SQLite and DuckDB don't support pool_size / pool_timeout
        if connection_string.startswith("sqlite") or connection_string.startswith("duckdb"):
            engine_kw.pop("pool_size", None)
            engine_kw.pop("pool_timeout", None)

        engine = create_engine(connection_string, **engine_kw)
        return engine, tunnel

    # ---- public API ----

    def list_supported_databases(self) -> List[Dict[str, Any]]:
        """Return metadata about every supported DB type (drives the dropdown)."""
        out = []
        for key, info in DB_TYPES.items():
            out.append(
                {
                    "db_type": key,
                    "label": info["label"],
                    "default_port": info["default_port"],
                    "default_driver": info["driver"],
                    "drivers_available": info["drivers_available"],
                    "placeholder_host": info.get("placeholder_host", ""),
                }
            )
        return out

    # --- test connection (the "Test Connection" button) ---

    def test_connection(self, config: DBConnectionConfig) -> Dict[str, Any]:
        """
        Test a database connection **without** saving it.
        Works for both public (direct) and private (SSH tunnel) databases.
        """
        tunnel: Optional[_SSHTunnelWrapper] = None
        try:
            cs = build_connection_string(config)
            engine, tunnel = self._create_engine(cs, config)

            # Connectivity check
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))

            insp = inspect(engine)
            tables = insp.get_table_names()
            views: List[str] = []
            try:
                views = insp.get_view_names()
            except Exception:
                pass
            schemas: List[str] = []
            try:
                schemas = insp.get_schema_names()
            except Exception:
                pass

            db_type = config.db_type or _infer_db_type(cs)
            engine.dispose()

            return {
                "ok": True,
                "message": "Connection successful",
                "db_type": db_type,
                "dialect": DB_TYPES.get(db_type, {}).get("label", db_type),
                "tables_count": len(tables),
                "views_count": len(views),
                "schemas": schemas[:20],
                "ssh_tunnel_active": config.ssh_tunnel is not None,
            }
        except ImportError as exc:
            raise ValueError(str(exc))
        except Exception as exc:
            raise ValueError(f"Connection failed: {exc}")
        finally:
            if tunnel:
                tunnel.stop()

    # --- connect & save (the "Next" button after test passes) ---

    def connect_and_save(self, config: DBConnectionConfig) -> Dict[str, Any]:
        """Test, save, and cache the engine for subsequent operations."""
        test_result = self.test_connection(config)

        cs = build_connection_string(config)
        conn_id = str(uuid.uuid4())[:8]
        db_type = config.db_type or _infer_db_type(cs)
        name = (config.name or "").strip() or (
            f"{DB_TYPES.get(db_type, {}).get('label', db_type)} connection"
        )

        engine, tunnel = self._create_engine(cs, config)
        self._engines[conn_id] = engine
        if tunnel:
            self._tunnels[conn_id] = tunnel

        record: Dict[str, Any] = {
            "id": conn_id,
            "name": name,
            "db_type": db_type,
            "connection_string": cs,
            "dialect": DB_TYPES.get(db_type, {}).get("label", db_type),
            "tables_count": test_result["tables_count"],
            "views_count": test_result.get("views_count", 0),
            "schemas": test_result.get("schemas", []),
            "ssh_tunnel_enabled": config.ssh_tunnel is not None,
            "ssh_host": (
                config.ssh_tunnel.ssh_host if config.ssh_tunnel else None
            ),
        }
        if config.ssh_tunnel:
            record["ssh_config"] = {
                "ssh_host": config.ssh_tunnel.ssh_host,
                "ssh_port": config.ssh_tunnel.ssh_port,
                "ssh_username": config.ssh_tunnel.ssh_username,
                "has_password": bool(config.ssh_tunnel.ssh_password),
                "has_private_key": bool(config.ssh_tunnel.ssh_private_key),
            }

        self._connections[conn_id] = record
        self._persist()

        return {
            "ok": True,
            "connection_id": conn_id,
            "name": name,
            "db_type": db_type,
            "dialect": record["dialect"],
            "tables_count": test_result["tables_count"],
            "views_count": test_result.get("views_count", 0),
            "schemas": test_result.get("schemas", []),
            "connection_string_masked": mask_connection_string(cs),
            "ssh_tunnel_active": config.ssh_tunnel is not None,
            "message": "Connection saved successfully",
        }

    # --- CRUD ---

    def list_connections(self) -> List[Dict[str, Any]]:
        out = []
        for c in self._connections.values():
            out.append(
                {
                    "id": c["id"],
                    "name": c.get("name", "Unnamed"),
                    "db_type": c.get("db_type", "unknown"),
                    "dialect": c.get("dialect", "unknown"),
                    "connection_string_masked": mask_connection_string(
                        c.get("connection_string", "")
                    ),
                    "tables_count": c.get("tables_count", 0),
                    "views_count": c.get("views_count", 0),
                    "ssh_tunnel_enabled": c.get("ssh_tunnel_enabled", False),
                    "ssh_host": c.get("ssh_host"),
                }
            )
        return out

    def get_connection(self, connection_id: str) -> Optional[Dict[str, Any]]:
        c = self._connections.get(connection_id)
        if not c:
            return None
        return {
            "id": c["id"],
            "name": c.get("name", "Unnamed"),
            "db_type": c.get("db_type", "unknown"),
            "dialect": c.get("dialect", "unknown"),
            "connection_string_masked": mask_connection_string(
                c.get("connection_string", "")
            ),
            "tables_count": c.get("tables_count", 0),
            "views_count": c.get("views_count", 0),
            "schemas": c.get("schemas", []),
            "ssh_tunnel_enabled": c.get("ssh_tunnel_enabled", False),
            "ssh_config": c.get("ssh_config"),
        }

    def delete_connection(self, connection_id: str) -> Dict[str, Any]:
        if connection_id not in self._connections:
            raise ValueError(f"Connection '{connection_id}' not found")

        engine = self._engines.pop(connection_id, None)
        if engine:
            engine.dispose()
        tunnel = self._tunnels.pop(connection_id, None)
        if tunnel:
            tunnel.stop()

        name = self._connections[connection_id].get("name", "Unnamed")
        del self._connections[connection_id]
        self._persist()
        return {"ok": True, "deleted_id": connection_id, "name": name}

    def update_connection(
        self, connection_id: str, updates: Dict[str, Any]
    ) -> Dict[str, Any]:
        if connection_id not in self._connections:
            raise ValueError(f"Connection '{connection_id}' not found")
        c = self._connections[connection_id]
        if "name" in updates:
            c["name"] = updates["name"]
        self._connections[connection_id] = c
        self._persist()
        return {"ok": True, "connection_id": connection_id, "name": c.get("name")}

    # ---- schema introspection ----

    def _get_engine(self, connection_id: str) -> Engine:
        if connection_id in self._engines:
            return self._engines[connection_id]
        c = self._connections.get(connection_id)
        if not c:
            raise ValueError(f"Connection '{connection_id}' not found")
        cs = c.get("connection_string")
        if not cs:
            raise ValueError(f"Connection '{connection_id}' has no connection_string")
        cfg = DBConnectionConfig(connection_string=cs)
        engine, tunnel = self._create_engine(cs, cfg)
        self._engines[connection_id] = engine
        if tunnel:
            self._tunnels[connection_id] = tunnel
        return engine

    def get_schemas(self, connection_id: str) -> Dict[str, Any]:
        engine = self._get_engine(connection_id)
        insp = inspect(engine)
        schemas: List[str] = []
        try:
            schemas = insp.get_schema_names()
        except Exception:
            pass
        return {"connection_id": connection_id, "schemas": schemas}

    def get_tables(
        self, connection_id: str, schema_name: Optional[str] = None
    ) -> Dict[str, Any]:
        engine = self._get_engine(connection_id)
        insp = inspect(engine)
        tables = insp.get_table_names(schema=schema_name)
        views: List[str] = []
        try:
            views = insp.get_view_names(schema=schema_name)
        except Exception:
            pass
        return {
            "connection_id": connection_id,
            "schema": schema_name or "default",
            "tables": tables,
            "views": views,
            "tables_count": len(tables),
            "views_count": len(views),
        }

    def get_full_schema(
        self, connection_id: str, schema_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Full introspection: columns, PKs, FKs, indexes, row counts."""
        engine = self._get_engine(connection_id)
        insp = inspect(engine)

        tables_info: List[Dict[str, Any]] = []
        for table in insp.get_table_names(schema=schema_name):
            cols = []
            try:
                for col in insp.get_columns(table, schema=schema_name):
                    cols.append(
                        {
                            "name": col["name"],
                            "type": str(col["type"]),
                            "nullable": col.get("nullable", True),
                            "default": (
                                str(col["default"]) if col.get("default") else None
                            ),
                            "autoincrement": col.get("autoincrement"),
                        }
                    )
            except Exception as e:
                cols = [{"error": str(e)}]

            pk: List[str] = []
            try:
                pk_info = insp.get_pk_constraint(table, schema=schema_name)
                pk = pk_info.get("constrained_columns", []) if pk_info else []
            except Exception:
                pass

            fks: List[Dict[str, Any]] = []
            try:
                for fk in insp.get_foreign_keys(table, schema=schema_name):
                    fks.append(
                        {
                            "constrained_columns": fk.get("constrained_columns", []),
                            "referred_table": fk.get("referred_table"),
                            "referred_columns": fk.get("referred_columns", []),
                            "referred_schema": fk.get("referred_schema"),
                        }
                    )
            except Exception:
                pass

            indexes: List[Dict[str, Any]] = []
            try:
                for idx in insp.get_indexes(table, schema=schema_name):
                    indexes.append(
                        {
                            "name": idx.get("name"),
                            "columns": idx.get("column_names", []),
                            "unique": idx.get("unique", False),
                        }
                    )
            except Exception:
                pass

            row_count = None
            try:
                sp = f'"{schema_name}".' if schema_name else ""
                with engine.connect() as conn:
                    r = conn.execute(text(f'SELECT COUNT(*) FROM {sp}"{table}"'))
                    row_count = r.scalar()
            except Exception:
                pass

            tables_info.append(
                {
                    "table_name": table,
                    "columns": cols,
                    "primary_key": pk,
                    "foreign_keys": fks,
                    "indexes": indexes,
                    "row_count": row_count,
                }
            )

        views_info: List[Dict[str, Any]] = []
        try:
            for view in insp.get_view_names(schema=schema_name):
                vcols = []
                try:
                    for col in insp.get_columns(view, schema=schema_name):
                        vcols.append(
                            {"name": col["name"], "type": str(col["type"])}
                        )
                except Exception:
                    pass
                views_info.append({"view_name": view, "columns": vcols})
        except Exception:
            pass

        conn_info = self._connections.get(connection_id, {})
        return {
            "connection_id": connection_id,
            "name": conn_info.get("name", "Unknown"),
            "db_type": conn_info.get("db_type", "unknown"),
            "schema": schema_name or "default",
            "tables": tables_info,
            "views": views_info,
            "tables_count": len(tables_info),
            "views_count": len(views_info),
        }

    def preview_table(
        self,
        connection_id: str,
        table_name: str,
        schema_name: Optional[str] = None,
        limit: int = 50,
    ) -> Dict[str, Any]:
        engine = self._get_engine(connection_id)
        sp = f'"{schema_name}".' if schema_name else ""
        db_type = self._connections.get(connection_id, {}).get("db_type", "")

        if db_type == "mssql":
            q = f"SELECT TOP {limit} * FROM {sp}[{table_name}]"
        elif db_type == "oracle":
            q = f'SELECT * FROM {sp}"{table_name}" WHERE ROWNUM <= {limit}'
        else:
            q = f'SELECT * FROM {sp}"{table_name}" LIMIT {limit}'

        try:
            with engine.connect() as conn:
                result = conn.execute(text(q))
                columns = list(result.keys())
                rows = [dict(zip(columns, row)) for row in result.fetchall()]
                return {
                    "connection_id": connection_id,
                    "table_name": table_name,
                    "columns": columns,
                    "rows": rows,
                    "row_count": len(rows),
                    "limited_to": limit,
                }
        except Exception as exc:
            raise ValueError(f"Error previewing table '{table_name}': {exc}")

    def execute_query(
        self,
        connection_id: str,
        query: str,
        limit: int = 100,
        timeout: int = 30,
    ) -> Dict[str, Any]:
        engine = self._get_engine(connection_id)
        q = query.strip()
        q_upper = q.upper()
        is_read = q_upper.startswith(("SELECT", "WITH", "SHOW", "DESCRIBE", "EXPLAIN"))

        try:
            with engine.connect() as conn:
                result = conn.execute(text(q))
                if is_read:
                    columns = list(result.keys())
                    rows = [
                        dict(zip(columns, row)) for row in result.fetchmany(limit)
                    ]
                    return {
                        "connection_id": connection_id,
                        "query": q,
                        "columns": columns,
                        "rows": rows,
                        "row_count": len(rows),
                        "limited_to": limit,
                        "type": "select",
                    }
                else:
                    conn.commit()
                    return {
                        "connection_id": connection_id,
                        "query": q,
                        "rows_affected": result.rowcount,
                        "type": "execute",
                        "message": f"Query executed. Rows affected: {result.rowcount}",
                    }
        except Exception as exc:
            raise ValueError(f"Query execution failed: {exc}")

    def invoke_with_question(
        self,
        connection_id: str,
        question: str,
        limit: int = 100,
        timeout: int = 30,
    ) -> Dict[str, Any]:
        """
        Answer a natural language question by generating SQL via LLM and executing it
        on the saved connection. Requires DB_INVOKE_LLM_URL (and optionally
        DB_INVOKE_LLM_MODEL) to be set.
        """
        conn = self._connections.get(connection_id)
        if not conn:
            raise ValueError(f"Connection '{connection_id}' not found")
        db_type = conn.get("db_type", "sqlite")
        schema_result = self.get_full_schema(connection_id, schema_name=None)
        schema_summary = _build_schema_summary_for_nl(schema_result)
        generated_sql, explanation = _generate_sql_via_llm(
            question, schema_summary, db_type, limit=limit
        )
        result = self.execute_query(
            connection_id=connection_id,
            query=generated_sql,
            limit=limit,
            timeout=timeout,
        )
        result["question"] = question
        result["generated_sql"] = generated_sql
        result["explanation"] = explanation
        # Convobi-style narrative summary for read results
        if result.get("type") == "select" and result.get("rows") is not None and result.get("columns"):
            summary_text = _summarize_result_for_question(
                question, result["columns"], result["rows"]
            )
            if summary_text:
                result["summary"] = summary_text
        return result

    # ---- cleanup ----

    def cleanup(self):
        for engine in self._engines.values():
            try:
                engine.dispose()
            except Exception:
                pass
        self._engines.clear()
        for tunnel in self._tunnels.values():
            try:
                tunnel.stop()
            except Exception:
                pass
        self._tunnels.clear()
