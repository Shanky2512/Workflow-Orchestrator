from __future__ import annotations
import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence
from sqlmodel import SQLModel, Field, Session, create_engine, select

# DB setup: default DB file (override via APP_DB_PATH)
DB_PATH = os.getenv("APP_DB_PATH", "apps_wizard_new.db")
DB_URI = f"sqlite:///{DB_PATH}"

# Ensure parent dir exists
Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)

engine = create_engine(
    DB_URI,
    connect_args={"check_same_thread": False},
    echo=False,
)

# Enforce foreign keys in SQLite
from sqlalchemy import event
from sqlite3 import Connection as SQLite3Connection

@event.listens_for(engine, "connect")
def _set_sqlite_pragma(dbapi_connection, connection_record):
    if isinstance(dbapi_connection, SQLite3Connection):
        cur = dbapi_connection.cursor()
        cur.execute("PRAGMA foreign_keys=ON;")
        cur.close()

def get_session() -> Session:
    """FastAPI dependency: yields a SQLModel session."""
    with Session(engine) as session:
        yield session

# -------------------------------------------------------------------------------------------------
# Association (link) tables
# -------------------------------------------------------------------------------------------------
class AppLlmLink(SQLModel, table=True):
    __tablename__ = "app_llm_link"
    app_id: str = Field(foreign_key="app.id", primary_key=True)
    llm_id: str = Field(foreign_key="llm.id", primary_key=True)


class AppDataSourceLink(SQLModel, table=True):
    __tablename__ = "app_datasource_link"
    app_id: str = Field(foreign_key="app.id", primary_key=True)
    data_source_id: str = Field(foreign_key="datasource.id", primary_key=True)


class AppGuardrailCategoryLink(SQLModel, table=True):
    __tablename__ = "app_guardrail_category_link"
    app_id: str = Field(foreign_key="app.id", primary_key=True)
    guardrail_category_id: str = Field(
        foreign_key="guardrailcategory.id", primary_key=True
    )


class AppDesignationLink(SQLModel, table=True):
    __tablename__ = "app_designation_link"
    app_id: str = Field(foreign_key="app.id", primary_key=True)
    designation_id: str = Field(foreign_key="designation.id", primary_key=True)


class AppBusinessUnitLink(SQLModel, table=True):
    __tablename__ = "app_businessunit_link"
    app_id: str = Field(foreign_key="app.id", primary_key=True)
    business_unit_id: str = Field(foreign_key="businessunit.id", primary_key=True)


class AppTagLink(SQLModel, table=True):
    __tablename__ = "app_tag_link"
    app_id: str = Field(foreign_key="app.id", primary_key=True)
    tag_id: str = Field(foreign_key="tag.id", primary_key=True)

# NEW: links to agents & workflows (multi-select at the App level)
class AppAgentLink(SQLModel, table=True):
    __tablename__ = "app_agent_link"
    app_id: str = Field(foreign_key="app.id", primary_key=True)
    agent_id: str = Field(foreign_key="agent.id", primary_key=True)


class AppWorkflowLink(SQLModel, table=True):
    __tablename__ = "app_workflow_link"
    app_id: str = Field(foreign_key="app.id", primary_key=True)
    workflow_id: str = Field(foreign_key="workflow.id", primary_key=True)

# -------------------------------------------------------------------------------------------------
# Core tables
# -------------------------------------------------------------------------------------------------
class AppModel(SQLModel, table=True):
    __tablename__ = "app"
    id: str = Field(default_factory=lambda: uuid.uuid4().hex, primary_key=True)
    name: str
    description: str
    status: str = Field(default="draft")  # draft\npublished\narchived

    # App-wide flags
    available_for_all_users: bool = Field(default=False)
    upload_enabled: bool = Field(default=False)

    # Context (step 3)
    welcome_prompt: Optional[str] = None
    disclaimer: Optional[str] = None
    sorry_message: Optional[str] = None
    starter_questions_json: Optional[str] = None  # JSON list of strings

    # Chat config (step 4)
    persona_id: Optional[str] = Field(default=None, foreign_key="persona.id")
    persona_text: Optional[str] = None
    guardrail_text: Optional[str] = None

    # Legacy single workflow defaults (kept for backward compatibility)
    workflow_id: Optional[str] = None
    workflow_name: Optional[str] = None
    workflow_mode: Optional[str] = "draft"  # "draft" \n "temp" \n "final"

    # Misc
    logo_url: Optional[str] = None
    created_by: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.utcnow(), nullable=False)
    updated_at: datetime = Field(default_factory=lambda: datetime.utcnow(), nullable=False)


class Llm(SQLModel, table=True):
    __tablename__ = "llm"
    id: str = Field(primary_key=True)
    name: str
    provider: Optional[str] = None
    model_name: Optional[str] = None
    base_url: Optional[str] = None
    api_key_env: Optional[str] = None
    api_key: Optional[str] = None
    is_default: bool = Field(default=False)


class DataSource(SQLModel, table=True):
    __tablename__ = "datasource"
    id: str = Field(primary_key=True)
    name: str
    kind: Optional[str] = None
    meta_json: Optional[str] = None


class Persona(SQLModel, table=True):
    __tablename__ = "persona"
    id: str = Field(primary_key=True)
    name: str


class GuardrailCategory(SQLModel, table=True):
    __tablename__ = "guardrailcategory"
    id: str = Field(primary_key=True)
    name: str


class Designation(SQLModel, table=True):
    __tablename__ = "designation"
    id: str = Field(primary_key=True)
    name: str


class BusinessUnit(SQLModel, table=True):
    __tablename__ = "businessunit"
    id: str = Field(primary_key=True)
    name: str


class Tag(SQLModel, table=True):
    __tablename__ = "tag"
    id: str = Field(primary_key=True)
    name: str


# NEW: Agent & Workflow catalogs
class Agent(SQLModel, table=True):
    __tablename__ = "agent"
    id: str = Field(primary_key=True)
    name: str
    description: Optional[str] = None
    supported_data_sources_json: Optional[str] = None
    tags_json: Optional[str] = None


class Workflow(SQLModel, table=True):
    __tablename__ = "workflow"
    id: str = Field(primary_key=True)
    name: str
    description: Optional[str] = None
    meta_json: Optional[str] = None

# -------------------------------------------------------------------------------------------------
# Chat session history (optional - retained)
# -------------------------------------------------------------------------------------------------
class ChatSession(SQLModel, table=True):
    __tablename__ = "chat_session"
    id: str = Field(default_factory=lambda: uuid.uuid4().hex, primary_key=True)
    app_id: str = Field(foreign_key="app.id")
    user_id: Optional[str] = None
    title: Optional[str] = None
    llm_id: Optional[str] = Field(default=None, foreign_key="llm.id")
    is_closed: bool = Field(default=False)
    started_at: datetime = Field(default_factory=lambda: datetime.utcnow(), nullable=False)
    updated_at: datetime = Field(default_factory=lambda: datetime.utcnow(), nullable=False)


class ChatMessage(SQLModel, table=True):
    __tablename__ = "chat_message"
    id: str = Field(default_factory=lambda: uuid.uuid4().hex, primary_key=True)
    session_id: str = Field(foreign_key="chat_session.id")
    role: str  # 'system' \n 'user' \n 'assistant'
    content: str
    created_at: datetime = Field(default_factory=lambda: datetime.utcnow(), nullable=False)

# --- NEW: attachments uploaded by users (optional link to a chat session) ---
class ChatAttachment(SQLModel, table=True):
    __tablename__ = "chat_attachment"
    id: str = Field(default_factory=lambda: uuid.uuid4().hex, primary_key=True)
    app_id: str = Field(foreign_key="app.id")
    session_id: Optional[str] = Field(default=None, foreign_key="chat_session.id")
    original_filename: str
    stored_filename: str
    public_url: Optional[str] = None  # e.g., /static/uploads/<stored_filename>
    mime_type: Optional[str] = None
    size_bytes: Optional[int] = None
    meta_json: Optional[str] = None  # extra metadata (e.g., quick type hints)
    created_at: datetime = Field(default_factory=lambda: datetime.utcnow(), nullable=False)

# -------------------------------------------------------------------------------------------------
# Pydantic Schemas
# -------------------------------------------------------------------------------------------------
from pydantic import BaseModel, Field as PydField

class AppCreate(BaseModel):
    name: str
    description: str
    created_by: Optional[str] = None
    # Removed legacy workflow_* fields from create payload


class AppBasicPatch(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    created_by: Optional[str] = None
    # NEW: allow changing orchestrator execution mode per app
    workflow_mode: Optional[str] = PydField(
        default=None,
        description="Allowed values: 'temp' or 'final'"
    )
    # Removed legacy workflow_* fields from basic patch payload


class AppSetupPatch(BaseModel):
    llm_ids: list[str] = PydField(default_factory=list)
    data_source_ids: list[str] = PydField(default_factory=list)
    # IMPORTANT: "skills" now means combined Agents + Workflows.
    # Expect IDs like "agent:<agent_id>" or "workflow:<workflow_id>"
    skill_ids: list[str] = PydField(default_factory=list)


class AccessPermissionPatch(BaseModel):
    designation_ids: list[str] = PydField(default_factory=list)
    business_unit_ids: list[str] = PydField(default_factory=list)
    tag_ids: list[str] = PydField(default_factory=list)
    available_for_all_users: bool = False
    upload_enabled: bool = False


class ContextPatch(BaseModel):
    welcome_prompt: str
    disclaimer: str
    sorry_message: str
    starter_questions: list[str] = PydField(default_factory=list)


class ChatConfigPatch(BaseModel):
    persona_id: Optional[str] = None
    persona_text: Optional[str] = None
    guardrail_category_ids: list[str] = PydField(default_factory=list)
    guardrail_text: Optional[str] = None


class LinkedItem(BaseModel):
    id: str
    name: str


class LlmItem(LinkedItem):
    provider: Optional[str] = None
    model_name: Optional[str] = None
    base_url: Optional[str] = None
    api_key_env: Optional[str] = None
    is_default: Optional[bool] = None


class AppDetail(BaseModel):
    id: str
    name: str
    description: str
    status: str

    # Step 1 bindings
    llms: list[LlmItem]
    data_sources: list[LinkedItem]
    # "skills" returns combined Agents + Workflows with prefixed IDs
    skills: list[LinkedItem]

    # Step 2 access
    designations: list[LinkedItem]
    business_units: list[LinkedItem]
    tags: list[LinkedItem]
    available_for_all_users: bool
    upload_enabled: bool

    # Step 3 context
    welcome_prompt: Optional[str]
    disclaimer: Optional[str]
    sorry_message: Optional[str]
    starter_questions: list[str]

    # Step 4 chat config
    persona: Optional[LinkedItem]
    persona_text: Optional[str]
    guardrail_categories: list[LinkedItem]
    guardrail_text: Optional[str]

    # Misc
    logo_url: Optional[str]
    created_by: Optional[str]
    created_at: datetime
    updated_at: datetime
    # Removed legacy workflow_* fields from response

# -------------------------------------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------------------------------------
def _sync_links(
    session: Session,
    app_row: AppModel,
    target_model: type[SQLModel],
    link_model: type[SQLModel],
    incoming_ids: Optional[Sequence[str]],
    link_fk_name: str,
) -> None:
    desired = set(i for i in (incoming_ids or []) if i and isinstance(i, str))
    curr_rows = session.exec(
        select(link_model).where(getattr(link_model, "app_id") == app_row.id)
    ).all()
    current = set(getattr(r, link_fk_name) for r in curr_rows)
    to_add = desired - current
    to_remove = current - desired

    if to_add:
        existing = set(
            r.id
            for r in session.exec(select(target_model).where(target_model.id.in_(to_add)))
        )
        missing = to_add - existing
        if missing:
            raise ValueError(
                f"Invalid {target_model.__tablename__} ids: {', '.join(sorted(missing))}"
            )
        for tid in to_add:
            row = link_model(app_id=app_row.id)
            setattr(row, link_fk_name, tid)
            session.add(row)

    if to_remove:
        for r in curr_rows:
            if getattr(r, link_fk_name) in to_remove:
                session.delete(r)


def _serialize_app(session: Session, app_row: AppModel) -> AppDetail:
    from sqlmodel import select

    llm_rows = session.exec(
        select(Llm).where(
            Llm.id.in_(
                select(AppLlmLink.llm_id).where(AppLlmLink.app_id == app_row.id)
            )
        )
    ).all()

    ds_rows = session.exec(
        select(DataSource).where(
            DataSource.id.in_(
                select(AppDataSourceLink.data_source_id).where(
                    AppDataSourceLink.app_id == app_row.id
                )
            )
        )
    ).all()

    # NEW: combined "skills" from Agents and Workflows
    agent_rows = session.exec(
        select(Agent).where(
            Agent.id.in_(select(AppAgentLink.agent_id).where(AppAgentLink.app_id == app_row.id))
        )
    ).all()
    wf_rows = session.exec(
        select(Workflow).where(
            Workflow.id.in_(
                select(AppWorkflowLink.workflow_id).where(
                    AppWorkflowLink.app_id == app_row.id
                )
            )
        )
    ).all()
    combined_skills = (
        [LinkedItem(id=f"agent:{r.id}", name=r.name) for r in agent_rows]
        + [LinkedItem(id=f"workflow:{r.id}", name=r.name) for r in wf_rows]
    )

    gr_rows = session.exec(
        select(GuardrailCategory).where(
            GuardrailCategory.id.in_(
                select(AppGuardrailCategoryLink.guardrail_category_id).where(
                    AppGuardrailCategoryLink.app_id == app_row.id
                )
            )
        )
    ).all()
    desig_rows = session.exec(
        select(Designation).where(
            Designation.id.in_(
                select(AppDesignationLink.designation_id).where(
                    AppDesignationLink.app_id == app_row.id
                )
            )
        )
    ).all()
    bu_rows = session.exec(
        select(BusinessUnit).where(
            BusinessUnit.id.in_(
                select(AppBusinessUnitLink.business_unit_id).where(
                    AppBusinessUnitLink.app_id == app_row.id
                )
            )
        )
    ).all()
    tag_rows = session.exec(
        select(Tag).where(
            Tag.id.in_(select(AppTagLink.tag_id).where(AppTagLink.app_id == app_row.id))
        )
    ).all()

    persona_obj = session.get(Persona, app_row.persona_id) if app_row.persona_id else None

    try:
        starter_questions = (
            json.loads(app_row.starter_questions_json)
            if app_row.starter_questions_json
            else []
        )
        if not isinstance(starter_questions, list):
            starter_questions = []
    except Exception:
        starter_questions = []

    return AppDetail(
        id=app_row.id,
        name=app_row.name,
        description=app_row.description,
        status=app_row.status,
        llms=[
            LlmItem(
                id=r.id,
                name=r.name,
                provider=r.provider,
                model_name=r.model_name,
                base_url=r.base_url,
                api_key_env=r.api_key_env,
                is_default=bool(r.is_default),
            )
            for r in llm_rows
        ],
        data_sources=[LinkedItem(id=r.id, name=r.name) for r in ds_rows],
        skills=combined_skills,
        designations=[LinkedItem(id=r.id, name=r.name) for r in desig_rows],
        business_units=[LinkedItem(id=r.id, name=r.name) for r in bu_rows],
        tags=[LinkedItem(id=r.id, name=r.name) for r in tag_rows],
        available_for_all_users=bool(app_row.available_for_all_users),
        upload_enabled=bool(app_row.upload_enabled),
        welcome_prompt=app_row.welcome_prompt,
        disclaimer=app_row.disclaimer,
        sorry_message=app_row.sorry_message,
        starter_questions=starter_questions,
        persona=(LinkedItem(id=persona_obj.id, name=persona_obj.name) if persona_obj else None),
        persona_text=app_row.persona_text,
        guardrail_categories=[LinkedItem(id=r.id, name=r.name) for r in gr_rows],
        guardrail_text=app_row.guardrail_text,
        logo_url=app_row.logo_url,
        created_by=app_row.created_by,
        created_at=app_row.created_at,
        updated_at=app_row.updated_at,
        # workflow_* intentionally not returned anymore
    )


def load_llms_from_json(session: Session, json_path: str = "llm_provider_onprem.json") -> None:
    """
    Load/refresh LLM catalog from a JSON file.
    Supports:
    1) A list of objects
    2) An object with {"models": [...] }
    """
    with open(json_path, "r", encoding="utf-8") as fp:
        raw = json.load(fp)
    items = raw.get("models") if isinstance(raw, dict) else raw
    if not isinstance(items, list):
        raise ValueError("LLM catalog JSON must be a list or an object with 'models'.")
    for it in items:
        llm_id = it["id"]
        row = session.get(Llm, llm_id)
        if not row:
            row = Llm(id=llm_id, name=it.get("name", llm_id))
        row.name = it.get("name", row.name)
        row.provider = it.get("provider")
        row.model_name = it.get("model_name")
        row.base_url = it.get("base_url")
        row.api_key = it.get("api_key")
        row.api_key_env = it.get("api_key_env", row.api_key_env)
        row.is_default = bool(it.get("is_default", False))
        session.add(row)
        session.commit()

# -------------------------------------------------------------------------------------------------
from sqlalchemy import text

def _column_names(conn, table: str) -> set[str]:
    rows = conn.exec_driver_sql(f"PRAGMA table_info({table});").fetchall()
    return {row[1] for row in rows}


def migrate_schema(engine) -> None:
    """Idempotent SQLite migrations (add columns introduced after first DB creation)."""
    with engine.begin() as conn:
        # DataSource: add kind/meta_json
        if "datasource" in {
            r[0] for r in conn.exec_driver_sql("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
        }:
            cols = _column_names(conn, "datasource")
            if "kind" not in cols:
                conn.exec_driver_sql("ALTER TABLE datasource ADD COLUMN kind VARCHAR;")
            if "meta_json" not in cols:
                conn.exec_driver_sql("ALTER TABLE datasource ADD COLUMN meta_json TEXT;")

        # Llm: add api_key/is_default if missing
        if "llm" in {
            r[0] for r in conn.exec_driver_sql("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
        }:
            cols = _column_names(conn, "llm")
            if "api_key" not in cols:
                conn.exec_driver_sql("ALTER TABLE llm ADD COLUMN api_key TEXT;")
            if "is_default" not in cols:
                conn.exec_driver_sql("ALTER TABLE llm ADD COLUMN is_default BOOLEAN DEFAULT 0;")

        # App: add workflow_* defaults if missing
        if "app" in {
            r[0] for r in conn.exec_driver_sql("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
        }:
            cols = _column_names(conn, "app")
            if "workflow_id" not in cols:
                conn.exec_driver_sql("ALTER TABLE app ADD COLUMN workflow_id VARCHAR;")
            if "workflow_name" not in cols:
                conn.exec_driver_sql("ALTER TABLE app ADD COLUMN workflow_name VARCHAR;")
            if "workflow_mode" not in cols:
                conn.exec_driver_sql("ALTER TABLE app ADD COLUMN workflow_mode VARCHAR;")


# -------------------------------------------------------------------------------------------------
# Seeding (non-LLM catalogs with dummy values)
# -------------------------------------------------------------------------------------------------
def _upsert(session: Session, model: type[SQLModel], id: str, **fields):
    row = session.get(model, id)
    if not row:
        row = model(id=id, **fields)
    else:
        for k, v in fields.items():
            setattr(row, k, v)
    session.add(row)
    return row


def seed_catalog(session: Session) -> None:
    """Seed catalog tables with dummy values if empty (excluding LLMs)."""
    # Personas
    if not session.exec(select(Persona).limit(1)).first():
        _upsert(session, Persona, "support-agent", name="Support Agent")
        _upsert(session, Persona, "hr-assistant", name="HR Assistant")
        _upsert(session, Persona, "sales-assistant", name="Sales Assistant")

    # Guardrail categories
    if not session.exec(select(GuardrailCategory).limit(1)).first():
        _upsert(session, GuardrailCategory, "safety", name="Safety")
        _upsert(session, GuardrailCategory, "pii", name="PII")
        _upsert(session, GuardrailCategory, "compliance", name="Compliance")

    # Designations
    if not session.exec(select(Designation).limit(1)).first():
        _upsert(session, Designation, "exec", name="Executive")
        _upsert(session, Designation, "mgr", name="Manager")
        _upsert(session, Designation, "ic", name="Individual Contributor")

    # Business units
    if not session.exec(select(BusinessUnit).limit(1)).first():
        _upsert(session, BusinessUnit, "bu-ops", name="Operations")
        _upsert(session, BusinessUnit, "bu-hr", name="Human Resources")
        _upsert(session, BusinessUnit, "bu-sales", name="Sales")

    # Tags
    if not session.exec(select(Tag).limit(1)).first():
        _upsert(session, Tag, "tag-internal", name="Internal")
        _upsert(session, Tag, "tag-external", name="External")
        _upsert(session, Tag, "tag-priority", name="Priority")

    # NEW: sample Agents
    if not session.exec(select(Agent).limit(1)).first():
        _upsert(session, Agent, "ag_sales_expert", name="Sales Expert", description="Specialist in CRM and pipeline analysis")
        _upsert(session, Agent, "ag_policy_checker", name="Policy Checker", description="Validates requests against HR and IT policies")

    # NEW: sample Workflows
    if not session.exec(select(Workflow).limit(1)).first():
        _upsert(session, Workflow, "wf_lead_to_opportunity", name="Lead → Opportunity", description="Qualify leads and create opportunities")
        _upsert(session, Workflow, "wf_policy_qna", name="Policy Q&A", description="Answer policy questions and cite sources")

    session.commit()


def seed_demo_app(session: Session) -> None:
    """Create a sample app with placeholder context (idempotent)."""
    app_row = session.exec(select(AppModel).where(AppModel.name == "Sample App")).first()
    if not app_row:
        app_row = AppModel(
            name="Sample App",
            description="Demo app seeded for first-time use.",
            status="draft",
            welcome_prompt="Welcome to Sample App!",
            disclaimer="For internal testing only.",
            sorry_message="Sorry, I couldn't find that.",
            starter_questions_json=json.dumps(["What can you do?"]),
            created_by="seeder",
        )
        session.add(app_row)
        session.commit()
        session.refresh(app_row)


def init_db_and_seed() -> None:
    """Create tables if missing and seed demo data (safe to run multiple times)."""
    SQLModel.metadata.create_all(engine)
    migrate_schema(engine)
    with Session(engine) as session:
        seed_catalog(session)
        json_path = os.getenv("LLM_CATALOG_PATH", "llm_provider_onprem.json")
        if Path(json_path).exists():
            load_llms_from_json(session, json_path=json_path)
        seed_demo_app(session)

# Run at import
init_db_and_seed()

__all__ = [
    # DB / session
    "get_session",
    # Models
    "AppModel",
    "Llm",
    "DataSource",
    "Persona",
    "GuardrailCategory",
    "Designation",
    "BusinessUnit",
    "Tag",
    "Agent",
    "Workflow",
    # Link tables
    "AppLlmLink",
    "AppDataSourceLink",
    "AppGuardrailCategoryLink",
    "AppDesignationLink",
    "AppBusinessUnitLink",
    "AppTagLink",
    "AppAgentLink",
    "AppWorkflowLink",
    # Chat history
    "ChatSession",
    "ChatMessage",
    "ChatAttachment",
    # Schemas
    "AppCreate",
    "AppBasicPatch",
    "AppSetupPatch",
    "AccessPermissionPatch",
    "ContextPatch",
    "ChatConfigPatch",
    "AppDetail",
    # Helpers
    "_sync_links",
    "_serialize_app",
    "load_llms_from_json",
]