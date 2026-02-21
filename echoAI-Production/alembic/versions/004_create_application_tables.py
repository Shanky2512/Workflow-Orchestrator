"""Create application orchestrator tables

Revision ID: 004
Revises: 003
Create Date: 2026-02-08

Creates all tables for the AI Application Orchestrator module:

Lookup / Catalog tables:
    - app_personas
    - app_guardrail_categories
    - app_designations
    - app_business_units
    - app_tags
    - app_llms
    - app_data_sources

Core table:
    - applications (with partial indexes on user_id, status)

Association / Link tables:
    - application_llm_links
    - application_skill_links
    - application_data_source_links
    - application_designation_links
    - application_business_unit_links
    - application_tag_links
    - application_guardrail_links

Chat tables:
    - application_chat_sessions
    - application_chat_messages

Document / RAG table:
    - application_documents

Audit trail:
    - application_execution_traces

Also seeds lookup tables with initial data and creates update triggers.

Target Database: PostgreSQL 16
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = '004'
down_revision: Union[str, None] = '003'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ========================================================================
    # LOOKUP / CATALOG TABLES (no FK dependencies — created first)
    # ========================================================================

    # TABLE: app_personas
    op.create_table(
        'app_personas',
        sa.Column('persona_id', postgresql.UUID(as_uuid=True), server_default=sa.text('gen_random_uuid()'), nullable=False),
        sa.Column('name', sa.String(100), nullable=False),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.PrimaryKeyConstraint('persona_id'),
        sa.UniqueConstraint('name', name='uq_app_personas_name'),
    )

    # TABLE: app_guardrail_categories
    op.create_table(
        'app_guardrail_categories',
        sa.Column('guardrail_category_id', postgresql.UUID(as_uuid=True), server_default=sa.text('gen_random_uuid()'), nullable=False),
        sa.Column('name', sa.String(100), nullable=False),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.PrimaryKeyConstraint('guardrail_category_id'),
        sa.UniqueConstraint('name', name='uq_app_guardrail_categories_name'),
    )

    # TABLE: app_designations
    op.create_table(
        'app_designations',
        sa.Column('designation_id', postgresql.UUID(as_uuid=True), server_default=sa.text('gen_random_uuid()'), nullable=False),
        sa.Column('name', sa.String(100), nullable=False),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.PrimaryKeyConstraint('designation_id'),
        sa.UniqueConstraint('name', name='uq_app_designations_name'),
    )

    # TABLE: app_business_units
    op.create_table(
        'app_business_units',
        sa.Column('business_unit_id', postgresql.UUID(as_uuid=True), server_default=sa.text('gen_random_uuid()'), nullable=False),
        sa.Column('name', sa.String(100), nullable=False),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.PrimaryKeyConstraint('business_unit_id'),
        sa.UniqueConstraint('name', name='uq_app_business_units_name'),
    )

    # TABLE: app_tags
    op.create_table(
        'app_tags',
        sa.Column('tag_id', postgresql.UUID(as_uuid=True), server_default=sa.text('gen_random_uuid()'), nullable=False),
        sa.Column('name', sa.String(100), nullable=False),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.PrimaryKeyConstraint('tag_id'),
        sa.UniqueConstraint('name', name='uq_app_tags_name'),
    )

    # TABLE: app_llms
    op.create_table(
        'app_llms',
        sa.Column('llm_id', postgresql.UUID(as_uuid=True), server_default=sa.text('gen_random_uuid()'), nullable=False),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('provider', sa.String(50), nullable=True),
        sa.Column('model_name', sa.String(255), nullable=True),
        sa.Column('base_url', sa.String(512), nullable=True),
        sa.Column('api_key_env', sa.String(100), nullable=True),
        sa.Column('is_default', sa.Boolean(), server_default=sa.text('FALSE'), nullable=False),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.PrimaryKeyConstraint('llm_id'),
    )

    # TABLE: app_data_sources
    op.create_table(
        'app_data_sources',
        sa.Column('data_source_id', sa.String(255), nullable=False),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('kind', sa.String(50), nullable=True),
        sa.Column('metadata', postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'{}'::jsonb"), nullable=True),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.PrimaryKeyConstraint('data_source_id'),
    )

    # ========================================================================
    # CORE TABLE: applications
    # ========================================================================
    op.create_table(
        'applications',
        sa.Column('application_id', postgresql.UUID(as_uuid=True), server_default=sa.text('gen_random_uuid()'), nullable=False),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('status', sa.String(20), server_default=sa.text("'draft'"), nullable=False),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('available_for_all_users', sa.Boolean(), server_default=sa.text('FALSE'), nullable=False),
        sa.Column('upload_enabled', sa.Boolean(), server_default=sa.text('FALSE'), nullable=False),
        sa.Column('welcome_prompt', sa.Text(), nullable=True),
        sa.Column('disclaimer', sa.Text(), nullable=True),
        sa.Column('sorry_message', sa.Text(), nullable=True),
        sa.Column('starter_questions', postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'[]'::jsonb"), nullable=True),
        sa.Column('persona_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('persona_text', sa.Text(), nullable=True),
        sa.Column('guardrail_text', sa.Text(), nullable=True),
        sa.Column('logo_url', sa.String(512), nullable=True),
        sa.Column('is_deleted', sa.Boolean(), server_default=sa.text('FALSE'), nullable=False),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.Column('updated_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.PrimaryKeyConstraint('application_id'),
        sa.ForeignKeyConstraint(['user_id'], ['users.user_id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['persona_id'], ['app_personas.persona_id']),
        sa.CheckConstraint("status IN ('draft', 'published', 'error')", name='valid_app_status'),
    )

    # Partial indexes on applications
    op.create_index(
        'idx_applications_user',
        'applications',
        ['user_id'],
        unique=False,
        postgresql_where=sa.text('is_deleted = FALSE'),
    )
    op.create_index(
        'idx_applications_status',
        'applications',
        ['user_id', 'status'],
        unique=False,
        postgresql_where=sa.text('is_deleted = FALSE'),
    )

    # Trigger: auto-update applications.updated_at (reuses existing function from migration 001)
    op.execute("""
        CREATE TRIGGER update_applications_updated_at
        BEFORE UPDATE ON applications
        FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
    """)

    # ========================================================================
    # ASSOCIATION / LINK TABLES
    # ========================================================================

    # TABLE: application_llm_links
    op.create_table(
        'application_llm_links',
        sa.Column('application_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('llm_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('role', sa.String(20), server_default=sa.text("'general'"), nullable=False),
        sa.PrimaryKeyConstraint('application_id', 'llm_id'),
        sa.ForeignKeyConstraint(['application_id'], ['applications.application_id'], ondelete='CASCADE'),
    )

    # TABLE: application_skill_links
    op.create_table(
        'application_skill_links',
        sa.Column('application_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('skill_type', sa.String(10), nullable=False),
        sa.Column('skill_id', sa.String(255), nullable=False),
        sa.Column('skill_name', sa.String(255), nullable=True),
        sa.Column('skill_description', sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint('application_id', 'skill_type', 'skill_id'),
        sa.ForeignKeyConstraint(['application_id'], ['applications.application_id'], ondelete='CASCADE'),
    )

    # TABLE: application_data_source_links
    op.create_table(
        'application_data_source_links',
        sa.Column('application_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('data_source_id', sa.String(255), nullable=False),
        sa.PrimaryKeyConstraint('application_id', 'data_source_id'),
        sa.ForeignKeyConstraint(['application_id'], ['applications.application_id'], ondelete='CASCADE'),
    )

    # TABLE: application_designation_links
    op.create_table(
        'application_designation_links',
        sa.Column('application_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('designation_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.PrimaryKeyConstraint('application_id', 'designation_id'),
        sa.ForeignKeyConstraint(['application_id'], ['applications.application_id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['designation_id'], ['app_designations.designation_id']),
    )

    # TABLE: application_business_unit_links
    op.create_table(
        'application_business_unit_links',
        sa.Column('application_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('business_unit_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.PrimaryKeyConstraint('application_id', 'business_unit_id'),
        sa.ForeignKeyConstraint(['application_id'], ['applications.application_id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['business_unit_id'], ['app_business_units.business_unit_id']),
    )

    # TABLE: application_tag_links
    op.create_table(
        'application_tag_links',
        sa.Column('application_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('tag_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.PrimaryKeyConstraint('application_id', 'tag_id'),
        sa.ForeignKeyConstraint(['application_id'], ['applications.application_id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['tag_id'], ['app_tags.tag_id']),
    )

    # TABLE: application_guardrail_links
    op.create_table(
        'application_guardrail_links',
        sa.Column('application_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('guardrail_category_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.PrimaryKeyConstraint('application_id', 'guardrail_category_id'),
        sa.ForeignKeyConstraint(['application_id'], ['applications.application_id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['guardrail_category_id'], ['app_guardrail_categories.guardrail_category_id']),
    )

    # ========================================================================
    # CHAT TABLES
    # ========================================================================

    # TABLE: application_chat_sessions
    op.create_table(
        'application_chat_sessions',
        sa.Column('chat_session_id', postgresql.UUID(as_uuid=True), server_default=sa.text('gen_random_uuid()'), nullable=False),
        sa.Column('application_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('title', sa.String(255), nullable=True),
        sa.Column('conversation_state', sa.String(30), server_default=sa.text("'awaiting_input'"), nullable=False),
        sa.Column('llm_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('is_closed', sa.Boolean(), server_default=sa.text('FALSE'), nullable=False),
        sa.Column('context_data', postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'{}'::jsonb"), nullable=True),
        sa.Column('started_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.Column('updated_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.PrimaryKeyConstraint('chat_session_id'),
        sa.ForeignKeyConstraint(['application_id'], ['applications.application_id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['user_id'], ['users.user_id']),
        sa.CheckConstraint(
            "conversation_state IN ('awaiting_input', 'awaiting_clarification', 'executing')",
            name='valid_conversation_state',
        ),
    )

    op.create_index(
        'idx_app_chat_sessions_user',
        'application_chat_sessions',
        ['application_id', 'user_id'],
        unique=False,
    )

    # Trigger: auto-update application_chat_sessions.updated_at
    op.execute("""
        CREATE TRIGGER update_app_chat_sessions_updated_at
        BEFORE UPDATE ON application_chat_sessions
        FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
    """)

    # TABLE: application_chat_messages
    op.create_table(
        'application_chat_messages',
        sa.Column('message_id', postgresql.UUID(as_uuid=True), server_default=sa.text('gen_random_uuid()'), nullable=False),
        sa.Column('chat_session_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('role', sa.String(20), nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('enhanced_prompt', sa.Text(), nullable=True),
        sa.Column('execution_trace', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('guardrail_flags', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.PrimaryKeyConstraint('message_id'),
        sa.ForeignKeyConstraint(['chat_session_id'], ['application_chat_sessions.chat_session_id'], ondelete='CASCADE'),
    )

    op.create_index(
        'idx_app_chat_messages_session',
        'application_chat_messages',
        ['chat_session_id', 'created_at'],
        unique=False,
    )

    # ========================================================================
    # DOCUMENT / RAG TABLE
    # ========================================================================

    # TABLE: application_documents
    op.create_table(
        'application_documents',
        sa.Column('document_id', postgresql.UUID(as_uuid=True), server_default=sa.text('gen_random_uuid()'), nullable=False),
        sa.Column('application_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('chat_session_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('original_filename', sa.String(512), nullable=False),
        sa.Column('stored_filename', sa.String(512), nullable=False),
        sa.Column('public_url', sa.String(512), nullable=True),
        sa.Column('mime_type', sa.String(100), nullable=True),
        sa.Column('size_bytes', sa.BigInteger(), nullable=True),
        sa.Column('processing_status', sa.String(20), server_default=sa.text("'pending'"), nullable=False),
        sa.Column('chunk_count', sa.Integer(), server_default=sa.text('0'), nullable=False),
        sa.Column('metadata', postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'{}'::jsonb"), nullable=True),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.PrimaryKeyConstraint('document_id'),
        sa.ForeignKeyConstraint(['application_id'], ['applications.application_id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['chat_session_id'], ['application_chat_sessions.chat_session_id']),
        sa.CheckConstraint(
            "processing_status IN ('pending', 'processing', 'ready', 'failed')",
            name='valid_doc_processing_status',
        ),
    )

    # ========================================================================
    # EXECUTION TRACE TABLE (AUDIT TRAIL)
    # ========================================================================

    # TABLE: application_execution_traces
    op.create_table(
        'application_execution_traces',
        sa.Column('trace_id', postgresql.UUID(as_uuid=True), server_default=sa.text('gen_random_uuid()'), nullable=False),
        sa.Column('application_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('chat_session_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('message_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('user_message', sa.Text(), nullable=False),
        sa.Column('enhanced_prompt', sa.Text(), nullable=True),
        sa.Column('orchestrator_plan', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('execution_result', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('skills_invoked', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('guardrail_input', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('guardrail_output', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('total_duration_ms', sa.Integer(), nullable=True),
        sa.Column('status', sa.String(20), server_default=sa.text("'pending'"), nullable=False),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.PrimaryKeyConstraint('trace_id'),
        sa.ForeignKeyConstraint(['application_id'], ['applications.application_id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['chat_session_id'], ['application_chat_sessions.chat_session_id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['message_id'], ['application_chat_messages.message_id']),
        sa.CheckConstraint(
            "status IN ('pending', 'running', 'completed', 'failed')",
            name='valid_exec_trace_status',
        ),
    )

    op.create_index(
        'idx_app_exec_traces_session',
        'application_execution_traces',
        ['chat_session_id', 'created_at'],
        unique=False,
    )

    # ========================================================================
    # SEED DATA — Lookup tables
    # ========================================================================

    # Personas
    op.execute("""
        INSERT INTO app_personas (persona_id, name) VALUES
        (gen_random_uuid(), 'Support Agent'),
        (gen_random_uuid(), 'HR Assistant'),
        (gen_random_uuid(), 'Sales Assistant'),
        (gen_random_uuid(), 'Legal Advisor'),
        (gen_random_uuid(), 'Technical Expert');
    """)

    # Guardrail Categories
    op.execute("""
        INSERT INTO app_guardrail_categories (guardrail_category_id, name) VALUES
        (gen_random_uuid(), 'Compliance'),
        (gen_random_uuid(), 'PII'),
        (gen_random_uuid(), 'Safety');
    """)

    # Designations
    op.execute("""
        INSERT INTO app_designations (designation_id, name) VALUES
        (gen_random_uuid(), 'Executive'),
        (gen_random_uuid(), 'Manager'),
        (gen_random_uuid(), 'Individual Contributor'),
        (gen_random_uuid(), 'Intern');
    """)

    # Business Units
    op.execute("""
        INSERT INTO app_business_units (business_unit_id, name) VALUES
        (gen_random_uuid(), 'Operations'),
        (gen_random_uuid(), 'Human Resources'),
        (gen_random_uuid(), 'Sales'),
        (gen_random_uuid(), 'Engineering'),
        (gen_random_uuid(), 'Legal'),
        (gen_random_uuid(), 'Finance');
    """)

    # Tags
    op.execute("""
        INSERT INTO app_tags (tag_id, name) VALUES
        (gen_random_uuid(), 'Internal'),
        (gen_random_uuid(), 'External'),
        (gen_random_uuid(), 'Priority'),
        (gen_random_uuid(), 'Experimental');
    """)

    # LLMs
    op.execute("""
        INSERT INTO app_llms (llm_id, name, provider, model_name, base_url, api_key_env, is_default) VALUES
        (gen_random_uuid(), 'GPT-4o', 'openai', 'gpt-4o', 'https://api.openai.com/v1', 'OPENAI_API_KEY', FALSE),
        (gen_random_uuid(), 'GPT-4o Mini', 'openai', 'gpt-4o-mini', 'https://api.openai.com/v1', 'OPENAI_API_KEY', FALSE),
        (gen_random_uuid(), 'Claude Sonnet', 'anthropic', 'claude-sonnet-4-20250514', 'https://api.anthropic.com', 'ANTHROPIC_API_KEY', FALSE),
        (gen_random_uuid(), 'OpenRouter Free', 'openrouter', 'liquid/lfm-2.5-1.2b-instruct:free', 'https://openrouter.ai/api/v1', 'OPENROUTER_API_KEY', TRUE),
        (gen_random_uuid(), 'Ollama Local', 'ollama', 'llama3', 'http://localhost:11434/v1', NULL, FALSE);
    """)

    # Data Sources
    op.execute("""
        INSERT INTO app_data_sources (data_source_id, name, kind, metadata) VALUES
        ('mcp_filesystem', 'Local Filesystem', 'mcp', '{"description": "Access local files"}'::jsonb),
        ('mcp_web_search', 'Web Search', 'mcp', '{"description": "Search the web"}'::jsonb),
        ('api_rest', 'REST API', 'api', '{"description": "Connect to REST APIs"}'::jsonb);
    """)


def downgrade() -> None:
    # ========================================================================
    # DROP TRIGGERS
    # ========================================================================
    op.execute("DROP TRIGGER IF EXISTS update_app_chat_sessions_updated_at ON application_chat_sessions;")
    op.execute("DROP TRIGGER IF EXISTS update_applications_updated_at ON applications;")

    # ========================================================================
    # DROP TABLES (reverse dependency order)
    # ========================================================================

    # Audit trail (depends on chat messages, chat sessions, applications)
    op.drop_table('application_execution_traces')

    # Documents (depends on applications, chat sessions)
    op.drop_table('application_documents')

    # Chat messages (depends on chat sessions)
    op.drop_table('application_chat_messages')

    # Chat sessions (depends on applications, users)
    op.drop_table('application_chat_sessions')

    # Link tables (depend on applications + lookups)
    op.drop_table('application_guardrail_links')
    op.drop_table('application_tag_links')
    op.drop_table('application_business_unit_links')
    op.drop_table('application_designation_links')
    op.drop_table('application_data_source_links')
    op.drop_table('application_skill_links')
    op.drop_table('application_llm_links')

    # Core table (depends on users + app_personas)
    op.drop_table('applications')

    # Lookup / catalog tables (no dependencies)
    op.drop_table('app_data_sources')
    op.drop_table('app_llms')
    op.drop_table('app_tags')
    op.drop_table('app_business_units')
    op.drop_table('app_designations')
    op.drop_table('app_guardrail_categories')
    op.drop_table('app_personas')
