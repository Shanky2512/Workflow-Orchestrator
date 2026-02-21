"""Initial schema for EchoAI database

Revision ID: 001
Revises:
Create Date: 2026-02-03

Creates all tables for EchoAI multi-tenant workflow orchestration platform:
- users: User accounts
- agents: AI agent definitions (simplified, stores complete JSON)
- workflows: Workflow definitions (simplified, stores complete JSON)
- workflow_versions: Immutable workflow version history
- chat_sessions: User chat sessions with embedded messages
- session_tool_configs: Tool configuration overrides per session
- tools: User-owned tool definitions
- executions: Workflow execution runs
- hitl_checkpoints: Human-in-the-loop checkpoint state

Target Database: PostgreSQL 16
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '001'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ========================================================================
    # TABLE: users
    # ========================================================================
    op.create_table(
        'users',
        sa.Column('user_id', postgresql.UUID(as_uuid=True), server_default=sa.text('gen_random_uuid()'), nullable=False),
        sa.Column('email', sa.String(255), nullable=False),
        sa.Column('display_name', sa.String(255), nullable=True),
        sa.Column('avatar_url', sa.String(500), nullable=True),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=True),
        sa.Column('updated_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=True),
        sa.Column('last_login_at', sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column('metadata', postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'{}'::jsonb"), nullable=True),
        sa.Column('is_active', sa.Boolean(), server_default=sa.text('TRUE'), nullable=True),
        sa.PrimaryKeyConstraint('user_id'),
        sa.UniqueConstraint('email', name='uq_users_email')
    )

    # Users indexes
    op.create_index('idx_users_email', 'users', ['email'], unique=False)
    op.create_index('idx_users_active', 'users', ['is_active'], unique=False, postgresql_where=sa.text('is_active = TRUE'))

    # ========================================================================
    # TABLE: agents (SIMPLIFIED - stores complete JSON in definition)
    # ========================================================================
    op.create_table(
        'agents',
        sa.Column('agent_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('definition', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('source_workflow_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=True),
        sa.Column('updated_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=True),
        sa.Column('is_deleted', sa.Boolean(), server_default=sa.text('FALSE'), nullable=True),
        sa.PrimaryKeyConstraint('agent_id'),
        sa.ForeignKeyConstraint(['user_id'], ['users.user_id'], ondelete='CASCADE'),
        sa.UniqueConstraint('user_id', 'name', name='unique_agent_name_per_user')
    )

    # Agents indexes
    op.create_index('idx_agents_user', 'agents', ['user_id'], unique=False)
    op.create_index('idx_agents_user_active', 'agents', ['user_id'], unique=False, postgresql_where=sa.text('is_deleted = FALSE'))
    op.create_index('idx_agents_source_workflow', 'agents', ['source_workflow_id'], unique=False)
    # Full-text search index on agent name
    op.execute("CREATE INDEX idx_agents_name_search ON agents USING gin(to_tsvector('english', name))")

    # ========================================================================
    # TABLE: workflows (SIMPLIFIED - stores complete JSON in definition)
    # ========================================================================
    op.create_table(
        'workflows',
        sa.Column('workflow_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('status', sa.String(20), server_default=sa.text("'draft'"), nullable=False),
        sa.Column('definition', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=True),
        sa.Column('updated_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=True),
        sa.Column('is_deleted', sa.Boolean(), server_default=sa.text('FALSE'), nullable=True),
        sa.PrimaryKeyConstraint('workflow_id'),
        sa.ForeignKeyConstraint(['user_id'], ['users.user_id'], ondelete='CASCADE'),
        sa.CheckConstraint("status IN ('draft', 'validated', 'final', 'archived')", name='valid_status')
    )

    # Workflows indexes
    op.create_index('idx_workflows_user', 'workflows', ['user_id'], unique=False)
    op.create_index('idx_workflows_user_status', 'workflows', ['user_id', 'status'], unique=False)
    op.create_index('idx_workflows_user_active', 'workflows', ['user_id'], unique=False, postgresql_where=sa.text('is_deleted = FALSE'))
    # Full-text search index on workflow name
    op.execute("CREATE INDEX idx_workflows_name_search ON workflows USING gin(to_tsvector('english', name))")

    # ========================================================================
    # TABLE: workflow_versions
    # ========================================================================
    op.create_table(
        'workflow_versions',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('workflow_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('version', sa.String(20), nullable=False),
        sa.Column('definition', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('status', sa.String(20), server_default=sa.text("'final'"), nullable=False),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=True),
        sa.Column('created_by', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('notes', sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['workflow_id'], ['workflows.workflow_id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['created_by'], ['users.user_id']),
        sa.UniqueConstraint('workflow_id', 'version', name='unique_workflow_version'),
        sa.CheckConstraint("status IN ('final', 'archived')", name='valid_version_status')
    )

    # Workflow versions indexes
    op.create_index('idx_workflow_versions_workflow', 'workflow_versions', ['workflow_id'], unique=False)

    # ========================================================================
    # TABLE: chat_sessions (with embedded messages JSONB)
    # ========================================================================
    op.create_table(
        'chat_sessions',
        sa.Column('session_id', postgresql.UUID(as_uuid=True), server_default=sa.text('gen_random_uuid()'), nullable=False),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('title', sa.String(255), server_default=sa.text("'New Chat'"), nullable=False),
        sa.Column('context_type', sa.String(20), server_default=sa.text("'general'"), nullable=False),
        sa.Column('context_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('workflow_mode', sa.String(20), nullable=True),
        sa.Column('workflow_version', sa.String(20), nullable=True),
        sa.Column('messages', postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'[]'::jsonb"), nullable=True),
        sa.Column('selected_tool_ids', postgresql.ARRAY(postgresql.UUID(as_uuid=True)), server_default=sa.text("'{}'"), nullable=True),
        sa.Column('context_data', postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'{}'::jsonb"), nullable=True),
        sa.Column('variables', postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'{}'::jsonb"), nullable=True),
        sa.Column('state_schema', postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'{}'::jsonb"), nullable=True),
        sa.Column('run_ids', postgresql.ARRAY(postgresql.UUID(as_uuid=True)), server_default=sa.text("'{}'"), nullable=True),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=True),
        sa.Column('last_activity', sa.TIMESTAMP(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=True),
        sa.Column('is_deleted', sa.Boolean(), server_default=sa.text('FALSE'), nullable=True),
        sa.PrimaryKeyConstraint('session_id'),
        sa.ForeignKeyConstraint(['user_id'], ['users.user_id'], ondelete='CASCADE'),
        sa.CheckConstraint("context_type IN ('general', 'workflow', 'agent', 'workflow_design')", name='valid_context_type')
    )

    # Chat sessions indexes
    op.create_index('idx_sessions_user', 'chat_sessions', ['user_id'], unique=False)
    op.create_index('idx_sessions_user_active', 'chat_sessions', ['user_id', 'last_activity'], unique=False, postgresql_where=sa.text('is_deleted = FALSE'))
    op.create_index('idx_sessions_context', 'chat_sessions', ['context_type', 'context_id'], unique=False)

    # ========================================================================
    # TABLE: tools
    # ========================================================================
    op.create_table(
        'tools',
        sa.Column('tool_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('tool_type', sa.String(20), server_default=sa.text("'local'"), nullable=False),
        sa.Column('definition', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('status', sa.String(20), server_default=sa.text("'active'"), nullable=True),
        sa.Column('version', sa.String(20), server_default=sa.text("'1.0'"), nullable=True),
        sa.Column('tags', postgresql.ARRAY(sa.Text()), server_default=sa.text("'{}'"), nullable=True),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=True),
        sa.Column('updated_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=True),
        sa.Column('is_deleted', sa.Boolean(), server_default=sa.text('FALSE'), nullable=True),
        sa.PrimaryKeyConstraint('tool_id'),
        sa.ForeignKeyConstraint(['user_id'], ['users.user_id'], ondelete='CASCADE'),
        sa.CheckConstraint("tool_type IN ('local', 'mcp', 'api', 'crewai', 'custom')", name='valid_tool_type'),
        sa.CheckConstraint("status IN ('active', 'deprecated', 'disabled')", name='valid_tool_status')
    )

    # Tools indexes
    op.create_index('idx_tools_user', 'tools', ['user_id'], unique=False)
    op.create_index('idx_tools_user_active', 'tools', ['user_id'], unique=False, postgresql_where=sa.text("is_deleted = FALSE AND status = 'active'"))
    op.create_index('idx_tools_type', 'tools', ['tool_type'], unique=False)
    op.create_index('idx_tools_tags', 'tools', ['tags'], unique=False, postgresql_using='gin')

    # ========================================================================
    # TABLE: session_tool_configs
    # ========================================================================
    op.create_table(
        'session_tool_configs',
        sa.Column('session_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('tool_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('config_overrides', postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'{}'::jsonb"), nullable=True),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=True),
        sa.Column('updated_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=True),
        sa.PrimaryKeyConstraint('session_id', 'tool_id'),
        sa.ForeignKeyConstraint(['session_id'], ['chat_sessions.session_id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['tool_id'], ['tools.tool_id'], ondelete='CASCADE')
    )

    # Session tool configs indexes
    op.create_index('idx_session_tool_configs_session', 'session_tool_configs', ['session_id'], unique=False)
    op.create_index('idx_session_tool_configs_tool', 'session_tool_configs', ['tool_id'], unique=False)

    # ========================================================================
    # TABLE: executions
    # ========================================================================
    op.create_table(
        'executions',
        sa.Column('run_id', postgresql.UUID(as_uuid=True), server_default=sa.text('gen_random_uuid()'), nullable=False),
        sa.Column('workflow_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('session_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('execution_mode', sa.String(20), nullable=False),
        sa.Column('workflow_version', sa.String(20), nullable=True),
        sa.Column('status', sa.String(30), server_default=sa.text("'queued'"), nullable=False),
        sa.Column('input_payload', postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'{}'::jsonb"), nullable=True),
        sa.Column('output', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('started_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=True),
        sa.Column('completed_at', sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column('duration_ms', sa.Integer(), nullable=True),
        sa.Column('agent_count', sa.Integer(), nullable=True),
        sa.PrimaryKeyConstraint('run_id'),
        sa.ForeignKeyConstraint(['workflow_id'], ['workflows.workflow_id']),
        sa.ForeignKeyConstraint(['user_id'], ['users.user_id']),
        sa.ForeignKeyConstraint(['session_id'], ['chat_sessions.session_id']),
        sa.CheckConstraint("execution_mode IN ('draft', 'test', 'final')", name='valid_exec_mode'),
        sa.CheckConstraint("status IN ('queued', 'running', 'hitl_waiting', 'hitl_approved', 'hitl_rejected', 'completed', 'failed', 'cancelled')", name='valid_exec_status')
    )

    # Executions indexes
    op.create_index('idx_executions_workflow', 'executions', ['workflow_id'], unique=False)
    op.create_index('idx_executions_user', 'executions', ['user_id'], unique=False)
    op.create_index('idx_executions_session', 'executions', ['session_id'], unique=False, postgresql_where=sa.text('session_id IS NOT NULL'))
    op.create_index('idx_executions_status', 'executions', ['status'], unique=False, postgresql_where=sa.text("status IN ('running', 'hitl_waiting')"))
    op.create_index('idx_executions_user_recent', 'executions', ['user_id', 'started_at'], unique=False)

    # ========================================================================
    # TABLE: hitl_checkpoints
    # ========================================================================
    op.create_table(
        'hitl_checkpoints',
        sa.Column('checkpoint_id', postgresql.UUID(as_uuid=True), server_default=sa.text('gen_random_uuid()'), nullable=False),
        sa.Column('run_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('status', sa.String(30), server_default=sa.text("'waiting_for_human'"), nullable=False),
        sa.Column('blocked_at', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('agent_output', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('tools_used', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('execution_metrics', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('state_snapshot', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('previous_decisions', postgresql.ARRAY(postgresql.JSONB(astext_type=sa.Text())), server_default=sa.text("'{}'"), nullable=True),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=True),
        sa.Column('resolved_at', sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column('resolved_by', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('resolution', sa.String(30), nullable=True),
        sa.Column('resolution_data', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.PrimaryKeyConstraint('checkpoint_id'),
        sa.ForeignKeyConstraint(['run_id'], ['executions.run_id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['resolved_by'], ['users.user_id']),
        sa.CheckConstraint("status IN ('waiting_for_human', 'approved', 'rejected', 'modified', 'deferred')", name='valid_hitl_status'),
        sa.CheckConstraint("resolution IN ('approve', 'reject', 'modify', 'defer', 'rerun') OR resolution IS NULL", name='valid_resolution')
    )

    # HITL checkpoints indexes
    op.create_index('idx_hitl_run', 'hitl_checkpoints', ['run_id'], unique=False)
    op.create_index('idx_hitl_pending', 'hitl_checkpoints', ['status'], unique=False, postgresql_where=sa.text("status = 'waiting_for_human'"))

    # ========================================================================
    # DATABASE FUNCTIONS
    # ========================================================================

    # Function: update_updated_at_column
    op.execute("""
        CREATE OR REPLACE FUNCTION update_updated_at_column()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at = CURRENT_TIMESTAMP;
            RETURN NEW;
        END;
        $$ language 'plpgsql';
    """)

    # Function: update_session_last_activity
    op.execute("""
        CREATE OR REPLACE FUNCTION update_session_last_activity()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.last_activity = CURRENT_TIMESTAMP;
            RETURN NEW;
        END;
        $$ language 'plpgsql';
    """)

    # ========================================================================
    # TRIGGERS
    # ========================================================================

    # Trigger: update users.updated_at
    op.execute("""
        CREATE TRIGGER update_users_updated_at
        BEFORE UPDATE ON users
        FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
    """)

    # Trigger: update agents.updated_at
    op.execute("""
        CREATE TRIGGER update_agents_updated_at
        BEFORE UPDATE ON agents
        FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
    """)

    # Trigger: update workflows.updated_at
    op.execute("""
        CREATE TRIGGER update_workflows_updated_at
        BEFORE UPDATE ON workflows
        FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
    """)

    # Trigger: update tools.updated_at
    op.execute("""
        CREATE TRIGGER update_tools_updated_at
        BEFORE UPDATE ON tools
        FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
    """)

    # Trigger: update session_tool_configs.updated_at
    op.execute("""
        CREATE TRIGGER update_session_tool_configs_updated_at
        BEFORE UPDATE ON session_tool_configs
        FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
    """)

    # Trigger: update chat_sessions.last_activity when messages are updated
    op.execute("""
        CREATE TRIGGER update_session_activity
        BEFORE UPDATE OF messages ON chat_sessions
        FOR EACH ROW EXECUTE FUNCTION update_session_last_activity();
    """)


def downgrade() -> None:
    # ========================================================================
    # DROP TRIGGERS
    # ========================================================================
    op.execute("DROP TRIGGER IF EXISTS update_session_activity ON chat_sessions;")
    op.execute("DROP TRIGGER IF EXISTS update_session_tool_configs_updated_at ON session_tool_configs;")
    op.execute("DROP TRIGGER IF EXISTS update_tools_updated_at ON tools;")
    op.execute("DROP TRIGGER IF EXISTS update_workflows_updated_at ON workflows;")
    op.execute("DROP TRIGGER IF EXISTS update_agents_updated_at ON agents;")
    op.execute("DROP TRIGGER IF EXISTS update_users_updated_at ON users;")

    # ========================================================================
    # DROP FUNCTIONS
    # ========================================================================
    op.execute("DROP FUNCTION IF EXISTS update_session_last_activity();")
    op.execute("DROP FUNCTION IF EXISTS update_updated_at_column();")

    # ========================================================================
    # DROP INDEXES (GIN indexes need explicit drop)
    # ========================================================================
    op.execute("DROP INDEX IF EXISTS idx_agents_name_search;")
    op.execute("DROP INDEX IF EXISTS idx_workflows_name_search;")

    # ========================================================================
    # DROP TABLES (in reverse dependency order)
    # ========================================================================
    op.drop_table('hitl_checkpoints')
    op.drop_table('executions')
    op.drop_table('session_tool_configs')
    op.drop_table('tools')
    op.drop_table('chat_sessions')
    op.drop_table('workflow_versions')
    op.drop_table('workflows')
    op.drop_table('agents')
    op.drop_table('users')
