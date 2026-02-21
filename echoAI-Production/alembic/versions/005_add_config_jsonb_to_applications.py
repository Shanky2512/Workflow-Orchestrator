"""Add config JSONB column to applications table

Revision ID: 005
Revises: 004
Create Date: 2026-02-08

Adds a consolidated config JSONB column to the applications table.
This column stores LLM bindings, skills, data sources, access control,
guardrails, and persona configuration in a single JSONB object,
reducing the need for complex joins across 7+ link tables.

The existing link tables are preserved for backward compatibility.
Both the JSONB column and link tables are kept in sync via dual-write
in the repository layer.

A GIN index is created on the config column for efficient JSONB queries.

Target Database: PostgreSQL 16
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '005'
down_revision: Union[str, None] = '004'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add config JSONB column with empty object default
    op.add_column(
        'applications',
        sa.Column(
            'config',
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=True,
        ),
    )

    # Create GIN index for efficient JSONB containment queries
    # e.g. WHERE config->'skills' @> '[{"skill_id": "..."}]'
    op.create_index(
        'idx_applications_config_gin',
        'applications',
        ['config'],
        postgresql_using='gin',
    )


def downgrade() -> None:
    op.drop_index('idx_applications_config_gin', table_name='applications')
    op.drop_column('applications', 'config')
