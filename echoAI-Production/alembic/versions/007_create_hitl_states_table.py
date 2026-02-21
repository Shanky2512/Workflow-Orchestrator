"""Create hitl_states table for HITL node DB persistence

Revision ID: 007
Revises: 006
Create Date: 2026-02-17

Adds a lightweight hitl_states table used by HITLDBManager to persist
HITL node state (waiting, approved, rejected, modified, deferred) to
the database. This supplements the file-based persistence fallback and
is separate from the hitl_checkpoints table (which stores full
checkpoint data for the review UI).

Target Database: PostgreSQL 16
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = '007'
down_revision: Union[str, None] = '006'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'hitl_states',
        sa.Column('run_id', sa.String(255), nullable=False),
        sa.Column('state', sa.String(50), nullable=False),
        sa.Column('data', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column(
            'updated_at',
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text('CURRENT_TIMESTAMP'),
            nullable=True,
        ),
        sa.PrimaryKeyConstraint('run_id'),
    )

    op.create_index('idx_hitl_states_state', 'hitl_states', ['state'], unique=False)


def downgrade() -> None:
    op.drop_index('idx_hitl_states_state', table_name='hitl_states')
    op.drop_table('hitl_states')
