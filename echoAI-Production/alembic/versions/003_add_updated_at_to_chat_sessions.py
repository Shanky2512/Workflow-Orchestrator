"""Add updated_at column to chat_sessions

Revision ID: 003
Revises: 002
Create Date: 2026-02-04

The chat_sessions table was missing the updated_at column that is inherited
from BaseModel. This migration adds it for consistency.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '003'
down_revision: Union[str, None] = '002'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add updated_at column to chat_sessions
    op.add_column(
        'chat_sessions',
        sa.Column(
            'updated_at',
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text('CURRENT_TIMESTAMP'),
            nullable=True
        )
    )

    # Set existing rows to have updated_at = created_at
    op.execute("UPDATE chat_sessions SET updated_at = created_at WHERE updated_at IS NULL")

    # Now make it non-nullable
    op.alter_column('chat_sessions', 'updated_at', nullable=False)

    # Add trigger for auto-updating updated_at
    op.execute("""
        CREATE TRIGGER update_chat_sessions_updated_at
        BEFORE UPDATE ON chat_sessions
        FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
    """)


def downgrade() -> None:
    # Drop trigger
    op.execute("DROP TRIGGER IF EXISTS update_chat_sessions_updated_at ON chat_sessions;")

    # Drop column
    op.drop_column('chat_sessions', 'updated_at')
