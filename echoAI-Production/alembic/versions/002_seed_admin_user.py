"""Seed admin user for EchoAI

Revision ID: 002
Revises: 001
Create Date: 2026-02-03

Creates the default system admin user for:
- Migrated data ownership (existing agents/workflows from JSON files)
- System-level operations
- Default user for development/testing

Admin User Details:
- user_id: 00000000-0000-0000-0000-000000000001 (fixed UUID)
- email: admin@echoai.local
- display_name: System Admin
- metadata: {"role": "admin", "migrated": true}
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '002'
down_revision: Union[str, None] = '001'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Insert default admin user with fixed UUID for consistent reference
    # This user will own all migrated data from JSON files
    op.execute("""
        INSERT INTO users (user_id, email, display_name, metadata, is_active)
        VALUES (
            '00000000-0000-0000-0000-000000000001',
            'admin@echoai.local',
            'System Admin',
            '{"role": "admin", "migrated": true, "system_user": true}'::jsonb,
            TRUE
        )
        ON CONFLICT (email) DO NOTHING;
    """)


def downgrade() -> None:
    # Remove the admin user
    # Note: This will CASCADE delete any agents/workflows owned by admin
    # In production, you would want to reassign ownership first
    op.execute("""
        DELETE FROM users
        WHERE user_id = '00000000-0000-0000-0000-000000000001';
    """)
