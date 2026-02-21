"""Change application_llm_links.llm_id from UUID to VARCHAR and add name column

Revision ID: 006
Revises: 005
Create Date: 2026-02-10

The LLM system uses string-based identifiers from JSON provider config files
(e.g. "ollama-qwen3-vl-8b", "gpt-4o") rather than UUIDs. This migration
changes the llm_id column from UUID to VARCHAR(255) to accept both UUID
strings and slug-style identifiers.

Also adds a 'name' column for denormalized LLM display name storage,
matching the pattern used by application_skill_links (skill_name).

Target Database: PostgreSQL 16
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '006'
down_revision: Union[str, None] = '005'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Drop the composite PK before altering the column type
    op.drop_constraint(
        'application_llm_links_pkey',
        'application_llm_links',
        type_='primary',
    )

    # Change llm_id from UUID to VARCHAR(255)
    op.alter_column(
        'application_llm_links',
        'llm_id',
        existing_type=sa.dialects.postgresql.UUID(as_uuid=True),
        type_=sa.String(255),
        existing_nullable=False,
        postgresql_using='llm_id::text',
    )

    # Recreate the composite PK with the new column type
    op.create_primary_key(
        'application_llm_links_pkey',
        'application_llm_links',
        ['application_id', 'llm_id'],
    )

    # Add name column for denormalized LLM display name
    op.add_column(
        'application_llm_links',
        sa.Column('name', sa.String(255), nullable=True),
    )


def downgrade() -> None:
    # Remove name column
    op.drop_column('application_llm_links', 'name')

    # Drop PK before altering back
    op.drop_constraint(
        'application_llm_links_pkey',
        'application_llm_links',
        type_='primary',
    )

    # Change llm_id back to UUID
    op.alter_column(
        'application_llm_links',
        'llm_id',
        existing_type=sa.String(255),
        type_=sa.dialects.postgresql.UUID(as_uuid=True),
        existing_nullable=False,
        postgresql_using='llm_id::uuid',
    )

    # Recreate original composite PK
    op.create_primary_key(
        'application_llm_links_pkey',
        'application_llm_links',
        ['application_id', 'llm_id'],
    )
