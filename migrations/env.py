import os
import sys
from logging.config import fileConfig

from sqlalchemy import engine_from_config
from sqlalchemy import pool

from alembic import context

# Add project root to sys.path so `api` package is importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.config import settings
from api.database import Base

# This is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Override the connection URL from application config.
# FOOTBALL_ALEMBIC_URL env var allows pointing at a different database
# (e.g. a scratch DB for baseline generation).
config.set_main_option(
    "sqlalchemy.url",
    os.environ.get("FOOTBALL_ALEMBIC_URL", settings.postgres_dsn),
)

# Interpret the config file for Python logging.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Target metadata = all ORM models in api/database.py.
target_metadata = Base.metadata

# Non-ORM tables (raw-SQL, created by rag/build_postgres_db.py) must never be
# touched by Alembic: no drops, no creates, no diffs.
NON_ORM_TABLES = {"matches", "teams"}


def include_object(object, name, type_, reflected, compare_to):
    """Only manage ORM tables (and their indexes/constraints)."""
    if type_ == "table":
        if reflected and name not in target_metadata.tables:
            return False
        if name in NON_ORM_TABLES:
            return False
    return True


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        include_object=include_object,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            include_object=include_object,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
