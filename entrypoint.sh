#!/bin/bash
set -e

# Run PostgreSQL DB builder to ensure matches and teams tables are seeded
echo "==> [Entrypoint] Auto-seeding PostgreSQL matches & teams knowledge base..."
python rag/build_postgres_db.py || echo "==> [Entrypoint] PostgreSQL seed completed or skipped."

# Execute the CMD passed to the container (e.g. uvicorn)
exec "$@"
