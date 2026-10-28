#!/bin/bash
set -e

echo "==> [Entrypoint] Zero-CLI Automated Container Bootstrap Starting..."

# 1. Ensure processed dataset and tactical profiles exist if raw data is present
if [ ! -f "/app/data/processed/processed_matches.csv" ]; then
    echo "==> [Entrypoint] processed_matches.csv missing; running data/preprocess.py..."
    python data/preprocess.py || echo "==> [Entrypoint] preprocess.py skipped or deferred."
fi

if [ ! -f "/app/rag/knowledge_base/team_tactics.json" ] && [ -f "/app/data/processed/processed_matches.csv" ]; then
    echo "==> [Entrypoint] team_tactics.json missing; running data/extract_tactics.py..."
    python data/extract_tactics.py || echo "==> [Entrypoint] extract_tactics.py skipped or deferred."
fi

# 2. Auto-export dynamic-batch ONNX GNN artifact if checkpoint exists and ONNX is missing
if [ ! -f "/app/models/onnx/tea_gnn.onnx" ] && [ -f "/app/models/checkpoints/tea_gnn_best.pt" ]; then
    echo "==> [Entrypoint] Exporting dynamic-batch ONNX GNN model (models/onnx/tea_gnn.onnx)..."
    python models/export_gnn.py --dynamic || echo "==> [Entrypoint] ONNX GNN export skipped or deferred."
fi

# 3. Seed PostgreSQL relational tables (matches, teams, match_corrections, team_overrides)
echo "==> [Entrypoint] Auto-seeding PostgreSQL matches & teams knowledge base..."
python rag/build_postgres_db.py || echo "==> [Entrypoint] PostgreSQL seed completed or skipped."

# 4. Seed Neo4j Knowledge Graph if KG_PROVIDER is neo4j and Neo4j is reachable
if [ "${KG_PROVIDER}" = "neo4j" ]; then
    echo "==> [Entrypoint] Checking & auto-seeding Neo4j Knowledge Graph..."
    python -c "
import os
try:
    from neo4j import GraphDatabase
    uri = os.getenv('NEO4J_URI', 'bolt://neo4j:7687')
    user = os.getenv('NEO4J_USER', 'neo4j')
    pw = os.getenv('NEO4J_PASSWORD', 'password')
    driver = GraphDatabase.driver(uri, auth=(user, pw))
    with driver.session() as s:
        cnt = s.run('MATCH (t:Team) RETURN count(t) AS c').single()['c']
    driver.close()
    if cnt == 0:
        import subprocess
        print('==> [Entrypoint] Neo4j has 0 Team nodes; running rag/build_neo4j_kg.py...')
        subprocess.run(['python', 'rag/build_neo4j_kg.py'], check=False)
    else:
        print(f'==> [Entrypoint] Neo4j already populated ({cnt} Team nodes).')
except Exception as e:
    print(f'==> [Entrypoint] Neo4j check deferred (FastAPI startup will retry): {e}')
" || true
fi

# 5. Seed pgvector HNSW index if VECTOR_PROVIDER is pgvector and table is empty
if [ "${VECTOR_PROVIDER}" = "pgvector" ]; then
    echo "==> [Entrypoint] Checking & auto-seeding pgvector index..."
    python -c "
import os
try:
    import psycopg2
    dsn = os.getenv('POSTGRES_DSN', 'postgresql://postgres:123@postgres:5432/football_rag')
    conn = psycopg2.connect(dsn)
    with conn.cursor() as cur:
        cur.execute(\"SELECT to_regclass('kb_vectors')\")
        exists = cur.fetchone()[0] is not None
        cnt = 0
        if exists:
            cur.execute('SELECT COUNT(*) FROM kb_vectors')
            cnt = cur.fetchone()[0]
    conn.close()
    if cnt == 0:
        import subprocess
        print('==> [Entrypoint] pgvector kb_vectors has 0 rows; running rag/build_pgvector_index.py...')
        subprocess.run(['python', 'rag/build_pgvector_index.py'], check=False)
    else:
        print(f'==> [Entrypoint] pgvector already populated ({cnt} vectors).')
except Exception as e:
    print(f'==> [Entrypoint] pgvector check deferred (FastAPI startup will retry): {e}')
" || true
fi

echo "==> [Entrypoint] Bootstrap complete. Launching main command: $@"
exec "$@"
