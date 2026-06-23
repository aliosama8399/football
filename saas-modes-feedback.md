# Football RAG SaaS — FastAPI Web Service (Complete Plan)

Build a production-grade FastAPI service that exposes the existing RAG orchestration system (GNN Expert 1 + Fine-tuned LLM Expert 2 + PostgreSQL + FAISS) as a RESTful and GraphQL web service.

---

## 🤖 Applying knowledge of `@[project-planner]`, `@[database-architect]`, and `@[backend-specialist]`...

---

## Overview & Goals
The goal is to wrap the existing Football RAG system into a web-accessible SaaS API. It supports:
1. **Unified Database Schema**: User accounts, chat contexts, messages, and feedback stored directly in the default (`public`) schema of PostgreSQL.
2. **Supervisor-led Onboarding**: Supervisors create account records first, yielding an activation token. Users activate by setting their password.
3. **Dual Chat Modes**: 
   - *Mode 1 (Prediction & Tactics)*: Auto-runs live GNN + LLM analysis and structures predictions.
   - *Mode 2 (General Football Chat)*: Open conversation with vector search grounding.
4. **Supervisor Feedback Loop**: Users request tactic edits or prediction overrides. A supervisor reviews them and approves them, triggering automatic database updates (updating `public.teams` tactics or registering prediction overrides).
5. **Strawberry GraphQL Endpoint**: Exposes a GraphQL query interface over the Neo4j Graph DB to retrieve tactical relationships.
6. **Async Task Execution**: Wraps block-heavy RAG/GNN processes using Python's `ThreadPoolExecutor`.

---

## Tech Stack & Architecture

- **Web Framework**: FastAPI (asyncio)
- **GraphQL**: Strawberry GraphQL (code-first schema definition)
- **Database 1 (SQL)**: PostgreSQL (using SQLAlchemy with `asyncpg` driver)
- **Database 2 (Graph)**: Neo4j (using official `neo4j` Python driver)
- **Security**: `passlib[bcrypt]` (password hashing), `python-jose` (JWT)
- **Concurrency**: `ThreadPoolExecutor` for execution of synchronous RAG pipeline calls

---

## File Structure

```
football/
├── api/
│   ├── __init__.py
│   ├── main.py                       # App startup, routing, lifespans, GraphQL mount
│   ├── config.py                     # Pydantic Settings
│   ├── database.py                   # Postgres SQLAlchemy models & connection
│   ├── graph_db.py                   # Neo4j connection helper
│   ├── auth.py                       # JWT generation & password hashing (bcrypt)
│   ├── async_rag.py                  # ThreadPool wrapper around FootballRAGSystem
│   ├── schemas.py                    # Pydantic v2 schemas
│   ├── dependencies.py               # Dependency injection helpers
│   ├── repositories/                 # Decoupled Repository Pattern (SOLID)
│   │   ├── __init__.py
│   │   ├── user_repo.py              # PostgreSQL: users operations
│   │   ├── chat_repo.py              # PostgreSQL: conversations & messages
│   │   ├── feedback_repo.py          # PostgreSQL: feedback & prediction overrides
│   │   └── graph_repo.py             # Neo4j: Cypher queries for teams & tactics
│   ├── graphql/                      # Strawberry GraphQL schema
│   │   ├── __init__.py
│   │   ├── schema.py                 # Strawberry Schema definition
│   │   ├── types.py                  # Strawberry type mappings
│   │   └── resolvers.py              # Graph queries resolvers
│   └── routes/
│       ├── __init__.py
│       ├── auth.py                   # User activation & login
│       ├── chat.py                   # Chat sessions (Prediction vs General)
│       ├── predictions.py            # live predictions / overrides check
│       ├── feedback.py               # User feedback submission
│       └── supervisor.py             # Supervisor onboarding & approvals
```

---

## Detailed Task Breakdown

### Task 1: Setup Configurations and Dependencies
* **Description**: Create `requirements-api.txt` and Pydantic `Settings` class in `api/config.py`. Expose variables for Postgres, Neo4j, JWT secret, and default admin credentials.
* **Agent / Skill**: `backend-specialist` / `python-patterns`
* **INPUT**: Environment configurations.
* **OUTPUT**: `requirements-api.txt` and `api/config.py`.
* **VERIFY**: Run `pip install -r requirements-api.txt` and import `Settings` in python to confirm default parameters load.

---

### Task 2: PostgreSQL Schema and SQLAlchemy Models
* **Description**: Create async SQLAlchemy connection utilities and define ORM models mapping to `users`, `conversations`, `messages`, `prediction_overrides`, and `feedback` in the default schema. Write database startup triggers to auto-create tables if missing.
* **Agent / Skill**: `database-architect` / `database-design`
* **INPUT**: SQL table specifications.
* **OUTPUT**: `api/database.py`.
* **VERIFY**: Start a Python shell, run db setup functions, and verify tables are created in the PostgreSQL database.

---

### Task 3: Neo4j Driver Connection Utility
* **Description**: Create `api/graph_db.py` to maintain a thread-safe Neo4j driver connection instance. Configure shutdown handlers to close the driver.
* **Agent / Skill**: `database-architect` / `database-design`
* **INPUT**: Neo4j credentials from configurations.
* **OUTPUT**: `api/graph_db.py`.
* **VERIFY**: Call connection method and verify a session can run a simple `RETURN 1` query.

---

### Task 4: Repository Layer (SOLID Separation)
* **Description**: Implement repositories under `api/repositories/` to separate data access concerns:
  - `UserRepository` & `ChatRepository` (Postgres operations)
  - `FeedbackRepository` (Postgres operations)
  - `TeamGraphRepository` (Neo4j Cypher queries for teams, tactics, and H2H)
* **Agent / Skill**: `backend-specialist` / `clean-code`
* **INPUT**: Postgres models and Neo4j driver connection.
* **OUTPUT**: `api/repositories/*.py`.
* **VERIFY**: Unit tests importing repositories and asserting mock queries return successfully.

---

### Task 5: Auth Layer & Onboarding Routes
* **Description**: Build password hashing (`bcrypt`), JWT token generation, route protection middleware (`get_current_user`, `require_supervisor`). Expose:
  - `POST /api/v1/supervisor/users` (creates pending user, returns token)
  - `POST /api/v1/auth/activate` (accepts token, sets password, activates)
  - `POST /api/v1/auth/login` (generates JWT)
* **Agent / Skill**: `security-auditor` / `vulnerability-scanner`
* **INPUT**: Password criteria and token configurations.
* **OUTPUT**: `api/auth.py` and `api/routes/auth.py`.
* **VERIFY**: Onboard a pending user, activate them with a password, and verify password hash is recorded and login works.

---

### Task 6: Strawberry GraphQL Routing
* **Description**: Define Strawberry GraphQL schemas and types representing the Neo4j team nodes and match relationships. Connect queries to the `TeamGraphRepository`. Mount the router onto `/graphql`.
* **Agent / Skill**: `backend-specialist` / `api-patterns`
* **INPUT**: Graph schema fields.
* **OUTPUT**: `api/graphql/*.py` and mounting in `api/main.py`.
* **VERIFY**: Open `/graphql` in a browser and query team tactical details using the GraphiQL web console.

---

### Task 7: Dual Chat Modes & Async RAG Execution
* **Description**: Wrap the synchronous RAG pipeline inside `api/async_rag.py`. Implement the `/api/v1/chat/conversations` endpoints:
  - For `mode = 'prediction'`: Inject prompt emphasizing live GNN predictions and tactical report context.
  - For `mode = 'general'`: Trigger generic grounding.
* **Agent / Skill**: `backend-specialist` / `python-patterns`
* **INPUT**: Synchronous RAG orchestrator.
* **OUTPUT**: `api/async_rag.py` and `api/routes/chat.py`.
* **VERIFY**: Issue chat messages to both prediction and general sessions, confirming correct prompts are loaded and history context is maintained.

---

### Task 8: Predictions Routing (with overrides)
* **Description**: Implement `POST /api/v1/predictions`. When requested, query `prediction_overrides` table. If a supervisor override exists, return it immediately; otherwise, run GNN (Expert 1) + LLM (Expert 2) prediction logic.
* **Agent / Skill**: `backend-specialist` / `api-patterns`
* **INPUT**: Teams to predict.
* **OUTPUT**: `api/routes/predictions.py`.
* **VERIFY**: Query a prediction for a match and assert live predictions run. Then insert an override, request prediction again, and verify the override is returned.

---

### Task 9: Feedback & Supervisor Approval Loop
* **Description**: Create feedback submission routes. Build the supervisor panel route `POST /api/v1/supervisor/feedback/{id}/review`:
  - If type is `modify_tactics` and approved: Update public `teams` table details directly.
  - If type is `update_prediction` and approved: Write to `prediction_overrides`.
* **Agent / Skill**: `backend-specialist` / `database-design`
* **INPUT**: Staged feedback records.
* **OUTPUT**: `api/routes/feedback.py` and `api/routes/supervisor.py`.
* **VERIFY**: Submit a feedback request, approve it as supervisor, and verify changes are applied automatically to the database.

---

## Phase X: Verification Checklist

1. Run security validation:
   ```bash
   python .agent/skills/vulnerability-scanner/scripts/security_scan.py .
   ```
2. Start API service:
   ```bash
   uvicorn api.main:app --reload
   ```
3. Execute onboarding tests:
   - Onboard pending user → activate with hashed password → login and confirm JWT generation.
4. Execute dual-mode test queries and assert RAG returns.
5. Run GraphQL query from `/graphql` playground.
6. Verify feedback approval triggers immediate updates to database tactical records.
7. Run complete project checklists.
