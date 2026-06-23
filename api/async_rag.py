import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional
from rag.rag_orchestrator import FootballRAGSystem

class AsyncRAGWrapper:
    """
    Asynchronous wrapper for FootballRAGSystem.
    Offloads blocking CPU-intensive and I/O-bound (sync psycopg2 / GNN loading / FAISS searches)
    RAG system operations to a ThreadPoolExecutor, ensuring the FastAPI event loop remains unblocked.
    """
    def __init__(self, rag_system: FootballRAGSystem):
        self.rag = rag_system
        self._executor = ThreadPoolExecutor(max_workers=4)

    async def query(self, question: str) -> str:
        """Asynchronously execute a general RAG query."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self.rag.query, question)

    async def predict_match(self, home_team: str, away_team: str) -> str:
        """Asynchronously execute the match prediction pipeline (GNN + LLM analysis)."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self.rag.predict_match, home_team, away_team)

    async def get_team_report(self, team_name: str) -> str:
        """Asynchronously generate a comprehensive team tactical report."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self.rag.get_team_report, team_name)

    async def compare_teams(self, team_a: str, team_b: str) -> str:
        """Asynchronously run detailed comparison analysis between two teams."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self.rag.compare_teams, team_a, team_b)

    async def get_gnn_prediction_structured(self, home_team: str, away_team: str) -> Optional[dict]:
        """Asynchronously run GNN (Expert 1) prediction returning structured outcome & probabilities."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self.rag.get_gnn_prediction_structured, home_team, away_team)

    def get_available_teams(self) -> List[str]:
        """Fetch available team list (runs fast in-memory, no offloading needed)."""
        return self.rag.get_available_teams()
