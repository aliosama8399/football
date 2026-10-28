"""
Football RAG Orchestrator
==========================
The single entry point for all RAG queries. Wires together:
  - KG Provider  (Neo4j or PostgreSQL)
  - Vector Provider (FAISS or pgvector)
  - LLM Provider (HuggingFace / Gemini / Ollama / etc.)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from concurrent.futures import ThreadPoolExecutor
import json
import logging
import re
import yaml
from pathlib import Path
from typing import Optional


from rag.providers.kg_provider         import get_kg_provider
from rag.providers.vector_provider     import get_vector_provider
from rag.providers.gnn_provider        import BasePredictionProvider  # ABC only
try:
    from rag.providers.gnn_provider_onnx import ONNXPredictionProvider as GNNPredictionProvider
except ImportError:
    from rag.providers.gnn_provider import GNNPredictionProvider  # fallback to torch if ONNX missing
from models.llm_providers              import get_llm_provider
from eval.langfuse_client             import trace_step

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_CFG_PATH = Path(__file__).parent.parent / "models" / "llm_config.yaml"


def _load_rag_cfg() -> dict:
    if not _CFG_PATH.exists():
        return {}
    with open(_CFG_PATH) as f:
        return yaml.safe_load(f).get("rag", {})


# ─────────────────────────────────────────────────────────────────────────────
# Team Name Extractor
# ─────────────────────────────────────────────────────────────────────────────

_TEAM_CACHE: list[str] = []

def _extract_team_names(query: str, known_teams: list[str]) -> list[str]:
    """
    Scan query for team mentions using the alias registry and fuzzy matching.
    """
    try:
        from rag.retrieval.team_linker import link_teams
        return link_teams(query, known_teams=known_teams, max_teams=2)
    except Exception:
        found = []
        q_lower = query.lower()
        for team in known_teams:
            if team.lower() in q_lower:
                found.append(team)
            if len(found) >= 2:
                break
        return found


# ─────────────────────────────────────────────────────────────────────────────
# RAG Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

class FootballRAGSystem:
    """
    Full hybrid Dual-RAG pipeline:
      1. Extract team entities from the query
      2. Concurrently pull structured facts from KG (Neo4j / PostgreSQL) and
         semantic match chunks from Vector Store (FAISS / pgvector)
      3. Assemble context → LLM generation
    """

    def __init__(
        self,
        kg: str = None,
        vector: str = None,
        llm: str = None,
        predictor: Optional[BasePredictionProvider] = None,
        **kwargs,
    ):
        cfg = _load_rag_cfg()
        kg_name     = kg     or cfg.get("kg_provider",     "neo4j")
        vector_name = vector or cfg.get("vector_provider", "faiss")
        llm_name    = llm if llm else cfg.get("llm_provider", "huggingface")

        logger.info("Initializing RAG — KG=%s | Vector=%s | LLM=%s",
                    kg_name, vector_name, llm_name)

        self.kg      = get_kg_provider(kg_name)
        self.vector  = get_vector_provider(vector_name)
        self.llm     = get_llm_provider(llm_name)
        self.cfg     = cfg
        self.hybrid  = kwargs.get("hybrid", bool(cfg.get("hybrid_retrieval", False)))
        self.rerank  = kwargs.get("rerank", bool(cfg.get("rerank_results", False)))
        self.predictor = predictor if predictor is not None else GNNPredictionProvider()
        self._retrieval_pool = ThreadPoolExecutor(max_workers=2)

        # Refresh team cache on initialization / hot-reload
        global _TEAM_CACHE
        try:
            fresh_teams = []
            for league in ["Premier_League", "La_Liga", "Serie_A", "Bundesliga", "Ligue_1"]:
                fresh_teams.extend(self.kg.get_league_teams(league))
            if fresh_teams:
                _TEAM_CACHE = list(set(fresh_teams))
                logger.info("Team cache loaded: %d teams", len(_TEAM_CACHE))
        except Exception as e:
            logger.warning("Could not load team cache: %s", e)

    # ── Context Assembly ──────────────────────────────────────────────────────

    def _get_kg_context(self, teams: list[str], query: str, as_of: str = None) -> str:
        """Build a KG context block for the given teams with optional as-of cutoff."""
        with trace_step("kg_retrieval", input={"teams": teams, "as_of": as_of}, metadata={"teams": teams, "as_of": as_of}) as span:
            parts = []

            for team in teams[:2]:  # Max 2 teams per query
                try:
                    parts.append(self.kg.format_team_context(team))
                except Exception as e:
                    logger.warning("KG retrieval failed for '%s': %s", team, e)

            if len(teams) == 2:
                try:
                    h2h = self.kg.get_head_to_head(teams[0], teams[1], limit=5, as_of=as_of)
                    if h2h:
                        lines = [f"\n=== Head-to-Head: {teams[0]} vs {teams[1]} (last 5) ==="]
                        for m in h2h:
                            lines.append(
                                f"  {m.get('date','?')} | "
                                f"{m.get('home_team','?')} {m.get('home_goals','?')}-"
                                f"{m.get('away_goals','?')} {m.get('away_team','?')} "
                                f"[{m.get('result','?')}] | {m.get('league','?')}"
                            )
                        parts.append("\n".join(lines))
                except Exception as e:
                    logger.warning("H2H retrieval failed: %s", e)

            kg_text = "\n\n".join(parts)
            span.update(output=kg_text)
            return kg_text

    @staticmethod
    def _clean_match_chunk(raw_text: str) -> str:
        """Strip noisy environmental telemetry to preserve token budget for core tactical data."""
        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
        kept = []
        for line in lines:
            if line.startswith("Referee strictness:") or line.startswith("Temp:") or line.startswith("Rain:") or line.startswith("Wind:"):
                continue
            if "Over 2.5:" in line and "Clean sheets:" in line:
                continue
            kept.append(line)
        return "\n".join(kept)

    def _get_vector_context(self, query: str, teams: list[str], as_of: str = None) -> str:
        """Semantic search for similar analyses and match stats with optional as-of cutoff."""
        enriched = f"{' '.join(teams)} {query}" if teams else query
        with trace_step("vector_retrieval", input={"query": enriched, "as_of": as_of}, metadata={"query": query, "as_of": as_of, "hybrid": self.hybrid, "rerank": self.rerank}) as span:
            k = self.cfg.get("vector_top_k", 5)
            results = []

            if self.hybrid:
                try:
                    from rag.retrieval.hybrid import HybridRetriever
                    retriever = HybridRetriever(self.vector)
                    results = retriever.search(enriched, k=k, as_of=as_of)
                except Exception as e:
                    logger.debug("Hybrid retrieval error: %s", e)
            elif hasattr(self.vector, "search"):
                try:
                    results = self.vector.search(enriched, k=k, as_of=as_of)
                except Exception as e:
                    logger.debug("vector.search error: %s", e)

            if results and self.rerank:
                try:
                    from rag.retrieval.reranker import rerank_documents
                    results = rerank_documents(query, results, top_k=k)
                except Exception as e:
                    logger.debug("Reranking error: %s", e)

            if results:
                seen_matches = set()
                direct_matches = []
                single_matches = []
                for r in results:
                    meta = r.get("metadata", {})
                    h = meta.get("home_team")
                    a = meta.get("away_team")
                    dt = meta.get("date")
                    key = (h, a, dt) if h and a else r.get("text", "")[:80]
                    if key in seen_matches:
                        continue
                    seen_matches.add(key)

                    cleaned = self._clean_match_chunk(r.get("text", ""))
                    if not cleaned:
                        continue

                    if len(teams) >= 2 and h and a:
                        if h in teams and a in teams:
                            direct_matches.append(cleaned)
                        elif h in teams or a in teams:
                            single_matches.append(cleaned)
                    else:
                        single_matches.append(cleaned)

                selected = direct_matches + single_matches
                if selected:
                    vec_text = "\n\n".join(f"[{i}] {txt}" for i, txt in enumerate(selected[:k], 1))
                    span.update(output=vec_text)
                    return vec_text

            vec_text = self.vector.format_vector_context(enriched, k=k)
            span.update(output=vec_text)
            return vec_text

    # ── Public API ───────────────────────────────────────────────────────────

    def query_with_context(self, question: str, as_of: str = None, teams: list[str] = None) -> dict:
        """
        Execute Dual-RAG retrieval (KG + Vector concurrently) and LLM generation.
        """
        if teams is None:
            with trace_step("entity_extraction", input={"question": question}, metadata={"question": question}) as span:
                teams = _extract_team_names(question, _TEAM_CACHE)
                span.update(output=teams)
                logger.info("Teams detected: %s", teams)
        else:
            logger.info("Using explicit teams: %s", teams)

        # Parallel Dual-RAG retrieval (KG + Vector simultaneously)
        kg_future = self._retrieval_pool.submit(self._get_kg_context, teams, question, as_of)
        vec_future = self._retrieval_pool.submit(self._get_vector_context, question, teams, as_of)
        kg_context = kg_future.result()
        vec_context = vec_future.result()

        contexts: list[str] = []
        if kg_context and kg_context.strip():
            contexts.append(kg_context.strip())
        if vec_context and vec_context.strip():
            contexts.append(vec_context.strip())

        if self.llm is None:
            fallback_text = json.dumps({
                "question": question,
                "teams_detected": teams,
                "kg_context": kg_context,
                "vector_context": vec_context,
                "note": "LLM not configured (set rag.llm_provider in llm_config.yaml).",
            }, ensure_ascii=False)
            return {
                "response": fallback_text,
                "retrieved_contexts": contexts,
                "kg_context": kg_context,
                "vector_context": vec_context,
                "teams": teams,
            }

        model_name = getattr(self.llm, "_backend", None) or getattr(self.llm, "model_name", None) or type(self.llm).__name__
        full_prompt = self._build_prompt(question, kg_context, vec_context)

        with trace_step("llm_generation", as_type="generation", model=str(model_name), input=full_prompt, metadata={"llm_type": type(self.llm).__name__}) as span:
            try:
                if hasattr(self.llm, "generate_with_context"):
                    res = self.llm.generate_with_context(
                        prompt=question,
                        kg_context=kg_context,
                        vector_context=vec_context,
                    )
                elif hasattr(self.llm, "generate"):
                    res = self.llm.generate(full_prompt)
                elif hasattr(self.llm, "_call_api"):
                    res = self.llm._call_api(full_prompt)
                else:
                    raise AttributeError(
                        f"LLM {type(self.llm).__name__} exposes neither generate() nor _call_api()."
                    )
                span.update(output=res)
                return {
                    "response": res,
                    "retrieved_contexts": contexts,
                    "kg_context": kg_context,
                    "vector_context": vec_context,
                    "teams": teams,
                }
            except Exception as e:
                logger.error("LLM inference failed in query_with_context(): %s. Returning structured context fallback.", e)
                fallback_res = json.dumps({
                    "question": question,
                    "teams_detected": teams,
                    "kg_context": kg_context,
                    "vector_context": vec_context,
                    "note": "LLM runtime failure; degraded to quantitative/context fallback.",
                }, ensure_ascii=False)
                span.update(output=fallback_res)
                return {
                    "response": fallback_res,
                    "retrieved_contexts": contexts,
                    "kg_context": kg_context,
                    "vector_context": vec_context,
                    "teams": teams,
                }

    def query(self, question: str, as_of: str = None, teams: list[str] = None) -> str:
        """General-purpose RAG query."""
        result = self.query_with_context(question, as_of=as_of, teams=teams)
        return result["response"]

    @staticmethod
    def _format_gnn_prediction_block(result: Optional[dict]) -> str:
        """Format a precomputed GNN prediction dict into a prompt block."""
        if result is None:
            return ""
        probs = result["probabilities"]
        return (
            f"\n=== Expert 1 (Graph Neural Network) Prediction ===\n"
            f"Predicted Outcome: {result['predicted_result']}\n"
            f"Probabilities: Home Win={probs['H']:.1%} | Draw={probs['D']:.1%} | Away Win={probs['A']:.1%}\n"
            f"=================================================="
        )

    def _get_gnn_prediction(self, home_team: str, away_team: str) -> str:
        """Return a human-readable GNN prediction block (for LLM prompts). Empty on failure."""
        result = self.predict_structured(home_team, away_team)
        return self._format_gnn_prediction_block(result)

    def predict_structured(self, home_team: str, away_team: str) -> Optional[dict]:
        """Return Expert 1's structured prediction (dict) or None on failure."""
        return self.predictor.predict(home_team, away_team)

    def predict_match(
        self,
        home_team: str,
        away_team: str,
        as_of: str = None,
        gnn_prediction: Optional[dict] = None,
    ) -> str:
        """
        Structured match prediction with full RAG context and GNN ensemble.
        Accepts optional pre-computed `gnn_prediction` so Expert 1 runs only once per request.
        """
        gnn_result = gnn_prediction if gnn_prediction is not None else self.predict_structured(home_team, away_team)

        if self.llm is None:
            return json.dumps({
                "home_team": home_team,
                "away_team": away_team,
                "gnn_prediction": gnn_result,
                "note": "LLM not configured; returning Expert 1 (GNN) result only.",
            }, ensure_ascii=False)

        gnn_pred_text = self._format_gnn_prediction_block(gnn_result)
        question = (
            f"You are Expert 2 (Fine-tuned LLM). Analyze the upcoming match between {home_team} (Home) and {away_team} (Away).\n"
            f"Consider their recent form, head-to-head history, and tactical styles (attack and defense).\n"
            f"{gnn_pred_text}\n\n"
            f"Provide your final match prediction with reasoning, integrating your analysis with Expert 1's prediction."
        )
        try:
            return self.query(question, as_of=as_of, teams=[home_team, away_team])
        except Exception as e:
            logger.error("Expert 2 (LLM) failed in predict_match: %s. Returning Expert 1 GNN only.", e)
            return json.dumps({
                "home_team": home_team,
                "away_team": away_team,
                "gnn_prediction": gnn_result,
                "note": "LLM runtime failure; returning Expert 1 (GNN) result only.",
            }, ensure_ascii=False)

    def predict_live_match(self, home_team: str, away_team: str, live_context: dict) -> str:
        """
        Expert 2 live-match narrative: pre-match GNN block + current match
        state (minute, score, live probs, drivers) -> coach-actionable analysis.
        """
        gnn_result = self.predict_structured(home_team, away_team)
        gnn_pred_text = self._format_gnn_prediction_block(gnn_result)

        live_block = (
            f"LIVE STATE (minute {live_context.get('minute', '?')}):\n"
            f"Current score: {live_context.get('home_goals', 0)} - {live_context.get('away_goals', 0)}\n"
            f"Live probabilities: Home Win={live_context.get('live_probs', {}).get('H', 0):.1%} | "
            f"Draw={live_context.get('live_probs', {}).get('D', 0):.1%} | "
            f"Away Win={live_context.get('live_probs', {}).get('A', 0):.1%}\n"
            f"Pre-match probabilities: Home Win={live_context.get('pre_probs', {}).get('H', 0):.1%} | "
            f"Draw={live_context.get('pre_probs', {}).get('D', 0):.1%} | "
            f"Away Win={live_context.get('pre_probs', {}).get('A', 0):.1%}\n"
            f"Expected final score: {live_context.get('expected_score', {}).get('home', '?')} - "
            f"{live_context.get('expected_score', {}).get('away', '?')}\n"
        )

        drivers = live_context.get("drivers", [])
        if drivers:
            lines = ["LIVE STAT DRIVERS (deviation from season averages):"]
            for d in drivers:
                lines.append(
                    f"  - {d.get('side', '?')} {d.get('label', d.get('stat', '?'))}: "
                    f"{d.get('direction', 'neutral')} (live {d.get('live', '?')}/min vs season {d.get('season_avg', '?')}/min, "
                    f"pace x{d.get('pace', '?')})"
                )
            live_block += "\n" + "\n".join(lines)

        if self.llm is None:
            return json.dumps({
                "home_team": home_team,
                "away_team": away_team,
                "gnn_prediction": gnn_result,
                "live_context": live_context,
                "note": "LLM not configured; returning structured live context only.",
            }, ensure_ascii=False)

        question = (
            f"You are Expert 2 (Fine-tuned LLM), an in-match tactical advisor to a coach.\n"
            f"You are watching {home_team} (Home) vs {away_team} (Away).\n"
            f"{gnn_pred_text}\n\n"
            f"{live_block}\n\n"
            f"Give a concise, coach-actionable analysis grounded ONLY in the live driver data.\n\n"
            f"Respond with a SINGLE JSON object and nothing else: no markdown fences, no prose "
            f"outside the object. Fill in real match-specific text for every field:\n"
            f"{{\n"
            f'  "match_state": {{"minute": {live_context.get("minute", "?")}, '
            f'"score": "{live_context.get("home_goals", 0)}-{live_context.get("away_goals", 0)}"}},\n'
            f'  "analysis": {{\n'
            f'    "who_controls_now": "<2-3 sentences explaining whether {home_team} or {away_team} controls the match at {live_context.get("minute", "?")}\' and why>",\n'
            f'    "why": [\n'
            f'      "<signal 1 with exact numbers from LIVE STAT DRIVERS>",\n'
            f'      "<signal 2 with exact numbers from LIVE STAT DRIVERS>",\n'
            f'      "<signal 3 with remaining time and expected final score>"\n'
            f'    ],\n'
            f'    "how_outlook_changed": "<2-3 sentences comparing pre-match probabilities vs live probabilities at {live_context.get("minute", "?")}\'>",\n'
            f'    "coach_recommendations": [\n'
            f'      {{"priority": 1, "action": "<concrete tactical adjustment for {home_team}>", "reason": "<data-backed reason from live state>"}},\n'
            f'      {{"priority": 2, "action": "<concrete tactical adjustment for {away_team}>", "reason": "<data-backed reason from live state>"}},\n'
            f'      {{"priority": 3, "action": "<key substitution or set-piece focus for final minutes>", "reason": "<data-backed reason from live state>"}}\n'
            f"    ]\n"
            f"  }}\n"
            f"}}\n\n"
            f"Rules: NEVER output placeholder words like 'bullet 1' or 'specific action'. "
            f"Every string must reference {home_team}, {away_team}, the {live_context.get('home_goals', 0)}-{live_context.get('away_goals', 0)} scoreline at {live_context.get('minute', '?')}', and the exact numbers above."
        )
        try:
            return self.query(question, teams=[home_team, away_team])
        except Exception as e:
            logger.error("Expert 2 (LLM) failed in predict_live_match: %s. Returning structured fallback.", e)
            return json.dumps({
                "home_team": home_team,
                "away_team": away_team,
                "gnn_prediction": gnn_result,
                "live_context": live_context,
                "note": "LLM runtime failure; returning structured live context only.",
            }, ensure_ascii=False)

    def get_team_report(self, team_name: str) -> str:
        """Generate a full team tactical report."""
        question = (
            f"Provide a comprehensive tactical analysis of {team_name}. "
            f"Describe their attacking style, defensive organization, "
            f"recent form, strengths, and weaknesses."
        )
        return self.query(question)

    def compare_teams(self, team_a: str, team_b: str) -> str:
        """Compare two teams tactically and statistically."""
        question = (
            f"Compare {team_a} and {team_b} in detail: "
            f"attacking tactics, defensive organization, recent form, "
            f"head-to-head record, and key statistical differences."
        )
        return self.query(question)

    def _build_prompt(self, question: str, kg_context: str, vector_context: str) -> str:
        """Build the final RAG prompt for providers without generate_with_context."""
        parts = [
            "You are an expert football tactical analyst with access to a comprehensive database.",
            "Base your analysis strictly on the provided factual match data, head-to-head records, and tactical profiles.",
            "When tactical data is complete, ground your reasoning in the numbers; if specific historical data points are absent, use balanced analytical football reasoning without fabricating false scores, player names, or statistics.",
            "",
        ]
        if kg_context:
            parts += ["## Retrieved Knowledge Graph Context", kg_context, ""]
        if vector_context:
            parts += ["## Retrieved Historical Analyses", vector_context, ""]
        parts += ["## Question", question]
        return "\n".join(parts)

    def get_available_teams(self) -> list[str]:
        """Return a sorted list of all available team names from the cached known teams."""
        global _TEAM_CACHE
        return sorted(list(_TEAM_CACHE))

    # ── Context Management ────────────────────────────────────────────────────

    def close(self):
        """Close database connections and thread pool."""
        try:
            self._retrieval_pool.shutdown(wait=False)
        except Exception:
            pass
        try:
            self.kg.close()
        except Exception:
            pass
