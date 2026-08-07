"""
KnowledgeBase — the provider-agnostic Q&A facade (core of the chat-KB plan).

    retrieve(question)            → ContextBundle (facts + tables + sources; NO LLM)
    ask(question, llm_name)       → Answer {content, provider, sources, bundle}
                                    - llm_name=None/"" → none-safe structured answer
                                    - llm_name="ollama"/"openai"/... → narrated answer
                                      through ANY registered provider

All internals are lazy and tolerant:
  - CSV loads on first question (once per process)
  - PostgreSQL via TeamProfileStore (per-team, TTL-cached, CSV fallback)
  - FAISS vector index loads only for semantic enrichment, absent ⇒ degraded
  - GNN predictor loads only for prediction intent, absent ⇒ noted in facts
"""

import json
import logging
import threading
from pathlib import Path
from typing import List, Optional

import yaml

from models.llm_providers import get_llm_provider
from rag.knowledge_base.config import kb_settings
from rag.knowledge_base.context import ContextBundle, SourceRef
from rag.knowledge_base.datastore import MatchDataStore
from rag.knowledge_base.intents import IntentClassifier, IntentResult
from rag.knowledge_base.team_store import TeamProfileStore

logger = logging.getLogger(__name__)

_CFG_PATH = Path(__file__).resolve().parent.parent.parent / "models" / "llm_config.yaml"


def _rag_cfg() -> dict:
    try:
        with open(_CFG_PATH, "r", encoding="utf-8") as f:
            return (yaml.safe_load(f) or {}).get("rag", {}) or {}
    except Exception:
        return {}

_SYSTEM_PROMPT = (
    "You are a football knowledge assistant. Answer ONLY from the provided "
    "CONTEXT block below. Cite the evidence you use with its [n] source markers. "
    "If the context does not contain the answer, say so clearly. Be concise and factual."
)

# stat intent key → (aggregate key, display label)
_STAT_AGG = {
    "goals": ("avg_goals_home", "avg goals (home)"),
    "xg": ("avg_xg", "avg xG"),
    "xga": ("avg_xga", "avg xGA"),
    "shots": ("avg_shots", "avg shots"),
    "sot": ("avg_sot", "avg shots on target"),
    "corners": ("avg_corners", "avg corners"),
    "fouls": ("avg_fouls", "avg fouls"),
    "yellows": ("avg_yellows", "avg yellows"),
    "clean_sheet": ("clean_sheet_rate", "clean sheet rate"),
}


def _llm_generate(llm, prompt: str) -> str:
    """Call any registered LLM: HF exposes generate(), others expose _call_api().

    Providers that accept json_mode get free-text narration (OpenAI); others
    fall back to their default call signature.
    """
    if hasattr(llm, "generate"):
        return llm.generate(prompt)
    if hasattr(llm, "_call_api"):
        try:
            return llm._call_api(prompt, json_mode=False)
        except TypeError:
            return llm._call_api(prompt)
    raise AttributeError(f"LLM {type(llm).__name__} exposes no generate()/_call_api()")


class Answer:
    """Result of KnowledgeBase.ask() — LLM-independent shape."""

    def __init__(self, content: str, provider: str, bundle: ContextBundle,
                 error: Optional[str] = None):
        self.content = content
        self.provider = provider
        self.bundle = bundle
        self.error = error

    def to_json(self) -> dict:
        return dict(
            content=self.content,
            provider=self.provider,
            error=self.error,
            sources=[s.to_json() for s in self.bundle.sources],
            bundle=self.bundle.to_json(),
        )


class KnowledgeBase:
    def __init__(self, store: Optional[MatchDataStore] = None,
                 team_store: Optional[TeamProfileStore] = None,
                 classifier: Optional[IntentClassifier] = None,
                 vector=None, predictor=None, top_k: Optional[int] = None):
        cfg = kb_settings()
        self._cfg = cfg
        self._store = store or MatchDataStore()
        self._team_store = team_store or TeamProfileStore(csv_store=self._store)
        self._classifier = classifier or IntentClassifier(self._store)

        # Lazy components (None until first needed; tolerant on failure).
        self._vector = vector
        self._predictor = predictor
        self._vector_tried = vector is not None
        self._predictor_tried = predictor is not None
        self._lock = threading.Lock()
        self._top_k = top_k or int(_rag_cfg().get("vector_top_k", 5))

    # ─────────────────────────────────────────────────────────────────────────
    # Lazy internals
    # ─────────────────────────────────────────────────────────────────────────

    def _get_vector(self):
        with self._lock:
            if self._vector_tried:
                return self._vector
            self._vector_tried = True
            try:
                from rag.providers.vector_provider import FAISSProvider
                self._vector = FAISSProvider()
                logger.info("KB: FAISS vector index attached (lazy)")
            except Exception as e:
                logger.warning("KB: vector index unavailable (%s) — semantic enrichment disabled", e)
                self._vector = None
            return self._vector

    def _get_predictor(self):
        with self._lock:
            if self._predictor_tried:
                return self._predictor
            self._predictor_tried = True
            try:
                from rag.providers.gnn_provider import GNNPredictionProvider
                self._predictor = GNNPredictionProvider()
                logger.info("KB: GNN predictor attached (lazy)")
            except Exception as e:
                logger.warning("KB: GNN predictor unavailable (%s) — prediction facts skipped", e)
                self._predictor = None
            return self._predictor

    # ─────────────────────────────────────────────────────────────────────────
    # Retrieval (no LLM)
    # ─────────────────────────────────────────────────────────────────────────

    def retrieve(self, question: str, prefer_prediction: bool = False) -> ContextBundle:
        try:
            r = self._classifier.classify(question, prefer_prediction=prefer_prediction)
        except Exception as e:
            # e.g. CSV missing → classify needs team names; degrade gracefully.
            logger.error("KB: intent classification failed (%s) — degraded retrieval", e)
            r = None
            bundle = ContextBundle(question, "general", params={})
            bundle.add_fact("kb_error",
                            f"Knowledge base data unavailable ({e}). "
                            "Check kb_csv_path / data pipeline.")
            return bundle

        bundle = ContextBundle(question, r.intent, r.teams, r.league, r.season,
                               params=r.to_json())
        try:
            getattr(self, f"_resolve_{r.intent}")(bundle, r)
        except Exception as e:
            logger.error("KB resolver '%s' failed: %s", r.intent, e)
            bundle.add_fact("retrieval_error", str(e))
        return bundle

    # ── prediction ───────────────────────────────────────────────────────────

    def _resolve_prediction(self, b: ContextBundle, r: IntentResult):
        if len(r.teams) != 2:
            b.add_fact("note", "Prediction needs two teams.")
            return
        a, c = r.teams
        self._team_facts(b, a)
        self._team_facts(b, c)
        self._h2h_table(b, a, c, 5)
        self._form_facts(b, a, 3)
        self._form_facts(b, c, 3)

        pred = None
        predictor = self._get_predictor()
        if predictor is not None:
            try:
                pred = predictor.predict(a, c)
            except Exception as e:
                logger.warning("KB: GNN predict failed: %s", e)
        if pred:
            probs = pred.get("probabilities", {})
            b.add_fact("expert1_predicted_result", pred.get("predicted_result"),
                       SourceRef("Expert 1 (GNN) prediction", source_type="gnn",
                                 team=f"{a} vs {c}"))
            b.add_fact("expert1_probabilities",
                       {k: round(float(v), 4) for k, v in probs.items()},
                       SourceRef("Expert 1 (GNN) prediction", source_type="gnn",
                                 team=f"{a} vs {c}"))
        else:
            b.add_fact("expert1_prediction", "unavailable (model not loaded)",
                       SourceRef("Expert 1 (GNN) prediction", source_type="gnn"))
        self._semantic(b, r, k=2)

    # ── team_profile ─────────────────────────────────────────────────────────

    def _resolve_team_profile(self, b: ContextBundle, r: IntentResult):
        team = r.teams[0] if r.teams else None
        if team is None:
            b.add_fact("note", "Which team? e.g. 'Tell me about Arsenal'.")
            return
        profile = self._team_store.get_profile(team)
        if not profile:
            b.add_fact("note", f"No profile found for '{team}'.")
            return
        src = SourceRef("Team profile", source_type="postgres", team=team,
                        league=profile.get("league"))
        b.add_fact("league", profile.get("league"), src)
        b.add_fact("total_matches", profile.get("total_matches"), src)
        b.add_fact("win_rate", _pct(profile.get("win_rate")), src)
        b.add_fact("avg_goals_home", profile.get("avg_goals_home"), src)
        b.add_fact("avg_goals_away", profile.get("avg_goals_away"), src)
        b.add_fact("avg_xg", profile.get("avg_xg"), src)
        b.add_fact("avg_xga", profile.get("avg_xga"), src)
        b.add_fact("clean_sheet_rate", _pct(profile.get("clean_sheet_rate")), src)
        if profile.get("attack_tactic") or profile.get("defense_tactic"):
            b.add_fact("tactics", _tactics_text(profile), src)
        if profile.get("strengths"):
            b.add_fact("strengths", "; ".join(profile["strengths"]), src)
        if profile.get("weaknesses"):
            b.add_fact("weaknesses", "; ".join(profile["weaknesses"]), src)
        if not b.league:
            b.league = profile.get("league")
        self._semantic(b, r, k=2, filter_meta={"doc_type": "team_profile"})

    # ── head_to_head ─────────────────────────────────────────────────────────

    def _resolve_head_to_head(self, b: ContextBundle, r: IntentResult):
        if len(r.teams) != 2:
            b.add_fact("note", "Head-to-head needs two teams.")
            return
        a, c = r.teams
        self._team_facts(b, a)
        self._team_facts(b, c)
        self._h2h_table(b, a, c, max(3, r.top_n))

    # ── recent_form ──────────────────────────────────────────────────────────

    def _resolve_recent_form(self, b: ContextBundle, r: IntentResult):
        team = r.teams[0] if r.teams else None
        if team is None:
            b.add_fact("note", "Which team's form? e.g. 'Arsenal last 5 matches'.")
            return
        self._team_facts(b, team)
        self._form_facts(b, team, max(1, min(r.top_n, 10)))

    # ── standings ────────────────────────────────────────────────────────────

    def _resolve_standings(self, b: ContextBundle, r: IntentResult):
        league = r.league
        if not league and r.teams:
            agg = self._store.aggregates_for(r.teams[0])
            league = agg.get("league") if agg else None
        if not league:
            b.add_fact("note",
                       "Which league? Available: " + ", ".join(self._store.league_names()))
            return
        rows = self._store.league_table(league=league, season=r.season)
        if not rows:
            b.add_fact("note", f"No standings found for {league} ({r.season or 'all seasons'}).")
            return
        b.league = league
        b.add_table(
            f"Standings — {league}" + (f" {r.season}" if r.season else ""),
            ["P", "Team", "Pld", "W", "D", "L", "GF", "GA", "GD", "Pts"],
            [[x["position"], x["team"], x["played"], x["wins"], x["draws"],
              x["losses"], x["goals_for"], x["goals_against"], x["goal_diff"],
              x["points"]] for x in rows[:r.top_n]],
            SourceRef(f"CSV standings {league}", source_type="csv", league=league,
                      season=r.season),
        )

    # ── stats_question ───────────────────────────────────────────────────────

    def _resolve_stats_question(self, b: ContextBundle, r: IntentResult):
        stat = r.stat or "goals"
        if len(r.teams) == 2:
            self._stats_table(b, r.teams[0], r.teams[1], stat)
        elif len(r.teams) == 1:
            self._team_facts(b, r.teams[0], stat)
        else:
            b.add_fact("note", "Which team? e.g. 'average goals of Real Madrid'.")

    # ── compare ──────────────────────────────────────────────────────────────

    def _resolve_compare(self, b: ContextBundle, r: IntentResult):
        if len(r.teams) != 2:
            b.add_fact("note", "Comparison needs two teams.")
            return
        a, c = r.teams
        self._stats_table(b, a, c, r.stat or "goals")
        self._team_facts(b, a)
        self._team_facts(b, c)
        self._semantic(b, r, k=2)

    # ── general ──────────────────────────────────────────────────────────────

    def _resolve_general(self, b: ContextBundle, r: IntentResult):
        self._semantic(b, r, k=self._top_k)
        if not b.vector_hits and not b.facts:
            b.add_fact("note",
                       "No structured or semantic sources matched. Try naming a team "
                       "or league (e.g. 'Arsenal form', 'Premier League standings').")

    # ── shared builders ──────────────────────────────────────────────────────

    def _team_facts(self, b: ContextBundle, team: str, stat: Optional[str] = None):
        agg = self._store.aggregates_for(team)
        if not agg:
            b.add_fact("note", f"Unknown team: {team}")
            return
        src = SourceRef(f"CSV team aggregates", source_type="csv", team=team,
                        league=agg.get("league"))
        if stat and stat in _STAT_AGG:
            key, label = _STAT_AGG[stat]
            b.add_fact(f"{team} {label}", agg.get(key), src)
            if stat == "goals":
                b.add_fact(f"{team} avg goals (away)", agg.get("avg_goals_away"), src)
            return
        b.add_fact(f"{team} win rate", _pct(agg.get("win_rate")), src)
        b.add_fact(f"{team} avg goals (home)", agg.get("avg_goals_home"), src)
        b.add_fact(f"{team} avg goals (away)", agg.get("avg_goals_away"), src)
        b.add_fact(f"{team} avg xG", agg.get("avg_xg"), src)
        b.add_fact(f"{team} avg xGA", agg.get("avg_xga"), src)
        b.add_fact(f"{team} clean sheet rate", _pct(agg.get("clean_sheet_rate")), src)

    def _h2h_table(self, b: ContextBundle, a: str, c: str, limit: int):
        rows = self._team_store.get_head_to_head(a, c, limit=limit)
        if not rows:
            b.add_fact(f"H2H {a} vs {c}", "no meetings found",
                       SourceRef("Head-to-head", source_type="postgres",
                                 team=f"{a} vs {c}"))
            return
        b.add_table(
            f"H2H last {len(rows)}: {a} vs {c}",
            ["Date", "Home", "Score", "Away", "League"],
            [[x.get("date"), x.get("home_team"), f"{x.get('home_goals')}-{x.get('away_goals')}",
              x.get("away_team"), x.get("league")] for x in rows],
            SourceRef("Head-to-head", source_type="postgres", team=f"{a} vs {c}"),
        )
        wins_a = sum(1 for x in rows if (x.get("home_team") == a and x.get("result") == "H")
                     or (x.get("away_team") == a and x.get("result") == "A"))
        wins_c = sum(1 for x in rows if (x.get("home_team") == c and x.get("result") == "H")
                     or (x.get("away_team") == c and x.get("result") == "A"))
        draws = len(rows) - wins_a - wins_c
        b.add_fact(f"H2H summary {a} vs {c}", f"{wins_a}W {draws}D {wins_c}L")

    def _form_facts(self, b: ContextBundle, team: str, n: int):
        rows = self._team_store.get_recent_form(team, n=n)
        if not rows:
            b.add_fact(f"{team} recent form", "no matches found")
            return
        b.add_table(
            f"{team} — last {len(rows)} matches",
            ["Date", "Home", "Score", "Away"],
            [[x.get("date"), x.get("home_team"),
              f"{x.get('home_goals')}-{x.get('away_goals')}", x.get("away_team")]
             for x in rows],
            SourceRef("Recent form", source_type="postgres", team=team),
        )

    def _stats_table(self, b: ContextBundle, a: str, c: str, stat: str):
        key, label = _STAT_AGG.get(stat, ("avg_xg", "avg xG"))
        agg_a = self._store.aggregates_for(a) or {}
        agg_c = self._store.aggregates_for(c) or {}
        if not agg_a or not agg_c:
            b.add_fact("note", "Unknown team in comparison.")
            return
        rows = [
            ["win_rate", _pct(agg_a.get("win_rate")), _pct(agg_c.get("win_rate"))],
            [label, _num(agg_a.get(key)), _num(agg_c.get(key))],
            ["avg goals (home)", _num(agg_a.get("avg_goals_home")), _num(agg_c.get("avg_goals_home"))],
            ["avg goals (away)", _num(agg_a.get("avg_goals_away")), _num(agg_c.get("avg_goals_away"))],
            ["clean sheet rate", _pct(agg_a.get("clean_sheet_rate")), _pct(agg_c.get("clean_sheet_rate"))],
        ]
        b.add_table(
            f"Compare — {a} vs {c}",
            ["Metric", a, c],
            rows,
            SourceRef("CSV team aggregates", source_type="csv", team=f"{a} vs {c}"),
        )

    def _semantic(self, b: ContextBundle, r: IntentResult, k: int,
                  filter_meta: Optional[dict] = None):
        vector = self._get_vector()
        if vector is None:
            return
        query = " ".join(r.teams + [b.question]) if r.teams else b.question
        try:
            hits = vector.search(query, k=k, filter_meta=filter_meta)
        except Exception as e:
            logger.warning("KB: semantic search failed: %s", e)
            return
        for h in hits:
            m = h["metadata"]
            src = SourceRef(
                f"{m.get('doc_type', 'doc')} — {m.get('team') or m.get('home_team') or '?'}",
                source_type="faiss", team=m.get("team") or m.get("home_team"),
                league=m.get("league"), season=m.get("season"),
                doc_id=m.get("match_id"),
            )
            b.add_vector_hit(h["text"], h["score"], m, src)

    # ─────────────────────────────────────────────────────────────────────────
    # Ask (LLM-optional)
    # ─────────────────────────────────────────────────────────────────────────

    def ask(self, question: str, llm_name: Optional[str] = None,
            prefer_prediction: bool = False, memory: Optional[str] = None) -> Answer:
        bundle = self.retrieve(question, prefer_prediction=prefer_prediction)

        # None-safe: llm_name=None → structured answer without any LLM.
        llm = get_llm_provider(llm_name) if llm_name else None
        if llm is None:
            return Answer(
                content=self._structured_content(bundle),
                provider="none",
                bundle=bundle,
            )

        memory_block = f"\nRECENT CONVERSATION MEMORY:\n{memory}\n" if memory else ""
        prompt = (
            f"{_SYSTEM_PROMPT}\n\n{memory_block}"
            f"{bundle.render()}\n\nQUESTION: {question}"
        )
        try:
            content = _llm_generate(llm, prompt)
            provider = getattr(llm, "provider_name", str(llm_name))
            return Answer(content=content, provider=provider, bundle=bundle)
        except Exception as e:
            logger.error("KB: LLM '%s' failed (%s) — falling back to structured answer", llm_name, e)
            return Answer(
                content=self._structured_content(bundle),
                provider="none",
                bundle=bundle,
                error=f"LLM '{llm_name}' failed: {e}",
            )

    @staticmethod
    def _structured_content(bundle: ContextBundle) -> str:
        """None-safe JSON payload: retrieved facts + tables + sources."""
        payload = {
            "answer_type": "structured",
            "intent": bundle.intent,
            "teams": bundle.teams,
            "facts": bundle.facts,
            "tables": bundle.tables,
            "sources": [s.to_json() for s in bundle.sources],
            "note": "LLM not configured; returning retrieved facts only.",
        }
        return json.dumps(payload, ensure_ascii=False)

    def status(self) -> dict:
        return dict(
            csv_rows=len(self._store.team_names()) if self._store else 0,
            team_store=self._team_store.status() if self._team_store else {},
            vector_loaded=self._get_vector() is not None,
            predictor_loaded=self._get_predictor() is not None,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Formatting helpers
# ─────────────────────────────────────────────────────────────────────────────

def _pct(v) -> str:
    try:
        return f"{float(v):.0%}"
    except (TypeError, ValueError):
        return "?"


def _num(v) -> str:
    try:
        return f"{float(v):.2f}"
    except (TypeError, ValueError):
        return "?"


def _tactics_text(profile: dict) -> str:
    parts = []
    for key in ("attack_headline", "defense_headline"):
        if profile.get(key):
            parts.append(profile[key])
    return " | ".join(parts) or "no tactic summary"
