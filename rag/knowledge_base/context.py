"""
ContextBundle — the provider-agnostic retrieval contract (core of the KB).

Every question resolves into a ContextBundle: facts (structured), tables
(markdown), vector hits (semantic), and SourceRef citations with ref numbers.
Any LLM (or `none`) consumes the same bundle:

    bundle.render()    → prompt-ready context block (markdown, citations [n])
    bundle.to_json()   → full serializable payload (REST / GraphQL / none-safe)
"""

import json
from typing import List, Optional


class SourceRef:
    """A single retrievable evidence item (citation)."""

    __slots__ = ("title", "text", "source_type", "team", "league", "season",
                 "doc_id", "ref")

    def __init__(self, title: str, text: str = "",
                 source_type: str = "csv", team=None, league=None,
                 season=None, doc_id=None, ref: int = 0):
        self.title = title
        self.text = text
        self.source_type = source_type          # csv | postgres | tactics | faiss | gnn | kb
        self.team = team
        self.league = league
        self.season = season
        self.doc_id = doc_id
        self.ref = ref                          # assigned by ContextBundle

    def to_json(self) -> dict:
        return dict(
            ref=self.ref, title=self.title, text=self.text,
            source_type=self.source_type, team=self.team,
            league=self.league, season=self.season, doc_id=self.doc_id,
        )

    def __repr__(self):
        return f"<SourceRef [{self.ref}] {self.source_type}: {self.title}>"


class ContextBundle:
    """Retrieval result for one question — LLM-agnostic facts + sources."""

    def __init__(self, question: str, intent: str = "general",
                 teams: Optional[List[str]] = None, league=None, season=None,
                 params: Optional[dict] = None):
        self.question = question
        self.intent = intent
        self.teams = list(teams) if teams else []
        self.league = league
        self.season = season
        self.params = dict(params) if params else {}
        self.facts: List[dict] = []             # {label, value, ref}
        self.tables: List[dict] = []            # {title, header, rows, ref}
        self.vector_hits: List[dict] = []       # {text, score, metadata, ref}
        self._sources: List[SourceRef] = []

    # ─────────────────────────────────────────────────────────────────────────
    # Population
    # ─────────────────────────────────────────────────────────────────────────

    def add_fact(self, label: str, value, source: Optional[SourceRef] = None) -> int:
        ref = self._register(source)
        self.facts.append({"label": label, "value": value, "ref": ref})
        return ref

    def add_table(self, title: str, header: List[str], rows: List[list],
                  source: Optional[SourceRef] = None) -> int:
        ref = self._register(source)
        self.tables.append({"title": title, "header": list(header),
                            "rows": rows, "ref": ref})
        return ref

    def add_vector_hit(self, text: str, score: float, metadata: Optional[dict] = None,
                       source: Optional[SourceRef] = None) -> int:
        ref = self._register(source)
        self.vector_hits.append({"text": text, "score": float(score),
                                 "metadata": metadata or {}, "ref": ref})
        return ref

    def _register(self, source: Optional[SourceRef]) -> int:
        if source is None:
            return 0
        source.ref = len(self._sources) + 1
        self._sources.append(source)
        return source.ref

    @property
    def sources(self) -> List[SourceRef]:
        return list(self._sources)

    # ─────────────────────────────────────────────────────────────────────────
    # Output
    # ─────────────────────────────────────────────────────────────────────────

    def render(self) -> str:
        """Prompt-ready context block with citation markers [n]."""
        parts = []
        head = "CONTEXT"
        bits = [f"intent={self.intent}"]
        if self.teams:
            bits.append(f"teams={', '.join(self.teams)}")
        if self.league:
            bits.append(f"league={self.league}")
        if self.season:
            bits.append(f"season={self.season}")
        parts.append(f"=== {head} ({'; '.join(bits)}) ===")

        if self.facts:
            parts.append("FACTS:")
            for f in self.facts:
                marker = f"[{f['ref']}]" if f["ref"] else ""
                parts.append(f"  • {f['label']}: {f['value']} {marker}".rstrip())

        for t in self.tables:
            parts.append(f"[{t['title']}]")
            parts.append("| " + " | ".join(str(c) for c in t["header"]) + " |")
            parts.append("|" + "|".join("---" for _ in t["header"]) + "|")
            for row in t["rows"]:
                parts.append("| " + " | ".join(str(c) for c in row) + " |")
            if t["ref"]:
                parts[-1] += f" [{t['ref']}]"

        if self.vector_hits:
            parts.append("SEMANTIC PRECEDENT (similar past matches/analyses):")
            for h in self.vector_hits:
                marker = f"[{h['ref']}]" if h["ref"] else ""
                parts.append(f"  • {h['text'][:300]} (score {h['score']:.3f}) {marker}".rstrip())

        if self._sources:
            parts.append("SOURCES:")
            for s in self._sources:
                loc = " | ".join(x for x in (s.league, s.season, s.team) if x)
                parts.append(f"  [{s.ref}] {s.source_type}: {s.title}"
                             + (f" ({loc})" if loc else ""))

        return "\n".join(parts)

    def to_json(self) -> dict:
        return dict(
            question=self.question,
            intent=self.intent,
            teams=self.teams,
            league=self.league,
            season=self.season,
            params=self.params,
            facts=self.facts,
            tables=self.tables,
            vector_hits=self.vector_hits,
            sources=[s.to_json() for s in self._sources],
        )

    def token_estimate(self) -> int:
        """Rough token count (chars / 4) for budget logging."""
        return len(self.render()) // 4

    def __repr__(self):
        return (f"<ContextBundle intent={self.intent} teams={self.teams} "
                f"facts={len(self.facts)} tables={len(self.tables)} "
                f"hits={len(self.vector_hits)} sources={len(self._sources)}>")
