"""
Rule-based intent router for the knowledge base (Decision D4).

Deterministic, cheap, fully testable. Classifies a question into one of:

    prediction | head_to_head | compare | standings | recent_form |
    stats_question | team_profile | general

and extracts parameters: teams (canonical CSV names, max 2), league, season,
stat, top_n. `prefer_prediction` lets the chat layer force prediction routing
for prediction-mode conversations (Phase 4).
"""

import re
from typing import List, Optional

from rag.knowledge_base.datastore import MatchDataStore

# (stat_key, aliases) — matched longest-first so "shots on target" wins over "shots"
_STAT_ALIASES: List[tuple] = [
    ("clean_sheet", ["clean sheet", "clean-sheet", "shutout"]),
    ("sot", ["shots on target", "sot", "on target"]),
    ("xga", ["xga", "expected goals against", "xg against"]),
    ("xg", ["expected goals", "xg", "expected xg"]),
    ("yellows", ["yellow cards", "yellows"]),
    ("corners", ["corners", "corner kick"]),
    ("shots", ["shots"]),
    ("fouls", ["fouls"]),
    ("goals", ["goals", "scoring", "score"]),
]

_PREDICTION_WORDS = [
    "predict", "prediction", "who will win", "who wins", "will win",
    "win the match", "expected result", "winner", "outcome", "odds",
    "probability", "probabilities", "forecast", "tip", "expert 1",
    "gnn", "beat", "win probability",
]

_H2H_WORDS = [
    "head to head", "head-to-head", "h2h", "previous meeting", "previous meetings",
    "past meeting", "past meetings", "meetings between", "history between",
    "record between", "last time they met", "when they last", "against each other",
    "vs", "versus",
]

_COMPARE_WORDS = [
    "compare", "comparison", "better team", "stronger team", "who is better",
    "who is stronger", "difference between", "which team is better",
]

_STANDINGS_WORDS = [
    "standing", "standings", "league table", " table", "rank", "ranked",
    "position", "top of the league", "bottom of the league", "promotion",
    "relegation", "championship race", "who is first", "who leads",
    "who is leading", "leads the", "leading the", "1st place", "title race",
    "top of",
]

_FORM_WORDS = [
    "form", "recent", "last match", "last game", "last matches", "last games",
    "run of", "current run", "streak", "in the last",
]

_STATS_WORDS = [
    "average", "avg", "stats", "statistics", "statistically", "how many",
    "stat", "numbers", "record",
]

_PROFILE_WORDS = [
    "profile", "tactics", "tactical", "style", "strengths", "weaknesses",
    "formation", "how do they play", "how they play", "attack", "defense",
    "defence", "playing",
]


class IntentResult:
    """Parsed intent + parameters for one question."""

    __slots__ = ("intent", "teams", "league", "season", "stat", "top_n", "reason")

    def __init__(self, intent: str, teams=None, league=None, season=None,
                 stat=None, top_n: int = 10, reason: str = ""):
        self.intent = intent
        self.teams: List[str] = teams or []
        self.league: Optional[str] = league
        self.season: Optional[str] = season
        self.stat: Optional[str] = stat
        self.top_n: int = top_n
        self.reason = reason

    def to_json(self) -> dict:
        return dict(
            intent=self.intent, teams=self.teams, league=self.league,
            season=self.season, stat=self.stat, top_n=self.top_n,
            reason=self.reason,
        )

    def __repr__(self):
        return (f"<IntentResult intent={self.intent} teams={self.teams} "
                f"league={self.league} season={self.season} stat={self.stat}>")


class IntentClassifier:
    """Rule-based classifier backed by MatchDataStore team/league/season names."""

    def __init__(self, store: MatchDataStore):
        self.store = store
        self._div_map: Optional[dict] = None

    # ─────────────────────────────────────────────────────────────────────────
    # Public entry
    # ─────────────────────────────────────────────────────────────────────────

    def classify(self, question: str, prefer_prediction: bool = False) -> IntentResult:
        q = _normalize(question)
        if not q:
            return IntentResult("general", reason="empty question")

        teams = self._extract_teams(q)
        league = self._extract_league(q)
        season = self._extract_season(q)
        stat = self._extract_stat(q)
        top_n = self._extract_top_n(q)

        has_2 = len(teams) == 2
        has_1 = len(teams) == 1

        # 1. Prediction — explicit intent OR forced by conversation mode.
        if prefer_prediction and has_2:
            return IntentResult("prediction", teams, league, season, stat, top_n,
                                "prediction-mode conversation with 2 teams")
        if has_2 and any(w in q for w in _PREDICTION_WORDS):
            return IntentResult("prediction", teams, league, season, stat, top_n,
                                "prediction keywords + 2 teams")

        # 2. Head-to-head — 2 teams + h2h words (bare "vs" counts too).
        if has_2 and any(w in q for w in _H2H_WORDS):
            return IntentResult("head_to_head", teams, league, season, stat, top_n,
                                "2 teams + h2h/vs keywords")

        # 3. Compare — 2 teams + comparison words.
        if has_2 and any(w in q for w in _COMPARE_WORDS):
            return IntentResult("compare", teams, league, season, stat, top_n,
                                "2 teams + comparison keywords")

        # 4. Standings.
        if any(w in q for w in _STANDINGS_WORDS):
            return IntentResult("standings", teams, league, season, stat, top_n,
                                "standings keywords")

        # 5. Recent form.
        if has_1 and (any(w in q for w in _FORM_WORDS) or _has_form_hint(q)):
            return IntentResult("recent_form", teams, league, season, stat, top_n,
                                "team + form keywords")

        # 6. Stats question.
        if stat is not None or any(w in q for w in _STATS_WORDS):
            return IntentResult("stats_question", teams, league, season, stat, top_n,
                                "stat keywords detected")

        # 7. Team profile.
        if has_1 and any(w in q for w in _PROFILE_WORDS):
            return IntentResult("team_profile", teams, league, season, stat, top_n,
                                "team + profile keywords")

        # 8. Fallbacks.
        if has_1:
            return IntentResult("team_profile", teams, league, season, stat, top_n,
                                "single team, default profile")
        return IntentResult("general", teams, league, season, stat, top_n,
                            "no intent matched")

    # ─────────────────────────────────────────────────────────────────────────
    # Parameter extraction
    # ─────────────────────────────────────────────────────────────────────────

    def _extract_teams(self, q: str) -> List[str]:
        """Longest-first substring scan (overlap-safe) over canonical names and
        registry aliases, ordered by position in the question."""
        from data.team_registry import TEAM_REGISTRY

        csv_names = set(self.store.team_names())

        alias_map = {}
        for league_map in TEAM_REGISTRY.values():
            for alias, canonical in league_map.items():
                alias_map[str(alias).strip().lower()] = canonical

        # (match_text, canonical_name) — canonical names + registry aliases,
        # all sorted longest-first so "Manchester City" wins over "Man City".
        candidates = [(n, n) for n in csv_names]
        for alias, canonical in alias_map.items():
            if canonical in csv_names:
                candidates.append((alias, canonical))
        candidates.sort(key=lambda c: len(c[0]), reverse=True)

        found: List[str] = []
        spans: List[tuple] = []
        for text, canonical in candidates:
            nl = text.lower()
            start, idx = 0, q.find(nl)
            while idx != -1:
                if not any(idx < s2 and idx + len(nl) > s1 for s1, s2 in spans):
                    found.append(canonical)
                    spans.append((idx, idx + len(nl)))
                    break
                start = idx + 1
                idx = q.find(nl, start)
            if len(found) >= 2:
                break

        # Dedupe + question order.
        ordered, seen = [], set()
        for _, canonical in sorted(zip(spans, found), key=lambda sf: sf[0][0]):
            if canonical not in seen:
                seen.add(canonical)
                ordered.append(canonical)

        if not ordered:
            cand = self.store.resolve_team(q)
            if cand:
                ordered.append(cand)
        return ordered[:2]

    def _extract_league(self, q: str) -> Optional[str]:
        if self._div_map is None:
            df = self.store.df
            if "Div" in df.columns and "League" in df.columns:
                self._div_map = df.groupby("Div")["League"].agg(
                    lambda s: s.dropna().mode()[0] if s.notna().any() else None
                ).to_dict()
            else:
                self._div_map = {}
        for code, name in self._div_map.items():
            if code.lower() in q:
                return name
        for name in self.store.league_names():
            if name.lower() in q:
                return name
            if name.lower().replace("_", " ") in q:
                return name
        return None

    def _extract_season(self, q: str) -> Optional[str]:
        seasons = set(self.store.df["Season"].dropna().astype(str).unique().tolist())
        for m in re.finditer(r"\b(?:20)?(\d{2})\s*[/\-–]\s*(\d{2})\b", q):
            tok = m.group(1) + m.group(2)
            if tok in seasons:
                return tok
        for s in seasons:
            if s in q:
                return s
        return None

    def _extract_stat(self, q: str) -> Optional[str]:
        for key, aliases in _STAT_ALIASES:
            for alias in aliases:
                if alias in q:
                    return key
        return None

    def _extract_top_n(self, q: str) -> int:
        for pat in (r"top\s+(\d+)", r"first\s+(\d+)", r"last\s+(\d+)",
                    r"(\d+)\s+(?:matches|games|teams|winners)"):
            m = re.search(pat, q)
            if m:
                n = int(m.group(1))
                if 1 <= n <= 50:
                    return n
        return 10


def _normalize(question: str) -> str:
    q = " " + " ".join(str(question).lower().split()) + " "
    q = q.replace("-", " ")
    q = re.sub(r"\s+", " ", q)
    return q


def _has_form_hint(q: str) -> bool:
    """'last 5 matches', 'recent 3 games', 'last ten' …"""
    if re.search(r"last\s+\d+\s+(?:match|matches|game|games|fixtures)", q):
        return True
    if re.search(r"recent\s+\d+\s+(?:match|matches|game|games)", q):
        return True
    return re.search(r"\b(?:last|past)\s+\d+\b", q) is not None
