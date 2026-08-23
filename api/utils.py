import json
import re


def strip_markdown(text: str) -> str:
    """Remove all markdown formatting from model responses for clean plain-text display."""
    text = re.sub(r'#{1,6}\s*', '', text)
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'\*(.+?)\*', r'\1', text)
    text = re.sub(r'\*{3}(.+?)\*{3}', r'\1', text)
    text = re.sub(r'`(.+?)`', r'\1', text)
    text = re.sub(r'^[\-=]{3,}\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


# ── Heuristic per-team analysis extraction ────────────────────────────────────

_STRENGTH_WORDS = (
    "strong", "strength", "solid", "domina", "effective", "capable", "potent",
    "clinical", "resilien", "superior", "quality", "momentum", "advantage",
    "in form", "excellent", "impressiv", "threat",
)
_WEAK_WORDS = (
    "weak", "vulnerab", "struggl", "conced", "leak", "poor", "fragil", "lack",
    "fail", "miss", "injur", "susceptible", "dreadful", "frail", "exposed",
    "disorganiz",
)
# Verdict / meta sentences are NOT strengths or weaknesses.
_META_WORDS = (
    "prediction", "predict", "points towards", "points firmly", "i am confident",
    "my analysis", "the underlying probabilit", "probabilities favor",
    "favour", "favor", "expected outcome", "this analysis", "verdict",
    "i believe", "executive summary", "clash of",
)


def _sentences(text: str):
    return [s.strip(" •*-–\t") for s in re.split(r"(?<=[.!?])\s+|\n+", text or "")]


# Clause splitter: lets comparative sentences ("X are strong, while Y struggle")
# contribute a bullet to EACH team instead of being discarded.
_CLAUSE_SPLIT = re.compile(
    r",?\s+\b(?:while|whilst|whereas|but|however|although|though|in contrast|"
    r"on the other hand|meanwhile)\b\s*|,\s*(?=and\s+(?:their|its|the)\b)|;\s*",
    re.IGNORECASE,
)


def _clauses(sentence: str):
    parts = [c.strip(" ,.;:") for c in _CLAUSE_SPLIT.split(sentence)]
    return [p for p in parts if len(p) >= 12] or ([sentence.strip()] if sentence.strip() else [])


def extract_team_analysis(text: str, home_team: str, away_team: str) -> dict:
    """
    Extract per-team strengths/weaknesses bullets from a FREE-TEXT tactical
    narrative (DENSE mode). Sentences are split into clauses on contrast
    conjunctions so a single comparative line can feed both teams. A clause
    is attributed when it mentions exactly one team and carries an explicit
    polarity keyword; meta/verdict commentary is still excluded.
    """
    out = {
        "home_team_analysis": {"strengths": [], "weaknesses": []},
        "away_team_analysis": {"strengths": [], "weaknesses": []},
    }
    if not text:
        return out

    def tokens(team: str):
        toks = [w for w in re.findall(r"[a-z]{4,}", (team or "").lower())]
        # well-known nicknames so e.g. "Wolves'" matches Wolverhampton
        aliases = {
            "wolverhampton": ["wolves"],
            "tottenham": ["spurs"],
        }
        for base, extras in aliases.items():
            if base in toks:
                toks.extend(extras)
        return list(dict.fromkeys(toks))

    h_toks, a_toks = tokens(home_team), tokens(away_team)

    def add(key: str, bucket: str, clause: str):
        lst = out[key][bucket]
        if len(lst) < 4 and clause not in lst:
            lst.append(clause)

    for s in _sentences(strip_markdown(text)):
        last_key = None  # pronoun clauses ("their potent press") inherit this
        for clause in _clauses(s):
            cl = clause.lower()
            if not (12 <= len(clause) <= 220):
                continue
            in_home = bool(h_toks) and any(t in cl for t in h_toks)
            in_away = bool(a_toks) and any(t in cl for t in a_toks)
            if in_home and in_away:
                key = None  # explicitly comparative clause — skip
            elif in_home:
                key = "home_team_analysis"
            elif in_away:
                key = "away_team_analysis"
            else:
                # possessive-pronoun clause continues the previous team's topic
                key = last_key if re.match(r"\s*(?:and|or|but|yet)?\s*(their|its|his|her)\b", cl) else None
            if key is None:
                continue
            is_weak = any(w in cl for w in _WEAK_WORDS)
            is_strong = any(w in cl for w in _STRENGTH_WORDS)
            if is_weak:
                add(key, "weaknesses", clause)
            elif is_strong:
                add(key, "strengths", clause)
            last_key = key
    return out


def get_phase_logger(name: str) -> "logging.Logger":
    """
    Logger that ALWAYS prints INFO phases regardless of uvicorn/root logging
    config: owns its StreamHandler and does not propagate.
    """
    import logging

    log = logging.getLogger(name)
    if not getattr(log, "_phase_handler_set", False):
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
        log.addHandler(h)
        log.setLevel(logging.INFO)
        log.propagate = False
        log._phase_handler_set = True
    return log


def extract_json_object(text: str):
    """Best-effort extraction of the first top-level JSON object from LLM output.

    Handles markdown code fences and surrounding prose. If the object is
    TRUNCATED (LLM hit its token budget mid-JSON), attempts to close the open
    braces/brackets at the last complete value so partial-but-useful content
    (e.g. match_state / analysis without the final recommendations array)
    is recovered instead of lost.
    Returns the parsed dict, or None when nothing usable is found.
    """
    if not text or not isinstance(text, str):
        return None
    start = text.find("{")
    if start == -1:
        return None

    def _loads(candidate: str):
        try:
            obj = json.loads(candidate)
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None

    # 1) Exact balanced object
    depth = 0
    in_str = False
    escaped = False
    end = -1
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if escaped:
                escaped = False
            elif c == "\\":
                escaped = True
            elif c == '"':
                in_str = False
        elif c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                end = i
                break
    if end != -1:
        obj = _loads(text[start:end + 1])
        if obj is not None:
            return obj

    # 2) Truncated object: brute-force repair. Walk the cut point backwards,
    #    close open brackets at each candidate position, and take the FIRST
    #    candidate that parses (recovers maximum content from LLM outputs that
    #    hit their token budget mid-JSON).
    frag = text[start:]
    limit = min(len(frag), len(frag))
    for k in range(0, 900):
        cut = len(frag) - k
        if cut <= 1:
            break
        cand = frag[:cut]
        # rescan candidate
        stack = []
        in_str = False
        escaped = False
        valid = True
        for c in cand:
            if in_str:
                if escaped:
                    escaped = False
                elif c == "\\":
                    escaped = True
                elif c == '"':
                    in_str = False
            else:
                if c == '"':
                    in_str = True
                elif c in "[{":
                    stack.append(c)
                elif c in "]}" :
                    if not stack:
                        valid = False
                        break
                    stack.pop()
        if not valid:
            continue
        tail = cand.rstrip()
        if not tail or tail[-1] in ",:":
            continue
        closers = "".join("]" if o == "[" else "}" for o in reversed(stack))
        obj = _loads(tail + ('"' if in_str else "") + closers)
        if obj is not None:
            return obj
    return None
