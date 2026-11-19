import json
import re


_SPECIAL_TOKENS_RE = re.compile(
    r"<\|(?:im_start|im_end|endoftext|assistant|user|system|pad)\|>",
    re.IGNORECASE,
)
_EXPERT1_ECHO_RE = re.compile(
    r"={2,}\s*Expert\s*1\s*\([^)]*\)\s*Prediction\s*={2,}\s*\n?"
    r"(?:Predicted Outcome:[^\n]*\n?)?"
    r"(?:Probabilities:[^\n]*\n?)?"
    r"(?:={3,}\s*\n?)?",
    re.IGNORECASE,
)
_PROMPT_ECHO_LINE_RE = re.compile(
    r"^\s*(?:===\s*Reasoning\s*===|"
    r"##\s*Retrieved\s+(?:Knowledge\s+Graph\s+Context|Historical\s+Analyses)|"
    r"##\s*User\s+Question|"
    r"You are Expert 2 \(Fine-tuned LLM\)\..*)\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def clean_slm_output(text: str) -> str:
    """
    Sanitize SLM output by removing thinking tags, chat special tokens,
    echoed RAG/Expert-1 prompt wrappers, and degenerate repetition loops
    without shrinking max_tokens.
    """
    if not text or not isinstance(text, str):
        return ""

    cleaned = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE)
    cleaned = re.sub(r"<think>[\s\S]*$", "", cleaned, flags=re.IGNORECASE)
    cleaned = _SPECIAL_TOKENS_RE.sub("", cleaned)
    cleaned = _EXPERT1_ECHO_RE.sub("", cleaned)
    cleaned = _PROMPT_ECHO_LINE_RE.sub("", cleaned)

    # Collapse pathological consecutive duplicate lines (keep up to 1 occurrence when repeated)
    dedup_lines = []
    last_norm = None
    repeat_count = 0
    for line in cleaned.splitlines():
        norm = re.sub(r"\s+", " ", line).strip().lower()
        if norm and norm == last_norm:
            repeat_count += 1
            if repeat_count >= 1:
                continue
        else:
            last_norm = norm if norm else None
            repeat_count = 0
        dedup_lines.append(line)
    cleaned = "\n".join(dedup_lines)

    # Collapse 3+ consecutive identical sentences within a paragraph
    out_paragraphs = []
    for para in re.split(r"\n{2,}", cleaned):
        sentences = re.split(r"(?<=[.!?])\s+", para.strip())
        kept_sentences = []
        prev_s = None
        run_len = 0
        for s in sentences:
            s_norm = re.sub(r"\s+", " ", s).strip().lower()
            if s_norm and s_norm == prev_s:
                run_len += 1
                if run_len >= 1:
                    continue
            else:
                prev_s = s_norm if s_norm else None
                run_len = 0
            if s:
                kept_sentences.append(s)
        if kept_sentences:
            out_paragraphs.append(" ".join(kept_sentences))

    cleaned = "\n\n".join(out_paragraphs)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def is_rag_fallback_payload(obj: object) -> bool:
    """Return True when obj is an internal RAG degraded-context dump rather than real analysis."""
    if not isinstance(obj, dict):
        return False
    if obj.get("fallback") is True:
        return True
    return "question" in obj and "kg_context" in obj and "note" in obj


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
    "in form", "excellent", "impressiv", "threat", "high-intensity", "disciplined",
    "minimizing", "clean sheet", "shot volume", "xg", "sot", "shots-on-target",
    "pace", "structured", "consistent", "control", "low foul",
)
_WEAK_WORDS = (
    "weak", "vulnerab", "struggl", "tendency to concede", "conced", "leak", "poor",
    "fragil", "lack of", "lacks", "lacking", "instability", "inability",
    "low-quality", "collapse", "high foul", "foul count", "conceding foul",
    "excessive foul", "susceptible", "dreadful", "frail", "exposed", "behind",
    "disorganiz", "limited", "fail to", "unable to disrupt", "outplayed",
    "frustrat", "porous", "overwhelmed", "less clinical",
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
# contribute a bullet to EACH team instead of being discarded, without splitting
# parenthetical adverbs mid-clause ("The main weakness, however, lies in ...").
_CLAUSE_SPLIT = re.compile(
    r",?\s+\b(?:while|whilst|whereas|but|although|though|in contrast|"
    r"on the other hand|meanwhile|conversely)\b\s*,?\s*"
    r"|,?\s+\bhowever\b\s*,?\s*(?!\s*(?:lies|is|was|are|has|have|remains|comes|can|will|may|could|would|should)\b)"
    r"|,\s*(?=and\s+(?:their|its|the)\b)|;\s*",
    re.IGNORECASE,
)


def _clauses(sentence: str):
    parts = [c.strip(" ,.;:") for c in _CLAUSE_SPLIT.split(sentence)]
    kept = [p for p in parts if len(p) >= 12 and len(re.findall(r"[A-Za-z0-9']+", p)) >= 4]
    return kept or ([sentence.strip()] if sentence.strip() and len(re.findall(r"[A-Za-z0-9']+", sentence)) >= 4 else [])


def _team_tokens(team: str) -> list[str]:
    toks = [w for w in re.findall(r"[a-z]{4,}", (team or "").lower())]
    aliases = {
        "wolverhampton": ["wolves"],
        "tottenham": ["spurs"],
    }
    for base, extras in aliases.items():
        if base in toks:
            toks.extend(extras)
    return list(dict.fromkeys(toks))


_SECTION_HEADING_RE = re.compile(
    r"^(?:\d+[\)\.]\s*)?(?:"
    r"match\s+verdict(?:\s*(?:&|and)\s*probability\s+integration)?|"
    r"headline\s*(?:&|and)\s*verdict|"
    r".*?\btactical\s+strengths?\s*(?:&|and)\s*weaknesses|"
    r"strengths?\s*(?:&|and)\s*weaknesses\s+of\s+.+|"
    r"key\s+tactical\s+battlegrounds.*|"
    r"tactical\s+synthesis.*"
    r")[\.\:\s]*$",
    re.IGNORECASE,
)
_OPPONENT_PREP_RE = re.compile(
    r"(?:against|versus|vs\.?|over|than|facing|stopping)\s+(?:the\s+)?$",
    re.IGNORECASE,
)
_NEGATED_STRENGTH_PHRASES = (
    "not as effective",
    "not as robust",
    "not as well-organized",
    "not negligible",
    "prone to conceding",
    "defensive cost",
    "defensive concession",
    "comes at a",
    "at a defensive cost",
    "not immune to",
)
_CONTINUATION_SUBJECT_RE = re.compile(
    r"\s*(?:and|or|but|yet|while|whilst)?\s*"
    r"(?:their|they|them|its|this|these|the\s+(?:main|primary|team|club|side|attack|defense|defensive))\b",
    re.IGNORECASE,
)


def _is_section_heading(text: str) -> bool:
    cleaned = text.strip(" \t\"'•*-–.:#")
    if not cleaned:
        return True
    return bool(_SECTION_HEADING_RE.match(cleaned))


def _is_incomplete_fragment(text: str) -> bool:
    """Return True if a trailing SLM slice is mid-parenthesis or stops on a dangling function word."""
    stripped = (text or "").strip()
    if not stripped:
        return True
    if stripped.count("(") != stripped.count(")"):
        return True
    return bool(
        re.search(
            r"(?:\(\d+|\b(?:to|for|of|in|at|by|with|the|a|an|their|its|and|or)\.?)\s*$",
            stripped,
            re.IGNORECASE,
        )
    )


def _first_team_match_idx(cl: str, full_name: str, unique_toks: list[str], all_toks: list[str]) -> int | None:
    if full_name:
        for m in re.finditer(rf"\b{re.escape(full_name)}\b", cl):
            if not _OPPONENT_PREP_RE.search(cl[:m.start()]):
                return m.start()
    candidates = unique_toks if unique_toks else all_toks
    best: int | None = None
    for tok in candidates:
        for m in re.finditer(rf"\b{re.escape(tok)}\b", cl):
            if _OPPONENT_PREP_RE.search(cl[:m.start()]):
                continue
            if best is None or m.start() < best:
                best = m.start()
            break
    return best


def _attribute_clause_team(
    cl: str,
    home_low: str,
    away_low: str,
    h_toks: list[str],
    a_toks: list[str],
    h_unique: list[str],
    a_unique: list[str],
) -> str | None:
    h_idx = _first_team_match_idx(cl, home_low, h_unique, h_toks)
    a_idx = _first_team_match_idx(cl, away_low, a_unique, a_toks)
    if h_idx is not None and a_idx is None:
        return "home_team_analysis"
    if a_idx is not None and h_idx is None:
        return "away_team_analysis"
    if h_idx is not None and a_idx is not None:
        if any(p in cl[:45] for p in ("between ", "clash", "both teams", "neither")):
            return None
        first_idx, second_idx = (h_idx, a_idx) if h_idx < a_idx else (a_idx, h_idx)
        if first_idx < 50 and second_idx >= 50 and (second_idx - first_idx) >= 25:
            return "home_team_analysis" if h_idx < a_idx else "away_team_analysis"
        return None
    return None


def _classify_polarity(cl: str) -> str | None:
    weak_hits = sum(1 for w in _WEAK_WORDS if w in cl)
    strong_hits = sum(1 for w in _STRENGTH_WORDS if w in cl)
    neg_hits = sum(1 for p in _NEGATED_STRENGTH_PHRASES if p in cl)
    if neg_hits:
        weak_hits += neg_hits * 2
        strong_hits = max(0, strong_hits - neg_hits)
    if "disrupting opposing" in cl or "disrupt opponent" in cl:
        strong_hits += 1
    if "cleansheets=" in cl and weak_hits > 0:
        # Negated clean-sheet mention in a concession clause ("tendency to concede shots (CleanSheets=0.59)")
        strong_hits = max(0, strong_hits - 1)
    if weak_hits > 0 and weak_hits >= strong_hits:
        return "weaknesses"
    if strong_hits > weak_hits:
        return "strengths"
    if any(m in cl for m in _META_WORDS):
        return None
    return "strengths" if any(m in cl for m in ("xg", "shot", "goal", "corner", "defens", "attack", "press")) else None


def extract_team_analysis(text: str, home_team: str, away_team: str) -> dict:
    """
    Extract per-team strengths/weaknesses bullets from a FREE-TEXT tactical
    narrative. Handles shared-city clubs (Manchester City vs Manchester United),
    section headings, multi-team comparative clauses, and negated strength phrasing.
    """
    out = {
        "home_team_analysis": {"strengths": [], "weaknesses": []},
        "away_team_analysis": {"strengths": [], "weaknesses": []},
    }
    if not text:
        return out

    h_toks, a_toks = _team_tokens(home_team), _team_tokens(away_team)
    h_unique = [t for t in h_toks if t not in a_toks]
    a_unique = [t for t in a_toks if t not in h_toks]
    home_low, away_low = (home_team or "").lower().strip(), (away_team or "").lower().strip()

    def add(key: str, bucket: str, clause: str) -> None:
        lst = out[key][bucket]
        if len(lst) < 4 and clause not in lst:
            lst.append(clause)

    section_key: str | None = None
    last_key: str | None = None
    for s in _sentences(strip_markdown(text)):
        if _is_section_heading(s):
            sec_key = _attribute_clause_team(s.lower(), home_low, away_low, h_toks, a_toks, h_unique, a_unique)
            section_key = sec_key
            last_key = sec_key
            continue
        for clause in _clauses(s):
            if _is_section_heading(clause):
                sec_key = _attribute_clause_team(clause.lower(), home_low, away_low, h_toks, a_toks, h_unique, a_unique)
                section_key = sec_key
                last_key = sec_key
                continue
            cl = clause.lower()
            if not (12 <= len(clause) <= 350) or _is_incomplete_fragment(clause):
                continue
            key = _attribute_clause_team(cl, home_low, away_low, h_toks, a_toks, h_unique, a_unique)
            if key is None:
                if _CONTINUATION_SUBJECT_RE.match(cl):
                    key = last_key or section_key
                elif section_key is not None:
                    key = section_key
            if key is None:
                continue
            bucket = _classify_polarity(cl)
            if bucket:
                add(key, bucket, clause)
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


_SIDE_WEAK_WORDS = (
    "weak", "vulnerab", "struggl", "conced", "leak", "poor", "fragil", "lack",
    "fail", "inability", "susceptible", "limited", "error", "expos",
)
_CONFIDENCE_KV_RE = re.compile(
    r"""['"]?\$?(?:confidence(?:_rating|_level)?|certainty)['"]?\s*[:=,]\s*['"]?(?:High|Medium|Low|[0-9]+(?:\.[0-9]+)?%?)['"]?\s*,?""",
    re.IGNORECASE,
)
_QUOTED_KEY_RE = re.compile(
    r"""['"]\$?[a-zA-Z_][a-zA-Z0-9_\-]*['"]\s*[:=,]\s*['"]?"""
)
_SEMI_QUOTED_KEY_RE = re.compile(
    r"""(?<![a-zA-Z0-9_])\$?[a-zA-Z_][a-zA-Z0-9_\-]*['"]\s*[:=,]\s*['"]?"""
)
_SNAKE_OR_DOLLAR_KEY_RE = re.compile(
    r"""(?<![a-zA-Z0-9_])(?:\$[a-zA-Z_][a-zA-Z0-9_]*|[a-zA-Z]+_[a-zA-Z0-9_]+)\s*[:=,]\s*['"]?"""
)
_TRAILING_KEY_FRAG_RE = re.compile(
    r"""(?:^|[\s,{])['"]?(?:\$[a-zA-Z_][a-zA-Z0-9_]*|[a-zA-Z]+_[a-zA-Z0-9_]+)['"]?\s*[:=,]?\s*['"]?\s*$""",
    re.MULTILINE,
)
_KV_PAIR_RE = re.compile(
    r"""(?:['"](\$?[a-zA-Z_][a-zA-Z0-9_\-]*)['"]|(?<![a-zA-Z0-9_])(\$?[a-zA-Z_][a-zA-Z0-9_\-]*)['"]|(?<![a-zA-Z0-9_])(\$[a-zA-Z_][a-zA-Z0-9_]*|[a-zA-Z]+_[a-zA-Z0-9_]+))\s*[:=,]\s*"((?:[^"\\\n]|\\.)+?)(?:"|\s*$)"""
)


def _is_snake_tag(text: str) -> bool:
    return bool(re.fullmatch(r"\$?[a-z0-9_]+", text.strip().lower()))


def _strip_code_fences(text: str) -> str:
    cleaned = re.sub(r"```(?:json|markdown|md)?\s*", "", text, flags=re.IGNORECASE)
    return cleaned.replace("```", "")


def _repair_missing_object_colons(text: str) -> str:
    """Replace `\"key\", <value>` with `\"key\": <value>` inside JSON objects without altering arrays."""
    out: list[str] = []
    containers: list[str] = []
    expect_stack: list[str] = []  # expect_key | expect_colon | expect_value | expect_comma
    in_str = False
    escaped = False
    i = 0
    n = len(text)

    while i < n:
        ch = text[i]
        if in_str:
            out.append(ch)
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
                if containers and containers[-1] == "{" and expect_stack:
                    if expect_stack[-1] == "expect_key":
                        expect_stack[-1] = "expect_colon"
                    elif expect_stack[-1] == "expect_value":
                        expect_stack[-1] = "expect_comma"
            i += 1
            continue

        if ch == '"':
            in_str = True
            out.append(ch)
            i += 1
            continue

        if ch in " \t\r\n":
            out.append(ch)
            i += 1
            continue

        if ch == "{":
            containers.append("{")
            expect_stack.append("expect_key")
            out.append(ch)
            i += 1
            continue

        if ch == "[":
            containers.append("[")
            out.append(ch)
            i += 1
            continue

        if ch == "}":
            if containers and containers[-1] == "{":
                containers.pop()
                if expect_stack:
                    expect_stack.pop()
                if containers and containers[-1] == "{" and expect_stack and expect_stack[-1] == "expect_value":
                    expect_stack[-1] = "expect_comma"
            out.append(ch)
            i += 1
            continue

        if ch == "]":
            if containers and containers[-1] == "[":
                containers.pop()
                if containers and containers[-1] == "{" and expect_stack and expect_stack[-1] == "expect_value":
                    expect_stack[-1] = "expect_comma"
            out.append(ch)
            i += 1
            continue

        if containers and containers[-1] == "{" and expect_stack:
            state = expect_stack[-1]
            if state == "expect_colon":
                if ch in (":", ",", "="):
                    out.append(":")
                    expect_stack[-1] = "expect_value"
                    i += 1
                    continue
                # Missing colon altogether before scalar
                out.append(": ")
                expect_stack[-1] = "expect_value"
                continue
            if state == "expect_comma" and ch == ",":
                out.append(ch)
                expect_stack[-1] = "expect_key"
                i += 1
                continue
            if state == "expect_value" and ch == ",":
                out.append(ch)
                expect_stack[-1] = "expect_key"
                i += 1
                continue

        out.append(ch)
        i += 1

    return "".join(out)


def repair_slm_json_text(text: str) -> str:
    """Repair common 0.6B SLM JSON slips (bad quotes, `$` key prefixes, `,` instead of `:`)."""
    if not text or not isinstance(text, str):
        return ""
    repaired = _strip_code_fences(text)
    repaired = re.sub(
        r"""(?<=[\{\,\n])\s*['"]?(\$?[a-zA-Z_][a-zA-Z0-9_\-]*)['"]?\s*[:=]\s*""",
        lambda m: f'\n"{m.group(1).lstrip("$")}": ',
        repaired,
    )
    repaired = _repair_missing_object_colons(repaired)
    # Collapse accidental blank value lines between keys inside objects (`"verdict": \n "other":`)
    repaired = re.sub(
        r':\s*(?=\n\s*"[^"\n]+"\s*:)',
        ': "", ',
        repaired,
    )
    # Remove trailing commas before closing `}` or `]`
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
    return repaired


def sanitize_slm_prose(text: str, keep_markdown: bool = False, min_words: int = 4) -> str:
    """Strip JSON braces, key labels, and truncated tokens so only clean readable prose remains."""
    if not text or not isinstance(text, str):
        return ""

    cleaned = clean_slm_output(text)
    cleaned = _strip_code_fences(cleaned)
    if not keep_markdown:
        cleaned = strip_markdown(cleaned)

    cleaned = (
        cleaned.replace('\\"', '"')
        .replace("\\n", "\n")
        .replace("\\t", " ")
    )
    cleaned = _CONFIDENCE_KV_RE.sub("", cleaned)
    cleaned = _QUOTED_KEY_RE.sub("", cleaned)
    cleaned = _SEMI_QUOTED_KEY_RE.sub("", cleaned)
    cleaned = _SNAKE_OR_DOLLAR_KEY_RE.sub("", cleaned)
    cleaned = _TRAILING_KEY_FRAG_RE.sub("", cleaned)
    cleaned = re.sub(r"[{}\[\]]+", "\n", cleaned)

    if keep_markdown:
        md_lines: list[str] = []
        seen_lines: set[str] = set()
        for raw_line in cleaned.splitlines():
            line = raw_line.strip(' \t",;')
            if not line or _is_snake_tag(line) or line.lower() in ("high", "medium", "low"):
                continue
            words = re.findall(r"[A-Za-z0-9']+", line)
            if not line.startswith("#") and not line.startswith("-") and len(words) < 2:
                continue
            norm = re.sub(r"\s+", " ", line).lower()
            if norm not in seen_lines:
                seen_lines.add(norm)
                md_lines.append(line)
        return re.sub(r"\n{3,}", "\n\n", "\n".join(md_lines)).strip()

    kept: list[str] = []
    seen_sents: set[str] = set()
    for chunk in re.split(r"\n+", cleaned):
        for cand in re.split(r"(?<=[.!?])\s+", chunk.strip()):
            sent = cand.strip(' \t",;•*-–')
            if sent.startswith("'"):
                sent = sent.lstrip("'")
            sent = sent.strip()
            if (
                not sent
                or _is_snake_tag(sent)
                or _is_section_heading(sent)
                or _is_incomplete_fragment(sent)
                or sent.lower() in ("high", "medium", "low")
            ):
                continue
            words = re.findall(r"[A-Za-z][A-Za-z0-9'\-]*", sent)
            if len(words) < min_words:
                continue
            if sent[-1] not in ".!?":
                sent += "."
            norm = re.sub(r"\s+", " ", sent).lower()
            if norm not in seen_sents:
                seen_sents.add(norm)
                kept.append(sent)
    return " ".join(kept)


def _merge_parsed_dicts(target: dict, source: dict) -> dict:
    for k, v in source.items():
        if isinstance(v, dict) and isinstance(target.get(k), dict):
            _merge_parsed_dicts(target[k], v)
        elif k not in target or not target[k]:
            if v not in (None, "", [], {}):
                target[k] = v
    return target


def extract_json_object(text: str):
    """Best-effort extraction of top-level JSON objects from SLM output.

    Scans for ALL `{...}` blocks (merging fields when an SLM emits consecutive
    objects like `{...}\\n{...}`), applies syntax repair on 0.6B slips (such as
    `\"key\", \"val\"` instead of `\"key\": \"val\"`), and repairs truncated objects
    when generation stopped mid-JSON. Returns the merged dict or None.
    """
    if not text or not isinstance(text, str):
        return None
    if text.find("{") == -1:
        return None

    def _loads(candidate: str):
        try:
            obj = json.loads(candidate)
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None

    def _parse_candidate(candidate: str):
        obj = _loads(candidate)
        if obj is not None:
            return obj
        return _loads(repair_slm_json_text(candidate))

    def _recover_truncated(frag: str):
        variants = [frag]
        rep = repair_slm_json_text(frag)
        if rep != frag:
            variants.append(rep)
        for source in variants:
            for k in range(0, min(900, len(source))):
                cut = len(source) - k
                if cut <= 1:
                    break
                cand = source[:cut]
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
                        elif c in "]}":
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

    merged: dict = {}
    pos = 0
    n = len(text)
    while pos < n:
        start = text.find("{", pos)
        if start == -1:
            break
        depth = 0
        in_str = False
        escaped = False
        end = -1
        for i in range(start, n):
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
            obj = _parse_candidate(text[start:end + 1])
            if obj is not None:
                _merge_parsed_dicts(merged, obj)
            else:
                rec = _recover_truncated(text[start:end + 1])
                if rec is not None:
                    _merge_parsed_dicts(merged, rec)
            pos = end + 1
        else:
            rec = _recover_truncated(text[start:])
            if rec is not None:
                _merge_parsed_dicts(merged, rec)
            break

    return merged if merged else None


def _extract_kv_pairs_regex(text: str) -> dict:
    """Fallback regex extractor for quoted values inside broken/unclosed JSON objects."""
    found: dict = {}
    for m in _KV_PAIR_RE.finditer(text):
        key = (m.group(1) or m.group(2) or m.group(3) or "").lstrip("$").strip()
        val = (m.group(4) or "").replace('\\"', '"').strip()
        if key and val and (key not in found or not found[key]):
            found[key] = val
    return found


def _strip_brace_blocks(text: str) -> str:
    """Remove top-level `{...}` blocks and any trailing unclosed `{...` fragment."""
    out: list[str] = []
    pos = 0
    n = len(text)
    while pos < n:
        start = text.find("{", pos)
        if start == -1:
            out.append(text[pos:])
            break
        out.append(text[pos:start])
        depth = 0
        in_str = False
        escaped = False
        end = -1
        for i in range(start, n):
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
        if end == -1:
            break
        pos = end + 1
    return "\n".join(part for part in out if part.strip())


def _normalize_team_side_block(block) -> dict:
    """Normalize dict/string/list SLM per-team blocks into sanitized strengths/weaknesses."""
    out = {"strengths": [], "weaknesses": []}
    if not block:
        return out

    def _add(bucket: str, item_text: str) -> None:
        cleaned = sanitize_slm_prose(item_text, min_words=2).rstrip(".")
        if item_text.strip().endswith((".", "!", "?")) and cleaned and cleaned[-1] not in ".!?":
            cleaned += "."
        if cleaned and cleaned not in out[bucket] and len(out[bucket]) < 4:
            out[bucket].append(cleaned)

    def _clean_list(raw_items) -> list[str]:
        if isinstance(raw_items, str):
            raw_items = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", raw_items) if s.strip()]
        if not isinstance(raw_items, list):
            return []
        res: list[str] = []
        for elem in raw_items:
            txt = sanitize_slm_prose(str(elem), min_words=2)
            if not str(elem).strip().endswith((".", "!", "?")) and txt.endswith("."):
                txt = txt[:-1]
            if txt and txt not in res:
                res.append(txt)
        return res

    if isinstance(block, dict):
        for k, v in block.items():
            kn = re.sub(r"[^a-z0-9]", "", str(k).lower())
            if any(w in kn for w in ("strength", "pro", "advantage")):
                for s in _clean_list(v):
                    _add("strengths", s)
            elif any(w in kn for w in ("weak", "con", "vulnerab", "risk")):
                for w in _clean_list(v):
                    _add("weaknesses", w)
            elif any(w in kn for w in ("evaluat", "summar", "analys", "descrip")):
                for sent in _clean_list(v):
                    if len(sent) < 12:
                        continue
                    target = "weaknesses" if any(m in sent.lower() for m in _SIDE_WEAK_WORDS) else "strengths"
                    _add(target, sent)
        if not out["strengths"] and isinstance(block.get("attributes"), list):
            for attr in block["attributes"][:4]:
                pretty = str(attr).replace("_", " ").strip().capitalize()
                if pretty and pretty not in out["strengths"]:
                    out["strengths"].append(pretty)
        return out

    for sent in _clean_list(block):
        if len(sent) < 12:
            continue
        target = "weaknesses" if any(m in sent.lower() for m in _SIDE_WEAK_WORDS) else "strengths"
        _add(target, sent)
    return out


def parse_slm_tactical_response(raw_text: str, home_team: str = "", away_team: str = "") -> dict:
    """Unified 4-layer parser converting noisy 0.6B SLM outputs into clean tactical fields."""
    cleaned = clean_slm_output(raw_text or "")
    parsed_json = extract_json_object(cleaned)
    if is_rag_fallback_payload(parsed_json):
        err_hint = parsed_json.get("error") or parsed_json.get("note") or "SLM inference unavailable"
        return {
            "is_fallback": True,
            "fallback_error": str(err_hint),
            "prediction_verdict": "",
            "confidence_rating": "",
            "tactical_matchup_summary": "",
            "home_team_analysis": {"strengths": [], "weaknesses": []},
            "away_team_analysis": {"strengths": [], "weaknesses": []},
        }

    merged: dict = {}
    if isinstance(parsed_json, dict):
        _merge_parsed_dicts(merged, parsed_json)
    _merge_parsed_dicts(merged, _extract_kv_pairs_regex(cleaned))

    verdicts: list[str] = []
    summaries: list[str] = []
    extra_prose: list[str] = []
    confidence = ""
    home_block: dict = {"strengths": [], "weaknesses": []}
    away_block: dict = {"strengths": [], "weaknesses": []}

    def _merge_side(dest: dict, incoming: dict) -> None:
        for bkt in ("strengths", "weaknesses"):
            for item in incoming.get(bkt, []):
                if item not in dest[bkt] and len(dest[bkt]) < 4:
                    dest[bkt].append(item)

    def _append_unique(target: list[str], txt: str) -> None:
        clean = sanitize_slm_prose(txt)
        if clean and clean not in target:
            target.append(clean)

    def _walk(node) -> None:
        nonlocal confidence
        if not isinstance(node, dict):
            return
        for raw_k, val in node.items():
            k_norm = re.sub(r"[^a-z0-9]", "", str(raw_k).lower())
            if k_norm in ("attributes", "fallback", "note", "error", "gnnprediction", "hometeam", "awayteam"):
                continue
            if "confid" in k_norm or k_norm in ("rating", "certainty"):
                m_conf = re.search(r"\b(high|medium|low)\b", str(val or ""), re.IGNORECASE)
                if m_conf and not confidence:
                    confidence = m_conf.group(1).capitalize()
                continue
            if "home" in k_norm and any(
                w in k_norm for w in ("team", "analysis", "strength", "weak", "pro", "con", "factor", "profile", "side")
            ):
                _merge_side(home_block, _normalize_team_side_block(val))
                if isinstance(val, dict):
                    for ek in ("evaluation", "summary", "analysis", "description"):
                        if isinstance(val.get(ek), str):
                            _append_unique(extra_prose, val[ek])
                elif isinstance(val, str):
                    _append_unique(extra_prose, val)
                continue
            if "away" in k_norm and any(
                w in k_norm for w in ("team", "analysis", "strength", "weak", "pro", "con", "factor", "profile", "side")
            ):
                _merge_side(away_block, _normalize_team_side_block(val))
                if isinstance(val, dict):
                    for ek in ("evaluation", "summary", "analysis", "description"):
                        if isinstance(val.get(ek), str):
                            _append_unique(extra_prose, val[ek])
                elif isinstance(val, str):
                    _append_unique(extra_prose, val)
                continue
            if isinstance(val, dict):
                _walk(val)
                continue
            val_text = " ".join(str(x) for x in val) if isinstance(val, list) else str(val or "")
            if any(
                s in k_norm for s in ("verdict", "verity", "prediction", "outcome", "winner", "conclusion", "result")
            ) and "probabilit" not in k_norm:
                _append_unique(verdicts, val_text)
            elif any(
                s in k_norm for s in ("summary", "matchup", "tactical", "evaluation", "analysis", "reasoning", "overview", "narrative")
            ):
                _append_unique(summaries, val_text)
            else:
                _append_unique(extra_prose, val_text)

    if merged:
        _walk(merged)

    if not confidence:
        m_raw_conf = _CONFIDENCE_KV_RE.search(cleaned)
        if m_raw_conf:
            m_lvl = re.search(r"\b(high|medium|low)\b", m_raw_conf.group(0), re.IGNORECASE)
            if m_lvl:
                confidence = m_lvl.group(1).capitalize()

    outside_prose = sanitize_slm_prose(_strip_brace_blocks(cleaned))
    if outside_prose:
        _append_unique(summaries, outside_prose)

    verdict_txt = sanitize_slm_prose(" ".join(verdicts[:1])) if verdicts else ""
    summary_parts = list(summaries)
    for part in extra_prose:
        if part and part not in summary_parts:
            summary_parts.append(part)
    if verdict_txt and verdict_txt not in summary_parts and len(summary_parts) < 2:
        summary_parts.append(verdict_txt)
    if not summary_parts:
        fallback_prose = sanitize_slm_prose(cleaned)
        if fallback_prose:
            summary_parts.append(fallback_prose)

    summary_txt = sanitize_slm_prose(" ".join(dict.fromkeys(summary_parts)))
    if not verdict_txt and summary_txt:
        first_sent = re.split(r"(?<=[.!?])\s+", summary_txt, maxsplit=1)[0].strip()
        if len(re.findall(r"[A-Za-z0-9']+", first_sent)) >= 4:
            verdict_txt = first_sent

    if home_team or away_team:
        heur_source = f"{verdict_txt}\n{outside_prose or summary_txt}" if "{" not in cleaned else f"{verdict_txt}\n{summary_txt}"
        if "{" not in cleaned and cleaned:
            heur_source = cleaned
        heur = extract_team_analysis(heur_source, home_team, away_team)
        _merge_side(home_block, heur.get("home_team_analysis", {}))
        _merge_side(away_block, heur.get("away_team_analysis", {}))

    return {
        "is_fallback": False,
        "fallback_error": "",
        "prediction_verdict": verdict_txt,
        "confidence_rating": confidence,
        "tactical_matchup_summary": summary_txt,
        "home_team_analysis": home_block,
        "away_team_analysis": away_block,
    }
