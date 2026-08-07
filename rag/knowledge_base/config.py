"""
Knowledge base settings — loaded once from models/llm_config.yaml (rag.kb_*),
with sane defaults when the file or keys are missing.
"""

from pathlib import Path
import yaml

_BASE_DIR = Path(__file__).resolve().parent.parent.parent
_CFG_PATH = _BASE_DIR / "models" / "llm_config.yaml"

_DEFAULTS = {
    "kb_csv_path": "data/processed/processed_matches.csv",
    "kb_tactics_path": "rag/knowledge_base/team_tactics.json",
    "kb_index_analyses": False,
    "kb_team_cache_ttl": 300,
    "kb_csv_fallback": True,
}

_cache = None


def kb_settings() -> dict:
    """Read-only dict of knowledge-base settings (cached per process)."""
    global _cache
    if _cache is None:
        merged = dict(_DEFAULTS)
        try:
            with open(_CFG_PATH, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            rag = cfg.get("rag", {}) or {}
            for key in _DEFAULTS:
                if key in rag:
                    merged[key] = rag[key]
        except Exception:
            pass
        _cache = merged
    return _cache
