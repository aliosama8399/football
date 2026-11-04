"""
Base LLM Provider Interface and Configuration Utilities
========================================================
Config-driven architecture: model names live in models/llm_config.yaml;
API keys live in the git-ignored .env file.
"""

from __future__ import annotations

import json
import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
import yaml

logger = logging.getLogger(__name__)

# ── Secrets: load .env (git-ignored) so os.getenv sees LLM API keys ───────────

try:
    from dotenv import load_dotenv
    _ENV_PATH = Path(__file__).resolve().parent.parent.parent / ".env"
    load_dotenv(_ENV_PATH, override=False)
except Exception:
    pass

# ── Config loader ─────────────────────────────────────────────────────────────

_CONFIG_PATH = Path(os.getenv("LLM_CONFIG_PATH", Path(__file__).resolve().parent.parent / "llm_config.yaml"))


def _load_config() -> dict:
    """Load llm_config.yaml once. Returns empty dict if file is missing."""
    if not _CONFIG_PATH.exists():
        return {}
    try:
        with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        logger.warning("Failed to load %s: %s", _CONFIG_PATH, e)
        return {}


def _provider_cfg(provider_name: str) -> dict:
    """Return the config block for a specific provider (never raises)."""
    cfg = _load_config()
    return cfg.get("providers", {}).get(provider_name, {})


def _gen_cfg() -> dict:
    """Return the generation parameters block."""
    return _load_config().get("generation", {})


def _resolve_api_key(provider_name: str, env_var: str, override: str | None) -> str:
    """
    Resolve API key with priority: override arg -> env var -> config file.
    Raises ValueError with a clear fix message if nothing is found.
    """
    if override:
        return override
    env_val = os.getenv(env_var, "").strip()
    if env_val:
        return env_val
    cfg_val = _provider_cfg(provider_name).get("api_key", "").strip()
    if cfg_val:
        return cfg_val
    raise ValueError(
        f"No API key found for '{provider_name}'.\n"
        f"Fix one of:\n"
        f"  * Set env var:  export {env_var}=your-key\n"
        f"  * Edit config:  models/llm_config.yaml -> providers.{provider_name}.api_key"
    )


def _resolve_model(provider_name: str, override: str | None) -> str:
    """
    Resolve model name with priority: override arg -> env var -> config file -> hard-coded fallback.
    """
    if override:
        return override
    env_model = os.getenv(f"{provider_name.upper()}_MODEL", "").strip()
    if env_model:
        return env_model
    cfg_model = _provider_cfg(provider_name).get("default_model", "").strip()
    if cfg_model:
        return cfg_model
    fallbacks = {
        "ollama":    "llama3.2",
        "openai":    "gpt-4o-mini",
        "gemini":    "gemini-3.8-flash",
        "anthropic": "claude-haiku-4-5-20251001",
    }
    return fallbacks.get(provider_name, "unknown-model")


# ── Base class ────────────────────────────────────────────────────────────────

class BaseLLMProvider(ABC):
    """
    Abstract base for all LLM/SLM providers.
    Subclasses implement _call_api(prompt) -> str.
    generate_explanation() and _build_prompt() are shared across providers.
    """

    provider_name: str = "base"

    def generate_explanation(self, match_context: dict, gnn_explanation: dict) -> str:
        """Public entry point: builds the prompt, calls the API, returns text."""
        prompt = self._build_prompt(match_context, gnn_explanation)
        try:
            return self._call_api(prompt)
        except Exception as e:
            return f"[{self.provider_name.upper()}] API error: {e}"

    @abstractmethod
    def _call_api(self, prompt: str) -> str:
        """Send prompt to the LLM and return the raw text response."""
        pass

    def _build_prompt(self, match_context: dict, gnn_explanation: dict) -> str:
        """Constructs the standard XAI prompt. Override in a subclass to customize."""
        home  = match_context["home_team"]
        away  = match_context["away_team"]
        pred  = match_context["prediction"]
        probs = match_context.get("probabilities", {"H": 0.0, "D": 0.0, "A": 0.0})

        node_feats = json.dumps(gnn_explanation.get('top_node_features', {}), indent=2)
        hist_matches = json.dumps(gnn_explanation.get('top_influencing_matches', []), indent=2)
        raw_stats = match_context.get("raw_stats_json", "")
        stats_block = f"\nDETAILED STATISTICAL & ENVIRONMENTAL PROFILE:\n{raw_stats}\n" if raw_stats else ""

        return f"""You are an elite football tactical analyst delivering a comprehensive professional pre-match scouting report.

FIXTURE: {home} (Home) vs {away} (Away)
AI MODEL PREDICTION: {pred}
PROBABILITY DISTRIBUTION: Home Win {probs.get('H', 0):.1%} | Draw {probs.get('D', 0):.1%} | Away Win {probs.get('A', 0):.1%}
{stats_block}
KEY STATISTICAL INDICATORS (recent 5-match rolling form):
{node_feats}

KEY HISTORICAL CONTEXT (influential recent encounters):
{hist_matches}

INSTRUCTIONS:
Write a comprehensive, deeply analytical tactical dossier (4 substantial, detailed paragraphs, ~500-700 words total).
Do NOT output JSON dictionaries or bullet points. Output pure narrative prose with these exact 4 section headings:

### {home} vs {away}: Tactical Analysis & Prediction

**HEADLINE & VERDICT**
Write a robust, authoritative paragraph stating the predicted outcome ({pred}) and the core tactical thesis. Detail why the data supports this outcome, contrasting current momentum, form trajectories, home/away dynamics, and stadium atmosphere.

**STRENGTHS & WEAKNESSES OF {home.upper()}**
Write an in-depth analytical paragraph examining {home}'s tactical identity. Detail their offensive creation volume, shot conversion, xG metrics, and pressing structure based on the data. Contrast with their defensive organization, concession rates, and disciplinary liabilities (fouls/cards). Keep this paragraph focused on {home}.

**STRENGTHS & WEAKNESSES OF {away.upper()}**
Write an in-depth analytical paragraph examining {away}'s tactical identity. Analyze their attacking transition speed, defensive block compactness, and resilience when absorbing pressure based on the statistics. Highlight specific away-match vulnerabilities. Keep this paragraph focused on {away}.

**TACTICAL SYNTHESIS & WHAT COULD CHANGE THE OUTCOME**
Write a comprehensive synthesis paragraph detailing how the tactical systems clash on the pitch. Analyze territorial control, transitional game-states, set-pieces, environmental conditions, and the referee's strictness profile, identifying specific catalysts that could disrupt or confirm the predicted outcome."""
