"""
LLM Provider Interfaces for Explainable AI (XAI)
=================================================
Config-driven architecture: model names live in llm_config.yaml;
API keys live in the git-ignored .env file (see .env.example).

Priority for API keys:
  1. Environment variable  (e.g. OPENAI_API_KEY — loaded from .env)
  2. llm_config.yaml       (api_key field — keep empty; env-first)
  3. Raises ValueError with a clear message

Priority for model names (at runtime):
  1. --model-name CLI argument
  2. llm_config.yaml       (default_model field)

Adding a new provider:
  1. Add its block to llm_config.yaml
  2. Subclass BaseLLMProvider below and implement _call_api()
  3. Register it in LLM_REGISTRY at the bottom — done.
"""

import os
import json
import yaml
from abc import ABC, abstractmethod
from pathlib import Path

# ── Secrets: load .env (git-ignored) so os.getenv sees LLM API keys ───────────

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)
except Exception:
    pass

# ── Config loader ─────────────────────────────────────────────────────────────

_CONFIG_PATH = Path(__file__).parent / "llm_config.yaml"


def _load_config() -> dict:
    """Load llm_config.yaml once. Returns empty dict if file is missing."""
    if not _CONFIG_PATH.exists():
        return {}
    with open(_CONFIG_PATH, "r") as f:
        return yaml.safe_load(f) or {}


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
        f"  * Edit config:  llm_config.yaml -> providers.{provider_name}.api_key"
    )


def _resolve_model(provider_name: str, override: str | None) -> str:
    """
    Resolve model name with priority: override arg -> config file -> hard-coded fallback.
    """
    if override:
        return override
    cfg_model = _provider_cfg(provider_name).get("default_model", "").strip()
    if cfg_model:
        return cfg_model
    # Hard-coded fallbacks — last resort, should not normally be reached
    fallbacks = {
        "ollama":    "llama3.2",
        "openai":    "gpt-4o-mini",
        "gemini":    "gemini-2.0-flash",
        "anthropic": "claude-haiku-4-5-20251001",
    }
    return fallbacks.get(provider_name, "unknown-model")


# ── Base class ────────────────────────────────────────────────────────────────

class BaseLLMProvider(ABC):
    """
    Abstract base for all LLM/SLM providers.
    Subclasses only need to implement _call_api(prompt) -> str.
    generate_explanation() and _build_prompt() are inherited and shared.
    """

    provider_name: str = "base"  # Override in each subclass

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
        """Constructs the standard XAI prompt. Override in a subclass to customise."""
        home  = match_context["home_team"]
        away  = match_context["away_team"]
        pred  = match_context["prediction"]
        probs = match_context.get("probabilities", {"H": 0.0, "D": 0.0, "A": 0.0})

        node_feats = json.dumps(gnn_explanation.get('top_node_features', {}), indent=2)
        hist_matches = json.dumps(gnn_explanation.get('top_influencing_matches', []), indent=2)

        return f"""You are an elite football tactical analyst API. Focus on deeply analytical insights backed by the provided quantitative data.

INPUT DATA:
FIXTURE: {home} (Home) vs {away} (Away)
PREDICTION: {pred}
PROBABILITIES: Home Win {probs.get('H', 0):.1%} | Draw {probs.get('D', 0):.1%} | Away Win {probs.get('A', 0):.1%}

KEY STATISTICAL INDICATORS (recent 5-match rolling form):
{node_feats}

KEY HISTORICAL CONTEXT (influential recent encounters):
{hist_matches}

INSTRUCTIONS:
You must output ONLY valid JSON. Do not include markdown codeblocks or any conversational text. Adhere strictly to this schema:
{{
  "prediction_verdict": "Clear, professional thesis stating why the predicted outcome is expected (1-2 sentences)",
  "confidence_rating": "High, Medium, or Low based on the probability",
  "home_team_analysis": {{
    "strengths": ["list 2-3 specific strengths based on stats"],
    "weaknesses": ["list 1-2 specific weaknesses"]
  }},
  "away_team_analysis": {{
    "strengths": ["list 2-3 specific strengths based on stats"],
    "weaknesses": ["list 1-2 specific weaknesses"]
  }},
  "tactical_matchup_summary": "A concise paragraph describing how the teams' styles and form will interact."
}}"""




# ── Concrete providers ────────────────────────────────────────────────────────

class OllamaProvider(BaseLLMProvider):
    """Local SLM via Ollama (Llama, Mistral, Phi, Gemma, ...)."""

    provider_name = "ollama"

    def __init__(self, model_name: str = "", api_url: str = "", **kwargs):
        cfg = _provider_cfg("ollama")
        self.model_name = _resolve_model("ollama", model_name or None)
        base_url = api_url or cfg.get("api_url", "http://localhost:11434/api/chat")
        # Newer Ollama versions removed /api/generate (HTTP 410); normalise to /api/chat.
        if base_url.rstrip("/").endswith("/api/generate"):
            base_url = base_url[: base_url.rfind("/api/generate")] + "/api/chat"
        self.api_url = base_url
        self.timeout    = cfg.get("timeout", 5000000000)
        try:
            import requests
            self._requests = requests
        except ImportError:
            raise ImportError("pip install requests")

    def _call_api(self, prompt: str) -> str:
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }
        try:
            r = self._requests.post(self.api_url, json=payload, timeout=self.timeout)
            if r.status_code in (404, 410):
                # Fall back to the legacy generate endpoint if chat is unavailable.
                legacy = self.api_url.replace("/api/chat", "/api/generate")
                if legacy != self.api_url:
                    r = self._requests.post(legacy, json={"model": self.model_name, "prompt": prompt, "stream": False}, timeout=self.timeout)
            r.raise_for_status()
            data = r.json()
            # /api/chat returns 'message.content'; /api/generate returns 'response'
            return data.get("message", {}).get("content") or data.get("response", "")
        except self._requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"Cannot reach Ollama at {self.api_url}.\n"
                f"  Start it with: ollama serve"
            )


class OpenAIProvider(BaseLLMProvider):
    """OpenAI GPT models (gpt-4o, gpt-4o-mini, o1-mini, ...)."""

    provider_name = "openai"

    def __init__(self, model_name: str = "", api_key: str = "", **kwargs):
        self.model_name  = _resolve_model("openai", model_name or None)
        self.api_key     = _resolve_api_key("openai", "OPENAI_API_KEY", api_key or None)
        self.timeout     = _provider_cfg("openai").get("timeout", 60)
        self.temperature = _gen_cfg().get("temperature", 0.7)
        self.max_tokens  = _gen_cfg().get("max_tokens", 2048)
        try:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key)
        except ImportError:
            raise ImportError("pip install openai")

    def _call_api(self, prompt: str, json_mode: bool = True) -> str:
        """
        Call the OpenAI chat API. json_mode forces a JSON object reply
        (fine-tuned tactical-analysis flow); set False for free-text narration.
        """
        kwargs = dict(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a professional football tactical analyst API designed to format inputs as JSON."},
                {"role": "user",   "content": prompt},
            ],
            temperature=self.temperature,
            timeout=self.timeout,
        )
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        # gpt-5.x / o-series models use max_completion_tokens; older models
        # use max_tokens. Try the modern parameter, fall back on a 400.
        try:
            kwargs["max_completion_tokens"] = self.max_tokens
            response = self._client.chat.completions.create(**kwargs)
        except Exception as e:
            err = str(e)
            if "max_completion_tokens" in err:
                kwargs.pop("max_completion_tokens", None)
                kwargs["max_tokens"] = self.max_tokens
                response = self._client.chat.completions.create(**kwargs)
            else:
                raise
        return response.choices[0].message.content


class GeminiProvider(BaseLLMProvider):
    """Google Gemini models (gemini-2.0-flash, gemini-1.5-pro, ...)."""

    provider_name = "gemini"

    def __init__(self, model_name: str = "", api_key: str = "", **kwargs):
        self.model_name  = _resolve_model("gemini", model_name or None)
        self.api_key     = _resolve_api_key("gemini", "GEMINI_API_KEY", api_key or None)
        self.timeout     = _provider_cfg("gemini").get("timeout", 60)
        self.temperature = _gen_cfg().get("temperature", 0.7)
        self.max_tokens  = _gen_cfg().get("max_tokens", 2048)

        # Try stable SDK first, fall back to preview SDK
        self._sdk = None
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self._client = genai.GenerativeModel(
                self.model_name,
                generation_config={
                    "temperature": self.temperature,
                    "max_output_tokens": self.max_tokens,
                    "response_mime_type": "application/json",
                },
            )
            self._sdk = "generativeai"
        except ImportError:
            try:
                from google import genai as genai_new
                self._client = genai_new.Client(api_key=self.api_key)
                self._sdk = "genai"
            except ImportError:
                raise ImportError(
                    "Install a Gemini SDK:\n"
                    "  pip install google-generativeai   (recommended)\n"
                    "  pip install google-genai          (preview)"
                )

    def _call_api(self, prompt: str) -> str:
        if self._sdk == "generativeai":
            response = self._client.generate_content(prompt)
            return response.text
            return response.text
        # Fallback for the newer google.genai SDK
        response = self._client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config={"response_mime_type": "application/json", "temperature": self.temperature, "max_output_tokens": self.max_tokens}
        )
        return response.text


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude models (claude-opus-4-6, claude-sonnet-4-6, claude-haiku-4-5, ...)."""

    provider_name = "anthropic"

    def __init__(self, model_name: str = "", api_key: str = "", **kwargs):
        self.model_name  = _resolve_model("anthropic", model_name or None)
        self.api_key     = _resolve_api_key("anthropic", "ANTHROPIC_API_KEY", api_key or None)
        self.timeout     = _provider_cfg("anthropic").get("timeout", 60)
        self.temperature = _gen_cfg().get("temperature", 0.7)
        self.max_tokens  = _gen_cfg().get("max_tokens", 1024)
        try:
            import anthropic
            self._client = anthropic.Anthropic(api_key=self.api_key)
        except ImportError:
            raise ImportError("pip install anthropic")

    def _call_api(self, prompt: str) -> str:
        message = self._client.messages.create(
            model=self.model_name,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text


# ── Registry & factory ────────────────────────────────────────────────────────
#
#  To add a new provider (e.g. Mistral, Cohere, Azure OpenAI):
#    1. Add a block for it in llm_config.yaml
#    2. Subclass BaseLLMProvider and implement _call_api()
#    3. Add one line here:  'myprovider': MyProvider
#  No other file needs changing.
#
#  HuggingFaceProvider is a standalone class (not a BaseLLMProvider subclass
#  because it does local inference). It is imported here and registered under
#  'huggingface' so get_llm_provider("huggingface") works seamlessly.
#
try:
    from models.hf_provider import HuggingFaceProvider as _HFProvider
    _HF_AVAILABLE = True
except ImportError:
    _HFProvider = None
    _HF_AVAILABLE = False

try:
    from models.onnx_llm_provider import OnnxLLMProvider as _OnnxProvider
    _ONNX_AVAILABLE = True
except ImportError:
    _OnnxProvider = None
    _ONNX_AVAILABLE = False


LLM_REGISTRY: dict[str, type] = {
    "ollama":       OllamaProvider,
    "openai":       OpenAIProvider,
    "gemini":       GeminiProvider,
    "anthropic":    AnthropicProvider,
    "onnx":         _OnnxProvider,   # ONNX export of Qwen3-0.6B (models/export/slm) — replaces huggingface
    "huggingface":  _OnnxProvider,   # alias: any huggingface request now serves ONNX (per user request)
}


def get_llm_provider(provider_type: str = "", **kwargs) -> BaseLLMProvider:
    """
    Factory function. Returns a ready-to-use provider instance.

    If provider_type is empty, reads default_provider from llm_config.yaml.
    kwargs forwarded to the provider: model_name, api_key, api_url (all optional).

    None-safe: passing "" / None / "none" returns None (LLM disabled) instead
    of raising, so callers can gracefully degrade when no LLM is configured.
    """
    if not provider_type or str(provider_type).strip().lower() in ("", "none", "null"):
        return None

    prov_clean = str(provider_type).strip().lower()

    if prov_clean in ("huggingface", "onnx") and _OnnxProvider is None:
        raise ValueError(
            "ONNX provider 'onnx' is registered but unavailable because "
            "optimum/onnxruntime could not be imported. Run: conda run -n football pip install \"optimum[onnx]\""
        )

    if prov_clean not in LLM_REGISTRY:
        raise ValueError(
            f"Invalid LLM provider '{provider_type}'. Supported providers: {sorted(LLM_REGISTRY.keys())}.\n"
            f"Check your rag.llm_provider setting in models/llm_config.yaml."
        )

    cls = LLM_REGISTRY[prov_clean]
    return cls(**kwargs)