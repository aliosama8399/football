"""
LLM Provider Package for Explainable AI (XAI) & RAG Narrative
==============================================================
Modular, file-per-provider architecture:
  - base.py       -> BaseLLMProvider & config helpers
  - ollama.py     -> OllamaProvider
  - openai.py     -> OpenAIProvider
  - gemini.py     -> GeminiProvider
  - anthropic.py  -> AnthropicProvider
  - trtllm.py     -> TrtLLMProvider (TensorRT-LLM sidecar)
  - vllm.py       -> VllmProvider (vLLM sidecar)
  - onnx.py       -> OnnxLLMProvider (Optimum / ONNX Runtime GenAI)
  - hf.py         -> HuggingFaceProvider (Transformers / PyTorch)
"""

from __future__ import annotations

import logging
from typing import Any

from models.llm_providers.base import (
    BaseLLMProvider,
    _load_config,
    _provider_cfg,
    _gen_cfg,
    _resolve_api_key,
    _resolve_model,
    resolve_default_provider,
)
from models.llm_providers.ollama import OllamaProvider
from models.llm_providers.openai import OpenAIProvider
from models.llm_providers.gemini import GeminiProvider
from models.llm_providers.anthropic import AnthropicProvider
from models.llm_providers.trtllm import TrtLLMProvider
from models.llm_providers.vllm import VllmProvider
from models.llm_providers.onnx import OnnxLLMProvider
from models.llm_providers.hf import HuggingFaceProvider

logger = logging.getLogger(__name__)

LLM_REGISTRY: dict[str, type] = {
    "ollama":       OllamaProvider,
    "openai":       OpenAIProvider,
    "gemini":       GeminiProvider,
    "anthropic":    AnthropicProvider,
    "huggingface":  HuggingFaceProvider,
    "onnx":         OnnxLLMProvider,
    "trtllm":       TrtLLMProvider,
    "vllm":         VllmProvider,
}


def get_llm_provider(provider_type: str = "", **kwargs: Any) -> Any:
    """
    Factory function. Returns a ready-to-use provider instance.

    If provider_type is empty, reads default_provider from models/llm_config.yaml.
    kwargs forwarded to the provider constructor.

    None-safe: passing "" / None / "none" returns None (LLM disabled).
    """
    if not provider_type or str(provider_type).strip().lower() in ("", "none", "null"):
        if str(provider_type).strip().lower() == "none":
            return None
        provider_type = resolve_default_provider()
        if not provider_type or str(provider_type).strip().lower() in ("", "none", "null"):
            return None

    prov_clean = str(provider_type).strip().lower()

    if prov_clean not in LLM_REGISTRY:
        raise ValueError(
            f"Invalid LLM provider '{provider_type}'. Supported providers: {sorted(LLM_REGISTRY.keys())}.\n"
            f"Check your rag.llm_provider setting in models/llm_config.yaml."
        )

    cls = LLM_REGISTRY[prov_clean]
    return cls(**kwargs)


__all__ = [
    "BaseLLMProvider",
    "OllamaProvider",
    "OpenAIProvider",
    "GeminiProvider",
    "AnthropicProvider",
    "TrtLLMProvider",
    "VllmProvider",
    "OnnxLLMProvider",
    "HuggingFaceProvider",
    "LLM_REGISTRY",
    "get_llm_provider",
    "resolve_default_provider",
    "_load_config",
    "_provider_cfg",
    "_gen_cfg",
    "_resolve_api_key",
    "_resolve_model",
]
