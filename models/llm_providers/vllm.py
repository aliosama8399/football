"""
vLLM Provider
=============
vLLM sidecar serving the finetuned HF model directly (OpenAI-compatible).
"""

from __future__ import annotations

import logging
import os
from models.llm_providers.base import (
    BaseLLMProvider,
    _gen_cfg,
    _provider_cfg,
)

logger = logging.getLogger(__name__)


class VllmProvider(BaseLLMProvider):
    """vLLM sidecar serving the finetuned HF model directly (no export).

    OpenAI-compatible endpoint from `vllm serve`. Runs via the compose
    sidecar: `docker compose --profile vllm up -d vllm`.

    Config (models/llm_config.yaml -> providers.vllm):
      api_url:    http://localhost:8356/v1   (host) — VLLM_API_URL env wins
      model_name: served model id (must match the name vllm registers)
      timeout:    300
    """

    provider_name = "vllm"

    def __init__(self, model_name: str = "", api_url: str = "", **kwargs):
        cfg = _provider_cfg("vllm")
        self.model_name  = (model_name or cfg.get("model_name", "")).strip() or \
            cfg.get("default_model", "aliosama8399/football-analysisN")
        self.api_url     = (
            os.environ.get("VLLM_API_URL")
            or api_url
            or cfg.get("api_url", "")
        ).strip().rstrip("/") or "http://localhost:8356/v1"
        self.timeout     = cfg.get("timeout", 300)
        self.temperature = _gen_cfg().get("temperature", 0.7)
        self.max_tokens  = _gen_cfg().get("max_tokens", 2048)
        if not self.api_url.endswith("/v1"):
            self.api_url += "/v1"
        try:
            from openai import OpenAI
            self._client = OpenAI(base_url=self.api_url, api_key="not-needed")
        except ImportError:
            raise ImportError("pip install openai")

    def generate(self, prompt: str) -> str:
        return self._call_api(prompt)

    def generate_with_context(self, prompt: str, kg_context: str = "", vector_context: str = "") -> str:
        kg_budget = int(os.getenv("FOOTBALL_ONNX_KG_CTX", 0)) or 8000
        vec_budget = int(os.getenv("FOOTBALL_ONNX_VEC_CTX", 0)) or 2500
        if kg_budget and len(kg_context) > kg_budget:
            kg_context = kg_context[:kg_budget].rsplit(" ", 1)[0] + " ..."
        if vec_budget and len(vector_context) > vec_budget:
            vector_context = vector_context[:vec_budget].rsplit(" ", 1)[0] + " ..."

        rag_prompt = ""
        if kg_context:
            rag_prompt += f"## Retrieved Knowledge Graph Context\n{kg_context}\n\n"
        if vector_context:
            rag_prompt += f"## Retrieved Historical Analyses\n{vector_context}\n\n"
        rag_prompt += f"## User Question\n{prompt}"
        return self._call_api(rag_prompt)

    def _call_api(self, prompt: str) -> str:
        logger.info("[LLM:VLLM] -> %s model=%s timeout=%d", self.api_url, self.model_name, self.timeout)
        system_content = (
            "You are an expert football tactical analyst. "
            "Base your analysis strictly on the provided factual match data, head-to-head records, and tactical profiles. "
            "When tactical data is complete, ground your reasoning in the numbers; if specific historical data points are absent, "
            "use balanced analytical football reasoning without fabricating false scores, player names, or statistics."
        )
        response = self._client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user",   "content": prompt}
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout=self.timeout,
        )
        out = response.choices[0].message.content
        logger.info("[LLM:VLLM] done -> %d chars", len(out or ""))
        return out
