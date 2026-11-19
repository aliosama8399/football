"""
vLLM Provider
=============
vLLM sidecar serving the finetuned HF model directly (OpenAI-compatible).
"""

from __future__ import annotations

import logging
import os
import re

from api.utils import clean_slm_output
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
        self.temperature = float(cfg.get("temperature", _gen_cfg().get("temperature", 0.2)))
        self.max_tokens  = int(cfg.get("max_tokens", _gen_cfg().get("max_tokens", 2048)))
        if not self.api_url.endswith("/v1"):
            self.api_url += "/v1"
        try:
            from openai import OpenAI
            self._client = OpenAI(base_url=self.api_url, api_key="not-needed")
        except ImportError:
            raise ImportError("pip install openai")

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        return self._call_api(prompt, system_prompt=system_prompt)

    def generate_with_context(
        self,
        prompt: str,
        kg_context: str = "",
        vector_context: str = "",
        system_prompt: str | None = None,
    ) -> str:
        # Model context limit is 2048 tokens. Budget contexts conservatively so
        # there is room for detailed generation without vLLM 400 Bad Request errors.
        kg_budget = int(os.getenv("FOOTBALL_ONNX_KG_CTX", 0)) or 1500
        vec_budget = int(os.getenv("FOOTBALL_ONNX_VEC_CTX", 0)) or 400
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
        return self._call_api(rag_prompt, system_prompt=system_prompt)

    def _call_api(self, prompt: str, system_prompt: str | None = None) -> str:
        logger.info("[LLM:VLLM] -> %s model=%s timeout=%d", self.api_url, self.model_name, self.timeout)
        system_content = system_prompt or (
            "You are an expert football tactical analyst. "
            "Base your analysis strictly on the provided factual match data, head-to-head records, and tactical profiles. "
            "When tactical data is complete, ground your reasoning in the numbers; if specific historical data points are absent, "
            "use balanced analytical football reasoning without fabricating false scores, player names, or statistics."
        )

        # Dynamic token budget clamp: ensure input_tokens + max_tokens <= 2048
        approx_chars = len(system_content) + len(prompt)
        est_input_tokens = int(approx_chars / 2.6) + 45
        safe_output_tokens = max(256, min(self.max_tokens, 2036 - est_input_tokens))

        try:
            response = self._client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
                max_tokens=safe_output_tokens,
                timeout=self.timeout,
            )
        except Exception as exc:
            err_msg = str(exc)
            if "maximum context length" not in err_msg and "input_tokens" not in err_msg:
                raise
            match = re.search(r"(?:at least|value=)\s*(\d+)", err_msg)
            actual_in = int(match.group(1)) if match else est_input_tokens + 200
            retry_tokens = max(128, min(self.max_tokens, 2032 - actual_in))
            logger.warning(
                "[LLM:VLLM] Context length retry: input_tokens=%d -> max_tokens=%d",
                actual_in,
                retry_tokens,
            )
            response = self._client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
                max_tokens=retry_tokens,
                timeout=self.timeout,
            )
        out = clean_slm_output(response.choices[0].message.content or "")
        logger.info("[LLM:VLLM] done -> %d chars", len(out))
        return out
