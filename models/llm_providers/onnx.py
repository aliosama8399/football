"""
ONNX LLM Provider — Qwen via Optimum ONNX Runtime / ONNX Runtime GenAI
======================================================================
Fast local execution for exported SLM weights.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
import yaml

logger = logging.getLogger(__name__)

DEFAULT_ONNX_DIR = Path(__file__).resolve().parent.parent / "export" / "slm"


class OnnxLLMProvider:
    """
    Local ONNX inference using Optimum ORTModelForCausalLM or onnxruntime-genai.
    Plugs into the existing LLM provider architecture.
    """

    provider_name = "onnx"

    def __init__(self, model_path: str | Path | None = None, max_new_tokens: int | None = None, temperature: float | None = None):
        self.model_path = Path(model_path) if model_path else self._resolve_onnx_path()
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        self._model = None
        self._tokenizer = None
        self._loaded = False
        self._backend = None  # "genai" | "ort" | "hf"

        cfg = self._load_cfg()
        if self.max_new_tokens is None:
            self.max_new_tokens = cfg.get("max_new_tokens", 2048)
        if self.temperature is None:
            self.temperature = cfg.get("temperature", 0.7)

    @staticmethod
    def _load_cfg() -> dict:
        cfg_path = Path(__file__).resolve().parent.parent / "llm_config.yaml"
        if not cfg_path.exists():
            return {}
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
                return data.get("providers", {}).get("onnx", {}) or data.get("providers", {}).get("huggingface", {})
        except Exception:
            return {}

    @staticmethod
    def _resolve_onnx_path() -> Path:
        env_path = os.getenv("FOOTBALL_ONNX_MODEL_PATH", "").strip()
        if env_path:
            return Path(env_path)
        cfg_path = Path(__file__).resolve().parent.parent / "llm_config.yaml"
        if cfg_path.exists():
            try:
                with open(cfg_path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
                    p = data.get("providers", {}).get("onnx", {}).get("model_path", "")
                    if p:
                        pp = Path(p)
                        if not pp.is_absolute():
                            pp = Path(__file__).resolve().parent.parent.parent / pp
                        return pp
            except Exception:
                pass
        return DEFAULT_ONNX_DIR

    # ── Lazy loading ──────────────────────────────────────────────────────────
    def _load(self):
        if self._loaded:
            return

        # ── Backend 1: onnxruntime-genai (CUDA, real-time) ────────────────────
        genai_cfg = self.model_path / "genai_config.json"
        if genai_cfg.exists():
            try:
                try:
                    import onnxruntime as _ort
                    providers = _ort.get_available_providers()
                    logger.info("[LLM] onnxruntime providers available: %s", providers)
                    if "CUDAExecutionProvider" not in providers:
                        logger.warning("[LLM] CUDAExecutionProvider NOT available — "
                                       "genai artifact '%s' was built for CUDA and will fail "
                                       "or fall back to CPU.", genai_cfg.parent.name)
                except Exception:
                    pass

                import numpy as np
                import onnxruntime_genai as og
                from transformers import AutoTokenizer

                logger.info("[LLM] Loading onnxruntime-genai (CUDA) artifact: %s", self.model_path)
                self._og, self._np = og, np
                self._gmodel = og.Model(str(self.model_path))
                self._gtok = og.Tokenizer(self._gmodel)
                self._tokenizer = AutoTokenizer.from_pretrained(
                    str(self.model_path), trust_remote_code=True
                )
                self._backend = "genai"
                self._is_onnx = True
                self._loaded = True
                device = "CUDA (GPU)" if os.getenv("FOOTBALL_ONNX_EP", "").lower() != "cpu" else "CPU"
                logger.info("[LLM] READY — backend=onnxruntime-genai | device=%s | artifact=%s",
                            device, self.model_path)
                return
            except Exception as e:
                logger.warning("[LLM] genai CUDA load failed (%s); trying optimum ORT path.",
                               type(e).__name__)

        # ── Backend 2: optimum ORTModelForCausalLM ────────────────────────────
        onnx_ok = self.model_path.exists() and (self.model_path / "model.onnx").exists()
        if onnx_ok:
            try:
                from transformers import AutoTokenizer
                from optimum.onnxruntime import ORTModelForCausalLM
                logger.info("Loading ONNX LLM from: %s", self.model_path)
                self._tokenizer = AutoTokenizer.from_pretrained(str(self.model_path), trust_remote_code=True)

                ep_choice = os.getenv("FOOTBALL_ONNX_EP", "cuda").strip().lower()
                providers = None
                if ep_choice != "default":
                    import onnxruntime as _ort
                    available = _ort.get_available_providers()
                    wanted = {"cuda": "CUDAExecutionProvider",
                              "cpu": "CPUExecutionProvider"}.get(ep_choice)
                    if wanted and wanted in available:
                        providers = [wanted, "CPUExecutionProvider"]
                kwargs = {}
                if providers:
                    kwargs["providers"] = providers
                    logger.info("ONNX LLM execution providers: %s", providers)
                else:
                    logger.info("ONNX LLM execution providers: default (%s)", ep_choice)
                self._model = ORTModelForCausalLM.from_pretrained(str(self.model_path), **kwargs)
                if getattr(self._model, "can_use_cache", False):
                    head_dim = getattr(self._model.config, "head_dim", None)
                    if head_dim and head_dim != getattr(self._model, "embed_size_per_head", None):
                        self._model.embed_size_per_head = head_dim
                        logger.info("Patched embed_size_per_head -> %s (config.head_dim)", head_dim)
                self._is_onnx = True
                self._loaded = True
                logger.info("ONNX LLM ready: %s", self.model_path)
                return
            except Exception as e:
                logger.warning("ONNX load failed (%s), falling back to HF torch: %s", type(e).__name__, e)

        # Fallback to HF torch (huggingface)
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        cfg_path = Path(__file__).resolve().parent.parent / "llm_config.yaml"
        hf_id = "aliosama8399/football-analysisN"
        hf_token = None
        if cfg_path.exists():
            try:
                with open(cfg_path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
                    hf_cfg = data.get("providers", {}).get("huggingface", {})
                    hf_id = hf_cfg.get("model_id", hf_id)
                    hf_token = hf_cfg.get("hf_token", "").strip() or os.getenv("HUGGINGFACE_HUB_TOKEN", "").strip() or None
            except Exception:
                pass

        logger.info("Loading fallback HF model: %s", hf_id)
        self._tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True, token=hf_token)
        self._model = AutoModelForCausalLM.from_pretrained(
            hf_id,
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
            token=hf_token,
        )
        if torch.cuda.is_available() and next(self._model.parameters()).device.type == "cpu":
            self._model = self._model.to("cuda")
        self._model.eval()
        self._is_onnx = False
        self._loaded = True
        logger.info("Fallback HF LLM ready on %s", next(self._model.parameters()).device)

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
        return self.generate(rag_prompt)

    # ── Core generation ───────────────────────────────────────────────────────
    def generate(self, prompt: str) -> str:
        self._load()

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert football tactical analyst. "
                    "Base your analysis strictly on the provided factual match data, head-to-head records, and tactical profiles. "
                    "When tactical data is complete, ground your reasoning in the numbers; if specific historical data points are absent, "
                    "use balanced analytical football reasoning without fabricating false scores, player names, or statistics."
                ),
            },
            {"role": "user", "content": prompt},
        ]

        try:
            text = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        except Exception:
            try:
                text = self._tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                text = prompt

        # ── genai CUDA fast path ─────────────────────────────────────────────
        if getattr(self, "_backend", None) == "genai":
            return self._generate_genai(text)

        inputs = self._tokenizer(text, return_tensors="pt")
        if not getattr(self, "_is_onnx", True):
            try:
                import torch
                device = next(self._model.parameters()).device
                inputs = {k: v.to(device) for k, v in inputs.items()}
            except Exception:
                pass

        import torch

        json_mode = ('"match_state"' in text) or ("SINGLE JSON" in text) or ("Tactical Analysis" in text)
        do_sample = (self.temperature > 0) and not json_mode
        use_cache = getattr(self._model, "can_use_cache", True) if hasattr(self, "_model") and self._model else True
        gen_kwargs = dict(
            max_new_tokens=self.max_new_tokens,
            temperature=max(self.temperature, 1e-4) if do_sample else 1.0,
            do_sample=do_sample,
            top_p=0.9 if do_sample else 1.0,
            repetition_penalty=1.15,
            no_repeat_ngram_size=3,
            pad_token_id=self._tokenizer.eos_token_id,
            use_cache=use_cache,
        )

        try:
            with torch.no_grad():
                outputs = self._model.generate(**inputs, **gen_kwargs)
        except Exception as e:
            if getattr(self, "_is_onnx", False):
                logger.warning("ONNX generate failed (%s), falling back to HF: %s", type(e).__name__, e)
                self._loaded = False
                self._is_onnx = False
                self._model = None
                self._tokenizer = None
                self._load()
                inputs = self._tokenizer(text, return_tensors="pt")
                try:
                    device = next(self._model.parameters()).device
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                except Exception:
                    pass
                with torch.no_grad():
                    outputs = self._model.generate(**inputs, **gen_kwargs)
            else:
                raise

        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def _generate_genai(self, text: str) -> str:
        og, np = self._og, self._np
        input_ids = self._gtok.encode(text)

        json_mode = ('"match_state"' in text) or ("SINGLE JSON" in text)
        do_sample = (self.temperature > 0) and not json_mode

        params = og.GeneratorParams(self._gmodel)
        params.set_search_options(
            max_length=len(input_ids) + self.max_new_tokens,
            do_sample=do_sample,
            temperature=max(self.temperature, 1e-4),
            repetition_penalty=1.1,
            no_repeat_ngram_size=3,
            top_p=0.9 if do_sample else 1.0,
        )
        generator = og.Generator(self._gmodel, params)
        generator.append_tokens(input_ids)

        out = ""
        while not generator.is_done():
            generator.generate_next_token()
            out += self._gtok.decode(np.asarray(generator.get_next_tokens(), dtype=np.int32))
        return out.strip()

    def generate_with_context(self, prompt: str, kg_context: str = "", vector_context: str = "") -> str:
        kg_budget = int(os.getenv("FOOTBALL_ONNX_KG_CTX", 0)) or 2000
        vec_budget = int(os.getenv("FOOTBALL_ONNX_VEC_CTX", 0)) or 1200
        try:
            cfg_path = Path(__file__).resolve().parent.parent / "llm_config.yaml"
            if cfg_path.exists():
                with open(cfg_path, "r", encoding="utf-8") as f:
                    _cfg = (yaml.safe_load(f) or {}).get("providers", {}).get("onnx", {})
                kg_budget = int(_cfg.get("kg_context_chars", kg_budget))
                vec_budget = int(_cfg.get("vector_context_chars", vec_budget))
        except Exception:
            pass

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
        return self.generate(rag_prompt)

    # ── Compatibility shims ───────────────────────────────────────────────────
    def _call_api(self, prompt: str) -> str:
        return self.generate(prompt)

    def generate_explanation(self, match_context: dict, gnn_explanation: dict) -> str:
        from models.llm_providers.base import BaseLLMProvider
        prompt = BaseLLMProvider._build_prompt(self, match_context, gnn_explanation)
        return self.generate(prompt)

    def analyze_match(self, match_data: dict) -> str:
        prompt = (
            f"Analyze: {match_data.get('home_team', '?')} vs {match_data.get('away_team', '?')}. "
            f"Prediction: {match_data.get('prediction', '?')}."
        )
        return self.generate(prompt)
