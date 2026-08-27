"""
ONNX LLM Provider — Qwen3-0.6B via Optimum ONNX Runtime
=======================================================
Replaces HuggingFaceProvider (hf_provider.py) with ONNX inference.

Source model:  aliosama8399/football-analysisN  (merged finetune)
Artifact:      models/export/slm/model.onnx  (+ tokenizer files)
Docs:          https://huggingface.co/docs/transformers/serialization

Mirrors HuggingFaceProvider interface so rag/rag_orchestrator.py and
models/llm_providers.py work with zero changes beyond registry.
"""

import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Default ONNX export dir (from models/export_slm.py)
DEFAULT_ONNX_DIR = Path(__file__).parent / "export" / "slm"


class OnnxLLMProvider:
    """
    Local ONNX inference using Optimum ORTModelForCausalLM.
    Plugs into the existing LLM provider architecture.
    """

    provider_name = "onnx"

    def __init__(self, model_path: str | Path | None = None, max_new_tokens: int | None = None, temperature: float | None = None):
        # Resolve ONNX dir: explicit arg -> llm_config.yaml -> default
        self.model_path = Path(model_path) if model_path else self._resolve_onnx_path()
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        self._model = None
        self._tokenizer = None
        self._loaded = False
        self._backend = None  # "genai" | "ort" | "hf" (set in _load)

        # Lazy-load generation params from llm_config.yaml if not overridden
        cfg = self._load_cfg()
        if self.max_new_tokens is None:
            self.max_new_tokens = cfg.get("max_new_tokens", 2048)
        if self.temperature is None:
            self.temperature = cfg.get("temperature", 0.7)

    @staticmethod
    def _load_cfg() -> dict:
        import yaml
        cfg_path = Path(__file__).parent / "llm_config.yaml"
        if not cfg_path.exists():
            return {}
        try:
            with open(cfg_path) as f:
                data = yaml.safe_load(f) or {}
                # Support both providers.onnx and providers.huggingface fallback
                onnx_cfg = data.get("providers", {}).get("onnx", {}) or data.get("providers", {}).get("huggingface", {})
                return onnx_cfg
        except Exception:
            return {}

    @staticmethod
    def _resolve_onnx_path() -> Path:
        # 1) explicit env override (used by docker-compose)
        env_path = os.getenv("FOOTBALL_ONNX_MODEL_PATH", "").strip()
        if env_path:
            return Path(env_path)
        import yaml
        cfg_path = Path(__file__).parent / "llm_config.yaml"
        if cfg_path.exists():
            try:
                with open(cfg_path) as f:
                    data = yaml.safe_load(f) or {}
                    p = data.get("providers", {}).get("onnx", {}).get("model_path", "")
                    if p:
                        pp = Path(p)
                        if not pp.is_absolute():
                            pp = Path(__file__).parent.parent / pp
                        return pp
            except Exception:
                pass
        return DEFAULT_ONNX_DIR

    # ── Lazy loading ──────────────────────────────────────────────────────────
    def _load(self):
        if self._loaded:
            return

        # ── Backend 1: onnxruntime-genai (CUDA, real-time) ────────────────────
        # Artifact built by onnxruntime_genai.models.builder (-p fp16 -e cuda):
        # has genai_config.json. Runs the KV-cache loop in C++ on GPU —
        # ~60-70 tok/s vs ~1 tok/s for optimum's CPU ORTModel path.
        genai_cfg = self.model_path / "genai_config.json"
        if genai_cfg.exists():
            try:
                # Report GPU visibility BEFORE loading so misconfig is obvious
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
                # transformers tokenizer only for apply_chat_template formatting
                self._tokenizer = AutoTokenizer.from_pretrained(
                    str(self.model_path), trust_remote_code=True)
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

        # ── Backend 2: optimum ORTModelForCausalLM (CPU by default) ──────────
        onnx_ok = self.model_path.exists() and (self.model_path / "model.onnx").exists()
        if onnx_ok:
            try:
                from transformers import AutoTokenizer
                from optimum.onnxruntime import ORTModelForCausalLM
                logger.info("Loading ONNX LLM from: %s", self.model_path)
                self._tokenizer = AutoTokenizer.from_pretrained(str(self.model_path), trust_remote_code=True)
                # Execution provider: CUDA by default when available (CPU decode of a
                # 0.6B fp16 model is ~10x slower than GPU). Override with
                # FOOTBALL_ONNX_EP=cpu|cuda|default.
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
                # optimum bug (modeling_decoder.py): embed_size_per_head is computed as
                # hidden_size // num_attention_heads (=64) for qwen3, ignoring
                # config.head_dim (=128) that the exported graph was traced with ->
                # "past_key_values.N.key Got 64 Expected 128" at generate time.
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

        # Fallback to HF torch (huggingface) — ensures predict_match still has LLM reasoning
        # even when ONNX export is missing or broken (e.g., optimum post-processing bug)
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import yaml

        cfg_path = Path(__file__).parent / "llm_config.yaml"
        hf_id = "aliosama8399/football-analysisN"
        hf_token = None
        if cfg_path.exists():
            try:
                with open(cfg_path) as f:
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

    # ── Core generation ───────────────────────────────────────────────────────
    def generate(self, prompt: str) -> str:
        self._load()

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert football tactical analyst. "
                    "Analyze the match, predict the most likely outcome, and deliver "
                    "a detailed tactical report covering both teams' strengths, "
                    "weaknesses, and key strategic factors."
                ),
            },
            {"role": "user", "content": prompt},
        ]

        try:
            # enable_thinking=False skips Qwen3's <think> reasoning block:
            # saves hundreds of tokens per response (2-5x faster + cleaner).
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
        # HF fallback needs inputs on model device
        if not getattr(self, "_is_onnx", True):
            try:
                import torch
                device = next(self._model.parameters()).device
                inputs = {k: v.to(device) for k, v in inputs.items()}
            except Exception:
                pass

        import torch

        # use_cache follows the loaded artifact: True for the merged with-past
        # export (fast, low-memory decode), False for a no-past export
        # ("text-generation" task). Harmless for the HF fallback path.
        json_mode = ('"match_state"' in text) or ("SINGLE JSON" in text) or ("Tactical Analysis" in text)
        do_sample = (self.temperature > 0) and not json_mode
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
            # ORT generate failed (e.g., past dim mismatch) -> fallback to HF torch
            if getattr(self, "_is_onnx", False):
                logger.warning("ONNX generate failed (%s), falling back to HF: %s", type(e).__name__, e)
                # force reload as HF
                self._loaded = False
                self._is_onnx = False
                # clear and reload as HF
                self._model = None
                self._tokenizer = None
                self._load()  # will now take HF path (since onnx missing or still broken, it will still try onnx then fallback)
                # if still onnx, force HF directly
                if getattr(self, "_is_onnx", False):
                    # manual HF load
                    import yaml
                    cfg_path = Path(__file__).parent / "llm_config.yaml"
                    hf_id = "aliosama8399/football-analysisN"
                    if cfg_path.exists():
                        try:
                            with open(cfg_path) as f:
                                hf_id = (yaml.safe_load(f) or {}).get("providers", {}).get("huggingface", {}).get("model_id", hf_id)
                        except Exception:
                            pass
                    from transformers import AutoTokenizer, AutoModelForCausalLM
                    hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN", "").strip() or None
                    self._tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True, token=hf_token)
                    self._model = AutoModelForCausalLM.from_pretrained(hf_id, dtype=torch.float16 if torch.cuda.is_available() else torch.float32, device_map="auto" if torch.cuda.is_available() else None, trust_remote_code=True, token=hf_token)
                    if torch.cuda.is_available() and next(self._model.parameters()).device.type == "cpu":
                        self._model = self._model.to("cuda")
                    self._model.eval()
                    self._is_onnx = False
                    inputs = self._tokenizer(text, return_tensors="pt")
                    try:
                        device = next(self._model.parameters()).device
                        inputs = {k: v.to(device) for k, v in inputs.items()}
                    except Exception:
                        pass
                else:
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
        """KV-cache generation loop on GPU via onnxruntime-genai (~60-70 tok/s)."""
        og, np = self._og, self._np
        input_ids = self._gtok.encode(text)

        # JSON-mode prompts (live coach advisor) use GREEDY decoding: sampled
        # tokens occasionally trigger early-EOS mid-JSON, truncating the
        # structured response. Greedy reliably completes the template.
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
        # Context budgets: the raw KG team profiles are huge (~6KB/team) and the
        # full RAG prompt otherwise explodes the ONNX logits tensor
        # ([seq x 151936] per decode step -> multi-GB allocations -> OOM).
        import yaml

        kg_budget = int(os.getenv("FOOTBALL_ONNX_KG_CTX", 0)) or 2000
        vec_budget = int(os.getenv("FOOTBALL_ONNX_VEC_CTX", 0)) or 1200
        try:
            cfg_path = Path(__file__).parent / "llm_config.yaml"
            if cfg_path.exists():
                with open(cfg_path) as f:
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
        """For BaseLLMProvider-style callers (generate_explanation)."""
        return self.generate(prompt)

    def generate_explanation(self, match_context: dict, gnn_explanation: dict) -> str:
        """Public entry point matching BaseLLMProvider.generate_explanation."""
        # Reuse the shared prompt builder via BaseLLMProvider
        from models.llm_providers import BaseLLMProvider
        # Borrow _build_prompt without inheriting
        tmp = BaseLLMProvider._build_prompt
        # Create a minimal instance to build prompt
        prompt = tmp(self, match_context, gnn_explanation)
        return self.generate(prompt)

    def analyze_match(self, match_data: dict) -> str:
        prompt = (
            f"Analyze: {match_data.get('home_team', '?')} vs {match_data.get('away_team', '?')}. "
            f"Prediction: {match_data.get('prediction', '?')}."
        )
        return self.generate(prompt)
