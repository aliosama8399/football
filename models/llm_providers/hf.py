"""
HuggingFace Provider — Fine-tuned Football Analysis Model
===========================================================
Loads the full merged fine-tuned model from HuggingFace directly via
AutoTokenizer + AutoModelForCausalLM (no PEFT / LoRA adapter needed).

Model: aliosama8399/football-analysisN  (merged fine-tune of Qwen3-0.6B)
"""

from __future__ import annotations

import importlib.util
import logging
import os
import shutil
import threading
from pathlib import Path

from api.utils import clean_slm_output
from models.llm_providers.base import BaseLLMProvider

logger = logging.getLogger(__name__)


def disable_triton_if_no_compiler() -> None:
    """Disable PyTorch 2.13+ torch._native Triton overrides when no C compiler is present."""
    if shutil.which(os.getenv("CC", "gcc")) or shutil.which("cc"):
        return
    try:
        import torch._native.registry as _native_reg

        dsl_map = getattr(_native_reg, "_dsl_name_to_lib_graph", {})
        triton_dsls = [k for k in dsl_map if "triton" in str(k).lower()] or ["triton"]
        _native_reg.deregister_op_overrides(disable_dsl_names=triton_dsls)
        logger.info("Disabled PyTorch _native Triton overrides (%s) — no C compiler in runtime image.", triton_dsls)
    except Exception:
        pass

DEFAULT_HF_SYSTEM_PROMPT = (
    "You are an expert football tactical analyst. "
    "Analyze the match, predict the most likely outcome, and deliver "
    "a detailed tactical report covering both teams' strengths, "
    "weaknesses, and key strategic factors."
)


class HuggingFaceProvider(BaseLLMProvider):
    """
    Local inference using a fully merged HuggingFace causal-LM.
    Plugs into the existing LLM provider architecture.
    """

    provider_name = "huggingface"

    def __init__(self, model_id: str | None = None, **kwargs):
        super().__init__()
        cfg = self._load_hf_cfg()

        local_merged = Path(__file__).resolve().parent.parent / "finetuned_qwen_unsloth" / "merged_16bit"
        if not model_id and local_merged.exists():
            self.model_id = str(local_merged)
            logger.info("Using local fine-tuned merged model: %s", self.model_id)
        else:
            self.model_id = model_id or cfg.get("model_id", "aliosama8399/football-analysisN")

        self.max_new_tokens = cfg.get("max_new_tokens", 1024)
        self.temperature    = cfg.get("temperature", 0.2)

        # Resolve HF token: config file → env var → None (public repos only)
        self.hf_token = (
            cfg.get("hf_token", "").strip()
            or os.getenv("HUGGINGFACE_HUB_TOKEN", "").strip()
            or None
        )

        self._model     = None
        self._tokenizer = None
        self._loaded    = False
        self._gen_lock  = threading.Lock()

    @staticmethod
    def _load_hf_cfg() -> dict:
        import yaml
        cfg_path = Path(__file__).resolve().parent.parent / "llm_config.yaml"
        if not cfg_path.exists():
            return {}
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f).get("providers", {}).get("huggingface", {})
        except Exception:
            return {}

    # ── Lazy loading ──────────────────────────────────────────────────────────
    def _load(self):
        if self._loaded:
            return

        import enum
        import torch
        import torch.utils._pytree as _pytree

        disable_triton_if_no_compiler()

        # PyTorch 2.7+ natively supports Enum subclasses in torch.compile; skip redundant registration
        _orig_reg = getattr(_pytree, "register_constant", None)
        if _orig_reg is not None and not getattr(_orig_reg, "_enum_guarded", False):
            def _safe_register_constant(cls):
                if isinstance(cls, type) and issubclass(cls, enum.Enum):
                    return cls
                return _orig_reg(cls)
            _safe_register_constant._enum_guarded = True  # type: ignore[attr-defined]
            _pytree.register_constant = _safe_register_constant

        from transformers import AutoTokenizer, AutoModelForCausalLM

        logger.info("Loading model: %s", self.model_id)

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                trust_remote_code=True,
                token=self.hf_token,
                fix_mistral_regex=True,
            )
        except TypeError:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                trust_remote_code=True,
                token=self.hf_token,
            )

        has_accelerate = importlib.util.find_spec("accelerate") is not None
        load_kwargs = {
            "dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
            "trust_remote_code": True,
            "token": self.hf_token,
        }
        if torch.cuda.is_available() and has_accelerate:
            load_kwargs["device_map"] = "auto"

        self._model = AutoModelForCausalLM.from_pretrained(self.model_id, **load_kwargs)

        # Force move to CUDA if loaded on CPU (or when accelerate is absent)
        if torch.cuda.is_available() and next(self._model.parameters()).device.type == "cpu":
            logger.info("Moving model to CUDA device...")
            self._model = self._model.to("cuda")

        self._model.eval()
        self._backend = "hf"

        self._loaded = True
        logger.info("HuggingFace provider ready. Device: %s",
                    next(self._model.parameters()).device)

    # ── Core generation ───────────────────────────────────────────────────────
    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        """
        Generate a response for a plain text prompt.
        Compatible with the existing BaseLLMProvider interface.
        """
        self._load()

        import torch

        messages = [
            {
                "role": "system",
                "content": system_prompt or DEFAULT_HF_SYSTEM_PROMPT,
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
                text = prompt  # fallback for models without chat template

        inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)
        do_sample = self.temperature > 0
        gen_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": do_sample,
            "repetition_penalty": 1.15,
            "no_repeat_ngram_size": 4,
            "pad_token_id": self._tokenizer.eos_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = self.temperature
            gen_kwargs["top_p"] = 0.85

        with self._gen_lock, torch.no_grad():
            outputs = self._model.generate(**inputs, **gen_kwargs)

        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        raw = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
        return clean_slm_output(raw)

    def generate_with_context(
        self,
        prompt: str,
        kg_context: str = "",
        vector_context: str = "",
        system_prompt: str | None = None,
    ) -> str:
        """RAG-aware generation: inject retrieved context before the user prompt."""
        rag_prompt = ""
        if kg_context:
            rag_prompt += f"## Retrieved Knowledge Graph Context\n{kg_context}\n\n"
        if vector_context:
            rag_prompt += f"## Retrieved Historical Analyses\n{vector_context}\n\n"
        rag_prompt += f"## User Question\n{prompt}"
        return self.generate(rag_prompt, system_prompt=system_prompt)

    # ── Interface compatibility ───────────────────────────────────────────────
    def _call_api(self, prompt: str) -> str:
        """Forward BaseLLMProvider prompt to local generate method."""
        return self.generate(prompt)

    def analyze_match(self, match_data: dict) -> str:
        """Compatibility wrapper for the existing explain_match.py pipeline."""
        prompt = (
            f"Analyze: {match_data.get('home_team', '?')} vs {match_data.get('away_team', '?')}. "
            f"Prediction: {match_data.get('prediction', '?')}."
        )
        return self.generate(prompt)
