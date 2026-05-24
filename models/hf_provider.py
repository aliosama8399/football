"""
HuggingFace Provider — Fine-tuned Football Analysis Model
===========================================================
Loads the full merged fine-tuned model from HuggingFace directly via
AutoTokenizer + AutoModelForCausalLM (no PEFT / LoRA adapter needed).

Model: aliosama8399/football-analysisN  (merged fine-tune of Qwen3-0.6B)

Registered in llm_providers.py so the RAG system can call
`get_llm_provider("huggingface")`.
"""

import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class HuggingFaceProvider:
    """
    Local inference using a fully merged HuggingFace causal-LM.
    Plugs into the existing LLM provider architecture.
    """

    def __init__(self, model_id: str = None):
        cfg = self._load_hf_cfg()

        self.model_id       = model_id or cfg.get("model_id", "aliosama8399/football-analysisN")
        self.max_new_tokens = cfg.get("max_new_tokens", 2048)
        self.temperature    = cfg.get("temperature", 0.7)

        # Resolve HF token: config file → env var → None (public repos only)
        self.hf_token = (
            cfg.get("hf_token", "").strip()
            or os.getenv("HUGGINGFACE_HUB_TOKEN", "").strip()
            or None
        )

        self._model     = None
        self._tokenizer = None
        self._loaded    = False

    @staticmethod
    def _load_hf_cfg() -> dict:
        import yaml
        cfg_path = Path(__file__).parent / "llm_config.yaml"
        if not cfg_path.exists():
            return {}
        with open(cfg_path) as f:
            return yaml.safe_load(f).get("providers", {}).get("huggingface", {})

    # ── Lazy loading ──────────────────────────────────────────────────────────
    def _load(self):
        if self._loaded:
            return

        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        logger.info("Loading model: %s", self.model_id)

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            token=self.hf_token,
        )

        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
            token=self.hf_token,
        )
        self._model.eval()

        self._loaded = True
        logger.info("HuggingFace provider ready. Device: %s",
                    next(self._model.parameters()).device)

    # ── Core generation ───────────────────────────────────────────────────────
    def generate(self, prompt: str) -> str:
        """
        Generate a response for a plain text prompt.
        Compatible with the existing BaseLLMProvider interface.
        """
        self._load()

        import torch

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
            text = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            text = prompt  # fallback for models without chat template

        inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def generate_with_context(self, prompt: str, kg_context: str = "", vector_context: str = "") -> str:
        """RAG-aware generation: inject retrieved context before the user prompt."""
        rag_prompt = ""
        if kg_context:
            rag_prompt += f"## Retrieved Knowledge Graph Context\n{kg_context}\n\n"
        if vector_context:
            rag_prompt += f"## Retrieved Historical Analyses\n{vector_context}\n\n"
        rag_prompt += f"## User Question\n{prompt}"
        return self.generate(rag_prompt)

    # ── Interface compatibility ───────────────────────────────────────────────
    def analyze_match(self, match_data: dict) -> str:
        """Compatibility wrapper for the existing explain_match.py pipeline."""
        prompt = (
            f"Analyze: {match_data.get('home_team', '?')} vs {match_data.get('away_team', '?')}. "
            f"Prediction: {match_data.get('prediction', '?')}."
        )
        return self.generate(prompt)
