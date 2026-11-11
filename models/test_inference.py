"""
Fast Inference & Tactical Report Generator for Fine-Tuned Football SLM
=====================================================================
Tests and generates tactical reports using fine-tuned Unsloth LoRA adapters
or merged 16-bit models on NVIDIA GTX 1660 Ti (6 GB VRAM) with fast 4-bit inference.

Usage:
  # 1. Run default sample tactical match analysis (loads real validation match):
  python models/test_inference.py

  # 2. Test another validation sample by index:
  python models/test_inference.py --val-index 1

  # 3. Run with custom prompt:
  python models/test_inference.py --prompt "Analyze: Real Madrid (Home) vs Barcelona (Away)..."

  # 4. Interactive chat mode:
  python models/test_inference.py --interactive
"""

from __future__ import annotations

import argparse
import enum
import json
import sys
from pathlib import Path

__test__ = False

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

try:
    import torch.utils._pytree as _pytree

    # PyTorch 2.7+ natively supports Enum subclasses in torch.compile; skip redundant registration
    _orig_reg = getattr(_pytree, "register_constant", None)
    if _orig_reg is not None and not getattr(_orig_reg, "_enum_guarded", False):
        def _safe_register_constant(cls):
            if isinstance(cls, type) and issubclass(cls, enum.Enum):
                return cls
            return _orig_reg(cls)
        _safe_register_constant._enum_guarded = True  # type: ignore[attr-defined]
        _pytree.register_constant = _safe_register_constant
except ImportError:
    _pytree = None  # type: ignore[assignment]

try:
    # CRITICAL: import unsloth first to suppress warnings and enable fast Triton kernels
    import unsloth  # noqa: F401, E402
    from unsloth import FastLanguageModel  # noqa: E402
except ImportError:
    FastLanguageModel = None  # type: ignore[assignment]

try:
    import transformers  # noqa: E402

    _orig_tok_from_pretrained = transformers.AutoTokenizer.from_pretrained
    if not getattr(_orig_tok_from_pretrained, "_mistral_regex_guarded", False):
        def _safe_tok_from_pretrained(*args, **kwargs):
            if "fix_mistral_regex" not in kwargs:
                kwargs["fix_mistral_regex"] = True
            return _orig_tok_from_pretrained(*args, **kwargs)
        _safe_tok_from_pretrained._mistral_regex_guarded = True
        transformers.AutoTokenizer.from_pretrained = _safe_tok_from_pretrained
except ImportError:
    transformers = None  # type: ignore[assignment]

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


def load_sample_prompt(val_index: int = 0) -> tuple[str, str]:
    """Load a real prompt and metadata from football_val.json."""
    val_file = BASE_DIR / "data" / "finetune" / "football_val.json"
    if val_file.exists():
        try:
            with open(val_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data and 0 <= val_index < len(data):
                record = data[val_index]
                user_msg = next((m["content"] for m in record["messages"] if m["role"] == "user"), "")
                try:
                    payload = json.loads(user_msg)
                    match_info = payload.get("match", {})
                    h = match_info.get("home_team", "Home")
                    a = match_info.get("away_team", "Away")
                    desc = f"{h} vs {a}"
                except Exception:
                    desc = f"Validation sample #{val_index}"
                return user_msg, desc
        except Exception as e:
            print(f"[WARN] Could not load validation record: {e}")

    # Fallback prompt if validation file cannot be read
    fallback = (
        '{"match": {"home_team": "Liverpool", "away_team": "Real Madrid", "league": "Champions_League"}, '
        '"statistics": {"home_form": {"Points per match (last 5)": 2.4, "Goals scored per match (last 5)": 2.8}}, '
        '"gnn_prediction": "Home Win", "gnn_probabilities": {"H": 0.52, "D": 0.26, "A": 0.22}}'
    )
    return fallback, "Liverpool vs Real Madrid (Fallback)"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for testing generation."""
    parser = argparse.ArgumentParser(description="Test fine-tuned football SLM generation.")
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/finetuned_qwen_unsloth/lora_adapter",
        help="Path to fine-tuned LoRA adapter or merged model directory.",
    )
    parser.add_argument(
        "--val-index",
        type=int,
        default=0,
        help="Index of validation sample to test from football_val.json (default: 0).",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="",
        help="Custom tactical prompt to analyze (overrides validation sample).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=384,
        help="Maximum new tokens to generate (default: 384).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Sampling temperature (default: 0.6).",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Nucleus sampling top-p (default: 0.9).",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Start an interactive session to query multiple matches in real time.",
    )
    return parser.parse_args()


def load_model_and_tokenizer(model_path: str):
    """Load the fine-tuned model and tokenizer with fast inference optimizations."""
    import torch

    path_obj = Path(model_path)
    if not path_obj.exists():
        fallback = path_obj.parent / "merged_16bit"
        if fallback.exists():
            print(f"[INFO] '{path_obj}' not found; falling back to merged model '{fallback}'")
            model_path = str(fallback)
        else:
            raise FileNotFoundError(
                f"Model directory '{model_path}' does not exist. "
                "Ensure training has finished and saved the model."
            )

    print(f"[INFO] Loading fine-tuned model from: {model_path} (4-bit inference, max_seq_len=2048)...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=2048,
        dtype=torch.float16,
        load_in_4bit=True,
    )
    if hasattr(tokenizer, "init_kwargs") and isinstance(tokenizer.init_kwargs, dict):
        tokenizer.init_kwargs["fix_mistral_regex"] = True
    setattr(tokenizer, "fix_mistral_regex", True)
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def generate_report(
    model,
    tokenizer,
    user_prompt: str,
    max_new_tokens: int = 384,
    temperature: float = 0.6,
    top_p: float = 0.9,
) -> str:
    """Format prompt with chat template, generate tokens, and print output."""
    import torch

    messages = [{"role": "user", "content": user_prompt}]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    attention_mask = torch.ones_like(inputs)

    print("\n" + "=" * 70)
    print("  TACTICAL ANALYSIS GENERATION")
    print("=" * 70, flush=True)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=temperature > 0.0,
            use_cache=True,
        )

    # Decode only the generated response tokens
    gen_tokens = outputs[0][inputs.shape[1] :]
    decoded = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

    print(decoded, flush=True)
    print("=" * 70 + "\n", flush=True)
    return decoded


def main() -> None:
    """Main entrypoint for model generation testing."""
    args = parse_args()
    model, tokenizer = load_model_and_tokenizer(args.model_path)

    if args.interactive:
        print("\n[INFO] Entered Interactive Match Analysis Mode (Ctrl+C or 'exit' to quit).\n")
        while True:
            try:
                user_input = input("\nEnter match prompt > ").strip()
                if not user_input or user_input.lower() in ("exit", "quit"):
                    break
                generate_report(
                    model=model,
                    tokenizer=tokenizer,
                    user_prompt=user_input,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )
            except KeyboardInterrupt:
                print("\nExiting interactive mode.")
                break
    else:
        if args.prompt.strip():
            prompt_text = args.prompt.strip()
            print("[INFO] Using custom prompt provided via CLI")
        else:
            prompt_text, match_desc = load_sample_prompt(args.val_index)
            print(f"[INFO] Using validation sample #{args.val_index} ({match_desc})")

        generate_report(
            model=model,
            tokenizer=tokenizer,
            user_prompt=prompt_text,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )


if __name__ == "__main__":
    main()
