"""Unit tests for MLOps release governance gates, manifest generation, and SHA-256 verification."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import models.upload_slm_hf as upload_mod


def _write_meta(meta_path: Path, payload: dict) -> None:
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(payload), encoding="utf-8")


def test_generate_model_card_contains_champion_metadata() -> None:
    """Verify generated Hugging Face model card embeds YAML frontmatter, run ID, and eval loss."""
    card = upload_mod.generate_model_card(
        repo_id="aliosama8399/football-analysis-PROD",
        meta={
            "run_id": "run_abc12345",
            "model_name": "unsloth/Qwen3-0.6B-unsloth-bnb-4bit",
            "metrics": {"eval_loss": 0.1627, "training_loss": 0.194},
            "promoted_at": "2026-11-14T12:00:00+00:00",
        },
    )
    assert card.startswith("---")
    assert "aliosama8399/football-analysis-PROD" in card
    assert "run_abc12345" in card
    assert "0.1627" in card


def test_upload_skips_when_already_uploaded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify duplicate uploads for an already-published champion run_id are skipped."""
    meta_path = tmp_path / "production_slm_meta.json"
    _write_meta(
        meta_path,
        {
            "production": True,
            "tag": "production",
            "run_id": "run_same_001",
            "uploaded_to_hf_run_id": "run_same_001",
            "newly_promoted": False,
            "metrics": {"eval_loss": 0.16},
        },
    )
    monkeypatch.setattr(upload_mod, "PRODUCTION_META_PATH", meta_path)

    res = upload_mod.upload_production_slm_to_hf(model_dir=tmp_path / "weights")
    assert res["status"] == "skipped"
    assert res["reason"] == "already_uploaded"


def test_upload_skips_when_not_newly_promoted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify upload skips cleanly when latest run did not beat the previous champion."""
    meta_path = tmp_path / "production_slm_meta.json"
    _write_meta(
        meta_path,
        {
            "production": True,
            "tag": "production",
            "run_id": "run_current_002",
            "uploaded_to_hf_run_id": "run_older_001",
            "newly_promoted": False,
            "metrics": {"eval_loss": 0.18},
        },
    )
    monkeypatch.setattr(upload_mod, "PRODUCTION_META_PATH", meta_path)

    res = upload_mod.upload_production_slm_to_hf(model_dir=tmp_path / "weights")
    assert res["status"] == "skipped"
    assert res["reason"] == "not_newly_promoted"


def test_upload_blocks_on_sla_quality_gate_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify SLA quality threshold blocks releases whose eval_loss exceeds ceiling."""
    meta_path = tmp_path / "production_slm_meta.json"
    _write_meta(
        meta_path,
        {
            "production": True,
            "tag": "production",
            "run_id": "run_poor_quality_003",
            "newly_promoted": True,
            "metrics": {"eval_loss": 5.0},
        },
    )
    monkeypatch.setattr(upload_mod, "PRODUCTION_META_PATH", meta_path)

    res = upload_mod.upload_production_slm_to_hf(model_dir=tmp_path / "weights")
    assert res["status"] == "skipped"
    assert res["reason"] == "sla_quality_gate_failed"


def test_upload_dry_run_creates_manifest_and_validates_sha256(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify dry-run validates artifacts, writes release_manifest.json, and embeds SHA-256 digest."""
    meta_path = tmp_path / "production_slm_meta.json"
    model_dir = tmp_path / "merged_16bit"
    model_dir.mkdir(parents=True)

    (model_dir / "model.safetensors").write_bytes(b"fake-safetensors-weights-binary-payload")
    (model_dir / "config.json").write_text('{"architectures": ["Qwen3ForCausalLM"]}', encoding="utf-8")

    _write_meta(
        meta_path,
        {
            "production": True,
            "tag": "production",
            "run_id": "run_champion_999",
            "previous_run_id": "run_champion_998",
            "newly_promoted": True,
            "metrics": {"eval_loss": 0.16, "training_loss": 0.19},
            "promoted_at": "2026-11-14T10:00:00+00:00",
        },
    )
    monkeypatch.setattr(upload_mod, "PRODUCTION_META_PATH", meta_path)
    monkeypatch.setenv("HF_TOKEN", "hf_test_token")

    res = upload_mod.upload_production_slm_to_hf(model_dir=model_dir, dry_run=True)
    assert res["status"] == "dry_run_success"

    manifest_file = model_dir / "release_manifest.json"
    assert manifest_file.exists()
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    assert manifest["run_id"] == "run_champion_999"
    assert manifest["previous_run_id"] == "run_champion_998"
    assert len(manifest["weights_sha256"]) == 64
    assert manifest["weights_size_bytes"] > 0

