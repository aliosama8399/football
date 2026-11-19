"""Unit tests for Airflow DAG gates, collector testability, and inference prompt loading."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pandas as pd
import pytest

from data.collectors.test_fbref import scrape_cl_sample
from data.dags.football_gnn_mlflow_dag import (
    is_season_finished_for_slm,
    upload_production_slm_to_hf_task,
)
from models.test_inference import load_sample_prompt


def _ensure_airflow_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide lightweight Airflow stubs in sys.modules when apache-airflow is not installed."""
    if "airflow" in sys.modules:
        return

    class _DummyDAG:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    class _DummyOp:
        def __init__(self, *args, **kwargs):
            pass

        def __rshift__(self, other):
            return other

    airflow_mod = ModuleType("airflow")
    airflow_mod.DAG = _DummyDAG  # type: ignore[attr-defined]
    ops_mod = ModuleType("airflow.operators")
    bash_mod = ModuleType("airflow.operators.bash")
    bash_mod.BashOperator = _DummyOp  # type: ignore[attr-defined]
    py_mod = ModuleType("airflow.operators.python")
    py_mod.PythonOperator = _DummyOp  # type: ignore[attr-defined]
    py_mod.ShortCircuitOperator = _DummyOp  # type: ignore[attr-defined]
    trig_mod = ModuleType("airflow.operators.trigger_dagrun")
    trig_mod.TriggerDagRunOperator = _DummyOp  # type: ignore[attr-defined]

    for name, mod in [
        ("airflow", airflow_mod),
        ("airflow.operators", ops_mod),
        ("airflow.operators.bash", bash_mod),
        ("airflow.operators.python", py_mod),
        ("airflow.operators.trigger_dagrun", trig_mod),
    ]:
        monkeypatch.setitem(sys.modules, name, mod)


def test_all_airflow_dags_import_cleanly(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify all three Airflow DAG modules import cleanly and expose core callables."""
    _ensure_airflow_stub(monkeypatch)

    master_dag = importlib.import_module("data.dags.football_master_pipeline_dag")
    gnn_dag = importlib.import_module("data.dags.football_gnn_mlflow_dag")
    rag_eval_dag = importlib.import_module("data.dags.football_rag_eval_dag")

    assert callable(master_dag.run_teacher_in_gpu_worker)
    assert callable(gnn_dag.run_slm_training_on_mlflow)
    assert callable(gnn_dag.upload_production_slm_to_hf_task)
    assert callable(rag_eval_dag.reload_api_singletons)
    assert callable(rag_eval_dag.run_rag_eval_on_worker)


def test_season_end_gatekeeper_logic() -> None:
    """Verify DAG Step 4 ShortCircuit gate respects force_slm_train and skip_slm flags."""
    assert is_season_finished_for_slm(dag_run=SimpleNamespace(conf={"force_slm_train": True})) is True
    assert is_season_finished_for_slm(dag_run=SimpleNamespace(conf={"skip_slm": True})) is False


def test_dag_step8_skips_non_champion() -> None:
    """Verify DAG Step 8 skips Hugging Face upload cleanly when skip_hf_upload is configured."""
    res = upload_production_slm_to_hf_task(dag_run=SimpleNamespace(conf={"skip_hf_upload": True}))
    assert res["status"] == "skipped"
    assert res["reason"] == "skip_hf_upload_conf"


def test_inference_load_sample_prompt() -> None:
    """Verify load_sample_prompt returns a populated tactical prompt string and matchup label."""
    prompt, desc = load_sample_prompt(0)
    assert isinstance(prompt, str) and len(prompt) > 20
    assert isinstance(desc, str) and len(desc) > 0


def test_fbref_collector_with_mock_client(tmp_path: Path) -> None:
    """Verify scrape_cl_sample queries injected client and persists CSV output without network calls."""

    class _MockFBref:
        def scrape_matches(self, year: int, league: str) -> pd.DataFrame:
            return pd.DataFrame(
                [
                    {"Date": "2023-09-19", "Home": "Arsenal", "Away": "PSV", "Year": year, "League": league},
                    {"Date": "2023-09-20", "Home": "Real Madrid", "Away": "Union Berlin", "Year": year, "League": league},
                ]
            )

    out_csv = tmp_path / "cl_sample.csv"
    df = scrape_cl_sample(output_csv=out_csv, fbref_client=_MockFBref())

    assert len(df) == 2
    assert out_csv.exists()
    written = pd.read_csv(out_csv)
    assert len(written) == 2
    assert set(written["Home"]) == {"Arsenal", "Real Madrid"}
