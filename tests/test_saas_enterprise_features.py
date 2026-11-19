"""Unit tests for Enterprise SaaS upgrades 1 to 5."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from api.schemas import PreMatchDossierRequest, TacticalSimulationRequest
from api.services.chat_service import ChatService
from api.services.job_runner import JobRunRecord, PipelineJobManager
from api.services.saas_tenant_service import SaaSTenantManager
from api.services.shadow_model_service import ShadowModelService
from api.services.tactical_simulator_service import TacticalSimulatorService


def test_saas_multi_tenancy_api_keys_and_quotas(tmp_path: Path) -> None:
    """Verify B2B API keys, subscription tiers, daily quota enforcement, and key revocation."""
    mgr = SaaSTenantManager(storage_path=tmp_path / "tenants.json")

    key_data = mgr.create_api_key("arsenal_club", label="scouting-sync")
    raw_key = key_data["api_key"]
    assert raw_key.startswith("fk_live_")

    verified = mgr.verify_api_key(raw_key)
    assert verified is not None
    assert verified["username"] == "arsenal_club"
    assert mgr.verify_api_key("fk_live_invalid_secret") is None

    downgraded = mgr.update_tenant_tier("arsenal_club", tier="free", organization="Arsenal Analytics")
    assert downgraded["tier"] == "free"
    assert downgraded["daily_limit"] == 50
    assert downgraded["organization"] == "Arsenal Analytics"

    for _ in range(15):
        res = mgr.check_and_consume_quota("arsenal_club")
        assert res["allowed"] is True

    rpm_blocked = mgr.check_and_consume_quota("arsenal_club")
    assert rpm_blocked["allowed"] is False
    assert rpm_blocked["reason"] == "rpm_exceeded"

    mgr._rpm_windows["arsenal_club"].clear()  # pylint: disable=protected-access
    blocked = mgr.check_and_consume_quota("arsenal_club", cost=36)
    assert blocked["allowed"] is False
    assert blocked["reason"] == "daily_quota_exceeded"

    upgraded = mgr.update_tenant_tier("arsenal_club", tier="club_enterprise")
    assert upgraded["daily_limit"] == 50000
    assert "b2b_api" in upgraded["features"]

    assert len(mgr.list_api_keys("arsenal_club")) == 1
    assert mgr.revoke_api_key("arsenal_club", key_data["key_id"]) is True
    assert mgr.verify_api_key(raw_key) is None


def test_sse_streaming_chat_service() -> None:
    """Verify SSE tactical chat streams stage, token, citations, and completion events."""
    saved_records = []

    class _MockChatRepo:
        async def get_conversation(self, conversation_id: int, user_id: int):
            if conversation_id == 1 and user_id == 7:
                return SimpleNamespace(id=1, user_id=7, mode="prediction")
            return None

        async def get_recent_messages(self, conversation_id: int, limit: int = 6):
            return [SimpleNamespace(sender="user", content="Prior context turn")]

        async def save_message(self, conversation_id: int, sender: str, content: str, sources=None):
            rec = SimpleNamespace(
                id=len(saved_records) + 10,
                conversation_id=conversation_id,
                sender=sender,
                content=content,
                sources=sources,
            )
            saved_records.append(rec)
            return rec

    class _MockKBRepo:
        def ask(self, question: str, llm_name=None, prefer_prediction=False, memory=None):
            source_obj = SimpleNamespace(to_json=lambda: {"type": "kg", "team": "Arsenal"})
            return SimpleNamespace(
                content="Arsenal high press forces midfield turnovers and increases home win probability.",
                provider="onnx",
                bundle=SimpleNamespace(sources=[source_obj]),
            )

    service = ChatService(_MockChatRepo(), _MockKBRepo())  # type: ignore[arg-type]

    async def _collect_stream():
        frames = []
        async for chunk in service.stream_post_message(1, 7, "Analyze Arsenal vs Chelsea"):
            frames.append(chunk)
        return frames

    chunks = asyncio.run(_collect_stream())
    joined = "".join(chunks)

    assert "event: stage" in joined
    assert "context_loaded" in joined
    assert "hybrid_rag_and_gnn" in joined
    assert "event: token" in joined
    assert "event: citations" in joined
    assert "event: done" in joined
    assert len(saved_records) == 2


def test_what_if_tactical_counterfactual_simulator() -> None:
    """Verify counterfactual simulator computes probabilities/xPts and requires live SLM output (zero static text)."""
    sim = TacticalSimulatorService()
    req = TacticalSimulationRequest(
        home_team="Arsenal",
        away_team="Real Madrid",
        home_formation="4-2-3-1",
        away_formation="4-4-2",
        tactical_preset="high_press",
        key_player_impact=0.10,
    )

    with pytest.raises(RuntimeError):
        sim.simulate(req, baseline_probs={"H": 0.45, "D": 0.25, "A": 0.30})

    captured = {}

    def _mock_slm(prompt: str, system_prompt: str | None = None) -> str:
        captured["prompt"] = prompt
        captured["system_prompt"] = system_prompt
        return json.dumps({
            "recommendation": (
                "Deploy Arsenal in 4-2-3-1 high_press to trap Real Madrid's double pivot and capitalize on the +xPts shift."
            ),
            "strengths_gained": [
                "Numerical superiority between Real Madrid's midfield and defensive lines",
                "High pressing triggers force hurried clearances",
            ],
            "vulnerabilities_exposed": [
                "Space behind full-backs during vertical transitions",
            ],
        })

    res = sim.simulate(
        req,
        baseline_probs={"H": 0.45, "D": 0.25, "A": 0.30},
        slm_generator=_mock_slm,
        llm_provider="huggingface",
    )

    total_sim_prob = sum(res.simulated_probabilities.values())
    assert abs(total_sim_prob - 1.0) < 1e-3
    assert res.simulated_probabilities["H"] > res.baseline_probabilities["H"]
    assert res.expected_points_delta > 0.0
    assert res.slm_generated is True
    assert res.llm_provider == "huggingface"
    assert "Real Madrid's double pivot" in res.recommendation
    assert "4-2-3-1" in captured["prompt"]
    assert len(res.tactical_tradeoffs["strengths_gained"]) == 2
    assert len(res.tactical_tradeoffs["vulnerabilities_exposed"]) == 1


def test_one_click_pre_match_scouting_dossier() -> None:
    """Verify dossier composer selects optimal counterfactual math and generates dynamic SLM briefing."""
    sim = TacticalSimulatorService()
    req = PreMatchDossierRequest(home_team="Liverpool", away_team="Chelsea", league="Premier_League")

    with pytest.raises(RuntimeError):
        sim.generate_dossier(req, baseline_probs={"H": 0.50, "D": 0.25, "A": 0.25})

    def _mock_dossier_slm(prompt: str, system_prompt: str | None = None) -> str:
        assert "Liverpool" in prompt and "Chelsea" in prompt
        return json.dumps({
            "optimal_recommendation": "Liverpool should press aggressively in a 4-3-3 to pin Chelsea deep.",
            "strengths_gained": ["Dominant central overload via Liverpool's midfield trio"],
            "vulnerabilities_exposed": ["Counter-attacking pace behind the right flank"],
            "scouting_priorities": [
                {"position": "MF", "rationale": "Ball-retaining holding midfielder to suppress Chelsea transitions"},
                {"position": "DF", "rationale": "Recovery center-back for high defensive line cover"},
            ],
            "executive_markdown": (
                "# Pre-Match Scouting Dossier: Liverpool vs Chelsea\n\n"
                "## Tactical Outlook\nLiverpool hold the home TEA-GNN advantage and gain maximum xPts via 4-3-3 high press.\n\n"
                "## Recruitment Focus\nTarget a press-resistant MF and high-pace DF cover."
            ),
        })

    res = sim.generate_dossier(
        req,
        baseline_probs={"H": 0.50, "D": 0.25, "A": 0.25},
        home_profile={"strengths": ["High counter-press"], "weaknesses": ["Midfield turnover transitions"]},
        away_profile={"strengths": ["Fast wide wingers"], "weaknesses": ["Set-piece aerial marking"]},
        slm_generator=_mock_dossier_slm,
        llm_provider="huggingface",
    )

    assert res.match_forecast["verdict"] == "H"
    assert res.optimal_formation_simulation["expected_points_simulated"] >= 1.75
    assert res.slm_generated is True
    assert any(item["position"] == "MF" for item in res.scouting_priorities)
    assert "Pre-Match Scouting Dossier: Liverpool vs Chelsea" in res.executive_markdown
    assert "Liverpool hold the home TEA-GNN advantage" in res.executive_markdown


def test_champion_challenger_shadow_and_job_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify shadow deployment metrics and persistent disk-backed pipeline job ledger."""
    shadow_path = tmp_path / "shadow.jsonl"
    shadow_svc = ShadowModelService(ledger_path=shadow_path)

    for _ in range(5):
        entry = shadow_svc.evaluate_shadow(
            home_team="Barcelona",
            away_team="Atletico Madrid",
            champion_probs={"H": 0.62, "D": 0.22, "A": 0.16},
            champion_result="H",
        )
        assert entry["verdict_agreement"] is True
        assert entry["js_divergence"] >= 0.0

    summary = shadow_svc.get_shadow_summary()
    assert summary["total_evaluated"] == 5
    assert summary["agreement_rate"] == 1.0
    assert summary["promotion_ready"] is True
    assert shadow_path.exists()

    mgr = PipelineJobManager()
    ledger_path = tmp_path / "jobs_ledger.jsonl"
    monkeypatch.setattr(mgr, "ledger_path", ledger_path)

    record = JobRunRecord(
        job_id="job_20261118_120000",
        status="SUCCESS",
        step=8,
        step_name="Completed Successfully",
        progress_pct=100,
        started_at="2026-11-18T12:00:00",
        ended_at="2026-11-18T12:02:10",
        duration_s=130.0,
        parameters={"gnn_model": "TEA-GNN"},
    )
    mgr._persist_job_to_ledger(record)  # pylint: disable=protected-access
    entries = mgr.get_persistent_ledger(limit=10)
    assert len(entries) == 1
    assert entries[0]["job_id"] == "job_20261118_120000"
    assert json.loads(ledger_path.read_text(encoding="utf-8").strip())["status"] == "SUCCESS"


def test_fod_provider_uses_dynamic_season_and_factory_avoids_concatenation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify FOD passes dynamic season year and factory does not append unplayed FOD-only players."""
    from data.player_providers.factory import _AllProvidersProxy
    from data.player_providers.fod import FodPlayerProvider
    from data.player_providers.schema import PlayerRecord

    captured_params = {}
    fod = FodPlayerProvider(api_key="test_fod_key")

    def _mock_get(path: str, params=None):
        captured_params["path"] = path
        captured_params["params"] = params
        return {
            "teams": [
                {
                    "name": "Liverpool FC",
                    "shortName": "Liverpool",
                    "squad": [
                        {"name": "Mohamed Salah", "position": "Offence", "dateOfBirth": "1992-06-15", "nationality": "Egypt", "shirtNumber": 11},
                    ],
                }
            ]
        }

    monkeypatch.setattr(fod, "_get", _mock_get)
    fod.fetch_team_squad("Liverpool", "E0", "2526")
    assert captured_params["params"] == {"season": "2025"}

    class _MockStatsProv:
        provider_name = "understat"

        def fetch_team_squad(self, team, league_code, season):
            return [
                PlayerRecord(
                    name="Mohamed Salah",
                    team=team,
                    league=league_code,
                    season=season,
                    source="understat",
                    position="FW",
                    minutes=2900.0,
                    goals=22.0,
                )
            ]

    class _MockFodProv:
        provider_name = "fod"

        def fetch_team_squad(self, team, league_code, season):
            return [
                PlayerRecord(
                    name="Mohamed Salah",
                    team=team,
                    league=league_code,
                    season=season,
                    source="fod",
                    age=33.0,
                    shirt_number=11,
                ),
                PlayerRecord(
                    name="Old Or Offseason Player",
                    team=team,
                    league=league_code,
                    season=season,
                    source="fod",
                    age=29.0,
                ),
            ]

    proxy = _AllProvidersProxy.__new__(_AllProvidersProxy)
    proxy._providers = [_MockStatsProv(), _MockFodProv()]  # pylint: disable=protected-access
    squad = proxy.fetch_team_squad("Liverpool", "E0", "2425")
    names = [p.name for p in squad]
    assert names == ["Mohamed Salah"]
    assert squad[0].shirt_number == 11


def test_api_football_historical_cache_and_fast_429_bailout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify historical caches never expire and HTTP 429 bails out immediately without sleeping."""
    import os
    import time
    from data.player_providers.api_football import ApiFootballPlayerProvider

    cache_file = tmp_path / "teams_E0_2024.json"
    cache_file.write_text(json.dumps([{"team": {"id": 40, "name": "Liverpool"}}]), encoding="utf-8")
    old_mtime = time.time() - (30 * 24 * 3600)
    os.utime(cache_file, (old_mtime, old_mtime))

    prov = ApiFootballPlayerProvider(api_key="dummy_key", cache_dir=tmp_path)
    prov._min_spacing_seconds = 0.0  # pylint: disable=protected-access
    teams = prov.fetch_league_teams("E0", "2024")
    assert teams == [{"id": 40, "name": "Liverpool"}]

    def _mock_429(*args, **kwargs):
        return SimpleNamespace(status_code=429, text="rate limit", json=lambda: {})

    monkeypatch.setattr("requests.get", _mock_429)
    t0 = time.monotonic()
    res = prov._request("/players", {"team": 40, "season": 2026})  # pylint: disable=protected-access
    elapsed = time.monotonic() - t0
    assert res is None
    assert elapsed < 1.0
    assert prov._rate_limited_until > time.time()  # pylint: disable=protected-access


def test_ui_contains_all_enterprise_saas_controls() -> None:
    """Verify Tactical Cockpit HTML exposes all 5 SaaS upgrades and Best-XI/Scout fixes."""
    html = (Path(__file__).resolve().parent.parent / "api" / "static" / "index.html").read_text(encoding="utf-8")
    for elem_id in (
        "simulator-tab",
        "saas-tab",
        "best11-season-select",
        "best11-formation-select",
        "scout-refresh",
        "shadow-metrics-box",
        "job-ledger-box",
        "hud-tier-badge",
    ):
        assert elem_id in html, f"Missing UI element ID: {elem_id}"
    assert 'value="2627"' in html


def test_scout_blocks_one_minute_cameo_from_ranking_first() -> None:
    """Verify Empirical Bayes shrinkage + minutes gate prevent 1-match cameo players from ranking #1."""
    from api.services.scout.service import ScoutService
    from api.services.scout.strategies.scoring import ForwardScoring
    from data.player_providers.schema import PlayerRecord

    cameo = PlayerRecord(
        name="Shumaira Mheuka",
        team="Chelsea",
        league="Premier_League",
        season="2425",
        source="fbref",
        position="FW",
        age=16.0,
        appearances=1,
        minutes=1.0,
        goals=0.0,
        assists=0.0,
        xg=0.025,
        xa=0.0,
        shots=1.0,
        key_passes=0.0,
    )
    regulars = [
        PlayerRecord(
            name=f"Regular Striker {idx}",
            team=f"Club {idx}",
            league="Premier_League",
            season="2425",
            source="fbref",
            position="FW",
            age=19.0,
            appearances=30,
            minutes=2200.0 + idx * 50.0,
            goals=12.0 + idx,
            assists=5.0,
            xg=11.0 + idx,
            xa=4.0,
            shots=68.0,
            key_passes=30.0,
        )
        for idx in range(12)
    ]

    # Direct scoring check: shrinkage + reliability keeps cameo strictly below regulars
    direct_scores = ForwardScoring().score([cameo, *regulars])
    assert direct_scores[0].record_score < direct_scores[1].record_score
    assert direct_scores[0].breakdown["minutes_reliability"] < 0.1

    class _MockStatsProv:
        api_key = ""

    class _MockRepo:
        stats_provider = _MockStatsProv()

        @staticmethod
        def identity_pool(league_codes, season, position, youth, exclude_team=None):
            return [cameo, *regulars]

        @staticmethod
        def enrich_with_stats(recs, season, refresh=False):
            return recs

        @staticmethod
        def attach_transfers(recs, refresh=False):
            return {}

    report = ScoutService(_MockRepo()).scout(("E0",), "2425", "FW", top=5, youth=True)
    ranked_names = [c["name"] for c in report["candidates"]]
    assert "Shumaira Mheuka" not in ranked_names
    assert all((c["stats"]["minutes"] or 0) >= 270 for c in report["candidates"])


def test_clean_slm_output_strips_think_and_echoes_without_truncating() -> None:
    """Verify clean_slm_output removes noise wrappers while preserving long tactical prose."""
    from api.utils import clean_slm_output, is_rag_fallback_payload

    long_analysis = " ".join(f"Tactical factor {i} boosts possession control." for i in range(60))
    noisy = (
        "<think>internal chain of thought should be removed</think>\n\n"
        "=== Expert 1 (Graph Neural Network) Prediction ===\n"
        "Predicted Outcome: Draw\n"
        "Probabilities: Home Win=31.9% | Draw=55.4% | Away Win=12.7%\n"
        "==================================================\n\n"
        "=== Reasoning ===\n"
        "You are Expert 2 (Fine-tuned LLM). Analyze the match.\n"
        + long_analysis
        + "\n<|im_end|>"
    )
    cleaned = clean_slm_output(noisy)
    assert "<think>" not in cleaned
    assert "<|im_end|>" not in cleaned
    assert "Expert 1 (Graph Neural Network)" not in cleaned
    assert "=== Reasoning ===" not in cleaned
    assert len(cleaned) > 1500
    assert "Tactical factor 59 boosts possession control." in cleaned
    assert is_rag_fallback_payload({"question": "x", "kg_context": "y", "note": "z"}) is True


def test_espn_provider_and_best11_multiseason_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify EspnPlayerProvider maps rosters cleanly and Best11Service resolves early-season squads."""
    from api.services.best11.service import Best11Service
    from data.player_providers.espn import EspnPlayerProvider
    from data.team_totals import TeamTotals

    espn = EspnPlayerProvider(rate_limit_sec=0.0)

    def _mock_get_json(url: str, params=None):
        if url.endswith("/teams"):
            if "eng.1" in url:
                return {"sports": [{"leagues": [{"teams": [{"team": {"id": "388", "displayName": "Coventry City"}}]}]}]}
            return {"sports": [{"leagues": [{"teams": [{"team": {"id": "369", "displayName": "Middlesbrough"}}]}]}]}
        roster_pos = ["G", "D", "D", "D", "D", "M", "M", "M", "F", "F", "F", "M", "D"]
        athletes = [
            {
                "id": str(100 + idx),
                "displayName": f"Player {idx}",
                "age": 24,
                "citizenship": "England",
                "jersey": str(idx + 1),
                "position": {"abbreviation": pos},
                "statistics": {
                    "splits": {
                        "categories": [
                            {
                                "name": "general",
                                "stats": [
                                    {"name": "appearances", "value": 8.0},
                                    {"name": "subIns", "value": 1.0},
                                    {"name": "totalGoals", "value": float(idx % 3)},
                                    {"name": "goalAssists", "value": 1.0},
                                    {"name": "totalShots", "value": 6.0},
                                    {"name": "shotsOnTarget", "value": 3.0},
                                    {"name": "saves", "value": 18.0 if pos == "G" else 0.0},
                                    {"name": "goalsConceded", "value": 9.0 if pos == "G" else 0.0},
                                    {"name": "yellowCards", "value": 1.0},
                                    {"name": "redCards", "value": 0.0},
                                ],
                            }
                        ]
                    }
                },
            }
            for idx, pos in enumerate(roster_pos)
        ]
        return {"athletes": athletes}

    monkeypatch.setattr(espn, "_get_json", _mock_get_json)
    cov_squad = espn.fetch_team_squad("Coventry", "E0", "2627")
    assert len(cov_squad) == 13
    assert cov_squad[0].position == "GK"
    assert cov_squad[0].extra["pos_list"] == ["GK"]
    assert cov_squad[0].minutes > 0

    boro_squad = espn.fetch_team_squad("Middlesbrough", "E0", "1617")
    assert len(boro_squad) == 13

    class _MockBest11Repo:
        @staticmethod
        def find_team_seasons(team, league_code):
            return ["2627"]

        @staticmethod
        def find_cached_seasons(team, league_code, provider="all"):
            return []

        @staticmethod
        def load_totals(league_code, season):
            if season == "2627":
                return {
                    "Coventry": TeamTotals(
                        team="Coventry", league_code=league_code, season="2627",
                        matches=10, gf=14.0, ga=12.0, xg=13.5, xga=11.0,
                        shots=110.0, shots_against=95.0, sot=42.0, sot_against=34.0,
                        corners=50.0, corners_against=40.0, clean_sheets=3,
                    )
                }
            return {}

        @staticmethod
        def load_squad(team, league_code, season, provider="all", refresh=False):
            return cov_squad

    best11 = Best11Service(_MockBest11Repo()).solve("Coventry", "E0", season="2425")
    assert "error" not in best11
    assert best11["season"] == "2627"
    assert len(best11["lineup"]) == 11


def test_prediction_service_parses_variant_slm_json_schemas() -> None:
    """Verify PredictionService unpacks nested/variant SLM JSON without returning empty summary."""
    from api.services.prediction_service import PredictionService

    slm_variant_payload = json.dumps({
        "prediction": (
            "The match is likely a low-scoring battle where Coventry's superior defensive discipline "
            "will limit Middlesbrough's ability to break through."
        ),
        "tactical_analysis": {
            "home_team_strength": {
                "attributes": ["high_defense_volume", "consistent_clean_sheet_rate"],
                "evaluation": (
                    "Coventry's defensive structure is characterized by a high foul count, "
                    "supporting stability and limiting opponent shots."
                ),
            },
            "away_team_strength": (
                "Middlesbrough exhibits a balanced attack, generating limited pressure against deep blocks "
                "and struggling to disrupt organized midfield lines."
            ),
        },
        "confidence_rating": "Medium",
    })

    class _MockFeedbackRepo:
        @staticmethod
        async def get_prediction_override(home_team, away_team, match_date):
            return None

    class _MockRagWrapper:
        @staticmethod
        def get_available_teams():
            return ["Coventry", "Middlesbrough"]

        @staticmethod
        async def predict_structured(home_team, away_team):
            return {"predicted_result": "Draw", "probabilities": {"H": 0.32, "D": 0.55, "A": 0.13}}

        @staticmethod
        async def predict_match(home_team, away_team, as_of=None, gnn_prediction=None):
            return slm_variant_payload

        @staticmethod
        async def get_team_profile(team_name):
            return {"strengths": [], "weaknesses": []}

    svc = PredictionService(_MockFeedbackRepo(), _MockRagWrapper())  # type: ignore[arg-type]
    resp = asyncio.run(svc.predict("Coventry", "Middlesbrough", None))

    assert resp.tactical_analysis
    assert "Analysis details not parsed" not in resp.tactical_analysis
    assert "Coventry's defensive structure" in resp.tactical_analysis
    assert resp.analysis_breakdown is not None
    assert resp.analysis_breakdown["tactical_matchup_summary"] == resp.tactical_analysis
    assert "low-scoring battle" in resp.analysis_breakdown["prediction_verdict"]
    assert len(resp.analysis_breakdown["home_team_analysis"]["strengths"]) >= 1
    assert len(resp.analysis_breakdown["away_team_analysis"]["weaknesses"]) >= 1


def test_parse_malformed_slm_pseudo_json_and_zero_brace_gate() -> None:
    """Verify 4-layer parser repairs comma-for-colon & strips broken keys/braces without leaking syntax."""
    from api.services.prediction_service import PredictionService
    from api.utils import parse_slm_tactical_response, sanitize_slm_prose

    broken_slm_output = (
        '{\n'
        '  "prediction_verity": "The home side, Manchester City, dominates both statistical metrics and '
        'defensive discipline, securing a superior clean-sheet record and consistent shot volume.",\n'
        '  "confidence_rating", "High"\n'
        '}\n'
        '{\n'
        '  "$verdict": \n'
        'tactical_matchsummary\': "\n'
    )

    parsed = parse_slm_tactical_response(broken_slm_output, "Man City", "Arsenal")
    expected_sentence = (
        "The home side, Manchester City, dominates both statistical metrics and defensive discipline, "
        "securing a superior clean-sheet record and consistent shot volume."
    )
    assert parsed["is_fallback"] is False
    assert parsed["prediction_verdict"] == expected_sentence
    assert parsed["tactical_matchup_summary"] == expected_sentence
    assert parsed["confidence_rating"] == "High"

    sanitized = sanitize_slm_prose(broken_slm_output)
    assert sanitized == expected_sentence
    for field_text in (
        parsed["prediction_verdict"],
        parsed["tactical_matchup_summary"],
        sanitized,
    ):
        for forbidden in ("{", "}", "$verdict", "prediction_verity", "confidence_rating", "tactical_matchsummary"):
            assert forbidden not in field_text

    class _MockFeedbackRepo:
        @staticmethod
        async def get_prediction_override(home_team, away_team, match_date):
            return None

    class _MockRagWrapper:
        @staticmethod
        def get_available_teams():
            return ["Man City", "Arsenal"]

        @staticmethod
        async def predict_structured(home_team, away_team):
            return {"predicted_result": "H", "probabilities": {"H": 0.61, "D": 0.22, "A": 0.17}}

        @staticmethod
        async def predict_match(home_team, away_team, as_of=None, gnn_prediction=None):
            return broken_slm_output

        @staticmethod
        async def get_team_profile(team_name):
            return {"strengths": [], "weaknesses": []}

    svc = PredictionService(_MockFeedbackRepo(), _MockRagWrapper())  # type: ignore[arg-type]
    resp = asyncio.run(svc.predict("Man City", "Arsenal", None))
    assert resp.tactical_analysis == expected_sentence
    assert resp.analysis_breakdown is not None
    assert resp.analysis_breakdown["confidence_rating"] == "High"
    assert resp.analysis_breakdown["prediction_verdict"] == expected_sentence
    assert resp.analysis_breakdown["tactical_matchup_summary"] == expected_sentence

    # Simulator handles broken JSON without leaking braces or keys
    sim = TacticalSimulatorService()
    sim_res = sim.simulate(
        TacticalSimulationRequest(
            home_team="Man City",
            away_team="Arsenal",
            home_formation="4-3-3",
            away_formation="4-2-3-1",
            tactical_preset="high_press",
        ),
        baseline_probs={"H": 0.61, "D": 0.22, "A": 0.17},
        slm_generator=lambda _p, _s=None: broken_slm_output,
        llm_provider="huggingface",
    )
    assert sim_res.recommendation == expected_sentence
    assert "{" not in sim_res.recommendation and "}" not in sim_res.recommendation


def test_vllm_canonical_provider_and_dynamic_shared_city_scouting() -> None:
    """Verify vllm is the canonical default provider and shared-city SLM prose extracts dynamically."""
    from api.services.prediction_service import PredictionService
    from api.utils import extract_team_analysis, parse_slm_tactical_response
    from models.llm_providers import resolve_default_provider

    assert resolve_default_provider() == "vllm"

    narrative = (
        "Manchester City's dominant shot volume (xG = 2.35) indicates a structured, high-intensity attack "
        "where corners and shots-on-target (SOT) contribute to disrupting opposing play. "
        "Their low foul rate (FoulsAgainst=0.95) reflects a disciplined defensive posture, "
        "minimizing chances for opponent possession. "
        "Conversely, Manchester United exhibits a higher tendency to concede shots (CleanSheets=0.59), "
        "indicating a lack of confidence or stability in stopping City's pace."
    )

    extracted = extract_team_analysis(narrative, "Manchester City", "Manchester United")
    home_str = extracted["home_team_analysis"]["strengths"]
    away_wk = extracted["away_team_analysis"]["weaknesses"]
    assert len(home_str) >= 2
    assert any("dominant shot volume" in s for s in home_str)
    assert any("low foul rate" in s for s in home_str)
    assert len(away_wk) >= 1
    assert any("concede shots" in w for w in away_wk)

    # Multi-key SLM JSON retains both summary and extra evaluation sentences (no truncation)
    multi_json = json.dumps({
        "summary": "Manchester City control territory through high-volume pressing.",
        "home_evaluation": "Their low foul rate reflects a disciplined defensive posture.",
        "away_evaluation": "Manchester United struggle with a tendency to concede shots in transition.",
    })
    parsed_multi = parse_slm_tactical_response(multi_json, "Manchester City", "Manchester United")
    assert "control territory" in parsed_multi["tactical_matchup_summary"]
    assert "disciplined defensive posture" in parsed_multi["tactical_matchup_summary"]
    assert "concede shots" in parsed_multi["tactical_matchup_summary"]

    # Verify PredictionService keeps SLM-extracted buckets AND backfills empty buckets from KG profile
    class _MockFeedbackRepo:
        @staticmethod
        async def get_prediction_override(home_team, away_team, match_date):
            return None

    class _MockRagWrapper:
        @staticmethod
        def get_available_teams():
            return ["Manchester City", "Manchester United"]

        @staticmethod
        async def predict_structured(home_team, away_team):
            return {"predicted_result": "Home Win", "probabilities": {"H": 0.65, "D": 0.20, "A": 0.15}}

        @staticmethod
        async def predict_match(home_team, away_team, as_of=None, gnn_prediction=None):
            return narrative

        @staticmethod
        async def get_team_profile(team_name):
            return {"strengths": [f"KG_STRENGTH_{team_name}"], "weaknesses": [f"KG_WEAK_{team_name}"]}

    svc = PredictionService(_MockFeedbackRepo(), _MockRagWrapper())  # type: ignore[arg-type]
    resp = asyncio.run(svc.predict("Manchester City", "Manchester United", None))
    assert resp.analysis_breakdown is not None
    home_blk = resp.analysis_breakdown["home_team_analysis"]
    away_blk = resp.analysis_breakdown["away_team_analysis"]
    # Home strengths extracted dynamically from SLM (not overwritten by KG backfill)
    assert len(home_blk["strengths"]) >= 2
    assert all("KG_STRENGTH" not in s for s in home_blk["strengths"])
    # Away weaknesses extracted dynamically from SLM (not overwritten by KG backfill)
    assert len(away_blk["weaknesses"]) >= 1
    assert all("KG_WEAK" not in w for w in away_blk["weaknesses"])
    # Empty buckets backfilled from KG so neither team ever shows an empty scouting card
    assert home_blk["weaknesses"] == ["KG_WEAK_Manchester City"]
    assert away_blk["strengths"] == ["KG_STRENGTH_Manchester United"]

    # Section headings must never pollute verdict/weaknesses, and negated strengths become weaknesses
    heading_prose = (
        "**Match Verdict & Probability Integration**\n"
        "My analysis points firmly towards a Home Win for Manchester City.\n\n"
        "**Manchester City's Tactical Strengths & Weaknesses**\n"
        "Manchester City are creating 6.0 shots on target per match.\n"
        "The main weakness, however, lies in conceding goals at a rate that is not negligible.\n\n"
        "**Manchester United's Tactical Strengths & Weaknesses**\n"
        "Their home attacking tally is solid, with 2.00 PPM.\n"
        "However, their defensive discipline is not as effective and they are prone to conceding goals."
    )
    parsed_headings = parse_slm_tactical_response(heading_prose, "Manchester City", "Manchester United")
    assert "Match Verdict & Probability Integration" not in parsed_headings["prediction_verdict"]
    assert "points firmly towards a Home Win" in parsed_headings["prediction_verdict"]
    home_wk = parsed_headings["home_team_analysis"]["weaknesses"]
    away_wk_h = parsed_headings["away_team_analysis"]["weaknesses"]
    assert all("Tactical Strengths" not in item for item in home_wk + away_wk_h)
    assert any("not negligible" in item for item in home_wk)
    assert any("not as effective" in item or "prone to conceding" in item for item in away_wk_h)

    # VllmProvider auto-clamps & retries on context length 400 error
    from models.llm_providers.vllm import VllmProvider

    vp = VllmProvider()
    calls: list[int] = []

    class _FakeChoice:
        class _Msg:
            content = "Clean retry output."
        message = _Msg()

    class _FakeResp:
        choices = [_FakeChoice()]

    def _fake_create(**kwargs):
        calls.append(kwargs["max_tokens"])
        if len(calls) == 1:
            raise ValueError(
                "Error code: 400 - maximum context length is 2048 tokens; "
                "your prompt contains at least 1412 input tokens (parameter=input_tokens, value=1412)"
            )
        return _FakeResp()

    vp._client.chat.completions.create = _fake_create  # type: ignore[method-assign]
    assert vp._call_api("x" * 100) == "Clean retry output."  # pylint: disable=protected-access
    assert len(calls) == 2 and calls[1] == 2032 - 1412
