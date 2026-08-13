"""
Unit tests for rag/live_adjustment.py — the pure-math live prediction layer.
Run: python -m pytest tests/test_live_adjustment.py -q
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from rag.live_adjustment import (
    outcome_probs,
    rates_from_probs,
    conditional_remaining_probs,
    expected_final_score,
    compute_pace_factors,
    blend_probs,
    extract_drivers,
    TOTAL_MINUTES,
)


def _norm(d):
    t = sum(d.values())
    return {k: v / t for k, v in d.items()}


class TestOutcomeProbs:
    def test_sums_to_one(self):
        for lh, la in ((0.5, 0.5), (1.4, 1.1), (3.0, 0.8), (0.1, 4.5)):
            p = outcome_probs(lh, la)
            assert abs(sum(p.values()) - 1.0) < 1e-6
            assert set(p) == {"H", "D", "A"}

    def test_strong_home_dominates(self):
        p = outcome_probs(3.0, 0.5)
        assert p["H"] > 0.8

    def test_equal_rates_favor_draw(self):
        p = outcome_probs(0.8, 0.8)
        assert p["D"] > p["H"] and p["D"] > p["A"]


class TestRatesFromProbs:
    def test_round_trip(self):
        for probs in ({"H": 0.5, "D": 0.28, "A": 0.22},
                      {"H": 0.3, "D": 0.3, "A": 0.4},
                      {"H": 0.15, "D": 0.2, "A": 0.65}):
            rates = rates_from_probs(probs)
            back = outcome_probs(rates["home"], rates["away"])
            for k in ("H", "D", "A"):
                assert abs(back[k] - probs[k]) < 0.03, (probs, rates, back)

    def test_uniform_rates_are_small(self):
        rates = rates_from_probs({"H": 1 / 3, "D": 1 / 3, "A": 1 / 3})
        assert 0.8 < rates["home"] < 1.6
        assert 0.8 < rates["away"] < 1.6


class TestConditionalRemaining:
    def test_sums_to_one(self):
        for hg, ag, m in ((0, 0, 60), (2, 0, 80), (1, 1, 30), (0, 3, 85)):
            p = conditional_remaining_probs(0.7, 0.6, hg, ag)
            assert abs(sum(p.values()) - 1.0) < 1e-6

    def test_lead_at_90_is_certain(self):
        p = conditional_remaining_probs(0.001, 0.001, 1, 0)
        assert p["H"] > 0.99

    def test_trailing_away_at_90_loses(self):
        p = conditional_remaining_probs(0.001, 0.001, 0, 1)
        assert p["A"] > 0.99

    def test_draw_at_90_is_draw(self):
        p = conditional_remaining_probs(0.001, 0.001, 2, 2)
        assert p["D"] > 0.99


class TestExpectedFinalScore:
    def test_basic(self):
        e = expected_final_score(1.4, 1.1, 60, 1, 0)
        # remaining fraction 30/90 = 1/3
        assert e["home"] == pytest.approx(1 + 1.4 / 3, abs=0.01)
        assert e["away"] == pytest.approx(0 + 1.1 / 3, abs=0.01)

    def test_scale_with_pace(self):
        e = expected_final_score(1.4, 1.1, 60, 1, 0, pace_h=2.0)
        assert e["home"] == pytest.approx(1 + 1.4 / 3 * 2.0, abs=0.01)


class TestComputePaceFactors:
    def test_no_stats_no_momentum(self):
        # No live stats provided: only score momentum can move the pace.
        # 0-0 at minute 30 vs ~0.47 expected goals each -> slight deflation.
        r = compute_pace_factors(
            minute=30, live_stats={k: None for k in ("xg", "shots", "sot", "corners", "fouls", "yellows")},
            season_avgs={}, goals=0, lmbda_pre=1.4, reds=0,
        )
        assert 0.8 < r["pace"] < 1.0
        # A team scoring ahead of expectation gets a boost.
        r2 = compute_pace_factors(
            minute=30, live_stats={k: None for k in ("xg", "shots", "sot", "corners", "fouls", "yellows")},
            season_avgs={}, goals=2, lmbda_pre=1.4, reds=0,
        )
        assert r2["pace"] > 1.0

    def test_overperformance_raises_pace(self):
        r = compute_pace_factors(
            minute=60,
            live_stats={"shots": 30, "xg": 3.0, "sot": 10, "corners": 12, "fouls": 10, "yellows": 2},
            season_avgs={"avg_shots": 12, "avg_xg": 1.2, "avg_sot": 4, "avg_corners": 5, "avg_fouls": 10, "avg_yellows": 2},
            goals=2, lmbda_pre=1.4, reds=0,
        )
        assert r["pace"] > 1.0

    def test_red_card_penalty(self):
        r0 = compute_pace_factors(60, {"shots": 5}, {"avg_shots": 12}, 0, 1.4, reds=0)
        r1 = compute_pace_factors(60, {"shots": 5}, {"avg_shots": 12}, 0, 1.4, reds=2)
        assert r1["pace"] < r0["pace"]
        assert r1["components"]["red_card"] == pytest.approx(0.5)

    def test_zero_minute_no_op(self):
        r = compute_pace_factors(0, {"shots": 5}, {"avg_shots": 12}, 0, 1.4, reds=0)
        assert r["pace"] == 1.0


class TestBlendProbs:
    def test_early_match_keeps_prior(self):
        b = blend_probs({"H": 0.6, "D": 0.25, "A": 0.15}, {"H": 0.1, "D": 0.2, "A": 0.7}, 0)
        assert b["H"] > 0.5

    def test_late_match_takes_live(self):
        b = blend_probs({"H": 0.6, "D": 0.25, "A": 0.15}, {"H": 0.1, "D": 0.2, "A": 0.7}, 90)
        assert b["A"] > 0.6

    def test_midpoint(self):
        b = blend_probs({"H": 0.6, "D": 0.25, "A": 0.15}, {"H": 0.2, "D": 0.2, "A": 0.6}, 45)
        assert abs(b["H"] - 0.4) < 0.05


class TestExtractDrivers:
    def test_ranks_overperformers_first(self):
        home = {"shots": 30, "xg": 2.5, "sot": 9, "corners": 10, "fouls": 8, "yellows": 1}
        away = {"shots": 3, "xg": 0.2, "sot": 1, "corners": 2, "fouls": 9, "yellows": 2}
        avgs = {"avg_shots": 12, "avg_xg": 1.2, "avg_sot": 4, "avg_corners": 5, "avg_fouls": 10, "avg_yellows": 2}
        drivers = extract_drivers(60, home, away, avgs, avgs, 0, 0, 0, 0, 1.4, 1.1)
        assert drivers, "drivers should not be empty"
        devs = [d["deviation"] for d in drivers]
        assert devs == sorted(devs, reverse=True)
        assert any(d["side"] == "home" for d in drivers)
        assert len(drivers) <= 10


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
