"""Smoke test: full LivePredictionService pipeline with a fake RAG wrapper (no DB/LLM)."""
import asyncio
import json
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.schemas import LivePredictionRequest
from api.services.live_prediction_service import LivePredictionService


class FakeRAG:
    def __init__(self):
        self.teams = ["Arsenal", "Chelsea", "Liverpool", "Man_City"]

    def get_available_teams(self):
        return self.teams

    async def predict_structured(self, home, away):
        # Arsenal vs Chelsea: mildly home-leaning prior
        if home == "Arsenal" and away == "Chelsea":
            return {"predicted_result": "Home Win", "probabilities": {"H": 0.45, "D": 0.29, "A": 0.26}}
        return None

    async def predict_live_match(self, home, away, ctx):
        return json.dumps({
            "match_state": {"minute": ctx.get("minute"), "score": f"{ctx.get('home_goals')}-{ctx.get('away_goals')}"},
            "analysis": {
                "who_controls_now": "Arsenal are controlling territory via corners and shots, but Wolves sit deep.",
                "why": [
                    "Home shots are above season pace while away xG is below pace.",
                    "Away yellows are over pace, signalling a stretched defensive block.",
                ],
                "how_outlook_changed": "The pre-match away edge has collapsed as Wolves fail to impose their attacking plan.",
                "coach_recommendations": [
                    {"priority": 1, "action": "Push both fullbacks high and widen the attack.", "reason": "Corners are over pace but shots-on-target are not."},
                    {"priority": 2, "action": "Bring on a second striker at 65'.", "reason": "Central occupation is missing in the final third."},
                ],
            },
        }, ensure_ascii=False)

    async def get_team_profile(self, team):
        if team == "Arsenal":
            return {"avg_xg": 1.9, "avg_xga": 0.9, "avg_shots": 15.0, "avg_sot": 5.2,
                    "avg_corners": 6.0, "avg_fouls": 10.0, "avg_yellows": 1.8,
                    "avg_goals_home": 2.1, "avg_goals_away": 1.4}
        return {"avg_xg": 1.4, "avg_xga": 1.2, "avg_shots": 12.0, "avg_sot": 4.0,
                "avg_corners": 5.0, "avg_fouls": 11.0, "avg_yellows": 2.0,
                "avg_goals_home": 1.8, "avg_goals_away": 1.3}


async def main():
    svc = LivePredictionService(FakeRAG())

    # Case 1: 60th minute, 1-0, no stats, explain=False
    r1 = await svc.predict(LivePredictionRequest(
        home_team="Arsenal", away_team="Chelsea", minute=60,
        home_goals=1, away_goals=0, explain=False))
    print("\nCASE 1 (1-0, 60', no stats):")
    print("  result:", r1.predicted_result, "| probs:", r1.probabilities)
    print("  pre  :", r1.pre_match_probabilities)
    print("  delta:", r1.delta)
    print("  expected score:", r1.expected_final_score)
    print("  drivers:", len(r1.key_drivers), "| explain:", r1.explain, "| src:", r1.source)
    assert sum(r1.probabilities.values()) - 1.0 < 0.01
    assert r1.explain is False
    assert r1.tactical_analysis is None

    # Case 2: 70th minute, 0-1 (behind), full stats, explain=True
    r2 = await svc.predict(LivePredictionRequest(
        home_team="Arsenal", away_team="Chelsea", minute=70,
        home_goals=0, away_goals=1,
        home_shots=18, home_sot=6, home_xg=2.4, home_corners=8, home_fouls=9,
        home_yellows=1, home_reds=0,
        away_shots=4, away_sot=1, away_xg=0.3, away_corners=1, away_fouls=12,
        away_yellows=3, away_reds=0,
        explain=True))
    print("\nCASE 2 (0-1, 70', full stats, explain=True):")
    print("  result:", r2.predicted_result, "| probs:", r2.probabilities)
    print("  drivers:", [(d["side"], d["label"], d["pace"]) for d in r2.key_drivers][:5])
    print("  analysis:", (r2.tactical_analysis or "")[:60])
    assert r2.explain is True
    assert r2.source == "live_model+llm"
    assert r2.tactical_analysis is not None
    bd = r2.tactical_breakdown
    assert bd and bd["analysis"]["coach_recommendations"], "tactical_breakdown missing"
    assert bd["analysis"]["who_controls_now"]
    assert bd["analysis"]["why"]
    assert bd["analysis"]["how_outlook_changed"]
    assert bd["match_state"]["minute"] == 70
    assert bd["match_state"]["score"] == "0-1"
    print("  breakdown keys:", sorted(bd["analysis"].keys()), "| recs:", len(bd["analysis"]["coach_recommendations"]))

    # Case 3: team not in pool -> HTTPException
    from fastapi import HTTPException
    try:
        await svc.predict(LivePredictionRequest(home_team="Unknown", away_team="Chelsea", minute=30))
        print("\nCASE 3: FAIL — should have raised")
    except HTTPException as e:
        print("\nCASE 3 (unknown team) -> HTTP 400 OK")

    print("\nSMOKE TEST PASSED")


asyncio.run(main())
