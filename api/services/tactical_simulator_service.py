"""Counterfactual Tactical Simulator & SLM-Powered Pre-Match Scouting Dossier Engine."""

from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from api.schemas import (
    PreMatchDossierRequest,
    PreMatchDossierResponse,
    TacticalSimulationRequest,
    TacticalSimulationResponse,
)
from api.utils import (
    clean_slm_output,
    extract_json_object,
    is_rag_fallback_payload,
    parse_slm_tactical_response,
    sanitize_slm_prose,
    strip_markdown,
)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
TEAM_TACTICS_PATH = BASE_DIR / "rag" / "knowledge_base" / "team_tactics.json"

SUPPORTED_FORMATIONS: Tuple[str, ...] = ("4-3-3", "4-2-3-1", "3-5-2", "4-4-2", "5-3-2", "3-4-3")

# Calibrated log-odds modifier for home team (delta_h, delta_d, delta_a) by (home_formation, away_formation)
FORMATION_SYNERGY_MATRIX: Dict[Tuple[str, str], Tuple[float, float, float]] = {
    ("4-3-3", "4-3-3"): (0.00, 0.00, 0.00),
    ("4-3-3", "4-2-3-1"): (-0.06, 0.04, 0.03),
    ("4-3-3", "3-5-2"): (0.09, -0.02, -0.07),
    ("4-3-3", "4-4-2"): (0.11, -0.04, -0.08),
    ("4-3-3", "5-3-2"): (-0.04, 0.07, -0.02),
    ("4-3-3", "3-4-3"): (0.05, -0.03, -0.03),
    ("4-2-3-1", "4-3-3"): (0.07, -0.01, -0.06),
    ("4-2-3-1", "4-2-3-1"): (0.00, 0.02, -0.01),
    ("4-2-3-1", "3-5-2"): (0.04, 0.01, -0.05),
    ("4-2-3-1", "4-4-2"): (0.12, -0.03, -0.09),
    ("4-2-3-1", "5-3-2"): (-0.02, 0.05, -0.03),
    ("4-2-3-1", "3-4-3"): (0.06, -0.02, -0.04),
    ("3-5-2", "4-3-3"): (-0.04, 0.02, 0.03),
    ("3-5-2", "4-2-3-1"): (-0.03, 0.02, 0.02),
    ("3-5-2", "3-5-2"): (0.00, 0.03, -0.02),
    ("3-5-2", "4-4-2"): (0.14, -0.03, -0.10),
    ("3-5-2", "5-3-2"): (0.05, 0.06, -0.08),
    ("3-5-2", "3-4-3"): (0.08, -0.04, -0.05),
    ("4-4-2", "4-3-3"): (-0.10, 0.02, 0.08),
    ("4-4-2", "4-2-3-1"): (-0.11, 0.01, 0.09),
    ("4-4-2", "3-5-2"): (-0.09, 0.02, 0.07),
    ("4-4-2", "4-4-2"): (0.00, 0.04, -0.03),
    ("4-4-2", "5-3-2"): (-0.05, 0.08, -0.02),
    ("4-4-2", "3-4-3"): (-0.07, -0.02, 0.08),
    ("5-3-2", "4-3-3"): (-0.05, 0.14, -0.06),
    ("5-3-2", "4-2-3-1"): (-0.04, 0.12, -0.05),
    ("5-3-2", "3-5-2"): (-0.06, 0.11, -0.03),
    ("5-3-2", "4-4-2"): (0.03, 0.09, -0.09),
    ("5-3-2", "5-3-2"): (-0.02, 0.15, -0.08),
    ("5-3-2", "3-4-3"): (-0.01, 0.10, -0.07),
    ("3-4-3", "4-3-3"): (0.04, -0.06, 0.02),
    ("3-4-3", "4-2-3-1"): (0.02, -0.05, 0.03),
    ("3-4-3", "3-5-2"): (-0.04, -0.03, 0.06),
    ("3-4-3", "4-4-2"): (0.10, -0.04, -0.06),
    ("3-4-3", "5-3-2"): (0.06, 0.02, -0.07),
    ("3-4-3", "3-4-3"): (0.00, -0.08, 0.05),
}

# Quantitative counterfactual parameters only — zero canned tactical text.
PRESET_MODIFIERS: Dict[str, Dict[str, Tuple[float, ...]]] = {
    "balanced": {"logits": (0.00, 0.00, 0.00), "xg_tempo": (0.00, 0.00)},
    "high_press": {"logits": (0.11, -0.09, -0.02), "xg_tempo": (0.28, 0.16)},
    "low_block": {"logits": (-0.08, 0.22, -0.10), "xg_tempo": (-0.34, -0.28)},
    "counter_attack": {"logits": (0.06, -0.03, -0.04), "xg_tempo": (0.14, -0.05)},
    "possession_overload": {"logits": (0.10, 0.04, -0.14), "xg_tempo": (0.18, -0.18)},
    "wing_play": {"logits": (0.08, -0.02, -0.06), "xg_tempo": (0.21, 0.06)},
}

SLMGenerator = Callable[[str, Optional[str]], str]

SIMULATOR_SYSTEM_PROMPT = (
    "You are Expert 2, a fine-tuned football tactical analyst. "
    "Analyze the counterfactual formation matchup, pressing preset, TEA-GNN probability shifts, "
    "and retrieved Knowledge Graph facts. Ground every claim strictly in the provided match context. "
    "Respond cleanly with the requested JSON object containing concrete match-specific tactical analysis."
)

DOSSIER_SYSTEM_PROMPT = (
    "You are Expert 2, a fine-tuned elite football match-preparation and recruitment analyst. "
    "Author an executive pre-match scouting dossier strictly grounded in the TEA-GNN forecast, "
    "optimal counterfactual gameplan, and retrieved Knowledge Graph team profiles. "
    "Respond cleanly with the requested JSON object."
)

_POS_KEYWORDS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("GK", ("keeper", "goalkeeper", "shot-stopper")),
    ("DF", ("center-back", "centre-back", "fullback", "defender", "defensive", "aerial", "set-piece", "high line")),
    ("MF", ("midfield", "pivot", "press", "progression", "transition", "control", "tempo")),
    ("FW", ("striker", "finisher", "forward", "winger", "box", "xg", "chance", "finishing")),
)


def _clean_items(items: Any, limit: int = 4) -> List[str]:
    if not isinstance(items, list):
        return []
    out: List[str] = []
    for raw in items:
        raw_str = str(raw).strip()
        txt = sanitize_slm_prose(raw_str, min_words=2)
        if not raw_str.endswith((".", "!", "?")) and txt.endswith("."):
            txt = txt[:-1]
        if txt and txt not in out:
            out.append(txt)
        if len(out) >= limit:
            break
    return out


def _slm_sentences(text: str) -> List[str]:
    cleaned = sanitize_slm_prose(text)
    parts = [s.strip(" •*-–\t\n") for s in re.split(r"(?<=[.!?])\s+|\n+", cleaned)]
    return [p for p in parts if len(p) >= 16]


def _extract_tradeoffs_from_slm_prose(text: str) -> Tuple[str, List[str], List[str]]:
    """Derive recommendation and tradeoff bullets directly from the SLM's own generated prose."""
    sentences = _slm_sentences(text)
    if not sentences:
        return "", [], []
    weak_markers = ("risk", "expos", "vulnerab", "conced", "weak", "counter", "space behind", "struggl", "gap")
    strengths: List[str] = []
    vulnerabilities: List[str] = []
    for sent in sentences:
        low = sent.lower()
        if any(m in low for m in weak_markers):
            if len(vulnerabilities) < 3 and sent not in vulnerabilities:
                vulnerabilities.append(sent)
        elif len(strengths) < 3 and sent not in strengths:
            strengths.append(sent)
    recommendation = sanitize_slm_prose(" ".join(sentences[:2]))
    return recommendation, strengths, vulnerabilities


def _infer_pos_tag(text: str) -> str:
    low = text.lower()
    for pos, words in _POS_KEYWORDS:
        if any(w in low for w in words):
            return pos
    return "MF"


def _extract_json_prose(node: Any) -> str:
    """Recursively collect readable tactical prose sentences from arbitrary SLM JSON dicts."""
    collected: List[str] = []

    def _walk(item: Any) -> None:
        if isinstance(item, str):
            txt = sanitize_slm_prose(item)
            if len(txt) >= 16 and txt not in collected:
                collected.append(txt)
        elif isinstance(item, list):
            for el in item:
                _walk(el)
        elif isinstance(item, dict):
            for key, val in item.items():
                if str(key).lower() in ("confidence_rating", "attributes"):
                    continue
                _walk(val)

    _walk(node)
    return sanitize_slm_prose(" ".join(collected))


class TacticalSimulatorService:
    """Counterfactual tactical engine paired with live fine-tuned SLM synthesis."""

    @staticmethod
    def _normalize_probs(raw: Optional[Dict[str, float]]) -> Dict[str, float]:
        fallback = {"H": 0.44, "D": 0.28, "A": 0.28}
        src = raw or fallback
        vals = {
            "H": max(1e-4, float(src.get("H", 0.44))),
            "D": max(1e-4, float(src.get("D", 0.28))),
            "A": max(1e-4, float(src.get("A", 0.28))),
        }
        total = vals["H"] + vals["D"] + vals["A"]
        return {k: round(v / total, 4) for k, v in vals.items()}

    @staticmethod
    def _apply_log_odds_shift(
        base: Dict[str, float], delta_h: float, delta_d: float, delta_a: float
    ) -> Dict[str, float]:
        scores = {
            "H": math.exp(math.log(max(1e-6, base["H"])) + delta_h),
            "D": math.exp(math.log(max(1e-6, base["D"])) + delta_d),
            "A": math.exp(math.log(max(1e-6, base["A"])) + delta_a),
        }
        total = scores["H"] + scores["D"] + scores["A"]
        h_val = round(scores["H"] / total, 4)
        d_val = round(scores["D"] / total, 4)
        a_val = round(max(0.0, 1.0 - h_val - d_val), 4)
        return {"H": h_val, "D": d_val, "A": a_val}

    def compute_counterfactual(
        self,
        req: TacticalSimulationRequest,
        baseline_probs: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Pure quantitative log-odds / xPts / xG counterfactual shift."""
        base = self._normalize_probs(baseline_probs)
        f_pair = (req.home_formation, req.away_formation)
        f_h, f_d, f_a = FORMATION_SYNERGY_MATRIX.get(f_pair, (0.0, 0.0, 0.0))

        preset_key = req.tactical_preset if req.tactical_preset in PRESET_MODIFIERS else "balanced"
        preset_cfg = PRESET_MODIFIERS[preset_key]
        p_h, p_d, p_a = preset_cfg["logits"]

        impact_shift = float(req.key_player_impact) * 1.6
        sim = self._apply_log_odds_shift(
            base,
            delta_h=f_h + p_h + impact_shift,
            delta_d=f_d + p_d - 0.3 * abs(impact_shift),
            delta_a=f_a + p_a - impact_shift,
        )

        deltas = {k: round(sim[k] - base[k], 4) for k in ("H", "D", "A")}
        exp_pts_base = round(3.0 * base["H"] + 1.0 * base["D"], 3)
        exp_pts_sim = round(3.0 * sim["H"] + 1.0 * sim["D"], 3)
        exp_pts_delta = round(exp_pts_sim - exp_pts_base, 3)

        xg_h_adj, xg_a_adj = preset_cfg["xg_tempo"]
        home_xg = round(max(0.4, 1.45 + (sim["H"] - base["A"]) * 0.9 + xg_h_adj), 2)
        away_xg = round(max(0.35, 1.15 + (sim["A"] - base["H"]) * 0.8 + xg_a_adj), 2)

        return {
            "preset_key": preset_key,
            "baseline_probabilities": base,
            "simulated_probabilities": sim,
            "probability_deltas": deltas,
            "expected_points_baseline": exp_pts_base,
            "expected_points_simulated": exp_pts_sim,
            "expected_points_delta": exp_pts_delta,
            "expected_goals": {"home_xg": home_xg, "away_xg": away_xg},
        }

    @staticmethod
    def _load_tactical_profiles(
        home_team: str,
        away_team: str,
        home_profile: Optional[Dict[str, Any]],
        away_profile: Optional[Dict[str, Any]],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        disk_data: Dict[str, Any] = {}
        if (home_profile is None or away_profile is None) and TEAM_TACTICS_PATH.exists():
            try:
                disk_data = json.loads(TEAM_TACTICS_PATH.read_text(encoding="utf-8"))
            except Exception:
                disk_data = {}

        empty_prof: Dict[str, Any] = {"strengths": [], "weaknesses": []}
        resolved_home = home_profile or disk_data.get(home_team) or dict(empty_prof)
        resolved_away = away_profile or disk_data.get(away_team) or dict(empty_prof)
        return resolved_home, resolved_away

    @staticmethod
    def _invoke_slm(slm_generator: Optional[SLMGenerator], prompt: str, system_prompt: str) -> Tuple[str, Optional[dict]]:
        if slm_generator is None:
            raise RuntimeError(
                "SLM generator is required for tactical simulation and dossiers; "
                "static canned responses are disabled."
            )
        raw = slm_generator(prompt, system_prompt)
        cleaned = clean_slm_output(raw or "")
        if not cleaned:
            raise RuntimeError("SLM returned an empty tactical response.")
        parsed = extract_json_object(cleaned)
        if is_rag_fallback_payload(parsed):
            err = parsed.get("error") or parsed.get("note") or "SLM offline"
            raise RuntimeError(f"SLM inference degraded: {err}")
        return cleaned, parsed if isinstance(parsed, dict) else None

    def simulate(
        self,
        req: TacticalSimulationRequest,
        baseline_probs: Optional[Dict[str, float]] = None,
        home_profile: Optional[Dict[str, Any]] = None,
        away_profile: Optional[Dict[str, Any]] = None,
        kg_context: str = "",
        slm_generator: Optional[SLMGenerator] = None,
        llm_provider: str = "vllm",
    ) -> TacticalSimulationResponse:
        """Evaluate quantitative counterfactual shifts and generate live SLM tactical reasoning."""
        metrics = self.compute_counterfactual(req, baseline_probs)
        h_prof, a_prof = self._load_tactical_profiles(req.home_team, req.away_team, home_profile, away_profile)
        base = metrics["baseline_probabilities"]
        sim = metrics["simulated_probabilities"]
        deltas = metrics["probability_deltas"]
        xg = metrics["expected_goals"]

        kg_snippet = kg_context.strip()[:1200] if kg_context else ""
        prompt = (
            f"Simulate tactical matchup: {req.home_team} (Home, formation {req.home_formation}) vs "
            f"{req.away_team} (Away, formation {req.away_formation}).\n"
            f"Selected Home Tactical Preset: {metrics['preset_key']} | Key Player Availability Impact: {req.key_player_impact:+.2f}\n"
            f"TEA-GNN Baseline Probabilities: H={base['H']:.1%}, D={base['D']:.1%}, A={base['A']:.1%}\n"
            f"Counterfactual Probabilities: H={sim['H']:.1%} ({deltas['H']:+.1%}), D={sim['D']:.1%} ({deltas['D']:+.1%}), A={sim['A']:.1%} ({deltas['A']:+.1%})\n"
            f"Expected Points: {metrics['expected_points_baseline']:.3f} -> {metrics['expected_points_simulated']:.3f} ({metrics['expected_points_delta']:+.3f} xPts)\n"
            f"Projected Expected Goals: {req.home_team} {xg['home_xg']} xG vs {req.away_team} {xg['away_xg']} xG\n"
            f"{req.home_team} Profile Strengths: {', '.join(h_prof.get('strengths', [])[:3]) or 'N/A'} | Weaknesses: {', '.join(h_prof.get('weaknesses', [])[:3]) or 'N/A'}\n"
            f"{req.away_team} Profile Strengths: {', '.join(a_prof.get('strengths', [])[:3]) or 'N/A'} | Weaknesses: {', '.join(a_prof.get('weaknesses', [])[:3]) or 'N/A'}\n"
            + (f"\nKnowledge Graph Context:\n{kg_snippet}\n" if kg_snippet else "")
            + "\nRespond ONLY with valid JSON (no markdown fences) using this schema:\n"
            "{\n"
            f'  "recommendation": "<2-3 concrete match-specific sentences advising {req.home_team} on whether to adopt {req.home_formation} with {metrics["preset_key"]} vs {req.away_team}>",\n'
            '  "strengths_gained": ["<specific tactical advantage 1>", "<specific tactical advantage 2>"],\n'
            '  "vulnerabilities_exposed": ["<specific tactical risk 1>", "<specific tactical risk 2>"]\n'
            "}"
        )

        cleaned, parsed = self._invoke_slm(slm_generator, prompt, SIMULATOR_SYSTEM_PROMPT)

        recommendation = ""
        strengths: List[str] = []
        vulnerabilities: List[str] = []
        if parsed:
            raw_rec = (
                parsed.get("recommendation")
                or parsed.get("optimal_recommendation")
                or parsed.get("tactical_summary")
                or parsed.get("tactical_matchup_summary")
                or parsed.get("prediction_verdict")
                or parsed.get("prediction")
                or parsed.get("evaluation")
                or parsed.get("summary")
                or ""
            )
            recommendation = sanitize_slm_prose(str(raw_rec))
            if not recommendation:
                recommendation = _extract_json_prose(parsed)
            strengths = _clean_items(parsed.get("strengths_gained"))
            vulnerabilities = _clean_items(parsed.get("vulnerabilities_exposed"))

        if not recommendation or (not strengths and not vulnerabilities):
            slm_fallback = parse_slm_tactical_response(cleaned, req.home_team, req.away_team)
            fallback_source = (
                recommendation
                or slm_fallback["tactical_matchup_summary"]
                or slm_fallback["prediction_verdict"]
                or (_extract_json_prose(parsed) if parsed else "")
                or sanitize_slm_prose(cleaned)
            )
            prose_rec, prose_str, prose_vuln = _extract_tradeoffs_from_slm_prose(fallback_source)
            if not recommendation:
                recommendation = prose_rec or sanitize_slm_prose(fallback_source)
            if not strengths:
                strengths = prose_str
            if not vulnerabilities:
                vulnerabilities = prose_vuln
        recommendation = sanitize_slm_prose(recommendation) or strip_markdown(
            f"{req.home_team} {req.home_formation} ({metrics['preset_key']}) vs {req.away_team} {req.away_formation}: "
            f"projected {sim['H']:.1%} home win ({deltas['H']:+.1%}, {metrics['expected_points_delta']:+.3f} xPts)."
        )

        return TacticalSimulationResponse(
            home_team=req.home_team,
            away_team=req.away_team,
            home_formation=req.home_formation,
            away_formation=req.away_formation,
            tactical_preset=metrics["preset_key"],
            baseline_probabilities=base,
            simulated_probabilities=sim,
            probability_deltas=deltas,
            expected_points_baseline=metrics["expected_points_baseline"],
            expected_points_simulated=metrics["expected_points_simulated"],
            expected_points_delta=metrics["expected_points_delta"],
            expected_goals=xg,
            tactical_tradeoffs={
                "strengths_gained": strengths,
                "vulnerabilities_exposed": vulnerabilities,
            },
            recommendation=recommendation,
            slm_generated=True,
            llm_provider=llm_provider,
        )

    def generate_dossier(
        self,
        req: PreMatchDossierRequest,
        baseline_probs: Optional[Dict[str, float]] = None,
        home_profile: Optional[Dict[str, Any]] = None,
        away_profile: Optional[Dict[str, Any]] = None,
        kg_context: str = "",
        slm_generator: Optional[SLMGenerator] = None,
        llm_provider: str = "vllm",
    ) -> PreMatchDossierResponse:
        """Sweep formations/presets mathematically, then synthesize the full dossier via live SLM."""
        base = self._normalize_probs(baseline_probs)
        candidate_formations = ("4-3-3", "4-2-3-1", "3-5-2", "4-4-2")
        candidate_presets = ("balanced", "high_press", "possession_overload", "counter_attack")

        best_req: Optional[TacticalSimulationRequest] = None
        best_math: Optional[Dict[str, Any]] = None
        for formation in candidate_formations:
            for preset in candidate_presets:
                cand_req = TacticalSimulationRequest(
                    home_team=req.home_team,
                    away_team=req.away_team,
                    home_formation=formation,
                    away_formation="4-3-3",
                    tactical_preset=preset,
                )
                cand_math = self.compute_counterfactual(cand_req, baseline_probs=base)
                if best_math is None or cand_math["expected_points_simulated"] > best_math["expected_points_simulated"]:
                    best_req = cand_req
                    best_math = cand_math

        assert best_req is not None and best_math is not None
        h_prof, a_prof = self._load_tactical_profiles(req.home_team, req.away_team, home_profile, away_profile)
        generated_at = datetime.now(timezone.utc).isoformat()
        verdict = max(base, key=lambda k: base[k])
        sim_probs = best_math["simulated_probabilities"]
        deltas = best_math["probability_deltas"]
        xg = best_math["expected_goals"]

        kg_snippet = kg_context.strip()[:1400] if kg_context else ""
        prompt = (
            f"Compose a complete Executive Pre-Match Scouting Dossier for {req.home_team} (Home) vs {req.away_team} (Away) in {req.league}.\n"
            f"TEA-GNN Forecast: Verdict={verdict} | Probabilities: H={base['H']:.1%}, D={base['D']:.1%}, A={base['A']:.1%}\n"
            f"Mathematically Optimal Home Setup: {best_req.home_formation} ({best_math['preset_key']}) vs Away 4-3-3\n"
            f"Counterfactual Outcome: {sim_probs['H']:.1%} H ({deltas['H']:+.1%}), {sim_probs['D']:.1%} D, {sim_probs['A']:.1%} A | "
            f"xPts={best_math['expected_points_simulated']:.3f} ({best_math['expected_points_delta']:+.3f}) | "
            f"Projected xG: {req.home_team} {xg['home_xg']} vs {req.away_team} {xg['away_xg']}\n"
            f"{req.home_team} Strengths: {', '.join(h_prof.get('strengths', [])[:3]) or 'N/A'} | Weaknesses: {', '.join(h_prof.get('weaknesses', [])[:3]) or 'N/A'}\n"
            f"{req.away_team} Strengths: {', '.join(a_prof.get('strengths', [])[:3]) or 'N/A'} | Weaknesses: {', '.join(a_prof.get('weaknesses', [])[:3]) or 'N/A'}\n"
            + (f"\nKnowledge Graph Match & Form Data:\n{kg_snippet}\n" if kg_snippet else "")
            + "\nRespond with valid JSON matching this schema:\n"
            "{\n"
            '  "optimal_recommendation": "<2-3 sentence tactical gameplan recommendation>",\n'
            '  "strengths_gained": ["<advantage 1>", "<advantage 2>"],\n'
            '  "vulnerabilities_exposed": ["<risk 1>", "<risk 2>"],\n'
            '  "scouting_priorities": [\n'
            '    {"position": "GK|DF|MF|FW", "rationale": "<data-backed recruitment or tactical profile note>"}\n'
            "  ],\n"
            '  "executive_markdown": "<Full multi-section Markdown dossier covering GNN Forecast, Optimal Counterfactual Gameplan, Opposition Threats & Vulnerabilities, and Recruitment Signals>"\n'
            "}"
        )

        cleaned, parsed = self._invoke_slm(slm_generator, prompt, DOSSIER_SYSTEM_PROMPT)

        exec_md = ""
        opt_rec = ""
        strengths: List[str] = []
        vulnerabilities: List[str] = []
        priorities: List[Dict[str, str]] = []

        if parsed:
            exec_md = sanitize_slm_prose(str(parsed.get("executive_markdown") or ""), keep_markdown=True)
            raw_opt = (
                parsed.get("optimal_recommendation")
                or parsed.get("recommendation")
                or parsed.get("tactical_summary")
                or parsed.get("tactical_matchup_summary")
                or parsed.get("prediction_verdict")
                or parsed.get("prediction")
                or parsed.get("evaluation")
                or ""
            )
            opt_rec = sanitize_slm_prose(str(raw_opt))
            if not opt_rec:
                opt_rec = _extract_json_prose(parsed)
            strengths = _clean_items(parsed.get("strengths_gained"))
            vulnerabilities = _clean_items(parsed.get("vulnerabilities_exposed"))
            raw_prio = parsed.get("scouting_priorities")
            if isinstance(raw_prio, list):
                for item in raw_prio:
                    if isinstance(item, dict) and item.get("rationale"):
                        pos = str(item.get("position") or "MF").upper()
                        if pos not in ("GK", "DF", "MF", "FW"):
                            pos = _infer_pos_tag(str(item["rationale"]))
                        rat_txt = sanitize_slm_prose(str(item["rationale"]), min_words=2)
                        if not str(item["rationale"]).strip().endswith((".", "!", "?")) and rat_txt.endswith("."):
                            rat_txt = rat_txt[:-1]
                        if rat_txt:
                            priorities.append({"position": pos, "rationale": rat_txt})

        if not opt_rec:
            slm_fallback = parse_slm_tactical_response(cleaned, req.home_team, req.away_team)
            opt_rec = slm_fallback["tactical_matchup_summary"] or slm_fallback["prediction_verdict"]

        if not exec_md:
            # SLM replied directly in Markdown/prose or used variant JSON keys
            if parsed:
                md_parts = [f"# Pre-Match Scouting Dossier: {req.home_team} vs {req.away_team} ({req.league})"]
                if opt_rec:
                    md_parts.append(f"## Tactical Recommendation\n{opt_rec}")
                elif prose_fallback := _extract_json_prose(parsed):
                    md_parts.append(f"## Tactical Analysis\n{prose_fallback}")
                if strengths:
                    md_parts.append("## Strengths Gained\n" + "\n".join(f"- {s}" for s in strengths))
                if vulnerabilities:
                    md_parts.append("## Vulnerabilities Exposed\n" + "\n".join(f"- {v}" for v in vulnerabilities))
                if priorities:
                    md_parts.append(
                        "## Recruitment / Squad Gap Signals\n"
                        + "\n".join(f"- **{p['position']}**: {p['rationale']}" for p in priorities)
                    )
                exec_md = "\n\n".join(md_parts)
            else:
                exec_md = sanitize_slm_prose(cleaned, keep_markdown=True)

        if not opt_rec or (not strengths and not vulnerabilities):
            prose_rec, prose_str, prose_vuln = _extract_tradeoffs_from_slm_prose(exec_md or cleaned)
            if not opt_rec:
                opt_rec = prose_rec or sanitize_slm_prose(exec_md)
            if not strengths:
                strengths = prose_str
            if not vulnerabilities:
                vulnerabilities = prose_vuln

        opt_rec = sanitize_slm_prose(opt_rec)
        exec_md = sanitize_slm_prose(exec_md, keep_markdown=True)
        if not priorities:
            for sent in _slm_sentences(exec_md or cleaned)[:3]:
                priorities.append({"position": _infer_pos_tag(sent), "rationale": sent})

        best_sim = TacticalSimulationResponse(
            home_team=best_req.home_team,
            away_team=best_req.away_team,
            home_formation=best_req.home_formation,
            away_formation=best_req.away_formation,
            tactical_preset=best_math["preset_key"],
            baseline_probabilities=base,
            simulated_probabilities=sim_probs,
            probability_deltas=deltas,
            expected_points_baseline=best_math["expected_points_baseline"],
            expected_points_simulated=best_math["expected_points_simulated"],
            expected_points_delta=best_math["expected_points_delta"],
            expected_goals=xg,
            tactical_tradeoffs={
                "strengths_gained": strengths,
                "vulnerabilities_exposed": vulnerabilities,
            },
            recommendation=opt_rec,
            slm_generated=True,
            llm_provider=llm_provider,
        )

        return PreMatchDossierResponse(
            home_team=req.home_team,
            away_team=req.away_team,
            league=req.league,
            generated_at=generated_at,
            match_forecast={"verdict": verdict, "probabilities": base},
            optimal_formation_simulation=best_sim.model_dump(),
            tactical_profiles={"home": h_prof, "away": a_prof},
            scouting_priorities=priorities[:3],
            executive_markdown=exec_md,
            slm_generated=True,
            llm_provider=llm_provider,
        )


tactical_simulator = TacticalSimulatorService()
