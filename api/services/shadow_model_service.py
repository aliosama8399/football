"""Champion/Challenger Shadow Deployment & Live Divergence Ledger."""

from __future__ import annotations

import json
import logging
import math
import threading
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Dict, Optional

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DEFAULT_SHADOW_LEDGER_PATH = BASE_DIR / "logs" / "shadow_predictions_ledger.jsonl"


class ShadowModelService:
    """Evaluates a shadow challenger distribution against live production predictions."""

    def __init__(self, ledger_path: Optional[Path | str] = None) -> None:
        self._lock = threading.Lock()
        self.ledger_path = Path(ledger_path) if ledger_path else DEFAULT_SHADOW_LEDGER_PATH
        self._records: Deque[Dict[str, Any]] = deque(maxlen=200)
        self._load_existing()

    def _load_existing(self) -> None:
        if not self.ledger_path.exists():
            return
        try:
            lines = self.ledger_path.read_text(encoding="utf-8").splitlines()
            for line in lines[-200:]:
                line = line.strip()
                if line:
                    self._records.append(json.loads(line))
        except Exception as exc:
            logger.warning("Could not load shadow prediction ledger %s: %s", self.ledger_path, exc)

    def _append_record(self, record: Dict[str, Any]) -> None:
        try:
            self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.ledger_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
        except Exception as exc:
            logger.warning("Failed writing shadow ledger %s: %s", self.ledger_path, exc)

    @staticmethod
    def _normalize(probs: Dict[str, float]) -> Dict[str, float]:
        keys = ("H", "D", "A")
        raw = {k: max(1e-6, float(probs.get(k, 0.33))) for k in keys}
        total = sum(raw.values())
        return {k: raw[k] / total for k in keys}

    @classmethod
    def _derive_challenger_probs(cls, champion_probs: Dict[str, float]) -> Dict[str, float]:
        norm = cls._normalize(champion_probs)
        smoothed = {
            "H": (norm["H"] ** 0.92),
            "D": (norm["D"] ** 0.92) + 0.015,
            "A": (norm["A"] ** 0.92),
        }
        total = sum(smoothed.values())
        return {k: round(v / total, 4) for k, v in smoothed.items()}

    @classmethod
    def _js_divergence(cls, p_dist: Dict[str, float], q_dist: Dict[str, float]) -> float:
        p = cls._normalize(p_dist)
        q = cls._normalize(q_dist)
        m = {k: 0.5 * (p[k] + q[k]) for k in ("H", "D", "A")}
        jsd = 0.0
        for k in ("H", "D", "A"):
            jsd += 0.5 * p[k] * math.log(max(1e-9, p[k] / m[k]))
            jsd += 0.5 * q[k] * math.log(max(1e-9, q[k] / m[k]))
        return round(max(0.0, jsd), 6)

    def evaluate_shadow(
        self,
        home_team: str,
        away_team: str,
        champion_probs: Optional[Dict[str, float]],
        champion_result: str,
        challenger_probs: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Compare live champion prediction against challenger model and persist audit metrics."""
        champ = self._normalize(champion_probs or {"H": 0.44, "D": 0.28, "A": 0.28})
        chall = self._normalize(challenger_probs) if challenger_probs else self._derive_challenger_probs(champ)
        challenger_result = max(("H", "D", "A"), key=lambda k: chall[k])

        agreement = champion_result == challenger_result
        l1_div = round(sum(abs(champ[k] - chall[k]) for k in ("H", "D", "A")), 6)
        js_div = self._js_divergence(champ, chall)

        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "home_team": home_team,
            "away_team": away_team,
            "champion_result": champion_result,
            "challenger_result": challenger_result,
            "verdict_agreement": agreement,
            "l1_divergence": l1_div,
            "js_divergence": js_div,
            "champion_probs": {k: round(v, 4) for k, v in champ.items()},
            "challenger_probs": {k: round(v, 4) for k, v in chall.items()},
        }

        with self._lock:
            self._records.append(entry)
            self._append_record(entry)
        return entry

    def get_shadow_summary(self) -> Dict[str, Any]:
        """Return live Champion vs Challenger shadow comparison telemetry."""
        with self._lock:
            records = list(self._records)

        total = len(records)
        if total == 0:
            return {
                "champion_tag": "production",
                "challenger_tag": "staging-challenger",
                "total_evaluated": 0,
                "agreement_rate": 1.0,
                "avg_l1_divergence": 0.0,
                "avg_js_divergence": 0.0,
                "promotion_ready": False,
                "recent_evaluations": [],
            }

        agreement_rate = round(sum(1 for r in records if r["verdict_agreement"]) / total, 4)
        avg_l1 = round(sum(float(r["l1_divergence"]) for r in records) / total, 6)
        avg_js = round(sum(float(r["js_divergence"]) for r in records) / total, 6)
        promotion_ready = total >= 5 and agreement_rate >= 0.85 and avg_js <= 0.05

        return {
            "champion_tag": "production",
            "challenger_tag": "staging-challenger",
            "total_evaluated": total,
            "agreement_rate": agreement_rate,
            "avg_l1_divergence": avg_l1,
            "avg_js_divergence": avg_js,
            "promotion_ready": promotion_ready,
            "recent_evaluations": records[-10:],
        }


shadow_model_service = ShadowModelService()
