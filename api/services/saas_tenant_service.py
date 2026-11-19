"""Enterprise SaaS Multi-Tenancy, B2B API Keys & Sliding-Window Quota Manager."""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import secrets
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DEFAULT_TENANT_STORAGE_PATH = BASE_DIR / "logs" / "saas_tenants.json"

TIER_CONFIGS: Dict[str, Dict[str, Any]] = {
    "free": {
        "daily_quota": 50,
        "rpm_limit": 15,
        "features": ["predictions", "chat"],
    },
    "pro": {
        "daily_quota": 1000,
        "rpm_limit": 120,
        "features": ["predictions", "chat", "simulate", "dossier", "stream"],
    },
    "club_enterprise": {
        "daily_quota": 50000,
        "rpm_limit": 1000,
        "features": ["predictions", "chat", "simulate", "dossier", "stream", "shadow", "b2b_api"],
    },
}


class SaaSTenantManager:
    """Thread-safe SaaS tenant plan, B2B API key, and sliding-window quota manager."""

    def __init__(self, storage_path: Optional[Path | str] = None) -> None:
        self._lock = threading.Lock()
        self.storage_path = Path(storage_path) if storage_path else DEFAULT_TENANT_STORAGE_PATH
        self._tenants: Dict[str, Dict[str, Any]] = {}
        self._api_keys: Dict[str, Dict[str, Any]] = {}
        self._rpm_windows: Dict[str, Deque[float]] = defaultdict(deque)
        self._load_state()

    def _load_state(self) -> None:
        if not self.storage_path.exists():
            return
        try:
            payload = json.loads(self.storage_path.read_text(encoding="utf-8"))
            self._tenants = dict(payload.get("tenants", {}))
            self._api_keys = dict(payload.get("api_keys", {}))
        except Exception as exc:
            logger.warning("Failed loading SaaS tenant state from %s: %s", self.storage_path, exc)

    def _persist_state(self) -> None:
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            data = {"tenants": self._tenants, "api_keys": self._api_keys}
            self.storage_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception as exc:
            logger.warning("Failed persisting SaaS tenant state to %s: %s", self.storage_path, exc)

    @staticmethod
    def _today_utc() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def _ensure_tenant_locked(self, username: str, default_role: str = "user") -> Dict[str, Any]:
        today = self._today_utc()
        if username not in self._tenants:
            default_tier = "club_enterprise" if default_role == "supervisor" else "pro"
            self._tenants[username] = {
                "username": username,
                "organization": f"{username}-workspace",
                "tier": default_tier,
                "daily_used": 0,
                "quota_date": today,
            }
            self._persist_state()
        tenant = self._tenants[username]
        if tenant.get("quota_date") != today:
            tenant["daily_used"] = 0
            tenant["quota_date"] = today
            self._persist_state()
        return tenant

    def get_tenant_profile(self, username: str, default_role: str = "user") -> Dict[str, Any]:
        """Return enriched tenant profile including active tier limits and features."""
        with self._lock:
            tenant = self._ensure_tenant_locked(username, default_role=default_role)
            tier = tenant.get("tier", "pro")
            cfg = TIER_CONFIGS.get(tier, TIER_CONFIGS["pro"])
            return {
                **tenant,
                "daily_limit": cfg["daily_quota"],
                "daily_remaining": max(0, cfg["daily_quota"] - int(tenant.get("daily_used", 0))),
                "rpm_limit": cfg["rpm_limit"],
                "features": list(cfg["features"]),
            }

    def update_tenant_tier(
        self, username: str, tier: str, organization: Optional[str] = None
    ) -> Dict[str, Any]:
        """Upgrade or downgrade a tenant subscription plan and workspace name."""
        if tier not in TIER_CONFIGS:
            raise ValueError(f"Unsupported tier '{tier}'. Choose one of: {list(TIER_CONFIGS)}")
        with self._lock:
            tenant = self._ensure_tenant_locked(username)
            tenant["tier"] = tier
            if organization is not None and organization.strip():
                tenant["organization"] = organization.strip()
            self._persist_state()
            cfg = TIER_CONFIGS[tier]
            return {
                **tenant,
                "daily_limit": cfg["daily_quota"],
                "daily_remaining": max(0, cfg["daily_quota"] - int(tenant.get("daily_used", 0))),
                "rpm_limit": cfg["rpm_limit"],
                "features": list(cfg["features"]),
            }

    def create_api_key(self, username: str, label: str = "default") -> Dict[str, Any]:
        """Mint a B2B API key, returning plaintext secret once and persisting only SHA-256 hash."""
        raw_key = f"fk_live_{secrets.token_hex(16)}"
        key_hash = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()
        key_id = secrets.token_hex(6)
        key_prefix = f"{raw_key[:14]}..."
        created_at = datetime.now(timezone.utc).isoformat()

        record = {
            "key_id": key_id,
            "key_prefix": key_prefix,
            "key_hash": key_hash,
            "label": label,
            "username": username,
            "revoked": False,
            "created_at": created_at,
        }
        with self._lock:
            self._ensure_tenant_locked(username)
            self._api_keys[key_id] = record
            self._persist_state()

        return {
            "key_id": key_id,
            "api_key": raw_key,
            "key_prefix": key_prefix,
            "label": label,
            "username": username,
            "created_at": created_at,
        }

    def verify_api_key(self, raw_key: str) -> Optional[Dict[str, Any]]:
        """Verify B2B API key using SHA-256 and constant-time hmac.compare_digest."""
        if not raw_key or not raw_key.startswith("fk_live_"):
            return None
        candidate_hash = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()

        with self._lock:
            for record in self._api_keys.values():
                if record.get("revoked"):
                    continue
                stored_hash = record.get("key_hash", "")
                if stored_hash and hmac.compare_digest(candidate_hash, stored_hash):
                    tenant = self._ensure_tenant_locked(record["username"])
                    cfg = TIER_CONFIGS.get(tenant["tier"], TIER_CONFIGS["pro"])
                    return {
                        "key_id": record["key_id"],
                        "key_prefix": record["key_prefix"],
                        "label": record["label"],
                        "username": record["username"],
                        "tenant": {**tenant, "features": list(cfg["features"])},
                    }
        return None

    def list_api_keys(self, username: str) -> List[Dict[str, Any]]:
        """List non-secret API key metadata owned by username."""
        with self._lock:
            return [
                {k: v for k, v in rec.items() if k != "key_hash"}
                for rec in self._api_keys.values()
                if rec.get("username") == username
            ]

    def revoke_api_key(self, username: str, key_id: str) -> bool:
        """Revoke an API key owned by username."""
        with self._lock:
            rec = self._api_keys.get(key_id)
            if not rec or rec.get("username") != username:
                return False
            rec["revoked"] = True
            self._persist_state()
            return True

    def check_and_consume_quota(
        self, username: str, role: str = "user", cost: int = 1
    ) -> Dict[str, Any]:
        """Enforce sliding-window RPM rate limit and daily request quota for a tenant."""
        now = time.time()
        with self._lock:
            tenant = self._ensure_tenant_locked(username, default_role=role)
            tier = tenant.get("tier", "pro")
            cfg = TIER_CONFIGS.get(tier, TIER_CONFIGS["pro"])
            rpm_limit = int(cfg["rpm_limit"])
            daily_limit = int(cfg["daily_quota"])

            window = self._rpm_windows[username]
            cutoff = now - 60.0
            while window and window[0] < cutoff:
                window.popleft()

            daily_used = int(tenant.get("daily_used", 0))
            rpm_remaining = max(0, rpm_limit - len(window))
            daily_remaining = max(0, daily_limit - daily_used)

            if len(window) >= rpm_limit:
                return {
                    "allowed": False,
                    "tier": tier,
                    "daily_limit": daily_limit,
                    "daily_remaining": daily_remaining,
                    "rpm_limit": rpm_limit,
                    "rpm_remaining": 0,
                    "reason": "rpm_exceeded",
                }

            if daily_used + cost > daily_limit:
                return {
                    "allowed": False,
                    "tier": tier,
                    "daily_limit": daily_limit,
                    "daily_remaining": daily_remaining,
                    "rpm_limit": rpm_limit,
                    "rpm_remaining": rpm_remaining,
                    "reason": "daily_quota_exceeded",
                }

            window.append(now)
            tenant["daily_used"] = daily_used + cost
            self._persist_state()

            return {
                "allowed": True,
                "tier": tier,
                "daily_limit": daily_limit,
                "daily_remaining": max(0, daily_limit - tenant["daily_used"]),
                "rpm_limit": rpm_limit,
                "rpm_remaining": max(0, rpm_limit - len(window)),
                "reason": "ok",
            }


saas_tenant_manager = SaaSTenantManager()
