"""Unit tests for authentication primitives, JWT lifecycle, and constant-time internal auth guards."""

from __future__ import annotations

import asyncio
from datetime import timedelta
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from api.auth import (
    create_access_token,
    decode_access_token,
    get_password_hash,
    verify_password,
)
from api.routes.jobs import _verify_supervisor_or_internal


def test_password_hash_and_verify_roundtrip() -> None:
    """Verify bcrypt hashing accepts valid passwords and rejects wrong passwords."""
    raw_secret = "Enterprise_SaaS_Password_2026!"
    hashed = get_password_hash(raw_secret)

    assert hashed != raw_secret
    assert verify_password(raw_secret, hashed) is True
    assert verify_password("Wrong_Password_123!", hashed) is False


def test_jwt_access_token_lifecycle() -> None:
    """Verify JWT signing and decoding preserve claims and reject tampered signatures."""
    token = create_access_token({"sub": "alice_supervisor", "role": "supervisor"})
    payload = decode_access_token(token)

    assert payload is not None
    assert payload["sub"] == "alice_supervisor"
    assert payload["role"] == "supervisor"

    tampered = f"{token[:-4]}xxxx"
    assert decode_access_token(tampered) is None


def test_expired_jwt_rejected() -> None:
    """Verify expired JWT tokens fail validation and decode to None."""
    expired_token = create_access_token(
        {"sub": "expired_user", "role": "user"},
        expires_delta=timedelta(seconds=-10),
    )
    assert decode_access_token(expired_token) is None


def test_verify_supervisor_or_internal_constant_time(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify constant-time internal token, supervisor JWT, and rejection of unauthenticated callers."""
    monkeypatch.setenv("INTERNAL_ADMIN_TOKEN", "prod_internal_secret_999")

    valid_internal_req = SimpleNamespace(
        headers={"X-Internal-Token": "prod_internal_secret_999"},
        client=SimpleNamespace(host="10.0.1.42"),
    )
    asyncio.run(_verify_supervisor_or_internal(valid_internal_req))

    sup_jwt = create_access_token({"sub": "admin", "role": "supervisor"})
    valid_jwt_req = SimpleNamespace(
        headers={"Authorization": f"Bearer {sup_jwt}"},
        client=SimpleNamespace(host="10.0.1.42"),
    )
    asyncio.run(_verify_supervisor_or_internal(valid_jwt_req))

    bad_req = SimpleNamespace(
        headers={"X-Internal-Token": "wrong_secret"},
        client=SimpleNamespace(host="203.0.113.10"),
    )
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(_verify_supervisor_or_internal(bad_req))
    assert exc_info.value.status_code == 401

