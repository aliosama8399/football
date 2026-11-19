import logging
from datetime import datetime, timedelta, timezone
from typing import Optional
from jose import JWTError, jwt
import bcrypt
from sqlalchemy.future import select
from sqlalchemy import func
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordBearer

from api.config import settings
from api.database import User
from api.services.saas_tenant_service import saas_tenant_manager

logger = logging.getLogger(__name__)

# OAuth2 Password Bearer Scheme pointing to our login route
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/v1/auth/login", auto_error=False)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify plain password against its hash (truncated to 72 bytes for bcrypt)."""
    try:
        plain_bytes = plain_password.encode("utf-8")[:72]
        hash_bytes = hashed_password.encode("utf-8")
        return bcrypt.checkpw(plain_bytes, hash_bytes)
    except Exception as e:
        logger.error(f"Password verification failed: {e}")
        return False

def get_password_hash(password: str) -> str:
    """Generate a hashed password (truncated to 72 bytes for bcrypt)."""
    plain_bytes = password.encode("utf-8")[:72]
    salt = bcrypt.gensalt()
    hashed_bytes = bcrypt.hashpw(plain_bytes, salt)
    return hashed_bytes.decode("utf-8")

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT access token with optional expiry."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=settings.access_token_expire_minutes)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)
    return encoded_jwt

def decode_access_token(token: str) -> Optional[dict]:
    """Decode and validate a JWT access token."""
    try:
        payload = jwt.decode(token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm])
        return payload
    except JWTError:
        return None

# ── FastAPI Route Guard Dependencies ──────────────────────────────────────────

async def get_current_user(
    request: Request = None,
    token: Optional[str] = Depends(oauth2_scheme),
) -> User:
    """
    FastAPI dependency authenticating via B2B `X-API-Key` or Bearer JWT,
    enforcing account activation and SaaS tenant rate/daily quotas.
    """
    from api.dependencies import get_db  # Import inside to avoid circular reference

    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    api_key_header = request.headers.get("X-API-Key", "") if request is not None else ""
    if api_key_header:
        key_meta = saas_tenant_manager.verify_api_key(api_key_header)
        if key_meta is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or revoked X-API-Key",
            )
        username = key_meta["username"]
        if request is not None and not getattr(request.state, "quota_checked", False):
            quota = saas_tenant_manager.check_and_consume_quota(username)
            request.state.quota_checked = True
            if not quota["allowed"]:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=f"SaaS quota exceeded ({quota['reason']})",
                )
        try:
            async for db in get_db():
                stmt = select(User).where(User.username == username)
                result = await db.execute(stmt)
                user = result.scalar_one_or_none()
                if user is not None:
                    return user
        except Exception:
            pass
        return User(
            id=0,
            username=username,
            email=f"{username}@api.tenant",
            role="user",
            is_active=True,
        )

    if token is None:
        raise credentials_exception

    payload = decode_access_token(token)
    if payload is None:
        raise credentials_exception

    username: Optional[str] = payload.get("sub")
    if username is None:
        raise credentials_exception

    async for db in get_db():
        stmt = select(User).where(User.username == username)
        result = await db.execute(stmt)
        user = result.scalar_one_or_none()

        if user is None:
            raise credentials_exception
        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Inactive user. Account must be activated first."
            )
        if request is None or not getattr(request.state, "quota_checked", False):
            quota = saas_tenant_manager.check_and_consume_quota(user.username, role=user.role)
            if request is not None:
                request.state.quota_checked = True
            if not quota["allowed"]:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=f"SaaS quota exceeded ({quota['reason']})",
                )
        return user

async def require_supervisor(
    current_user: User = Depends(get_current_user)
) -> User:
    """FastAPI dependency to restrict endpoints to supervisors only."""
    if current_user.role != "supervisor":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Permission denied. Supervisor role required."
        )
    return current_user

# ── Supervisor Seeder ──────────────────────────────────────────────────────────

async def seed_supervisor_user(db: AsyncSession) -> None:
    """
    Checks if database is empty. If so, seeds the default supervisor user.
    """
    stmt = select(func.count(User.id))
    result = await db.execute(stmt)
    count = result.scalar()
    
    if count == 0:
        logger.info("No users found in database. Seeding default supervisor user...")
        admin = User(
            username=settings.football_admin_username,
            email=settings.football_admin_email,
            hashed_password=get_password_hash(settings.football_admin_password),
            role="supervisor",
            is_active=True
        )
        db.add(admin)
        await db.commit()
        logger.info(f"Default supervisor user '{settings.football_admin_username}' successfully seeded.")
