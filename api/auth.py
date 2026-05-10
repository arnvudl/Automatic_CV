"""
auth.py — Utilitaires JWT : hachage, création/vérification de token, dépendance FastAPI.
"""

import os
from datetime import datetime, timedelta
from typing import Optional

import bcrypt
import jwt
from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

SECRET_KEY  = os.getenv("JWT_SECRET", "luminary-dev-secret-CHANGE-IN-PROD")
ALGORITHM   = "HS256"
EXPIRE_HOURS = int(os.getenv("JWT_EXPIRE_HOURS", "24"))

_bearer = HTTPBearer()


def hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))


def create_token(data: dict, expire_hours: int = EXPIRE_HOURS) -> str:
    payload = {**data, "exp": datetime.utcnow() + timedelta(hours=expire_hours)}
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> dict:
    return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer),
) -> dict:
    """Dépendance FastAPI — valide le Bearer token et retourne le payload."""
    try:
        return decode_token(credentials.credentials)
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expiré")
    except Exception:
        raise HTTPException(status_code=401, detail="Token invalide")


# Version optionnelle (routes publiques qui veulent juste savoir si c'est auth)
async def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(
        HTTPBearer(auto_error=False)
    ),
) -> Optional[dict]:
    if not credentials:
        return None
    try:
        return decode_token(credentials.credentials)
    except Exception:
        return None
