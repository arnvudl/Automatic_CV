"""
routers/auth.py — Authentification JWT : login, me, gestion utilisateurs.
"""

import uuid
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from api.auth import hash_password, verify_password, create_token, get_current_user
from api.database import get_db, User as UserModel

router = APIRouter(prefix="/auth", tags=["auth"])
logger = logging.getLogger("cv_api")


# ── Pydantic ──────────────────────────────────────────────────────────
class LoginRequest(BaseModel):
    email:    str
    password: str

class UserCreate(BaseModel):
    email:    str
    name:     str
    password: str
    role:     str = "recruiter"  # admin | recruiter

class UserUpdate(BaseModel):
    name:     Optional[str] = None
    role:     Optional[str] = None
    password: Optional[str] = None
    is_active: Optional[bool] = None


def _user_to_dict(u: UserModel) -> dict:
    return {
        "user_id":    u.user_id,
        "email":      u.email,
        "name":       u.name,
        "role":       u.role,
        "is_active":  u.is_active,
        "created_at": u.created_at.isoformat() if u.created_at else None,
    }


# ── POST /auth/login ──────────────────────────────────────────────────
@router.post("/login")
def login(body: LoginRequest):
    """Retourne un JWT si les credentials sont valides."""
    with get_db() as db:
        user = db.query(UserModel).filter(
            UserModel.email == body.email.lower().strip(),
            UserModel.is_active == True,
        ).first()
        if not user or not verify_password(body.password, user.password_hash):
            raise HTTPException(401, "Email ou mot de passe incorrect")

    token = create_token({
        "sub":   user.user_id,
        "email": user.email,
        "name":  user.name,
        "role":  user.role,
    })
    return {
        "access_token": token,
        "token_type":   "bearer",
        "user": _user_to_dict(user),
    }


# ── GET /auth/me ──────────────────────────────────────────────────────
@router.get("/me")
def me(current_user: dict = Depends(get_current_user)):
    """Retourne l'utilisateur courant à partir du token."""
    return current_user


# ── GET /auth/users ───────────────────────────────────────────────────
@router.get("/users")
def list_users(current_user: dict = Depends(get_current_user)):
    """Liste tous les utilisateurs (admin uniquement)."""
    if current_user.get("role") != "admin":
        raise HTTPException(403, "Réservé aux administrateurs")
    with get_db() as db:
        users = db.query(UserModel).all()
        return [_user_to_dict(u) for u in users]


# ── POST /auth/users ──────────────────────────────────────────────────
@router.post("/users", status_code=201)
def create_user(body: UserCreate, current_user: dict = Depends(get_current_user)):
    """Crée un nouvel utilisateur (admin uniquement)."""
    if current_user.get("role") != "admin":
        raise HTTPException(403, "Réservé aux administrateurs")
    with get_db() as db:
        existing = db.query(UserModel).filter(UserModel.email == body.email.lower().strip()).first()
        if existing:
            raise HTTPException(409, "Email déjà utilisé")
        user = UserModel(
            user_id=uuid.uuid4().hex,
            email=body.email.lower().strip(),
            name=body.name,
            password_hash=hash_password(body.password),
            role=body.role,
        )
        db.add(user)
        db.flush()
        return _user_to_dict(user)


# ── PATCH /auth/users/{user_id} ───────────────────────────────────────
@router.patch("/users/{user_id}")
def update_user(
    user_id: str,
    body: UserUpdate,
    current_user: dict = Depends(get_current_user),
):
    """Met à jour un utilisateur (admin ou soi-même)."""
    is_admin = current_user.get("role") == "admin"
    is_self  = current_user.get("sub") == user_id
    if not is_admin and not is_self:
        raise HTTPException(403, "Accès refusé")
    with get_db() as db:
        user = db.get(UserModel, user_id)
        if not user:
            raise HTTPException(404, "Utilisateur introuvable")
        if body.name     is not None: user.name      = body.name
        if body.password is not None: user.password_hash = hash_password(body.password)
        if is_admin:
            if body.role      is not None: user.role      = body.role
            if body.is_active is not None: user.is_active = body.is_active
        db.flush()
        return _user_to_dict(user)


# ── DELETE /auth/users/{user_id} ──────────────────────────────────────
@router.delete("/users/{user_id}")
def delete_user(user_id: str, current_user: dict = Depends(get_current_user)):
    """Désactive un utilisateur (admin uniquement, pas de suppression réelle)."""
    if current_user.get("role") != "admin":
        raise HTTPException(403, "Réservé aux administrateurs")
    if current_user.get("sub") == user_id:
        raise HTTPException(400, "Impossible de se désactiver soi-même")
    with get_db() as db:
        user = db.get(UserModel, user_id)
        if not user:
            raise HTTPException(404, "Utilisateur introuvable")
        user.is_active = False
        db.flush()
    return {"deactivated": True, "user_id": user_id}
