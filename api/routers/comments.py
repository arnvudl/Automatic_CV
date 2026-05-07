"""
routers/comments.py — Commentaires RH sur les candidats.
"""

import json
import uuid
from datetime import datetime

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from api.auth import get_current_user
from api.config import COMMENTS_FILE

router = APIRouter(tags=["comments"], dependencies=[Depends(get_current_user)])


def _load() -> dict:
    if not COMMENTS_FILE.exists():
        return {}
    return json.loads(COMMENTS_FILE.read_text(encoding="utf-8"))


def _save(data: dict):
    COMMENTS_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


class CommentCreate(BaseModel):
    author: str
    text:   str

class CommentUpdate(BaseModel):
    text: str


@router.get("/comments/{candidate_id}")
def get_comments(candidate_id: str):
    return _load().get(candidate_id, [])


@router.post("/comments/{candidate_id}")
def add_comment(candidate_id: str, body: CommentCreate):
    data   = _load()
    thread = data.get(candidate_id, [])
    comment = {
        "id":         uuid.uuid4().hex,
        "author":     body.author,
        "text":       body.text,
        "created_at": datetime.now().isoformat(),
        "updated_at": None,
    }
    thread.append(comment)
    data[candidate_id] = thread
    _save(data)
    return comment


@router.patch("/comments/{candidate_id}/{comment_id}")
def update_comment(candidate_id: str, comment_id: str, body: CommentUpdate):
    data   = _load()
    thread = data.get(candidate_id, [])
    for c in thread:
        if c["id"] == comment_id:
            c["text"]       = body.text
            c["updated_at"] = datetime.now().isoformat()
            data[candidate_id] = thread
            _save(data)
            return c
    raise HTTPException(404, "Commentaire introuvable.")


@router.delete("/comments/{candidate_id}/{comment_id}")
def delete_comment(candidate_id: str, comment_id: str):
    data = _load()
    data[candidate_id] = [c for c in data.get(candidate_id, []) if c["id"] != comment_id]
    _save(data)
    return {"deleted": comment_id}
