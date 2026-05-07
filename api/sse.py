"""
sse.py — État global SSE et fonction de broadcast.
Importé par main.py (endpoint /events) et par les routers qui émettent des événements.
"""

import asyncio
from typing import Set

_sse_clients: Set[asyncio.Queue] = set()


async def broadcast(event: dict):
    """Envoie un événement SSE à tous les clients connectés."""
    dead = set()
    for q in _sse_clients:
        try:
            q.put_nowait(event)
        except asyncio.QueueFull:
            dead.add(q)
    _sse_clients.difference_update(dead)
