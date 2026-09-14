import asyncio
from collections.abc import Awaitable, Callable


class ClientPool:
    """Process-wide keyed clients; sessions must never close a borrowed client."""
    def __init__(self) -> None: self._clients: dict[str, object] = {}; self._lock = asyncio.Lock()
    async def get(self, key: str, factory: Callable[[], Awaitable[object]]) -> object:
        async with self._lock:
            if key not in self._clients: self._clients[key] = await factory()
            return self._clients[key]
    async def close(self) -> None:
        for client in self._clients.values():
            close = getattr(client, "close", None)
            if close: result = close(); await result if hasattr(result, "__await__") else None
        self._clients.clear()
