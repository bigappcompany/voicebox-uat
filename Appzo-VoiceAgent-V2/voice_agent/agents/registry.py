import asyncio
from collections.abc import Awaitable, Callable

from .bundle import AgentBundle


class AgentBundleRegistry:
    def __init__(self, loader: Callable[[str], Awaitable[AgentBundle]] | None = None) -> None:
        self._bundles: dict[tuple[str, str], AgentBundle] = {}
        self._loader = loader
        self._lock = asyncio.Lock()

    async def put(self, bundle: AgentBundle) -> None:
        async with self._lock:
            self._bundles[(bundle.agent_id, bundle.version)] = bundle

    async def get(self, agent_id: str, version: str | None = None) -> AgentBundle:
        async with self._lock:
            matches = [b for (aid, ver), b in self._bundles.items() if aid == agent_id and (version is None or ver == version)]
        if matches:
            return sorted(matches, key=lambda item: item.version)[-1]
        if self._loader is None:
            raise KeyError(f"No compiled bundle for {agent_id!r} version={version!r}")
        bundle = await self._loader(agent_id)
        await self.put(bundle)
        return bundle

    async def refresh(self, agent_id: str) -> None:
        if self._loader is None:
            return
        await self.put(await self._loader(agent_id))
