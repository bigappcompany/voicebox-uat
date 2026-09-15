"""Cartesia WebSocket connection adapter for the pinned Pipecat release."""
import asyncio
import socket
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from loguru import logger
from pipecat.services.cartesia.tts import CartesiaTTSService


class ResilientCartesiaTTSService(CartesiaTTSService):
    """Use the documented header auth and IPv4-only server-side WSS path.

    The installed Pipecat service places the API key in the URI and lets
    ``websockets`` prefer IPv6. Cartesia documents ``X-API-Key`` for the
    handshake, and the current deployment's IPv6 CloudFront route stalls.
    Keep credentials out of the URI and pin this outbound provider client to
    the verified IPv4 route. V1 retains its stock service as rollback.
    """

    async def _websocket_connect(self, uri, **kwargs):
        parsed = urlsplit(uri)
        safe_query = urlencode(
            [(key, value) for key, value in parse_qsl(parsed.query, keep_blank_values=True) if key != "api_key"]
        )
        uri = urlunsplit((parsed.scheme, parsed.netloc, parsed.path, safe_query, parsed.fragment))
        headers = dict(kwargs.pop("additional_headers", {}) or {})
        headers["X-API-Key"] = self._api_key
        kwargs["additional_headers"] = headers
        kwargs.setdefault("proxy", None)
        kwargs.setdefault("family", socket.AF_INET)
        kwargs.setdefault("open_timeout", 2.5)
        for attempt in range(3):
            try:
                return await super()._websocket_connect(uri, **kwargs)
            except (TimeoutError, OSError):
                logger.warning("Cartesia connection attempt {} failed", attempt + 1)
                if attempt == 2:
                    raise
                await asyncio.sleep(.15 * (attempt + 1))
