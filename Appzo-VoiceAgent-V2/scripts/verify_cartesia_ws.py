"""Verify only the Cartesia WebSocket handshake used by the V2 media path."""

import asyncio
import argparse
import os
import socket
import time
from urllib.parse import urlencode

from dotenv import load_dotenv
from websockets.asyncio.client import connect


async def verify(env_file: str | None = None) -> int:
    load_dotenv(env_file, override=True)
    api_key = os.getenv("CARTESIA_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("CARTESIA_API_KEY is required")
    # Cartesia's current WebSocket API authenticates the handshake with this
    # header. Keep credentials out of both the URI and all log output.
    url = "wss://api.cartesia.ai/tts/websocket?" + urlencode(
        {"cartesia_version": "2026-03-01"}
    )
    started = time.perf_counter()
    try:
        async with connect(
            url,
            additional_headers={"X-API-Key": api_key},
            # This is a server-side connection. Avoid ambient OS/proxy
            # discovery, which can hang a direct WSS upgrade on macOS even
            # when direct HTTPS and the authenticated upgrade both work.
            proxy=None,
            # The current host advertises IPv6 but cannot complete this
            # CloudFront WebSocket route over IPv6. Pinning the server-side
            # provider connection to IPv4 avoids the stalled first address.
            family=socket.AF_INET,
            open_timeout=5,
            max_size=None,
        ):
            elapsed_ms = round((time.perf_counter() - started) * 1000)
    except Exception as exc:
        print(
            f"Cartesia WebSocket handshake failed ({type(exc).__name__}). "
            "V2 auto mode will use its audible HTTP fallback for this call."
        )
        return 1
    print(f"Cartesia WebSocket handshake succeeded in {elapsed_ms} ms")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--env-file",
        help="Optional dotenv file to test. Credentials and the WebSocket URL are never printed.",
    )
    args = parser.parse_args()
    raise SystemExit(asyncio.run(verify(args.env_file)))
