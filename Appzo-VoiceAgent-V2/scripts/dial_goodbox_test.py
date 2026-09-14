#!/usr/bin/env python3
"""Start a GoodBox agent test call without exposing browser credentials."""

import argparse
import asyncio
import os

import httpx
from dotenv import load_dotenv


def required(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value or value.startswith("<"):
        raise SystemExit(f"{name} must be set in .env")
    return value


async def dial(destination: str) -> None:
    url = required("GOODBOX_VOICE_CAMPAIGN_TEST_URL")
    token = required("GOODBOX_API_TOKEN")
    org_code = required("GOODBOX_ORG_CODE")
    payload = {
        "chatbot_id": required("GOODBOX_CHATBOT_ID"),
        "phone_id": required("GOODBOX_PHONE_ID"),
        "phone_number": destination,
        "custom_variables": {},
    }
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "X-Org-Code": org_code,
    }
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(url, json=payload, headers=headers)
        response.raise_for_status()
    print(f"GoodBox accepted test call to {destination} ({response.status_code}).")


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Place a GoodBox agent test call.")
    parser.add_argument("--to", default=os.getenv("GOODBOX_TEST_DESTINATION", ""))
    args = parser.parse_args()
    destination = args.to.strip()
    if not destination.startswith("+"):
        raise SystemExit("Use an E.164 destination, for example +918274828890")
    asyncio.run(dial(destination))


if __name__ == "__main__":
    main()
