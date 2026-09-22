#!/usr/bin/env python3
"""Start a GoodBox agent test call without exposing browser credentials."""

import argparse
import asyncio
import os
from urllib.parse import urlencode

import httpx
from dotenv import load_dotenv


def required(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value or value.startswith("<"):
        raise SystemExit(f"{name} must be set in .env")
    return value


async def dial_via_goodbox(destination: str) -> None:
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


async def dial_direct_to_local(destination: str) -> None:
    """Place a Plivo call whose answer URL is this local public server.

    The Goodbox test-campaign endpoint owns its answer URL and may route to a
    remote voice worker. Direct mode is deliberately explicit and does not
    mutate the Plivo application or phone configuration.
    """
    public_base_url = required("PUBLIC_BASE_URL")
    phone_id = required("GOODBOX_PHONE_ID")
    chatbot_id = required("GOODBOX_CHATBOT_ID")
    auth_id = required("PLIVO_AUTH_ID")
    auth_token = required("PLIVO_AUTH_TOKEN")

    # Prefer an explicit source number. If omitted, read the non-secret number
    # metadata from Goodbox; provider credentials remain local-only env vars.
    source_number = os.getenv("PLIVO_SOURCE_NUMBER", "").strip()
    async with httpx.AsyncClient(timeout=30) as client:
        if not source_number:
            base_url = required("GOODBOX_VOICE_CAMPAIGN_TEST_URL").split("/v1/", 1)[0]
            headers = {
                "Authorization": f"Bearer {required('GOODBOX_API_TOKEN')}",
                "X-Org-Code": required("GOODBOX_ORG_CODE"),
            }
            phone_response = await client.get(
                f"{base_url}/v1/phones/{phone_id}", headers=headers
            )
            phone_response.raise_for_status()
            phone_body = phone_response.json()
            phone_data = phone_body.get("data", phone_body)
            source_number = str(phone_data.get("destination_number") or "").strip()
        if not source_number:
            raise SystemExit("Set PLIVO_SOURCE_NUMBER to the Plivo caller ID.")

        query = urlencode({"chatbot_id": chatbot_id})
        answer_url = (
            f"{public_base_url.rstrip('/')}/v1/plivo/callback/{phone_id}?{query}"
        )
        response = await client.post(
            f"https://api.plivo.com/v1/Account/{auth_id}/Call/",
            auth=httpx.BasicAuth(auth_id, auth_token),
            json={
                "from": source_number,
                "to": destination,
                "answer_url": answer_url,
                "answer_method": "POST",
            },
        )
        response.raise_for_status()
        body = response.json()
    request_uuid = body.get("request_uuid") or body.get("api_id") or "accepted"
    print(f"Plivo accepted local test call ({request_uuid}).")
    print(f"Answer URL: {answer_url}")
    print("When answered, /health plivo_callback_count and plivo_media_count must increase.")


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Place a GoodBox agent test call.")
    parser.add_argument("--to", default=os.getenv("GOODBOX_TEST_DESTINATION", ""))
    parser.add_argument(
        "--route",
        choices=("goodbox", "local"),
        default=os.getenv("GOODBOX_TEST_ROUTE", "goodbox"),
        help="Use Goodbox's remote call controller or force this local PUBLIC_BASE_URL.",
    )
    args = parser.parse_args()
    destination = args.to.strip()
    if not destination.startswith("+"):
        raise SystemExit("Use an E.164 destination, for example +918274828890")
    if args.route == "local":
        asyncio.run(dial_direct_to_local(destination))
    else:
        asyncio.run(dial_via_goodbox(destination))


if __name__ == "__main__":
    main()
