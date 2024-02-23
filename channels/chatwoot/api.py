import logging
import threading
from typing import Text

import aiohttp

from .base import IChatwootAPI

logger = logging.getLogger(__name__)

ctx = threading.local()


def get_session() -> aiohttp.ClientSession:
    if not hasattr(ctx, "http_session"):
        ctx.http_session = aiohttp.ClientSession()
    return ctx.http_session


class ChatwootAPI(IChatwootAPI):
    __slots__ = ("_api_access_token", "_account_id", "_url")

    def __init__(self, api_access_token: Text, account_id: str, url: Text):
        self._api_access_token = api_access_token
        self._account_id = account_id
        self._url = url

    async def create_conversation(self, inbox_id, contact_id, source_id) -> int:
        url = f"{self._url}/api/v1/accounts/{self._account_id}/conversations"
        body = {
            "inbox_id": f"{inbox_id}",
            "contact_id": f"{contact_id}",
            "source_id": f"{source_id}"
        }
        async with get_session().post(url, json=body, headers={"api_access_token": self._api_access_token}) as resp:
            if resp.status != 200:
                logger.error(
                    f"Response from Chatwoot is not 200:\n"
                    f"url: {url},\n"
                    f"code: {resp.status},\n"
                    f"body: {await resp.text()}"
                )
            resp_json = await resp.json()
        conversation_id = resp_json.get("id")
        return conversation_id
