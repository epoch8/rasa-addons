import abc
import logging
import threading
from typing import Any

import aiohttp

logger = logging.getLogger(__name__)

ctx = threading.local()


def get_session() -> aiohttp.ClientSession:
    if not hasattr(ctx, "http_session"):
        ctx.http_session = aiohttp.ClientSession(raise_for_status=True)
    return ctx.http_session


class ITranslateAPI(abc.ABC):
    @abc.abstractmethod
    async def google_translate(
            self,
            source_language: str,
            target_language: str,
            data: list[dict[str, Any]]
    ) -> dict[str, Any]:
        pass


class TranslateAPI(ITranslateAPI):
    __slots__ = ("_url", "_token")

    def __init__(self, url: str, token: str):
        self._url = url
        self._token = token

    async def google_translate(
            self,
            source_language: str,
            target_language: str,
            data: list[dict[str, Any]]
    ) -> dict[str, Any]:
        url = f"{self._url}/api/v1/translation/google/"
        header = {"x-api-key": self._token}
        body = {
            "data": data,
            "source": source_language,
            "target": target_language,
        }
        async with get_session().post(url, json=body, headers=header) as resp:
            resp_json = await resp.json()
            logger.info(resp_json)
        return resp_json
