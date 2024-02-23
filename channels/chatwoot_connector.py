import asyncio
import logging
import threading
from traceback import format_exc
from typing import List, Text, Dict, Any, Callable, Awaitable, Optional

import aiohttp
from rasa.core.channels.channel import InputChannel, OutputChannel, UserMessage
from sanic import Blueprint, response
from sanic.request import Request
from sanic.response import HTTPResponse

from apis.settings import translation_service_settings
from apis.translation_api import TranslateAPI
from channels.chatwoot.api import ChatwootAPI
from channels.chatwoot.chatwoot_channel_factory import ChatwootChannelFactory

logger = logging.getLogger(__name__)

ctx = threading.local()


def get_session() -> aiohttp.ClientSession:
    if not hasattr(ctx, "http_session"):
        ctx.http_session = aiohttp.ClientSession(raise_for_status=True)
    return ctx.http_session


class ChatwootOutput(OutputChannel):
    def __init__(
            self,
            api_access_token: str,
            url: str,
            account_id: str,
            buttons_type: Optional[str],
    ) -> None:
        self.api_access_token = api_access_token
        self.url = url
        self.account_id = account_id
        self.buttons_type = buttons_type

    @classmethod
    def name(cls) -> Text:
        return "chatwoot"

    async def send_text_message(
            self, recipient_id: Text, text: Text, **kwargs: Any
    ) -> None:
        """https://www.chatwoot.com/developers/api/#operation/create-a-new-message-in-a-conversation"""

        url = (
            f"{self.url}/api/v1/accounts/{self.account_id}/conversations/{recipient_id}/messages"
        )

        body = {"content": text, "message_type": "outgoing", "private": False}

        buttons = kwargs.get("buttons")
        if buttons:
            items = []
            for button in buttons:
                title = button.get("title", "")
                items.append({"title": title, "value": button.get("payload", title)})

            body["content_type"] = "input_select"
            body["content_attributes"] = {"items": items, "buttons_layout": self.buttons_type}

        logger.info(f"Sending message: {body}")
        async with get_session().post(url, json=body, headers={"api_access_token": self.api_access_token}) as resp:
            if resp.status != 200:
                logger.info(
                    f"Response from Chatwoot is not 200:\n"
                    f"url: {url},\n"
                    f"code: {resp.status},\n"
                    f"body: {await resp.text()}"
                )

    async def send_text_with_buttons(
            self,
            recipient_id: Text,
            text: Text,
            buttons: List[Dict[Text, Any]],
            **kwargs: Any,
    ) -> None:
        await self.send_text_message(recipient_id, text, buttons=buttons)


class ChatwootInput(InputChannel):
    _chatwoot_channel_factory = ChatwootChannelFactory()

    def __init__(
            self,
            api_access_token: str,
            url: str,
            account_id: str,
            buttons_type: Optional[str],
    ) -> None:
        self.api_access_token = api_access_token
        self.url = url
        self.account_id = account_id
        self.buttons_type = buttons_type
        self._chatwoot_api = ChatwootAPI(api_access_token, account_id, url)
        self._translate_api = TranslateAPI(translation_service_settings.URL, translation_service_settings.TOKEN)
        self._background_task = set()

    @classmethod
    def name(cls) -> Text:
        return "chatwoot"

    @classmethod
    def from_credentials(cls, credentials: Dict[str, str]) -> InputChannel:
        if not credentials:
            cls.raise_missing_credentials_exception()

        return cls(
            credentials.get("api_access_token"),
            credentials.get("url"),
            credentials.get("account_id"),
            credentials.get("buttons_type"),
        )

    def _logger_task_status(self, task):
        task_name = task.get_name()
        try:
            task.result()
            logger.info(f"{task_name} is finished: {task.done()}")
        except asyncio.CancelledError as e:
            logger.exception(e)
            logger.error(f"{task_name} Canceled Error!")
        except asyncio.InvalidStateError as e:
            logger.error(f"{task_name} Invalid State Error!")
        return self._background_task.discard

    def blueprint(
            self, on_new_message: Callable[[UserMessage], Awaitable[Any]]
    ) -> Blueprint:
        custom_webhook = Blueprint("custom_webhook_chatwoot", __name__)

        # noinspection PyUnusedLocal
        @custom_webhook.route("/", methods=["GET"])
        async def health(request: Request) -> HTTPResponse:
            return response.json({"status": "ok"})

        @custom_webhook.route("/webhook", methods=["POST"])
        async def receive(request: Request) -> HTTPResponse:
            req_json = request.json

            # TODO remove
            logger.info(f"Data: {req_json}")
            chatwoot_channel = self._chatwoot_channel_factory(request, self._chatwoot_api, self._translate_api)
            if chatwoot_channel is None:
                return response.text("Ok")
            input_channel = self.name()
            user_message = await chatwoot_channel.get_user_message(self.get_output_channel(), input_channel)
            if user_message is None:
                return response.text("Ok")
            try:
                task = asyncio.create_task(on_new_message(user_message))
                self._background_task.add(task)
                task.add_done_callback(self._logger_task_status(task))
            except Exception:
                logger.exception(
                    f"An exception occured while handling user message:\n{format_exc()}"
                )
            return response.text("success")

        return custom_webhook

    def get_output_channel(self) -> OutputChannel:
        return ChatwootOutput(
            self.api_access_token,
            self.url,
            self.account_id,
            self.buttons_type,
        )
