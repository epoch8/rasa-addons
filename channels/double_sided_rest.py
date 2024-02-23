import asyncio
import logging
import threading
from http import HTTPStatus
from traceback import format_exc
from typing import List
from typing import Text, Dict, Callable, Awaitable, Any

import aiohttp
from rasa.core.channels.channel import InputChannel, OutputChannel, UserMessage
from sanic import Blueprint, response
from sanic.request import Request
from sanic.response import HTTPResponse

ctx = threading.local()

logger = logging.getLogger(__name__)


def get_session() -> aiohttp.ClientSession:
    if not hasattr(ctx, "http_session"):
        ctx.http_session = aiohttp.ClientSession(raise_for_status=True)
    return ctx.http_session


class DoubleSidedRestOutput(OutputChannel):

    def __init__(
            self,
            url: str,
            token: str,
            system_user_id: str
    ):
        self._url = url
        self._token = token
        self._system_user_id = system_user_id

    @classmethod
    def name(cls) -> Text:
        return "double_sided_rest"

    async def send_text_message(
            self,
            recipient_id: Text,
            text: Text,
            **kwargs: Any,
    ) -> None:
        url = f"{self._url}/rest/{self._system_user_id}/{self._token}/rasa.add.message.json"
        body = {"text": text, "sender_id": recipient_id}
        buttons = kwargs.get("buttons")
        if buttons:
            items = []
            for button in buttons:
                title = button.get("title", "")
                items.append({"title": title, "value": button.get("payload", title)})
            body["buttons"] = items

        logger.info(f"Sending message: {body}")
        # TODO: discuss with Evraz where put the token
        async with get_session().post(url, json=body, ssl=False) as resp:
            if resp.status != 200:
                logger.error(
                    f"Response from API is not 200:\n"
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


class DoubleSidedRestInput(InputChannel):
    def __init__(self):
        self._background_task = set()

    @classmethod
    def name(cls) -> Text:
        return "double_sided_rest"

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

    @staticmethod
    def __error_message(status: HTTPStatus, detail_message: str | None = None):
        return response.json(
            status=status,
            body={"status": "Error", "code": status, "detail": detail_message})

    def blueprint(self, on_new_message: Callable[[UserMessage], Awaitable[Any]]) -> Blueprint:
        custom_webhook = Blueprint("custom_webhook_double_sided_rest", __name__)

        @custom_webhook.route("/", methods=["GET"])
        async def health(_: Request) -> HTTPResponse:
            return response.json({"status": "ok"})

        @custom_webhook.route("/webhook", methods=["POST"])
        async def receive(request: Request) -> HTTPResponse:
            req_json = request.json

            logger.info(f"Data: {req_json}")
            message = req_json.get("message")
            sender_id = req_json.get("sender")
            token = req_json.get("token")
            system_user_id = req_json.get("systemUserId")
            url = req_json.get("callback")
            if token is None:
                return self.__error_message(HTTPStatus.UNAUTHORIZED, "Unauthorized Error!")
            if message is None:
                return self.__error_message(HTTPStatus.UNPROCESSABLE_ENTITY, "Field message is None!")
            if sender_id is None:
                return self.__error_message(HTTPStatus.UNPROCESSABLE_ENTITY, "Field sender is None!")
            if system_user_id is None:
                return self.__error_message(HTTPStatus.UNPROCESSABLE_ENTITY, "Field systemUserId is None!")
            if url is None:
                return self.__error_message(HTTPStatus.UNPROCESSABLE_ENTITY, "Field url is None!")
            output_channel = self.get_output_channel(token, system_user_id, url)
            user_message = UserMessage(
                message,
                output_channel,
                sender_id,
                input_channel=self.name(),
                metadata={},
            )
            try:
                task = asyncio.create_task(on_new_message(user_message))
                self._background_task.add(task)
                task.add_done_callback(self._logger_task_status(task))
            except Exception:
                logger.exception(
                    f"An exception occured while handling user message:\n{format_exc()}"
                )
            return response.json(status=HTTPStatus.CREATED, body={"status": "ok"})

        return custom_webhook

    @staticmethod
    def get_output_channel(token: str, system_user_id: str, url: str) -> OutputChannel:
        return DoubleSidedRestOutput(url, token, system_user_id)
