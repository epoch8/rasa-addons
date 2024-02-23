from typing import Dict, Text, Any

from sanic.request import Request

from channels.chatwoot.base import IMessageMetaData


class MessageMetaData(IMessageMetaData):
    __slots__ = ("_request", "_kwargs")

    def __init__(self, request: Request, kwargs=None):
        self._request = request
        self._kwargs = kwargs

    @classmethod
    def simple_metadata(cls, request: Request):
        return cls(request)

    @classmethod
    def custom_metadata(cls, request: Request, **kwargs):
        return cls(request, kwargs)

    @property
    def metadata(self) -> Dict[Text, Any]:
        data = self._request.json
        if self._kwargs is None:
            self._kwargs = {}
        try:
            telegram_username = self._kwargs.get("telegram_username",
                                                 data["conversation"]["meta"]["sender"]["additional_attributes"].get(
                                                     "username"))
        except KeyError:
            telegram_username = None

        result = {
            "contact_id": self._kwargs.get("contact_id", data.get("sender", {}).get("id")),
            "contact_name": self._kwargs.get("contact_name", data.get("sender", {}).get("name")),
            "account_id": self._kwargs.get("account_id", data.get("account", {}).get("id")),
            "inbox_id": self._kwargs.get("inbox_id", data.get("inbox", {}).get("id")),
            "conversation_id": self._kwargs.get("conversation_id", data.get("conversation", {}).get("id")),
            "telegram_username": telegram_username,
            "messages": self._kwargs.get("messages", data.get("conversation", {}).get("messages"))
        }
        return result
