import abc
from typing import Text, Optional

from rasa.core.channels.channel import UserMessage, OutputChannel
from sanic.request import Request


class IMessageMetaData(abc.ABC):
    @classmethod
    @abc.abstractmethod
    def simple_metadata(cls, request: Request):
        pass

    @classmethod
    @abc.abstractmethod
    def custom_metadata(cls, request: Request, **kwargs):
        pass


class IChatwootUserMessage(abc.ABC):
    @abc.abstractmethod
    async def get_user_message(self, output_channel: OutputChannel, input_channel: Text) -> Optional[UserMessage]:
        pass


class IChatwootAPI(abc.ABC):
    @abc.abstractmethod
    async def create_conversation(self, inbox_id, contact_id, source_id) -> int:
        pass
