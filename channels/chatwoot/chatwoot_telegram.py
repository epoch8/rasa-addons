import logging
from typing import Text, Optional

from rasa.core.channels.channel import UserMessage, OutputChannel

from channels.chatwoot.abstact import AChatwootUserMessage
from channels.chatwoot.base import IChatwootUserMessage


logger = logging.getLogger(__name__)


class ChatwootUserMessageTelegram(AChatwootUserMessage, IChatwootUserMessage):

    async def get_user_message(self, output_channel: OutputChannel, input_channel: Text) -> Optional[UserMessage]:
        if self._check_empty_message():
            return None
        metadata = self._message_metadata.simple_metadata(self._request).metadata
        text = self._request_json["content"]
        sender_id = metadata["conversation_id"]
        text = self._get_intent_by_attach_or_text(metadata.get("messages"), text)
        logger.info(f"Got message: {text}")

        if not text:
            return None

        return UserMessage(text, output_channel, sender_id, input_channel=input_channel, metadata=metadata)
