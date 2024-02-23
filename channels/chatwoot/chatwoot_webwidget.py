import logging
from typing import Text, Optional

from rasa.core.channels.channel import UserMessage, OutputChannel

from channels.chatwoot.abstact import AChatwootUserMessage
from channels.chatwoot.base import IChatwootUserMessage

logger = logging.getLogger(__name__)


class ChatwootUserMessageWidget(AChatwootUserMessage, IChatwootUserMessage):
    def __get_intent_from_widget_button(self) -> Optional[Text]:
        content_attributes = self._request_json["content_attributes"]["submitted_values"][0]
        intent = content_attributes.get("value")
        return intent

    def __check_webwidget_trigger(self) -> bool:
        event_info = self._request_json.get("event_info")
        if event_info is None:
            return False
        return self._request_json.get("event") == "webwidget_triggered" and self._request_json.get(
            "current_conversation") is None and not event_info.get("pre_chat_form_enabled")

    def __check_start_message(self) -> bool:
        conversation = self._request_json.get("conversation")
        event = self._request_json.get("event")
        if conversation is None:
            return False
        return conversation.get("agent_last_seen_at") == 0 and event != "message_created"

    def __check_button_message(self) -> bool:
        return self._request_json["event"] == "message_updated" and self._request_json.get("content_attributes")

    async def get_user_message(self, output_channel: OutputChannel, input_channel: Text) -> Optional[UserMessage]:
        if self.__check_webwidget_trigger():
            logger.info("Webwidget Trigger")
            text = self._request_json["event_info"].get("start_message")
            if text is None:
                text = "/get_started"
            inbox_id = self._request_json["inbox"]["id"]
            contact_id = self._request_json["contact"]["id"]
            source_id = self._request_json["source_id"]
            sender_id = await self._api.create_conversation(inbox_id, contact_id, source_id)
            metadata = self._message_metadata.custom_metadata(
                self._request,
                contact_id=contact_id,
                account_id=contact_id,
                conversation_id=sender_id,
            ).metadata
        elif self.__check_button_message():
            logger.info("Button Trigger")
            text = self.__get_intent_from_widget_button()
            metadata = self._message_metadata.simple_metadata(self._request).metadata
            sender_id = metadata.get("conversation_id")
        elif self._check_empty_message():
            logger.info("Empty Message")
            return None
        else:
            logger.info("simple content text")
            text = self._request_json.get("content")
            metadata = self._message_metadata.simple_metadata(self._request).metadata
            sender_id = str(metadata.get("conversation_id"))

        text = self._get_intent_by_attach_or_text(metadata.get("messages"), text)

        logger.info(f"Got message: {text}")

        if not text:
            return None
        if text[0] != "/":
            content = await self._translate_message(text)
            metadata["source_message"] = text
        else:
            content = text

        return UserMessage(content, output_channel, sender_id, input_channel=input_channel, metadata=metadata)
