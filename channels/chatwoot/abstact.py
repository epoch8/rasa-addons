import logging
import abc
from typing import Optional, List, Text, Dict, Any

from sanic.request import Request

from apis.settings import translation_service_settings
from apis.translation_api import ITranslateAPI
from channels.chatwoot.base import IChatwootAPI
from channels.chatwoot.enums import AttachmentsEnum
from channels.chatwoot.message_metadata import MessageMetaData
from channels.settings.chatwoot import ChatwootChannelsBaseSettings, chatwoot_channels_base_settings


logger = logging.getLogger(__name__)


class AChatwootUserMessage(abc.ABC):
    _message_metadata = MessageMetaData
    _settings: ChatwootChannelsBaseSettings = chatwoot_channels_base_settings

    __slots__ = ("_request", "_request_json", "_api", "_translation_service_api")

    def __init__(self, request: Request, api: IChatwootAPI, translation_service_api: ITranslateAPI):
        self._request = request
        self._request_json = request.json
        self._api = api
        self._translation_service_api = translation_service_api

    @staticmethod
    def _check_attachments(messages: Optional[List[Dict[Text, Any]]]) -> AttachmentsEnum:
        if messages is None:
            return AttachmentsEnum.MEDIA_IS_NONE
        for message in messages:
            attachments = message.get("attachments")
            if attachments is None:
                return AttachmentsEnum.MEDIA_IS_NONE
            for attachment in attachments:
                file_type = attachment.get("file_type")
                if file_type is None:
                    continue
                if file_type == "audio":
                    return AttachmentsEnum.AUDIO_MESSAGE
            return AttachmentsEnum.ANOTHER_MEDIA_CONTENT
        return AttachmentsEnum.MEDIA_IS_NONE

    def _get_intent_by_attach_or_text(self, messages: Optional[List[Dict[Text, Any]]], text: str) -> Text:
        match self._check_attachments(messages):
            case AttachmentsEnum.AUDIO_MESSAGE:
                return self._settings.AUDIO_CONTENT_INTENT
            case AttachmentsEnum.ANOTHER_MEDIA_CONTENT:
                return self._settings.MEDIA_CONTENT_INTENT
            case _:
                ...
        return text

    def _check_empty_message(self) -> bool:
        return self._request_json["event"] != "message_created" \
            or self._request_json["message_type"] == "outgoing" \
            or self._request_json["conversation"]["status"] != "pending"

    async def _translate_message(self, text: str) -> str:
        sender = self._request_json.get("sender")
        if sender is None:
            return text
        custom_attributes = sender.get("custom_attributes")
        if custom_attributes is None:
            return text
        target_language = translation_service_settings.DEFAULT_LANGUAGE
        source_language = custom_attributes.get("language", translation_service_settings.DEFAULT_LANGUAGE)
        if source_language is None or source_language == target_language:
            logger.warning(f"{source_language=}!")
            return text
        data = [{"text": text}]
        result = await self._translation_service_api.google_translate(source_language, target_language, data)
        answer_text = result.get("text")
        if not answer_text:
            return text
        return answer_text

