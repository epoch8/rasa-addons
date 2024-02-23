import logging
from typing import Optional

from sanic.request import Request

from apis.translation_api import ITranslateAPI
from channels.chatwoot.base import IChatwootUserMessage, IChatwootAPI
from channels.chatwoot.chatwoot_telegram import ChatwootUserMessageTelegram
from channels.chatwoot.chatwoot_webwidget import ChatwootUserMessageWidget
from channels.chatwoot.enums import ChannelEnums, ChannelEventEnums

logger = logging.getLogger(__name__)


class ChatwootChannelFactory:
    def __call__(
            self,
            request: Request,
            chatwoot_api: IChatwootAPI,
            translate_api: ITranslateAPI
    ) -> Optional[IChatwootUserMessage]:
        data_json = request.json
        channel = None
        if data_json.get("event") == ChannelEventEnums.WEB_WIDGET_TRIGGERED.value:
            channel = ChannelEnums.WEB_WIDGET.value
        if channel is None:
            channel = data_json["conversation"]["channel"]

        if channel == ChannelEnums.WEB_WIDGET.value:
            logger.info(f"The {channel} channel is used!")
            return self._chatwoot_user_message_widget(request, chatwoot_api, translate_api)
        elif channel == ChannelEnums.TELEGRAM.value:
            logger.info(f"The {channel} channel us user!")
            return self._chatwoot_user_message_telegram(request, chatwoot_api, translate_api)
        logger.warning(f"This {channel} is not yet supported!")
        return None

    @staticmethod
    def _chatwoot_user_message_telegram(
            request: Request,
            chatwoot_api: IChatwootAPI,
            translate_api: ITranslateAPI,
    ) -> IChatwootUserMessage:
        return ChatwootUserMessageTelegram(request, chatwoot_api, translate_api)

    @staticmethod
    def _chatwoot_user_message_widget(
            request: Request,
            chatwoot_api: IChatwootAPI,
            translate_api: ITranslateAPI,
    ) -> IChatwootUserMessage:
        return ChatwootUserMessageWidget(request, chatwoot_api, translate_api)
