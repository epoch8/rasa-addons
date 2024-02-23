from enum import Enum


class ChannelEnums(Enum):
    WEB_WIDGET = "Channel::WebWidget"
    TELEGRAM = "Channel::Telegram"


class ChannelEventEnums(Enum):
    WEB_WIDGET_TRIGGERED = "webwidget_triggered"


class AttachmentsEnum(Enum):
    AUDIO_MESSAGE = 0
    ANOTHER_MEDIA_CONTENT = 1
    MEDIA_IS_NONE = 2

