from pydantic import BaseSettings, Field


class ChatwootChannelsBaseSettings(BaseSettings):
    MEDIA_CONTENT_INTENT: str = Field("/media_content", env="MEDIA_CONTENT_INTENT")
    AUDIO_CONTENT_INTENT: str = Field("/audio_content", env="AUDIO_CONTENT_INTENT")


chatwoot_channels_base_settings = ChatwootChannelsBaseSettings()
