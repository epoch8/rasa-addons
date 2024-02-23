from pydantic import BaseSettings, Field


class TranslationServiceSettings(BaseSettings):
    URL: str = Field("", env="TRANSLATION_SERVICE_URL")
    TOKEN: str = Field("", env="TRANSLATION_SERVICE_TOKEN")
    DEFAULT_LANGUAGE: str = Field("ru", env="TRANSLATION_SERVICE_DEFAULT_LANGUAGE")


translation_service_settings = TranslationServiceSettings()
