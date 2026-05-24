import os
from pathlib import Path

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel


class SupplierConfig(BaseModel):
    enabled: bool = True
    location: str
    url: str
    extraction_method: str = "text_first"
    price_type: str = "gross"
    default_vat_rate: float = 0.07
    preferred_tier: str = "middle"
    region: str | None = None
    store_code: str | None = None
    relevant_categories: list[str] | None = None


class VisionApiConfig(BaseModel):
    provider: str = "gemini"
    model: str = "gemini-2.5-flash"
    max_retries: int = 3
    temperature: float = 0.1
    max_concurrent_requests: int = 10
    min_request_interval_seconds: float = 0.0
    api_key: str = ""


class TwilioConfig(BaseModel):
    enabled: bool = True
    account_sid: str = ""
    auth_token: str = ""
    to_number: str = ""
    from_number: str = ""
    messaging_service_sid: str | None = None


class PipelineConfig(BaseModel):
    image_dpi: int = 300
    image_format: str = "png"
    focus_categories: list[str] = []
    fuzzy_match_threshold: int = 80


class StorageConfig(BaseModel):
    base_dir: str = "."
    data_dir: str = "data"
    images_dir: str = "images"
    parsed_dir: str = "parsed"
    reports_dir: str = "reports"
    reference_dir: str = "reference"
    logs_dir: str = "logs"

    def resolve(self, sub_dir: str) -> Path:
        return Path(self.base_dir) / getattr(self, sub_dir)


class AppConfig(BaseModel):
    suppliers: dict[str, SupplierConfig]
    vision_api: VisionApiConfig
    twilio: TwilioConfig = TwilioConfig()
    pipeline: PipelineConfig
    storage: StorageConfig


def load_config(config_path: str = "config.yaml") -> AppConfig:
    load_dotenv()
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    config = AppConfig(**raw)
    config.vision_api.api_key = (
        os.environ.get("GEMINI_API_KEY_PRIVATE")
        or os.environ.get("GEMINI_API_KEY_NEW")
        or os.environ.get("GEMINI_API_KEY-new")
        or os.environ.get("GEMINI_API_KEY", "")
    )
    config.vision_api.model = os.environ.get("GEMINI_MODEL", config.vision_api.model)
    if os.environ.get("GEMINI_MAX_RETRIES"):
        config.vision_api.max_retries = int(os.environ["GEMINI_MAX_RETRIES"])
    if os.environ.get("GEMINI_MAX_CONCURRENT_REQUESTS"):
        config.vision_api.max_concurrent_requests = int(os.environ["GEMINI_MAX_CONCURRENT_REQUESTS"])
    if os.environ.get("GEMINI_MIN_REQUEST_INTERVAL_SECONDS"):
        config.vision_api.min_request_interval_seconds = float(os.environ["GEMINI_MIN_REQUEST_INTERVAL_SECONDS"])
    config.twilio.account_sid = (
        os.environ.get("TWILIO_ACCOUNT_SID")
        or os.environ.get("Account_SID", "")
    )
    config.twilio.auth_token = (
        os.environ.get("TWILIO_AUTH_TOKEN")
        or os.environ.get("AuthToken_twilio", "")
    )
    config.twilio.to_number = (
        os.environ.get("TWILIO_TO_NUMBER")
        or os.environ.get("SMS_TO_NUMBER", "")
    )
    config.twilio.from_number = (
        os.environ.get("TWILIO_FROM_NUMBER")
        or os.environ.get("Twilio_number", "")
    )
    config.twilio.messaging_service_sid = (
        os.environ.get("TWILIO_MESSAGING_SERVICE_SID")
        or os.environ.get("Messaging_Service_SID")
        or None
    )
    config.twilio.enabled = os.environ.get("TWILIO_ENABLED", "true").strip().lower() not in {
        "0", "false", "no", "off",
    }

    return config
