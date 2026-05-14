import logging
from typing import Callable

from src.config import TwilioConfig


def missing_twilio_fields(config: TwilioConfig) -> list[str]:
    missing = []
    if not config.account_sid:
        missing.append("TWILIO_ACCOUNT_SID/Account_SID")
    if not config.auth_token:
        missing.append("TWILIO_AUTH_TOKEN/AuthToken_twilio")
    if not config.to_number:
        missing.append("TWILIO_TO_NUMBER")
    if not config.from_number and not config.messaging_service_sid:
        missing.append("TWILIO_FROM_NUMBER/Twilio_number or TWILIO_MESSAGING_SERVICE_SID")
    return missing


def twilio_is_configured(config: TwilioConfig) -> bool:
    return config.enabled and not missing_twilio_fields(config)


def build_pipeline_sms_text(
    *,
    stage: str,
    week: int,
    year: int,
    suppliers: list[str],
    extra: str | None = None,
) -> str:
    suppliers_text = ", ".join(suppliers) if suppliers else "all suppliers"
    base = f"Birkenhof {stage} for KW{week:02d}_{year}. Suppliers: {suppliers_text}."
    if extra:
        base = f"{base} {extra}"
    return base[:1200]


def _default_client_factory(account_sid: str, auth_token: str):
    from twilio.rest import Client

    return Client(account_sid, auth_token)


def send_twilio_sms(
    config: TwilioConfig,
    text: str,
    logger: logging.Logger | None = None,
    client_factory: Callable | None = None,
) -> bool:
    logger = logger or logging.getLogger("birkenhof.notify.twilio")

    if not config.enabled:
        logger.debug("Twilio SMS notifications are disabled")
        return False

    missing = missing_twilio_fields(config)
    if missing:
        logger.warning(
            "Twilio SMS not sent, missing configuration: %s",
            ", ".join(missing),
        )
        return False

    factory = client_factory or _default_client_factory
    try:
        client = factory(config.account_sid, config.auth_token)
        create_kwargs = {
            "body": text,
            "to": config.to_number,
        }
        if config.messaging_service_sid:
            create_kwargs["messaging_service_sid"] = config.messaging_service_sid
        else:
            create_kwargs["from_"] = config.from_number

        message = client.messages.create(**create_kwargs)
        message_sid = getattr(message, "sid", None)
        logger.info("Twilio SMS sent%s", f" ({message_sid})" if message_sid else "")
        return True
    except Exception as exc:
        logger.warning("Twilio SMS request failed: %s", exc)
        return False
