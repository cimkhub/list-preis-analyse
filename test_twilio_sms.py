from src.config import TwilioConfig, load_config
from src.notify.twilio_sms import (
    build_pipeline_sms_text,
    send_twilio_sms,
    twilio_is_configured,
)


def test_twilio_is_configured_requires_recipient_and_sender():
    config = TwilioConfig(
        account_sid="AC123",
        auth_token="token_123",
        to_number="+49123456789",
        from_number="+49876543210",
    )

    assert twilio_is_configured(config) is True
    assert twilio_is_configured(TwilioConfig(account_sid="AC123", auth_token="token_123")) is False
    assert twilio_is_configured(
        TwilioConfig(account_sid="AC123", auth_token="token_123", to_number="+49123456789")
    ) is False


def test_send_twilio_sms_uses_messaging_service_sid():
    calls = {}

    class DummyMessages:
        def create(self, **kwargs):
            calls["kwargs"] = kwargs

            class DummyMessage:
                sid = "SM123"

            return DummyMessage()

    class DummyClient:
        def __init__(self, account_sid, auth_token):
            calls["account_sid"] = account_sid
            calls["auth_token"] = auth_token
            self.messages = DummyMessages()

    config = TwilioConfig(
        account_sid="AC123",
        auth_token="token_123",
        to_number="+49123456789",
        messaging_service_sid="MG123",
    )

    sent = send_twilio_sms(
        config,
        "Birkenhof pipeline started",
        client_factory=DummyClient,
    )

    assert sent is True
    assert calls["account_sid"] == "AC123"
    assert calls["auth_token"] == "token_123"
    assert calls["kwargs"] == {
        "body": "Birkenhof pipeline started",
        "to": "+49123456789",
        "messaging_service_sid": "MG123",
    }


def test_load_config_supports_existing_twilio_env_aliases(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
suppliers:
  metro:
    enabled: true
    location: "goslar"
    url: "https://www.metro.de/standorte/goslar"
vision_api: {}
pipeline: {}
storage: {}
""".strip(),
        encoding="utf-8",
    )

    monkeypatch.setenv("Account_SID", "AC123")
    monkeypatch.setenv("AuthToken_twilio", "token_123")
    monkeypatch.setenv("Twilio_number", "+49876543210")
    monkeypatch.setenv("TWILIO_TO_NUMBER", "+49123456789")
    monkeypatch.setenv("TWILIO_MESSAGING_SERVICE_SID", "MG123")

    config = load_config(str(config_path))

    assert config.twilio.account_sid == "AC123"
    assert config.twilio.auth_token == "token_123"
    assert config.twilio.from_number == "+49876543210"
    assert config.twilio.to_number == "+49123456789"
    assert config.twilio.messaging_service_sid == "MG123"


def test_build_pipeline_sms_text_is_short_and_informative():
    text = build_pipeline_sms_text(
        stage="pipeline completed",
        week=15,
        year=2026,
        suppliers=["metro", "edeka"],
        extra="Done. Documents found: 12.",
    )

    assert "KW15_2026" in text
    assert "metro, edeka" in text
    assert "Documents found: 12." in text
    assert len(text) < 1200
