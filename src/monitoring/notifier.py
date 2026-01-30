"""Notification channels: Telegram, Slack, and webhook.

Sends trading alerts and system notifications to external channels.
Configurable via environment variables.

Usage:
    notifier = TelegramNotifier(bot_token="...", chat_id="...")
    notifier.send_alert(alert)

    slack = SlackNotifier(webhook_url="...")
    slack.send_alert(alert)
"""

import json
import os
from abc import ABC, abstractmethod
from typing import Any

import requests
from loguru import logger

from src.monitoring.alerts import Alert, AlertLevel


class BaseNotifier(ABC):
    """Abstract base for notification channels."""

    @abstractmethod
    def send_alert(self, alert: Alert) -> bool:
        """Send an alert notification. Returns True on success."""

    @abstractmethod
    def send_message(self, message: str) -> bool:
        """Send a plain text message. Returns True on success."""


class TelegramNotifier(BaseNotifier):
    """Send notifications via Telegram Bot API.

    Requires:
        TELEGRAM_BOT_TOKEN: Bot token from @BotFather
        TELEGRAM_CHAT_ID: Chat or group ID to send to
    """

    BASE_URL = "https://api.telegram.org/bot{token}/sendMessage"

    def __init__(
        self,
        bot_token: str | None = None,
        chat_id: str | None = None,
    ) -> None:
        self.bot_token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID", "")

        if not self.bot_token or not self.chat_id:
            logger.warning("TelegramNotifier: Missing bot_token or chat_id")

    def _format_alert(self, alert: Alert) -> str:
        """Format alert as Telegram message with emoji indicators."""
        level_emoji = {
            AlertLevel.INFO: "ℹ️",
            AlertLevel.WARNING: "⚠️",
            AlertLevel.CRITICAL: "🚨",
        }
        emoji = level_emoji.get(alert.level, "📊")
        lines = [
            f"{emoji} *Medal Alert: {alert.alert_type.value}*",
            f"Level: {alert.level.value}",
            f"Message: {alert.message}",
            f"Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}",
        ]
        if alert.data:
            data_str = ", ".join(f"{k}={v}" for k, v in alert.data.items() if not isinstance(v, (dict, list)))
            lines.append(f"Data: {data_str}")
        return "\n".join(lines)

    def send_alert(self, alert: Alert) -> bool:
        """Send alert via Telegram."""
        return self.send_message(self._format_alert(alert), parse_mode="Markdown")

    def send_message(self, message: str, parse_mode: str = "Markdown") -> bool:
        """Send message via Telegram Bot API."""
        if not self.bot_token or not self.chat_id:
            logger.debug("TelegramNotifier: Not configured, skipping")
            return False

        url = self.BASE_URL.format(token=self.bot_token)
        payload = {
            "chat_id": self.chat_id,
            "text": message,
            "parse_mode": parse_mode,
        }

        try:
            resp = requests.post(url, json=payload, timeout=10)
            if resp.status_code == 200:
                logger.info("Telegram: Message sent")
                return True
            logger.error("Telegram: Failed ({}): {}", resp.status_code, resp.text)
            return False
        except Exception as e:
            logger.error("Telegram: Error sending message: {}", e)
            return False


class SlackNotifier(BaseNotifier):
    """Send notifications via Slack Incoming Webhook.

    Requires:
        SLACK_WEBHOOK_URL: Incoming webhook URL from Slack app
    """

    def __init__(self, webhook_url: str | None = None) -> None:
        self.webhook_url = webhook_url or os.getenv("SLACK_WEBHOOK_URL", "")

        if not self.webhook_url:
            logger.warning("SlackNotifier: Missing webhook_url")

    def _format_alert(self, alert: Alert) -> dict:
        """Format alert as Slack message block."""
        level_emoji = {
            AlertLevel.INFO: ":information_source:",
            AlertLevel.WARNING: ":warning:",
            AlertLevel.CRITICAL: ":rotating_light:",
        }
        emoji = level_emoji.get(alert.level, ":chart_with_upwards_trend:")

        return {
            "blocks": [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": f"{emoji} Medal Alert: {alert.alert_type.value}",
                    },
                },
                {
                    "type": "section",
                    "fields": [
                        {"type": "mrkdwn", "text": f"*Level:*\n{alert.level.value}"},
                        {"type": "mrkdwn", "text": f"*Type:*\n{alert.alert_type.value}"},
                        {"type": "mrkdwn", "text": f"*Message:*\n{alert.message}"},
                        {"type": "mrkdwn", "text": f"*Time:*\n{alert.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}"},
                    ],
                },
            ],
        }

    def send_alert(self, alert: Alert) -> bool:
        """Send alert via Slack webhook."""
        if not self.webhook_url:
            logger.debug("SlackNotifier: Not configured, skipping")
            return False

        payload = self._format_alert(alert)

        try:
            resp = requests.post(self.webhook_url, json=payload, timeout=10)
            if resp.status_code == 200:
                logger.info("Slack: Message sent")
                return True
            logger.error("Slack: Failed ({}): {}", resp.status_code, resp.text)
            return False
        except Exception as e:
            logger.error("Slack: Error sending message: {}", e)
            return False

    def send_message(self, message: str) -> bool:
        """Send plain text via Slack webhook."""
        if not self.webhook_url:
            return False

        try:
            resp = requests.post(
                self.webhook_url,
                json={"text": message},
                timeout=10,
            )
            return resp.status_code == 200
        except Exception as e:
            logger.error("Slack: Error: {}", e)
            return False


class WebhookNotifier(BaseNotifier):
    """Send notifications to a generic webhook URL.

    Supports any HTTP endpoint that accepts JSON POST requests.
    """

    def __init__(self, url: str | None = None, headers: dict | None = None) -> None:
        self.url = url or os.getenv("WEBHOOK_URL", "")
        self.headers = headers or {"Content-Type": "application/json"}

    def send_alert(self, alert: Alert) -> bool:
        """Send alert as JSON to webhook."""
        if not self.url:
            return False

        payload = {
            "type": alert.alert_type.value,
            "level": alert.level.value,
            "message": alert.message,
            "timestamp": alert.timestamp.isoformat(),
            "data": alert.data,
        }

        try:
            resp = requests.post(self.url, json=payload, headers=self.headers, timeout=10)
            return resp.status_code in (200, 201, 202)
        except Exception as e:
            logger.error("Webhook: Error: {}", e)
            return False

    def send_message(self, message: str) -> bool:
        """Send plain text as JSON to webhook."""
        if not self.url:
            return False

        try:
            resp = requests.post(
                self.url,
                json={"message": message},
                headers=self.headers,
                timeout=10,
            )
            return resp.status_code in (200, 201, 202)
        except Exception as e:
            logger.error("Webhook: Error: {}", e)
            return False


class MultiNotifier(BaseNotifier):
    """Dispatches alerts to multiple channels.

    Usage:
        multi = MultiNotifier([
            TelegramNotifier(),
            SlackNotifier(),
        ])
        multi.send_alert(alert)
    """

    def __init__(self, notifiers: list[BaseNotifier] | None = None) -> None:
        self.notifiers = notifiers or []

    def add(self, notifier: BaseNotifier) -> None:
        """Add a notification channel."""
        self.notifiers.append(notifier)

    def send_alert(self, alert: Alert) -> bool:
        """Send to all channels. Returns True if any succeeded."""
        results = []
        for n in self.notifiers:
            try:
                results.append(n.send_alert(alert))
            except Exception as e:
                logger.error("MultiNotifier: Channel failed: {}", e)
                results.append(False)
        return any(results)

    def send_message(self, message: str) -> bool:
        """Send to all channels."""
        results = []
        for n in self.notifiers:
            try:
                results.append(n.send_message(message))
            except Exception as e:
                results.append(False)
        return any(results)


def create_notifier_from_env() -> MultiNotifier:
    """Create a MultiNotifier configured from environment variables.

    Checks for:
        TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID
        SLACK_WEBHOOK_URL
        WEBHOOK_URL
    """
    multi = MultiNotifier()

    if os.getenv("TELEGRAM_BOT_TOKEN") and os.getenv("TELEGRAM_CHAT_ID"):
        multi.add(TelegramNotifier())
        logger.info("Notifier: Telegram enabled")

    if os.getenv("SLACK_WEBHOOK_URL"):
        multi.add(SlackNotifier())
        logger.info("Notifier: Slack enabled")

    if os.getenv("WEBHOOK_URL"):
        multi.add(WebhookNotifier())
        logger.info("Notifier: Webhook enabled")

    if not multi.notifiers:
        logger.info("Notifier: No channels configured (alerts will be logged only)")

    return multi
