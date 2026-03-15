"""Desktop notification handler for LLM feedback."""

import logging
from datetime import datetime

from desktop_notifier import DesktopNotifier

from models.types import LLMResponse

logger = logging.getLogger(__name__)


def create_notification_handler() -> DesktopNotifier:
    """Create and return a DesktopNotifier instance."""
    return DesktopNotifier(
        app_name="Cognitive Mirror",
        notification_limit=5,
    )


def show_llm_feedback_notification(response: LLMResponse) -> None:
    """
    Display a desktop notification for LLM feedback.

    Intended to be used as a subscriber callback in PipelineController.

    Args:
        response: LLMResponse object containing feedback text and timestamp.
    """
    try:
        notifier = create_notification_handler()

        # Format timestamp for display
        timestamp_str = datetime.fromtimestamp(response.timestamp).strftime(
            "%H:%M:%S"
        )

        # Truncate feedback if it's too long (notifications work best with brief text)
        feedback = response.feedback_text
        max_length = 250
        if len(feedback) > max_length:
            feedback = feedback[: max_length - 3] + "..."

        notifier.send(
            title="Cognitive Feedback",
            message=feedback,
            timeout=5,
        )

        logger.debug(
            "Displayed notification at %s: %s",
            timestamp_str,
            response.feedback_text[:50],
        )
    except Exception as e:
        logger.error("Failed to show notification: %s", e)
