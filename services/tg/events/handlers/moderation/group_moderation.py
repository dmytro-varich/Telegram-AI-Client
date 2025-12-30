import logging
from typing import Dict, Any, Optional, Union

from services.tg.events.handlers.base import BaseHandler
from services.ai.moderation import ModerationService
from services.tg.events.event import MessageEvent, UserStatusEvent, ChatActionEvent
from services.tg.events.enums import ChatType

logger = logging.getLogger(__name__)

class GroupModerationHandler(BaseHandler):
    """Handler for moderating messages in channels."""
    
    def __init__(
        self, 
        service: Optional[ModerationService] = None, 
        monitored_groups: Optional[set] = None, 
        *, 
        send_logs_to: Union[str, int, None] = None,
        send_warnings: bool = False
    ):
        """
        Initialize moderation handler.
        
        Args:
            service: AI moderation service (optional, created if None)
            monitored_groups: Set of group/chat IDs to monitor (optional)
            send_logs: Where to send moderation logs (chat ID or username)
            send_warnings: Whether to send warnings to users on violations
        """
        self.moderation_service = service or ModerationService()
        self.monitored_groups = monitored_groups
        self.send_logs_to = send_logs_to
        self.send_warnings = send_warnings

    def can_handle(self, event: Union[MessageEvent, UserStatusEvent, ChatActionEvent]) -> bool:
        """Only handle MessageEvent in groups"""
        
        logger.debug(f"GroupModerationHandler checking event: type={event.__class__.__name__}, chat_type={event.chat_type}, is_outgoing={event.is_outgoing}")

        if not isinstance(event, MessageEvent):
            return False

        # Must be group chat
        if event.chat_type != ChatType.GROUP:
            return False
        
        # Must not be outgoing or service message
        if event.is_outgoing and event.is_service:
            return False
                
        # Check if group is in monitored list (if specified)
        if self.monitored_groups is not None and abs(event.chat_id) not in self.monitored_groups:
            return False
        
        if self.send_logs_to and abs(self.send_logs_to) == abs(event.chat_id):
            return False
        
        return True
                    
    def handle(self, event: MessageEvent) -> None:
        """Process and moderate the message event."""
        if not isinstance(event, MessageEvent):
            logger.warning("GroupModerationHandler received non-MessageEvent")
            return
        
        logger.info(f"Moderating message {event.message_id} in chat {event.chat_id}")
        
        # Call moderation service
        result = self.moderation_service.moderate_message(event)
        
        # Execute based on moderation result
        if result.should_delete:
            logger.warning("Deleting message %s. Reason: %s", event.message_id, result.reason)
            is_delete = event.client.delete_message(event.chat_id, event.message_id)
            if is_delete:
                logger.info("Message %s deleted successfully.", event.message_id)
                if self.send_logs_to:
                    log_text = (
                        f"🗑 <b>Message deleted</b>\n\n"
                        f"🧾 <b>Message ID:</b> <code>{event.message_id}</code>\n"
                        f"💬 <b>Chat ID:</b> <code>{event.chat_id}</code>\n"
                        f"👤 <b>User:</b> {event.sender.full_name} ({event.sender.mention})\n"
                        f"🆔 <b>User ID:</b> <code>{event.sender_id}</code>\n"
                        f"📞 <b>Phone:</b> {event.sender.phone}\n"
                        f"📝 <b>Content:</b> «{event.text}»\n"
                        f"🖼 <b>Media:</b> {event.media.media_type if event.has_media else 'text'}\n"
                        f"⚠️ <b>Reason:</b> {result.reason}\n"
                        f"📊 <b>Confidence:</b> {result.confidence:.2f}\n"
                    )
                    event.client.send_message(self.send_logs_to, log_text, 'html')
                if self.send_warnings:
                    warning_text = (
                        f"❗️ {event.sender.mention}\n Your message in this chat was removed due to violation of community guidelines.\n\n"
                        f"Reason: {result.reason}\n"
                        f"Please adhere to the rules to avoid further actions."
                    )
                    event.client.send_message(event.chat_id, warning_text, 'html')
            else:
                logger.error("Failed to delete message %s.", event.message_id)
        else:
            logger.debug("Message %s passed moderation.", event.message_id)
            