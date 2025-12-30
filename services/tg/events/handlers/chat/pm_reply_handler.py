import logging
from typing import Optional, Union, Set

from services.tg.events.handlers.base import BaseHandler
from services.tg.events.event import MessageEvent
from services.tg.events.enums import ChatType
from services.ai.chat.agent import ChatAgent

logger = logging.getLogger(__name__)


class PMReplyHandler(BaseHandler):
    """Handler for replying to private messages using a chat agent."""
    
    def __init__(
        self, 
        agent: Optional[ChatAgent] = None, 
        monitored_users: Optional[set] = None, 
        escalation_chat_id: Optional[int] = None
    ):
        """
        Initialize PM reply handler.
        
        Args:
            agent: Chat agent to generate replies (optional, created if None)
            users: Set of user IDs to respond to (optional)
            escalation_chat_id: Where to send moderation logs (chat ID or username)
        """
        self.agent = agent 
        self.monitored_users = monitored_users
        self.escalation_chat_id = escalation_chat_id
        logger.info("PMReplyHandler initialized")

    def can_handle(self, event: MessageEvent) -> bool:
        """Only handle private message events with text."""
        
        logger.debug(f"PMReplyHandler checking event: type={event.__class__.__name__}, chat_type={event.chat_type}, is_outgoing={event.is_outgoing}")
        
        if not isinstance(event, MessageEvent):
            return False

        # Must be private chat
        if event.chat_type != ChatType.PRIVATE:
            return False
        
        # Must not be outgoing or service message
        if event.is_outgoing:
            return False
        
        if event.is_service:
            return False
        
        if not event.text or not event.text.strip():
            return False
        
        if self.monitored_users is not None and event.sender_id not in self.monitored_users:
            return False
        
        return True

    def handle(self, event: MessageEvent) -> None:
        """Generate and send reply to private message."""
        logger.info(
            f"Handling PM from {event.sender.full_name} (@{event.sender.username}): "
            f"'{event.text[:50]}...'"
        )
        
        try:
            # Mark message as read
            event.client.mark_read(event.chat_id)
            
            # Generate response using agent (agent handles RAG, history, etc.)
            response = self.agent.generate_response(event=event)
            
            # Check if escalation needed
            if response.should_escalate:
                logger.warning(
                    f"⚠️ Escalation required: {response.escalation_reason} "
                    f"(confidence: {response.confidence})"
                )
                
                # Send escalation notification
                if self.escalation_chat_id:
                    escalation_text = (
                        f"🔔 Escalation Required\n\n"
                        f"User: {event.sender.full_name} (@{event.sender.username})\n"
                        f"User ID: {event.sender_id}\n"
                        f"Chat ID: {event.chat_id}\n\n"
                        f"Question: {event.text}\n\n"
                        f"Reason: {response.escalation_reason}\n"
                        f"Confidence: {response.confidence:.2f}\n\n"
                        f"Auto-reply sent: {response.message}"
                    )
                    event.client.send_message(self.escalation_chat_id, escalation_text)
            
            # Send reply to user
            sent_message = event.client.send_message(
                peer=event.chat_id,
                text=response.to_telegram_message(),
            )
            
            if sent_message:
                status = "(escalated)" if response.should_escalate else "[OK]"
                logger.info(f"{status} Replied to message {event.message_id}")
            else:
                logger.error(f"Failed to send reply to message {event.message_id}")
        except Exception as e:
            logger.error(f"Error handling PM reply: {e}")