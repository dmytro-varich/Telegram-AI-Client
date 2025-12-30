import logging
from typing import Optional, List, Dict, Any, Callable

from services.tg.client.base import BaseTelegramClient
from services.tg.config import TDLibConfig
from telegram.client import Telegram

from services.tg.events.router import EventRouter

logger = logging.getLogger(__name__)


class TDLibClient(BaseTelegramClient):
    """
    TDLib implementation of BaseTelegramClient.
    
    This client uses the TDLib library to interact with Telegram's API.
    """
    
    def __init__(self, config: TDLibConfig):
        """
        Initialize TDLib client with configuration.
        
        Args:
            config: TDLib-specific configuration object.
        """
        self.config = config
        self.client: Optional[Telegram] = None
          
    def start(self) -> bool:
        try: 
            logger.info("Starting TDLib client: %s", self.config.name)
            self.client = Telegram(
                api_id=self.config.api_id,
                api_hash=self.config.api_hash,
                phone=self.config.phone,
                database_encryption_key=self.config.db_enc_key,
                files_directory=self.config.files_directory,  
                library_path=self.config.library_path,
            )

            self.client.login()
            logger.info("TDLib client started successfully: %s", self.config.name)
            return True
        except Exception as e:
            logger.exception("Failed to start TDLib client %s: %s", self.config.name, str(e))
            return False

    def stop(self) -> bool:
        if self.client:
            logger.info("Stopping TDLib client: %s", self.config.name)
            self.client.stop()
            logger.info("TDLib client stopped: %s", self.config.name)
            return True
        logger.warning("TDLib client not running: %s", self.config.name)
        return False

    def send_message(
        self,
        peer: str | int,
        text: str,
        parse_mode: Optional[str] = None,
        *,
        message_thread_id: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        if not self.client:
            logger.error("Cannot send message: TDLib client is not initialized")
            return None
    
        chat_id = self._resolve_peer(peer)
        
        if chat_id is None:
            logger.error("Failed to resolve peer: %s", peer)
            return None
        
        return self._send(chat_id, text, parse_mode, message_thread_id)
        
    def delete_message(self, chat_id: int, message_id: int, revoke: bool = True) -> bool:
        if not self.client:
            logger.error("Cannot delete message: TDLib client is not initialized")
            return False
        
        try:
            result = self.client.call_method('deleteMessages', {
                'chat_id': chat_id,
                'message_ids': [message_id],
                'revoke': revoke  # Whether to delete for all users
            })
            result.wait()
            
            if result.error:
                logger.error("Error deleting message %s in chat %s: %s", 
                           message_id, chat_id, result.error_info)
                return False
            
            logger.info("Deleted message %s in chat %s", message_id, chat_id)
            return True
        except Exception as e:
            logger.exception("Exception while deleting message: %s", str(e))
            return False

    def get_me(self) -> Optional[Dict[str, Any]]:
        if not self.client:
            logger.warning("TDLib client is not initialized")
            return None
            
        result = self.client.get_me()
        result.wait()
        
        if result.error:
            logger.error("Error retrieving account info: %s", result.error_info)
            return None
        
        return result.update

    def get_user(self, user_peer: str | int) -> Optional[Dict[str, Any]]:
        if not self.client: 
            logger.error("Cannot get user: TDLib client is not initialized")
            return None
       
        try:             
            user_id = self._resolve_peer(user_peer)
            if user_id is None:
                return None
            
            result = self.client.call_method('getUser', {'user_id': user_id})
            result.wait()

            if result.error:
                logger.error("Error retrieving user %s: %s", user_peer, result.error_info)
                return None
            
            logger.info("Retrieved user %s", user_peer)
            return result.update
        except Exception as e:
            logger.exception("Exception while retrieving user %s: %s", user_peer, str(e))
            return None
    
    def get_chat(self, chat_peer: str | int) -> Optional[Dict[str, Any]]:
        if not self.client:
            logger.error("Cannot get chat: TDLib client is not initialized")
            return None
        
        try: 
            chat_id = self._resolve_peer(chat_peer)
            if chat_id is None:
                return None
            
            result = self.client.call_method('getChat', {'chat_id': chat_id})
            result.wait()
            
            if result.error:
                logger.error("Error getting chat %s: %s", chat_peer, result.error_info)
                return None
            
            logger.info("Retrieved chat %s", chat_peer)
            return result.update
        except Exception as e:
            logger.exception("Exception while getting chat %s: %s", chat_peer, str(e))
            return None
        
    def get_history(
        self,
        chat_peer: str | int,
        limit: int = 10,
        from_message_id: int = 0,
        *,
        offset: int = 0,
        only_local: bool = False
    ) -> Optional[List[Dict[str, Any]]]:
        if not self.client:
            logger.error("Cannot get history: TDLib client is not initialized")
            return None
        
        try:
            chat_id = self._resolve_peer(chat_peer)
            if chat_id is None:
                return None
            
            result = self.client.call_method('getChatHistory', {
                'chat_id': chat_id,
                'from_message_id': from_message_id,
                'offset': offset,
                'limit': limit,
                'only_local': only_local
            })
            result.wait()
            
            if result.error:
                logger.error("Error getting history for chat %s: %s", chat_peer, result.error_info)
                return None
            
            messages = result.update.get('messages', [])
            logger.info("Retrieved %d messages from chat %s", len(messages), chat_peer)
            return messages
        except Exception as e:
            logger.exception("Exception while getting history for chat %s: %s", chat_peer, str(e))
            return None

    def listen(self, router: EventRouter) -> None:
        """Register event router to handle all updates.
        
         Args:
            router: MessageEvent router that processes incoming updates.
        """
        if not self.client:
            logger.error("Cannot register handler: TDLib client is not initialized")
            return
        
        try:
            self.client.add_message_handler(lambda update: router.route(update, self))
            logger.info("Registered event router for client %s", self.config.name)
        except Exception as e:
            logger.exception("Exception while registering router: %s", str(e))
    
    # --- Additional Methods ---
    def mark_read(self, chat_peer: str | int) -> bool:
        """
        Mark all messages in a chat as read.
        
        Args:
            chat_peer: Chat identifier (username or numeric ID).
            
        Returns:
            bool: True if successful, False otherwise.
        """ 
        if not self.client:
            logger.error("Cannot mark read: TDLib client is not initialized")
            return False
        
        chat_id = self._resolve_peer(chat_peer)
        if chat_id is None:
            logger.error("Failed to resolve peer for marking read: %s", chat_peer)
            return False
        
        try:
            result = self.client.call_method('viewMessages', {
                'chat_id': chat_id,
                'message_ids': []  # Empty list marks all as read
            })
            result.wait()
            
            if result.error:
                logger.error("Error marking messages as read in chat %s: %s", 
                           chat_peer, result.error_info)
                return False
            
            logger.info("Marked messages as read in chat %s", chat_peer)
            return True
        except Exception as e:
            logger.exception("Exception while marking messages as read in chat %s: %s", 
                             chat_peer, str(e))
            return False
        
    # --- Private Helper Methods (TDLib-specific) ---
        
    def _resolve_peer(self, peer: str | int) -> Optional[int]:
        """
        Resolve peer identifier to TDLib chat_id.
        
        Tries multiple ID formats to find the correct chat:
        - Username lookup via searchPublicChat
        - Direct numeric ID
        - Negative ID (for groups)
        - Supergroup format (-100xxxxxxxxx)
        
        Args:
            peer: Username or numeric identifier.
        
        Returns:
            Optional[int]: Resolved chat_id, or None if resolution fails.
        """
        if not self.client:
            return None
        
        # Handle username
        if isinstance(peer, str) and not peer.lstrip('-').isdigit():
            username = peer.lstrip('@')
            result = self.client.call_method('searchPublicChat', {'username': username})
            result.wait()
            
            if not result.error:
                return result.update.get('id')
            
            logger.error("Failed to resolve username '%s': %s", username, result.error_info)
            return None
        
        # Handle numeric ID - try different formats
        try:
            peer_id = int(peer)
        except (ValueError, TypeError):
            logger.error("Invalid peer format: %s", peer)
            return None
        
        # Try different ID formats
        id_variants = [
            peer_id,
            abs(peer_id),
            -abs(peer_id),
            int('-100' + str(abs(peer_id))) if not str(abs(peer_id)).startswith('100') else -abs(peer_id),
        ]
        
        id_variants = list(dict.fromkeys(id_variants))
        
        for chat_id in id_variants:
            result = self.client.call_method('getChat', {'chat_id': chat_id})
            result.wait()
            
            if not result.error:
                logger.debug("Resolved peer %s to chat_id %s", peer, chat_id)
                return chat_id
        
        logger.error("Failed to resolve peer %s", peer)
        return None
    
    def _send(
        self,
        chat_id: int,
        text: str,
        parse_mode: Optional[str] = None,
        message_thread_id: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Internal method to send a message (TDLib-specific).
        
        Args:
            chat_id: Resolved chat identifier.
            text: Message text.
            parse_mode: Formatting mode.
            message_thread_id: Thread ID for topics.
        
        Returns:
            Optional[Dict]: Sent message object, or None on error.
        """
        try:
            input_message_content = self._parse_text(text, parse_mode)
            
            send_params = {
                'chat_id': chat_id,
                'input_message_content': input_message_content
            }
            
            if message_thread_id is not None:
                send_params['message_thread_id'] = message_thread_id
        
            result = self.client.call_method('sendMessage', send_params)
            result.wait()
            
            if result.error:
                logger.error("Error sending message to chat %s: %s", chat_id, result.error_info)
                return None
            
            logger.info("Message sent to chat %s", chat_id)
            return result.update
        except Exception as e:
            logger.exception("Exception while sending message to chat %s: %s", chat_id, str(e))
            return None
        
    def _parse_text(self, text: str, parse_mode: Optional[str] = None) -> Dict[str, Any]:
        """
        Parse text with formatting entities (TDLib-specific).
        
        Args:
            text: Plain text to parse.
            parse_mode: Formatting mode ('html', 'markdown', or None).
        
        Returns:
            Dict containing formatted text structure for TDLib.
        """
        input_message_content = {
            '@type': 'inputMessageText',
            'text': {
                '@type': 'formattedText',
                'text': text,
                'entities': []
            }
        }
        
        if parse_mode:
            parse_type = {
                'html': 'textParseModeHTML',
                'markdown': 'textParseModeMarkdown'
            }.get(parse_mode.lower(), 'textParseModeMarkdown')
            
            parse_result = self.client.call_method('parseTextEntities', {
                'text': text,
                'parse_mode': {'@type': parse_type}
            })
            parse_result.wait()

            if not parse_result.error:
                input_message_content['text'] = parse_result.update
            else:
                logger.warning("Failed to parse entities: %s. Sending as plain text.", parse_result.error_info)
        
        return input_message_content