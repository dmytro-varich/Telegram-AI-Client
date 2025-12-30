from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any, Callable
from services.tg.events.router import EventRouter

class BaseTelegramClient(ABC):
    """
    Abstract base class for Telegram clients.
    
    This class defines the common interface that all Telegram client implementations
    (TDLib, Telethon, Pyrogram, etc.) must follow to ensure compatibility.
    
    Attributes:
        config: Client configuration object containing credentials and settings.
    """
    
    # --- Lifecycle Management ---
    
    @abstractmethod
    def start(self) -> bool:
        """
        Initialize and start the Telegram client.
        
        This method should:
        - Initialize the underlying client library
        - Authenticate the user (if not already authenticated)
        - Start the update processing loop
        
        Returns:
            bool: True if client started successfully, False otherwise.
        """
        pass

    @abstractmethod
    def stop(self) -> bool:
        """
        Gracefully shut down the Telegram client.
        
        This method should:
        - Stop the update processing loop
        - Close all active connections
        - Clean up resources
        
        Returns:
            bool: True if client stopped successfully, False otherwise.
        """
        pass

    # --- Sending and Managing Messages ---

    @abstractmethod
    def send_message(
        self,
        peer: str | int,
        text: str,
        parse_mode: Optional[str] = None,
        *,
        message_thread_id: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Send a text message to a chat.
        
        Args:
            peer: Chat identifier (username like '@username' or numeric chat ID).
            text: The text content of the message.
            parse_mode: Text formatting mode ('html', 'markdown', or None for plain text).
            message_thread_id: Thread/topic ID for supergroups (optional).
        
        Returns:
            Optional[Dict[str, Any]]: Message object if successful, None otherwise.
            
        Example:
            >>> client.send_message('@username', 'Hello!')
            >>> client.send_message(123456, '<b>Bold</b>', parse_mode='html')
        """
        pass

    @abstractmethod
    def delete_message(
        self,
        chat_id: int,
        message_id: int,
        revoke: bool = True
    ) -> bool:
        """
        Delete a message from a chat.
        
        Args:
            chat_id: The chat identifier.
            message_id: The message identifier to delete.
            revoke: If True, delete for all chat participants. If False, delete only for self.
        
        Returns:
            bool: True if deletion was successful, False otherwise.
        """
        pass

    # --- Data Retrieval (Getters) ---

    @abstractmethod
    def get_me(self) -> Optional[Dict[str, Any]]:
        """
        Retrieve information about the current authenticated account.
        
        Returns:
            Optional[Dict[str, Any]]: User object containing account information, or None on error.
            
        Example:
            >>> me = client.get_me()
            >>> print(me['first_name'])
        """
        pass

    @abstractmethod
    def get_user(self, user_peer: str | int) -> Optional[Dict[str, Any]]:
        """
        Retrieve information about a user.
        
        Args:
            user_peer: User identifier (username or numeric user ID).
        
        Returns:
            Optional[Dict[str, Any]]: User object if successful, None otherwise.
            
        Example:
            >>> user = client.get_user('@username')
            >>> user = client.get_user(123456789)
        """
        pass
    
    @abstractmethod
    def get_chat(self, chat_peer: str | int) -> Optional[Dict[str, Any]]:
        """
        Retrieve detailed information about a chat.
        
        Args:
            chat_peer: Chat identifier (username or numeric ID).
        
        Returns:
            Optional[Dict[str, Any]]: Chat object if successful, None otherwise.
            
        Example:
            >>> chat = client.get_chat('@channelname')
            >>> chat = client.get_chat(-1001234567890)
        """
        pass

    @abstractmethod
    def get_history(
        self,
        chat_peer: str | int,
        limit: int = 10,
        from_message_id: int = 0,
        *,
        offset: int = 0,
        only_local: bool = False
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Retrieve message history from a chat.
        
        Args:
            chat_peer: Chat identifier (username or numeric ID).
            limit: Maximum number of messages to retrieve (default: 10).
            from_message_id: Message ID to start from (0 for latest messages).
            offset: Number of messages to skip (default: 0).
            only_local: If True, retrieve only locally cached messages (default: False).
        
        Returns:
            Optional[List[Dict[str, Any]]]: List of message objects if successful, None otherwise.
            
        Example:
            >>> messages = client.get_history('@channel', limit=50)
            >>> recent_messages = client.get_history(123456, limit=10)
        """
        pass

    # --- Event Handling ---

    @abstractmethod
    def listen(self, router: EventRouter) -> None:
        """
        Register an update handler to receive real-time updates.
        
        Args:
            router: An EventRouter instance to handle incoming updates.
            
        Returns:
            None
        """
        pass
    
    # --- Additional Methods ---
    
    @abstractmethod
    def mark_read(self, chat_peer: str | int) -> bool:
        """
        Mark all messages in a chat as read.
        
        Args:
            chat_peer: Chat identifier (username or numeric ID).
            
        Returns:
            bool: True if successful, False otherwise.
        """ 