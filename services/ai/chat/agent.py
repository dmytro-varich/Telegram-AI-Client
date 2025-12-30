import logging
from typing import Optional, List, Dict

from services.ai.chat.base import BaseChatModel
from services.ai.rag.retriever import Retriever
from services.ai.chat.response import ChatResponse
from services.tg.events import MessageEvent

logger = logging.getLogger(__name__)


class ChatAgent:
    """
    Chat agent that orchestrates conversation logic.
    
    Combines:
    - System prompt (personality/instructions)
    - RAG retriever (knowledge base)
    - Chat model (AI API)
    - Conversation history (memory)
    """
    
    def __init__(
        self,
        chat_model: BaseChatModel,
        system_prompt: str = "You are a helpful assistant.",
        retriever: Optional[Retriever] = None,
        max_history: int = 10
    ):
        """
        Initialize chat agent.
        
        Args:
            chat_model: AI model for generating responses
            system_prompt: System instructions/personality
            retriever: RAG retriever for knowledge base (optional)
            max_history: Maximum conversation history to keep
        """
        self.chat_model = chat_model
        self.system_prompt = system_prompt
        self.retriever = retriever
        self.max_history = max_history
        
        # Conversation history per user
        self._conversations: Dict[int, List[Dict[str, str]]] = {}
        
        logger.info("ChatAgent initialized with system prompt: '%s...'", system_prompt[:50])
        if retriever:
            logger.info("RAG retriever enabled")
            
    def generate_response(
        self, 
        event: MessageEvent, 
        clear_history: bool = False
    ) -> ChatResponse:
        """
        Generate response to user message.
        
        Args:
            event: Normalized message event
            clear_history: Whether to clear conversation history
        
        Returns:
            Generated response text
        """
        user_id = event.sender_id 
        user_message = ''

        # Clear history if requested
        if clear_history:
            self._conversations[user_id] = []
        
        # Get conversation history
        history = self._conversations.get(user_id, [])
        
        # Text-only message
        if not event.has_media and event.text:
            user_message = event.text
        
        # Media with caption
        if event.has_media and event.media:
            
            # Voice note
            if event.media.media_type in ('voicenote', 'voice'):
                # Download voice from Telegram
                audio_data = self._download_file(event.client, event.media.file_id)

                if not audio_data:
                    # Fallback to caption moderation only
                    if event.media.caption:
                        user_message = transcription    
                                
                # Transcribe voice to text
                transcription = self._transcribe_voice(audio_data)

                if transcription:
                    user_message = transcription

        # Build context from RAG if available
        rag_context = ""
        if self.retriever:
            logger.debug("Retrieving relevant documents from knowledge base")
            documents = self.retriever.retrieve(user_message, top_k=3)
            
            if documents:
                rag_context = "\n\n".join([
                    f"[Document {i+1}]\n{doc['document']}"
                    for i, doc in enumerate(documents)
                ])
                logger.debug(f"Retrieved {len(documents)} relevant documents")
        
        # Generate response
        response = self.chat_model.generate(
            system_prompt=self.system_prompt,
            user_message=user_message,
            conversation_history=history,
            rag_context=rag_context
        )
        
        # Update conversation history (only if not escalated)
        if not response.should_escalate:
            history.append({"role": "user", "content": user_message})
            history.append({"role": "assistant", "content": response.message})
            
            # Keep only last N messages
            if len(history) > self.max_history * 2:
                history = history[-self.max_history * 2:]
            
            self._conversations[user_id] = history
        
        return response

    def _download_file(self, client, file_id: str) -> bytes | None:
        """
        Download file from Telegram client.
        
        Args:
            client: TDLib client instance
            file_id: File ID to download
            
        Returns:
            File binary data or None
        """
        try: 
            # Get file info
            file_result = client.client.call_method('getFile', params={'file_id': file_id})
            file_result.wait()
            
            if file_result.error:
                logging.error(f"Failed to get file info: {file_result.error}")
                return None
            
            file_info = file_result.update
            local_path = file_info.get('local', {}).get('path', '')
            is_downloaded = file_info.get('local', {}).get('is_downloading_completed', False)
            
            # Download if not already downloaded
            if not local_path or not is_downloaded:
                logger.debug(f"Downloading file {file_id}...")
                download_result = client.client.call_method(
                    'downloadFile',
                    params={
                        'file_id': file_id,
                        'priority': 32,
                        'synchronous': True
                    }
                )
                download_result.wait()
                
                if download_result.error:
                    logger.error(f"Error downloading file: {download_result.error}")
                    return None
                
                local_path = download_result.update.get('local', {}).get('path', '')
            
            # Read file as bytes
            if local_path and os.path.exists(local_path):
                with open(local_path, 'rb') as f:
                    file_bytes = f.read()
                
                logger.info(f"Downloaded file {file_id}: {len(file_bytes)} bytes")
                return file_bytes
            
            logger.error(f"File path not found for {file_id}")
            return None
        except Exception as e:
            logging.error(f"Exception occurred while downloading file: {e}")
            return None
        
    def _transcribe_voice(self, audio_data: bytes) -> str | None:
        """
        Transcribe voice audio to text using Whisper.
        
        Args:
            audio_data: Audio file binary data
            
        Returns:
            Transcribed text or None
        """ 
        try: 
            # Lazy load Whisper model
            if not self._whisper_model:
                import whisper
                logger.info("Loading Whisper model (base)...")
                self._whisper_model = whisper.load_model("base")  # tiny/base/small/medium/large
                logger.info("Whisper model loaded successfully")
            
            # Save audio data to temporary file (Whisper needs file path)
            with tempfile.NamedTemporaryFile(suffix='.ogg', delete=False) as temp_file:
                temp_file.write(audio_data)
                temp_path = temp_file.name
                
            try:
                # Transcribe audio
                logger.debug(f"Transcribing audio file: {temp_path}")
                result = self._whisper_model.transcribe(
                    temp_path,
                    language=None,  # Auto-detect language
                    fp16=False      # Disable FP16 for CPU compatibility
                )
                
                transcribed_text = result['text'].strip()
                detected_language = result.get('language', 'unknown')
                
                logger.info(
                    f"Voice transcribed: '{transcribed_text[:50]}...' (lang: {detected_language})"
                )
                
                return transcribed_text
            finally:
                # Clean up temporary file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        except ImportError: 
            logger.error("Whisper not installed. Install with: pip install openai-whisper")
            return None
        except Exception as e:
            logger.exception(f"Failed to transcribe voice: {e}")
            return None
            
    