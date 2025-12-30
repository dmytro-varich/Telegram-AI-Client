import logging
import time
import os

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

from services.tg.client.manager import TelegramClientManager
from services.tg.client import TDLibClient
from services.tg.utils import load_tdlib_account
from services.tg.events.router import EventRouter

# Handlers
from services.tg.events.handlers.chat import PMReplyHandler
from services.tg.events.handlers.moderation import GroupModerationHandler

# AI Services
from services.ai.chat.agent import ChatAgent
from services.ai.chat.openai import OpenAIGPTModel
from services.ai.moderation.service import ModerationService
from services.ai.moderation.openai import OpenAIModerationModel

# Knowledge Base & RAG
from services.knowledge_base import KnowledgeBase
from services.knowledge_base.sources.pdf_source import PdfTextSource
from services.knowledge_base.stores.chroma_store import ChromaVectorStore
from services.ai.rag.retriever import Retriever

from config import (
    LIBRARY_PATH, FOLDER_ACCOUNTS, MONITORED_USERS_FILE, MONITORED_GROUPS_FILE, 
    LOGS_ID_CHAT, MODERATE_ID_CHAT,
    OPENAI_API_KEY, OPENAI_CHAT_MODEL, SYSTEM_PROMPT_FILE, 
    PDF_KNOWLEDGE_BASE, EMBEDDING_MODEL
)
from utils import load_users, load_groups
from utils.files import get_account_files
from services.ai.utils import load_prompt

logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", 
    handlers=[
        logging.FileHandler("logs/main.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def main():
    logger.info("="*20)
    logger.info("Starting Telegram AI Agent Bot")
    logger.info("Features: Private Message Reply (RAG) + Group Moderation")
    logger.info("="*20)
    
    monitored_users = load_users(MONITORED_USERS_FILE)
    monitored_groups = load_groups(MONITORED_GROUPS_FILE)
    
    logger.info(f"Monitored Users: {len(monitored_users)}")
    logger.info(f"Monitored Groups: {len(monitored_groups)}")
    
    monitored_users_ids = set()
    monitored_groups_ids = set()
    
    for user in monitored_users:
        uid = user.get("id")
        username = user.get("username")
        
        if uid:
            monitored_users_ids.add(uid)
        elif username:
            monitored_users_ids.add(username)
                
    for group in monitored_groups:
        uid = group.get("id")
        
        if uid:
            monitored_groups_ids.add(uid)
        
    # =========================================================================
    # 1. Initialize RAG Knowledge Base for PM Replies
    # =========================================================================
    logger.info("Initializing Knowledge Base for RAG...")
    
    system_prompt = load_prompt(SYSTEM_PROMPT_FILE)
    logger.info(f"Loaded system prompt: '{system_prompt[:50]}...'")
    
    # Configure OpenAI embedding function
    embedding_function = OpenAIEmbeddingFunction(
        api_key=OPENAI_API_KEY,
        model_name=EMBEDDING_MODEL
    )
    
    # Create ChromaDB client and collection
    chroma_client = chromadb.PersistentClient(path="./data/chroma")
    collection = chroma_client.get_or_create_collection(
        name="company_knowledge",
        metadata={"hnsw:space": "cosine"}, 
        embedding_function=embedding_function
    )
    
    # Wrap collection in store
    vector_store = ChromaVectorStore(collection=collection)
    
    # Create PDF source
    pdf_source = PdfTextSource(file_path=PDF_KNOWLEDGE_BASE)
    
    # Create knowledge base
    kb = KnowledgeBase(
        source=pdf_source,
        store=vector_store,
        chunk_size=500,
        chunk_overlap=50
    )
    
    # Build or load index
    force_rebuild = os.getenv('REBUILD_KB', 'false').lower() == 'true'
    indexed_count = kb.build_index(force_rebuild=force_rebuild)
    
    if indexed_count == 0:
        logger.error("Failed to build knowledge base index!")
        return
    
    logger.info(f"✅ Knowledge base ready: {indexed_count} chunks indexed")
    
    # Create retriever
    retriever = Retriever(kb)
    
    # =========================================================================
    # 2. Initialize AI Models and Services
    # =========================================================================
    logger.info("Initializing AI models...")
    
    # Chat model for PM replies
    chat_model = OpenAIGPTModel(
        api_key=OPENAI_API_KEY,
        model=OPENAI_CHAT_MODEL
    )
    
    # Moderation model for groups
    moderation_model = OpenAIModerationModel(api_key=OPENAI_API_KEY)
    
    # Create services
    chat_agent = ChatAgent(
        chat_model=chat_model,
        system_prompt=system_prompt,
        retriever=retriever,  
        max_history=3
    )
    
    moderation_service = ModerationService(model=moderation_model)
    
    logger.info("✅ AI models and services initialized")
    
    # =========================================================================
    # 3. Create Event Handlers
    # =========================================================================
    logger.info("Creating event handlers...")
    
    # PM Reply Handler (with RAG)
    pm_handler = PMReplyHandler(
        agent=chat_agent,
        # monitored_users=monitored_users_ids,
        escalation_chat_id=MODERATE_ID_CHAT
    )
    
    # Group Moderation Handler
    group_moderation_handler = GroupModerationHandler(
        service=moderation_service,
        # monitored_groups=monitored_groups_ids,
        send_logs_to=LOGS_ID_CHAT
    )
    
    # Register handlers in router
    router = EventRouter()
    router.add_handler(pm_handler)
    router.add_handler(group_moderation_handler)
    
    logger.info("✅ Handlers registered: PM Reply, Group Moderation")
    
    # =========================================================================
    # 4. Setup Telegram Clients
    # =========================================================================
    logger.info("Setting up Telegram clients...")
    
    manager = TelegramClientManager()
    account_files = get_account_files(FOLDER_ACCOUNTS)
    
    for account_file in account_files:
        config = load_tdlib_account(account_file, library_path=LIBRARY_PATH)
        client = TDLibClient(config)
        manager.add_client(config.name, client)
    
    # Start clients and connect router
    manager.start_all()
    
    for name, client in manager.clients.items():
        client.listen(router)
        logger.info(f"✅ Client '{name}' connected to router")
    
    # =========================================================================
    # 5. Run Bot
    # =========================================================================
    logger.info("="*20)
    logger.info("🤖 Telegram AI Agent Bot is running")
    logger.info(f"📚 Knowledge Base: {indexed_count} chunks")
    logger.info(f"🛡️  Group Moderation: Active")
    logger.info(f"💬 PM Replies: Active (RAG-enabled)")
    logger.info("Press Ctrl+C to stop")
    logger.info("="*20)
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("="*20)
        logger.info("Stopping all clients...")
        manager.stop_all()
        logger.info("✅ Program terminated.")
        logger.info("="*20)


if __name__ == '__main__':
    main()