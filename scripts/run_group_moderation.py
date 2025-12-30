import time
import logging

from services.tg.client.manager import TelegramClientManager
from services.tg.client import TDLibClient
from services.tg.utils import load_tdlib_account
from services.tg.events.router import EventRouter

# Handlers
from services.tg.events.handlers.moderation import GroupModerationHandler

# Services
from services.ai.moderation.service import ModerationService

# Models
from services.ai.moderation.openai import OpenAIModerationModel

from config import LIBRARY_PATH, FOLDER_ACCOUNTS, OPENAI_API_KEY
from utils.files import get_account_files

logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

def main():
    # 1. Initialize AI models
    openai_api_key = OPENAI_API_KEY
    
    openai_moderation_model = OpenAIModerationModel(api_key=openai_api_key)
    
    # 2. Initialize services
    moderation_service = ModerationService(model=openai_moderation_model)
    
    # 3. Create handlers
    group_moderation_handler = GroupModerationHandler(moderation_service, )
    
    # 2. Router
    router = EventRouter()
    
    # 4. Create router and register handlers
    router = EventRouter()
    router.add_handler(group_moderation_handler)

    # 5. Setup clients
    manager = TelegramClientManager()
    account_files = get_account_files(FOLDER_ACCOUNTS)
    
    for account_file in account_files:
        config = load_tdlib_account(account_file, library_path=LIBRARY_PATH)
        client = TDLibClient(config)
        manager.add_client(config.name, client)

    # 6. Start clients and connect router
    manager.start_all()
    
    for name, client in manager.clients.items():
        client.listen(router)
    
    # 7. Run
    logger.info("Bot is running. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Stopping all clients...")
        manager.stop_all()
        logger.info("Program terminated.")

if __name__ == "__main__":
    main()