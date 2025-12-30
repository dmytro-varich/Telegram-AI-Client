import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Common path to TDLib
LIBRARY_PATH = str(os.getenv("LIBRARY_PATH", ""))

# Folders
FOLDER_ACCOUNTS = 'accounts'
FOLDER_PROXIES = 'proxies'
FOLDER_DATA = 'data'
FOLDER_LOGS = 'logs'

# Monitored Users/Groups
MONITORED_USERS_FILE = 'data/monitored_users.csv'
MONITORED_GROUPS_FILE = 'data/monitored_groups.csv'

# API Keys 
OPENAI_API_KEY = str(os.getenv("OPENAI_API_KEY", ""))

# AI Models 
OPENAI_CHAT_MODEL = 'gpt-4o-mini'

# Prompt Files
SYSTEM_PROMPT_FILE = 'config/prompts/private_kb_responder.txt'

# Knowledge Base Settings
PDF_KNOWLEDGE_BASE = os.path.join(FOLDER_DATA, 'knowledge_base.pdf')
EMBEDDING_MODEL = 'text-embedding-3-small'

# API KEY FOR DETECT LANGUAGE
DETECT_LANG_KEY = str(os.getenv("DETECT_LANG_KEY", ""))

# Target Ids Chat
LOGS_ID_CHAT = 5058141109
MODERATE_ID_CHAT = 4995500181


