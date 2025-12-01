from pathlib import Path
from django.conf import settings
import os

PRIMARY_GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "AIzaSyCN_flhR6pXNOvQWjZSMAwe_t1DnI_O8IM")
GEMINI_API_KEYS = [
    PRIMARY_GOOGLE_API_KEY,
    os.getenv("GOOGLE_API_KEY_1", "AIzaSyDw2a1VhB3MXps3ldFUMyYvi65OTIMqFfM"),
    os.getenv("GOOGLE_API_KEY_2", "AIzaSyDats92Eac1yPpk4Z9soGf4nCCiBTh1P64"),
    os.getenv("GOOGLE_API_KEY_3", "AIzaSyBVEQzc89kQ1072ji4xR9wMPtBlzvqCIlY"),
    os.getenv("GOOGLE_API_KEY_4", "AIzaSyBCAqTBZSg7wXK_Jg-JnXW0rZkRJ-VRU64"),
    os.getenv("GOOGLE_API_KEY_5", "AIzaSyDoT41uDC4u212LEnJPS0BPmKKjI4QyWZA"),
    os.getenv("GOOGLE_API_KEY_6", "AIzaSyATDdCrhGScStHL5lIYoebslaPiKrOCyeg"),
    os.getenv("GOOGLE_API_KEY_7", "AIzaSyDrlorbNo7shE5rWi5Gtx2RuJIIuMz4Sn8"),
    os.getenv("GOOGLE_API_KEY_8", "AIzaSyA9K5ZoNuE0Iobdj8VHgM7QstL9s86m3OM"),
    os.getenv("GOOGLE_API_KEY_9", "AIzaSyBwMxv58Vhd8XmW_H3nLddOTl-q0riefw4"),
    os.getenv("GOOGLE_API_KEY_10", "AIzaSyAGGJVO1u8BbyitKJCW4_Q6QhI1xzzmRM4"),
    os.getenv("GOOGLE_API_KEY_11", "AIzaSyDAlOtNLb4WyqCNZKPCTHXOkd_yQ67WVCM"),
    os.getenv("GOOGLE_API_KEY_12", "AIzaSyAIlwECvROhT76rmCQYsKX8h9IOJg9jmlQ"),
    os.getenv("GOOGLE_API_KEY_13", "AIzaSyBgTxPnQ_Sesi3n2SPEDagVpZyzmXw6e6s"),
]
GEMINI_API_KEYS = [key for key in GEMINI_API_KEYS if key]

CHROMA_PERSIST_PATH = Path(settings.BASE_DIR) / "chroma_db_store"
CHROMA_PERSIST_PATH.mkdir(parents=True, exist_ok=True)

CHROMA_COLLECTION_NAME = "missing_people_profiles"
EMBEDDING_MODEL_NAME = "models/text-embedding-004"
DETAIL_COLUMN_NAME = "Chi tiet_merged"

BATCH_SIZE_LLM = 3
MAX_CONCURRENT_REQUESTS_LLM = len(GEMINI_API_KEYS)
MAX_RETRIES_LLM = 5
INITIAL_RETRY_DELAY_LLM = 5
BATCH_GROUP_DELAY_LLM = 2

