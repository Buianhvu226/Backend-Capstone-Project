import os
from django.conf import settings
from pathlib import Path

# --- API Keys Configuration ---
PRIMARY_GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "AIzaSyCN_flhR6pXNOvQWjZSMAwe_t1DnI_O8IM")
GEMINI_API_KEYS = [
    PRIMARY_GOOGLE_API_KEY,
    os.getenv("GOOGLE_API_KEY_1", "AIzaSyA9M1sGANeRrsLWkEJ6oQV26EnhY0l1DIg"),
    os.getenv("GOOGLE_API_KEY_2", "AIzaSyC2q4B7RntxfKXRDMGqyFGj24h_Qd9rXvI"),
    os.getenv("GOOGLE_API_KEY_3", "AIzaSyBVEQzc89kQ1072ji4xR9wMPtBlzvqCIlY"),
    os.getenv("GOOGLE_API_KEY_4", "AIzaSyBCAqTBZSg7wXK_Jg-JnXW0rZkRJ-VRU64"),
    os.getenv("GOOGLE_API_KEY_5", "AIzaSyDoT41uDC4u212LEnJPS0BPmKKjI4QyWZA"),
    os.getenv("GOOGLE_API_KEY_6", "AIzaSyATDdCrhGScStHL5lIYoebslaPiKrOCyeg"),
    os.getenv("GOOGLE_API_KEY_7", "AIzaSyDrlorbNo7shE5rWi5Gtx2RuJIIuMz4Sn8"),
    os.getenv("GOOGLE_API_KEY_8", "AIzaSyA9K5ZoNuE0Iobdj8VHgM7QstL9s86m3OM"),
    os.getenv("GOOGLE_API_KEY_9", "AIzaSyBwMxv58Vhd8XmW_H3nLddOTl-q0riefw4"),
    os.getenv("GOOGLE_API_KEY_10", "AIzaSyAGGJVO1u8BbyitKJCW4_Q6QhI1xzzmRM4"),
    os.getenv("GOOGLE_API_KEY_11", "AIzaSyDats92Eac1yPpk4Z9soGf4nCCiBTh1P64"),
    os.getenv("GOOGLE_API_KEY_12", "AIzaSyAIlwECvROhT76rmCQYsKX8h9IOJg9jmlQ"),
    os.getenv("GOOGLE_API_KEY_13", "AIzaSyBgTxPnQ_Sesi3n2SPEDagVpZyzmXw6e6s"),
    os.getenv("GOOGLE_API_KEY_14", "AIzaSyDyOtYSzMUfg49MyRJH2_5I9kQKbX-tDiY"),
    os.getenv("GOOGLE_API_KEY_15", "AIzaSyBtL_H3W6KDLHoda_afc3dM7Vjo5f6CjO0"),
    os.getenv("GOOGLE_API_KEY_16", "AIzaSyDEH482LFeiUIU8nb0FkhW_oAWnUivU6eQ"),
    os.getenv("GOOGLE_API_KEY_17", "AIzaSyCvg9pOxGF2q5p1dDxU2KfvQfhuohWt3I0"),
    os.getenv("GOOGLE_API_KEY_18", "AIzaSyDkUH1itL-dR1JbEhxP55xAbuRXIhCke7Y"),
    os.getenv("GOOGLE_API_KEY_19", "AIzaSyDAlOtNLb4WyqCNZKPCTHXOkd_yQ67WVCM"),
    os.getenv("GOOGLE_API_KEY_20", "AIzaSyDw2a1VhB3MXps3ldFUMyYvi65OTIMqFfM"),
    os.getenv("GOOGLE_API_KEY_21", "AIzaSyB2Em85QndktYfOeGD5rTkft0J-i_ATauc"),
]
GEMINI_API_KEYS = [key for key in GEMINI_API_KEYS if key]  # Loại bỏ key rỗng

# --- ChromaDB and Embedding Configuration ---
# F:\Capstone-Project\BE\capstone_project\chroma_db_store
# CHROMA_PERSIST_PATH = "F:\\Capstone-Project\\BE\\capstone_project\\chroma_db_store"
CHROMA_PERSIST_PATH = str(Path(settings.BASE_DIR) / "chroma_db_store")

CHROMA_COLLECTION_NAME = "missing_people_profiles"
EMBEDDING_MODEL_NAME = "models/text-embedding-004"
DETAIL_COLUMN_NAME = "Chi tiet_merged"

# --- LLM Configuration ---
BATCH_SIZE_LLM = 200
# MAX_CONCURRENT_REQUESTS_LLM = len(GEMINI_API_KEYS)
MAX_CONCURRENT_REQUESTS_LLM = 1
MAX_RETRIES_LLM = 1
INITIAL_RETRY_DELAY_LLM = 5  # Giây
BATCH_GROUP_DELAY_LLM = 2  # Có thể giảm delay này vì đang dùng nhiều key

# --- Django Integration ---
# Setup Django environment if running as a standalone script
import sys
import os
from pathlib import Path

# Add project path to sys.path
# sys.path.append("F:\\Capstone-Project\\BE\\capstone_project")
# os.environ.setdefault("DJANGO_SETTINGS_MODULE", "capstone_project.settings")
# Tự động thêm project path vào sys.path dựa trên vị trí file hiện tại
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR / "capstone_project"))

# Thiết lập biến môi trường cho Django
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "capstone_project.settings")

# Initialize Django
try:
    django.setup()
except Exception as e:
    print(f"Warning: Could not initialize Django: {e}")