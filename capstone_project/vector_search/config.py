import os
from django.conf import settings
from pathlib import Path
from dotenv import load_dotenv
import django

# --- API Keys Configuration ---
# Tải biến môi trường từ file .env ở thư mục gốc dự án (nếu có)
ENV_PATH = Path(settings.BASE_DIR) / ".env"
load_dotenv(dotenv_path=ENV_PATH, override=False)

# Không hardcode khóa; bắt buộc lấy từ biến môi trường
PRIMARY_GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

def _collect_gemini_keys(max_index=50):
    keys = []
    names = ["GOOGLE_API_KEY"] + [f"GOOGLE_API_KEY_{i}" for i in range(1, max_index + 1)]
    seen = set()
    for name in names:
        value = os.getenv(name)
        if value and value not in seen:
            keys.append(value)
            seen.add(value)
    return keys

GEMINI_API_KEYS = _collect_gemini_keys()

# --- ChromaDB and Embedding Configuration ---
# F:\Capstone-Project\BE\capstone_project\chroma_db_store
# CHROMA_PERSIST_PATH = "F:\\Capstone-Project\\BE\\capstone_project\\chroma_db_store"
CHROMA_PERSIST_PATH = str(Path(settings.BASE_DIR) / "chroma_db_store")

CHROMA_COLLECTION_NAME = "missing_people_profiles"
EMBEDDING_MODEL_NAME = "models/text-embedding-004"
DETAIL_COLUMN_NAME = "Chi tiet_merged"

# --- LLM Configuration ---
BATCH_SIZE_LLM = 100
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