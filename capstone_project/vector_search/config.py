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

# --- Vector DB Selection ---
# Mặc định dùng Qdrant, có thể chuyển về ChromaDB hoặc Pinecone bằng env vars
USE_QDRANT = os.getenv("USE_QDRANT", "true").lower() == "true"
USE_PINECONE = os.getenv("USE_PINECONE", "false").lower() == "true"
USE_CHROMADB = os.getenv("USE_CHROMADB", "false").lower() == "true"

# --- Qdrant Configuration ---
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_URL = os.getenv("QDRANT_URL")  # Cloud: https://xxxxx.qdrant.io
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")  # Cloud API key
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "missing_people_profiles")

# --- Pinecone Configuration (đã comment - không dùng nữa) ---
# USE_PINECONE = os.getenv("USE_PINECONE", "true").lower() == "true"
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_HOST = os.getenv("PINECONE_INDEX_HOST")  # v3 host dạng https://XXXX-XXXX.svc.XXXX.pinecone.io
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")  # tùy chọn nếu dùng client cần tên
PINECONE_TOP_K = int(os.getenv("PINECONE_TOP_K", "1000"))

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