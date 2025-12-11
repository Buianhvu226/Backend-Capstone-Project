"""
Qdrant client helper để thay thế ChromaDB và Pinecone
"""
import os
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

# Cấu hình từ environment variables
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_URL = os.getenv("QDRANT_URL")  # Cloud: https://xxxxx.qdrant.io
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")  # Cloud API key
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "missing_people_profiles")

_qdrant_client = None
_qdrant_collection = None

def get_qdrant_client():
    """Lấy Qdrant client (singleton)"""
    global _qdrant_client
    if _qdrant_client is None:
        if QDRANT_URL:
            # Cloud Qdrant
            _qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
            print(f"Đã kết nối Qdrant Cloud: {QDRANT_URL}")
        else:
            # Local Qdrant
            _qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
            print(f"Đã kết nối Qdrant Local: {QDRANT_HOST}:{QDRANT_PORT}")
    return _qdrant_client

def get_qdrant_collection():
    """Lấy collection Qdrant (singleton) - chỉ trả về tên collection, không kiểm tra info"""
    global _qdrant_collection
    if _qdrant_collection is None:
        client = get_qdrant_client()
        # Kiểm tra collection có tồn tại không (bỏ qua lỗi version mismatch)
        try:
            # Thử kiểm tra collection có tồn tại không bằng cách list collections
            collections = client.get_collections()
            collection_exists = any(c.name == QDRANT_COLLECTION_NAME for c in collections.collections)
            if collection_exists:
                _qdrant_collection = QDRANT_COLLECTION_NAME
                print(f"Đã xác nhận collection '{QDRANT_COLLECTION_NAME}' tồn tại.")
            else:
                print(f"Collection '{QDRANT_COLLECTION_NAME}' chưa tồn tại. Vui lòng tạo collection trước.")
                return None
        except Exception as e:
            # Bỏ qua lỗi version mismatch, vẫn trả về collection name nếu có thể
            print(f"⚠️  Không thể kiểm tra collection info (có thể do version mismatch): {e}")
            print(f"⚠️  Vẫn sử dụng collection '{QDRANT_COLLECTION_NAME}' (giả định đã tồn tại).")
            _qdrant_collection = QDRANT_COLLECTION_NAME
    return _qdrant_collection

def initialize_qdrant():
    """Khởi tạo Qdrant (tương tự initialize_vector_db cho ChromaDB)"""
    collection_name = get_qdrant_collection()
    if collection_name:
        return get_qdrant_client()
    return None

