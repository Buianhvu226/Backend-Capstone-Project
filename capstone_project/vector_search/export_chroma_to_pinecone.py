"""
Export toàn bộ vector từ ChromaDB hiện tại ra JSONL để chuẩn bị nhập Pinecone.
Không thay đổi dữ liệu/logic hiện hữu; chỉ đọc và ghi ra file export.

Output: ./exports/chroma_pinecone_export.jsonl
  - Mỗi dòng: {"id": "...", "values": [...], "metadata": {...}, "document": "..."}
"""

import json
from pathlib import Path

import chromadb

try:
    from chromadb.config import Settings
except ImportError:
    Settings = None  # chromadb 0.3.x có thể không cần Settings

# Không phụ thuộc Django settings để tránh lỗi cấu hình khi chạy module trực tiếp
BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_PERSIST_PATH = BASE_DIR / "chroma_db_store"
CHROMA_COLLECTION_NAME = "missing_people_profiles"


OUTPUT_DIR = Path("exports")
OUTPUT_FILE = OUTPUT_DIR / "chroma_pinecone_export.jsonl"
BATCH_SIZE = 500


def get_client():
    """
    Tạo client tương thích cả chromadb 0.3.x và các phiên bản mới hơn.
    Ưu tiên PersistentClient, fallback sang Client với Settings.
    """
    try:
        return chromadb.PersistentClient(path=str(CHROMA_PERSIST_PATH))
    except AttributeError:
        if Settings is None:
            return chromadb.Client()
        return chromadb.Client(
            settings=Settings(
                persist_directory=str(CHROMA_PERSIST_PATH),
                chroma_db_impl="duckdb+parquet",
            )
        )


def export_chroma():
    OUTPUT_DIR.mkdir(exist_ok=True)

    client = get_client()
    collection = client.get_or_create_collection(name=CHROMA_COLLECTION_NAME)
    total = collection.count()
    print(f"Collection: {CHROMA_COLLECTION_NAME} | Total records: {total}")

    written = 0
    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        offset = 0
        while True:
            batch = collection.get(
                include=["embeddings", "metadatas", "documents"],
                limit=BATCH_SIZE,
                offset=offset,
            )
            ids = batch.get("ids", [])
            if not ids:
                break

            embeddings = batch.get("embeddings", [])
            metadatas = batch.get("metadatas", [])
            documents = batch.get("documents", [])

            for i, item_id in enumerate(ids):
                # Chuẩn hóa embedding về list (np.ndarray -> list)
                emb = embeddings[i] if i < len(embeddings) else []
                try:
                    emb = emb.tolist() if hasattr(emb, "tolist") else list(emb)
                except Exception:
                    emb = []

                meta = metadatas[i] if i < len(metadatas) else {}
                doc = documents[i] if i < len(documents) else None

                record = {
                    "id": item_id,
                    "values": emb,
                    "metadata": meta,
                    "document": doc,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1

            offset += len(ids)
            print(f"Đã xuất {written}/{total} bản ghi...")

    print(f"Hoàn tất. Đã ghi {written} bản ghi vào {OUTPUT_FILE.resolve()}")


if __name__ == "__main__":
    export_chroma()

