import chromadb
from chromadb.utils import embedding_functions
from django.conf import settings
from pathlib import Path

# --- Config (copy từ gem_vectorDB.py) ---
# CHROMA_PERSIST_PATH = "F:\\Capstone-Project\\BE\\capstone_project\\chroma_db_store"
CHROMA_PERSIST_PATH = Path(settings.BASE_DIR) / "chroma_db_store"
CHROMA_COLLECTION_NAME = "missing_people_profiles"
EMBEDDING_MODEL_NAME = "models/text-embedding-004"
PRIMARY_GOOGLE_API_KEY = "AIzaSyCN_flhR6pXNOvQWjZSMAwe_t1DnI_O8IM"

def initialize_vector_db():
    """Khởi tạo ChromaDB client và collection."""
    print(f"Khởi tạo ChromaDB client với đường dẫn lưu trữ: {CHROMA_PERSIST_PATH}")
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_PATH)
    google_ef = embedding_functions.GoogleGenerativeAiEmbeddingFunction(
        api_key=PRIMARY_GOOGLE_API_KEY,
        model_name=EMBEDDING_MODEL_NAME
    )
    print(f"Sử dụng embedding model: {EMBEDDING_MODEL_NAME}")
    try:
        collection = client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            embedding_function=google_ef,
            metadata={"hnsw:space": "cosine"}
        )
        print(f"Đã kết nối/tạo collection '{CHROMA_COLLECTION_NAME}' thành công.")
        print(f"Số lượng hồ sơ hiện có trong collection: {collection.count()}")
        return collection
    except Exception as e:
        print(f"Lỗi khi kết nối/tạo collection ChromaDB: {e}")
        exit()

def list_profiles(limit=20):
    collection = initialize_vector_db()
    # Correct way: get all and access 'ids'
    all_ids = collection.get()['ids']
    print(f"Tổng số profile trong ChromaDB: {len(all_ids)}")
    print(f"Hiển thị {min(limit, len(all_ids))} profile đầu tiên:")
    for pid in all_ids[:limit]:
        print(f"- ID: {pid}")

def show_profile_detail(profile_id):
    collection = initialize_vector_db()
    result = collection.get(ids=[str(profile_id)], include=['metadatas'])
    if result['metadatas']:
        print(f"Metadata cho profile {profile_id}:")
        print(result['metadatas'][0])
    else:
        print(f"Không tìm thấy profile với ID {profile_id} trong ChromaDB.")

def delete_profiles(profile_ids):
    collection = initialize_vector_db()
    str_ids = [str(pid) for pid in profile_ids]
    collection.delete(ids=str_ids)
    print(f"Đã xóa các profile với ID: {str_ids} khỏi ChromaDB.")

if __name__ == "__main__":
    while True:
        print("Chọn thao tác:")
        print("1. Liệt kê profile")
        print("2. Xem chi tiết profile")
        print("3. Xóa profile")
        print("q. Thoát chương trình")
        choice = input("Nhập lựa chọn (1/2/3/q): ").strip()
        if choice == "1":
            limit = input("Số lượng profile muốn xem (mặc định 20): ").strip()
            limit = int(limit) if limit.isdigit() else 20
            list_profiles(limit)
        elif choice == "2":
            pid = input("Nhập ID profile muốn xem: ").strip()
            show_profile_detail(pid)
        elif choice == "3":
            ids = input("Nhập các ID profile muốn xóa (cách nhau bởi dấu phẩy): ").strip()
            id_list = [i.strip() for i in ids.split(",") if i.strip()]
            delete_profiles(id_list)
        elif choice.lower() == "q":
            print("Đã thoát chương trình.")
            break
        else:
            print("Lựa chọn không hợp lệ.")


# import chromadb
# from chromadb.config import Settings

# # Cấu hình
# CHROMA_PERSIST_PATH = "F:\\Capstone-Project\\BE\\capstone_project\\chroma_db_store"
# CHROMA_COLLECTION_NAME = "missing_people_profiles"
# BATCH_SIZE = 100  # Số lượng item mỗi batch

# # Khởi tạo Chroma client
# client = chromadb.Client(Settings(
#     persist_directory=CHROMA_PERSIST_PATH,
#     anonymized_telemetry=False
# ))

# # Lấy collection
# collection = client.get_collection(name=CHROMA_COLLECTION_NAME)

# # Lấy tổng số lượng item
# count = collection.count()
# print(f"Tổng số item trong collection: {count}")

# # Duyệt từng batch và hiển thị đầy đủ thông tin
# for start in range(0, count, BATCH_SIZE):
#     print(f"\n--- Batch {start // BATCH_SIZE + 1} ---\n")

#     # Truy vấn theo batch, dùng limit + offset
#     results = collection.get(
#         include=["embeddings", "documents", "metadatas"],
#         offset=start,
#         limit=BATCH_SIZE
#     )

#     # Hiển thị từng item
#     for idx, item_id in enumerate(results["ids"]):
#         print(f"Item {start + idx + 1}:")
#         print(f"  ID: {item_id}")
#         print(f"  Document: {results['documents'][idx]}")
#         print(f"  Metadata: {results['metadatas'][idx]}")
#         print(f"  Embedding (first 10 dims): {results['embeddings'][idx][:10]}")  # In gọn phần đầu embedding
#         print("-" * 50)
