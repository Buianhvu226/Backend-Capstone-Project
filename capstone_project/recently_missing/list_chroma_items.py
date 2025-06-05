import chromadb
from django.conf import settings
from pathlib import Path

# Cấu hình
# CHROMA_DB_PATH = "F:/Capstone-Project/BE/capstone_project/chroma_db_face"
CHROMA_DB_PATH = Path(settings.BASE_DIR) / 'chroma_db_face'
COLLECTION_NAME = "face_report"

def connect_to_chroma():
    return chromadb.PersistentClient(path=CHROMA_DB_PATH)

def list_items(collection):
    results = collection.get()
    print("\n📦 DANH SÁCH ITEM TRONG COLLECTION:")
    for i in range(len(results['ids'])):
        print(f"\n--- Item {i+1} ---")
        print(f"ID: {results['ids'][i]}")
        if 'metadatas' in results and results['metadatas']:
            print(f"Metadata: {results['metadatas'][i]}")
        if 'documents' in results and results['documents']:
            print(f"Document: {results['documents'][i]}")
        if 'embeddings' in results and results['embeddings']:
            print(f"Embedding (rút gọn): {results['embeddings'][i][:5]}...")
    print(f"\n✅ Tổng cộng: {len(results['ids'])} item(s).")

def delete_item_by_id(collection, item_id):
    try:
        collection.delete(ids=[item_id])
        print(f"🗑️ Đã xóa item có ID: {item_id}")
    except Exception as e:
        print(f"❌ Lỗi khi xóa item: {str(e)}")

if __name__ == "__main__":
    client = connect_to_chroma()
    collection = client.get_or_create_collection(COLLECTION_NAME)

    while True:
        print("\n--- QUẢN LÝ CHROMA DB ---")
        print("1. Xem danh sách items")
        print("2. Xóa item theo ID")
        print("0. Thoát")
        choice = input("👉 Chọn thao tác: ")

        if choice == "1":
            list_items(collection)
        elif choice == "2":
            item_id = input("🔍 Nhập ID cần xóa: ").strip()
            delete_item_by_id(collection, item_id)
        elif choice == "0":
            break
        else:
            print("❗ Lựa chọn không hợp lệ. Thử lại.")
