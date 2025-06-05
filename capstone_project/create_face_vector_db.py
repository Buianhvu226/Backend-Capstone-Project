import chromadb
from django.conf import settings
from pathlib import Path

# Đường dẫn lưu DB
# persist_directory = "F:\\Capstone-Project\\BE\\capstone_project\\chroma_db_face"
persist_directory = Path(settings.BASE_DIR) / "chroma_db_face"
# Khởi tạo client Chroma theo chuẩn mới
client = chromadb.PersistentClient(path=persist_directory)

# Tạo hoặc lấy collection 'face_report'
collection_name = "face_report"
collection = client.get_or_create_collection(name=collection_name)

print(f"✅ Collection '{collection_name}' đã được tạo tại: {persist_directory}")

# import chromadb

# client = chromadb.PersistentClient(path="F:/Capstone-Project/BE/capstone_project/chroma_db_face")
# client.delete_collection("face_report")
# print("✅ Đã xóa collection face_report")

# # Tạo lại (nó sẽ tự nhận vector 512 từ lần thêm đầu tiên)
# collection = client.get_or_create_collection("face_report")
# print("✅ Collection mới đã sẵn sàng")

