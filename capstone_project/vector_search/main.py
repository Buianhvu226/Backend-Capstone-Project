import os
import sys
import django
from pathlib import Path

# Setup Django environment
# sys.path.append("F:\\Capstone-Project\\BE\\capstone_project")
# os.environ.setdefault("DJANGO_SETTINGS_MODULE", "capstone_project.settings")
# django.setup()

# Setup Django environment
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR / "capstone_project"))

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "capstone_project.settings")

django.setup()

from .embedding import initialize_vector_db
from .chromadb_utils import embed_and_upsert_profiles
from .search import search_combined_chroma
from .db_utils import fetch_profiles_from_db

def search_loop(df_original, collection):
    """Vòng lặp nhận truy vấn từ người dùng và thực hiện tìm kiếm."""
    while True:
        user_query = input(f"\nNhập mô tả tìm kiếm (hoặc 'q' để thoát): ")
        if user_query.lower() == 'q':
            break
        if not user_query.strip():
            print("Vui lòng nhập mô tả tìm kiếm.")
            continue

        # Gọi hàm tìm kiếm chính
        search_combined_chroma(df_original, collection, user_query)

def main():
    print("Bắt đầu quy trình xử lý hồ sơ và tìm kiếm...")

    # 1. Khởi tạo Vector DB
    collection = initialize_vector_db()
    if collection is None:
        print("Không thể khởi tạo Vector DB. Thoát chương trình.")
        return

    # 2. Đọc dữ liệu từ database
    try:
        df = fetch_profiles_from_db()
        if df.empty:
            print("Không có dữ liệu hồ sơ. Thoát chương trình.")
            return
            
        print(f"Đã đọc dữ liệu từ database ({len(df)} hồ sơ)")
        print("Thông tin DataFrame:")
        df.info()

    except Exception as e:
        print(f"Lỗi không xác định khi đọc dữ liệu từ database: {e}")
        return

    # 3. Kiểm tra và Upsert hồ sơ mới (Nếu người dùng chọn)
    check_and_upsert = input("Bạn có muốn kiểm tra và upsert hồ sơ mới từ database vào Vector DB không? (yes/no): ").lower()
    if check_and_upsert == 'yes':
        current_db_count = 0
        try:
            current_db_count = collection.count()
        except Exception as e:
            print(f"Lỗi khi lấy số lượng hồ sơ từ ChromaDB: {e}. Tiếp tục mà không upsert.")
            current_db_count = -1

        if current_db_count != -1:
            total_db_count = len(df)
            print(f"Số hồ sơ hiện tại trong ChromaDB: {current_db_count}")
            print(f"Số hồ sơ trong database: {total_db_count}")

            if total_db_count > current_db_count:
                num_new_profiles = total_db_count - current_db_count
                print(f"Phát hiện {num_new_profiles} hồ sơ mới trong database.")

                # Lấy các hồ sơ mới cần thêm
                new_profiles_df = df.iloc[current_db_count:]

                print(f"Bắt đầu embedding và upsert {len(new_profiles_df)} hồ sơ mới...")
                embed_and_upsert_profiles(new_profiles_df, collection)
            else:
                print("Không có hồ sơ mới nào trong database cần được thêm vào ChromaDB.")
    else:
        print("Bỏ qua bước kiểm tra và upsert hồ sơ mới.")
        try:
            print(f"Số lượng hồ sơ hiện tại trong collection '{collection.name}': {collection.count()}")
        except Exception as e:
            print(f"Lỗi khi lấy số lượng hồ sơ từ ChromaDB: {e}")

    # 4. Bắt đầu vòng lặp tìm kiếm
    search_loop(df, collection)

    print("\nChương trình đã kết thúc.")

if __name__ == "__main__":
    main()