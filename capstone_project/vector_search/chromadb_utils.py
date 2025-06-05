import pandas as pd
from tqdm import tqdm
import time

from .config import DETAIL_COLUMN_NAME
from .embedding import get_embedding

def embed_and_upsert_profiles(df, collection, batch_size_chroma=100):
    """Tạo embedding và upsert dữ liệu vào ChromaDB theo lô."""
    print(f"\nBắt đầu quá trình embedding và upsert vào collection '{collection.name}'...")
    profiles_to_upsert = []
    processed_count = 0
    failed_count = 0  # Đếm số lượng không thể embed

    total_profiles = len(df)
    progress_bar = tqdm(total=total_profiles, desc="Embedding & Upserting")

    for index, row in df.iterrows():
        # Use row['id'] instead of index
        profile_id = int(row['id'])
        text_to_embed = row.get(DETAIL_COLUMN_NAME)
        if pd.isna(text_to_embed) or not isinstance(text_to_embed, str) or not text_to_embed.strip():
            progress_bar.update(1)
            continue  # Bỏ qua nếu dữ liệu không hợp lệ

        # Hàm get_embedding giờ sẽ tự động thử lại mạnh mẽ
        embedding = get_embedding(text_to_embed, task_type="RETRIEVAL_DOCUMENT")

        if embedding:
            # Chuẩn bị metadata - chỉ lưu các kiểu dữ liệu cơ bản
            metadata = {}
            for col in ['Tiêu đề', 'Họ và tên', 'Năm sinh', 'Năm thất lạc']:
                if col in row:
                    value = row[col]
                    if pd.isna(value):
                        metadata[col] = ""
                    elif isinstance(value, (str, int, float, bool)):
                        metadata[col] = value
                    else:
                        metadata[col] = str(value)
            metadata = {k: "" if pd.isna(v) else v for k, v in metadata.items()}

            profiles_to_upsert.append({
                "id": str(index),
                "embedding": embedding,
                "metadata": metadata
            })

            # Upsert theo lô
            if len(profiles_to_upsert) >= batch_size_chroma:
                try:
                    collection.upsert(
                        ids=[p["id"] for p in profiles_to_upsert],
                        embeddings=[p["embedding"] for p in profiles_to_upsert],
                        metadatas=[p["metadata"] for p in profiles_to_upsert]
                    )
                    processed_count += len(profiles_to_upsert)
                    profiles_to_upsert = []
                except Exception as e:
                    print(f"\nLỗi khi upsert batch vào ChromaDB: {e}")
                    # Ghi log lỗi upsert nhưng vẫn tiếp tục
                    failed_count += len(profiles_to_upsert)  # Coi như batch này lỗi
                    profiles_to_upsert = []  # Reset batch lỗi

        else:
            # Nếu get_embedding trả về None (sau khi đã thử lại rất nhiều lần hoặc gặp lỗi fatal)
            print(f"Không thể tạo embedding cho hồ sơ index {index} sau nhiều lần thử. Bỏ qua hồ sơ này.")
            failed_count += 1

        progress_bar.update(1)

    # Upsert phần còn lại
    if profiles_to_upsert:
        try:
            collection.upsert(
                ids=[p["id"] for p in profiles_to_upsert],
                embeddings=[p["embedding"] for p in profiles_to_upsert],
                metadatas=[p["metadata"] for p in profiles_to_upsert]
            )
            processed_count += len(profiles_to_upsert)
        except Exception as e:
            print(f"\nLỗi khi upsert batch cuối cùng vào ChromaDB: {e}")
            failed_count += len(profiles_to_upsert)

    progress_bar.close()
    print(f"\nHoàn thành embedding và upsert.")
    print(f"Tổng cộng {processed_count}/{total_profiles} hồ sơ hợp lệ đã được xử lý và upsert thành công.")
    if failed_count > 0:
        print(f"Cảnh báo: {failed_count} hồ sơ không thể tạo embedding hoặc upsert.")
    print(f"Số lượng hồ sơ cuối cùng trong collection: {collection.count()}")