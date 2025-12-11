"""
Script để import dữ liệu từ file JSONL vào Qdrant.
Chạy: python vector_search/import_to_qdrant.py
"""
import json
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"
if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH)
    print(f"Đã load file .env từ: {ENV_PATH}")
else:
    print(f"Không tìm thấy file .env tại: {ENV_PATH}")
    print("Sử dụng biến môi trường hệ thống hoặc giá trị mặc định")

# Import trực tiếp từ package qdrant_client (không phải file local)
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# Cấu hình
JSONL_PATH = BASE_DIR / "exports" / "chroma_pinecone_export.jsonl"
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "missing_people_profiles")

# Qdrant connection - ưu tiên Cloud, sau đó mới dùng Local
QDRANT_URL = os.getenv("QDRANT_URL")  # Cloud: https://xxxxx.qdrant.io
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")  # Cloud API key
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))

# Debug: Hiển thị cấu hình
print(f"\n=== Cấu hình Qdrant ===")
if QDRANT_URL:
    print(f"Mode: Cloud")
    print(f"URL: {QDRANT_URL}")
    print(f"API Key: {'*' * 10 if QDRANT_API_KEY else 'CHƯA CẤU HÌNH'}")
else:
    print(f"Mode: Local")
    print(f"Host: {QDRANT_HOST}:{QDRANT_PORT}")
print(f"Collection: {COLLECTION_NAME}")
print("=" * 30 + "\n")

def get_qdrant_client():
    """Tạo Qdrant client"""
    if QDRANT_URL:
        # Cloud Qdrant
        return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    else:
        # Local Qdrant
        return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

def import_jsonl_to_qdrant():
    """Import dữ liệu từ JSONL vào Qdrant"""
    print(f"Đang đọc file: {JSONL_PATH}")
    
    if not JSONL_PATH.exists():
        print(f"Lỗi: Không tìm thấy file {JSONL_PATH}")
        return
    
    # Kết nối Qdrant
    print("Đang kết nối Qdrant...")
    client = get_qdrant_client()
    
    # Đọc file và xác định dimension
    print("Đang đọc file để xác định dimension...")
    first_line = True
    dimension = None
    
    with open(JSONL_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                dimension = len(data['values'])
                break
    
    if dimension is None:
        print("Lỗi: Không thể xác định dimension từ file")
        return
    
    print(f"Dimension: {dimension}")
    
    # Tạo collection nếu chưa có
    try:
        collections = client.get_collections().collections
        collection_exists = any(c.name == COLLECTION_NAME for c in collections)
        
        if collection_exists:
            print(f"Collection '{COLLECTION_NAME}' đã tồn tại. Xóa và tạo lại...")
            client.delete_collection(COLLECTION_NAME)
        
        print(f"Đang tạo collection '{COLLECTION_NAME}' với dimension {dimension}...")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=dimension,
                distance=Distance.COSINE
            )
        )
        print(f"Đã tạo collection '{COLLECTION_NAME}' thành công!")
    except Exception as e:
        print(f"Lỗi khi tạo collection: {e}")
        return
    
    # Import dữ liệu
    print("Đang import dữ liệu...")
    batch_size = 100
    points = []
    total_imported = 0
    
    with open(JSONL_PATH, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            
            try:
                data = json.loads(line)
                
                # Convert ID sang integer (Qdrant yêu cầu integer hoặc UUID, không chấp nhận string số)
                point_id = data['id']
                try:
                    # Thử convert sang integer
                    point_id = int(point_id)
                except (ValueError, TypeError):
                    # Nếu không convert được, giữ nguyên (có thể là UUID)
                    pass
                
                # Tạo point
                point = PointStruct(
                    id=point_id,
                    vector=data['values'],
                    payload=data.get('metadata', {})
                )
                points.append(point)
                
                # Upsert theo batch
                if len(points) >= batch_size:
                    try:
                        client.upsert(
                            collection_name=COLLECTION_NAME,
                            points=points
                        )
                        total_imported += len(points)
                        print(f"Đã import {total_imported} records...")
                        points = []
                    except Exception as e:
                        print(f"\n❌ LỖI KHI UPSERT BATCH (dòng {line_num - batch_size + 1} đến {line_num}):")
                        print(f"Chi tiết: {e}")
                        print(f"\nDừng import ngay lập tức!")
                        raise  # Dừng ngay lập tức
            
            except json.JSONDecodeError as e:
                print(f"\n❌ LỖI PARSE JSON ở dòng {line_num}: {e}")
                print(f"Dừng import ngay lập tức!")
                raise  # Dừng ngay lập tức
            except Exception as e:
                print(f"\n❌ LỖI ở dòng {line_num}: {e}")
                print(f"Dừng import ngay lập tức!")
                raise  # Dừng ngay lập tức
        
        # Upsert phần còn lại
        if points:
            try:
                client.upsert(
                    collection_name=COLLECTION_NAME,
                    points=points
                )
                total_imported += len(points)
            except Exception as e:
                print(f"\n❌ LỖI KHI UPSERT BATCH CUỐI CÙNG:")
                print(f"Chi tiết: {e}")
                print(f"\nDừng import ngay lập tức!")
                raise  # Dừng ngay lập tức
    
    print(f"\n✅ Hoàn thành! Đã import {total_imported} records vào Qdrant.")
    
    # Kiểm tra số lượng (bỏ qua lỗi nếu có do version mismatch)
    try:
        collection_info = client.get_collection(COLLECTION_NAME)
        print(f"✅ Số lượng vectors trong collection: {collection_info.points_count}")
    except Exception as e:
        print(f"⚠️  Không thể kiểm tra số lượng vectors (có thể do version mismatch): {e}")
        print(f"✅ Nhưng import đã hoàn thành với {total_imported} records!")

if __name__ == "__main__":
    import_jsonl_to_qdrant()

