import time
import pandas as pd
import google.generativeai as genai
import chromadb
from chromadb.utils import embedding_functions
import requests
from google.api_core import exceptions as google_exceptions

from .config import (
    PRIMARY_GOOGLE_API_KEY,
    GEMINI_API_KEYS,
    CHROMA_PERSIST_PATH,
    CHROMA_COLLECTION_NAME,
    EMBEDDING_MODEL_NAME
)

# --- Khởi tạo Google AI ---
try:
    genai.configure(api_key=PRIMARY_GOOGLE_API_KEY)
    print("Đã cấu hình Google AI API Key chính thành công.")
except Exception as e:
    print(f"Lỗi cấu hình Google AI API Key chính: {e}")

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
        return None

def get_embedding(text, task_type, model=EMBEDDING_MODEL_NAME, api_keys=None, 
                 max_wait_time=120, max_consecutive_failures_per_key=3, max_total_attempts=15):
    """Lấy embedding từ Google API với cơ chế thử lại mạnh mẽ và xoay vòng API key."""
    if not isinstance(text, str) or not text.strip() or pd.isna(text):
        print(f"Cảnh báo: Dữ liệu đầu vào không hợp lệ cho embedding: {text[:50]}...")
        return None

    if api_keys is None or not api_keys:
        api_keys = [PRIMARY_GOOGLE_API_KEY]  # Sử dụng key chính nếu không có danh sách

    text = text[:8000]  # Giới hạn độ dài
    total_attempts = 0
    current_key_index = 0
    consecutive_failures_with_current_key = 0
    current_wait_time = 5  # Thời gian chờ ban đầu (giây)

    while total_attempts < max_total_attempts:
        current_api_key = api_keys[current_key_index]
        total_attempts += 1

        try:
            # Cấu hình key hiện tại cho thư viện genai
            genai.configure(api_key=current_api_key)

            result = genai.embed_content(
                model=model,
                content=text,
                task_type=task_type
            )
            return result["embedding"]

        # Lỗi cụ thể từ Google API
        except google_exceptions.ResourceExhausted as e:  # Lỗi Quota (429)
            error_type = "ResourceExhausted (429)"
            consecutive_failures_with_current_key += 1
        except google_exceptions.ServiceUnavailable as e:  # Lỗi Server (503)
            error_type = "ServiceUnavailable (503)"
            consecutive_failures_with_current_key += 1
        except google_exceptions.DeadlineExceeded as e:  # Lỗi Timeout phía Google
            error_type = "DeadlineExceeded"
            consecutive_failures_with_current_key += 1
        except google_exceptions.InternalServerError as e:  # Lỗi Server (500)
            error_type = "InternalServerError (500)"
            consecutive_failures_with_current_key += 1
        except google_exceptions.InvalidArgument as e:  # Lỗi đầu vào (400)
            print(f"Lỗi InvalidArgument khi tạo embedding cho '{text[:50]}...' (Key index {current_key_index}): {e}. Đây có thể là lỗi dữ liệu không hợp lệ. Bỏ qua.")
            return None  # Không thử lại lỗi này
        except google_exceptions.PermissionDenied as e:  # Lỗi API Key hoặc quyền
            error_type = f"PermissionDenied (Key index {current_key_index})"
            print(f"Lỗi API key không hợp lệ hoặc bị từ chối (Key index {current_key_index}): {e}. Chuyển sang key tiếp theo.")
            # Chuyển key ngay lập tức khi gặp lỗi này
            consecutive_failures_with_current_key = max_consecutive_failures_per_key  # Buộc chuyển key

        # Lỗi kết nối mạng chung
        except requests.exceptions.Timeout as e:
            error_type = "Request Timeout"
            consecutive_failures_with_current_key += 1
        except requests.exceptions.ConnectionError as e:
            error_type = "Connection Error"
            consecutive_failures_with_current_key += 1

        # Các lỗi không mong muốn khác
        except Exception as e:
            error_type = f"Unknown Exception ({type(e).__name__})"
            print(f"Lỗi chi tiết khi tạo embedding (Key index {current_key_index}): {e}")
            consecutive_failures_with_current_key += 1

        # Xử lý thử lại và chuyển key
        print(f"Lỗi '{error_type}' khi tạo embedding cho '{text[:50]}...' (Key index {current_key_index}, Thử lại tổng {total_attempts}/{max_total_attempts}).")

        # Kiểm tra nếu cần chuyển key
        if consecutive_failures_with_current_key >= max_consecutive_failures_per_key:
            current_key_index = (current_key_index + 1) % len(api_keys)
            consecutive_failures_with_current_key = 0
            current_wait_time = 5  # Reset thời gian chờ khi chuyển key
            print(f"Đã gặp {max_consecutive_failures_per_key} lỗi liên tiếp. Chuyển sang API key index {current_key_index}.")
            if current_key_index == 0 and total_attempts > len(api_keys):  # Đã thử hết 1 vòng các key
                print("Đã thử hết các API key ít nhất một lần, tăng thời gian chờ...")
            continue  # Bỏ qua phần sleep và thử lại ngay với key mới

        # Nếu chưa chuyển key, thực hiện chờ và tăng thời gian chờ
        print(f"Chờ {current_wait_time} giây trước khi thử lại...")
        time.sleep(current_wait_time)
        current_wait_time = min(current_wait_time * 2, max_wait_time)

    # Nếu thoát khỏi vòng lặp do hết max_total_attempts
    print(f"Không thể tạo embedding cho '{text[:50]}...' sau {max_total_attempts} lần thử với các API key khác nhau.")
    return None