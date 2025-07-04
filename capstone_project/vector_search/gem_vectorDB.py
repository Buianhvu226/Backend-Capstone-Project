import pandas as pd
# from sklearn.metrics.pairwise import cosine_similarity # Không cần trực tiếp nữa
import numpy as np
import os
import time
from tqdm import tqdm
import google.generativeai as genai
import chromadb # Thư viện Vector DB
from chromadb.utils import embedding_functions # Để dùng embedding function của Gemini
import math # Để xử lý NaN
from concurrent.futures import ThreadPoolExecutor
import requests
import json
from google.api_core import exceptions as google_exceptions # Import exceptions của Google

from django.conf import settings
from pathlib import Path

# --- Cấu hình ---
PRIMARY_GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "AIzaSyCN_flhR6pXNOvQWjZSMAwe_t1DnI_O8IM")
GEMINI_API_KEYS = [
    PRIMARY_GOOGLE_API_KEY,
    os.getenv("GOOGLE_API_KEY_1", "AIzaSyDw2a1VhB3MXps3ldFUMyYvi65OTIMqFfM"),
    os.getenv("GOOGLE_API_KEY_2", "AIzaSyDats92Eac1yPpk4Z9soGf4nCCiBTh1P64"),
    os.getenv("GOOGLE_API_KEY_3", "AIzaSyBVEQzc89kQ1072ji4xR9wMPtBlzvqCIlY"),
    os.getenv("GOOGLE_API_KEY_4", "AIzaSyBCAqTBZSg7wXK_Jg-JnXW0rZkRJ-VRU64"),
    os.getenv("GOOGLE_API_KEY_5", "AIzaSyDoT41uDC4u212LEnJPS0BPmKKjI4QyWZA"),
    os.getenv("GOOGLE_API_KEY_6", "AIzaSyATDdCrhGScStHL5lIYoebslaPiKrOCyeg"),
    os.getenv("GOOGLE_API_KEY_7", "AIzaSyDrlorbNo7shE5rWi5Gtx2RuJIIuMz4Sn8"),
    os.getenv("GOOGLE_API_KEY_8", "AIzaSyA9K5ZoNuE0Iobdj8VHgM7QstL9s86m3OM"),
    os.getenv("GOOGLE_API_KEY_9", "AIzaSyBwMxv58Vhd8XmW_H3nLddOTl-q0riefw4"),
    os.getenv("GOOGLE_API_KEY_10", "AIzaSyAGGJVO1u8BbyitKJCW4_Q6QhI1xzzmRM4"),
    os.getenv("GOOGLE_API_KEY_11", "AIzaSyDAlOtNLb4WyqCNZKPCTHXOkd_yQ67WVCM"),
    os.getenv("GOOGLE_API_KEY_12", "AIzaSyAIlwECvROhT76rmCQYsKX8h9IOJg9jmlQ"),
    os.getenv("GOOGLE_API_KEY_13", "AIzaSyBgTxPnQ_Sesi3n2SPEDagVpZyzmXw6e6s"),
]
GEMINI_API_KEYS = [key for key in GEMINI_API_KEYS if key] # Loại bỏ key rỗng

# --- Cấu hình ChromaDB và Embedding ---
# F:\Capstone-Project\BE\capstone_project\chroma_db_store
# CHROMA_PERSIST_PATH = "F:\\Capstone-Project\\BE\\capstone_project\\chroma_db_store"
CHROMA_PERSIST_PATH = Path(settings.BASE_DIR) / "chroma_db_store"
CHROMA_PERSIST_PATH.mkdir(parents=True, exist_ok=True)

CHROMA_COLLECTION_NAME = "missing_people_profiles"  # Tên collection trong ChromaDB
EMBEDDING_MODEL_NAME = "models/text-embedding-004"  # Model embedding của Google
DETAIL_COLUMN_NAME = "Chi tiet_merged"  # Tên cột chứa nội dung chi tiết
# CSV_PATH = "F:\\missing_people(NCHCCCL)\\data\\profiles_detailed_data_final.csv"

# Cấu hình LLM (giữ nguyên hoặc điều chỉnh nếu cần)
BATCH_SIZE_LLM = 3
MAX_CONCURRENT_REQUESTS_LLM = len(GEMINI_API_KEYS) # Tận dụng tối đa số key
MAX_RETRIES_LLM = 5
INITIAL_RETRY_DELAY_LLM = 5 # Giây
BATCH_GROUP_DELAY_LLM = 2 # Có thể giảm delay này vì đang dùng nhiều key

# --- Khởi tạo Google AI ---
try:
    genai.configure(api_key=PRIMARY_GOOGLE_API_KEY)
    print("Đã cấu hình Google AI API Key chính thành công.")
except Exception as e:
    print(f"Lỗi cấu hình Google AI API Key chính: {e}")
    exit()

# --- Khởi tạo ChromaDB Client và Collection ---
def initialize_vector_db():
    """Khởi tạo ChromaDB client và collection."""
    print(f"Khởi tạo ChromaDB client với đường dẫn lưu trữ: {CHROMA_PERSIST_PATH}")
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_PATH)
    google_ef = embedding_functions.GoogleGenerativeAiEmbeddingFunction(api_key=PRIMARY_GOOGLE_API_KEY, model_name=EMBEDDING_MODEL_NAME)
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

# --- Hàm lấy Embedding (Giữ nguyên, nhưng đảm bảo model khớp với ChromaDB) ---
def get_embedding(text, task_type, model=EMBEDDING_MODEL_NAME, api_keys=None, max_wait_time=120, max_consecutive_failures_per_key=3, max_total_attempts=15):
    """Lấy embedding từ Google API với cơ chế thử lại mạnh mẽ và xoay vòng API key."""
    if not isinstance(text, str) or not text.strip() or pd.isna(text):
        print(f"Cảnh báo: Dữ liệu đầu vào không hợp lệ cho embedding: {text[:50]}...")
        return None

    if api_keys is None or not api_keys:
        api_keys = [PRIMARY_GOOGLE_API_KEY] # Sử dụng key chính nếu không có danh sách

    text = text[:8000] # Giới hạn độ dài
    total_attempts = 0
    current_key_index = 0
    consecutive_failures_with_current_key = 0
    current_wait_time = 5 # Thời gian chờ ban đầu (giây)

    while total_attempts < max_total_attempts:
        current_api_key = api_keys[current_key_index]
        total_attempts += 1

        try:
            # Cấu hình key hiện tại cho thư viện genai
            # Lưu ý: Thao tác này ảnh hưởng đến cấu hình toàn cục của genai
            genai.configure(api_key=current_api_key)

            # print(f"Attempt {total_attempts}: Trying key index {current_key_index} for '{text[:30]}...'") # Debug

            result = genai.embed_content(
                model=model,
                content=text,
                task_type=task_type
            )
            # print(f"Thành công embedding với key index {current_key_index} cho: {text[:30]}...") # Debug
            # Reset cấu hình về key chính sau khi thành công (tùy chọn, để tránh ảnh hưởng các phần khác)
            # genai.configure(api_key=PRIMARY_GOOGLE_API_KEY)
            return result["embedding"]

        # Lỗi cụ thể từ Google API
        except google_exceptions.ResourceExhausted as e: # Lỗi Quota (429)
            error_type = "ResourceExhausted (429)"
            consecutive_failures_with_current_key += 1
        except google_exceptions.ServiceUnavailable as e: # Lỗi Server (503)
            error_type = "ServiceUnavailable (503)"
            consecutive_failures_with_current_key += 1
        except google_exceptions.DeadlineExceeded as e: # Lỗi Timeout phía Google
             error_type = "DeadlineExceeded"
             consecutive_failures_with_current_key += 1
        except google_exceptions.InternalServerError as e: # Lỗi Server (500)
             error_type = "InternalServerError (500)"
             consecutive_failures_with_current_key += 1
        except google_exceptions.InvalidArgument as e: # Lỗi đầu vào (400)
             print(f"Lỗi InvalidArgument khi tạo embedding cho '{text[:50]}...' (Key index {current_key_index}): {e}. Đây có thể là lỗi dữ liệu không hợp lệ. Bỏ qua.")
             # Reset cấu hình về key chính
             # genai.configure(api_key=PRIMARY_GOOGLE_API_KEY)
             return None # Không thử lại lỗi này
        except google_exceptions.PermissionDenied as e: # Lỗi API Key hoặc quyền
             error_type = f"PermissionDenied (Key index {current_key_index})"
             print(f"Lỗi API key không hợp lệ hoặc bị từ chối (Key index {current_key_index}): {e}. Chuyển sang key tiếp theo.")
             # Chuyển key ngay lập tức khi gặp lỗi này
             consecutive_failures_with_current_key = max_consecutive_failures_per_key # Buộc chuyển key

        # Lỗi kết nối mạng chung
        except requests.exceptions.Timeout as e:
            error_type = "Request Timeout"
            consecutive_failures_with_current_key += 1
        except requests.exceptions.ConnectionError as e:
            error_type = "Connection Error"
            consecutive_failures_with_current_key += 1

        # Các lỗi không mong muốn khác (bao gồm cả ValueError bạn gặp)
        except Exception as e:
            error_type = f"Unknown Exception ({type(e).__name__})"
            # In lỗi cụ thể để chẩn đoán ValueError
            print(f"Lỗi chi tiết khi tạo embedding (Key index {current_key_index}): {e}")
            consecutive_failures_with_current_key += 1

        # Xử lý thử lại và chuyển key
        print(f"Lỗi '{error_type}' khi tạo embedding cho '{text[:50]}...' (Key index {current_key_index}, Thử lại tổng {total_attempts}/{max_total_attempts}).")

        # Kiểm tra nếu cần chuyển key
        if consecutive_failures_with_current_key >= max_consecutive_failures_per_key:
            current_key_index = (current_key_index + 1) % len(api_keys)
            consecutive_failures_with_current_key = 0
            current_wait_time = 5 # Reset thời gian chờ khi chuyển key
            print(f"Đã gặp {max_consecutive_failures_per_key} lỗi liên tiếp. Chuyển sang API key index {current_key_index}.")
            if current_key_index == 0 and total_attempts > len(api_keys): # Đã thử hết 1 vòng các key
                 print("Đã thử hết các API key ít nhất một lần, tăng thời gian chờ...")
                 # Có thể tăng thời gian chờ cơ bản nếu muốn sau mỗi vòng lặp key
                 # current_wait_time = min(current_wait_time * 1.5, 30) # Ví dụ
            # Không sleep ngay, để vòng lặp tiếp theo thử key mới
            continue # Bỏ qua phần sleep và thử lại ngay với key mới

        # Nếu chưa chuyển key, thực hiện chờ và tăng thời gian chờ
        print(f"Chờ {current_wait_time} giây trước khi thử lại...")
        time.sleep(current_wait_time)
        current_wait_time = min(current_wait_time * 2, max_wait_time)

    # Nếu thoát khỏi vòng lặp do hết max_total_attempts
    print(f"Không thể tạo embedding cho '{text[:50]}...' sau {max_total_attempts} lần thử với các API key khác nhau.")
    # Reset cấu hình về key chính
    # genai.configure(api_key=PRIMARY_GOOGLE_API_KEY)
    return None

# --- Hàm Embed và Upsert dữ liệu vào ChromaDB ---
def embed_and_upsert_profiles(df, collection, batch_size_chroma=100):
    """Tạo embedding và upsert dữ liệu vào ChromaDB theo lô."""
    print(f"\nBắt đầu quá trình embedding và upsert vào collection '{collection.name}'...")
    profiles_to_upsert = []
    processed_count = 0
    failed_count = 0 # Đếm số lượng không thể embed

    total_profiles = len(df)
    progress_bar = tqdm(total=total_profiles, desc="Embedding & Upserting")

    for index, row in df.iterrows():
        text_to_embed = row.get(DETAIL_COLUMN_NAME)
        if pd.isna(text_to_embed) or not isinstance(text_to_embed, str) or not text_to_embed.strip():
            progress_bar.update(1)
            continue # Bỏ qua nếu dữ liệu không hợp lệ

        # Hàm get_embedding giờ sẽ tự động thử lại mạnh mẽ
        embedding = get_embedding(text_to_embed, task_type="RETRIEVAL_DOCUMENT")

        if embedding:
            # Chuẩn bị metadata - chỉ lưu các kiểu dữ liệu cơ bản
            metadata = {}
            for col in ['Tiêu đề', 'Họ và tên', 'Link', 'Năm sinh', 'Năm thất lạc']:
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
                    failed_count += len(profiles_to_upsert) # Coi như batch này lỗi
                    profiles_to_upsert = [] # Reset batch lỗi

        else:
            # Nếu get_embedding trả về None (sau khi đã thử lại rất nhiều lần hoặc gặp lỗi fatal)
            print(f"Không thể tạo embedding cho hồ sơ index {index} sau nhiều lần thử. Bỏ qua hồ sơ này.")
            failed_count += 1

        progress_bar.update(1)
        # KHÔNG CẦN time.sleep ở đây nữa, vì get_embedding đã xử lý việc chờ
        # time.sleep(0.05) # <--- XÓA DÒNG NÀY

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

# --- Hàm xác minh hồ sơ bằng LLM (Prompt đã được cải thiện ở lần trước) ---
def verify_profiles_with_llm(query, profiles_data, api_key):
    """Verify profiles using direct HTTP requests to Gemini API with specific key."""
    # ... (phần tạo profile_strings và prompt giữ nguyên) ...
    profile_strings = []
    for profile in profiles_data:
        profile_id = profile.get('id') if isinstance(profile, dict) else profile.name
        title = profile.get('Tiêu đề', 'N/A')
        name = profile.get('Họ và tên', 'N/A')
        detail_source = profile.get('metadata', {}) if isinstance(profile, dict) and 'metadata' in profile else profile
        detail = detail_source.get(DETAIL_COLUMN_NAME, 'N/A')
        detail = str(detail).replace('\\', '/')[:1000]

        profile_str = f"""
Index: {profile_id}
Tiêu đề: {title}
Họ tên: {name}
Chi tiết: {detail}
{"-"*40}"""
        profile_strings.append(profile_str)

    prompt = f"""Bạn là một chuyên gia phân tích hồ sơ tìm kiếm người thân thất lạc cực kỳ tỉ mỉ và chính xác. Nhiệm vụ của bạn là phân tích các hồ sơ dưới đây và chỉ xác định những hồ sơ nào mô tả **chính xác cùng một người** và **cùng một hoàn cảnh thất lạc** như được nêu trong Yêu cầu tìm kiếm.

Hãy so sánh **cực kỳ cẩn thận** các chi tiết nhận dạng cốt lõi:
- **Họ tên người thất lạc:** Phải khớp hoặc rất tương đồng.
- **Tên cha, mẹ, anh chị em (nếu có trong yêu cầu):** Phải khớp hoặc rất tương đồng.
- **Năm sinh:** Phải khớp hoặc gần đúng.
- **Quê quán/Địa chỉ liên quan:** Phải khớp hoặc có liên quan logic.
- **Hoàn cảnh thất lạc (thời gian, địa điểm, sự kiện chính):** Phải tương đồng đáng kể.

**Quy tắc loại trừ quan trọng:**
- Nếu **Họ tên người thất lạc** trong hồ sơ **khác biệt rõ ràng** so với yêu cầu, hãy **LOẠI BỎ** hồ sơ đó NGAY LẬP TỨC, bất kể các chi tiết khác có trùng khớp hay không.
- Nếu **tên cha mẹ hoặc anh chị em** (khi được cung cấp trong yêu cầu) trong hồ sơ **hoàn toàn khác biệt**, hồ sơ đó rất có thể **KHÔNG PHÙ HỢP** và cần được xem xét loại bỏ.
- Sự trùng khớp **chỉ** về địa danh hoặc năm sinh là **KHÔNG ĐỦ** để kết luận hồ sơ phù hợp nếu các tên riêng cốt lõi và hoàn cảnh thất lạc khác biệt.

Mỗi hồ sơ có Index gốc, Tiêu đề, Họ tên và Chi tiết mô tả.

Yêu cầu tìm kiếm:
{query}
------------------------------------

Các hồ sơ cần kiểm tra:
------------------------------------
{"".join(profile_strings)}
------------------------------------

Hãy trả về **chỉ các Index gốc** (là các chuỗi ID dạng số) của những hồ sơ mà bạn **rất chắc chắn** (high confidence) là phù hợp dựa trên **tất cả các tiêu chí cốt lõi** nêu trên. Mỗi index trên một dòng. Nếu không có hồ sơ nào thực sự phù hợp, trả về 'none'.
"""

    api_endpoint = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key,
    }
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        # Có thể thêm generationConfig và safetySettings nếu cần
        "generationConfig": {
             "temperature": 0.2,
             "maxOutputTokens": 256
        },
        "safetySettings": [
            { "category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE" },
            { "category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE" },
            { "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE" },
            { "category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE" }
        ]
    }

    for attempt in range(MAX_RETRIES_LLM):
        try:
            response = requests.post(api_endpoint, headers=headers, json=payload, timeout=60) # Thêm timeout

            # Kiểm tra lỗi Rate Limit (429) hoặc Server Error (5xx)
            if response.status_code == 429:
                error_type = "Rate Limit (429)"
                # Không cần phân tích response.json() vì lỗi là do header
            elif response.status_code >= 500:
                error_type = f"Server Error ({response.status_code})"
                # Có thể thử phân tích lỗi chi tiết hơn từ response nếu có
                try:
                    error_detail = response.json().get('error', {}).get('message', response.text)
                    print(f"  Server error detail: {error_detail}")
                except json.JSONDecodeError:
                    print(f"  Server error response (non-JSON): {response.text}")
            elif response.status_code != 200:
                # Các lỗi khác (ví dụ: 400 Bad Request, 403 Forbidden/API Key invalid)
                error_type = f"HTTP Error {response.status_code}"
                error_detail = "Unknown error"
                try:
                    error_json = response.json().get('error', {})
                    error_detail = error_json.get('message', response.text)
                    # Kiểm tra lỗi API key cụ thể
                    if "API key not valid" in error_detail:
                         print(f"Lỗi API Key không hợp lệ (Key: ...{api_key[-4:]}). Ngừng thử lại với key này.")
                         return [] # Trả về list rỗng
                except json.JSONDecodeError:
                     error_detail = response.text # Nếu response không phải JSON
                print(f"Lỗi không thể thử lại ({error_type}) khi gọi Gemini API (Key ...{api_key[-4:]}): {error_detail}")
                return [] # Không thử lại các lỗi client khác 429

            # Nếu là lỗi có thể thử lại (429, 5xx)
            if response.status_code == 429 or response.status_code >= 500:
                 if attempt < MAX_RETRIES_LLM - 1:
                    wait_time = INITIAL_RETRY_DELAY_LLM * (2 ** attempt)
                    print(f"Lỗi '{error_type}' (Key ...{api_key[-4:]}). Retrying in {wait_time} seconds... (Attempt {attempt+1}/{MAX_RETRIES_LLM})")
                    time.sleep(wait_time)
                    continue # Thử lại vòng lặp
                 else:
                    print(f"Không thể xác minh batch sau {MAX_RETRIES_LLM} lần thử do lỗi '{error_type}' (Key ...{api_key[-4:]}).")
                    return [] # Hết số lần thử

            # Nếu thành công (status_code == 200)
            response_data = response.json()

            # Trích xuất text một cách an toàn
            try:
                # Kiểm tra xem có bị block do safety settings không
                if response_data.get('promptFeedback', {}).get('blockReason'):
                    block_reason = response_data['promptFeedback']['blockReason']
                    print(f"Cảnh báo: Yêu cầu bị chặn do safety settings (Key ...{api_key[-4:]}): {block_reason}")
                    return []

                # Kiểm tra cấu trúc response chuẩn
                generated_text = response_data['candidates'][0]['content']['parts'][0]['text']

                if generated_text:
                    if generated_text.strip().lower() == 'none':
                        return [] # Không có kết quả phù hợp
                    # Tách và chuyển đổi index
                    matched_indices_str = [idx.strip() for idx in generated_text.split('\n') if idx.strip().isdigit()]
                    return matched_indices_str
                else:
                    print(f"Cảnh báo: Gemini API trả về phản hồi thành công nhưng text rỗng (Key ...{api_key[-4:]}).")
                    return []
            except (KeyError, IndexError, TypeError) as e:
                print(f"Lỗi khi phân tích response thành công từ Gemini API (Key ...{api_key[-4:]}): {e}")
                print(f"  Response data: {response_data}")
                return [] # Coi như lỗi

        except requests.exceptions.RequestException as e:
            # Lỗi mạng (Timeout, ConnectionError, etc.)
            error_type = f"Network Error ({type(e).__name__})"
            if attempt < MAX_RETRIES_LLM - 1:
                wait_time = INITIAL_RETRY_DELAY_LLM * (2 ** attempt)
                print(f"Lỗi '{error_type}' (Key ...{api_key[-4:]}). Retrying in {wait_time} seconds... (Attempt {attempt+1}/{MAX_RETRIES_LLM})")
                time.sleep(wait_time)
            else:
                print(f"Không thể xác minh batch sau {MAX_RETRIES_LLM} lần thử do lỗi '{error_type}' (Key ...{api_key[-4:]}).")
                return [] # Hết số lần thử

    return [] # Trả về list rỗng nếu vòng lặp kết thúc mà không thành công

# --- Hàm xác minh song song (Cập nhật để xử lý profile_data) ---
def parallel_verify(query, ranked_profiles_data, max_profiles=300):
    """Perform parallel verification of profiles using multiple API keys."""
    max_profiles = min(max_profiles, len(ranked_profiles_data))
    profiles_to_verify = ranked_profiles_data[:max_profiles]
    print(f"Xử lý {max_profiles} hồ sơ có điểm số cao nhất để xác minh bằng LLM")

    if not profiles_to_verify:
        return []

    batches = [profiles_to_verify[i:i + BATCH_SIZE_LLM]
               for i in range(0, len(profiles_to_verify), BATCH_SIZE_LLM)]
    print(f"Chia {len(profiles_to_verify)} hồ sơ thành {len(batches)} batch, mỗi batch tối đa {BATCH_SIZE_LLM} hồ sơ")

    verified_indices_str = set()
    num_api_keys = len(GEMINI_API_KEYS)
    print(f"Sử dụng {num_api_keys} API key để phân phối tải (lưu ý: generate_content dùng key chính)")

    batch_groups = [batches[i:i + MAX_CONCURRENT_REQUESTS_LLM]
                    for i in range(0, len(batches), MAX_CONCURRENT_REQUESTS_LLM)]

    total_batches_processed = 0
    with tqdm(total=len(batches), desc="Verifying Batches (LLM)") as pbar_llm:
        for group_idx, batch_group in enumerate(batch_groups):
            with ThreadPoolExecutor(max_workers=min(len(batch_group), MAX_CONCURRENT_REQUESTS_LLM)) as executor:
                futures = {}
                for i, batch in enumerate(batch_group):
                    if not batch: continue
                    # Vẫn xoay vòng key để log lỗi cho đúng key nếu có vấn đề khác
                    api_key_index = (total_batches_processed + i) % num_api_keys
                    api_key = GEMINI_API_KEYS[api_key_index]
                    future = executor.submit(verify_profiles_with_llm, query, batch, api_key)
                    futures[future] = total_batches_processed + i + 1 # Lưu index batch để debug

                for future in futures:
                    batch_index_debug = futures[future] # Lấy index batch để debug
                    try:
                        # verify_profiles_with_llm giờ trả về list (có thể rỗng)
                        result = future.result()
                        if result: # Kiểm tra nếu list không rỗng
                            verified_indices_str.update(result) # Dùng set để tự động loại bỏ trùng lặp
                    except Exception as e:
                        # Lỗi xảy ra khi lấy kết quả từ future (ít khả năng hơn vì lỗi đã được xử lý bên trong)
                        print(f"Lỗi nghiêm trọng khi lấy kết quả của batch {batch_index_debug}: {e}")

            total_batches_processed += len(batch_group)
            pbar_llm.update(len(batch_group))

            if group_idx < len(batch_groups) - 1:
                 # print(f"Chờ {BATCH_GROUP_DELAY_LLM} giây trước khi xử lý nhóm batch tiếp theo...") # Có thể bỏ comment nếu cần giảm tải
                 time.sleep(BATCH_GROUP_DELAY_LLM)

    return list(verified_indices_str) # Trả về list các ID string đã xác minh

# --- Hàm trích xuất từ khóa từ truy vấn bằng Gemini ---
def extract_keywords_gemini(query, model="gemini-2.0-flash"):
    """Trích xuất các từ khóa quan trọng từ truy vấn bằng Gemini (có ví dụ)."""
    prompt = f"""Phân tích các hồ sơ tìm kiếm người thân thất lạc sau và trích xuất các từ khóa quan trọng có thể dùng để tìm kiếm thông tin liên quan đến người mất tích. Trả về một danh sách các từ khóa và những từ có khả năng liên quan. Lưu ý tên riêng có thể phân tích nhỏ hơn thành tên riêng (ví dụ: Lê Thị Hạnh => Hạnh). Từ khóa liên quan có thể được sinh ra từ các từ khóa chính (ví dụ: chiến tranh => xung đột, chạy giặc, vượt biên, di cư,...) hoặc từ các từ khóa khác trong đoạn văn bản. Vậy nhiệm vụ của bạn là trích xuất các từ khóa quan trọng nhất có thể dùng để tìm kiếm thông tin liên quan đến người mất tích và các từ khóa liên quan có thể sinh ra từ các từ khóa chính. Các từ khóa này có thể là tên riêng, địa danh, năm sinh, địa chỉ, đặc điểm nhận dạng, ký ức hoặc các thông tin khác... . Hãy trả về danh sách các từ khóa và các từ khóa liên quan có thể sinh ra từ các từ khóa chính.

Ví dụ 1:
Đoạn văn bản: Chị Lê Thị Mỹ Duyên tìm bác Lê Viết Thi, đi vượt biên mất liên lạc khoảng năm 1978. Ông Lê Viết Thi sinh năm 1946, quê Quảng Nam. Bố mẹ là cụ Lê Viết Y và cụ Nguyễn Thị Ca. Anh chị em trong gia đình là Viết, Thơ, Dũng, Chung, Mười, Sỹ và Tượng. Khoảng năm 1978, ông Lê Viết Thi đi vượt biên. Từ đó, gia đình không còn nghe tin tức gì về ông.
Các từ khóa quan trọng: Lê Thị Mỹ Duyên, Duyên, Lê Viết Thi, Thi, vượt biên, di cư, chiến tranh, chạy giặc, 1978, 1946, Quảng Nam, Lê Viết Y, Y Nguyễn Thị Ca, Ca, Viết, Thơ, Dũng, Chung, Mười, Sỹ, Tượng

Ví dụ 2:
Đoạn văn bản: Chị Lê Thị Toàn tìm anh Lê Văn Thương, mất liên lạc năm 1984 tại ga Đông Hà, Quảng Trị. Vào năm 1984, gia đình ông Tiên và bà Tẻo từ Thanh Hóa di chuyển vào Quảng Trị. Khi đến ga Đông Hà (Quảng Trị), vì hoàn cảnh quá khó khăn, ông Tiên bị tật ở chân, còn bà Tẻo không minh mẫn nên bà Tèo đã mang con trai Lê Văn Thương vừa mới sinh cho một người phụ nữ ở ga Đông Hà. Người phụ nữ đó có cho bà Tẻo một ít tiền rồi ôm anh Thương đi mất.
Các từ khóa quan trọng: Lê Thị Toàn, Toàn, Lê Văn Thương, Thương, 1984, Đông Hà, Quảng Trị, Tiên, Tẻo, Thanh Hóa, Quảng Trị, di chuyển, di cư, khó khăn, thiếu thốn, nghèo khổ, tật, khiếm khuyết, không minh mẫn, thần kinh, tâm thần, mới sinh, sơ sinh, mới đẻ.

Ví dụ 3:
Đoạn văn bản: Chị Nguyễn Thị Yến tìm ba Nguyễn Văn Đã mất liên lạc năm 1977. Ông Nguyễn Văn Đã, sinh năm 1939, không rõ quê quán. Khoảng năm 1970, bà Vũ Thị Hải gặp ông Nguyễn Văn Đã ở nông trường Sao Đỏ tại Mộc Châu, Sơn La. Ông Đã phụ trách lái xe lương thực cho nông trường. Sau khi sinh chị Yến, ông muốn đưa hai mẹ con về quê ông nhưng bà Hải biết ông Đã đã có vợ ở quê nên không đồng ý và đem con về khu tập thể nhà máy nước Nam Định. Ông Đã vẫn thường lái xe về thăm con gái. Năm 1979, bà Hải mang con về quê bà sinh sống, từ đó chị Yến không hay tin gì về ba nữa.
Các từ khóa quan trọng: Nguyễn Thị Yến, Yến, Nguyễn Văn Đã, Đã, 1977, 1939, 1970, Vũ Thị Hải, Hải, nông trường, Sao Đỏ, Mộc Châu, Sơn La, lái xe, lương thực, nông trường, làm nông, nông nghiệp, khu tập thể, nhà máy nước, Nam Định, 1979

*Chú ý: những từ khóa nào phổ biến, phổ thông quá thì bỏ qua như: gia đình, anh, em, vợ, chồng, tìm kiếm, thất lạc, mất tích, mất liên lạc, không rõ quê quán, không rõ địa chỉ, không rõ thông tin, không rõ năm sinh, không rõ đặc điểm nhận dạng, không rõ ký ức...

Đoạn văn bản hiện tại:
{query}

Các từ khóa quan trọng:"""

    try:
        response = genai.GenerativeModel(model).generate_content(prompt)
        if response.text:
            keywords_str = response.text.strip()
            keywords = [kw.strip() for kw in keywords_str.split(',') + keywords_str.split('\n') if kw.strip()]
            return keywords
        else:
            print("Gemini không trả về kết quả trích xuất từ khóa.")
            return []
    except Exception as e:
        print(f"Lỗi khi gọi Gemini để trích xuất từ khóa: {e}")
        return []
    
# --- Hàm Tìm kiếm kết hợp ChromaDB và Từ khóa ---
def search_combined_chroma(df_original, collection, user_query, top_n_final=100):
    """
    Thực hiện tìm kiếm kết hợp:
    1. Tìm tất cả hồ sơ có ít nhất 1 từ khóa trùng khớp
    2. Thực hiện vector search trên tất cả hồ sơ
    3. Tính tổng điểm = điểm tương đồng vector + (số từ khóa khớp × 0.05)
    4. Chọn top_n_final hồ sơ có tổng điểm cao nhất để LLM lọc tiếp
    """
    print("\n--- Bắt đầu Tìm kiếm (Kết hợp Từ Khóa và Vector Search -> LLM) ---")

    # --- Bước 1: Trích xuất từ khóa và tìm tất cả hồ sơ có ít nhất 1 từ khóa trùng khớp ---
    keywords = extract_keywords_gemini(user_query)
    print("Từ khóa trích xuất từ Gemini:", keywords)

    # Tìm kiếm các hồ sơ chứa ít nhất một từ khóa
    keyword_match_indices = set()
    keyword_match_counts = {}

    if keywords:
        print("Đang tìm kiếm hồ sơ chứa từ khóa...")
        for keyword in keywords:
            for col in df_original.columns:
                if df_original[col].dtype == object:  # Chỉ tìm kiếm trên các cột dạng văn bản
                    try:
                        matches = df_original[col].str.contains(keyword, case=False, na=False)
                        matched_indices = df_original[matches].index
                        keyword_match_indices.update(matched_indices)

                        # Đếm số lượng từ khóa khớp cho từng hồ sơ
                        for idx in matched_indices:
                            keyword_match_counts[idx] = keyword_match_counts.get(idx, 0) + 1
                    except Exception as e:
                        continue

    # Lọc các indices hợp lệ
    valid_keyword_indices = [idx for idx in keyword_match_indices if idx in df_original.index]

    if not valid_keyword_indices:
        print("\nKhông tìm thấy hồ sơ nào khớp với từ khóa. Sẽ tìm kiếm bằng vector search trên toàn bộ dữ liệu.")
        # Nếu không có hồ sơ nào khớp từ khóa, tất cả các hồ sơ đều có số từ khóa khớp = 0
        keyword_match_counts = {idx: 0 for idx in df_original.index}
    else:
        print(f"\nTìm thấy {len(valid_keyword_indices)} hồ sơ khớp với ít nhất một từ khóa.")
        # Hiển thị phân bố số lượng từ khóa khớp
        keyword_counts = list(keyword_match_counts.values())
        print(f"Thống kê số lượng từ khóa khớp: Min={min(keyword_counts)}, Max={max(keyword_counts)}, Avg={sum(keyword_counts)/len(keyword_counts):.2f}")

    # --- Bước 2: Thực hiện vector search trên toàn bộ collection ---
    print("\nĐang tạo embedding cho truy vấn...")
    query_embedding = get_embedding(user_query, task_type="RETRIEVAL_QUERY")

    if query_embedding is None:
        print("Lỗi: Không thể tạo embedding cho truy vấn. Sử dụng chỉ kết quả từ khóa.")

        if not valid_keyword_indices:
            print("Không tìm thấy kết quả từ khóa và không thể thực hiện vector search. Kết thúc tìm kiếm.")
            return None

        # Nếu không thể tạo embedding, xếp hạng chỉ dựa trên số lượng từ khóa khớp
        ranked_by_keywords = sorted(keyword_match_counts.items(), key=lambda x: x[1], reverse=True)
        top_keyword_profiles = ranked_by_keywords[:top_n_final]

        profiles_for_llm = []
        for idx, _ in top_keyword_profiles:
            try:
                profile_data = df_original.loc[idx].copy()
                profile_data['id'] = str(idx)
                profiles_for_llm.append(profile_data)
            except KeyError:
                continue

        # Xác minh bằng LLM
        print(f"\nĐang xác minh {len(profiles_for_llm)} kết quả với Gemini LLM...")
        verified_indices_str = parallel_verify(user_query, profiles_for_llm, max_profiles=len(profiles_for_llm))

        # Hiển thị kết quả cuối
        if verified_indices_str:
            print(f"\n=== {len(verified_indices_str)} KẾT QUẢ PHÙ HỢP NHẤT SAU KHI LỌC BẰNG LLM ===")
            for id_str in verified_indices_str:
                try:
                    idx = int(id_str)
                    profile = df_original.loc[idx]
                    print(f"\nIndex: {idx}")
                    print(f"Số từ khóa khớp: {keyword_match_counts.get(idx, 0)}")
                    print(f"Tiêu đề: {profile.get('Tiêu đề', 'N/A')}")
                    print(f"Họ và tên: {profile.get('Họ và tên', 'N/A')}")
                    print(f"Chi tiết: {str(profile.get(DETAIL_COLUMN_NAME, 'N/A'))[:500]}...")
                    print(f"Link: {profile.get('Link', 'N/A')}")
                    print("-" * 60)
                except Exception as e:
                    print(f"Lỗi khi hiển thị hồ sơ {id_str}: {e}")
        else:
            print("\nKhông tìm thấy hồ sơ nào phù hợp sau khi xác minh bằng LLM.")

        return verified_indices_str

    # Thực hiện vector search
    print("Đang thực hiện vector search trên toàn bộ collection...")
    try:
        vector_results = collection.query(
            query_embeddings=[query_embedding],
            n_results=collection.count(),  # Lấy tất cả kết quả có thể
            include=['metadatas', 'distances']
        )
    except Exception as e:
        print(f"Lỗi khi truy vấn ChromaDB: {e}")
        return None

    # --- Bước 3: Kết hợp kết quả từ khóa và vector ---
    print("Đang kết hợp kết quả từ khóa và vector...")

    # Lấy dữ liệu từ vector search
    vector_distances = {}
    if vector_results and vector_results.get('ids') and vector_results['ids'][0]:
        vector_ids = vector_results['ids'][0]
        distances = vector_results['distances'][0]

        # Chuyển distances thành điểm tương đồng (1 - distance)
        for i, id_str in enumerate(vector_ids):
            try:
                idx = int(id_str)
                vector_distances[idx] = 1 - distances[i]  # Chuyển sang similarity score (0-1)
            except ValueError:
                continue
    else:
        print("Không nhận được kết quả từ vector search.")
        return None

    # Tính tổng điểm kết hợp
    combined_scores = {}
    KEYWORD_BONUS = 0.05  # Hệ số điểm thưởng cho mỗi từ khóa khớp

    # Duyệt qua tất cả hồ sơ trong dataset gốc
    for idx in df_original.index:
        # Lấy điểm vector (nếu có)
        vector_score = vector_distances.get(idx, 0)

        # Lấy số lượng từ khóa khớp (nếu có)
        keyword_count = keyword_match_counts.get(idx, 0)

        # Tính tổng điểm
        total_score = vector_score + keyword_count * KEYWORD_BONUS

        # Chỉ lưu các hồ sơ có điểm > 0 (có khớp vector hoặc từ khóa)
        if total_score > 0:
            combined_scores[idx] = total_score

    if not combined_scores:
        print("Không tìm thấy hồ sơ nào có điểm kết hợp > 0.")
        return None

    # --- Bước 4: Xếp hạng và chọn top_n_final ---
    print(f"Đang xếp hạng {len(combined_scores)} hồ sơ theo điểm kết hợp...")

    # Sắp xếp theo tổng điểm giảm dần
    ranked_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)

    # Lấy top_n_final kết quả
    top_results = ranked_results[:top_n_final]

    # Hiển thị top 10 kết quả
    print(f"\n--- Top {min(10, len(top_results))} Kết quả (Theo Điểm Kết Hợp, Trước LLM) ---")
    for i, (idx, score) in enumerate(top_results[:10]):
        try:
            profile = df_original.loc[idx]
            kw_count = keyword_match_counts.get(idx, 0)
            kw_score = kw_count * KEYWORD_BONUS
            vec_score = vector_distances.get(idx, 0)

            print(f"  Rank: {i+1} | Tổng điểm: {score:.4f} (Vector: {vec_score:.4f}, KW: {kw_score:.4f}) | "
                  f"Từ khóa: {kw_count} | ID: {idx} | "
                  f"Tiêu đề: {profile.get('Tiêu đề', 'N/A')} | Họ và tên: {profile.get('Họ và tên', 'N/A')}")
        except KeyError:
            print(f"  Rank: {i+1} | Tổng điểm: {score:.4f} | ID: {idx} | Lỗi: Không tìm thấy hồ sơ.")

    if len(top_results) > 10:
        print(f"  ... và {len(top_results) - 10} kết quả khác")

    # --- Bước 5: Chuẩn bị dữ liệu cho LLM ---
    profiles_for_llm = []
    for idx, _ in top_results:
        try:
            profile_data = df_original.loc[idx].copy()
            profile_data['id'] = str(idx)  # Thêm ID dạng string
            profiles_for_llm.append(profile_data)
        except KeyError:
            continue

    if not profiles_for_llm:
        print("Không tìm thấy hồ sơ hợp lệ để gửi đến LLM.")
        return None

    # --- Bước 6: Xác minh bằng LLM ---
    print(f"\nĐang xác minh {len(profiles_for_llm)} kết quả với Gemini LLM...")
    verified_indices_str = parallel_verify(user_query, profiles_for_llm, max_profiles=len(profiles_for_llm))

    # --- Bước 7: Hiển thị kết quả cuối cùng sau LLM ---
    if verified_indices_str:
        print(f"\n=== {len(verified_indices_str)} KẾT QUẢ PHÙ HỢP NHẤT SAU KHI LỌC BẰNG LLM ===")
        verified_indices_int = [int(id_str) for id_str in verified_indices_str if id_str.isdigit()]

        # Xếp hạng lại các kết quả đã được LLM xác minh theo thứ tự điểm kết hợp
        verified_with_scores = [(idx, combined_scores.get(idx, 0)) for idx in verified_indices_int if idx in combined_scores]
        verified_with_scores.sort(key=lambda x: x[1], reverse=True)

        for i, (idx, score) in enumerate(verified_with_scores):
            try:
                profile = df_original.loc[idx]
                kw_count = keyword_match_counts.get(idx, 0)
                kw_score = kw_count * KEYWORD_BONUS
                vec_score = vector_distances.get(idx, 0)

                print(f"\n{i+1}. ID: {idx}")
                print(f"Tổng điểm: {score:.4f} (Vector: {vec_score:.4f}, KW: {kw_score:.4f}, Từ khóa khớp: {kw_count})")
                print(f"Tiêu đề: {profile.get('Tiêu đề', 'N/A')}")
                print(f"Họ và tên: {profile.get('Họ và tên', 'N/A')}")
                print(f"Chi tiết: {str(profile.get(DETAIL_COLUMN_NAME, 'N/A'))[:500]}...")
                print(f"Link: {profile.get('Link', 'N/A')}")
                print("-" * 60)
            except Exception as e:
                print(f"Lỗi khi hiển thị hồ sơ {idx}: {e}")
    else:
        print("\nKhông tìm thấy hồ sơ nào phù hợp sau khi xác minh bằng LLM.")

    return verified_indices_str

# --- Hàm Chính (Cập nhật) ---
def main():
    print("Bắt đầu quy trình xử lý hồ sơ và tìm kiếm...")

    # 1. Khởi tạo Vector DB
    collection = initialize_vector_db()

    # 2. Đọc dữ liệu CSV
    try:
        df = pd.read_csv(CSV_PATH, on_bad_lines='skip', encoding='utf-8')
        df.columns = df.columns.str.strip() # Làm sạch tên cột
        print(f"Đã đọc dữ liệu từ {CSV_PATH} ({len(df)} hồ sơ)")

        if DETAIL_COLUMN_NAME not in df.columns:
            print(f"Lỗi: Không tìm thấy cột '{DETAIL_COLUMN_NAME}' trong file CSV.")
            print(f"Các cột hiện có: {df.columns.tolist()}")
            return

        # Quan trọng: Reset index để đảm bảo index liên tục từ 0
        # Điều này quan trọng nếu CSV gốc không có index chuẩn hoặc bị thiếu dòng
        df.reset_index(drop=True, inplace=True)
        print("Đã reset index của DataFrame.")
        print("Thông tin DataFrame sau khi reset index:")
        df.info()
        # print("\n5 dòng đầu:")
        # print(df.head())

    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file {CSV_PATH}")
        return
    except Exception as e:
        print(f"Lỗi không xác định khi đọc file CSV: {e}")
        return

    # 3. Embed và Upsert dữ liệu vào ChromaDB
    # Bước này sẽ tạo embedding cho TẤT CẢ hồ sơ trong CSV và upsert vào DB.
    # ChromaDB tự xử lý việc cập nhật nếu ID đã tồn tại.
    # Trong thực tế, bạn có thể muốn có cơ chế chỉ embed và upsert
    # những hồ sơ mới hoặc đã thay đổi kể từ lần chạy trước.
    embed_and_upsert_profiles(df, collection)

    # 4. Thử nghiệm Tìm kiếm
    search_combined_chroma(df, collection, user_query=None) # Gọi vòng lặp tìm kiếm

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

# --- Chạy chương trình ---
if __name__ == "__main__":
    # 1. Khởi tạo Vector DB
    collection = initialize_vector_db()
    if collection is None: # Thoát nếu không khởi tạo được DB
        exit()

    # 2. Đọc dữ liệu CSV
    try:
        # Đảm bảo đường dẫn đúng
        df_main = pd.read_csv(CSV_PATH, on_bad_lines='skip', encoding='utf-8')
        df_main.columns = df_main.columns.str.strip()
        print(f"Đã đọc dữ liệu từ {CSV_PATH} ({len(df_main)} hồ sơ)")

        if DETAIL_COLUMN_NAME not in df_main.columns:
            print(f"Lỗi: Không tìm thấy cột '{DETAIL_COLUMN_NAME}'.")
            exit()

        # Quan trọng: Reset index để đảm bảo index liên tục từ 0
        df_main.reset_index(drop=True, inplace=True)
        print("Đã reset index của DataFrame.")

    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file {CSV_PATH}")
        exit()
    except Exception as e:
        print(f"Lỗi không xác định khi đọc file CSV: {e}")
        exit()

    # 3. Kiểm tra và Upsert hồ sơ mới (Nếu người dùng chọn)
    check_and_upsert = input("Bạn có muốn kiểm tra và upsert hồ sơ mới từ CSV vào Vector DB không? (yes/no): ").lower()
    if check_and_upsert == 'yes':
        current_db_count = 0
        try:
            current_db_count = collection.count()
        except Exception as e:
            print(f"Lỗi khi lấy số lượng hồ sơ từ ChromaDB: {e}. Tiếp tục mà không upsert.")
            current_db_count = -1 # Đánh dấu lỗi

        if current_db_count != -1:
            total_csv_count = len(df_main)
            print(f"Số hồ sơ hiện tại trong DB: {current_db_count}")
            print(f"Số hồ sơ trong file CSV: {total_csv_count}")

            if total_csv_count > current_db_count:
                num_new_profiles = total_csv_count - current_db_count
                print(f"Phát hiện {num_new_profiles} hồ sơ mới trong CSV.")

                # Lấy các hồ sơ mới cần thêm
                new_profiles_df = df_main.iloc[current_db_count:]

                print(f"Bắt đầu embedding và upsert {len(new_profiles_df)} hồ sơ mới...")
                # Gọi hàm upsert chỉ với dữ liệu mới
                # Hàm embed_and_upsert_profiles sẽ sử dụng index của new_profiles_df làm ID
                # và dùng PRIMARY_GOOGLE_API_KEY (do không truyền api_keys vào đây)
                embed_and_upsert_profiles(new_profiles_df, collection)
            else:
                print("Không có hồ sơ mới nào trong file CSV cần được thêm vào DB.")
    else:
         print("Bỏ qua bước kiểm tra và upsert hồ sơ mới.")
         try:
             print(f"Số lượng hồ sơ hiện tại trong collection '{collection.name}': {collection.count()}")
         except Exception as e:
             print(f"Lỗi khi lấy số lượng hồ sơ từ ChromaDB: {e}")


    # 4. Bắt đầu vòng lặp tìm kiếm
    search_loop(df_main, collection)

    print("\nChương trình đã kết thúc.")
    