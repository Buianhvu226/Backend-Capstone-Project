import time
import json
import requests
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import google.generativeai as genai
import random # Import random for jitter
import re # Import regular expressions module

from .config import (
    PRIMARY_GOOGLE_API_KEY,
    GEMINI_API_KEYS,
    DETAIL_COLUMN_NAME,
    BATCH_SIZE_LLM,
    MAX_CONCURRENT_REQUESTS_LLM,
    MAX_RETRIES_LLM,
    INITIAL_RETRY_DELAY_LLM,
    BATCH_GROUP_DELAY_LLM
)

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
        }
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
def extract_keywords_gemini(query, model="gemini-1.5-flash-latest"): # Use a valid model name
    """Trích xuất các từ khóa quan trọng từ truy vấn bằng Gemini (có ví dụ và làm sạch)."""
    prompt = f"""Phân tích các hồ sơ tìm kiếm người thân thất lạc sau và trích xuất các từ khóa quan trọng có thể dùng để tìm kiếm thông tin liên quan đến người mất tích. Trả về một danh sách các từ khóa và những từ có khả năng liên quan. Lưu ý tên riêng có thể phân tích nhỏ hơn thành tên riêng (ví dụ: Lê Thị Hạnh => Hạnh). Từ khóa liên quan có thể được sinh ra từ các từ khóa chính (ví dụ: chiến tranh => xung đột, chạy giặc, vượt biên, di cư,...) hoặc từ các từ khóa khác trong đoạn văn bản. Vậy nhiệm vụ của bạn là trích xuất các từ khóa quan trọng nhất có thể dùng để tìm kiếm thông tin liên quan đến người mất tích và các từ khóa liên quan có thể sinh ra từ các từ khóa chính. Các từ khóa này có thể là tên riêng, địa danh, năm sinh, địa chỉ, đặc điểm nhận dạng, ký ức hoặc các thông tin khác... . Hãy trả về danh sách các từ khóa và các từ khóa liên quan có thể sinh ra từ các từ khóa chính, mỗi từ khóa cách nhau bởi dấu phẩy.

Ví dụ 1:
Đoạn văn bản: Chị Lê Thị Mỹ Duyên tìm bác Lê Viết Thi, đi vượt biên mất liên lạc khoảng năm 1978. Ông Lê Viết Thi sinh năm 1946, quê Quảng Nam. Bố mẹ là cụ Lê Viết Y và cụ Nguyễn Thị Ca. Anh chị em trong gia đình là Viết, Thơ, Dũng, Chung, Mười, Sỹ và Tượng. Khoảng năm 1978, ông Lê Viết Thi đi vượt biên. Từ đó, gia đình không còn nghe tin tức gì về ông.
Các từ khóa quan trọng: Lê Thị Mỹ Duyên, Duyên, Lê Viết Thi, Thi, vượt biên, di cư, chiến tranh, chạy giặc, 1978, 1946, Quảng Nam, Lê Viết Y, Y, Nguyễn Thị Ca, Ca, Viết, Thơ, Dũng, Chung, Mười, Sỹ, Tượng

Ví dụ 2:
Đoạn văn bản: Chị Lê Thị Toàn tìm anh Lê Văn Thương, mất liên lạc năm 1984 tại ga Đông Hà, Quảng Trị. Vào năm 1984, gia đình ông Tiên và bà Tẻo từ Thanh Hóa di chuyển vào Quảng Trị. Khi đến ga Đông Hà (Quảng Trị), vì hoàn cảnh quá khó khăn, ông Tiên bị tật ở chân, còn bà Tẻo không minh mẫn nên bà Tèo đã mang con trai Lê Văn Thương vừa mới sinh cho một người phụ nữ ở ga Đông Hà. Người phụ nữ đó có cho bà Tẻo một ít tiền rồi ôm anh Thương đi mất.
Các từ khóa quan trọng: Lê Thị Toàn, Toàn, Lê Văn Thương, Thương, 1984, Đông Hà, Quảng Trị, Tiên, Tẻo, Thanh Hóa, Quảng Trị, di chuyển, di cư, khó khăn, thiếu thốn, nghèo khổ, tật, khiếm khuyết, không minh mẫn, thần kinh, tâm thần, mới sinh, sơ sinh, mới đẻ

Ví dụ 3:
Đoạn văn bản: Chị Nguyễn Thị Yến tìm ba Nguyễn Văn Đã mất liên lạc năm 1977. Ông Nguyễn Văn Đã, sinh năm 1939, không rõ quê quán. Khoảng năm 1970, bà Vũ Thị Hải gặp ông Nguyễn Văn Đã ở nông trường Sao Đỏ tại Mộc Châu, Sơn La. Ông Đã phụ trách lái xe lương thực cho nông trường. Sau khi sinh chị Yến, ông muốn đưa hai mẹ con về quê ông nhưng bà Hải biết ông Đã đã có vợ ở quê nên không đồng ý và đem con về khu tập thể nhà máy nước Nam Định. Ông Đã vẫn thường lái xe về thăm con gái. Năm 1979, bà Hải mang con về quê bà sinh sống, từ đó chị Yến không hay tin gì về ba nữa.
Các từ khóa quan trọng: Nguyễn Thị Yến, Yến, Nguyễn Văn Đã, Đã, 1977, 1939, 1970, Vũ Thị Hải, Hải, nông trường, Sao Đỏ, Mộc Châu, Sơn La, lái xe, lương thực, nông trường, làm nông, nông nghiệp, khu tập thể, nhà máy nước, Nam Định, 1979

*Chú ý: những từ khóa nào phổ biến, phổ thông quá thì bỏ qua như: gia đình, anh, em, vợ, chồng, tìm kiếm, thất lạc, mất tích, mất liên lạc, không rõ quê quán, không rõ địa chỉ, không rõ thông tin, không rõ năm sinh, không rõ đặc điểm nhận dạng, không rõ ký ức...

Đoạn văn bản hiện tại:
{query}

Các từ khóa quan trọng:""" # Updated prompt to ask for comma-separated list

    try:
        # Ensure the primary key is configured for the SDK call
        if not PRIMARY_GOOGLE_API_KEY:
             print("Lỗi: PRIMARY_GOOGLE_API_KEY chưa được cấu hình trong config.py cho SDK.")
             return []
        genai.configure(api_key=PRIMARY_GOOGLE_API_KEY)

        # Use the correct model name if different from default
        model_instance = genai.GenerativeModel(model) # Use the model parameter
        response = model_instance.generate_content(prompt)

        if response.text:
            keywords_str = response.text.strip()
            print(f"Từ khóa gốc từ Gemini: {keywords_str}") # Log raw output

            # --- Start of Cleaning Logic ---
            # 1. Remove potential introductory phrases (adjust regex as needed)
            keywords_str = re.sub(r"^(Dựa trên.*?:|Các từ khóa quan trọng là:|Đây là các từ khóa:)\s*", "", keywords_str, flags=re.IGNORECASE | re.MULTILINE).strip()

            # 2. Remove markdown list markers (*, -) and bold markers (**)
            keywords_str = re.sub(r'^[\*\-\s]+', '', keywords_str, flags=re.MULTILINE) # Remove list markers at start of lines
            keywords_str = keywords_str.replace('**', '') # Remove bold markers

            # 3. Replace newlines with commas to standardize the separator
            keywords_str = keywords_str.replace('\n', ',')

            # 4. Split by comma, strip whitespace, remove empty strings
            raw_keywords = [kw.strip() for kw in keywords_str.split(',') if kw.strip()]

            # 5. Remove duplicates while preserving order
            unique_keywords = list(dict.fromkeys(raw_keywords))
            # --- End of Cleaning Logic ---

            print(f"Từ khóa đã làm sạch: {unique_keywords}") # Log cleaned output
            return unique_keywords
        else:
            print("Gemini không trả về kết quả trích xuất từ khóa.")
            return []
    # Add specific exception handling for Gemini API errors if possible
    except Exception as e:
        print(f"Lỗi khi gọi Gemini để trích xuất từ khóa: {e}")
        # You might want to check the type of exception, e.g., RateLimitError from the SDK
        return []