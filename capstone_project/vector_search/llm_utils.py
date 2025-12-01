import time
import json
import requests
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import google.generativeai as genai
import random # Import random for jitter
import re # Import regular expressions module
from random import choice

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

    prompt = f"""Bạn là chuyên gia phân tích hồ sơ tìm kiếm người thân thất lạc với khả năng nhận diện pattern phức tạp. Nhiệm vụ của bạn là tìm những hồ sơ có khả năng mô tả **cùng một người** và **cùng một hoàn cảnh thất lạc** với yêu cầu tìm kiếm bên dưới.

## CÁC YẾU TỐ QUAN TRỌNG (THEO MỨC ĐỘ ƯU TIÊN):

### 1. **TÊN CHA/MẸ – YẾU TỐ ƯU TIÊN CAO NHẤT**
- Ít thay đổi theo thời gian → nếu khớp chính xác, rất có khả năng cùng người
- Nếu khác hoàn toàn → cần có các yếu tố khác để bù trừ

### 2. **TÊN ANH CHỊ EM VÀ CẤU TRÚC GIA ĐÌNH**
- Tên, số lượng, vị trí trong gia đình ("con út", "con thứ ba")

### 3. **HOÀN CẢNH THẤT LẠC**
- Sự kiện đặc trưng: "lạc trong chiến tranh", "được đem cho đi nuôi", "bỏ nhà đi", "vượt biên", ...
- Thời điểm: năm, độ tuổi
- Cách thức: bị lạc, được đưa đi, chiến tranh

### 4. **THÔNG TIN NGƯỜI THẤT LẠC**
- Chú ý ràng tên có thể thay đổi do: nhận nuôi, đặt lại tên, biệt danh, ...
- Nếu khác tên nhưng các yếu tố khác khớp → vẫn coi là khả năng cao
- Đặc điểm nhận dạng đặc biệt trước khi thất lạc: "tật ở chân", "sẹo", "vết bớt", ...

### 5. **THÔNG TIN ĐỊA LÝ**
- Địa điểm thất lạc, nơi sinh sống trước đó, tỉnh/thành gốc

### 6. **HOÀN CẢNH GIA ĐÌNH**
- Nghề nghiệp cha mẹ, điều kiện xã hội (chiến tranh, di cư...)

## NGUYÊN TẮC ĐÁNH GIÁ:

### **NÊN GỢI Ý** khi:
- Có **ít nhất 2 yếu tố quan trọng khớp rõ ràng**
- Không có mâu thuẫn lớn về thời gian, địa điểm
- Có logic hợp lý giữa các chi tiết (vd: cùng thời điểm, cùng sự kiện đặc biệt, manh mối rõ ràng)

### **LOẠI BỎ NGAY** nếu:
- Tên cha mẹ hoàn toàn khác
- Hoàn cảnh thất lạc hoặc thời gian khác biệt rõ ràng

## LƯU Ý QUAN TRỌNG:

- Chỉ gợi ý hồ sơ nếu bạn **thật sự thấy có khả năng liên quan**.  
- Nếu có nhiều hồ sơ phù hợp, **chỉ chọn tối đa 20 hồ sơ tốt nhất**, xếp theo mức độ khớp từ cao xuống thấp.  
- Không cần cố gợi ý nếu không đủ thông tin hoặc thấy không thuyết phục.
- Hãy đọc kỹ để hiểu rõ nội dung của từng hồ sơ để so sánh với yêu cầu tìm kiếm. Chú ý phân tích vào các thông tin, manh mối trước khi thất lạc thay vì những chi tiết sau khi thất lạc.
- Ví dụ, nếu yêu cầu tìm kiếm là "Gia đình đang tìm con trai tên Long thất lạc năm 90 tại Sài Gòn", thì các hồ sơ có tên con trai là Long, mất tích vào khoảng năm 90, tại Sài Gòn hoặc có cha mẹ tên giống nhau sẽ được ưu tiên hơn. Các hồ sơ có tên con trai khác, thời gian và địa điểm khác biệt rõ ràng sẽ bị loại bỏ ngay.
Yêu cầu tìm kiếm:
{query}
------------------------------------

Các hồ sơ cần kiểm tra:
{"".join(profile_strings)}

---

 **Hãy trả về duy nhất danh sách các Index** của hồ sơ phù hợp, mỗi index trên một dòng. Nếu không có hồ sơ phù hợp, trả về đúng từ `none`.
"""

    api_endpoint = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
    # api_endpoint = "https://generativelanguage.googleapis.com/v1/models/gemini-2.5-flash:generateContent"
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key,
    }
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        # Có thể thêm generationConfig và safetySettings nếu cần
        "generationConfig": {
             "temperature": 0.2,
             "maxOutputTokens": 100
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
    max_profiles = min(max_profiles, len(ranked_profiles_data))
    profiles_to_verify = ranked_profiles_data[:max_profiles]
    print(f"Xử lý {max_profiles} hồ sơ có điểm số cao nhất để xác minh bằng LLM")

    if not profiles_to_verify:
        return []

    print(f"Xử lý {len(profiles_to_verify)} hồ sơ trong 1 lần gọi API")

    verified_indices_str = set()

    # 🔁 Thử nhiều key nếu gặp lỗi
    for api_key in random.sample(GEMINI_API_KEYS, len(GEMINI_API_KEYS)):
        print(f"Thử xác minh với API key ...{api_key[-4:]}")
        result = verify_profiles_with_llm(query, profiles_to_verify, api_key)
        if result:
            verified_indices_str.update(result)
            print(f"✅ Đã xác minh thành công {len(result)} hồ sơ với key ...{api_key[-4:]}")
            break  # Nếu thành công thì dừng lại
        else:
            print(f"❌ Không xác minh được với key ...{api_key[-4:]}, thử key khác.")

    return list(verified_indices_str)

# --- Hàm trích xuất từ khóa từ truy vấn bằng Gemini ---
def extract_keywords_gemini(query, model="gemini-2.5-flash"): # Use a valid model name
    """Trích xuất các từ khóa quan trọng từ truy vấn bằng Gemini (có ví dụ và làm sạch)."""
    prompt = f"""Phân tích yêu cầu / hồ sơ tìm kiếm người thân thất lạc sau và trích xuất các từ khóa quan trọng có thể dùng để tìm kiếm thông tin liên quan đến người mất tích. Trả về một danh sách các từ khóa và những từ có khả năng liên quan. Lưu ý tên riêng có thể phân tích nhỏ hơn thành tên riêng (ví dụ: Lê Thị Hạnh => Hạnh). Từ khóa liên quan có thể được sinh ra từ các từ khóa chính (ví dụ: chiến tranh => xung đột, chạy giặc, vượt biên, di cư,...) hoặc từ các từ khóa khác trong đoạn văn bản. Vậy nhiệm vụ của bạn là trích xuất các từ khóa quan trọng nhất có thể dùng để tìm kiếm thông tin liên quan đến người mất tích và các từ khóa liên quan có thể sinh ra từ các từ khóa chính. Các từ khóa này có thể là tên riêng, địa danh, năm sinh, địa chỉ, đặc điểm nhận dạng, ký ức hoặc các thông tin khác... . Hãy trả về danh sách các từ khóa và các từ khóa liên quan có thể sinh ra từ các từ khóa chính, mỗi từ khóa cách nhau bởi dấu phẩy.

Ví dụ 1:
Đoạn văn bản: Chị Lê Thị Mỹ Duyên tìm bác Lê Viết Thi, đi vượt biên mất liên lạc khoảng năm 1978. Ông Lê Viết Thi sinh năm 1946, quê Quảng Nam. Bố mẹ là cụ Lê Viết Y và cụ Nguyễn Thị Ca. Anh chị em trong gia đình là Viết, Thơ, Dũng, Chung, Mười, Sỹ và Tượng. Khoảng năm 1978, ông Lê Viết Thi đi vượt biên. Từ đó, gia đình không còn nghe tin tức gì về ông.
Các từ khóa quan trọng: Lê Thị Mỹ Duyên, Duyên, Lê Viết Thi, Thi, vượt biên, di cư, chiến tranh, chạy giặc, 1978, 1946, Quảng Nam, Lê Viết Y, Y, Nguyễn Thị Ca, Ca, Viết, Thơ, Dũng, Chung, Mười, Sỹ, Tượng

Ví dụ 2:
Đoạn văn bản: Chị Lê Thị Toàn tìm anh Lê Văn Thương, mất liên lạc năm 1984 tại ga Đông Hà, Quảng Trị. Vào năm 1984, gia đình ông Tiên và bà Tẻo từ Thanh Hóa di chuyển vào Quảng Trị. Khi đến ga Đông Hà (Quảng Trị), vì hoàn cảnh quá khó khăn, ông Tiên bị tật ở chân, còn bà Tẻo không minh mẫn nên bà Tèo đã mang con trai Lê Văn Thương vừa mới sinh cho một người phụ nữ ở ga Đông Hà. Người phụ nữ đó có cho bà Tẻo một ít tiền rồi ôm anh Thương đi mất.
Các từ khóa quan trọng: Lê Thị Toàn, Toàn, Lê Văn Thương, Thương, 1984, Đông Hà, Quảng Trị, Tiên, Tẻo, Thanh Hóa, Quảng Trị, di chuyển, di cư, khó khăn, thiếu thốn, nghèo khổ, tật, khiếm khuyết, không minh mẫn, thần kinh, tâm thần, mới sinh, sơ sinh, mới đẻ

Ví dụ 3:
Đoạn văn bản: Chị Nguyễn Thị Yến tìm ba Nguyễn Văn Đã mất liên lạc năm 1977. Ông Nguyễn Văn Đã, sinh năm 1939, không rõ quê quán. Khoảng năm 1970, bà Vũ Thị Hải gặp ông Nguyễn Văn Đã ở nông trường Sao Đỏ tại Mộc Châu, Sơn La. Ông Đã phụ trách lái xe lương thực cho nông trường. Sau khi sinh chị Yến, ông muốn đưa hai mẹ con về quê ông nhưng bà Hải biết ông Đã đã có vợ ở quê nên không đồng ý và đem con về khu tập thể nhà máy nước Nam Định. Ông Đã vẫn thường lái xe về thăm con gái. Năm 1979, bà Hải mang con về quê bà sinh sống, từ đó chị Yến không hay tin gì về ba nữa.
Các từ khóa quan trọng: Nguyễn Thị Yến, Yến, Nguyễn Văn Đã, Đã, 1977, 1939, 1970, Vũ Thị Hải, Hải, nông trường, Sao Đỏ, Mộc Châu, Sơn La, lái xe, lương thực, nông trường, làm nông, nông nghiệp, khu tập thể, nhà máy nước, Nam Định, 1979

*Chú ý: những từ khóa nào phổ biến, phổ thông quá thì bỏ qua như: gia đình, anh, em, vợ, chồng, tìm kiếm, thất lạc, mất tích, mất liên lạc, không rõ quê quán, không rõ địa chỉ, không rõ thông tin, không rõ năm sinh, không rõ đặc điểm nhận dạng, không rõ ký ức... . Và nên lấy các từ khóa, thông tin, từ những nội dung chi tiết hồ sơ trước khi thất lạc.

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