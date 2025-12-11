from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

from .embedding import initialize_vector_db
from .db_utils import fetch_profiles_from_db
from .search import search_combined_chroma, search_combined_qdrant, search_combined_pinecone
from .qdrant_helper import get_qdrant_client, get_qdrant_collection
from .config import USE_QDRANT, USE_PINECONE, USE_CHROMADB, QDRANT_COLLECTION_NAME
# from .pinecone_client import get_pinecone_index  # Đã comment - không dùng Pinecone nữa
import json
import requests
import time
from .config import PRIMARY_GOOGLE_API_KEY, MAX_RETRIES_LLM, INITIAL_RETRY_DELAY_LLM
import re
from queue_list.queue import add_user_to_queue, remove_user_from_queue, is_user_turn

class ProfileSearchAPIView(APIView):
    """
    API endpoint for searching profiles using a user query.
    POST body: { "query": "..." }
    """
    def moderate_content(self, query):
        """
        Kiểm duyệt nội dung tìm kiếm sử dụng Gemini API để đảm bảo nội dung phù hợp
        Trả về tuple (is_appropriate, feedback)
        - is_appropriate: True nếu nội dung phù hợp, False nếu không
        - feedback: Phản hồi từ hệ thống kiểm duyệt
        """
        try:
            # Kiểm tra API key
            if not PRIMARY_GOOGLE_API_KEY:
                print("Lỗi: PRIMARY_GOOGLE_API_KEY chưa được cấu hình trong config.py")
                return True, "Không thể kiểm duyệt nội dung do thiếu API key, cho phép tìm kiếm tạm thời"
                
            # Cấu hình API endpoint và headers
            api_endpoint = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
            headers = {
                "Content-Type": "application/json",
                "x-goog-api-key": PRIMARY_GOOGLE_API_KEY,
            }
            
            # Tạo prompt kiểm duyệt
            prompt = f"""
            Tưởng tượng bạn là một nhà kiểm duyệt các nội dung tìm kiếm cho một chương trình tìm kiếm người thất lạc.
            Hãy kiểm duyệt nội dung truy vấn tìm kiếm dưới đây và xác định xem nó có phù hợp để sử dụng trong hệ thống tìm kiếm người thất lạc hay không.
            
            Nội dung cần kiểm duyệt:
            {query}
            
            Hãy kiểm tra các tiêu chí sau:
            1. Không chứa nội dung bạo lực, phân biệt chủng tộc, tôn giáo, giới tính
            2. Không chứa ngôn từ xúc phạm, thô tục
            3. Không chứa thông tin cá nhân nhạy cảm không liên quan đến việc tìm kiếm người thất lạc
            4. Không chứa nội dung quảng cáo, spam
            5. Không chứa nội dung lừa đảo, giả mạo
            
            Trả về kết quả dưới dạng JSON với các trường sau:
            {{
                "is_appropriate": true/false,
                "feedback": "Lý do tại sao nội dung (không) phù hợp"
            }}
            * Lưu ý: Không nên quá khắt khe trong nội dung kiểm duyệt. Nội dùng tìm kiếm khá là đa dạng nên không phải lúc nào xuất
            hiện những từ ngữ liên quan đến tiêu chuẩn là bị cho là không phù hợp. Hãy phân tích nội dung thật kỹ để tránh đưa ra những kết luận quá 
            khắt khe khiến cho việc xử lý tìm kiếm cho người dùng gặp thất bại.
            Ví dụ: 
            + Tìm kiếm con lai thì là nội dung bình thường, không phải là nội dung kì thị - phân biệt
            + Những hồ sơ nào có liên quan đến đặc điểm nhận dạng trên cơ thể cũng là nội dung bình thường, không phải là nội dung thô tục, khiêu dâm, ...
            """
            
            # Cấu hình payload với safety settings
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0.5,
                    "maxOutputTokens": 1024,
                },
            }
            
            # Gọi API với cơ chế retry
            for attempt in range(MAX_RETRIES_LLM):
                try:
                    response = requests.post(api_endpoint, headers=headers, json=payload, timeout=60)
                    
                    # Xử lý các mã lỗi
                    if response.status_code == 429:
                        error_type = "Rate Limit (429)"
                    elif response.status_code >= 500:
                        error_type = f"Server Error ({response.status_code})"
                        try:
                            error_detail = response.json().get('error', {}).get('message', response.text)
                            print(f"  Server error detail: {error_detail}")
                        except json.JSONDecodeError:
                            print(f"  Server error response (non-JSON): {response.text}")
                    elif response.status_code != 200:
                        error_type = f"HTTP Error {response.status_code}"
                        error_detail = "Unknown error"
                        try:
                            error_json = response.json().get('error', {})
                            error_detail = error_json.get('message', response.text)
                            if "API key not valid" in error_detail:
                                print(f"Lỗi API Key không hợp lệ (Key: ...{PRIMARY_GOOGLE_API_KEY[-4:]}). Ngừng thử lại.")
                                return True, "Không thể kiểm duyệt nội dung do lỗi API key, cho phép tìm kiếm tạm thời"
                        except json.JSONDecodeError:
                            error_detail = response.text
                        print(f"Lỗi không thể thử lại ({error_type}) khi gọi Gemini API: {error_detail}")
                        return True, "Không thể kiểm duyệt nội dung do lỗi API, cho phép tìm kiếm tạm thời"
                    
                    # Xử lý lỗi có thể thử lại
                    if response.status_code == 429 or response.status_code >= 500:
                        if attempt < MAX_RETRIES_LLM - 1:
                            wait_time = INITIAL_RETRY_DELAY_LLM * (2 ** attempt)
                            print(f"Lỗi '{error_type}'. Thử lại sau {wait_time} giây... (Lần {attempt+1}/{MAX_RETRIES_LLM})")
                            time.sleep(wait_time)
                            continue
                        else:
                            print(f"Không thể kiểm duyệt nội dung sau {MAX_RETRIES_LLM} lần thử do lỗi '{error_type}'.")
                            return True, "Không thể kiểm duyệt nội dung do lỗi API, cho phép tìm kiếm tạm thời"
                    
                    # Xử lý phản hồi thành công
                    if response.status_code == 200:
                        response_data = response.json()
                        
                        # Kiểm tra xem có bị block do safety settings không
                        if response_data.get('promptFeedback', {}).get('blockReason'):
                            block_reason = response_data['promptFeedback']['blockReason']
                            print(f"Cảnh báo: Yêu cầu bị chặn do safety settings: {block_reason}")
                            return True, "Không thể kiểm duyệt nội dung do bị chặn bởi safety settings, cho phép tìm kiếm tạm thời"
                        
                        # Trích xuất text từ phản hồi
                        try:
                            generated_text = response_data['candidates'][0]['content']['parts'][0]['text']
                            
                            if generated_text:
                                # Tìm và trích xuất phần JSON từ phản hồi
                                try:
                                    # Tìm chuỗi JSON trong phản hồi
                                    json_match = re.search(r'({[\s\S]*})', generated_text)
                                    if json_match:
                                        json_str = json_match.group(1)
                                        moderation_result = json.loads(json_str)
                                        return moderation_result.get('is_appropriate', True), moderation_result.get('feedback', "Nội dung phù hợp")
                                    else:
                                        print("Không tìm thấy chuỗi JSON trong phản hồi của LLM")
                                        return True, "Không thể phân tích kết quả kiểm duyệt, cho phép tìm kiếm tạm thời"
                                except json.JSONDecodeError as e:
                                    print(f"Không thể phân tích JSON từ phản hồi LLM: {e}")
                                    print(f"Phản hồi gốc: {generated_text}")
                                    return True, "Không thể phân tích kết quả kiểm duyệt, cho phép tìm kiếm tạm thời"
                            else:
                                print("Gemini API trả về phản hồi thành công nhưng text rỗng.")
                                return True, "Không thể kiểm duyệt nội dung do phản hồi rỗng, cho phép tìm kiếm tạm thời"
                        except (KeyError, IndexError, TypeError) as e:
                            print(f"Lỗi khi phân tích response thành công từ Gemini API: {e}")
                            print(f"Response data: {response_data}")
                            return True, "Không thể kiểm duyệt nội dung do lỗi phân tích phản hồi, cho phép tìm kiếm tạm thời"
                
                except requests.exceptions.RequestException as e:
                    # Xử lý lỗi mạng
                    error_type = f"Network Error ({type(e).__name__})"
                    if attempt < MAX_RETRIES_LLM - 1:
                        wait_time = INITIAL_RETRY_DELAY_LLM * (2 ** attempt)
                        print(f"Lỗi '{error_type}'. Thử lại sau {wait_time} giây... (Lần {attempt+1}/{MAX_RETRIES_LLM})")
                        time.sleep(wait_time)
                    else:
                        print(f"Không thể kiểm duyệt nội dung sau {MAX_RETRIES_LLM} lần thử do lỗi '{error_type}'.")
                        return True, "Không thể kiểm duyệt nội dung do lỗi mạng, cho phép tìm kiếm tạm thời"
            
            return True, "Không thể kiểm duyệt nội dung sau nhiều lần thử, cho phép tìm kiếm tạm thời"  # Trả về True nếu vòng lặp kết thúc mà không thành công
                
        except Exception as e:
            print(f"Lỗi khi kiểm duyệt nội dung: {str(e)}")
            return True, f"Không thể kiểm duyệt nội dung do lỗi: {str(e)}, cho phép tìm kiếm tạm thời"
    
    def post(self, request):
        user_id = request.user.id
        # Thêm người dùng vào hàng đợi
        add_user_to_queue(user_id)

        # Chờ đến lượt của người dùng
        while not is_user_turn(user_id):
            time.sleep(0.1)  # Chờ 100ms rồi kiểm tra lại

        try:

            user_query = request.data.get("query", "").strip()
            if not user_query:
                return Response({"error": "Missing or empty 'query'."}, status=status.HTTP_400_BAD_REQUEST)

            # Kiểm duyệt nội dung trước khi xử lý
            # is_appropriate, feedback = self.moderate_content(user_query)
            # if not is_appropriate:
            #     # Trả về lỗi nếu nội dung không phù hợp
            #     return Response({
            #         "error": f"Nội dung tìm kiếm không phù hợp: {feedback}"
            #     }, status=status.HTTP_400_BAD_REQUEST)

            # Initialize vector DB và fetch profiles
            df = fetch_profiles_from_db()
            if df.empty:
                return Response({"error": "No profiles found in database."}, status=status.HTTP_404_NOT_FOUND)

            # Run the search - ưu tiên Qdrant, sau đó Pinecone, cuối cùng ChromaDB
            try:
                if USE_QDRANT:
                    # Sử dụng Qdrant
                    qdrant_client = get_qdrant_client()
                    collection_name = get_qdrant_collection()
                    if collection_name is None:
                        return Response({"error": "Qdrant collection unavailable."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
                    id_list = search_combined_qdrant(
                        df, qdrant_client, collection_name, user_query, top_n_final=100, return_json=True, user=self.request.user
                    )
                elif USE_PINECONE:
                    # Sử dụng Pinecone (đã comment - không dùng nữa)
                    # from .pinecone_client import get_pinecone_index
                    # index = get_pinecone_index()
                    # if index is None:
                    #     return Response({"error": "Pinecone index unavailable."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
                    # from .search import search_combined_pinecone
                    # id_list = search_combined_pinecone(
                    #     df, index, user_query, top_n_final=100, return_json=True, user=self.request.user
                    # )
                    return Response({"error": "Pinecone is disabled. Please use Qdrant."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
                else:
                    # Sử dụng ChromaDB (đã comment - không dùng nữa)
                    # collection = initialize_vector_db()
                    # if collection is None:
                    #     return Response({"error": "Vector DB unavailable."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
                    # id_list = search_combined_chroma(
                    #     df, collection, user_query, top_n_final=100, return_json=True, user=self.request.user
                    # )
                    return Response({"error": "ChromaDB is disabled. Please use Qdrant."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
                if not id_list:
                    return Response({"results": []})

                # Kiểm tra xem id_list là list of dicts (từ return_json=True) hay list of IDs
                if isinstance(id_list, list) and len(id_list) > 0 and isinstance(id_list[0], dict):
                    # Đã là list of dicts từ search function (return_json=True)
                    detailed_results = []
                    for result_dict in id_list:
                        # Lấy các field từ result_dict (đã có đầy đủ từ search function)
                        detailed_results.append({
                            "id": result_dict.get('id', ''),
                            "title": result_dict.get('title', ''),
                            "full_name": result_dict.get('full_name', ''),
                            "losing_year": result_dict.get('losing_year', ''),
                            "born_year": result_dict.get('born_year', ''),
                            "name_of_father": result_dict.get('name_of_father', ''),
                            "name_of_mother": result_dict.get('name_of_mother', ''),
                            "siblings": result_dict.get('siblings', ''),
                            "detail": result_dict.get('detail', ''),
                        })
                else:
                    # Là list of IDs, cần build detailed_results từ DataFrame
                    detailed_results = []
                    for idx in id_list:
                        # idx may be string or int, ensure correct type for DataFrame lookup
                        try:
                            profile = df.loc[int(idx)] if int(idx) in df.index else df[df['id'] == int(idx)].iloc[0]
                        except Exception:
                            continue

                        detailed_results.append({
                            "id": profile.get('id', ''),
                            "title": profile.get('Tiêu đề', ''),
                            "full_name": profile.get('Họ và tên', ''),
                            "losing_year": profile.get('Năm thất lạc', ''),
                            "born_year": profile.get('Năm sinh', ''),
                            "name_of_father": profile.get('Tên cha', ''),
                            "name_of_mother": profile.get('Tên mẹ', ''),
                            "siblings": profile.get('Anh chị em', ''),
                            "detail": str(profile.get('Chi tiet_merged', '')),
                        })

                print("Tìm theo truy vấn detailed_results:", detailed_results)

                return Response({"results": detailed_results})
            except Exception as e:
                return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            
        finally:
            # Xóa người dùng khỏi hàng đợi sau khi tìm kiếm xong
            remove_user_from_queue(user_id)