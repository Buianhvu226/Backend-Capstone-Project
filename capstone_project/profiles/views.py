from rest_framework import viewsets, permissions, status, filters
from rest_framework.decorators import action
from rest_framework.response import Response
from django.db.models import Q
from .models import Profile, ProfileMatchSuggestion, ProfileImage
from .serializers import ProfileSerializer, ProfileCreateSerializer, ProfileMatchSuggestionSerializer
from notifications.utils import create_notification
import json
import requests
from django.conf import settings
from vector_search.gem_vectorDB import get_embedding, initialize_vector_db, CHROMA_COLLECTION_NAME, DETAIL_COLUMN_NAME
from vector_search.db_utils import find_similar_profiles
from vector_search.views_api import ProfileSearchAPIView
from vector_search.db_utils import fetch_profiles_from_db
from vector_search.embedding import initialize_vector_db
from vector_search.search import search_combined_chroma
from rest_framework.permissions import AllowAny
from rest_framework.permissions import IsAuthenticated  # <-- Add this import
from django_filters.rest_framework import DjangoFilterBackend
import django_filters
from datetime import datetime
from django.db.models import Count

# Define ProfileFilter class
class ProfileFilter(django_filters.FilterSet):
    full_name = django_filters.CharFilter(lookup_expr='icontains')
    born_year = django_filters.CharFilter(lookup_expr='exact')
    losing_year = django_filters.CharFilter(lookup_expr='exact')
    name_of_father = django_filters.CharFilter(lookup_expr='icontains')
    name_of_mother = django_filters.CharFilter(lookup_expr='icontains')
    siblings = django_filters.CharFilter(lookup_expr='icontains')
    status = django_filters.CharFilter(lookup_expr='exact')

    class Meta:
        model = Profile
        fields = [
            'full_name',
            'born_year',
            'losing_year',
            'name_of_father',
            'name_of_mother',
            'siblings',
            'status'
        ]
    created_at_after = django_filters.DateTimeFilter(field_name='created_at', lookup_expr='gte')
    created_at_before = django_filters.DateTimeFilter(field_name='created_at', lookup_expr='lte')
    
    class Meta:
        model = Profile
        fields = ['status', 'created_at_after', 'created_at_before']

class IsOwnerOrReadOnly(permissions.BasePermission):
    def has_object_permission(self, request, view, obj):
        if request.method in permissions.SAFE_METHODS:
            return True
        return obj.user == request.user

class ProfileViewSet(viewsets.ModelViewSet):
    queryset = Profile.objects.all().order_by('-created_at') # Sắp xếp theo created_at giảm dần
    # permission_classes = [permissions.IsAuthenticated, IsOwnerOrReadOnly] # Xóa dòng này hoặc comment lại
    filter_backends = [filters.SearchFilter, filters.OrderingFilter, DjangoFilterBackend]
    search_fields = ['title', 'full_name', 'description']
    ordering_fields = ['created_at', 'born_year', 'losing_year']
    filterset_class = ProfileFilter
    def get_serializer_class(self):
        if self.action == 'create' or self.action == 'update' or self.action == 'partial_update':
            return ProfileCreateSerializer
        return ProfileSerializer

    def get_permissions(self):
        """
        Instantiates and returns the list of permissions that this view requires.
        """
        if self.action == 'list' or self.action == 'retrieve' or self.action == 'advanced_search': # Thêm 'advanced_search'
            permission_classes = [AllowAny]
        else:
            permission_classes = [permissions.IsAuthenticated, IsOwnerOrReadOnly]
        return [permission() for permission in permission_classes]
    
    def extract_profile_info(self, title, description):
        """
        Sử dụng LLM để trích xuất thông tin từ tiêu đề và mô tả
        """
        prompt = f"""
        Hãy trích xuất các thông tin sau từ đoạn văn bản dưới đây:
        - Họ và tên đầy đủ của người bị thất lạc
        - Tên các anh chị em (nếu có)
        - Tên của cha (nếu có)
        - Tên của mẹ (nếu có)
        - Năm sinh (nếu có)
        - Năm thất lạc (nếu có)

        Tiêu đề: {title}
        Mô tả: {description}

        Trả về kết quả dưới dạng JSON với các trường sau:
        {{
            "full_name": "Tên đầy đủ của người bị thất lạc",
            "siblings": "Tên các anh chị em, phân cách bằng dấu phẩy",
            "name_of_father": "Tên của cha",
            "name_of_mother": "Tên của mẹ",
            "born_year": "Năm sinh",
            "losing_year": "Năm thất lạc"
        }}

        Nếu không tìm thấy thông tin nào, hãy để trống giá trị tương ứng.
        """
        
        try:
            # Import các thư viện cần thiết từ vector_search
            from vector_search.config import PRIMARY_GOOGLE_API_KEY
            import json
            import requests
            import time
            
            # Kiểm tra API key
            if not PRIMARY_GOOGLE_API_KEY:
                print("Lỗi: PRIMARY_GOOGLE_API_KEY chưa được cấu hình trong config.py")
                return {}
                
            # Cấu hình API endpoint và headers
            api_endpoint = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
            headers = {
                "Content-Type": "application/json",
                "x-goog-api-key": PRIMARY_GOOGLE_API_KEY,
            }
            
            # Cấu hình payload
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0.1,
                    "maxOutputTokens": 256
                }
            }
            
            # Gọi API với cơ chế retry
            from vector_search.config import MAX_RETRIES_LLM, INITIAL_RETRY_DELAY_LLM
            
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
                                return {}
                        except json.JSONDecodeError:
                            error_detail = response.text
                        print(f"Lỗi không thể thử lại ({error_type}) khi gọi Gemini API: {error_detail}")
                        return {}
                    
                    # Xử lý lỗi có thể thử lại
                    if response.status_code == 429 or response.status_code >= 500:
                        if attempt < MAX_RETRIES_LLM - 1:
                            wait_time = INITIAL_RETRY_DELAY_LLM * (2 ** attempt)
                            print(f"Lỗi '{error_type}'. Thử lại sau {wait_time} giây... (Lần {attempt+1}/{MAX_RETRIES_LLM})")
                            time.sleep(wait_time)
                            continue
                        else:
                            print(f"Không thể trích xuất thông tin sau {MAX_RETRIES_LLM} lần thử do lỗi '{error_type}'.")
                            return {}
                    
                    # Xử lý phản hồi thành công
                    if response.status_code == 200:
                        response_data = response.json()
                        
                        # Kiểm tra xem có bị block do safety settings không
                        if response_data.get('promptFeedback', {}).get('blockReason'):
                            block_reason = response_data['promptFeedback']['blockReason']
                            print(f"Cảnh báo: Yêu cầu bị chặn do safety settings: {block_reason}")
                            return {}
                        
                        # Trích xuất text từ phản hồi
                        try:
                            generated_text = response_data['candidates'][0]['content']['parts'][0]['text']
                            
                            if generated_text:
                                # Tìm và trích xuất phần JSON từ phản hồi
                                try:
                                    # Tìm chuỗi JSON trong phản hồi
                                    import re
                                    json_match = re.search(r'({[\s\S]*})', generated_text)
                                    if json_match:
                                        json_str = json_match.group(1)
                                        extracted_data = json.loads(json_str)
                                        return extracted_data
                                    else:
                                        print("Không tìm thấy chuỗi JSON trong phản hồi của LLM")
                                        return {}
                                except json.JSONDecodeError as e:
                                    print(f"Không thể phân tích JSON từ phản hồi LLM: {e}")
                                    print(f"Phản hồi gốc: {generated_text}")
                                    return {}
                            else:
                                print("Gemini API trả về phản hồi thành công nhưng text rỗng.")
                                return {}
                        except (KeyError, IndexError, TypeError) as e:
                            print(f"Lỗi khi phân tích response thành công từ Gemini API: {e}")
                            print(f"Response data: {response_data}")
                            return {}
                
                except requests.exceptions.RequestException as e:
                    # Xử lý lỗi mạng
                    error_type = f"Network Error ({type(e).__name__})"
                    if attempt < MAX_RETRIES_LLM - 1:
                        wait_time = INITIAL_RETRY_DELAY_LLM * (2 ** attempt)
                        print(f"Lỗi '{error_type}'. Thử lại sau {wait_time} giây... (Lần {attempt+1}/{MAX_RETRIES_LLM})")
                        time.sleep(wait_time)
                    else:
                        print(f"Không thể trích xuất thông tin sau {MAX_RETRIES_LLM} lần thử do lỗi '{error_type}'.")
                        return {}
            
            return {}  # Trả về dict rỗng nếu vòng lặp kết thúc mà không thành công
                
        except Exception as e:
            print(f"Lỗi khi trích xuất thông tin: {str(e)}")
            return {}
    
    def generate_full_description(self, title, description, extracted_info):
        """
        Tạo mô tả đầy đủ từ tiêu đề, mô tả gốc và thông tin đã trích xuất
        """
        full_name = extracted_info.get('full_name', '')
        siblings = extracted_info.get('siblings', '')
        name_of_father = extracted_info.get('name_of_father', '')
        name_of_mother = extracted_info.get('name_of_mother', '')
        born_year = extracted_info.get('born_year', '')
        losing_year = extracted_info.get('losing_year', '')
        
        # Tạo phần thông tin cơ bản
        basic_info = []
        if born_year:
            basic_info.append(f"Năm sinh của người bị thất lạc là {born_year}")
        if losing_year:
            basic_info.append(f"thời gian thất lạc là vào {losing_year}")
        if name_of_father:
            basic_info.append(f"tên cha người bị thất lạc: {name_of_father}")
        if name_of_mother:
            basic_info.append(f"tên mẹ người bị thất lạc: {name_of_mother}")
        if siblings:
            basic_info.append(f"tên các anh chị em: {siblings}")
        
        # Kết hợp thành mô tả đầy đủ
        full_description = description
        
        # Thêm phần thông tin cơ bản nếu có
        if basic_info:
            basic_info_text = "Thông tin cơ bản: " + "; ".join(basic_info) + "."
            if not full_description.endswith('.'):
                full_description += '.'
            full_description += " " + basic_info_text
        
        return full_description
    
    def perform_create(self, serializer):
        title = serializer.validated_data.get('title', '')
        description = serializer.validated_data.get('description', '')

        # Trích xuất thông tin từ tiêu đề và mô tả
        extracted_info = self.extract_profile_info(title, description)

        # Tạo thông báo đang trong quá trình tạo hồ sơ
        from notifications.utils import create_notification
        create_notification(
            user=self.request.user,
            notification_type='profile_creating',
            content=f'Đang trích xuất thông tin từ tiêu đề và mô tả...',
            additional_data={
                'text': f'Đang trích xuất thông tin hồ sơ từ tiêu đề {title} và mô tả {description}',
            }
        )

        # Cập nhật các trường đã trích xuất vào validated_data (dù có hay không)
        for field in ['full_name', 'siblings', 'name_of_father', 'name_of_mother', 'born_year', 'losing_year']:
            serializer.validated_data[field] = extracted_info.get(field, "")

        # Tạo mô tả đầy đủ
        full_description = self.generate_full_description(title, description, extracted_info)
        serializer.validated_data['description'] = full_description

        # Thông báo đã tạo mô tả đầy đủ
        from notifications.utils import create_notification
        create_notification(
            user=self.request.user,
            notification_type='profile_creating',
            content=f'Hồ sơ đã được tạo thành công.',
            additional_data={
                'text': f'Hồ sơ đã được tạo thành công với mô tả đầy đủ: {full_description}',
            }
        )

        # Lưu profile
        profile = serializer.save(user=self.request.user)

        # Thông báo đã lưu profile
        from notifications.utils import create_notification
        create_notification(
            user=self.request.user,
            notification_type='profile_creating',
            content=f'Hồ sơ đã được lưu tạm thời thành công.',
            additional_data={
                'text': f'Hồ sơ đã được lưu tạm thời thành công với mô tả đầy đủ: {full_description}',
            }
        )
        
        # Xử lý embedding và vector search
        detail_text = getattr(profile, DETAIL_COLUMN_NAME, None)
        if not detail_text:
            detail_text = profile.description

        embedding = get_embedding(
            detail_text,
            task_type="RETRIEVAL_DOCUMENT"
        )

        if embedding:
            collection = initialize_vector_db()
            metadata = {
                "Tiêu đề": profile.title or "",
                "Họ và tên": profile.full_name or "",
                "Năm sinh": getattr(profile, "born_year", "") or "",
                "Năm thất lạc": getattr(profile, "losing_year", "") or "",
                "id": profile.id if profile.id is not None else "",
            }
            try:
                collection.upsert(
                    ids=[str(profile.id)],
                    embeddings=[embedding],
                    metadatas=[metadata]
                )
                print(f"Profile {profile.id} embedded and upserted into ChromaDB.")

                # Thông báo đã lưu profile
                from notifications.utils import create_notification
                create_notification(
                    user=self.request.user,
                    notification_type='profile_creating',
                    content=f'Hồ sơ {profile.title} đã được lưu vào ChromaDB thành công.',
                    additional_data={
                        'text': f'Hồ sơ {profile.title} đã được lưu vào ChromaDB thành công với mô tả đầy đủ',
                    }
                )
            except Exception as e:
                print(f"Error upserting profile {profile.id} into ChromaDB: {e}")
        else:
            print(f"Could not create embedding for profile {profile.id}.")
            
        # --- Gợi ý hồ sơ tương tự và lưu vào bảng ProfileMatchSuggestion ---
        from profiles.models import ProfileMatchSuggestion, Profile as ProfileModel
        from notifications.utils import create_notification
        from django.db.models import Q
        
        # Lấy toàn bộ profiles dưới dạng DataFrame
        collection = initialize_vector_db()
        if collection is None:
            return Response({"error": "Vector DB unavailable."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        df = fetch_profiles_from_db()
        if df.empty:
            return Response({"error": "No profiles found in database."}, status=status.HTTP_404_NOT_FOUND)
        else:
            from notifications.utils import create_notification
            create_notification(
                user=self.request.user,
                notification_type='profile_creating',
                content=f'Đang tải toàn bộ hồ sơ từ cơ sở dữ liệu...',
                additional_data={
                    'text': f'Đang tải toàn bộ hồ sơ từ cơ sở dữ liệu...',
                }
            )
        
        # Sử dụng mô tả chi tiết để tìm kiếm các hồ sơ phù hợp nhất
        id_list = search_combined_chroma(
            df, collection, detail_text, top_n_final=22, return_json=True, user=self.request.user
        )
        if not id_list:
            return Response({"results": []})

        # For each ID, build a detailed result dict
        detailed_results = []
        for idx in id_list:
            # idx may be string or int, ensure correct type for DataFrame lookup
            try:
                profile_row = df.loc[int(idx)] if int(idx) in df.index else df[df['id'] == int(idx)].iloc[0]
            except Exception:
                continue

            detailed_results.append({
                "id": profile_row .get('id', ''),
                "title": profile_row .get('Tiêu đề', ''),
                "full_name": profile_row .get('Họ và tên', ''),
                "losing_year": profile_row .get('Năm thất lạc', ''),
                "born_year": profile_row .get('Năm sinh', ''),
                "name_of_father": profile_row .get('Tên cha', ''),
                "name_of_mother": profile_row .get('Tên mẹ', ''),
                "siblings": profile_row .get('Anh chị em', ''),
                "detail": str(profile_row .get('Chi tiet_merged', '')),
            })

        detailed_results = [result for result in detailed_results if result['id'] != str(profile.id)]
        print("Tìm theo truy vấn detailed_results (trong quá trình tạo):", detailed_results)
        
        # Lấy các đối tượng Profile tương ứng
        similar_profiles = ProfileModel.objects.filter(id__in=[int(result['id']) for result in detailed_results])
        
        # Tạo thông báo cho người tạo hồ sơ mới
        if similar_profiles:
            # Danh sách thông tin của các hồ sơ khớp để hiển thị trong thông báo
            match_info = [{"id": p.id, "title": p.title} for p in similar_profiles]
            
            # Gửi thông báo cho người tạo hồ sơ
            create_notification(
                user=self.request.user,
                notification_type='profile_created_with_matches',
                content=f'Hồ sơ "{profile.title}" đã được tạo thành công với {len(similar_profiles) - 1} gợi ý phù hợp.',
                related_entity_id=profile.id,
                additional_data={
                    'matching_profiles': match_info,
                    'profile_id': profile.id
                }
            )
        else:
            # Thông báo khi không có hồ sơ phù hợp
            create_notification(
                user=self.request.user,
                notification_type='profile_created',
                content=f'Hồ sơ "{profile.title}" đã được tạo thành công.',
                related_entity_id=profile.id
            )
        
        # Thêm từng cặp vào bảng ProfileMatchSuggestion nếu chưa tồn tại
        # và gửi thông báo cho chủ sở hữu của hồ sơ phù hợp
        for similar in similar_profiles:
            # Bỏ qua nếu profile1 và profile2 là cùng một hồ sơ phù hợp
            if profile.id == similar.id:
                continue
                
            # Kiểm tra xem đã tồn tại gợi ý nào chưa
            exists = ProfileMatchSuggestion.objects.filter(
                Q(profile1=profile, profile2=similar) | 
                Q(profile1=similar, profile2=profile)
            ).exists()
            
            if not exists:
                # Tạo gợi ý match
                suggestion = ProfileMatchSuggestion.objects.create(
                    profile1=profile,
                    profile2=similar
                )
                
                # Gửi thông báo cho chủ sở hữu của hồ sơ được khớp
                if hasattr(similar, 'user') and similar.user:
                    create_notification(
                        user=similar.user,
                        notification_type='new_match',
                        content=f'Có hồ sơ mới "{profile.title}" phù hợp với hồ sơ "{similar.title}" của bạn.',
                        related_entity_id=profile.id,
                        additional_data={
                            'matching_profile_id': profile.id,
                            'matching_profile_title': profile.title,
                            'your_profile_id': similar.id,
                            'your_profile_title': similar.title,
                            'suggestion_id': suggestion.id
                        }
                    )
        
        # Lưu lại danh sách các hồ sơ tương tự để sử dụng trong create() method
        self.similar_profiles = similar_profiles
        return profile

    def create(self, request, *args, **kwargs):
        response = super().create(request, *args, **kwargs)
        similar_profiles = getattr(self, 'similar_profiles', [])
        serializer = ProfileSerializer(similar_profiles, many=True, context={'request': request})
        response.data['suggested_profiles'] = serializer.data
        return response

    def destroy(self, request, *args, **kwargs):
        instance = self.get_object()
        profile_id = str(instance.id)
        # Delete from ChromaDB
        try:
            collection = initialize_vector_db()
            collection.delete(ids=[profile_id])
            print(f"Đã xóa profile {profile_id} khỏi ChromaDB.")
        except Exception as e:
            print(f"Lỗi khi xóa profile {profile_id} khỏi ChromaDB: {e}")
        # Delete from database
        return super().destroy(request, *args, **kwargs)
    
    @action(detail=True, methods=['get'])
    def suggested_profiles(self, request, pk=None):
        profile = self.get_object() # Đây là profile có id = pk (id truy vấn)
        
        # Tìm các gợi ý trong đó profile hiện tại (pk) là profile1
        # và lấy ra các profile2 tương ứng làm hồ sơ gợi ý.
        suggestions = ProfileMatchSuggestion.objects.filter(profile1=profile)
        
        result = []
        for suggestion in suggestions:
            # Hồ sơ được gợi ý chính là profile2 của bản ghi suggestion
            result.append(suggestion.profile2)
        
        # Luôn truyền request vào context để tính toán is_owner
        serializer = ProfileSerializer(result, many=True, context={'request': request})
        return Response(serializer.data)
    
    @action(detail=False, methods=['get'], permission_classes=[IsAuthenticated])
    def all_suggested_profiles(self, request):
        """
        Lấy tất cả các hồ sơ được gợi ý từ các hồ sơ của user hiện tại.
        Mỗi hồ sơ gợi ý sẽ có thêm thông tin về hồ sơ gốc đã tạo ra gợi ý.
        """
        # Lấy ID của tất cả hồ sơ thuộc về user hiện tại
        user_profile_ids = Profile.objects.filter(user=request.user).values_list('id', flat=True)
        
        # Lấy tất cả các đề xuất match từ các hồ sơ của user
        suggestions = ProfileMatchSuggestion.objects.filter(profile1_id__in=user_profile_ids).select_related(
            'profile1', 'profile2'
        ).order_by('-created_at')
        
        # Xử lý phân trang
        page = self.paginate_queryset(suggestions)
        if page is not None:
            suggestions = page
        
        # Tạo kết quả
        result = []
        for suggestion in suggestions:
            # Serialize profile2 (hồ sơ được gợi ý)
            profile_data = ProfileSerializer(suggestion.profile2, context={'request': request}).data
            
            # Thêm thông tin về profile1 (hồ sơ gốc)
            profile_data['suggested_from_profile'] = {
                'id': suggestion.profile1.id,
                'title': suggestion.profile1.title,
                'created_at': suggestion.profile1.created_at
            }
            
            # Thêm thông tin về suggestion
            profile_data['suggestion_info'] = {
                'id': suggestion.id,
                'match_status': suggestion.match_status,
                'created_at': suggestion.created_at
            }
            
            result.append(profile_data)
        
        # Trả về kết quả với phân trang nếu có
        if hasattr(self, 'get_paginated_response') and page is not None:
            return self.get_paginated_response(result)
        
        # Nếu không phân trang
        return Response(result)
    
    @action(detail=False, methods=['get'], permission_classes=[IsAuthenticated])
    def all_referenced_profiles(self, request):
        """
        Lấy tất cả các hồ sơ tham chiếu đến các hồ sơ của user hiện tại.
        Ngược với all_suggested_profiles, endpoint này trả về các hồ sơ của người khác
        mà được gợi ý đến hồ sơ của user hiện tại.
        Mỗi hồ sơ tham chiếu sẽ có thêm thông tin về hồ sơ của user được tham chiếu.
        """
        # Lấy ID của tất cả hồ sơ thuộc về user hiện tại
        user_profile_ids = Profile.objects.filter(user=request.user).values_list('id', flat=True)
        
        # Lấy tất cả các đề xuất match mà profile2 là hồ sơ của user hiện tại
        # Điều này ngược với all_suggested_profiles (lấy profile1_id__in)
        suggestions = ProfileMatchSuggestion.objects.filter(profile2_id__in=user_profile_ids).select_related(
            'profile1', 'profile2'
        ).order_by('-created_at')
        
        # Xử lý phân trang
        page = self.paginate_queryset(suggestions)
        if page is not None:
            suggestions = page
        
        # Tạo kết quả
        result = []
        for suggestion in suggestions:
            # Serialize profile1 (hồ sơ tham chiếu - của người khác)
            profile_data = ProfileSerializer(suggestion.profile1, context={'request': request}).data
            
            # Thêm thông tin về profile2 (hồ sơ của user hiện tại được tham chiếu)
            profile_data['referenced_to_profile'] = {
                'id': suggestion.profile2.id,
                'title': suggestion.profile2.title,
                'created_at': suggestion.profile2.created_at
            }
            
            # Thêm thông tin về suggestion
            profile_data['suggestion_info'] = {
                'id': suggestion.id,
                'match_status': suggestion.match_status,
                'created_at': suggestion.created_at
            }
            
            result.append(profile_data)
        
        # Trả về kết quả với phân trang nếu có
        if hasattr(self, 'get_paginated_response') and page is not None:
            return self.get_paginated_response(result)
        
        # Nếu không phân trang
        return Response(result)
    
    @action(detail=False, methods=['get'])
    def my_profiles(self, request):
        profiles = Profile.objects.filter(user=request.user).order_by('-created_at') # Sắp xếp theo created_at giảm dần
        serializer = ProfileSerializer(profiles, many=True, context={'request': request})
        return Response(serializer.data)
    
    @action(detail=False, methods=['post'])
    def search_by_description(self, request):
        description = request.data.get('description', '')
        if not description:
            return Response({"error": "Description is required"}, status=status.HTTP_400_BAD_REQUEST)
        
        # Create a temporary profile object to generate embedding
        temp_profile = Profile(description=description)
        similar_profiles = find_similar_profiles(temp_profile, top_k=10)
        
        serializer = ProfileSerializer(similar_profiles, many=True, context={'request': request})
        return Response(serializer.data)

    @action(detail=False, methods=['get'], permission_classes=[IsAuthenticated])
    def my_all_profiles(self, request):
        """
        Lấy danh sách hồ sơ do user hiện tại tạo, có phân trang.
        """
        queryset = Profile.objects.filter(user=request.user).order_by('-created_at')
        page = self.paginate_queryset(queryset)
        if page is not None:
            serializer = ProfileSerializer(page, many=True, context={'request': request})
            return self.get_paginated_response(serializer.data)
        serializer = ProfileSerializer(queryset, many=True, context={'request': request})
        return Response(serializer.data)

    def retrieve(self, request, *args, **kwargs):
        """
        Override phương thức retrieve để thêm context và xử lý logic xác thực chủ sở hữu
        """
        instance = self.get_object()
        serializer = self.get_serializer(instance, context={'request': request})
        
        data = serializer.data
        
        # Cung cấp thông tin truy cập vào API debug cho admin
        if request.user.is_authenticated and request.user.is_staff:
            data['_debug'] = {
                'profile_user_id': instance.user.id if instance.user else None,
                'request_user_id': request.user.id,
                'timestamp': str(datetime.now()),
            }
        
        return Response(data)

    @action(detail=False, methods=['get'], permission_classes=[IsAuthenticated])
    def matching_notifications(self, request):
        """
        Lấy danh sách thông báo về hồ sơ mới phù hợp
        """
        from notifications.models import Notification
        from notifications.serializers import NotificationSerializer
        
        # Lấy tất cả thông báo loại 'new_match' của user hiện tại
        notifications = Notification.objects.filter(
            user=request.user,
            type='new_match',
            is_read=False
        ).order_by('-created_at')
        
        # Phân trang nếu cần
        page = self.paginate_queryset(notifications)
        if page is not None:
            serializer = NotificationSerializer(page, many=True)
            return self.get_paginated_response(serializer.data)
        
        serializer = NotificationSerializer(notifications, many=True)
        return Response(serializer.data)

    @action(detail=True, methods=['post'])
    def start_conversation(self, request, pk=None):
        from chats.models import ChatParticipant, ChatSession
        from notifications.models import Notification
        
        profile = self.get_object()
        
        # Nếu người dùng đang cố liên hệ với profile của chính mình
        if request.user.id == profile.user.id:
            return Response(
                {"error": "Bạn không thể bắt đầu cuộc trò chuyện với chính mình"}, 
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Kiểm tra xem đã có cuộc trò chuyện nào giữa hai người dùng chưa
        user_participations = ChatParticipant.objects.filter(user=request.user).values_list('session_id', flat=True)
        other_user_participations = ChatParticipant.objects.filter(user=profile.user).values_list('session_id', flat=True)
        
        common_sessions = set(user_participations).intersection(set(other_user_participations))
        
        if common_sessions:
            # Trả về session ID nếu đã tồn tại
            session = ChatSession.objects.get(id=list(common_sessions)[0])
            return Response({"session_id": session.id})
        
        # Tạo chat session mới
        session = ChatSession.objects.create(related_profile=profile)
        
        # Thêm người tham gia
        ChatParticipant.objects.create(session=session, user=request.user)
        ChatParticipant.objects.create(session=session, user=profile.user)
        
        # Tạo thông báo cho người dùng kia
        Notification.objects.create(
            user=profile.user,
            type='message',
            content=f"{request.user.get_full_name() or request.user.username} đã bắt đầu cuộc trò chuyện với bạn",
            related_entity_id=session.id
        )
        
        return Response({"session_id": session.id}, status=status.HTTP_201_CREATED)

    @action(detail=True, methods=['post'], permission_classes=[IsAuthenticated])
    def upload_image(self, request, pk=None):
        """
        Nhận URL hình ảnh từ client và lưu vào bảng ProfileImage
        """
        profile = self.get_object()
        
        # Kiểm tra quyền sở hữu profile
        if request.user != profile.user:
            return Response(
                {"error": "Bạn không có quyền thêm ảnh vào hồ sơ này"}, 
                status=status.HTTP_403_FORBIDDEN
            )
        
        # Lấy URL và mô tả từ request
        image_url = request.data.get('image_url')
        description = request.data.get('description', '')
        
        if not image_url:
            return Response({"error": "URL hình ảnh là bắt buộc"}, status=status.HTTP_400_BAD_REQUEST)
        
        # Kiểm tra xem profile đã có ảnh chưa, nếu có thì cập nhật thay vì tạo mới
        profile_image = ProfileImage.objects.filter(profile=profile).first()
        
        if profile_image:
            # Cập nhật ảnh đã tồn tại
            profile_image.image = image_url
            profile_image.description = description
            profile_image.save()
            message = "Cập nhật ảnh hồ sơ thành công"
        else:
            # Tạo bản ghi ProfileImage mới
            profile_image = ProfileImage.objects.create(
                profile=profile,
                image=image_url,
                description=description
            )
            message = "Thêm ảnh hồ sơ thành công"
        
        # Trả về thông tin ảnh đã lưu
        return Response({
            "message": message,
            "image_data": {
                "id": profile_image.id,
                "image_url": profile_image.image,
                "description": profile_image.description,
                "created_at": profile_image.created_at
            }
        }, status=status.HTTP_200_OK)

    @action(detail=False, methods=['get'], permission_classes=[permissions.IsAuthenticated])
    def statistics(self, request):
        """
        Endpoint để thống kê số lượng hồ sơ theo trạng thái và số lượng cặp hồ sơ ghép đôi theo trạng thái
        """
        # Thống kê số lượng hồ sơ theo trạng thái
        profile_stats = Profile.objects.values('status').annotate(count=Count('id'))
        profile_stats_dict = {}
        for stat in profile_stats:
            profile_stats_dict[stat['status']] = stat['count']
        
        # Đảm bảo tất cả trạng thái đều có trong kết quả, ngay cả khi không có hồ sơ nào
        for status, _ in Profile.STATUS_CHOICES:
            if status not in profile_stats_dict:
                profile_stats_dict[status] = 0
        
        # Thống kê số lượng cặp hồ sơ ghép đôi theo trạng thái
        match_stats = ProfileMatchSuggestion.objects.values('match_status').annotate(count=Count('id'))
        match_stats_dict = {}
        for stat in match_stats:
            match_stats_dict[stat['match_status']] = stat['count']
        
        # Đảm bảo tất cả trạng thái đều có trong kết quả, ngay cả khi không có cặp ghép đôi nào
        for status, _ in ProfileMatchSuggestion.STATUS_CHOICES:
            if status not in match_stats_dict:
                match_stats_dict[status] = 0
        
        # Tổng số hồ sơ và cặp ghép đôi
        total_profiles = Profile.objects.count()
        total_matches = ProfileMatchSuggestion.objects.count()
        
        return Response({
            'profiles': {
                'total': total_profiles,
                'by_status': profile_stats_dict
            },
            'matches': {
                'total': total_matches,
                'by_status': match_stats_dict
            }
        })

class ProfileMatchSuggestionViewSet(viewsets.ReadOnlyModelViewSet):
    serializer_class = ProfileMatchSuggestionSerializer
    permission_classes = [permissions.IsAuthenticated]
    
    def get_queryset(self):
        user = self.request.user
        user_profiles = Profile.objects.filter(user=user).values_list('id', flat=True)
        return ProfileMatchSuggestion.objects.filter(
            Q(profile1_id__in=user_profiles) | Q(profile2_id__in=user_profiles)
        )
    
    @action(detail=True, methods=['post'])
    def update_status(self, request, pk=None):
        suggestion = self.get_object()
        status = request.data.get('status')
        
        if status not in ['accepted', 'rejected']:
            return Response({"error": "Invalid status"}, status=status.HTTP_400_BAD_REQUEST)
        
        suggestion.match_status = status
        suggestion.save()
        
        serializer = self.get_serializer(suggestion)
        return Response(serializer.data)
