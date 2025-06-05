from rest_framework import generics, status, permissions
from rest_framework.decorators import api_view, permission_classes
from rest_framework.response import Response
from django.shortcuts import get_object_or_404
from django.db.models import Q
from .models import RecentlyMissingReport, MissingPersonMatchResult
from .serializers import (
    RecentlyMissingReportSerializer,
    RecentlyMissingReportListSerializer,
    RecentlyMissingReportCreateSerializer,
    RecentlyMissingReportUpdateSerializer,
    MissingPersonMatchResultSerializer
)
import requests
import numpy as np
from PIL import Image
from io import BytesIO
import face_recognition
import chromadb
from sklearn.neighbors import KDTree
from concurrent.futures import ThreadPoolExecutor
import queue
import cv2
import google.generativeai as genai
import json
from rest_framework import permissions, status
from notifications.utils import create_notification
from rest_framework.permissions import AllowAny
from rest_framework.permissions import IsAuthenticated
from django.db.models import Count
from django.conf import settings
from pathlib import Path

genai.configure(api_key="AIzaSyCN_flhR6pXNOvQWjZSMAwe_t1DnI_O8IM")
task_queue = queue.Queue()

def get_chroma_collection():
    # client = chromadb.PersistentClient(path="F:\\Capstone-Project\\BE\\capstone_project\\chroma_db_face")

    # Tạo đường dẫn trực tiếp từ BASE_DIR
    chroma_db_path = Path(settings.BASE_DIR) / "chroma_db_face"

    # Đảm bảo thư mục tồn tại
    chroma_db_path.mkdir(parents=True, exist_ok=True)

    # Khởi tạo client
    client = chromadb.PersistentClient(path=str(chroma_db_path))
    return client.get_or_create_collection("face_report")

def encode_face_from_url(image_url, scale=0.5):
    try:
        response = requests.get(image_url)
        if response.status_code != 200:
            raise ValueError("Không thể tải ảnh từ URL.")
        img = Image.open(BytesIO(response.content)).convert('RGB')
        img_array = np.array(img)
        small_image = cv2.resize(img_array, (0, 0), fx=scale, fy=scale)
        face_locations = face_recognition.face_locations(small_image, model="hog")
        if not face_locations:
            raise ValueError("Không phát hiện khuôn mặt trong ảnh.")
        face_encodings = face_recognition.face_encodings(small_image, face_locations)
        if not face_encodings:
            raise ValueError("Không thể mã hóa khuôn mặt.")
        top, right, bottom, left = face_locations[0]
        top, right, bottom, left = [int(coord / scale) for coord in (top, right, bottom, left)]
        face_image = img_array[top:bottom, left:right]
        return face_encodings[0].tolist(), face_image
    except Exception as e:
        raise ValueError(f"Lỗi mã hóa khuôn mặt: {str(e)}")

def save_embedding_to_chroma(report_id, embedding_vector):
    try:
        collection = get_chroma_collection()
        collection.add(
            ids=[str(report_id)],
            embeddings=[embedding_vector],
            metadatas=[{"report_id": report_id}]
        )
    except Exception as e:
        raise RuntimeError(f"Lỗi khi lưu vào ChromaDB: {str(e)}")

def match_face_task(report_id, embedding, collection, k=10):
    try:
        results = collection.query(
            query_embeddings=[embedding],
            n_results=k
        )
        matches = []
        for distance, metadata in zip(results['distances'][0], results['metadatas'][0]):
            matched_report_id = int(metadata['report_id'])
            if matched_report_id != report_id and distance <= 0.22:  # Độ giống >= 78%
                match_score = round((1 - distance) * 100, 2)
                if not MissingPersonMatchResult.objects.filter(
                    report1_id=min(report_id, matched_report_id),
                    report2_id=max(report_id, matched_report_id)
                ).exists():
                    # Gán llm_confidence theo thang đánh giá mới
                    if 78 <= match_score < 82:
                        confidence = 'uncertain'
                    elif 82 <= match_score < 85:
                        confidence = 'moderate'
                    elif 85 <= match_score < 88:
                        confidence = 'fairly_confident'
                    elif 88 <= match_score < 92:
                        confidence = 'confident'
                    else:  # match_score >= 92
                        confidence = 'highly_confident'
                        # Tạo thông báo cho người dùng khi match_score >= 92
                    matches.append({
                        'report1_id': min(report_id, matched_report_id),
                        'report2_id': max(report_id, matched_report_id),
                        'face_match_score': match_score,
                        'llm_confidence': confidence
                    })
        return matches
    except Exception as e:
        print(f"Lỗi khi so khớp khuôn mặt: {str(e)}")
        return []

def find_and_save_similar_reports(report_id, embedding):
    try:
        collection = get_chroma_collection()
        with ThreadPoolExecutor(max_workers=4) as executor:
            future = executor.submit(match_face_task, report_id, embedding, collection)
            task_queue.put(future)
            matches = future.result()
        
        # Tạo các match result và gửi thông báo nếu match_score >= 92
        for match in matches:
            # Tạo match result
            match_result = MissingPersonMatchResult.objects.create(**match)
            
            # Gửi thông báo cho người dùng khi match_score >= 92
            if match['face_match_score'] >= 92:
                # Lấy thông tin các báo cáo
                current_report = RecentlyMissingReport.objects.get(id=report_id)
                other_report = RecentlyMissingReport.objects.get(id=match['report2_id'] if match['report1_id'] == report_id else match['report1_id'])
                
                # Gửi thông báo cho chủ sở hữu của báo cáo hiện tại (báo cáo vừa được upload ảnh)
                if hasattr(current_report, 'user') and current_report.user:
                    create_notification(
                        user=current_report.user,
                        notification_type='high_face_match',
                        content=f'Hệ thống đã tìm thấy một kết quả khớp khuôn mặt cao ({match["face_match_score"]:.1f}%) giữa báo cáo "{current_report.title}" của bạn và báo cáo "{other_report.title}".',
                        related_entity_id=match_result.id,
                        additional_data={
                            'match_id': match_result.id,
                            'your_report_id': current_report.id,
                            'your_report_title': current_report.title,
                            'matching_report_id': other_report.id,
                            'matching_report_title': other_report.title,
                            'face_match_score': match['face_match_score'],
                            'match_type': 'face_recognition'
                        }
                    )
                
                # Gửi thông báo cho chủ sở hữu của báo cáo khớp
                if hasattr(other_report, 'user') and other_report.user:
                    create_notification(
                        user=other_report.user,
                        notification_type='high_face_match',
                        content=f'Hệ thống đã tìm thấy một kết quả khớp khuôn mặt cao ({match["face_match_score"]:.1f}%) giữa báo cáo "{other_report.title}" của bạn và báo cáo "{current_report.title}".',
                        related_entity_id=match_result.id,
                        additional_data={
                            'match_id': match_result.id,
                            'your_report_id': other_report.id,
                            'your_report_title': other_report.title,
                            'matching_report_id': current_report.id,
                            'matching_report_title': current_report.title,
                            'face_match_score': match['face_match_score'],
                            'match_type': 'face_recognition'
                        }
                    )
        
        while not task_queue.empty():
            task_queue.get().result()
    except Exception as e:
        raise RuntimeError(f"Lỗi khi tìm kiếm và lưu kết quả khớp: {str(e)}")

def analyze_reports_with_gemini(current_report, other_reports):
    """Gọi Gemini API để phân tích và so sánh các báo cáo với thông tin chi tiết hơn."""
    model = genai.GenerativeModel('gemini-2.0-flash')
    
    # Hàm helper để format thông tin liên hệ CHI TIẾT
    def format_contact_info_detailed(report):
        contact_info = {
            'formatted_text': "Không có thông tin",
            'contact_list': [],
            'family_structure': {},
            'contact_count': 0
        }
        
        if hasattr(report, 'contact_persons_list') and report.contact_persons_list:
            # Sử dụng contact_persons_list nếu có
            contact_info['contact_list'] = report.contact_persons_list
            contact_info['contact_count'] = len(report.contact_persons_list)
            
            # Phân tích cấu trúc gia đình
            family_roles = {}
            for contact in report.contact_persons_list:
                relationship = contact.get('relationship', 'Không xác định')
                name = contact.get('name', 'Không có tên')
                family_roles[relationship] = name
            
            contact_info['family_structure'] = family_roles
            contact_info['formatted_text'] = "; ".join([f"{c['relationship']}: {c['name']}" for c in report.contact_persons_list])
            
        elif hasattr(report, 'contact_persons') and report.contact_persons:
            try:
                # Fallback cho contact_persons dict/json
                if isinstance(report.contact_persons, dict):
                    contacts = report.contact_persons
                elif isinstance(report.contact_persons, str):
                    contacts = json.loads(report.contact_persons)
                else:
                    contacts = {}
                
                contact_info['family_structure'] = contacts
                contact_info['contact_count'] = len(contacts)
                contact_info['formatted_text'] = "; ".join([f"{k}: {v}" for k, v in contacts.items()])
                
                # Convert to list format
                contact_info['contact_list'] = [{'relationship': k, 'name': v} for k, v in contacts.items()]
                
            except:
                contact_info['formatted_text'] = str(report.contact_persons)
        
        return contact_info
    
    # Hàm helper để format thông tin báo cáo với phân tích gia đình
    def format_report_info_enhanced(report):
        contact_details = format_contact_info_detailed(report)
        
        # Phân tích cấu trúc gia đình
        family_analysis = ""
        if contact_details['contact_count'] > 0:
            family_analysis = f"""
  - Số lượng người liên hệ: {contact_details['contact_count']}
  - Cấu trúc gia đình: {', '.join(contact_details['family_structure'].keys())}
  - Chi tiết người liên hệ: {contact_details['formatted_text']}"""
        else:
            family_analysis = "\n  - Thông tin liên hệ: Không có thông tin"
        
        return f"""- ID: {report.id}
  - Tiêu đề: {getattr(report, 'title', 'Không có tiêu đề')}
  - Loại hồ sơ: {getattr(report, 'profile_type', 'Không xác định')} ({'Người tìm kiếm' if getattr(report, 'profile_type', '') == 'seeker' else 'Người cung cấp thông tin' if getattr(report, 'profile_type', '') == 'provider' else 'Không xác định'})
  - Tên: {report.name or 'Không xác định'}
  - Tuổi: {report.age or 'Không xác định'}
  - Ngày mất tích: {report.missing_date or 'Không xác định'}
  - Địa điểm: {report.location or 'Không xác định'}
  - Mô tả chi tiết: {report.description or 'Không có mô tả'}{family_analysis}
  - Trạng thái: {getattr(report, 'status', 'Không xác định')}
  - Ngày tạo báo cáo: {getattr(report, 'created_at', 'Không xác định')}
  - Có ảnh: {'Có' if getattr(report, 'image_url', None) else 'Không'}"""

    # Format thông tin báo cáo hiện tại
    current_info = format_report_info_enhanced(current_report)
    
    # Format thông tin các báo cáo khác
    reports_list = "\n\n".join([format_report_info_enhanced(report) for report in other_reports])
    
    prompt = f"""
Bạn là chuyên gia phân tích báo cáo người mất tích với nhiều năm kinh nghiệm. Nhiệm vụ của bạn là so sánh báo cáo hiện tại với từng báo cáo trong danh sách để tìm ra mức độ tương đồng và khả năng cùng là một người.

**CÁC YẾU TỐ CẦN PHÂN TÍCH CHI TIẾT:**
1. **Thông tin cá nhân**: Tên, tuổi, giới tính (nếu có)
2. **Thời gian**: Ngày mất tích, thời gian tạo báo cáo (phần này không quá quan trọng vì thời gian mất tích có thể khác nhau)
3. **Địa danh được nhắc đến**: Nơi mất tích, khu vực sinh sống, trường học, nơi sinh hoạt,...
4. **Đặc điểm ngoại hình**: Chiều cao, cân nặng, màu tóc, đặc điểm đặc biệt
5. **Trang phục**: Quần áo mặc lần cuối
6. **Hành vi/Thói quen**: Các hoạt động thường ngày, thói quen hành vi cá nhân
7. **Sức khỏe**: Tình trạng sức khỏe, bệnh lý đặc biệt, đặc biệt là tình trạng tâm thần
8. **Yếu tố khác**: Các yếu tố khác mà bạn thấy có thể ảnh hưởng đến khả năng cùng là một người
9. **THÔNG TIN GIA ĐÌNH VÀ NGƯỜI LIÊN HỆ** (QUAN TRỌNG):
   - So sánh cấu trúc gia đình (có cùng các mối quan hệ không?)
   - So sánh tên người liên hệ (có trùng nhau không?)
   - Phân tích số lượng người liên hệ (gia đình lớn/nhỏ)
   - Kiểm tra logic mối quan hệ (ví dụ: báo cáo A có "con trai: Nguyễn Văn X", báo cáo B có "cha: Nguyễn Văn X")

**HƯỚNG DẪN PHÂN TÍCH GIA ĐÌNH:**
- Nếu 2 báo cáo có cùng người liên hệ với tên giống nhau → khả năng cao cùng gia đình
- Nếu báo cáo A có "con trai: X" và báo cáo B có "cha: X" → có thể cùng người nhưng từ góc nhìn khác nhau
- Nếu cấu trúc gia đình hoàn toàn khác nhau → ít khả năng cùng người
- Chú ý các mối quan hệ đảo ngược (ví dụ: "mẹ của A" vs "con của B")

BÁOO CÁO HIỆN TẠI:
{current_info}

DANH SÁCH BÁO CÁO CẦN SO SÁNH:
{reports_list}

**YÊU CẦU ĐỊNH DẠNG JSON CHÍNH XÁC:**
Hãy trả về kết quả theo định dạng JSON mẫu sau đây với các key bằng tiếng Việt (riêng key "summary" cần phải có và phải giống cấu trúc bên dưới):

```json
[
  {{
    "report_id": 1,
    "analysis": {{
      "Mức độ tương đồng về tên": "Phân tích so sánh tên - giống/khác ở đâu, mức độ tương đồng",
      "Mức độ tương đồng về tuổi": "So sánh tuổi - chênh lệch bao nhiêu, có hợp lý không",
      "Mức độ tương đồng về địa danh": "So sánh địa danh - cùng khu vực hay khác xa, khoảng cách, hoặc là các địa danh được nhắc đến có sự tương đồng nào không",
      "Mức độ tương đồng về thời gian": "So sánh thời gian mất tích - có cách xa nhau quá không (thực tế độ tương đồng về thời gian không quá quan trọng)",
      "Đặc điểm ngoại hình": "So sánh đặc điểm ngoại hình - những điểm giống và khác",
      "Trang phục": "So sánh trang phục - có trùng khớp gì không",
      "Phân tích thông tin gia đình": "PHÂN TÍCH CHI TIẾT: So sánh cấu trúc gia đình, tên người liên hệ, mối quan hệ, số lượng thành viên, có người liên hệ trùng nhau không, có logic mối quan hệ đảo ngược không",
      "Hành vi và thói quen": "So sánh hành vi/thói quen - có điểm chung nào",
      "Tình trạng sức khỏe": "So sánh tình trạng sức khỏe - có vấn đề gì giống nhau",
      "Yếu tố khác": "Các yếu tố khác đáng chú ý trong quá trình so sánh",
      "summary": {{
        "conclusion": "match|partial|no_match",
        "conclusion_text": "Tóm tắt ngắn gọn về kết luận cuối cùng, ưu tiên đề cập đến thông tin gia đình nếu có điểm đáng chú ý"
      }}
    }}
  }}
]
```

**LƯU Ý QUAN TRỌNG:**
- **Thông tin gia đình là yếu tố quyết định**: Nếu có cùng người liên hệ hoặc mối quan hệ logic → xem xét kỹ khả năng "match"
- Ưu tiên phân tích các trường hợp có thể cùng gia đình nhưng báo cáo từ người khác nhau
- Chú ý các mối quan hệ có thể đảo ngược hoặc nhìn từ góc độ khác
- Key "Phân tích thông tin gia đình" là BẮT BUỘC và phải phân tích chi tiết
- Conclusion cần cân nhắc đặc biệt thông tin gia đình khi đưa ra kết luận
- "match" là khi các thông tin hầu như đều khớp, "partial" là khi có những điểm chung nhưng không hoàn toàn khớp, "no_match" là khi không có điểm chung hoặc khớp quá ít
"""

    # Phần còn lại của hàm giữ nguyên...
    try:
        response = model.generate_content(prompt)
        result_text = response.text.strip()
        
        # Tìm và trích xuất JSON từ response
        import re
        json_pattern = r'```json\s*(\[.*?\])\s*```'
        json_match = re.search(json_pattern, result_text, re.DOTALL)
        
        if json_match:
            json_text = json_match.group(1)
            try:
                final_results = json.loads(json_text)
                
                # Validate và ensure summary format
                for result in final_results:
                    if 'analysis' in result and 'summary' in result['analysis']:
                        summary = result['analysis']['summary']
                        if not isinstance(summary, dict):
                            result['analysis']['summary'] = {
                                "conclusion": "no_match",
                                "conclusion_text": "Lỗi định dạng summary"
                            }
                        else:
                            # Ensure required fields
                            if 'conclusion' not in summary:
                                summary['conclusion'] = 'no_match'
                            if 'conclusion_text' not in summary:
                                summary['conclusion_text'] = 'Không có kết luận chi tiết'
                            
                            # Validate conclusion values
                            if summary['conclusion'] not in ['match', 'partial', 'no_match']:
                                summary['conclusion'] = 'no_match'
                    else:
                        # Add missing summary
                        if 'analysis' not in result:
                            result['analysis'] = {}
                        result['analysis']['summary'] = {
                            "conclusion": "no_match",
                            "conclusion_text": "Thiếu thông tin phân tích"
                        }
                
                return final_results
            except json.JSONDecodeError as je:
                print(f"Lỗi parse JSON: {str(je)}")
                print(f"JSON text: {json_text}")
        else:
            # Fallback logic giữ nguyên...
            try:
                json_start = result_text.find('[')
                json_end = result_text.rfind(']') + 1
                if json_start != -1 and json_end > json_start:
                    json_text = result_text[json_start:json_end]
                    final_results = json.loads(json_text)
                    
                    # Apply same validation as above
                    for result in final_results:
                        if 'analysis' not in result:
                            result['analysis'] = {}
                        if 'summary' not in result['analysis']:
                            result['analysis']['summary'] = {
                                "conclusion": "no_match",
                                "conclusion_text": "Phân tích không đầy đủ"
                            }
                        else:
                            summary = result['analysis']['summary']
                            if not isinstance(summary, dict):
                                result['analysis']['summary'] = {
                                    "conclusion": "no_match",
                                    "conclusion_text": "Lỗi định dạng summary"
                                }
                            else:
                                if 'conclusion' not in summary:
                                    summary['conclusion'] = 'no_match'
                                if 'conclusion_text' not in summary:
                                    summary['conclusion_text'] = 'Không có kết luận chi tiết'
                                if summary['conclusion'] not in ['match', 'partial', 'no_match']:
                                    summary['conclusion'] = 'no_match'
                    
                    return final_results
            except (json.JSONDecodeError, ValueError):
                pass
        
        # Nếu không parse được JSON, tạo response mặc định
        print(f"Không thể parse JSON từ Gemini response: {result_text}")
        return [
            {
                "report_id": report.id,
                "analysis": {
                    "parsing_error": f"Lỗi parse JSON. Raw response: {result_text[:200]}...",
                    "Mức độ tương đồng về tên": "Lỗi phân tích tự động",
                    "Mức độ tương đồng về tuổi": "Lỗi phân tích tự động", 
                    "Mức độ tương đồng về địa điểm": "Lỗi phân tích tự động",
                    "Mức độ tương đồng về thời gian": "Lỗi phân tích tự động",
                    "Phân tích thông tin gia đình": "Lỗi phân tích thông tin gia đình",
                    "summary": {
                        "conclusion": "no_match",
                        "conclusion_text": "Lỗi trong quá trình phân tích tự động"
                    }
                }
            } 
            for report in other_reports
        ]
        
    except Exception as e:
        print(f"Lỗi khi gọi Gemini API: {str(e)}")
        return [
            {
                "report_id": report.id,
                "analysis": {
                    "api_error": f"Lỗi API: {str(e)}",
                    "Mức độ tương đồng về tên": f"Lỗi khi gọi AI: {str(e)}",
                    "Mức độ tương đồng về tuổi": f"Lỗi khi gọi AI: {str(e)}",
                    "Mức độ tương đồng về địa điểm": f"Lỗi khi gọi AI: {str(e)}",
                    "Mức độ tương đồng về thời gian": f"Lỗi khi gọi AI: {str(e)}",
                    "Phân tích thông tin gia đình": f"Lỗi phân tích gia đình: {str(e)}",
                    "summary": {
                        "conclusion": "no_match", 
                        "conclusion_text": f"Lỗi khi gọi AI: {str(e)}"
                    }
                }
            } 
            for report in other_reports
        ]

class RecentlyMissingReportListCreateView(generics.ListCreateAPIView):
    permission_classes = [permissions.IsAuthenticated]
    
    def get_serializer_class(self):
        if self.request.method == 'GET':
            return RecentlyMissingReportListSerializer
        return RecentlyMissingReportCreateSerializer
    
    def get_queryset(self):
        return RecentlyMissingReport.objects.filter(user=self.request.user)
    
    def perform_create(self, serializer):
        instance = serializer.save(user=self.request.user)
        image_url = self.request.data.get('image_url')
        if image_url:
            try:
                embedding, _ = encode_face_from_url(image_url)
                save_embedding_to_chroma(instance.id, embedding)
                find_and_save_similar_reports(instance.id, embedding)
            except Exception as e:
                print(f"Error processing embedding: {str(e)}")
    
    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        self.perform_create(serializer)
        instance = serializer.instance
        response_serializer = RecentlyMissingReportSerializer(instance, context={'request': request})
        return Response(response_serializer.data, status=status.HTTP_201_CREATED)

class RecentlyMissingReportPublicListView(generics.ListAPIView):
    serializer_class = RecentlyMissingReportListSerializer
    permission_classes = [permissions.AllowAny]
    
    def get_queryset(self):
        queryset = RecentlyMissingReport.objects.filter(status='active')
        profile_type = self.request.query_params.get('profile_type')
        if profile_type:
            queryset = queryset.filter(profile_type=profile_type)
        location = self.request.query_params.get('location')
        if location:
            queryset = queryset.filter(location__icontains=location)
        name = self.request.query_params.get('name')
        if name:
            queryset = queryset.filter(name__icontains=name)
        return queryset.order_by('-created_at')



class RecentlyMissingReportDetailView(generics.RetrieveUpdateDestroyAPIView):
    def get_serializer_class(self):
        if self.request.method == 'GET':
            return RecentlyMissingReportSerializer
        return RecentlyMissingReportUpdateSerializer
    
    def get_permissions(self):
        if self.request.method == 'GET':
            return [permissions.AllowAny()]
        return [permissions.IsAuthenticated()]
    
    def get_queryset(self):
        if self.request.method == 'GET':
            return RecentlyMissingReport.objects.all()
        return RecentlyMissingReport.objects.filter(user=self.request.user)

    def destroy(self, request, *args, **kwargs):
        instance = self.get_object()
        if instance.user != request.user:
            return Response({"error": "Bạn không có quyền xóa báo cáo này."}, status=status.HTTP_403_FORBIDDEN)
        try:
            delete_embedding_from_chroma(instance.id)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        self.perform_destroy(instance)
        return Response(status=status.HTTP_204_NO_CONTENT)

@api_view(['POST'])
@permission_classes([permissions.IsAuthenticated])
def upload_missing_report_image(request, report_id):
    report = get_object_or_404(RecentlyMissingReport, id=report_id, user=request.user)
    image_url = request.data.get('image_url')
    if not image_url:
        return Response({'error': 'Thiếu URL ảnh.'}, status=status.HTTP_400_BAD_REQUEST)
    
    try:
        embedding, _ = encode_face_from_url(image_url)
        save_embedding_to_chroma(report_id, embedding)
        find_and_save_similar_reports(report_id, embedding)
    except Exception as e:
        return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)
    
    report.image_url = image_url
    report.save()
    serializer = RecentlyMissingReportSerializer(report, context={'request': request})
    return Response({
        'message': 'Upload ảnh và lưu embedding thành công.',
        'report': serializer.data
    }, status=status.HTTP_200_OK)

@api_view(['GET'])
@permission_classes([permissions.AllowAny])
def get_missing_report_matches(request, report_id):
    try:
        report = get_object_or_404(RecentlyMissingReport, id=report_id)
        matches = MissingPersonMatchResult.objects.filter(
            Q(report1=report) | Q(report2=report)
        ).order_by('-created_at')
        serializer = MissingPersonMatchResultSerializer(matches, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)
    except Exception as e:
        return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['POST'])
@permission_classes([permissions.IsAuthenticated])
def update_match_status(request, match_id):
    try:
        match = get_object_or_404(MissingPersonMatchResult, id=match_id)
        if request.user not in [match.report1.user, match.report2.user]:
            return Response({'error': 'Bạn không có quyền cập nhật kết quả khớp này'}, status=status.HTTP_403_FORBIDDEN)
        new_status = request.data.get('status')
        if new_status not in ['accepted', 'rejected', 'pending']:
            return Response({'error': 'Trạng thái không hợp lệ'}, status=status.HTTP_400_BAD_REQUEST)
        match.status = new_status
        match.save()
        serializer = MissingPersonMatchResultSerializer(match)
        return Response(serializer.data, status=status.HTTP_200_OK)
    except Exception as e:
        return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['POST'])
@permission_classes([permissions.IsAuthenticated])
def analyze_matches_with_llm(request):
    try:
        current_report_id = request.data.get('current_report_id')
        other_report_ids = request.data.get('other_report_ids', [])
        if not current_report_id or not other_report_ids:
            return Response({'error': 'Thiếu current_report_id hoặc other_report_ids'}, status=status.HTTP_400_BAD_REQUEST)
        
        # Convert current_report_id to integer
        try:
            current_report_id = int(current_report_id)
        except (ValueError, TypeError):
            return Response({'error': 'current_report_id phải là một số nguyên'}, status=status.HTTP_400_BAD_REQUEST)

        current_report = get_object_or_404(RecentlyMissingReport, id=current_report_id)
        other_reports = RecentlyMissingReport.objects.filter(id__in=other_report_ids)
        
        # Chia thành batch (10 báo cáo mỗi batch)
        batch_size = 10
        results = []
        for i in range(0, len(other_reports), batch_size):
            batch = other_reports[i:i + batch_size]
            batch_results = analyze_reports_with_gemini(current_report, batch)
            results.extend(batch_results)
        
        # Cập nhật kết quả vào MissingPersonMatchResult với cấu trúc JSON mới
        for result in results:
            report_id = result['report_id']
            match = MissingPersonMatchResult.objects.filter(
                Q(report1_id=min(current_report_id, report_id), report2_id=max(current_report_id, report_id))
            ).first()
            if match:
                # Chỉ lưu analysis object vào llm_analysis
                match.llm_analysis = result['analysis']
                match.save()
                # Cập nhật kết quả phân tích vào llm_confidence
                if result['analysis']['summary']['conclusion']:
                    match.llm_confidence = result['analysis']['summary']['conclusion']
                    match.save()
                # Gửi thông báo cho chủ sở hữu của hồ sơ được khớp nếu kết luận là "match"
                if result['analysis']['summary']['conclusion'] == 'match':
                    # Lấy thông tin báo cáo hiện tại và báo cáo khớp
                    other_report = RecentlyMissingReport.objects.get(id=report_id)
                    
                    # Gửi thông báo cho chủ sở hữu của báo cáo hiện tại
                    if hasattr(current_report, 'user') and current_report.user:
                        create_notification(
                            user=current_report.user,
                            notification_type='match_found',
                            content=f'AI đã tìm thấy một kết quả khớp hoàn toàn giữa báo cáo "{current_report.title}" của bạn và báo cáo "{other_report.title}".',
                            related_entity_id=match.id,
                            additional_data={
                                'match_id': match.id,
                                'your_report_id': current_report.id,
                                'your_report_title': current_report.title,
                                'matching_report_id': other_report.id,
                                'matching_report_title': other_report.title,
                                'face_match_score': match.face_match_score,
                                'match_conclusion': 'match'
                            }
                        )
                    
                    # Gửi thông báo cho chủ sở hữu của báo cáo khớp
                    if hasattr(other_report, 'user') and other_report.user:
                        create_notification(
                            user=other_report.user,
                            notification_type='match_found',
                            content=f'AI đã tìm thấy một kết quả khớp hoàn toàn giữa báo cáo "{other_report.title}" của bạn và báo cáo "{current_report.title}".',
                            related_entity_id=match.id,
                            additional_data={
                                'match_id': match.id,
                                'your_report_id': other_report.id,
                                'your_report_title': other_report.title,
                                'matching_report_id': current_report.id,
                                'matching_report_title': current_report.title,
                                'face_match_score': match.face_match_score,
                                'match_conclusion': 'match'
                            }
                        )


        
        return Response({'message': 'Phân tích LLM hoàn tất', 'results': results}, status=status.HTTP_200_OK)
    except Exception as e:
        import traceback
        print(f"Error in analyze_matches_with_llm: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

def delete_embedding_from_chroma(report_id):
    try:
        collection = get_chroma_collection()
        collection.delete(ids=[str(report_id)])
    except Exception as e:
        raise RuntimeError(f"Lỗi khi xóa embedding từ ChromaDB: {str(e)}")

# Thêm vào cuối file views.py

class MyReportsListView(generics.ListAPIView):
    """
    Endpoint chuyên dụng để lấy danh sách các report của user hiện tại
    """
    serializer_class = RecentlyMissingReportListSerializer
    permission_classes = [permissions.IsAuthenticated]
    
    def get_queryset(self):
        return RecentlyMissingReport.objects.filter(
            user=self.request.user
        ).order_by('-created_at')

# Thống kê
@api_view(['GET'])
@permission_classes([permissions.IsAuthenticated])
def statistics(request):
    """
    Endpoint để thống kê số lượng báo cáo theo trạng thái và số lượng cặp báo cáo ghép đôi theo trạng thái
    """
    # Thống kê số lượng báo cáo theo trạng thái
    report_stats = RecentlyMissingReport.objects.values('status').annotate(count=Count('id'))
    report_stats_dict = {}
    for stat in report_stats:
        report_stats_dict[stat['status']] = stat['count']
    
    # Đảm bảo tất cả trạng thái đều có trong kết quả, ngay cả khi không có báo cáo nào
    for status, _ in RecentlyMissingReport.STATUS_CHOICES:
        if status not in report_stats_dict:
            report_stats_dict[status] = 0
    
    # Thống kê số lượng cặp báo cáo ghép đôi theo trạng thái
    match_stats = MissingPersonMatchResult.objects.values('status').annotate(count=Count('id'))
    match_stats_dict = {}
    for stat in match_stats:
        match_stats_dict[stat['status']] = stat['count']
    
    # Đảm bảo tất cả trạng thái đều có trong kết quả, ngay cả khi không có cặp ghép đôi nào
    for status, _ in MissingPersonMatchResult.STATUS_CHOICES:
        if status not in match_stats_dict:
            match_stats_dict[status] = 0
    
    # Tổng số báo cáo và cặp ghép đôi
    total_reports = RecentlyMissingReport.objects.count()
    total_matches = MissingPersonMatchResult.objects.count()
    
    return Response({
        'reports': {
            'total': total_reports,
            'by_status': report_stats_dict
        },
        'matches': {
            'total': total_matches,
            'by_status': match_stats_dict
        }
    })