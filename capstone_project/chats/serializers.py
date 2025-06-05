from rest_framework import serializers
from django.contrib.auth import get_user_model
from .models import ChatSession, ChatParticipant, Message
from profiles.models import Profile
from recently_missing.models import RecentlyMissingReport  # Import model báo cáo người mất tích

User = get_user_model()

class MessageSerializer(serializers.ModelSerializer):
    class Meta:
        model = Message
        fields = ['id', 'sender', 'content', 'created_at', 'is_read']

class ChatSessionSerializer(serializers.ModelSerializer):
    participant_id = serializers.IntegerField(write_only=True)
    related_profile_id = serializers.IntegerField(required=False, allow_null=True)  # Không bắt buộc
    related_report_id = serializers.IntegerField(required=False, allow_null=True)   # Thêm trường related_report_id
    other_user = serializers.SerializerMethodField()
    unread_count = serializers.SerializerMethodField()
    last_message = serializers.SerializerMethodField()
    related_entity = serializers.SerializerMethodField()  # Thêm trường để hiển thị thông tin về entity liên quan
    
    class Meta:
        model = ChatSession
        fields = [
            'id', 'participant_id', 'related_profile_id', 'related_report_id', 
            'other_user', 'unread_count', 'created_at', 'updated_at', 
            'last_message', 'related_entity'
        ]
        read_only_fields = ['id']
    
    def get_other_user(self, obj):
        request = self.context.get('request')
        if not request or not hasattr(request, 'user'):
            return None
            
        current_user = request.user
        participant = obj.participants.exclude(user=current_user).first()
        if not participant:
            return None
            
        user = participant.user
        return {
            'id': user.id,
            'username': user.username,
            'first_name': user.first_name,
            'last_name': user.last_name,
            'avatar': user.profile.avatar.url if hasattr(user, 'profile') and user.profile.avatar else None
        }
    
    def get_related_entity(self, obj):
        """Trả về thông tin về entity liên quan (profile hoặc report)"""
        if obj.related_profile:
            # Sửa cách truy cập images
            # image_url = None
            # # Kiểm tra xem images có phải là RelatedManager không
            # if hasattr(obj.related_profile, 'images'):
            #     if hasattr(obj.related_profile.images, 'all'):  # Nếu là RelatedManager
            #         first_image = obj.related_profile.images.first()
            #         # Fix: Access the image field within the ProfileImage object
            #         if first_image:
            #             # Try different possible field names for the image
            #             if hasattr(first_image, 'image') and first_image.image:
            #                 image_url = first_image.image.url
            #             elif hasattr(first_image, 'url') and first_image.url:
            #                 image_url = first_image.url
            #             elif hasattr(first_image, 'file') and first_image.file:
            #                 image_url = first_image.file.url
            #             elif hasattr(first_image, 'str') and first_image.str:
            #                 image_url = first_image.str.url
            #     elif isinstance(obj.related_profile.images, list):  # Nếu là list
            #         if obj.related_profile.images:
            #             first_image = obj.related_profile.images[0]
            #             if hasattr(first_image, 'image') and first_image.image:
            #                 image_url = first_image.image.url
            #             elif hasattr(first_image, 'url'):
            #                 image_url = first_image.url
            #     elif isinstance(obj.related_profile.images, str):  # Nếu là string
            #         image_url = obj.related_profile.images
            
            return {
                'type': 'profile',
                'id': obj.related_profile.id,
                'title': obj.related_profile.title,
                # 'image_url': image_url
            }
        elif obj.related_report:
            return {
                'type': 'report',
                'id': obj.related_report.id,
                'title': obj.related_report.title,
                # 'image_url': obj.related_report.image_url if hasattr(obj.related_report, 'image_url') else None
            }
        return None
    
    def get_unread_count(self, obj):
        request = self.context.get('request')
        if not request or not hasattr(request, 'user'):
            return 0
            
        return obj.messages.filter(is_read=False).exclude(sender=request.user).count()
    
    def get_last_message(self, obj):
        last_message = obj.messages.order_by('-created_at').first()
        if not last_message:
            return None
            
        return {
            'id': last_message.id,
            'content': last_message.content,
            'sender_id': last_message.sender_id,
            'created_at': last_message.created_at
        }
    
    def validate(self, data):
        """Kiểm tra xem có ít nhất một trong hai trường related_profile_id hoặc related_report_id"""
        related_profile_id = data.get('related_profile_id')
        related_report_id = data.get('related_report_id')
        
        if not related_profile_id and not related_report_id:
            raise serializers.ValidationError(
                "Phải cung cấp ít nhất một trong hai trường: related_profile_id hoặc related_report_id"
            )
        
        if related_profile_id and related_report_id:
            raise serializers.ValidationError(
                "Chỉ được cung cấp một trong hai trường: related_profile_id hoặc related_report_id"
            )
            
        return data
    
    def create(self, validated_data):
        participant_id = validated_data.pop('participant_id')
        related_profile_id = validated_data.pop('related_profile_id', None)
        related_report_id = validated_data.pop('related_report_id', None)
        current_user = self.context['request'].user
        
        # Kiểm tra người dùng có tồn tại không
        try:
            other_user = User.objects.get(id=participant_id)
        except User.DoesNotExist:
            raise serializers.ValidationError({"participant_id": "User not found"})
        
        # Kiểm tra xem hồ sơ có tồn tại không (nếu related_profile_id được cung cấp)
        profile = None
        if related_profile_id:
            try:
                profile = Profile.objects.get(id=related_profile_id)
            except Profile.DoesNotExist:
                raise serializers.ValidationError({"related_profile_id": "Profile not found"})
        
        # Kiểm tra xem báo cáo có tồn tại không (nếu related_report_id được cung cấp)
        report = None
        if related_report_id:
            try:
                report = RecentlyMissingReport.objects.get(id=related_report_id)
            except RecentlyMissingReport.DoesNotExist:
                raise serializers.ValidationError({"related_report_id": "Report not found"})
        
        # Kiểm tra xem đã có session chung giữa hai người dùng chưa
        user_sessions = ChatParticipant.objects.filter(user=current_user).values_list('session', flat=True)
        existing_session = ChatParticipant.objects.filter(user=other_user, session__id__in=user_sessions).first()
        
        if existing_session:
            # Nếu đã có session chung, kiểm tra và cập nhật related_profile_id hoặc related_report_id nếu cần
            session = existing_session.session
            
            if related_profile_id and not session.related_profile_id:
                session.related_profile = profile
                session.save()
            
            if related_report_id and not session.related_report_id:
                session.related_report = report
                session.save()
                
            return session
        
        # Tạo session mới nếu chưa có
        chat_session = ChatSession.objects.create(
            related_profile=profile,
            related_report=report
        )
        
        # Thêm người dùng hiện tại vào session
        ChatParticipant.objects.create(session=chat_session, user=current_user)
        
        # Thêm người dùng được chọn vào session
        ChatParticipant.objects.create(session=chat_session, user=other_user)
        
        return chat_session