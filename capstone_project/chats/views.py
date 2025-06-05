from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from django.db.models import Q, Max, Prefetch
from .models import ChatSession, Message, ChatParticipant
from .serializers import ChatSessionSerializer, MessageSerializer
from notifications.models import Notification
from django.views.decorators.csrf import csrf_exempt
from rest_framework.decorators import api_view

class ChatSessionViewSet(viewsets.ModelViewSet):
    serializer_class = ChatSessionSerializer
    permission_classes = [IsAuthenticated]
    
    def get_queryset(self):
        # Chỉ lấy các session mà người dùng hiện tại tham gia
        user_sessions = ChatParticipant.objects.filter(user=self.request.user).values_list('session', flat=True)
        return ChatSession.objects.filter(id__in=user_sessions).order_by('-updated_at')
    
    def get_serializer_context(self):
        context = super().get_serializer_context()
        context.update({"request": self.request})
        return context
    
    @action(detail=True, methods=['get'])
    def messages(self, request, pk=None):
        """Lấy tất cả tin nhắn trong một phiên chat"""
        session = self.get_object()
        messages = session.messages.all()
        
        # Đánh dấu các tin nhắn chưa đọc của người dùng khác là đã đọc
        unread_messages = messages.filter(is_read=False).exclude(sender=request.user)
        unread_count = unread_messages.count()
        
        if unread_count > 0:
            unread_messages.update(is_read=True)
            
        serializer = MessageSerializer(messages, many=True)
        return Response(serializer.data)
    
    @action(detail=True, methods=['post'])
    def send_message(self, request, pk=None):
        """Gửi tin nhắn mới trong một phiên chat"""
        session = self.get_object()
        content = request.data.get('content')
        
        if not content or content.strip() == '':
            return Response({"error": "Message content is required"}, status=status.HTTP_400_BAD_REQUEST)
        
        # Tạo tin nhắn mới
        message = Message.objects.create(
            session=session,
            sender=request.user,
            content=content.strip()
        )
        
        # Cập nhật thời gian session
        session.save()  # .save() sẽ tự động cập nhật updated_at
        
        # Tạo thông báo cho người dùng khác trong phiên chat
        other_participants = session.participants.exclude(user=request.user)
        
        # Xác định loại entity và thông tin liên quan
        entity_info = ""
        additional_data = {}
        
        if session.related_profile:
            entity_type = "profile"
            entity_id = session.related_profile.id
            entity_title = session.related_profile.title
            entity_info = f" về hồ sơ '{entity_title}'"
            additional_data = {
                'entity_type': 'profile',
                'entity_id': entity_id,
                'entity_title': entity_title
            }
        elif session.related_report:
            entity_type = "report"
            entity_id = session.related_report.id
            entity_title = session.related_report.title
            entity_info = f" về báo cáo '{entity_title}'"
            additional_data = {
                'entity_type': 'report',
                'entity_id': entity_id,
                'entity_title': entity_title
            }
        
        for participant in other_participants:
            Notification.objects.create(
                user=participant.user,
                type='message',
                content=f"Tin nhắn mới từ {request.user.get_full_name() or request.user.username}{entity_info}",
                related_entity_id=session.id,
                additional_data=additional_data
            )
        
        serializer = MessageSerializer(message)
        return Response(serializer.data, status=status.HTTP_201_CREATED)

class MessageViewSet(viewsets.ReadOnlyModelViewSet):
    """ViewSet chỉ đọc cho các tin nhắn"""
    serializer_class = MessageSerializer
    permission_classes = [IsAuthenticated]
    
    def get_queryset(self):
        # Chỉ cho phép người dùng xem tin nhắn từ các phiên họ tham gia
        user_sessions = ChatParticipant.objects.filter(user=self.request.user).values_list('session', flat=True)
        return Message.objects.filter(session__id__in=user_sessions)
    
    @action(detail=True, methods=['post'])
    def mark_as_read(self, request, pk=None):
        """Đánh dấu tin nhắn đã đọc"""
        try:
            message = self.get_object()
            # Chỉ đánh dấu tin nhắn của người khác và chưa đọc
            if message.sender != request.user and not message.is_read:
                message.is_read = True
                message.save()
            return Response({"status": "message marked as read"})
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)

@api_view(['GET'])
def api_endpoints(request):
    """Liệt kê tất cả các API endpoints cho ứng dụng chat"""
    data = {
        'chat_sessions': {
            'list_create': request.build_absolute_uri('/api/chats/chatsessions/'),
            'retrieve': request.build_absolute_uri('/api/chats/chatsessions/{id}/'),
            'messages': request.build_absolute_uri('/api/chats/chatsessions/{id}/messages/'),
            'send_message': request.build_absolute_uri('/api/chats/chatsessions/{id}/send_message/'),
        },
        'messages': {
            'list': request.build_absolute_uri('/api/chats/messages/'),
            'retrieve': request.build_absolute_uri('/api/chats/messages/{id}/'),
            'mark_as_read': request.build_absolute_uri('/api/chats/messages/{id}/mark_as_read/'),
        }
    }
    return Response(data)