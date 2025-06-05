import os
from django.conf import settings
import json
import requests
from .models import Notification

def create_notification(user, notification_type, content, related_entity_id=None, additional_data=None):
    """
    Tạo thông báo trong database
    
    Args:
        user: User nhận thông báo
        notification_type: Loại thông báo ('match', 'profile_created', 'new_match', etc.)
        content: Nội dung thông báo
        related_entity_id: ID của đối tượng liên quan (nếu có)
        additional_data: Dữ liệu bổ sung dưới dạng dict (các ID, titles, etc.)
    """
    # Tạo thông báo trong database - lưu vào PostgreSQL của Supabase
    notification = Notification.objects.create(
        user=user,
        type=notification_type,
        content=content,
        related_entity_id=str(related_entity_id) if related_entity_id else None,
        is_read=False,
        additional_data=additional_data or {}
    )
    return notification

def mark_notification_as_read(notification_id):
    """
    Đánh dấu thông báo đã đọc
    """
    try:
        notification = Notification.objects.get(id=notification_id)
        notification.is_read = True
        notification.save()
        return True
    except Notification.DoesNotExist:
        return False

def mark_all_notifications_as_read(user_id):
    """
    Đánh dấu tất cả thông báo của user là đã đọc
    """
    Notification.objects.filter(user_id=user_id, is_read=False).update(is_read=True)
    return True