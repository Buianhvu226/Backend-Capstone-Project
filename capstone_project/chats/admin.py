from django.contrib import admin
from .models import ChatSession, ChatParticipant, Message

class ChatParticipantInline(admin.TabularInline):
    model = ChatParticipant
    extra = 0

class MessageInline(admin.TabularInline):
    model = Message
    extra = 0
    readonly_fields = ['created_at']
    ordering = ['-created_at']

@admin.register(ChatSession)
class ChatSessionAdmin(admin.ModelAdmin):
    list_display = ['id', 'created_at', 'updated_at', 'get_participants']
    inlines = [ChatParticipantInline, MessageInline]
    readonly_fields = ['created_at', 'updated_at']
    
    def get_participants(self, obj):
        return ", ".join([p.user.username for p in obj.participants.all()])
    get_participants.short_description = 'Participants'

@admin.register(Message)
class MessageAdmin(admin.ModelAdmin):
    list_display = ['id', 'sender', 'content_preview', 'created_at', 'is_read', 'session']
    list_filter = ['is_read', 'created_at']
    search_fields = ['content', 'sender__username', 'session__id']
    readonly_fields = ['created_at']
    
    def content_preview(self, obj):
        return obj.content[:50] + '...' if len(obj.content) > 50 else obj.content
    content_preview.short_description = 'Content'
