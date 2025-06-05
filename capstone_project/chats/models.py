from django.db import models
from django.conf import settings
from profiles.models import Profile
from recently_missing.models import RecentlyMissingReport

class ChatSession(models.Model):
    """Chat sessions between users"""
    id = models.IntegerField(primary_key=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    related_profile = models.ForeignKey(
        Profile, 
        on_delete=models.SET_NULL, 
        null=True, 
        blank=True,
        help_text="Profile that initiated the chat"
    )
    related_report = models.ForeignKey(
        RecentlyMissingReport,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        help_text="Missing person report that initiated the chat"
    )
    
    def __str__(self):
        if self.related_profile:
            return f"Chat Session {self.id} (Profile: {self.related_profile.id})"
        elif self.related_report:
            return f"Chat Session {self.id} (Report: {self.related_report.id})"
        return f"Chat Session {self.id}"
    
    def save(self, *args, **kwargs):
        if self.id is None:
            last = ChatSession.objects.order_by('-id').first()
            self.id = 1 if last is None else last.id + 1
        super().save(*args, **kwargs)

class ChatParticipant(models.Model):
    """Participants in a chat session"""
    id = models.IntegerField(primary_key=True)
    session = models.ForeignKey(ChatSession, on_delete=models.CASCADE, related_name='participants')
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,  # Thay thế User bằng settings.AUTH_USER_MODEL
        on_delete=models.CASCADE
    )
    joined_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        unique_together = ['session', 'user']
    
    def __str__(self):
        return f"{self.user.username} in session {self.session.id}"
    
    def save(self, *args, **kwargs):
        if self.id is None:
            last = ChatParticipant.objects.order_by('-id').first()
            self.id = 1 if last is None else last.id + 1
        super().save(*args, **kwargs)

class Message(models.Model):
    """Messages in a chat session"""
    id = models.IntegerField(primary_key=True)
    session = models.ForeignKey(ChatSession, on_delete=models.CASCADE, related_name='messages')
    sender = models.ForeignKey(
        settings.AUTH_USER_MODEL,  # Thay thế User bằng settings.AUTH_USER_MODEL
        on_delete=models.CASCADE
    )
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    is_read = models.BooleanField(default=False)
    
    class Meta:
        ordering = ['created_at']
    
    def __str__(self):
        return f"Message from {self.sender.username} at {self.created_at}"
    
    def save(self, *args, **kwargs):
        if self.id is None:
            last = Message.objects.order_by('-id').first()
            self.id = 1 if last is None else last.id + 1
        super().save(*args, **kwargs)

