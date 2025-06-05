from django.db import models
from django.conf import settings

class Notification(models.Model):
    """Notifications for users"""
    id = models.IntegerField(primary_key=True)
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='notifications')
    type = models.CharField(max_length=50)
    content = models.TextField()
    related_entity_id = models.CharField(max_length=255, null=True, blank=True)
    is_read = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    additional_data = models.JSONField(default=dict, blank=True)
    
    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.type} for {self.user.username}"
    
    def save(self, *args, **kwargs):
        if self.id is None:
            last = Notification.objects.order_by('-id').first()
            self.id = 1 if last is None else last.id + 1
        super().save(*args, **kwargs)
