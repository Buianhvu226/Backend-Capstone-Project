# recently_missing/models.py
from django.db import models
from accounts.models import User

class RecentlyMissingReport(models.Model):  # ✅ Đổi từ RecentlyMissingProfile
    PROFILE_TYPE_CHOICES = [
        ('seeker', 'Người đi tìm'),
        ('finder', 'Người cung cấp thông tin'),
    ]
    STATUS_CHOICES = [
        ('active', 'Đang tìm kiếm'),
        ('closed', 'Đã đóng'),
        ('found', 'Đã tìm thấy'),
    ]
    
    id = models.IntegerField(primary_key=True)
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='missing_reports')  # ✅ Đổi related_name
    profile_type = models.CharField(max_length=10, choices=PROFILE_TYPE_CHOICES)
    title = models.CharField(max_length=255)
    name = models.CharField(max_length=255, blank=True)
    age = models.IntegerField(null=True, blank=True)
    missing_date = models.DateField(null=True, blank=True)
    location = models.CharField(max_length=255, blank=True)
    description = models.TextField(blank=True)
    contact_persons = models.JSONField(default=dict, blank=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='active')
    image_url = models.URLField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"Missing Report: {self.title} ({self.profile_type})"
    
    def save(self, *args, **kwargs):
        if self.id is None:
            last = RecentlyMissingReport.objects.order_by('-id').first()
            self.id = 1 if last is None else last.id + 1
        super().save(*args, **kwargs)

    @property
    def formatted_contact_persons(self):
        """Trả về chuỗi formatted của các người thân"""
        if not self.contact_persons:
            return ""
        
        contacts = []
        for relationship, name in self.contact_persons.items():
            if name and name.strip() and relationship and relationship.strip():
                contacts.append(f"{relationship}: {name}")
        
        return "; ".join(contacts)

    @property
    def contact_persons_list(self):
        """Trả về danh sách người thân dạng list"""
        if not self.contact_persons:
            return []
        
        contacts = []
        for relationship, name in self.contact_persons.items():
            if name and name.strip() and relationship and relationship.strip():
                contacts.append({
                    'relationship': relationship.strip(),
                    'name': name.strip()
                })
        
        return contacts

    class Meta:
        ordering = ['-created_at']
        verbose_name = "Recently Missing Report"
        verbose_name_plural = "Recently Missing Reports"

class MissingPersonMatchResult(models.Model):
    STATUS_CHOICES = [
        ('pending', 'Đang chờ'),
        ('accepted', 'Đã chấp nhận'),
        ('rejected', 'Đã từ chối'),
    ]
    
    id = models.IntegerField(primary_key=True)
    report1 = models.ForeignKey(RecentlyMissingReport, on_delete=models.CASCADE, related_name='matches_as_report1')  # ✅ Đổi tên
    report2 = models.ForeignKey(RecentlyMissingReport, on_delete=models.CASCADE, related_name='matches_as_report2')  # ✅ Đổi tên
    face_match_score = models.FloatField()
    llm_analysis = models.JSONField(null=True, blank=True)
    llm_confidence = models.CharField(max_length=20, choices=[('uncertain', 'Chưa chắc chắn'), ('moderate', 'Có thể trung bình'), ('fairly_confident', 'Trung bình chắc chắn'), ('confident', 'Khá chắc chắn'), ('highly_confident', 'Rất chắc chắn'), ('match', 'Khớp hoàn toàn'), ('partial', 'Khớp một phần'), ('no_match', 'Không khớp')], blank=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    created_at = models.DateTimeField(auto_now_add=True)

    def save(self, *args, **kwargs):
        if self.id is None:
            last = MissingPersonMatchResult.objects.order_by('-id').first()
            self.id = 1 if last is None else last.id + 1
        super().save(*args, **kwargs)

    class Meta:
        unique_together = ['report1', 'report2']
        verbose_name = "Missing Person Match Result"
        verbose_name_plural = "Missing Person Match Results"