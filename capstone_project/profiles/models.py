from django.db import models
from django.conf import settings
from django.utils.translation import gettext_lazy as _

class Profile(models.Model):
    id = models.IntegerField(primary_key=True)  # Thêm dòng này, tự quản lý id
    
    STATUS_CHOICES = (
        ('active', _('Active')),
        ('found', _('Found')),
        ('closed', _('Closed')),
    )
    
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='profiles')
    title = models.CharField(_("Title"), max_length=255)
    full_name = models.CharField(_("Name of missing person"), max_length=255, null=True, blank=True)
    name_of_father = models.CharField(_("Father's name"), max_length=255, null=True, blank=True)
    name_of_mother = models.CharField(_("Mother's name"), max_length=255, null=True, blank=True)
    born_year = models.CharField(_("Year of birth"), max_length=255, null=True, blank=True)
    losing_year = models.CharField(_("Year of disappearance"), max_length=255, null=True, blank=True)
    siblings = models.TextField(_("Siblings"), null=True, blank=True, help_text=_("List of siblings as plain text"))  # Changed from JSONField to TextField
    description = models.TextField(_("Detailed description"))
    status = models.CharField(_("Status"), max_length=20, choices=STATUS_CHOICES, default='active')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.title
    
    def save(self, *args, **kwargs):
        if self.id is None:
            last = Profile.objects.order_by('-id').first()
            self.id = 1 if last is None else last.id + 1
        super().save(*args, **kwargs)

class ProfileImage(models.Model):
    """Images associated with a profile"""
    id = models.IntegerField(primary_key=True)
    profile = models.ForeignKey(Profile, on_delete=models.CASCADE, related_name='images')
    image = models.CharField(_("Image URL"), max_length=255)
    description = models.CharField(_("Description"), max_length=255, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Image for {self.profile.title}"
    def save(self, *args, **kwargs):
        if self.id is None:
            last = ProfileImage.objects.order_by('-id').first()
            self.id = 1 if last is None else last.id + 1
        super().save(*args, **kwargs)

class ProfileMatchSuggestion(models.Model):
    """Match suggestions between profiles"""
    STATUS_CHOICES = (
        ('pending', _('Pending')),
        ('accepted', _('Accepted')),
        ('rejected', _('Rejected')),
    )
    
    id = models.IntegerField(primary_key=True)  # Thêm dòng này để tự quản lý id
    profile1 = models.ForeignKey(Profile, on_delete=models.CASCADE, related_name='suggestions_as_profile1')
    profile2 = models.ForeignKey(Profile, on_delete=models.CASCADE, related_name='suggestions_as_profile2')
    match_status = models.CharField(_("Match status"), max_length=20, choices=STATUS_CHOICES, default='pending')
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ('profile1', 'profile2')

    def save(self, *args, **kwargs):
        if self.id is None:
            last = ProfileMatchSuggestion.objects.order_by('-id').first()
            self.id = 1 if last is None else last.id + 1
        super().save(*args, **kwargs)

    def __str__(self):
        return f"Match between {self.profile1.title} and {self.profile2.title}"
