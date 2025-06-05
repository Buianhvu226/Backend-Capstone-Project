from django.db import models
from django.contrib.auth.models import AbstractUser
from django.utils.translation import gettext_lazy as _
from rest_framework_simplejwt.tokens import RefreshToken

class User(AbstractUser):
    """Custom user model with additional fields"""
    email = models.EmailField(_("Email address"), unique=True)
    full_name = models.CharField(_("Full name"), max_length=255, blank=True)
    address = models.CharField(_("Address"), max_length=255, blank=True)
    phone = models.CharField(_("Phone number"), max_length=20, blank=True)
    avatar_url = models.ImageField(_("Avatar"), upload_to='avatars/', null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    @property
    def token(self):
        refresh = RefreshToken.for_user(self)
        return {
            'access': str(refresh.access_token),
            'refresh': str(refresh)
        }

    def __str__(self):
        return self.username
