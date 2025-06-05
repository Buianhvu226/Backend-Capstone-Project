from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from rest_framework.routers import DefaultRouter
from rest_framework_simplejwt.views import TokenObtainPairView, TokenRefreshView
from accounts.views import UserRegistrationView, CustomTokenObtainPairView, UserLogoutView, UserProfileView, UserViewSet
from profiles.views import ProfileViewSet, ProfileMatchSuggestionViewSet
from notifications.views import NotificationViewSet

# Create a router for our viewsets
router = DefaultRouter()
# router.register(r'profiles', ProfileViewSet, basename='profile')
router.register(r'match-suggestions', ProfileMatchSuggestionViewSet, basename='match-suggestion')
router.register(r'notifications', NotificationViewSet, basename='notification')
# router.register(r'chats', ChatSessionViewSet, basename='chat')
router.register(r'users', UserViewSet, basename='user')

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include('profiles.urls')), # Or whatever prefix you want for your profiles app
    path('api/', include(router.urls)),
    path('api/', include('vector_search.urls')),  # <-- Make sure this line exists
    path('api/auth/register/', UserRegistrationView.as_view(), name='register'),
    path('api/auth/login/', CustomTokenObtainPairView.as_view(), name='login'),
    path('api/auth/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
    path('api/auth/logout/', UserLogoutView.as_view(), name='logout'),
    path('api/auth/account/', UserProfileView.as_view(), name='user-account'),
    # JWT Token endpoints
    path('api/token/', TokenObtainPairView.as_view(), name='token_obtain_pair'),
    path('api/token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
    path('api/notifications/', include('notifications.urls')),
    path('api/chats/', include('chats.urls')),
    path('api/recently-missing/', include('recently_missing.urls')),
]

# Serve media files in development
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
