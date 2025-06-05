from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import ProfileViewSet, ProfileMatchSuggestionViewSet # Make sure to import ProfileViewSet

router = DefaultRouter()
router.register(r'profiles', ProfileViewSet, basename='profile')
# If you have ProfileMatchSuggestionViewSet, register it as well
# router.register(r'profile-match-suggestions', ProfileMatchSuggestionViewSet, basename='profilematchsuggestion')


urlpatterns = [
    path('', include(router.urls)),
]