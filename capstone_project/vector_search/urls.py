from django.urls import path
from .views_api import ProfileSearchAPIView
from django.http import JsonResponse

urlpatterns = [
    path('search-profiles/', ProfileSearchAPIView.as_view(), name='search-profiles'),
]