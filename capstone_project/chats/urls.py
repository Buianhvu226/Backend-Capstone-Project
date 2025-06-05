from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import ChatSessionViewSet, MessageViewSet, api_endpoints

router = DefaultRouter()
router.register(r'chatsessions', ChatSessionViewSet, basename='chatsession')
router.register(r'messages', MessageViewSet, basename='message')

urlpatterns = [
    path('', include(router.urls)),
    path('endpoints/', api_endpoints, name='chat-api-endpoints'),
]