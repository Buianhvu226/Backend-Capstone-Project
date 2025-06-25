from django.urls import path
from .views import CommentListCreateView, like_comment, reply_to_comment

urlpatterns = [
    path('profiles/<int:profile_id>/comments/', CommentListCreateView.as_view(), name='comment-list-create'),
    path('comments/<int:comment_id>/like/', like_comment, name='like-comment'),
    path('comments/<int:comment_id>/reply/', reply_to_comment, name='reply-comment'),
]