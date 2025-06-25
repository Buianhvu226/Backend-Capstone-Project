from rest_framework import serializers
from .models import Comment, Reply

class ReplySerializer(serializers.ModelSerializer):
    user_name = serializers.CharField(source='user.username', read_only=True)
    created_at = serializers.DateTimeField(format="%d/%m/%Y %H:%M")

    class Meta:
        model = Reply
        fields = ['id', 'user_name', 'content', 'created_at', 'likes', 'is_liked']

class CommentSerializer(serializers.ModelSerializer):
    user_name = serializers.CharField(source='user.username', read_only=True)
    created_at = serializers.DateTimeField(format="%d/%m/%Y %H:%M")
    replies = ReplySerializer(many=True, read_only=True)

    class Meta:
        model = Comment
        fields = ['id', 'user_name', 'content', 'created_at', 'likes', 'is_liked', 'replies']

class CommentCreateSerializer(serializers.ModelSerializer):
    class Meta:
        model = Comment
        fields = ['content']

class ReplyCreateSerializer(serializers.ModelSerializer):
    class Meta:
        model = Reply
        fields = ['content', 'comment']