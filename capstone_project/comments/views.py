from rest_framework import generics, status
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from rest_framework.decorators import api_view, permission_classes
from .models import Comment, Reply
from .serializers import CommentSerializer, CommentCreateSerializer, ReplyCreateSerializer
from profiles.models import Profile

class CommentListCreateView(generics.ListCreateAPIView):
    permission_classes = [IsAuthenticated]
    serializer_class = CommentSerializer

    def get_queryset(self):
        profile_id = self.kwargs['profile_id']
        return Comment.objects.filter(profile_id=profile_id)

    def get_serializer_class(self):
        if self.request.method == 'POST':
            return CommentCreateSerializer
        return CommentSerializer

    def perform_create(self, serializer):
        profile_id = self.kwargs['profile_id']
        profile = Profile.objects.get(id=profile_id)
        serializer.save(user=self.request.user, profile=profile)

    def list(self, request, *args, **kwargs):
        queryset = self.get_queryset()
        page = request.query_params.get('page', 1)
        per_page = 2  # Giới hạn số bình luận mỗi trang
        start = (int(page) - 1) * per_page
        end = start + per_page
        paginated_queryset = queryset[start:end]
        serializer = self.get_serializer(paginated_queryset, many=True)
        return Response({
            'comments': serializer.data,
            'has_more': len(queryset) > end
        })

    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        profile_id = self.kwargs['profile_id']
        profile = Profile.objects.get(id=profile_id)
        comment = serializer.save(user=request.user, profile=profile)
        # Serialize lại bằng CommentSerializer để trả về dữ liệu đầy đủ
        full_serializer = CommentSerializer(comment)
        return Response(full_serializer.data, status=status.HTTP_201_CREATED)

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def like_comment(request, comment_id):
    try:
        comment = Comment.objects.get(id=comment_id)
        comment.is_liked = not comment.is_liked
        if comment.is_liked:
            comment.likes += 1
        else:
            comment.likes = max(0, comment.likes - 1)
        comment.save()
        serializer = CommentSerializer(comment)
        return Response(serializer.data, status=status.HTTP_200_OK)
    except Comment.DoesNotExist:
        return Response({'error': 'Bình luận không tồn tại'}, status=status.HTTP_404_NOT_FOUND)

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def reply_to_comment(request, comment_id):
    try:
        comment = Comment.objects.get(id=comment_id)
        serializer = ReplyCreateSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save(user=request.user, comment=comment)
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    except Comment.DoesNotExist:
        return Response({'error': 'Bình luận không tồn tại'}, status=status.HTTP_404_NOT_FOUND)