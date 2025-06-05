from rest_framework import serializers
from .models import Profile, ProfileMatchSuggestion, ProfileImage

class ProfileSerializer(serializers.ModelSerializer):
    # Thêm các kiểm tra an toàn cho các trường liên quan đến user
    username = serializers.SerializerMethodField()
    user_id = serializers.SerializerMethodField()
    images = serializers.SerializerMethodField()  # Vẫn giữ tên là images để tương thích với code hiện tại
    is_owner = serializers.SerializerMethodField()
    
    class Meta:
        model = Profile
        fields = ['id', 
                  'user_id',        
                  'username',       
                  'title', 'full_name', 'born_year', 'losing_year', 
                  'name_of_father', 'name_of_mother', 'siblings', 'description', 
                  'status', 'created_at', 'is_owner', 'images']
    
    def get_username(self, obj):
        """Lấy username an toàn"""
        if hasattr(obj, 'user') and obj.user:
            return obj.user.username
        return None
    
    def get_user_id(self, obj):
        """Lấy user_id an toàn"""
        if hasattr(obj, 'user') and obj.user:
            return obj.user.id
        return None
    
    def get_images(self, obj):
        """Lấy URL hình ảnh từ bảng ProfileImage"""
        profile_image = ProfileImage.objects.filter(profile=obj).first()
        if profile_image:
            return profile_image.image
        return None
    
    def get_is_owner(self, obj):
        """Xác định người dùng hiện tại có phải là chủ sở hữu của hồ sơ hay không"""
        request = self.context.get('request')
        if request and hasattr(request, 'user'):
            return obj.user == request.user
        return False

class ProfileCreateSerializer(serializers.ModelSerializer):
    images = serializers.ListField(
        child=serializers.ImageField(),
        required=False,
        write_only=True
    )
    siblings = serializers.JSONField(required=False) 

    class Meta:
        model = Profile
        fields = [
            'id',
            'title', 'full_name', 'siblings', 
            'name_of_father', 'name_of_mother', 'born_year', 'losing_year',
            'description', 'status', 'images'
        ]
        read_only_fields = ['id']  # <-- Add this line
    
    def create(self, validated_data):
        images_data = validated_data.pop('images', [])
        # XÓA siblings_data = validated_data.pop('siblings', [])
        
        validated_data['user'] = self.context['request'].user
        profile = Profile.objects.create(**validated_data)
        
        for image in images_data:
            ProfileImage.objects.create(profile=profile, image=image)
        
        # XÓA for sibling_data in siblings_data: ...
        
        return profile

class ProfileMatchSuggestionSerializer(serializers.ModelSerializer):
    profile1_details = ProfileSerializer(source='profile1', read_only=True)
    profile2_details = ProfileSerializer(source='profile2', read_only=True)
    
    class Meta:
        model = ProfileMatchSuggestion
        fields = [
            'id', 'profile1', 'profile2', 'match_score', 'match_status', 
            'created_at', 'profile1_details', 'profile2_details'
        ]
        read_only_fields = ['id', 'created_at', 'profile1_details', 'profile2_details']