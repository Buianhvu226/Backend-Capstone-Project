from rest_framework import serializers
from .models import RecentlyMissingReport, MissingPersonMatchResult  # ✅ Đổi tên import


class RecentlyMissingReportSerializer(serializers.ModelSerializer):  # ✅ Đổi tên
    username = serializers.CharField(source='user.username', read_only=True)
    is_owner = serializers.SerializerMethodField()
    formatted_contact_persons = serializers.CharField(read_only=True)
    contact_persons_list = serializers.ListField(read_only=True)

    class Meta:
        model = RecentlyMissingReport  # ✅ Đổi tên model
        fields = [
            'id', 'username', 'profile_type', 'title', 'name', 'age',
            'missing_date', 'location', 'description', 'contact_persons',
            'formatted_contact_persons', 'contact_persons_list', 'status',
            'image_url', 'created_at', 'updated_at', 'is_owner'
        ]
        read_only_fields = ['id', 'created_at', 'updated_at', 'username', 'is_owner',
                           'formatted_contact_persons', 'contact_persons_list']

    def get_is_owner(self, obj):
        request = self.context.get('request')
        if request and request.user.is_authenticated:
            return obj.user == request.user
        return False

    def validate_contact_persons(self, value):
        """Validate contact_persons JSON structure"""
        if not value:
            return {}

        if not isinstance(value, dict):
            raise serializers.ValidationError("contact_persons phải là một object JSON")

        cleaned_data = {}
        for relationship, name in value.items():
            if not isinstance(relationship, str) or not isinstance(name, str):
                raise serializers.ValidationError("Mỗi entry phải có key và value là string")

            clean_relationship = relationship.strip()
            clean_name = name.strip()

            if not clean_relationship:
                raise serializers.ValidationError("Tên mối quan hệ không được để trống")

            if len(clean_relationship) > 50:
                raise serializers.ValidationError("Tên mối quan hệ không được vượt quá 50 ký tự")

            if len(clean_name) > 100:
                raise serializers.ValidationError("Tên người thân không được vượt quá 100 ký tự")

            if clean_name:
                cleaned_data[clean_relationship] = clean_name

        return cleaned_data


class RecentlyMissingReportListSerializer(serializers.ModelSerializer):  # ✅ Đổi tên
    """Serializer tối ưu cho list view"""
    username = serializers.CharField(source='user.username', read_only=True)
    is_owner = serializers.SerializerMethodField()
    formatted_contact_persons = serializers.CharField(read_only=True)

    class Meta:
        model = RecentlyMissingReport  # ✅ Đổi tên model
        fields = [
            'id', 'username', 'profile_type', 'title', 'name', 'age',
            'missing_date', 'location', 'contact_persons', 'formatted_contact_persons',
            'status', 'image_url', 'created_at', 'updated_at', 'is_owner'
        ]
        read_only_fields = ['id', 'created_at', 'updated_at', 'username', 'is_owner', 'formatted_contact_persons']

    def get_is_owner(self, obj):
        request = self.context.get('request')
        if request and request.user.is_authenticated:
            return obj.user == request.user
        return False


class RecentlyMissingReportCreateSerializer(serializers.ModelSerializer):  # ✅ Đổi tên
    class Meta:
        model = RecentlyMissingReport  # ✅ Đổi tên model
        fields = [
            'profile_type', 'title', 'name', 'age', 'missing_date',
            'location', 'description', 'contact_persons', 'status'
        ]

    def validate_contact_persons(self, value):
        """Validate contact_persons JSON structure cho create"""
        if value is None:
            return {}

        if not isinstance(value, dict):
            raise serializers.ValidationError("contact_persons phải là một object JSON")

        cleaned_data = {}
        for relationship, name in value.items():
            if not isinstance(relationship, str) or not isinstance(name, str):
                raise serializers.ValidationError("Mỗi entry phải có key và value là string")

            clean_relationship = relationship.strip()
            clean_name = name.strip()

            if not clean_relationship:
                continue

            if len(clean_relationship) > 50:
                raise serializers.ValidationError(f"Tên mối quan hệ '{clean_relationship}' quá dài (tối đa 50 ký tự)")

            if len(clean_name) > 100:
                raise serializers.ValidationError(f"Tên người thân '{clean_name}' quá dài (tối đa 100 ký tự)")

            if clean_name:
                cleaned_data[clean_relationship] = clean_name

        return cleaned_data

    def create(self, validated_data):
        validated_data['user'] = self.context['request'].user
        return super().create(validated_data)


class RecentlyMissingReportUpdateSerializer(serializers.ModelSerializer):  # ✅ Đổi tên
    """Serializer cho update"""
    class Meta:
        model = RecentlyMissingReport  # ✅ Đổi tên model
        fields = [
            'profile_type', 'title', 'name', 'age', 'missing_date',
            'location', 'description', 'contact_persons', 'status'
        ]

    def validate_contact_persons(self, value):
        """Validate contact_persons JSON structure cho update"""
        if value is None:
            return {}

        if not isinstance(value, dict):
            raise serializers.ValidationError("contact_persons phải là một object JSON")

        cleaned_data = {}
        for relationship, name in value.items():
            if not isinstance(relationship, str) or not isinstance(name, str):
                raise serializers.ValidationError("Mỗi entry phải có key và value là string")

            clean_relationship = relationship.strip()
            clean_name = name.strip()

            if not clean_relationship:
                continue

            if len(clean_relationship) > 50:
                raise serializers.ValidationError(f"Tên mối quan hệ '{clean_relationship}' quá dài (tối đa 50 ký tự)")

            if len(clean_name) > 100:
                raise serializers.ValidationError(f"Tên người thân '{clean_name}' quá dài (tối đa 100 ký tự)")

            if clean_name:
                cleaned_data[clean_relationship] = clean_name

        return cleaned_data


class MissingPersonMatchResultSerializer(serializers.ModelSerializer):
    report1_title = serializers.CharField(source='report1.title', read_only=True)
    report2_title = serializers.CharField(source='report2.title', read_only=True)
    report1_id = serializers.IntegerField(source='report1.id', read_only=True)
    report2_id = serializers.IntegerField(source='report2.id', read_only=True)
    report1_image_url = serializers.URLField(source='report1.image_url', read_only=True)
    report2_image_url = serializers.URLField(source='report2.image_url', read_only=True)
    report1_profile_type = serializers.CharField(source='report1.profile_type', read_only=True)
    report2_profile_type = serializers.CharField(source='report2.profile_type', read_only=True)
    report1_user_id = serializers.IntegerField(source='report1.user.id', read_only=True)
    report2_user_id = serializers.IntegerField(source='report2.user.id', read_only=True)
    report1_username = serializers.CharField(source='report1.user.username', read_only=True)
    report2_username = serializers.CharField(source='report2.user.username', read_only=True)
    report1_created_at = serializers.DateTimeField(source='report1.created_at', read_only=True)
    report2_created_at = serializers.DateTimeField(source='report2.created_at', read_only=True)
    report1_status = serializers.CharField(source='report1.status', read_only=True)
    report2_status = serializers.CharField(source='report2.status', read_only=True)

    class Meta:
        model = MissingPersonMatchResult
        fields = [
            'id', 'report1_id', 'report2_id', 'report1_title', 'report2_title',
            'report1_image_url', 'report2_image_url', 'report1_profile_type', 'report2_profile_type', 'report1_user_id', 'report2_user_id',
            'report1_username', 'report2_username', 'report1_created_at', 'report2_created_at',
            'report1_status', 'report2_status', 'face_match_score', 'llm_confidence', 'llm_analysis', 'status', 'created_at'
        ]