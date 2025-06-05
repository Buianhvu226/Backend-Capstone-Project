from rest_framework import serializers
from django.contrib.auth import get_user_model
from django.contrib.auth.password_validation import validate_password
from rest_framework_simplejwt.serializers import TokenObtainPairSerializer

User = get_user_model()

class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ['id', 'username', 'email', 'full_name', 'address', 'phone', 'avatar_url','is_active']
        read_only_fields = ['id']

class UserRegistrationSerializer(serializers.ModelSerializer):
    password = serializers.CharField(write_only=True, required=True)

    class Meta:
        model = User
        fields = ['email', 'password', 'full_name', 'address', 'phone']
        read_only_fields = []
        extra_kwargs = {
            'full_name': {'required': False, 'allow_blank': True},
            'address': {'required': False, 'allow_blank': True},
            'phone': {'required': False, 'allow_blank': True},
        }

    def create(self, validated_data):
        email = validated_data['email']
        user = User.objects.create_user(
            username=email,  # Auto-generate username from email
            email=email,
            password=validated_data['password'],
            full_name=validated_data.get('full_name', ''),
            address=validated_data.get('address', ''),
            phone=validated_data.get('phone', '')
        )
        return user

class CustomTokenObtainPairSerializer(TokenObtainPairSerializer):
    def validate(self, attrs):
        try:
            data = super().validate(attrs)
            user = self.user
            user_serializer = UserSerializer(user)
            data['user'] = user_serializer.data
            data['user']['is_admin'] = user.is_staff or user.is_superuser
            return data
        except Exception as e:
            print(f"Authentication error: {str(e)}")
            raise