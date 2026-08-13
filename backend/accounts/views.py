from rest_framework.decorators import api_view, authentication_classes, permission_classes
from rest_framework.response import Response
from rest_framework import status
from rest_framework.views import APIView
from rest_framework.permissions import IsAuthenticated, AllowAny
from django.contrib.auth.hashers import make_password, check_password
from django.views.decorators.csrf import csrf_exempt
from django.utils.timezone import now
from .models import User
from .serializers import RegisterSerializer, UserSerializer
from .utils import generate_jwt, decode_jwt
import re
import os
import requests
import logging

logger = logging.getLogger(__name__)
EMAIL_REGEX = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")

@csrf_exempt
@api_view(['POST'])
@permission_classes([AllowAny])
def register_view(request):
    try:
        data = request.data
        email = data.get('email', '').lower().strip()
        full_name = data.get('full_name', '').strip()
        password = data.get('password', '')
        
        print(f"Registering user with email: {email}, full_name: {full_name}, password: {password}s")

        if not email or not password or not full_name:
            return Response({'message': 'All fields are required.', 'success': False}, status=400)

        if not EMAIL_REGEX.match(email):
            return Response({'message': 'Please provide a valid email address.', 'success': False}, status=400)

        if len(password) < 6:
            return Response({'message': 'Password must be at least 6 characters long.', 'success': False}, status=400)

        if User.objects.filter(email=email).exists():
            return Response({'message': 'User already exists with this email.', 'success': False}, status=400)

        serializer = RegisterSerializer(data={
            'email': email,
            'full_name': full_name,
            'password': password
        })

        if serializer.is_valid():
            user = serializer.save()
            token = generate_jwt(user)

            response = Response({
                'message': 'Account created successfully.',
                'success': True,
                'user': UserSerializer(user).data,
                'token': token
            }, status=201)

            response.set_cookie(
                key='Authorization',
                value='Bearer ' + token,
                httponly=True,
                secure=os.environ.get('DJANGO_ENV') == 'production'
            )
            return response

        return Response({'message': 'Invalid data.', 'success': False}, status=400)
    except Exception as e:
        print(f"Error occured during registration: {str(e)}")
        return Response({
            'error': str(e),
            'success': False,
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@csrf_exempt
@api_view(['POST'])
@permission_classes([AllowAny])
def login_view(request):
    try:
        data = request.data
        email = data.get('email', '').lower().strip()
        password = data.get('password', '')

        print(f"Logging in user with email: {email}, password: {password}")

        if not email or not password:
            return Response({'message': 'Email and password are required.', 'success': False}, status=400)

        if not EMAIL_REGEX.match(email):
            return Response({'message': 'Please provide a valid email address.', 'success': False}, status=400)

        try:
            user = User.objects.get(email=email)
        except User.DoesNotExist:
            return Response({'message': 'Invalid email or password.', 'success': False}, status=400)

        if not user.check_password(password):
            return Response({'message': 'Invalid email or password.', 'success': False}, status=400)

        token = generate_jwt(user)

        response = Response({
            'message': 'Logged in successfully.',
            'success': True,
            'user': UserSerializer(user).data,
            'token': token
        })

        response.set_cookie(
            key='Authorization',
            value='Bearer ' + token,
            httponly=True,
            secure=os.environ.get('DJANGO_ENV') == 'production'
        )

        return response
    except Exception as e:
        print(f"Error occured during login: {str(e)}")
        return Response({
            'error': str(e),
            'success': False,
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def logout_view(request):
    response = Response({
        'message': 'Logged out successfully.',
        'success': True
    }, status=status.HTTP_200_OK)
    response.delete_cookie('Authorization')
    return response

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def update_premium_status(request):
    user = request.user

    if not user or not user.id:
        return Response({
            "message": "Authentication required.",
            "success": False,
            "error": "UNAUTHORIZED"
        }, status=status.HTTP_401_UNAUTHORIZED)

    user_id = str(user.id)

    try:
        user = User.objects.get(id=user_id)
    except User.DoesNotExist:
        print(f"Premium update attempted for non-existent user: {user_id}")
        return Response({
            "message": "User not found.",
            "success": False,
            "error": "USER_NOT_FOUND"
        }, status=status.HTTP_404_NOT_FOUND)

    if user.premium_user:
        return Response({
            "message": "User is already a premium member.",
            "success": True,
            "data": {
                "premium_status": True,
                "message": "No changes made"
            }
        }, status=status.HTTP_200_OK)

    user.premium_user = True
    user.save()

    print(f"Premium status updated successfully for user: {user_id}")

    return Response({
        "message": "Premium status updated successfully.",
        "success": True,
        "data": {
            "premium_status": user.premium_user,
            "updated_at": now().isoformat()
        }
    }, status=status.HTTP_200_OK)

class UserProfileView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        user = request.user
        serialized = UserSerializer(user)
        return Response(serialized.data, status=status.HTTP_200_OK)

def get_user_from_token(request):
    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith('Bearer '):
        return None
    token = auth_header.split(' ')[1]
    payload = decode_jwt(token)
    if not payload:
        return None
    try:
        return User.objects.get(id=payload['user_id'])
    except User.DoesNotExist:
        return None

@api_view(['PUT'])
@permission_classes([IsAuthenticated])
def update_liked_model(request, model_id):
    user = request.user
    
    if not user.liked_models:
        user.liked_models = []
    
    already_liked = model_id in user.liked_models
    state = "dislike" if already_liked else "like"
    
    if already_liked:
        user.liked_models.remove(model_id)
    else:
        user.liked_models.append(model_id)
    
    try:
        token = request.headers.get('Authorization', '').replace('Bearer ', '')
        ml_api_url = f"http://127.0.0.1:8000/api/v1/trained-model/update-model/{model_id}/"
        
        response = requests.put(
            ml_api_url,
            json={"state": state},
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {token}"
            },
            timeout=10
        )
        
        if response.status_code == 404:
            return Response({
                "message": "Model not found on the ML service.",
                "success": False,
                "error": "MODEL_NOT_FOUND"
            }, status=status.HTTP_404_NOT_FOUND)
        
        if response.status_code >= 400:
            return Response({
                "message": "Failed to update model status on ML service.",
                "success": False,
                "error": "EXTERNAL_API_ERROR"
            }, status=status.HTTP_400_BAD_REQUEST)
            
    except requests.exceptions.RequestException:
        return Response({
            "message": "Unable to connect to ML service.",
            "success": False,
            "error": "SERVICE_CONNECTION_ERROR"
        }, status=status.HTTP_503_SERVICE_UNAVAILABLE)
    
    try:
        user.save()
        
        return Response({
            "message": f"Model {'added to' if state == 'like' else 'removed from'} liked models successfully.",
            "success": True,
            "data": {
                "action": state,
                "model_id": model_id,
                "total_liked_models": len(user.liked_models)
            }
        }, status=status.HTTP_200_OK)
        
    except Exception as e:
        logger.error(f"Error saving user liked models: {str(e)}")
        return Response({
            "message": "Failed to save liked model status.",
            "success": False,
            "error": "DATABASE_SAVE_ERROR"
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)