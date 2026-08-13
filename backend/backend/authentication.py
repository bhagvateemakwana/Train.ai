import jwt
from django.conf import settings
from rest_framework import authentication, exceptions
from accounts.models import User

class HeaderJWTAuthentication(authentication.BaseAuthentication):
    def authenticate(self, request):

        auth_header = request.META.get('HTTP_AUTHORIZATION')

        if not auth_header or not auth_header.startswith('Bearer '):
            return None

        token = auth_header.split(' ')[1]

        try:
            payload = jwt.decode(token, settings.SECRET_KEY, algorithms=["HS256"])
            print("Token is valid")
        except jwt.ExpiredSignatureError:
            print("Token has expired")
            raise exceptions.AuthenticationFailed('Token has expired')
        except jwt.InvalidTokenError as e:
            print(f'Invalid token: {str(e)}')
            raise exceptions.AuthenticationFailed(f'Invalid token: {str(e)}')

        user_id = payload.get('user_id')
        if not user_id:
            raise exceptions.AuthenticationFailed('Invalid token payload: missing user_id')

        try:
            user = User.objects.get(id=user_id)
        except User.DoesNotExist:
            raise exceptions.AuthenticationFailed('User not found')

        return (user, None)