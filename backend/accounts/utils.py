import jwt
from datetime import datetime, timedelta
from django.conf import settings

SECRET_KEY = settings.SECRET_KEY

def generate_jwt(user):
    payload = {
        'user_id': user.id,
        'email': user.email,
        'exp': datetime.utcnow() + timedelta(days=1),
        'iat': datetime.utcnow(),
    }
    token = jwt.encode(payload, SECRET_KEY, algorithm='HS256')
    return token

def decode_jwt(token):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=['HS256'])
        return payload
    except jwt.ExpiredSignatureError:
        print("token has expired")
        return None
    except jwt.InvalidTokenError:
        print("token is invalid")
        return None
