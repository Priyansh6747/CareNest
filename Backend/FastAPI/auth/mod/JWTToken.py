from datetime import datetime , timedelta
from jose import JWTError , jwt
import os
from dotenv import load_dotenv

from auth.mod.models import TokenData

load_dotenv()
secret_key = os.getenv('SECRET_KEY')
algorithm = 'HS256'
access_token_expire_minutes = os.getenv('ACCESS_TOKEN_EXPIRE_MINUTES')

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=int(access_token_expire_minutes))
    to_encode.update({'exp': expire})
    encoded_jwt = jwt.encode(to_encode, secret_key, algorithm=algorithm)
    return encoded_jwt

def verify_access_token(token: str, credentials_exception):
    try:
        payload = jwt.decode(token, secret_key, algorithms=[algorithm])
        username: str = payload.get('sub')
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
        return token_data  # Add this return statement
    except JWTError:
        raise credentials_exception

