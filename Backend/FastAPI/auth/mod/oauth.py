from fastapi import Depends, HTTPException, status
from auth.mod.JWTToken import verify_access_token
from fastapi.security import OAuth2PasswordBearer
from passlib.context import CryptContext
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")


pwd_cxt = CryptContext(schemes=['bcrypt'], deprecated='auto')


def bcrypt(password):
    return pwd_cxt.hash(password)

def verify_password(plain_password, hashed_password):
    return pwd_cxt.verify(plain_password, hashed_password)

def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    return verify_access_token(token, credentials_exception)