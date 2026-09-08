"""Light verification of the CRM's JWTs.

We don't enforce permissions here -- we only confirm the token is valid
(signature + expiry) and pull the caller's user_id from `sub`. The raw token
is then threaded through to the CRM on every crm_client call so the CRM can
enforce its own authorization.
"""
from dataclasses import dataclass

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt

from app.config import JWT_ALGORITHM, JWT_SECRET

# tokenUrl is nominal -- tokens are issued by the CRM, not this service. It's
# only here so FastAPI's docs UI knows to render a bearer-token auth flow.
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/token", auto_error=False)


@dataclass
class AuthedUser:
    user_id: str
    token: str


def get_current_user(token: str | None = Depends(oauth2_scheme)) -> AuthedUser:
    if not token:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing bearer token")

    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except JWTError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or expired token")

    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Token missing subject")

    return AuthedUser(user_id=user_id, token=token)
