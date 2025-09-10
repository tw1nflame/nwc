import os
import jwt
from fastapi import Request, HTTPException, status
from functools import wraps

SUPABASE_JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET", "your_supabase_jwt_secret")

def require_authentication(func):

    @wraps(func)
    async def wrapper(*args, request: Request, **kwargs):
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing or invalid Authorization header")
        token = auth_header.split(" ", 1)[1]
        try:
            # Allow 10 minutes leeway for clock skew between issuer and this service.
            payload = jwt.decode(
                token,
                SUPABASE_JWT_SECRET,
                algorithms=["HS256"],
                audience="authenticated",
                leeway=600,
            )
            request.state.user = payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token expired")
        except jwt.InvalidTokenError as e:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
        return await func(*args, request=request, **kwargs)
    return wrapper
