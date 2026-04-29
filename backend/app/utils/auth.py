import os
from functools import lru_cache, wraps

import jwt
from jwt import PyJWKClient
from fastapi import HTTPException, Request, status


@lru_cache(maxsize=1)
def _get_keycloak_settings() -> dict:
    url = os.getenv("KEYCLOAK_URL", "").rstrip("/")
    realm = os.getenv("KEYCLOAK_REALM", "")
    client_id = os.getenv("KEYCLOAK_CLIENT_ID", "")

    issuer = os.getenv("KEYCLOAK_ISSUER", "")
    if not issuer:
        if not url or not realm:
            raise RuntimeError(
                "Missing Keycloak env vars: set KEYCLOAK_URL and KEYCLOAK_REALM (or KEYCLOAK_ISSUER)."
            )
        issuer = f"{url}/realms/{realm}"

    jwks_url = os.getenv("KEYCLOAK_JWKS_URL", "")
    if not jwks_url:
        jwks_url = f"{issuer}/protocol/openid-connect/certs"

    leeway_seconds = int(os.getenv("KEYCLOAK_LEEWAY_SECONDS", "60"))
    algorithms = [a.strip() for a in os.getenv("KEYCLOAK_ALGORITHMS", "RS256").split(",") if a.strip()]

    if not client_id:
        raise RuntimeError("Missing Keycloak env var: set KEYCLOAK_CLIENT_ID")

    return {
        "issuer": issuer,
        "jwks_url": jwks_url,
        "client_id": client_id,
        "leeway_seconds": leeway_seconds,
        "algorithms": algorithms,
    }


@lru_cache(maxsize=1)
def _get_jwks_client() -> PyJWKClient:
    settings = _get_keycloak_settings()
    return PyJWKClient(settings["jwks_url"])


def _decode_keycloak_token(token: str) -> dict:
    settings = _get_keycloak_settings()
    jwks_client = _get_jwks_client()

    signing_key = jwks_client.get_signing_key_from_jwt(token).key

    # Keycloak access tokens often have aud=['account', ...] and the clientId is present as 'azp'.
    # We verify issuer + signature + exp and additionally bind token to the expected client via azp.
    payload = jwt.decode(
        token,
        signing_key,
        algorithms=settings["algorithms"],
        issuer=settings["issuer"],
        options={"verify_aud": False},
        leeway=settings["leeway_seconds"],
    )

    azp = payload.get("azp")
    if azp != settings["client_id"]:
        raise jwt.InvalidTokenError("Invalid azp")

    return payload


def require_authentication(func):
    @wraps(func)
    async def wrapper(*args, request: Request, **kwargs):
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing or invalid Authorization header",
            )

        token = auth_header.split(" ", 1)[1].strip()
        if not token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing or invalid Authorization header",
            )

        try:
            payload = _decode_keycloak_token(token)
            request.state.user = payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token expired")
        except Exception:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

        return await func(*args, request=request, **kwargs)

    return wrapper
