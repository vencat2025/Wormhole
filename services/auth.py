import secrets
from fastapi import Security, HTTPException, status
from fastapi.security.api_key import APIKeyHeader
from config import settings

api_key_header = APIKeyHeader(name="Authorization", auto_error=False)

async def verify_api_key(api_key_header_val: str = Security(api_key_header)):
    """
    Validates Bearer API Key from request Authorization header.
    Format: 'Authorization: Bearer wh_live_...'
    If settings.ENABLE_AUTH is False, allows unauthenticated dev access.
    """
    if not settings.ENABLE_AUTH:
        return "unauthenticated-dev-user"

    if not settings.VALID_API_KEYS:
        # Auth is on but no key was configured. Refusing is the only safe
        # answer: allowing the request would leave an endpoint the operator
        # believes is protected wide open.
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="ENABLE_AUTH is set but no WORMHOLE_API_KEYS are configured; refusing all requests."
        )

    if not api_key_header_val:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header. Provide 'Authorization: Bearer wh_live_...' key."
        )

    # Strip 'Bearer ' prefix if present
    token = api_key_header_val.replace("Bearer ", "").replace("bearer ", "").strip()

    for valid_key in settings.VALID_API_KEYS:
        if secrets.compare_digest(token, valid_key):
            return token

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid API Key. Unauthorized enterprise request."
    )
