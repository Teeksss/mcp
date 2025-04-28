from fastapi import APIRouter, Request
from starlette.responses import RedirectResponse
from src.services.auth.oauth import oauth

router = APIRouter(prefix="/api/v1/oauth", tags=["OAuth2"])

@router.get("/login/google")
async def login_via_google(request: Request):
    redirect_uri = request.url_for('auth_google_callback')
    return await oauth.google.authorize_redirect(request, redirect_uri)

@router.get("/login/google/callback")
async def auth_google_callback(request: Request):
    token = await oauth.google.authorize_access_token(request)
    user_info = await oauth.google.parse_id_token(request, token)
    # user_info['email'], user_info['sub'] gibi alanları kullan
    # Kullanıcıya JWT ver ve frontend'e redirect et
    jwt_token = create_jwt_for_user(user_info)
    response = RedirectResponse(url=f"/?token={jwt_token}")
    return response