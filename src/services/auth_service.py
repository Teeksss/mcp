from datetime import datetime, timedelta
from typing import Optional, Dict, List
import jwt
import bcrypt
from dataclasses import dataclass
import logging
from enum import Enum
import secrets
import base64
from fastapi import HTTPException, Security
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel

logger = logging.getLogger(__name__)

class UserRole(Enum):
    ADMIN = "admin"
    USER = "user"
    GUEST = "guest"

class Permission(Enum):
    READ_MODELS = "read:models"
    WRITE_MODELS = "write:models"
    MANAGE_USERS = "manage:users"
    VIEW_STATS = "view:stats"
    MANAGE_SYSTEM = "manage:system"

@dataclass
class AuthConfig:
    secret_key: str
    token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7
    password_min_length: int = 8
    max_failed_attempts: int = 5
    lockout_duration_minutes: int = 15
    enable_2fa: bool = True

class User(BaseModel):
    id: str
    username: str
    email: str
    hashed_password: str
    role: UserRole
    permissions: List[Permission]
    failed_attempts: int = 0
    last_failed_attempt: Optional[datetime] = None
    is_locked: bool = False
    is_active: bool = True
    two_factor_enabled: bool = False
    two_factor_secret: Optional[str] = None

class AuthService:
    def __init__(self, config: AuthConfig):
        self.config = config
        self.oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
        self._user_cache: Dict[str, User] = {}
    
    def _hash_password(self, password: str) -> str:
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password.encode(), salt).decode()
    
    def _verify_password(self, plain_password: str, hashed_password: str) -> bool:
        return bcrypt.checkpw(
            plain_password.encode(),
            hashed_password.encode()
        )
    
    def create_tokens(self, user: User) -> Dict[str, str]:
        # Create access token
        access_token_expires = datetime.utcnow() + timedelta(
            minutes=self.config.token_expire_minutes
        )
        access_token = self._create_jwt_token(
            data={
                "sub": user.username,
                "exp": access_token_expires,
                "type": "access",
                "role": user.role.value,
                "permissions": [p.value for p in user.permissions]
            }
        )
        
        # Create refresh token
        refresh_token_expires = datetime.utcnow() + timedelta(
            days=self.config.refresh_token_expire_days
        )
        refresh_token = self._create_jwt_token(
            data={
                "sub": user.username,
                "exp": refresh_token_expires,
                "type": "refresh"
            }
        )
        
        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer"
        }
    
    def _create_jwt_token(self, data: Dict) -> str:
        return jwt.encode(
            data,
            self.config.secret_key,
            algorithm="HS256"
        )
    
    async def authenticate_user(
        self,
        username: str,
        password: str,
        two_factor_code: Optional[str] = None
    ) -> Optional[User]:
        user = await self._get_user(username)
        
        if not user or not user.is_active:
            return None
        
        # Check if user is locked
        if user.is_locked:
            lockout_end = user.last_failed_attempt + timedelta(
                minutes=self.config.lockout_duration_minutes
            )
            if datetime.utcnow() < lockout_end:
                raise HTTPException(
                    status_code=403,
                    detail="Account is locked. Please try again later."
                )
            # Reset lockout if duration has passed
            user.is_locked = False
            user.failed_attempts = 0
        
        # Verify password
        if not self._verify_password(password, user.hashed_password):
            await self._handle_failed_login(user)
            return None
        
        # Verify 2FA if enabled
        if user.two_factor_enabled:
            if not two_factor_code:
                raise HTTPException(
                    status_code=400,
                    detail="2FA code required"
                )
            if not self._verify_2fa(user, two_factor_code):
                await self._handle_failed_login(user)
                return None
        
        # Reset failed attempts on successful login
        user.failed_attempts = 0
        user.last_failed_attempt = None
        return user
    
    async def _handle_failed_login(self, user: User):
        user.failed_attempts += 1
        user.last_failed_attempt = datetime.utcnow()
        
        if user.failed_attempts >= self.config.max_failed_attempts:
            user.is_locked = True
            logger.warning(f"Account locked for user: {user.username}")
        
        await self._update_user(user)
    
    def _verify_2fa(self, user: User, code: str) -> bool:
        if not user.two_factor_secret:
            return False
        
        try:
            totp = pyotp.TOTP(user.two_factor_secret)
            return totp.verify(code)
        except Exception as e:
            logger.error(f"2FA verification error: {e}")
            return False
    
    def setup_2fa(self, user: User) -> str:
        """Setup 2FA for a user and return the secret"""
        secret = pyotp.random_base32()
        user.two_factor_secret = secret
        user.two_factor_enabled = True
        return secret
    
    async def get_current_user(
        self,
        token: str = Security(OAuth2PasswordBearer(tokenUrl="token"))
    ) -> User:
        try:
            payload = jwt.decode(
                token,
                self.config.secret_key,
                algorithms=["HS256"]
            )
            username = payload.get("sub")
            if not username:
                raise HTTPException(
                    status_code=401,
                    detail="Invalid authentication credentials"
                )
            
            user = await self._get_user(username)
            if not user or not user.is_active:
                raise HTTPException(
                    status_code=401,
                    detail="User not found or inactive"
                )
            
            return user
            
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=401,
                detail="Token has expired"
            )
        except jwt.JWTError:
            raise HTTPException(
                status_code=401,
                detail="Could not validate credentials"
            )
    
    def verify_permission(
        self,
        user: User,
        required_permission: Permission
    ) -> bool:
        return (
            user.role == UserRole.ADMIN or
            required_permission in user.permissions
        )