from datetime import datetime, timedelta
import jwt
import bcrypt
from fastapi import HTTPException, Security, Depends
from fastapi.security import OAuth2PasswordBearer, OAuth2AuthorizationCodeBearer
from typing import Optional, Dict, List
import logging
import pyotp
import secrets
from enum import Enum
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session
from src.database.models import User, UserSession
from src.config import settings

logger = logging.getLogger(__name__)

class UserRole(Enum):
    SUPER_ADMIN = "super_admin"
    ADMIN = "admin"
    POWER_USER = "power_user"
    USER = "user"
    GUEST = "guest"

class Permission(Enum):
    MANAGE_USERS = "manage:users"
    MANAGE_MODELS = "manage:models"
    VIEW_ANALYTICS = "view:analytics"
    EXECUTE_MODELS = "execute:models"
    MANAGE_SYSTEM = "manage:system"

class AuthService:
    def __init__(self, db: Session):
        self.db = db
        self.oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
        self.oauth2_code_scheme = OAuth2AuthorizationCodeBearer(
            authorizationUrl="authorize",
            tokenUrl="token"
        )
        
        # Initialize TOTP for 2FA
        self.totp = pyotp.TOTP(settings.TOTP_SECRET)
    
    async def create_user(
        self,
        email: EmailStr,
        password: str,
        role: UserRole,
        permissions: List[Permission]
    ) -> User:
        try:
            # Check if user exists
            if await self.get_user_by_email(email):
                raise HTTPException(
                    status_code=400,
                    detail="User already exists"
                )
            
            # Hash password
            hashed_password = self._hash_password(password)
            
            # Create user
            user = User(
                email=email,
                hashed_password=hashed_password,
                role=role,
                permissions=permissions,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            
            self.db.add(user)
            await self.db.commit()
            await self.db.refresh(user)
            
            logger.info(f"Created new user: {email}")
            return user
            
        except Exception as e:
            logger.error(f"User creation failed: {e}")
            raise HTTPException(
                status_code=500,
                detail="User creation failed"
            )
    
    async def authenticate(
        self,
        email: str,
        password: str,
        two_factor_code: Optional[str] = None
    ) -> Dict:
        try:
            # Get user
            user = await self.get_user_by_email(email)
            if not user:
                raise HTTPException(
                    status_code=401,
                    detail="Invalid credentials"
                )
            
            # Verify password
            if not self._verify_password(password, user.hashed_password):
                await self._handle_failed_login(user)
                raise HTTPException(
                    status_code=401,
                    detail="Invalid credentials"
                )
            
            # Check if 2FA is required
            if user.two_factor_enabled:
                if not two_factor_code:
                    raise HTTPException(
                        status_code=400,
                        detail="2FA code required"
                    )
                if not self._verify_2fa(two_factor_code):
                    await self._handle_failed_login(user)
                    raise HTTPException(
                        status_code=401,
                        detail="Invalid 2FA code"
                    )
            
            # Create session
            session = await self._create_session(user)
            
            # Generate tokens
            access_token = self._create_access_token(user, session.id)
            refresh_token = self._create_refresh_token(user, session.id)
            
            return {
                "access_token": access_token,
                "refresh_token": refresh_token,
                "token_type": "bearer",
                "user": {
                    "id": user.id,
                    "email": user.email,
                    "role": user.role,
                    "permissions": user.permissions
                }
            }
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            raise HTTPException(
                status_code=500,
                detail="Authentication failed"
            )
    
    async def _create_session(self, user: User) -> UserSession:
        """Create a new user session"""
        session = UserSession(
            user_id=user.id,
            session_id=secrets.token_urlsafe(32),
            created_at=datetime.utcnow(),
            expires_at=datetime.utcnow() + timedelta(days=7),
            user_agent=request.headers.get("User-Agent"),
            ip_address=request.client.host
        )
        
        self.db.add(session)
        await self.db.commit()
        await self.db.refresh(session)
        
        return session
    
    def _create_access_token(
        self,
        user: User,
        session_id: str
    ) -> str:
        """Create JWT access token"""
        expires = datetime.utcnow() + timedelta(
            minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES
        )
        
        to_encode = {
            "sub": str(user.id),
            "exp": expires,
            "type": "access",
            "session": session_id,
            "role": user.role.value,
            "permissions": [p.value for p in user.permissions]
        }
        
        return jwt.encode(
            to_encode,
            settings.JWT_SECRET_KEY,
            algorithm=settings.JWT_ALGORITHM
        )
    
    async def verify_token(self, token: str) -> Dict:
        """Verify JWT token and return payload"""
        try:
            payload = jwt.decode(
                token,
                settings.JWT_SECRET_KEY,
                algorithms=[settings.JWT_ALGORITHM]
            )
            
            # Verify session
            session = await self._get_session(payload["session"])
            if not session or session.is_expired():
                raise HTTPException(
                    status_code=401,
                    detail="Session expired"
                )
            
            return payload
            
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=401,
                detail="Token expired"
            )
        except jwt.JWTError:
            raise HTTPException(
                status_code=401,
                detail="Invalid token"
            )
    
    def _hash_password(self, password: str) -> str:
        """Hash password using bcrypt"""
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password.encode(), salt).decode()
    
    def _verify_password(
        self,
        plain_password: str,
        hashed_password: str
    ) -> bool:
        """Verify password against hash"""
        return bcrypt.checkpw(
            plain_password.encode(),
            hashed_password.encode()
        )
    
    async def _handle_failed_login(self, user: User):
        """Handle failed login attempt"""
        user.failed_login_attempts += 1
        user.last_failed_login = datetime.utcnow()
        
        if user.failed_login_attempts >= settings.MAX_LOGIN_ATTEMPTS:
            user.is_locked = True
            user.locked_until = datetime.utcnow() + timedelta(
                minutes=settings.ACCOUNT_LOCKOUT_MINUTES
            )
        
        await self.db.commit()