from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
import asyncio
import logging
import jwt
from cryptography.fernet import Fernet
import bcrypt
from prometheus_client import Counter, Gauge, Histogram
import aioredis
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class SecurityConfig:
    jwt_secret: str
    jwt_algorithm: str = "HS256"
    jwt_expiry: int = 3600  # 1 hour
    max_login_attempts: int = 5
    lockout_duration: int = 300  # 5 minutes
    password_min_length: int = 12
    mfa_enabled: bool = True
    ip_rate_limit: int = 100
    session_timeout: int = 1800  # 30 minutes

class AdvancedSecurityManager:
    def __init__(
        self,
        config: SecurityConfig,
        redis_client: aioredis.Redis
    ):
        self.config = config
        self.redis = redis_client
        self.metrics = self._setup_metrics()
        
        # Encryption
        self.fernet = Fernet(Fernet.generate_key())
        
        # Current context
        self.timestamp = datetime.utcnow()
        self.current_user = "Teeksss"
        
        # Start security monitoring
        asyncio.create_task(self._monitor_security_events())
    
    def _setup_metrics(self) -> Dict:
        """Initialize security metrics"""
        return {
            "auth_attempts": Counter(
                "security_auth_attempts_total",
                "Total authentication attempts",
                ["status", "method"]
            ),
            "failed_logins": Counter(
                "security_failed_logins_total",
                "Total failed login attempts",
                ["user", "reason"]
            ),
            "blocked_ips": Gauge(
                "security_blocked_ips_total",
                "Total blocked IP addresses"
            ),
            "mfa_verifications": Counter(
                "security_mfa_verifications_total",
                "Total MFA verifications",
                ["status"]
            ),
            "security_events": Counter(
                "security_events_total",
                "Total security events",
                ["type", "severity"]
            )
        }
    
    async def authenticate_user(
        self,
        username: str,
        password: str,
        ip_address: str,
        mfa_code: Optional[str] = None
    ) -> Dict:
        """Authenticate user with enhanced security"""
        try:
            # Check IP rate limiting
            if await self._is_ip_rate_limited(ip_address):
                self.metrics["auth_attempts"].labels(
                    status="blocked",
                    method="ip_rate_limit"
                ).inc()
                raise SecurityException("Rate limit exceeded")
            
            # Check account lockout
            if await self._is_account_locked(username):
                self.metrics["auth_attempts"].labels(
                    status="blocked",
                    method="account_lockout"
                ).inc()
                raise SecurityException("Account locked")
            
            # Verify password
            if not await self._verify_password(username, password):
                await self._handle_failed_login(username, ip_address)
                raise SecurityException("Invalid credentials")
            
            # Check MFA if enabled
            if self.config.mfa_enabled:
                if not mfa_code:
                    raise SecurityException("MFA code required")
                if not await self._verify_mfa(username, mfa_code):
                    raise SecurityException("Invalid MFA code")
            
            # Generate tokens
            access_token = await self._generate_access_token(username)
            refresh_token = await self._generate_refresh_token(username)
            
            # Create session
            session_id = await self._create_session(
                username,
                ip_address
            )
            
            # Update metrics
            self.metrics["auth_attempts"].labels(
                status="success",
                method="password"
            ).inc()
            
            return {
                "access_token": access_token,
                "refresh_token": refresh_token,
                "session_id": session_id,
                "expires_in": self.config.jwt_expiry
            }
            
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            raise
    
    async def validate_token(
        self,
        token: str,
        token_type: str = "access"
    ) -> Dict:
        """Validate JWT token"""
        try:
            # Decode token
            payload = jwt.decode(
                token,
                self.config.jwt_secret,
                algorithms=[self.config.jwt_algorithm]
            )
            
            # Check token type
            if payload.get("type") != token_type:
                raise SecurityException("Invalid token type")
            
            # Check expiration
            if datetime.fromtimestamp(payload["exp"]) < self.timestamp:
                raise SecurityException("Token expired")
            
            # Verify session
            if not await self._verify_session(
                payload["sub"],
                payload.get("session_id")
            ):
                raise SecurityException("Invalid session")
            
            return payload
            
        except jwt.InvalidTokenError as e:
            logger.error(f"Token validation failed: {e}")
            raise SecurityException("Invalid token")
    
    async def refresh_access_token(
        self,
        refresh_token: str
    ) -> Dict:
        """Refresh access token"""
        try:
            # Validate refresh token
            payload = await self.validate_token(
                refresh_token,
                "refresh"
            )
            
            # Generate new access token
            access_token = await self._generate_access_token(
                payload["sub"]
            )
            
            return {
                "access_token": access_token,
                "expires_in": self.config.jwt_expiry
            }
            
        except Exception as e:
            logger.error(f"Token refresh failed: {e}")
            raise
    
    async def encrypt_sensitive_data(
        self,
        data: Union[str, Dict],
        context: Optional[Dict] = None
    ) -> str:
        """Encrypt sensitive data"""
        try:
            # Convert dict to string if needed
            if isinstance(data, dict):
                data = json.dumps(data)
            
            # Add context if provided
            if context:
                data = f"{json.dumps(context)}|{data}"
            
            # Encrypt
            encrypted = self.fernet.encrypt(data.encode())
            return encrypted.decode()
            
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            raise
    
    async def decrypt_sensitive_data(
        self,
        encrypted_data: str,
        context: Optional[Dict] = None
    ) -> Union[str, Dict]:
        """Decrypt sensitive data"""
        try:
            # Decrypt
            decrypted = self.fernet.decrypt(
                encrypted_data.encode()
            ).decode()
            
            # Split context if exists
            if "|" in decrypted:
                stored_context, data = decrypted.split("|", 1)
                stored_context = json.loads(stored_context)
                
                # Verify context
                if context and stored_context != context:
                    raise SecurityException("Invalid decryption context")
            else:
                data = decrypted
            
            # Try parse as JSON
            try:
                return json.loads(data)
            except json.JSONDecodeError:
                return data
            
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise
    
    async def authorize_request(
        self,
        user: Dict,
        resource: str,
        action: str
    ) -> bool:
        """Authorize user request"""
        try:
            # Get user roles and permissions
            roles = await self._get_user_roles(user["sub"])
            permissions = await self._get_role_permissions(roles)
            
            # Check permission
            required_permission = f"{resource}:{action}"
            
            if required_permission in permissions:
                # Log authorization
                await self._log_security_event(
                    "authorization",
                    "info",
                    {
                        "user": user["sub"],
                        "resource": resource,
                        "action": action,
                        "allowed": True
                    }
                )
                return True
            
            # Log denied access
            await self._log_security_event(
                "authorization",
                "warning",
                {
                    "user": user["sub"],
                    "resource": resource,
                    "action": action,
                    "allowed": False
                }
            )
            return False
            
        except Exception as e:
            logger.error(f"Authorization failed: {e}")
            raise
    
    async def validate_input(
        self,
        data: Dict,
        schema: Dict,
        sanitize: bool = True
    ) -> Dict:
        """Validate and sanitize input"""
        try:
            validated = {}
            
            for field, value in data.items():
                if field not in schema:
                    continue
                
                field_schema = schema[field]
                
                # Type validation
                if not isinstance(value, field_schema["type"]):
                    raise ValidationError(
                        f"Invalid type for {field}"
                    )
                
                # Pattern validation
                if "pattern" in field_schema:
                    if not re.match(field_schema["pattern"], str(value)):
                        raise ValidationError(
                            f"Invalid format for {field}"
                        )
                
                # Sanitization
                if sanitize:
                    value = self._sanitize_input(
                        value,
                        field_schema.get("sanitize", [])
                    )
                
                validated[field] = value
            
            return validated
            
        except Exception as e:
            logger.error(f"Input validation failed: {e}")
            raise
    
    async def _verify_password(
        self,
        username: str,
        password: str
    ) -> bool:
        """Verify password hash"""
        # Implementation depends on password storage
        pass
    
    async def _verify_mfa(
        self,
        username: str,
        mfa_code: str
    ) -> bool:
        """Verify MFA code"""
        # Implementation depends on MFA method
        pass
    
    async def _create_session(
        self,
        username: str,
        ip_address: str
    ) -> str:
        """Create new session"""
        session_id = str(uuid.uuid4())
        
        await self.redis.setex(
            f"session:{session_id}",
            self.config.session_timeout,
            json.dumps({
                "username": username,
                "ip_address": ip_address,
                "created_at": self.timestamp.isoformat()
            })
        )
        
        return session_id
    
    async def _verify_session(
        self,
        username: str,
        session_id: str
    ) -> bool:
        """Verify active session"""
        session_data = await self.redis.get(f"session:{session_id}")
        
        if not session_data:
            return False
        
        session = json.loads(session_data)
        return session["username"] == username
    
    async def _monitor_security_events(self):
        """Monitor security events"""
        while True:
            try:
                # Process security events
                events = await self._get_security_events()
                
                for event in events:
                    await self._analyze_security_event(event)
                
                # Clean up expired sessions
                await self._cleanup_expired_sessions()
                
            except Exception as e:
                logger.error(f"Security monitoring failed: {e}")
            
            await asyncio.sleep(60)
    
    async def _analyze_security_event(self, event: Dict):
        """Analyze security event for threats"""
        # Implementation depends on security rules
        pass

class SecurityException(Exception):
    """Custom security exception"""
    pass

class ValidationError(Exception):
    """Input validation error"""
    pass