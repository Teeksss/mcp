import pytest
from unittest.mock import Mock, patch
from datetime import datetime, timedelta
from src.services.security.auth_service import AuthService, UserRole, Permission

@pytest.fixture
def mock_db():
    """Create a mock database session"""
    return Mock()

@pytest.fixture
def auth_service(mock_db):
    """Create an AuthService instance with mock db"""
    return AuthService(mock_db)

@pytest.mark.asyncio
async def test_user_creation():
    """Test user creation process"""
    auth_service = auth_service(mock_db)
    
    user = await auth_service.create_user(
        email="test@example.com",
        password="SecurePass123!",
        role=UserRole.USER,
        permissions=[Permission.EXECUTE_MODELS]
    )
    
    assert user is not None
    assert user.email == "test@example.com"
    assert user.role == UserRole.USER
    assert Permission.EXECUTE_MODELS in user.permissions

@pytest.mark.asyncio
async def test_user_authentication():
    """Test user authentication process"""
    auth_service = auth_service(mock_db)
    
    # Create test user
    await auth_service.create_user(
        email="test@example.com",
        password="SecurePass123!",
        role=UserRole.USER,
        permissions=[Permission.EXECUTE_MODELS]
    )
    
    # Test successful authentication
    result = await auth_service.authenticate(
        email="test@example.com",
        password="SecurePass123!"
    )
    
    assert result is not None
    assert "access_token" in result
    assert "refresh_token" in result
    
    # Test failed authentication
    with pytest.raises(HTTPException):
        await auth_service.authenticate(
            email="test@example.com",
            password="WrongPassword"
        )

@pytest.mark.asyncio
async def test_token_verification():
    """Test token verification process"""
    auth_service = auth_service(mock_db)
    
    # Create test user and get tokens
    user_data = await auth_service.authenticate(
        email="test@example.com",
        password="SecurePass123!"
    )
    
    # Verify valid token
    payload = await auth_service.verify_token(
        user_data["access_token"]
    )
    
    assert payload is not None
    assert payload["sub"] == str(user_data["user"]["id"])
    
    # Test expired token
    with pytest.raises(HTTPException):
        expired_token = auth_service._create_access_token(
            user_data["user"],
            "test_session",
            expires_delta=timedelta(seconds=-1)
        )
        await auth_service.verify_token(expired_token)