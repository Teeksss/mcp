class MCPError(Exception):
    """Base exception for MCP server"""
    pass

class ConfigurationError(MCPError):
    """Configuration related errors"""
    pass

class ValidationError(MCPError):
    """Data validation errors"""
    pass

class ResourceNotFoundError(MCPError):
    """Resource not found errors"""
    pass

class ServiceError(MCPError):
    """Service operation errors"""
    pass

class AuthenticationError(MCPError):
    """Authentication related errors"""
    pass

class RateLimitError(MCPError):
    """Rate limiting errors"""
    pass

class DatabaseError(MCPError):
    """Database operation errors"""
    pass

class StorageError(MCPError):
    """Storage operation errors"""
    pass

class TrainingError(MCPError):
    """Model training errors"""
    pass

class DeploymentError(MCPError):
    """Model deployment errors"""
    pass

class InferenceError(MCPError):
    """Model inference errors"""
    pass

class StreamingError(MCPError):
    """Event streaming errors"""
    pass

class FeatureStoreError(MCPError):
    """Feature store errors"""
    pass