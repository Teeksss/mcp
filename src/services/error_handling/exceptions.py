class MCPException(Exception):
    pass

class ModelNotFound(MCPException):
    pass

class UnauthorizedAccess(MCPException):
    pass