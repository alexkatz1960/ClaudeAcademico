 # ==========================================
# INTERFACES/FASTAPI_BACKEND/CORE/SECURITY.PY
# Authentication & Security - Enterprise Grade
# Sistema de Traducción Académica v2.2
# ==========================================

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, APIKeyHeader
from fastapi.security.utils import get_authorization_scheme_param
from jose import JWTError, jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any, Union
from pydantic import BaseModel, Field
from enum import Enum
import structlog
import hashlib
import secrets
import time
from functools import wraps
from collections import defaultdict
import ipaddress

# Import configuration
from .config import get_settings

logger = structlog.get_logger(__name__)

# ==========================================
# SECURITY ENUMS AND CONSTANTS
# ==========================================

class UserRole(str, Enum):
    ADMIN = "admin"
    EDITOR = "editor"
    REVIEWER = "reviewer"
    VIEWER = "viewer"
    API_CLIENT = "api_client"

class PermissionLevel(str, Enum):
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    ADMIN = "admin"

class AuthMethod(str, Enum):
    JWT = "jwt"
    API_KEY = "api_key"
    BASIC = "basic"

# Security constants
JWT_ALGORITHM = "HS256"
API_KEY_HEADER_NAME = "X-API-Key"
RATE_LIMIT_EXCEED_MESSAGE = "Rate limit exceeded"

# ==========================================
# AUTHENTICATION MODELS
# ==========================================

class User(BaseModel):
    """Modelo de usuario para autenticación"""
    username: str = Field(..., description="Nombre de usuario único")
    email: str = Field(..., description="Email del usuario")
    full_name: str = Field(..., description="Nombre completo")
    role: UserRole = Field(..., description="Rol del usuario")
    is_active: bool = Field(default=True, description="Usuario activo")
    permissions: List[PermissionLevel] = Field(default_factory=list, description="Permisos específicos")
    created_at: datetime = Field(default_factory=datetime.now)
    last_login: Optional[datetime] = None

class TokenData(BaseModel):
    """Datos del token JWT"""
    username: Optional[str] = None
    role: Optional[UserRole] = None
    permissions: List[PermissionLevel] = Field(default_factory=list)
    exp: Optional[datetime] = None

class Token(BaseModel):
    """Respuesta de token"""
    access_token: str = Field(..., description="Token de acceso JWT")
    token_type: str = Field(default="bearer", description="Tipo de token")
    expires_at: datetime = Field(..., description="Fecha de expiración")
    user: User = Field(..., description="Información del usuario")

class APIKeyInfo(BaseModel):
    """Información de API key"""
    key_id: str = Field(..., description="ID único de la API key")
    name: str = Field(..., description="Nombre descriptivo")
    role: UserRole = Field(..., description="Rol asociado")
    permissions: List[PermissionLevel] = Field(default_factory=list)
    is_active: bool = Field(default=True)
    created_at: datetime = Field(default_factory=datetime.now)
    last_used: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    rate_limit: Optional[int] = None  # requests per minute

# ==========================================
# USER DATABASE (CONFIGURABLE)
# ==========================================

class UserDatabase:
    """
    Base de datos de usuarios en memoria - Enterprise configurable
    
    Para un sistema editorial interno, usamos usuarios configurables
    en lugar de una tabla de base de datos completa.
    """
    
    def __init__(self):
        self.users: Dict[str, User] = {}
        self.api_keys: Dict[str, APIKeyInfo] = {}
        self._init_default_users()
    
    def _init_default_users(self):
        """Inicializar usuarios por defecto"""
        settings = get_settings()
        
        # Admin user
        admin_user = User(
            username="admin",
            email="admin@claudeacademico.com",
            full_name="System Administrator",
            role=UserRole.ADMIN,
            permissions=[PermissionLevel.READ, PermissionLevel.WRITE, PermissionLevel.DELETE, PermissionLevel.ADMIN]
        )
        self.users["admin"] = admin_user
        
        # Editor user
        editor_user = User(
            username="editor",
            email="editor@claudeacademico.com",
            full_name="Chief Editor",
            role=UserRole.EDITOR,
            permissions=[PermissionLevel.READ, PermissionLevel.WRITE]
        )
        self.users["editor"] = editor_user
        
        # Reviewer user
        reviewer_user = User(
            username="reviewer",
            email="reviewer@claudeacademico.com",
            full_name="Quality Reviewer",
            role=UserRole.REVIEWER,
            permissions=[PermissionLevel.READ]
        )
        self.users["reviewer"] = reviewer_user
        
        # API client for external integrations
        if not settings.is_development:
            api_client = User(
                username="api_client",
                email="api@claudeacademico.com",
                full_name="External API Client",
                role=UserRole.API_CLIENT,
                permissions=[PermissionLevel.READ, PermissionLevel.WRITE]
            )
            self.users["api_client"] = api_client
        
        logger.info(f"Initialized {len(self.users)} default users")
    
    def get_user(self, username: str) -> Optional[User]:
        """Obtener usuario por nombre"""
        return self.users.get(username)
    
    def validate_user_credentials(self, username: str, password: str) -> Optional[User]:
        """
        Validar credenciales de usuario
        
        En un sistema real, esto verificaría contra base de datos con hashes.
        Para este sistema editorial, usamos validación simplificada.
        """
        user = self.get_user(username)
        if not user or not user.is_active:
            return None
        
        # Simple password validation for demo/internal use
        # In production, this would hash and compare properly
        expected_password = self._get_user_password(username)
        if password == expected_password:
            user.last_login = datetime.now()
            return user
        
        return None
    
    def _get_user_password(self, username: str) -> str:
        """Obtener contraseña esperada para usuario (configuración simple)"""
        settings = get_settings()
        
        # In development, use simple passwords
        if settings.is_development:
            password_map = {
                "admin": "admin123",
                "editor": "editor123",
                "reviewer": "reviewer123",
                "api_client": "api123"
            }
            return password_map.get(username, "default123")
        
        # In production, use environment variables or config
        return f"{username}_{settings.secret_key[:8]}"
    
    def create_api_key(self, name: str, role: UserRole, permissions: List[PermissionLevel], 
                      expires_days: Optional[int] = None) -> tuple[str, APIKeyInfo]:
        """Crear nueva API key"""
        
        # Generate secure API key
        api_key = self._generate_api_key()
        key_id = hashlib.sha256(api_key.encode()).hexdigest()[:16]
        
        # Create API key info
        expires_at = datetime.now() + timedelta(days=expires_days) if expires_days else None
        
        api_key_info = APIKeyInfo(
            key_id=key_id,
            name=name,
            role=role,
            permissions=permissions,
            expires_at=expires_at
        )
        
        self.api_keys[api_key] = api_key_info
        
        logger.info(f"Created API key '{name}' with role {role.value}")
        return api_key, api_key_info
    
    def validate_api_key(self, api_key: str) -> Optional[APIKeyInfo]:
        """Validar API key"""
        api_key_info = self.api_keys.get(api_key)
        
        if not api_key_info or not api_key_info.is_active:
            return None
        
        # Check expiration
        if api_key_info.expires_at and datetime.now() > api_key_info.expires_at:
            logger.warning(f"API key {api_key_info.key_id} has expired")
            return None
        
        # Update last used
        api_key_info.last_used = datetime.now()
        
        return api_key_info
    
    def _generate_api_key(self) -> str:
        """Generar API key segura"""
        return f"ca_{secrets.token_urlsafe(32)}"

# Global user database instance
user_db = UserDatabase()

# ==========================================
# PASSWORD HASHING
# ==========================================

class PasswordManager:
    """Gestor de contraseñas con hashing seguro"""
    
    def __init__(self):
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verificar contraseña"""
        return self.pwd_context.verify(plain_password, hashed_password)
    
    def get_password_hash(self, password: str) -> str:
        """Obtener hash de contraseña"""
        return self.pwd_context.hash(password)

password_manager = PasswordManager()

# ==========================================
# JWT TOKEN MANAGEMENT
# ==========================================

class JWTManager:
    """Gestor de tokens JWT"""
    
    def __init__(self):
        self.settings = get_settings()
    
    def create_access_token(self, user: User, expires_delta: Optional[timedelta] = None) -> str:
        """Crear token de acceso JWT"""
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=self.settings.access_token_expire_minutes)
        
        to_encode = {
            "sub": user.username,
            "role": user.role.value,
            "permissions": [p.value for p in user.permissions],
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "access"
        }
        
        encoded_jwt = jwt.encode(to_encode, self.settings.secret_key, algorithm=JWT_ALGORITHM)
        
        logger.info(f"Created JWT token for user {user.username} (role: {user.role.value})")
        
        return encoded_jwt
    
    def verify_token(self, token: str) -> Optional[TokenData]:
        """Verificar y decodificar token JWT"""
        
        try:
            payload = jwt.decode(token, self.settings.secret_key, algorithms=[JWT_ALGORITHM])
            
            username: str = payload.get("sub")
            if username is None:
                return None
            
            role_str = payload.get("role")
            role = UserRole(role_str) if role_str else None
            
            permissions_str = payload.get("permissions", [])
            permissions = [PermissionLevel(p) for p in permissions_str]
            
            exp_timestamp = payload.get("exp")
            exp = datetime.fromtimestamp(exp_timestamp) if exp_timestamp else None
            
            return TokenData(
                username=username,
                role=role,
                permissions=permissions,
                exp=exp
            )
            
        except JWTError as e:
            logger.warning(f"JWT verification failed: {e}")
            return None
        except ValueError as e:
            logger.warning(f"Invalid role or permission in JWT: {e}")
            return None

jwt_manager = JWTManager()

# ==========================================
# RATE LIMITING
# ==========================================

class RateLimiter:
    """Rate limiter in-memory simple pero efectivo"""
    
    def __init__(self):
        self.requests: Dict[str, List[float]] = defaultdict(list)
        self.settings = get_settings()
    
    def is_allowed(self, identifier: str, max_requests: Optional[int] = None, 
                   window_seconds: Optional[int] = None) -> bool:
        """Verificar si la request está permitida"""
        
        if not self.settings.enable_rate_limiting:
            return True
        
        max_requests = max_requests or self.settings.rate_limit_requests
        window_seconds = window_seconds or self.settings.rate_limit_window
        
        now = time.time()
        window_start = now - window_seconds
        
        # Clean old requests
        self.requests[identifier] = [
            req_time for req_time in self.requests[identifier] 
            if req_time > window_start
        ]
        
        # Check if under limit
        if len(self.requests[identifier]) >= max_requests:
            logger.warning(f"Rate limit exceeded for {identifier}")
            return False
        
        # Add current request
        self.requests[identifier].append(now)
        return True
    
    def get_remaining_requests(self, identifier: str, max_requests: Optional[int] = None) -> int:
        """Obtener requests restantes en la ventana actual"""
        max_requests = max_requests or self.settings.rate_limit_requests
        current_requests = len(self.requests.get(identifier, []))
        return max(0, max_requests - current_requests)

rate_limiter = RateLimiter()

# ==========================================
# SECURITY HEADERS
# ==========================================

def get_security_headers() -> Dict[str, str]:
    """Obtener headers de seguridad según configuración"""
    settings = get_settings()
    security_config = settings.security_config
    
    headers = {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "X-XSS-Protection": "1; mode=block",
        "Referrer-Policy": "strict-origin-when-cross-origin",
    }
    
    if security_config.get("strict_transport_security", False):
        headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    
    if security_config.get("content_security_policy", False):
        headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: https:; "
            "connect-src 'self'"
        )
    
    return headers

# ==========================================
# AUTHENTICATION DEPENDENCIES
# ==========================================

# Security schemes
bearer_security = HTTPBearer(auto_error=False)
api_key_security = APIKeyHeader(name=API_KEY_HEADER_NAME, auto_error=False)

async def get_current_user_jwt(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_security)
) -> Optional[User]:
    """Obtener usuario actual desde JWT token"""
    
    if not credentials:
        return None
    
    # Verify token
    token_data = jwt_manager.verify_token(credentials.credentials)
    if not token_data:
        return None
    
    # Get user from database
    user = user_db.get_user(token_data.username)
    if not user or not user.is_active:
        return None
    
    # Check rate limiting
    identifier = f"user:{user.username}"
    if not rate_limiter.is_allowed(identifier):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=RATE_LIMIT_EXCEED_MESSAGE
        )
    
    # Log access
    logger.info(
        "User authenticated via JWT",
        username=user.username,
        role=user.role.value,
        method="jwt",
        ip=request.client.host if request.client else "unknown"
    )
    
    return user

async def get_current_user_api_key(
    request: Request,
    api_key: Optional[str] = Depends(api_key_security)
) -> Optional[User]:
    """Obtener usuario actual desde API key"""
    
    if not api_key:
        return None
    
    # Validate API key
    api_key_info = user_db.validate_api_key(api_key)
    if not api_key_info:
        return None
    
    # Check rate limiting (API keys may have custom limits)
    identifier = f"api_key:{api_key_info.key_id}"
    max_requests = api_key_info.rate_limit or None
    
    if not rate_limiter.is_allowed(identifier, max_requests):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=RATE_LIMIT_EXCEED_MESSAGE
        )
    
    # Create user object from API key info
    user = User(
        username=f"api_key_{api_key_info.key_id}",
        email=f"{api_key_info.key_id}@api.claudeacademico.com",
        full_name=api_key_info.name,
        role=api_key_info.role,
        permissions=api_key_info.permissions
    )
    
    # Log access
    logger.info(
        "User authenticated via API key",
        api_key_id=api_key_info.key_id,
        api_key_name=api_key_info.name,
        role=api_key_info.role.value,
        method="api_key",
        ip=request.client.host if request.client else "unknown"
    )
    
    return user

async def get_current_user(
    request: Request,
    jwt_user: Optional[User] = Depends(get_current_user_jwt),
    api_key_user: Optional[User] = Depends(get_current_user_api_key)
) -> Optional[User]:
    """Obtener usuario actual desde cualquier método de autenticación"""
    
    # Priority: JWT first, then API key
    user = jwt_user or api_key_user
    
    if user:
        # Add security headers to response
        request.state.security_headers = get_security_headers()
    
    return user

async def require_authenticated_user(
    current_user: Optional[User] = Depends(get_current_user)
) -> User:
    """Requerir usuario autenticado (dependencia obligatoria)"""
    
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    return current_user

# ==========================================
# PERMISSION CHECKING
# ==========================================

def require_permission(permission: PermissionLevel):
    """Decorator/dependency para requerir permiso específico"""
    
    def permission_dependency(current_user: User = Depends(require_authenticated_user)) -> User:
        if permission not in current_user.permissions:
            logger.warning(
                f"Permission denied for user {current_user.username}: "
                f"required {permission.value}, has {[p.value for p in current_user.permissions]}"
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission '{permission.value}' required"
            )
        return current_user
    
    return permission_dependency

def require_role(role: UserRole):
    """Decorator/dependency para requerir rol específico"""
    
    def role_dependency(current_user: User = Depends(require_authenticated_user)) -> User:
        if current_user.role != role:
            logger.warning(
                f"Role denied for user {current_user.username}: "
                f"required {role.value}, has {current_user.role.value}"
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{role.value}' required"
            )
        return current_user
    
    return role_dependency

def require_admin():
    """Dependency para requerir rol de administrador"""
    return require_role(UserRole.ADMIN)

def require_editor_or_admin():
    """Dependency para requerir rol de editor o admin"""
    
    def editor_admin_dependency(current_user: User = Depends(require_authenticated_user)) -> User:
        allowed_roles = [UserRole.EDITOR, UserRole.ADMIN]
        if current_user.role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Editor or Admin role required"
            )
        return current_user
    
    return editor_admin_dependency

# ==========================================
# AUTHENTICATION ENDPOINTS HELPERS
# ==========================================

async def authenticate_user(username: str, password: str) -> Optional[User]:
    """Autenticar usuario con credenciales"""
    user = user_db.validate_user_credentials(username, password)
    
    if user:
        logger.info(f"User {username} authenticated successfully")
    else:
        logger.warning(f"Authentication failed for user {username}")
    
    return user

async def create_user_token(user: User) -> Token:
    """Crear token para usuario autenticado"""
    settings = get_settings()
    
    access_token_expires = timedelta(minutes=settings.access_token_expire_minutes)
    access_token = jwt_manager.create_access_token(user, expires_delta=access_token_expires)
    
    expires_at = datetime.now() + access_token_expires
    
    return Token(
        access_token=access_token,
        expires_at=expires_at,
        user=user
    )

# ==========================================
# IP ALLOWLIST/BLOCKLIST
# ==========================================

class IPFilter:
    """Filtro de IPs para security adicional"""
    
    def __init__(self):
        self.settings = get_settings()
        self.blocked_ips: set = set()
        self.allowed_networks: List[ipaddress.IPv4Network] = []
        
        # In production, load from config
        if self.settings.is_production:
            # Allow only internal networks by default
            self.allowed_networks = [
                ipaddress.IPv4Network('10.0.0.0/8'),
                ipaddress.IPv4Network('172.16.0.0/12'),
                ipaddress.IPv4Network('192.168.0.0/16'),
                ipaddress.IPv4Network('127.0.0.0/8')
            ]
    
    def is_ip_allowed(self, ip_address: str) -> bool:
        """Verificar si IP está permitida"""
        if not self.settings.is_production:
            return True  # Allow all in development
        
        try:
            ip = ipaddress.IPv4Address(ip_address)
            
            # Check if blocked
            if ip_address in self.blocked_ips:
                return False
            
            # Check if in allowed networks
            if self.allowed_networks:
                return any(ip in network for network in self.allowed_networks)
            
            return True
            
        except ipaddress.AddressValueError:
            logger.warning(f"Invalid IP address: {ip_address}")
            return False
    
    def block_ip(self, ip_address: str, reason: str = "Security violation"):
        """Bloquear IP"""
        self.blocked_ips.add(ip_address)
        logger.warning(f"Blocked IP {ip_address}: {reason}")

ip_filter = IPFilter()

async def check_ip_allowed(request: Request):
    """Dependency para verificar IP permitida"""
    if request.client:
        client_ip = request.client.host
        if not ip_filter.is_ip_allowed(client_ip):
            logger.warning(f"Access denied for IP: {client_ip}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied from this IP address"
            )

# ==========================================
# AUDIT LOGGING
# ==========================================

class SecurityAuditor:
    """Auditor de eventos de seguridad"""
    
    def __init__(self):
        self.logger = structlog.get_logger("security.audit")
    
    def log_login_attempt(self, username: str, success: bool, ip: str, method: AuthMethod):
        """Log de intento de login"""
        self.logger.info(
            "Login attempt",
            username=username,
            success=success,
            ip_address=ip,
            auth_method=method.value,
            timestamp=datetime.now().isoformat()
        )
    
    def log_permission_check(self, username: str, permission: str, granted: bool, resource: str):
        """Log de verificación de permisos"""
        self.logger.info(
            "Permission check",
            username=username,
            permission=permission,
            granted=granted,
            resource=resource,
            timestamp=datetime.now().isoformat()
        )
    
    def log_rate_limit_exceeded(self, identifier: str, ip: str):
        """Log de rate limit excedido"""
        self.logger.warning(
            "Rate limit exceeded",
            identifier=identifier,
            ip_address=ip,
            timestamp=datetime.now().isoformat()
        )
    
    def log_security_event(self, event_type: str, details: Dict[str, Any]):
        """Log de evento de seguridad genérico"""
        self.logger.warning(
            "Security event",
            event_type=event_type,
            details=details,
            timestamp=datetime.now().isoformat()
        )

security_auditor = SecurityAuditor()

# ==========================================
# EXPORT FUNCTIONS AND CLASSES
# ==========================================

__all__ = [
    # Enums
    'UserRole',
    'PermissionLevel', 
    'AuthMethod',
    
    # Models
    'User',
    'TokenData',
    'Token',
    'APIKeyInfo',
    
    # Managers
    'UserDatabase',
    'PasswordManager',
    'JWTManager',
    'RateLimiter',
    'IPFilter',
    'SecurityAuditor',
    
    # Dependencies
    'get_current_user',
    'require_authenticated_user',
    'require_permission',
    'require_role',
    'require_admin',
    'require_editor_or_admin',
    'check_ip_allowed',
    
    # Utilities
    'authenticate_user',
    'create_user_token',
    'get_security_headers',
    
    # Instances
    'user_db',
    'password_manager',
    'jwt_manager',
    'rate_limiter',
    'ip_filter',
    'security_auditor'
]
