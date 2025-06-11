 # ==========================================
# INTERFACES/FASTAPI_BACKEND/CORE/CONFIG.PY
# Configuration Management - Enterprise Grade
# Sistema de Traducción Académica v2.2
# ==========================================

from pydantic import BaseSettings, Field, validator
from typing import Optional, List, Dict, Any, Union
from pathlib import Path
from enum import Enum
import os
import logging
import structlog
from functools import lru_cache

# Import database configurations
from ...database.database import (
    Environment, DatabaseConfig, get_config_for_environment,
    get_database_manager
)

logger = structlog.get_logger(__name__)

# ==========================================
# ENUMS AND CONSTANTS
# ==========================================

class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class CORSPolicy(str, Enum):
    STRICT = "strict"
    DEVELOPMENT = "development"
    PERMISSIVE = "permissive"

class SecurityLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    ENTERPRISE = "enterprise"

# ==========================================
# CORE SETTINGS CLASS
# ==========================================

class Settings(BaseSettings):
    """
    Configuración central del sistema FastAPI - Enterprise Grade
    
    Integra con el sistema de configuración de base de datos existente
    y proporciona configuración específica para la API REST.
    """
    
    # ==========================================
    # APPLICATION METADATA
    # ==========================================
    
    app_name: str = Field(
        default="ClaudeAcademico API v2.2",
        description="Nombre de la aplicación"
    )
    
    app_version: str = Field(
        default="2.2.0",
        description="Versión del sistema"
    )
    
    app_description: str = Field(
        default="Sistema de Traducción Académica - Enterprise API",
        description="Descripción de la aplicación"
    )
    
    app_contact: Dict[str, str] = Field(
        default={
            "name": "ClaudeAcademico Development Team",
            "email": "dev@claudeacademico.com",
            "url": "https://claudeacademico.com"
        },
        description="Información de contacto"
    )
    
    # ==========================================
    # ENVIRONMENT AND RUNTIME
    # ==========================================
    
    environment: Environment = Field(
        default=Environment.DEVELOPMENT,
        description="Entorno de ejecución"
    )
    
    debug: bool = Field(
        default=True,
        description="Modo debug activado"
    )
    
    testing: bool = Field(
        default=False,
        description="Modo testing activado"
    )
    
    reload: bool = Field(
        default=True,
        description="Auto-reload en desarrollo"
    )
    
    # ==========================================
    # SERVER CONFIGURATION
    # ==========================================
    
    host: str = Field(
        default="0.0.0.0",
        description="Host del servidor"
    )
    
    port: int = Field(
        default=8000,
        description="Puerto del servidor"
    )
    
    workers: int = Field(
        default=1,
        description="Número de workers"
    )
    
    # ==========================================
    # SECURITY CONFIGURATION
    # ==========================================
    
    secret_key: str = Field(
        default="your-super-secret-key-change-in-production",
        description="Clave secreta para JWT y hashing"
    )
    
    algorithm: str = Field(
        default="HS256",
        description="Algoritmo para JWT"
    )
    
    access_token_expire_minutes: int = Field(
        default=30,
        description="Expiración de tokens en minutos"
    )
    
    security_level: SecurityLevel = Field(
        default=SecurityLevel.MEDIUM,
        description="Nivel de seguridad del sistema"
    )
    
    allowed_hosts: List[str] = Field(
        default=["*"],
        description="Hosts permitidos"
    )
    
    # ==========================================
    # CORS CONFIGURATION
    # ==========================================
    
    cors_policy: CORSPolicy = Field(
        default=CORSPolicy.DEVELOPMENT,
        description="Política de CORS"
    )
    
    cors_origins: List[str] = Field(
        default=[
            "http://localhost:3000",
            "http://localhost:8501",  # Streamlit
            "http://127.0.0.1:3000",
            "http://127.0.0.1:8501"
        ],
        description="Orígenes permitidos para CORS"
    )
    
    cors_methods: List[str] = Field(
        default=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
        description="Métodos HTTP permitidos"
    )
    
    cors_headers: List[str] = Field(
        default=["*"],
        description="Headers permitidos"
    )
    
    # ==========================================
    # LOGGING CONFIGURATION
    # ==========================================
    
    log_level: LogLevel = Field(
        default=LogLevel.INFO,
        description="Nivel de logging"
    )
    
    log_format: str = Field(
        default="json",
        description="Formato de logs (json|text)"
    )
    
    log_file: Optional[str] = Field(
        default=None,
        description="Archivo de logs (None = stdout)"
    )
    
    enable_request_logging: bool = Field(
        default=True,
        description="Habilitar logging de requests"
    )
    
    enable_sql_logging: bool = Field(
        default=False,
        description="Habilitar logging de SQL queries"
    )
    
    # ==========================================
    # DATABASE INTEGRATION
    # ==========================================
    
    @property
    def database_config(self) -> DatabaseConfig:
        """Obtener configuración de base de datos"""
        return get_config_for_environment(self.environment.value)
    
    @property
    def database_url(self) -> str:
        """Obtener URL de base de datos"""
        return self.database_config.get_database_url()
    
    # ==========================================
    # API CONFIGURATION
    # ==========================================
    
    api_v1_prefix: str = Field(
        default="/api/v1",
        description="Prefijo para API v1"
    )
    
    docs_url: Optional[str] = Field(
        default="/docs",
        description="URL de documentación Swagger"
    )
    
    redoc_url: Optional[str] = Field(
        default="/redoc",
        description="URL de documentación ReDoc"
    )
    
    openapi_url: Optional[str] = Field(
        default="/openapi.json",
        description="URL del schema OpenAPI"
    )
    
    # ==========================================
    # RATE LIMITING
    # ==========================================
    
    enable_rate_limiting: bool = Field(
        default=True,
        description="Habilitar rate limiting"
    )
    
    rate_limit_requests: int = Field(
        default=100,
        description="Requests por minuto por IP"
    )
    
    rate_limit_window: int = Field(
        default=60,
        description="Ventana de tiempo en segundos"
    )
    
    # ==========================================
    # PROCESSING CONFIGURATION
    # ==========================================
    
    max_concurrent_books: int = Field(
        default=5,
        description="Máximo de libros procesados concurrentemente"
    )
    
    processing_timeout_minutes: int = Field(
        default=120,
        description="Timeout para procesamiento de libros"
    )
    
    max_file_size_mb: int = Field(
        default=50,
        description="Tamaño máximo de archivo en MB"
    )
    
    allowed_file_extensions: List[str] = Field(
        default=[".pdf", ".docx"],
        description="Extensiones de archivo permitidas"
    )
    
    # ==========================================
    # EXTERNAL APIS CONFIGURATION
    # ==========================================
    
    deepl_api_key: Optional[str] = Field(
        default=None,
        description="API key para DeepL"
    )
    
    claude_api_key: Optional[str] = Field(
        default=None,
        description="API key para Claude/Anthropic"
    )
    
    abbyy_api_key: Optional[str] = Field(
        default=None,
        description="API key para ABBYY"
    )
    
    api_timeout_seconds: int = Field(
        default=30,
        description="Timeout para llamadas a APIs externas"
    )
    
    api_retry_attempts: int = Field(
        default=3,
        description="Intentos de retry para APIs"
    )
    
    # ==========================================
    # CACHE CONFIGURATION
    # ==========================================
    
    enable_caching: bool = Field(
        default=True,
        description="Habilitar cache de respuestas"
    )
    
    cache_ttl_seconds: int = Field(
        default=300,
        description="TTL por defecto del cache"
    )
    
    redis_url: Optional[str] = Field(
        default="redis://localhost:6379/0",
        description="URL de Redis para cache"
    )
    
    # ==========================================
    # MONITORING AND HEALTH
    # ==========================================
    
    enable_metrics: bool = Field(
        default=True,
        description="Habilitar métricas de aplicación"
    )
    
    health_check_interval: int = Field(
        default=30,
        description="Intervalo de health checks en segundos"
    )
    
    metrics_endpoint: str = Field(
        default="/metrics",
        description="Endpoint para métricas"
    )
    
    health_endpoint: str = Field(
        default="/health",
        description="Endpoint para health check"
    )
    
    # ==========================================
    # PERFORMANCE SETTINGS
    # ==========================================
    
    request_timeout_seconds: int = Field(
        default=30,
        description="Timeout para requests HTTP"
    )
    
    max_request_size_mb: int = Field(
        default=16,
        description="Tamaño máximo de request"
    )
    
    gzip_minimum_size: int = Field(
        default=1000,
        description="Tamaño mínimo para compresión gzip"
    )
    
    # ==========================================
    # DEVELOPMENT SETTINGS
    # ==========================================
    
    enable_debug_toolbar: bool = Field(
        default=False,
        description="Habilitar toolbar de debug"
    )
    
    enable_profiler: bool = Field(
        default=False,
        description="Habilitar profiler de performance"
    )
    
    mock_external_apis: bool = Field(
        default=False,
        description="Usar mocks para APIs externas"
    )
    
    # ==========================================
    # VALIDATORS
    # ==========================================
    
    @validator('environment')
    def validate_environment(cls, v):
        """Validar entorno de ejecución"""
        if isinstance(v, str):
            return Environment(v.lower())
        return v
    
    @validator('cors_policy')
    def validate_cors_policy(cls, v):
        """Validar política de CORS"""
        if isinstance(v, str):
            return CORSPolicy(v.lower())
        return v
    
    @validator('security_level')
    def validate_security_level(cls, v):
        """Validar nivel de seguridad"""
        if isinstance(v, str):
            return SecurityLevel(v.lower())
        return v
    
    @validator('log_level')
    def validate_log_level(cls, v):
        """Validar nivel de log"""
        if isinstance(v, str):
            return LogLevel(v.upper())
        return v
    
    @validator('port')
    def validate_port(cls, v):
        """Validar puerto"""
        if not 1 <= v <= 65535:
            raise ValueError('Puerto debe estar entre 1 y 65535')
        return v
    
    @validator('workers')
    def validate_workers(cls, v):
        """Validar número de workers"""
        if v < 1:
            raise ValueError('Número de workers debe ser mayor a 0')
        return v
    
    @validator('max_file_size_mb')
    def validate_max_file_size(cls, v):
        """Validar tamaño máximo de archivo"""
        if v < 1 or v > 1000:
            raise ValueError('Tamaño de archivo debe estar entre 1MB y 1GB')
        return v
    
    # ==========================================
    # COMPUTED PROPERTIES
    # ==========================================
    
    @property
    def is_development(self) -> bool:
        """Indica si está en modo desarrollo"""
        return self.environment == Environment.DEVELOPMENT
    
    @property
    def is_production(self) -> bool:
        """Indica si está en modo producción"""
        return self.environment == Environment.PRODUCTION
    
    @property
    def is_testing(self) -> bool:
        """Indica si está en modo testing"""
        return self.environment == Environment.TESTING
    
    @property
    def cors_config(self) -> Dict[str, Any]:
        """Configuración de CORS basada en política"""
        base_config = {
            "allow_credentials": True,
            "allow_methods": self.cors_methods,
            "allow_headers": self.cors_headers,
        }
        
        if self.cors_policy == CORSPolicy.STRICT:
            base_config["allow_origins"] = []
        elif self.cors_policy == CORSPolicy.DEVELOPMENT:
            base_config["allow_origins"] = self.cors_origins
        elif self.cors_policy == CORSPolicy.PERMISSIVE:
            base_config["allow_origins"] = ["*"]
            
        return base_config
    
    @property
    def security_config(self) -> Dict[str, Any]:
        """Configuración de seguridad basada en nivel"""
        if self.security_level == SecurityLevel.LOW:
            return {
                "require_https": False,
                "strict_transport_security": False,
                "content_security_policy": False
            }
        elif self.security_level == SecurityLevel.MEDIUM:
            return {
                "require_https": False,
                "strict_transport_security": True,
                "content_security_policy": True
            }
        elif self.security_level == SecurityLevel.HIGH:
            return {
                "require_https": True,
                "strict_transport_security": True,
                "content_security_policy": True
            }
        else:  # ENTERPRISE
            return {
                "require_https": True,
                "strict_transport_security": True,
                "content_security_policy": True,
                "enable_audit_logging": True,
                "enable_encryption": True
            }
    
    @property
    def logging_config(self) -> Dict[str, Any]:
        """Configuración de logging estructurado"""
        return {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "json": {
                    "format": "%(asctime)s %(name)s %(levelname)s %(message)s",
                    "class": "pythonjsonlogger.jsonlogger.JsonFormatter"
                },
                "text": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                }
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "level": self.log_level.value,
                    "formatter": self.log_format,
                    "stream": "ext://sys.stdout"
                }
            },
            "loggers": {
                "claudeacademico": {
                    "level": self.log_level.value,
                    "handlers": ["console"],
                    "propagate": False
                },
                "uvicorn": {
                    "level": "INFO",
                    "handlers": ["console"],
                    "propagate": False
                }
            },
            "root": {
                "level": "WARNING",
                "handlers": ["console"]
            }
        }
    
    # ==========================================
    # CONFIGURATION CLASS
    # ==========================================
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        env_prefix = "CLAUDEACADEMICO_"
        
        # Allow extra fields for extensibility
        extra = "allow"
        
        # Validation configuration
        validate_assignment = True
        use_enum_values = True
        
        # Schema configuration
        schema_extra = {
            "example": {
                "app_name": "ClaudeAcademico API v2.2",
                "environment": "development",
                "debug": True,
                "host": "0.0.0.0",
                "port": 8000,
                "log_level": "INFO",
                "cors_policy": "development",
                "security_level": "medium"
            }
        }

# ==========================================
# CONFIGURATION FACTORY FUNCTIONS
# ==========================================

@lru_cache()
def get_settings() -> Settings:
    """
    Factory function para obtener configuración (singleton con cache)
    
    Utiliza lru_cache para garantizar una sola instancia
    """
    try:
        settings = Settings()
        
        # Configure structured logging
        configure_logging(settings)
        
        # Log configuration load
        logger.info(
            "Configuration loaded successfully",
            environment=settings.environment.value,
            debug=settings.debug,
            api_prefix=settings.api_v1_prefix
        )
        
        return settings
        
    except Exception as e:
        # Fallback logging if structured logging fails
        logging.error(f"Failed to load configuration: {e}")
        raise

def configure_logging(settings: Settings):
    """Configurar logging estructurado basado en settings"""
    try:
        import logging.config
        
        logging_config = settings.logging_config
        
        # Add file handler if specified
        if settings.log_file:
            file_handler = {
                "class": "logging.FileHandler",
                "level": settings.log_level.value,
                "formatter": settings.log_format,
                "filename": settings.log_file,
                "mode": "a"
            }
            logging_config["handlers"]["file"] = file_handler
            
            # Add file handler to loggers
            for logger_name in logging_config["loggers"]:
                logging_config["loggers"][logger_name]["handlers"].append("file")
        
        logging.config.dictConfig(logging_config)
        
        # Configure structlog
        structlog.configure(
            processors=[
                structlog.processors.TimeStamper(fmt="ISO"),
                structlog.processors.add_log_level,
                structlog.processors.JSONRenderer() if settings.log_format == "json" 
                else structlog.dev.ConsoleRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.WriteLoggerFactory(),
            wrapper_class=structlog.make_filtering_bound_logger(
                getattr(logging, settings.log_level.value)
            ),
            cache_logger_on_first_use=True,
        )
        
    except Exception as e:
        logging.error(f"Failed to configure logging: {e}")
        raise

def get_database_manager_from_settings():
    """Obtener database manager configurado según settings"""
    try:
        settings = get_settings()
        
        # El database manager ya está configurado en el módulo database
        # Solo necesitamos asegurar que use la configuración correcta
        db_manager = get_database_manager()
        
        # Verificar que el entorno coincida
        if db_manager.config.environment != settings.environment:
            logger.warning(
                "Environment mismatch between API and database configuration",
                api_env=settings.environment.value,
                db_env=db_manager.config.environment.value
            )
        
        return db_manager
        
    except Exception as e:
        logger.error("Failed to get database manager from settings", error=str(e))
        raise

# ==========================================
# ENVIRONMENT-SPECIFIC CONFIGURATIONS
# ==========================================

def get_development_settings() -> Settings:
    """Configuración específica para desarrollo"""
    return Settings(
        environment=Environment.DEVELOPMENT,
        debug=True,
        reload=True,
        log_level=LogLevel.DEBUG,
        cors_policy=CORSPolicy.DEVELOPMENT,
        security_level=SecurityLevel.LOW,
        enable_debug_toolbar=True,
        mock_external_apis=True
    )

def get_testing_settings() -> Settings:
    """Configuración específica para testing"""
    return Settings(
        environment=Environment.TESTING,
        debug=False,
        testing=True,
        log_level=LogLevel.WARNING,
        cors_policy=CORSPolicy.STRICT,
        security_level=SecurityLevel.MEDIUM,
        docs_url=None,  # Disable docs in testing
        redoc_url=None,
        openapi_url=None
    )

def get_production_settings() -> Settings:
    """Configuración específica para producción"""
    return Settings(
        environment=Environment.PRODUCTION,
        debug=False,
        reload=False,
        log_level=LogLevel.INFO,
        cors_policy=CORSPolicy.STRICT,
        security_level=SecurityLevel.ENTERPRISE,
        enable_debug_toolbar=False,
        mock_external_apis=False,
        docs_url=None,  # Security: disable docs in production
        redoc_url=None
    )

# ==========================================
# CONFIGURATION VALIDATION
# ==========================================

def validate_configuration(settings: Settings) -> bool:
    """Validar configuración completa del sistema"""
    try:
        validation_logger = structlog.get_logger("config.validation")
        
        # Validate database connectivity
        db_manager = get_database_manager_from_settings()
        if not db_manager.check_connection():
            validation_logger.error("Database connection failed")
            return False
        
        # Validate required API keys in production
        if settings.is_production:
            required_keys = ['deepl_api_key', 'claude_api_key', 'abbyy_api_key']
            for key in required_keys:
                if not getattr(settings, key):
                    validation_logger.error(f"Missing required API key in production: {key}")
                    return False
        
        # Validate security settings
        if settings.is_production and settings.security_level == SecurityLevel.LOW:
            validation_logger.warning("Low security level in production environment")
        
        # Validate file size limits
        if settings.max_file_size_mb > 100:
            validation_logger.warning(f"Large file size limit: {settings.max_file_size_mb}MB")
        
        validation_logger.info(
            "Configuration validation passed",
            environment=settings.environment.value,
            security_level=settings.security_level.value
        )
        
        return True
        
    except Exception as e:
        validation_logger.error("Configuration validation failed", error=str(e))
        return False

# ==========================================
# EXPORT FUNCTIONS
# ==========================================

# Main settings instance (singleton)
settings = get_settings()

# Export primary functions and classes
__all__ = [
    # Core classes
    'Settings',
    'LogLevel',
    'CORSPolicy', 
    'SecurityLevel',
    'Environment',
    
    # Factory functions
    'get_settings',
    'get_development_settings',
    'get_testing_settings', 
    'get_production_settings',
    
    # Utility functions
    'configure_logging',
    'get_database_manager_from_settings',
    'validate_configuration',
    
    # Main instance
    'settings'
]
