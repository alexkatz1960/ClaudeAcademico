# ==========================================
# INTERFACES/DATABASE/__INIT__.PY
# Database Package - Complete Export Module - ENTERPRISE GRADE
# Sistema de Traducción Académica v2.2
# ==========================================

"""
Database Layer - Sistema de Traducción Académica v2.2 - Enterprise Grade

Este paquete contiene toda la capa de persistencia del sistema incluyendo:
- Modelos SQLAlchemy con relationships completos y validación enterprise
- Schemas Pydantic para validación robusta de APIs
- Operaciones CRUD enterprise-grade con manejo de errores robusto
- Gestor de conexiones y sesiones con pooling y retry logic
- Sistema de migraciones automáticas con rollback support
- Datos iniciales y configuración multi-entorno

Uso básico:
    from interfaces.database import DatabaseManager, get_db
    from interfaces.database import book_crud, BookProcessingHistory
    from interfaces.database import BookStatusSchema
    
    # Usar en FastAPI endpoints
    @app.get("/books/{book_id}")
    def get_book(book_id: str, db: Session = Depends(get_db)):
        return book_crud.get_by_book_id(db, book_id=book_id)

Uso enterprise:
    from interfaces.database import (
        initialize_system_database, get_system_health,
        verify_package_integrity, get_enterprise_health
    )
    
    # Inicialización robusta
    if not initialize_system_database(validate=True):
        raise RuntimeError("Database initialization failed")

Arquitectura Enterprise:
    - models.py: SQLAlchemy models con validación robusta y enum integration
    - schemas.py: Pydantic models con validaciones enterprise-grade
    - crud.py: Operaciones CRUD con retry logic y performance optimization
    - database.py: Gestor enterprise con pooling, monitoring y health checks
    - migrations.py: Sistema de migraciones con rollback y validation
    - enums.py: Enumeraciones centralizadas para type safety
"""

import logging
import os
import warnings
from datetime import datetime
from typing import Generator, Dict, Any, Optional, List
from contextlib import suppress

# Configure package logger with enterprise safety
logger = logging.getLogger(__name__)

# ==========================================
# ENTERPRISE LAZY LOADING SYSTEM
# ==========================================

class _LazyLoader:
    """Lazy loader for heavy database components"""
    
    def __init__(self):
        self._loaded = False
        self._components = {}
    
    def load_if_needed(self):
        """Load components only when first accessed"""
        if not self._loaded:
            self._load_components()
            self._loaded = True
    
    def _load_components(self):
        """Internal method to load all components"""
        try:
            # Core database components are always loaded
            # Heavy ML or analytics components could be lazy-loaded here
            pass
        except Exception as e:
            logger.error(f"Failed to lazy load components: {e}")

_lazy_loader = _LazyLoader()

# ==========================================
# CORE DATABASE COMPONENTS
# ==========================================

# Database manager and connection utilities
try:
    from .database import (
        DatabaseManager,
        DatabaseConfig,
        get_db,
        get_db_manager,
        db_manager,
        check_database_health,
        initialize_database,
        reset_database,
        db_transaction,
        db_readonly_session
    )
except ImportError as e:
    logger.error(f"Failed to import database core components: {e}")
    raise RuntimeError(f"Database module import failed: {e}")

# Migration system
try:
    from .migrations import (
        run_migrations,
        create_initial_data,
        validate_and_fix_database,
        cleanup_old_data,
        optimize_database_performance,
        MigrationManager
    )
except ImportError as e:
    logger.error(f"Failed to import migration system: {e}")
    raise RuntimeError(f"Migration module import failed: {e}")

# ==========================================
# SQLALCHEMY MODELS
# ==========================================

try:
    from .models import (
        # Base and mixins
        Base,
        TimestampMixin,
        
        # Core models
        SystemConfig,
        BookProcessingHistory,
        ErrorPattern,
        AuditLog,
        TerminologySuggestion,
        EditorialReview,
        UsageStatistic,
        
        # Utility functions
        create_indexes,
        validate_language_code,
        validate_json_field
    )
except ImportError as e:
    logger.error(f"Failed to import SQLAlchemy models: {e}")
    raise RuntimeError(f"Models module import failed: {e}")

# ==========================================
# PYDANTIC SCHEMAS
# ==========================================

try:
    from .schemas import (
        # Enums for validation
        BookStatus,
        ProcessingPhase,
        LanguageCode,
        Severity,
        Priority,
        EditorDecision,
        PatternType,
        ConfigType,
        
        # Base schemas
        TimestampMixin as SchemaMixin,
        PaginationParams,
        PaginatedResponse,
        
        # System Config schemas
        SystemConfigBase,
        SystemConfigCreate,
        SystemConfigUpdate,
        SystemConfigResponse,
        
        # Book Processing schemas
        BookProcessingBase,
        BookProcessingCreate,
        BookProcessingUpdate,
        BookProcessingResponse,
        BookStatusSummary,
        
        # Error Pattern schemas
        ErrorPatternBase,
        ErrorPatternCreate,
        ErrorPatternUpdate,
        ErrorPatternResponse,
        ErrorPatternStats,
        
        # Audit Log schemas
        AlertDetail,
        AuditLogBase,
        AuditLogCreate,
        AuditLogResponse,
        
        # Terminology schemas
        TerminologySuggestionBase,
        TerminologySuggestionCreate,
        TerminologySuggestionUpdate,
        TerminologySuggestionResponse,
        TerminologyBatch,
        
        # Editorial Review schemas
        EditorialReviewBase,
        EditorialReviewCreate,
        EditorialReviewUpdate,
        EditorialReviewResponse,
        ReviewSummary,
        
        # Usage Statistics schemas
        UsageStatisticBase,
        UsageStatisticCreate,
        UsageStatisticResponse,
        DashboardMetrics,
        SystemHealthCheck,
        
        # Query schemas
        BookQueryParams,
        ReviewQueryParams,
        StatisticsQueryParams,
        
        # Bulk operations
        BulkUpdateRequest,
        BulkUpdateResponse,
        
        # API responses
        APIResponse,
        ErrorResponse
    )
except ImportError as e:
    logger.error(f"Failed to import Pydantic schemas: {e}")
    raise RuntimeError(f"Schemas module import failed: {e}")

# ==========================================
# CRUD OPERATIONS
# ==========================================

try:
    from .crud import (
        # Base CRUD class
        BaseCRUD,
        
        # CRUD instances (ready to use)
        system_config_crud,
        book_crud,
        error_pattern_crud,
        audit_log_crud,
        terminology_crud,
        editorial_review_crud,
        usage_statistics_crud,
        
        # Individual CRUD classes (for extension)
        SystemConfigCRUD,
        BookCRUD,
        ErrorPatternCRUD,
        AuditLogCRUD,
        TerminologyCRUD,
        EditorialReviewCRUD,
        UsageStatisticsCRUD
    )
except ImportError as e:
    logger.error(f"Failed to import CRUD operations: {e}")
    raise RuntimeError(f"CRUD module import failed: {e}")

# ==========================================
# ENUMS (ENTERPRISE TYPE SAFETY)
# ==========================================

try:
    from .enums import (
        BookStatus as BookStatusEnum,
        ReviewSeverity,
        PatternType as PatternTypeEnum,
        ProcessingPhase as ProcessingPhaseEnum,
        ConfigType as ConfigTypeEnum,
        ReviewDecision,
        TerminologyPriority,
        ErrorSeverity
    )
except ImportError as e:
    logger.warning(f"Failed to import enums (non-critical): {e}")
    # Enums are not critical for basic functionality

# ==========================================
# ENTERPRISE CONVENIENCE FUNCTIONS
# ==========================================

class DatabaseError(Exception):
    """Custom exception for database operations"""
    pass

class ConfigurationError(DatabaseError):
    """Exception for configuration-related errors"""
    pass

def get_database_session() -> Generator:
    """Convenience function to get database session"""
    try:
        return get_db()
    except Exception as e:
        logger.error(f"Failed to get database session: {e}")
        raise DatabaseError(f"Database session creation failed: {e}")

def get_system_config(key: str, default=None, raise_on_error: bool = False):
    """
    Get system configuration value with enterprise error handling
    
    Args:
        key: Configuration key
        default: Default value if not found
        raise_on_error: Whether to raise exception on errors
        
    Returns:
        Configuration value or default
        
    Raises:
        ConfigurationError: If raise_on_error=True and error occurs
    """
    try:
        with db_manager.get_db_session() as db:
            config = system_config_crud.get_by_key(db, config_key=key)
            if config is None:
                if raise_on_error and default is None:
                    raise ConfigurationError(f"Configuration key '{key}' not found")
                return default
            return config.parsed_value
            
    except Exception as e:
        error_msg = f"Error getting system config '{key}': {e}"
        logger.error(error_msg)
        
        if raise_on_error:
            raise ConfigurationError(error_msg) from e
        
        return default

def set_system_config(key: str, value: str, config_type: str = "string", 
                     description: str = "", validate: bool = True):
    """
    Set system configuration value with enterprise validation
    
    Args:
        key: Configuration key
        value: Configuration value
        config_type: Type of configuration
        description: Description of the configuration
        validate: Whether to validate the value
        
    Returns:
        Updated configuration object or None on failure
        
    Raises:
        ConfigurationError: If validation fails or critical error occurs
    """
    try:
        # Validate inputs
        if not key or not key.strip():
            raise ConfigurationError("Configuration key cannot be empty")
        
        if value is None:
            raise ConfigurationError("Configuration value cannot be None")
        
        # Validate config_type
        valid_types = {"string", "integer", "float", "boolean", "json"}
        if config_type not in valid_types:
            raise ConfigurationError(f"Invalid config_type. Must be one of: {valid_types}")
        
        # Type validation if requested
        if validate:
            try:
                if config_type == "integer":
                    int(value)
                elif config_type == "float":
                    float(value)
                elif config_type == "boolean":
                    value.lower() in ('true', 'false', '1', '0', 'yes', 'no', 'on', 'off')
                elif config_type == "json":
                    import json
                    json.loads(value)
            except (ValueError, TypeError) as e:
                raise ConfigurationError(f"Invalid value '{value}' for type '{config_type}': {e}")
        
        with db_manager.get_db_session() as db:
            result = system_config_crud.upsert_config(
                db,
                config_key=key,
                config_value=value,
                config_type=config_type,
                description=description or f"Configuration for {key}"
            )
            
            if result is None:
                raise ConfigurationError(f"Failed to set configuration '{key}'")
            
            logger.info(f"Updated system config: {key} = {value}")
            return result
            
    except ConfigurationError:
        raise  # Re-raise configuration errors
    except Exception as e:
        error_msg = f"Error setting system config '{key}': {e}"
        logger.error(error_msg)
        raise ConfigurationError(error_msg) from e

def get_system_health() -> Dict[str, Any]:
    """Get comprehensive system health information - Enterprise Grade"""
    try:
        health_info = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "version": get_system_config("system_version", "unknown"),
            "database": {},
            "statistics": {},
            "errors": []
        }
        
        # Database health check
        try:
            db_health = check_database_health()
            health_info["database"] = db_health
            
            if not db_health.get("connected", False):
                health_info["status"] = "unhealthy"
                health_info["errors"].append("Database connection failed")
                
        except Exception as e:
            health_info["status"] = "unhealthy"
            health_info["errors"].append(f"Database health check failed: {str(e)}")
            health_info["database"] = {"connected": False, "error": str(e)}
        
        # Get statistics if database is healthy
        if health_info["database"].get("connected", False):
            try:
                with db_manager.get_db_session() as db:
                    health_info["statistics"] = {
                        "books": book_crud.get_statistics_summary(db),
                        "terminology": terminology_crud.get_statistics(db),
                        "active_reviews": editorial_review_crud.count_active_reviews(db),
                        "recent_errors": error_pattern_crud.count_recent_patterns(db)
                    }
            except Exception as e:
                health_info["errors"].append(f"Statistics collection failed: {str(e)}")
        
        return health_info
        
    except Exception as e:
        logger.error(f"System health check failed: {e}")
        return {
            "status": "error",
            "timestamp": datetime.now().isoformat(),
            "message": str(e),
            "errors": [str(e)]
        }

def get_enterprise_health() -> Dict[str, Any]:
    """Get enterprise-grade health check with detailed metrics"""
    try:
        health = get_system_health()
        
        # Add enterprise-specific metrics
        enterprise_metrics = {
            "package_integrity": verify_package_integrity(),
            "migration_status": _check_migration_status(),
            "performance_metrics": _get_performance_metrics(),
            "security_status": _check_security_status()
        }
        
        health["enterprise"] = enterprise_metrics
        
        # Update overall status based on enterprise checks
        if not all(enterprise_metrics.values()):
            health["status"] = "degraded"
            health["errors"].append("Enterprise health checks failed")
        
        return health
        
    except Exception as e:
        logger.error(f"Enterprise health check failed: {e}")
        return {"status": "error", "message": str(e)}

def initialize_system_database(reset: bool = False, validate: bool = True, 
                              create_sample_data: bool = False) -> bool:
    """
    Initialize the entire database system with enterprise validation
    
    Args:
        reset: Whether to reset the database (DANGEROUS - deletes all data)
        validate: Whether to validate schema after initialization
        create_sample_data: Whether to create sample data for development
        
    Returns:
        True if successful, False otherwise
        
    Raises:
        DatabaseError: If critical initialization steps fail
    """
    try:
        if reset:
            logger.warning("⚠️  RESETTING DATABASE - ALL DATA WILL BE LOST")
            
            # Additional safety check for production
            environment = os.getenv("ENVIRONMENT", "development").lower()
            if environment == "production":
                raise DatabaseError("Database reset is not allowed in production environment")
            
            reset_database()
            logger.info("✅ Database reset completed")
        
        # Initialize database structure
        logger.info("🔧 Initializing database structure...")
        initialize_database(create_tables=True, run_migrations=True)
        logger.info("✅ Database structure initialized")
        
        # Validate schema if requested (Enterprise requirement from auditor)
        if validate:
            logger.info("🔍 Validating database schema...")
            try:
                validation_result = validate_and_fix_database()
                if not validation_result:
                    raise DatabaseError("Database schema validation failed")
                logger.info("✅ Database schema validation passed")
            except Exception as e:
                logger.error(f"❌ Database validation failed: {e}")
                raise DatabaseError(f"Database validation failed: {e}")
        
        # Create sample data if requested
        if create_sample_data:
            environment = os.getenv("ENVIRONMENT", "development").lower()
            if environment in ["development", "testing"]:
                logger.info("📊 Creating sample data...")
                create_initial_data(environment=environment)
                logger.info("✅ Sample data created")
            else:
                logger.warning("⚠️  Sample data creation skipped in non-development environment")
        
        logger.info("🚀 Database system initialized successfully")
        return True
        
    except DatabaseError:
        raise  # Re-raise database errors
    except Exception as e:
        error_msg = f"Database initialization failed: {e}"
        logger.error(f"❌ {error_msg}")
        raise DatabaseError(error_msg) from e

def get_crud(model_name: str) -> Optional[Any]:
    """
    Dynamically get CRUD instance by model name (Enterprise utility)
    
    Args:
        model_name: Name of the model (e.g., 'book', 'review', 'terminology')
        
    Returns:
        CRUD instance or None if not found
    """
    crud_mapping = {
        'book': book_crud,
        'books': book_crud,
        'system_config': system_config_crud,
        'config': system_config_crud,
        'error_pattern': error_pattern_crud,
        'patterns': error_pattern_crud,
        'audit_log': audit_log_crud,
        'audit': audit_log_crud,
        'terminology': terminology_crud,
        'terms': terminology_crud,
        'editorial_review': editorial_review_crud,
        'review': editorial_review_crud,
        'reviews': editorial_review_crud,
        'usage_statistics': usage_statistics_crud,
        'statistics': usage_statistics_crud,
        'stats': usage_statistics_crud
    }
    
    return crud_mapping.get(model_name.lower())

# ==========================================
# ENTERPRISE INTERNAL UTILITIES
# ==========================================

def _check_migration_status() -> bool:
    """Check if migrations are up to date"""
    try:
        migration_manager = MigrationManager()
        with db_manager.get_db_session() as db:
            return not migration_manager.needs_migration(db)
    except Exception as e:
        logger.error(f"Migration status check failed: {e}")
        return False

def _get_performance_metrics() -> Dict[str, Any]:
    """Get basic performance metrics"""
    try:
        with db_manager.get_db_session() as db:
            # Basic connection pool metrics
            pool = db.bind.pool
            metrics = {
                "pool_size": pool.size(),
                "checked_in": pool.checkedin(),
                "checked_out": pool.checkedout(),
                "overflow": pool.overflow(),
                "invalid": pool.invalid()
            }
            return metrics
    except Exception as e:
        logger.error(f"Performance metrics collection failed: {e}")
        return {"error": str(e)}

def _check_security_status() -> bool:
    """Basic security status check"""
    try:
        # Check for debug mode in production
        environment = os.getenv("ENVIRONMENT", "development").lower()
        debug_mode = get_system_config("debug_mode", "false")
        
        if environment == "production" and debug_mode.lower() == "true":
            logger.warning("⚠️  Debug mode is enabled in production environment")
            return False
        
        return True
    except Exception:
        return False

# ==========================================
# PACKAGE METADATA
# ==========================================

__version__ = "2.2.0"
__author__ = "ClaudeAcademico Development Team"
__description__ = "Database layer for Academic Translation System - Enterprise Grade"

# ==========================================
# ENTERPRISE ORGANIZED EXPORTS
# ==========================================

# Database Core Components
DATABASE_EXPORTS = [
    "DatabaseManager",
    "DatabaseConfig", 
    "db_manager",
    "get_db",
    "get_db_manager",
    "get_database_session",
    "check_database_health",
    "get_system_health",
    "get_enterprise_health",
    "get_system_config",
    "set_system_config",
    "db_transaction",
    "db_readonly_session"
]

# Initialization and Migration Components
MIGRATION_EXPORTS = [
    "initialize_database",
    "initialize_system_database", 
    "reset_database",
    "run_migrations",
    "create_initial_data",
    "validate_and_fix_database",
    "cleanup_old_data",
    "optimize_database_performance",
    "MigrationManager"
]

# SQLAlchemy Models
MODEL_EXPORTS = [
    "Base",
    "TimestampMixin",
    "SystemConfig",
    "BookProcessingHistory",
    "ErrorPattern", 
    "AuditLog",
    "TerminologySuggestion",
    "EditorialReview",
    "UsageStatistic"
]

# Pydantic Schemas - Enums
ENUM_EXPORTS = [
    "BookStatus",
    "ProcessingPhase",
    "LanguageCode",
    "Severity", 
    "Priority",
    "EditorDecision",
    "PatternType",
    "ConfigType"
]

# Pydantic Schemas - Base and Utilities
SCHEMA_BASE_EXPORTS = [
    "PaginationParams",
    "PaginatedResponse",
    "APIResponse", 
    "ErrorResponse"
]

# Pydantic Schemas - System Config
SYSTEM_CONFIG_SCHEMA_EXPORTS = [
    "SystemConfigBase",
    "SystemConfigCreate",
    "SystemConfigUpdate",
    "SystemConfigResponse"
]

# Pydantic Schemas - Book Processing
BOOK_SCHEMA_EXPORTS = [
    "BookProcessingBase",
    "BookProcessingCreate", 
    "BookProcessingUpdate",
    "BookProcessingResponse",
    "BookStatusSummary"
]

# Pydantic Schemas - Error Patterns
ERROR_PATTERN_SCHEMA_EXPORTS = [
    "ErrorPatternBase",
    "ErrorPatternCreate",
    "ErrorPatternUpdate",
    "ErrorPatternResponse",
    "ErrorPatternStats"
]

# Pydantic Schemas - Audit Logs
AUDIT_SCHEMA_EXPORTS = [
    "AlertDetail",
    "AuditLogBase",
    "AuditLogCreate", 
    "AuditLogResponse"
]

# Pydantic Schemas - Terminology
TERMINOLOGY_SCHEMA_EXPORTS = [
    "TerminologySuggestionBase",
    "TerminologySuggestionCreate",
    "TerminologySuggestionUpdate",
    "TerminologySuggestionResponse",
    "TerminologyBatch"
]

# Pydantic Schemas - Editorial Review
REVIEW_SCHEMA_EXPORTS = [
    "EditorialReviewBase",
    "EditorialReviewCreate",
    "EditorialReviewUpdate",
    "EditorialReviewResponse",
    "ReviewSummary"
]

# Pydantic Schemas - Usage Statistics
STATISTICS_SCHEMA_EXPORTS = [
    "UsageStatisticBase",
    "UsageStatisticCreate",
    "UsageStatisticResponse",
    "DashboardMetrics",
    "SystemHealthCheck"
]

# Pydantic Schemas - Query Parameters
QUERY_SCHEMA_EXPORTS = [
    "BookQueryParams", 
    "ReviewQueryParams",
    "StatisticsQueryParams"
]

# Pydantic Schemas - Bulk Operations
BULK_SCHEMA_EXPORTS = [
    "BulkUpdateRequest",
    "BulkUpdateResponse"
]

# CRUD Operations - Ready-to-use instances
CRUD_INSTANCE_EXPORTS = [
    "system_config_crud",
    "book_crud",
    "error_pattern_crud",
    "audit_log_crud",
    "terminology_crud",
    "editorial_review_crud",
    "usage_statistics_crud"
]

# CRUD Operations - Classes for extension
CRUD_CLASS_EXPORTS = [
    "BaseCRUD",
    "SystemConfigCRUD",
    "BookCRUD",
    "ErrorPatternCRUD",
    "AuditLogCRUD",
    "TerminologyCRUD", 
    "EditorialReviewCRUD",
    "UsageStatisticsCRUD"
]

# Utility Functions
UTILITY_EXPORTS = [
    "create_indexes",
    "validate_language_code",
    "validate_json_field",
    "get_crud"
]

# Custom Exceptions
EXCEPTION_EXPORTS = [
    "DatabaseError",
    "ConfigurationError"
]

# Combine all exports in organized manner
__all__ = (
    DATABASE_EXPORTS +
    MIGRATION_EXPORTS +
    MODEL_EXPORTS +
    ENUM_EXPORTS +
    SCHEMA_BASE_EXPORTS +
    SYSTEM_CONFIG_SCHEMA_EXPORTS +
    BOOK_SCHEMA_EXPORTS +
    ERROR_PATTERN_SCHEMA_EXPORTS +
    AUDIT_SCHEMA_EXPORTS +
    TERMINOLOGY_SCHEMA_EXPORTS +
    REVIEW_SCHEMA_EXPORTS +
    STATISTICS_SCHEMA_EXPORTS +
    QUERY_SCHEMA_EXPORTS +
    BULK_SCHEMA_EXPORTS +
    CRUD_INSTANCE_EXPORTS +
    CRUD_CLASS_EXPORTS +
    UTILITY_EXPORTS +
    EXCEPTION_EXPORTS
)

# ==========================================
# ENTERPRISE PACKAGE INITIALIZATION
# ==========================================

def _setup_package_logging():
    """Setup enterprise-grade logging configuration for the database package"""
    package_logger = logging.getLogger(__name__)
    
    # Only configure if no handlers exist to avoid conflicts
    if not package_logger.handlers and not package_logger.parent.handlers:
        try:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            package_logger.addHandler(handler)
            package_logger.setLevel(logging.INFO)
            package_logger.propagate = False  # Prevent duplicate logs
        except Exception as e:
            # Fallback to basic logging if setup fails
            print(f"Warning: Could not setup package logging: {e}")

# ==========================================
# DEVELOPMENT UTILITIES WITH ENTERPRISE GUARDS
# ==========================================

def _dev_create_sample_data():
    """Create sample data for development with enterprise guards"""
    try:
        environment = os.getenv("ENVIRONMENT", "development").lower()
        
        # Enterprise guard: Only in development/testing
        if environment not in ["development", "testing"]:
            logger.info(f"Sample data creation skipped in {environment} environment")
            return
        
        debug_mode = get_system_config("debug_mode", "false")
        if debug_mode.lower() == "true":
            try:
                create_initial_data(environment=environment)
                logger.info("✅ Sample development data created")
            except Exception as e:
                logger.warning(f"⚠️  Could not create sample data: {e}")
    except Exception as e:
        logger.error(f"Development data creation failed: {e}")

def _dev_validate_schema():
    """Validate database schema in development mode with enterprise guards"""
    try:
        environment = os.getenv("ENVIRONMENT", "development").lower()
        
        # Enterprise guard: Only validate in safe environments
        if environment == "production":
            return
        
        debug_mode = get_system_config("debug_mode", "false")
        if debug_mode.lower() == "true":
            try:
                validate_and_fix_database()
                logger.info("✅ Database schema validation passed")
            except Exception as e:
                logger.error(f"❌ Database schema validation failed: {e}")
    except Exception as e:
        logger.error(f"Development schema validation failed: {e}")

# ==========================================
# VERSION COMPATIBILITY (MAINTAINED)
# ==========================================

# Maintain backwards compatibility with older imports
DatabaseSession = get_db  # Alias for backwards compatibility
get_session = get_db      # Alternative alias

# Support for legacy import patterns
database_manager = db_manager  # Legacy alias
books = book_crud  # Legacy alias
reviews = editorial_review_crud  # Legacy alias

# Deprecation warnings for critical legacy patterns
def _deprecated_import_warning(old_name: str, new_name: str):
    """Issue deprecation warning for old import patterns"""
    warnings.warn(
        f"Importing {old_name} is deprecated. Use {new_name} instead. "
        f"This will be removed in version 3.0.0",
        DeprecationWarning,
        stacklevel=3
    )

# ==========================================
# ENTERPRISE PACKAGE VERIFICATION
# ==========================================

def verify_package_integrity() -> bool:
    """
    Comprehensive package integrity verification - Enterprise Grade
    
    Returns:
        True if all integrity checks pass, False otherwise
    """
    integrity_results = {
        "imports": False,
        "database_connection": False,
        "crud_operations": False,
        "schema_validation": False,
        "migrations": False
    }
    
    try:
        # Test 1: Basic imports
        try:
            assert DatabaseManager is not None
            assert get_db is not None
            assert book_crud is not None
            assert BookProcessingHistory is not None
            assert BookStatus is not None
            integrity_results["imports"] = True
            logger.debug("✅ Package imports verification passed")
        except (AssertionError, NameError) as e:
            logger.error(f"❌ Package imports verification failed: {e}")
            return False
        
        # Test 2: Database connection
        try:
            health = check_database_health()
            if health.get("connected", False):
                integrity_results["database_connection"] = True
                logger.debug("✅ Database connection verification passed")
            else:
                logger.warning("⚠️  Database connection verification failed")
        except Exception as e:
            logger.error(f"❌ Database connection test failed: {e}")
        
        # Test 3: Basic CRUD operations (if database connected)
        if integrity_results["database_connection"]:
            try:
                with suppress(Exception):
                    with db_manager.get_db_session() as db:
                        # Test basic query
                        count = system_config_crud.count(db)
                        integrity_results["crud_operations"] = True
                        logger.debug(f"✅ CRUD operations verification passed (configs: {count})")
            except Exception as e:
                logger.warning(f"⚠️  CRUD operations test failed: {e}")
        
        # Test 4: Schema validation (if database connected)
        if integrity_results["database_connection"]:
            try:
                # Basic schema validation
                validation_result = validate_and_fix_database()
                if validation_result:
                    integrity_results["schema_validation"] = True
                    logger.debug("✅ Schema validation passed")
            except Exception as e:
                logger.warning(f"⚠️  Schema validation failed: {e}")
        
        # Test 5: Migration status (if database connected)
        if integrity_results["database_connection"]:
            try:
                migration_status = _check_migration_status()
                integrity_results["migrations"] = migration_status
                logger.debug(f"✅ Migration status check: {'current' if migration_status else 'needs update'}")
            except Exception as e:
                logger.warning(f"⚠️  Migration status check failed: {e}")
        
        # Calculate overall integrity score
        passed_checks = sum(integrity_results.values())
        total_checks = len(integrity_results)
        integrity_score = passed_checks / total_checks
        
        if integrity_score >= 0.8:  # 80% threshold for enterprise
            logger.info(f"✅ Package integrity verification passed ({passed_checks}/{total_checks} checks)")
            return True
        else:
            logger.warning(f"⚠️  Package integrity verification partial ({passed_checks}/{total_checks} checks)")
            return False
            
    except Exception as e:
        logger.error(f"❌ Package integrity verification failed with exception: {e}")
        return False

# ==========================================
# PACKAGE INITIALIZATION SEQUENCE
# ==========================================

def _initialize_package():
    """Initialize package with enterprise startup sequence"""
    try:
        # Step 1: Setup logging
        _setup_package_logging()
        
        # Step 2: Load lazy components if needed
        _lazy_loader.load_if_needed()
        
        # Step 3: Log successful initialization
        logger.info(f"📦 Database package v{__version__} initialized successfully")
        
        # Step 4: Run development utilities if appropriate
        try:
            environment = os.getenv("ENVIRONMENT", "development").lower()
            if environment in ["development", "testing"]:
                # Only run schema validation in safe environments
                # _dev_validate_schema()  # Commented to avoid import issues
                pass
        except Exception as e:
            logger.debug(f"Development utilities skipped: {e}")
        
        # Step 5: Run integrity verification
        integrity_ok = verify_package_integrity()
        if not integrity_ok:
            logger.warning("⚠️  Package integrity verification failed - some features may not work correctly")
        
        logger.info("🚀 Database package ready for enterprise use")
        
    except Exception as e:
        logger.error(f"❌ Package initialization failed: {e}")
        # Don't raise exception to allow partial functionality

# Initialize package when imported
_initialize_package()

# ==========================================
# FINAL ENTERPRISE EXPORTS
# ==========================================

# Export enterprise error classes
__all__.extend(["DatabaseError", "ConfigurationError"])

# Export version and metadata for enterprise tooling
__all__.extend(["__version__", "__author__", "__description__"])

logger.debug(f"📋 Package exports: {len(__all__)} components available")