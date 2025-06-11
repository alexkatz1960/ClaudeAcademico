# ==========================================
# INTERFACES/DATABASE/DATABASE.PY
# Database Connection & Session Management - Enterprise
# Sistema de Traducción Académica v2.2
# ==========================================

from sqlalchemy import create_engine, event, text, Column, String, Integer, DateTime
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
from sqlalchemy.ext.declarative import declarative_base
from contextlib import contextmanager
from typing import Generator, Optional, Dict, Any, Type
import logging
import structlog
import os
import time
from pathlib import Path
import sqlite3
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
import json
from dataclasses import dataclass

logger = structlog.get_logger(__name__)

# ==========================================
# SCHEMA VERSION MANAGEMENT
# ==========================================

class SchemaVersion:
    """Gestión de versiones de esquema de base de datos"""
    CURRENT_VERSION = "2.2.0"
    COMPATIBLE_VERSIONS = ["2.1.0", "2.2.0"]

# Tabla para tracking de versión de esquema
SchemaVersionBase = declarative_base()

class DatabaseSchemaVersion(SchemaVersionBase):
    __tablename__ = 'schema_version'
    
    id = Column(Integer, primary_key=True)
    version = Column(String(20), nullable=False)
    applied_at = Column(DateTime, default=datetime.now)
    description = Column(String(200))

# ==========================================
# ENVIRONMENT CONFIGURATIONS
# ==========================================

class Environment(str, Enum):
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"

@dataclass
class DatabaseConfig(ABC):
    """Configuración base abstracta de base de datos"""
    echo: bool = False
    pool_pre_ping: bool = True
    pool_recycle: int = 3600
    environment: Environment = Environment.DEVELOPMENT
    
    @abstractmethod
    def get_database_url(self) -> str:
        """Obtener URL de base de datos"""
        pass
    
    @abstractmethod
    def get_engine_config(self) -> Dict[str, Any]:
        """Obtener configuración específica del engine"""
        pass

@dataclass
class DevelopmentConfig(DatabaseConfig):
    """Configuración para desarrollo"""
    echo: bool = True
    environment: Environment = Environment.DEVELOPMENT
    
    def get_database_url(self) -> str:
        url = os.getenv("DATABASE_URL")
        if url:
            return url
            
        base_path = Path(__file__).parent.parent.parent
        db_path = base_path / "data" / "dev_translation_system.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        return f"sqlite:///{db_path}"
    
    def get_engine_config(self) -> Dict[str, Any]:
        return {
            "echo": self.echo,
            "poolclass": StaticPool,
            "connect_args": {
                "check_same_thread": False,
                "timeout": 20,
                "isolation_level": None
            }
        }

@dataclass 
class TestingConfig(DatabaseConfig):
    """Configuración para testing"""
    echo: bool = False
    environment: Environment = Environment.TESTING
    
    def get_database_url(self) -> str:
        # Tests usan base de datos en memoria
        return "sqlite:///:memory:"
    
    def get_engine_config(self) -> Dict[str, Any]:
        return {
            "echo": False,
            "poolclass": StaticPool,
            "connect_args": {
                "check_same_thread": False,
                "isolation_level": None
            }
        }

@dataclass
class StagingConfig(DatabaseConfig):
    """Configuración para staging"""
    echo: bool = False
    pool_pre_ping: bool = True
    environment: Environment = Environment.STAGING
    
    def get_database_url(self) -> str:
        url = os.getenv("DATABASE_URL")
        if not url:
            raise ValueError("DATABASE_URL environment variable required for staging")
        return url
    
    def get_engine_config(self) -> Dict[str, Any]:
        if self.get_database_url().startswith("sqlite"):
            return {
                "echo": self.echo,
                "poolclass": StaticPool,
                "connect_args": {"check_same_thread": False, "timeout": 30}
            }
        else:
            return {
                "echo": self.echo,
                "pool_pre_ping": self.pool_pre_ping,
                "pool_recycle": self.pool_recycle,
                "pool_size": 10,
                "max_overflow": 20
            }

@dataclass
class ProductionConfig(DatabaseConfig):
    """Configuración para producción"""
    echo: bool = False
    pool_pre_ping: bool = True
    pool_recycle: int = 1800  # 30 minutes
    environment: Environment = Environment.PRODUCTION
    
    def get_database_url(self) -> str:
        url = os.getenv("DATABASE_URL")
        if not url:
            raise ValueError("DATABASE_URL environment variable required for production")
        return url
    
    def get_engine_config(self) -> Dict[str, Any]:
        if self.get_database_url().startswith("sqlite"):
            # SQLite en producción (solo para casos específicos)
            base_path = Path(__file__).parent.parent.parent
            db_path = base_path / "data" / "prod_translation_system.db"
            db_path.parent.mkdir(parents=True, exist_ok=True)
            
            return {
                "echo": False,
                "poolclass": StaticPool,
                "connect_args": {"check_same_thread": False, "timeout": 60}
            }
        else:
            # PostgreSQL/MySQL en producción
            return {
                "echo": False,
                "pool_pre_ping": True,
                "pool_recycle": self.pool_recycle,
                "pool_size": 20,
                "max_overflow": 30,
                "pool_timeout": 30
            }

def get_config_for_environment(env: Optional[str] = None) -> DatabaseConfig:
    """Factory para obtener configuración según entorno"""
    if env is None:
        env = os.getenv("ENVIRONMENT", "development").lower()
    
    config_map = {
        "development": DevelopmentConfig,
        "dev": DevelopmentConfig,
        "testing": TestingConfig,
        "test": TestingConfig,
        "staging": StagingConfig,
        "stage": StagingConfig,
        "production": ProductionConfig,
        "prod": ProductionConfig
    }
    
    config_class = config_map.get(env, DevelopmentConfig)
    return config_class()

# ==========================================
# STRUCTURED LOGGING SETUP
# ==========================================

def setup_structured_logging():
    """Configurar logging estructurado para la base de datos"""
    
    # Configurar structlog
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="ISO"),
            structlog.processors.add_log_level,
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.WriteLoggerFactory(),
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        cache_logger_on_first_use=True,
    )
    
    # Configurar logging de SQLAlchemy
    sqlalchemy_logger = logging.getLogger('sqlalchemy.engine')
    sqlalchemy_logger.setLevel(logging.WARNING)
    sqlalchemy_logger.propagate = False
    
    if not sqlalchemy_logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        sqlalchemy_logger.addHandler(handler)

# ==========================================
# DATABASE MANAGER - ENTERPRISE
# ==========================================

class DatabaseManager:
    """
    Gestor empresarial de base de datos con configuración por entorno,
    control de versiones y logging estructurado
    """
    
    def __init__(self, config: Optional[DatabaseConfig] = None):
        self.config = config or get_config_for_environment()
        self.engine: Optional[Engine] = None
        self.SessionLocal: Optional[sessionmaker] = None
        self._schema_validated = False
        
        # Setup structured logging
        self.logger = structlog.get_logger("database").bind(
            component="database_manager",
            environment=self.config.environment.value
        )
        
        self._initialize_engine()
    
    def _initialize_engine(self):
        """Inicializar engine de SQLAlchemy con configuración por entorno"""
        try:
            database_url = self.config.get_database_url()
            engine_config = self.config.get_engine_config()
            
            self.engine = create_engine(database_url, **engine_config)
            
            # Configurar SQLite si es necesario
            if database_url.startswith("sqlite"):
                self._configure_sqlite()
            
            # Crear sessionmaker
            self.SessionLocal = sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=self.engine
            )
            
            self.logger.info(
                "Database engine initialized",
                database_type=database_url.split("://")[0],
                environment=self.config.environment.value,
                url_safe=self._get_safe_url()
            )
            
        except Exception as e:
            self.logger.error(
                "Error initializing database engine",
                error=str(e),
                environment=self.config.environment.value
            )
            raise
    
    def _configure_sqlite(self):
        """Configurar SQLite para optimización"""
        @event.listens_for(self.engine, "connect")
        def set_sqlite_pragma(dbapi_connection, connection_record):
            """Configurar PRAGMA settings para SQLite"""
            cursor = dbapi_connection.cursor()
            
            try:
                # Performance optimizations
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.execute("PRAGMA synchronous=NORMAL")
                cursor.execute("PRAGMA cache_size=10000")
                cursor.execute("PRAGMA temp_store=memory")
                cursor.execute("PRAGMA mmap_size=268435456")
                
                # Foreign key enforcement
                cursor.execute("PRAGMA foreign_keys=ON")
                
                # Increase timeouts
                cursor.execute("PRAGMA busy_timeout=30000")
                
                self.logger.debug("SQLite PRAGMA settings applied")
                
            except Exception as e:
                self.logger.warning("Error applying SQLite PRAGMA", error=str(e))
            finally:
                cursor.close()
    
    def _get_safe_url(self) -> str:
        """Obtener URL de base de datos sin credenciales para logging"""
        url = self.config.get_database_url()
        if "@" in url:
            parts = url.split("@")
            if len(parts) > 1:
                protocol_user = parts[0].split("://")
                if len(protocol_user) > 1:
                    return f"{protocol_user[0]}://***@{parts[1]}"
        return url.replace(str(Path.home()), "~")  # Hide home path
    
    def validate_schema_version(self) -> bool:
        """Validar versión de esquema de base de datos"""
        if self._schema_validated:
            return True
            
        try:
            # Crear tabla de schema_version si no existe
            SchemaVersionBase.metadata.create_all(bind=self.engine)
            
            with self.get_db_session() as session:
                # Verificar versión actual
                current_record = session.query(DatabaseSchemaVersion).order_by(
                    DatabaseSchemaVersion.id.desc()
                ).first()
                
                if not current_record:
                    # Primera instalación - crear registro inicial
                    new_record = DatabaseSchemaVersion(
                        version=SchemaVersion.CURRENT_VERSION,
                        description="Initial schema version"
                    )
                    session.add(new_record)
                    session.commit()
                    
                    self.logger.info(
                        "Schema version initialized",
                        version=SchemaVersion.CURRENT_VERSION
                    )
                    self._schema_validated = True
                    return True
                
                # Verificar compatibilidad
                if current_record.version not in SchemaVersion.COMPATIBLE_VERSIONS:
                    self.logger.error(
                        "Incompatible schema version",
                        current_version=current_record.version,
                        required_versions=SchemaVersion.COMPATIBLE_VERSIONS
                    )
                    raise ValueError(
                        f"Schema version {current_record.version} is incompatible. "
                        f"Compatible versions: {SchemaVersion.COMPATIBLE_VERSIONS}"
                    )
                
                # Actualizar a versión actual si es diferente
                if current_record.version != SchemaVersion.CURRENT_VERSION:
                    new_record = DatabaseSchemaVersion(
                        version=SchemaVersion.CURRENT_VERSION,
                        description=f"Updated from {current_record.version}"
                    )
                    session.add(new_record)
                    session.commit()
                    
                    self.logger.info(
                        "Schema version updated",
                        from_version=current_record.version,
                        to_version=SchemaVersion.CURRENT_VERSION
                    )
                
                self._schema_validated = True
                return True
                
        except Exception as e:
            self.logger.error("Schema validation failed", error=str(e))
            raise
    
    def create_all_tables(self):
        """Crear todas las tablas en la base de datos"""
        try:
            # Validar esquema primero
            self.validate_schema_version()
            
            # Importar modelos dinámicamente para evitar dependencias circulares
            from .models import Base
            
            Base.metadata.create_all(bind=self.engine)
            
            self.logger.info(
                "All database tables created successfully",
                environment=self.config.environment.value
            )
        except Exception as e:
            self.logger.error("Error creating database tables", error=str(e))
            raise
    
    def drop_all_tables(self):
        """Eliminar todas las tablas (¡CUIDADO!)"""
        try:
            if self.config.environment == Environment.PRODUCTION:
                raise ValueError("Cannot drop tables in production environment")
            
            from .models import Base
            
            Base.metadata.drop_all(bind=self.engine)
            SchemaVersionBase.metadata.drop_all(bind=self.engine)
            
            self.logger.warning(
                "All database tables dropped",
                environment=self.config.environment.value
            )
        except Exception as e:
            self.logger.error("Error dropping database tables", error=str(e))
            raise
    
    def get_session(self) -> Session:
        """Obtener nueva sesión de base de datos"""
        if not self.SessionLocal:
            raise RuntimeError("Database not initialized")
        
        # Validar esquema en primera sesión
        if not self._schema_validated:
            self.validate_schema_version()
            
        return self.SessionLocal()
    
    @contextmanager
    def get_db_session(self) -> Generator[Session, None, None]:
        """Context manager para sesiones de base de datos"""
        session = self.get_session()
        start_time = time.time()
        
        try:
            yield session
            duration_ms = (time.time() - start_time) * 1000
            self.logger.debug("Database session completed", duration_ms=duration_ms)
        except Exception as e:
            session.rollback()
            duration_ms = (time.time() - start_time) * 1000
            self.logger.error(
                "Database session error",
                error=str(e),
                duration_ms=duration_ms
            )
            raise
        finally:
            session.close()
    
    def execute_raw_sql(self, sql: str, params: Optional[dict] = None) -> any:
        """Ejecutar SQL crudo con logging estructurado"""
        start_time = time.time()
        
        try:
            with self.engine.begin() as conn:
                result = conn.execute(text(sql), params or {})
                duration_ms = (time.time() - start_time) * 1000
                
                self.logger.info(
                    "Raw SQL executed",
                    sql_preview=sql[:100] + "..." if len(sql) > 100 else sql,
                    duration_ms=duration_ms,
                    param_count=len(params) if params else 0
                )
                
                return result
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.logger.error(
                "Raw SQL execution failed",
                sql_preview=sql[:100] + "..." if len(sql) > 100 else sql,
                error=str(e),
                duration_ms=duration_ms
            )
            raise
    
    def check_connection(self) -> bool:
        """Verificar conexión a la base de datos con retry automático"""
        def _check():
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        
        try:
            return execute_with_retry(_check, max_retries=3, delay=1.0)
        except Exception as e:
            self.logger.error("Database connection check failed", error=str(e))
            return False
    
    def get_database_info(self) -> dict:
        """Obtener información de la base de datos con logging estructurado"""
        try:
            with self.engine.connect() as conn:
                database_url = self.config.get_database_url()
                
                if database_url.startswith("sqlite"):
                    # SQLite specific queries
                    result = conn.execute(text("PRAGMA database_list"))
                    db_info = result.fetchall()
                    
                    result = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))
                    tables = [row[0] for row in result.fetchall()]
                    
                    # Get database file size
                    db_path = database_url.replace("sqlite:///", "")
                    file_size = 0
                    if db_path != ":memory:" and os.path.exists(db_path):
                        file_size = os.path.getsize(db_path)
                    
                    info = {
                        "type": "SQLite",
                        "path": db_path,
                        "size_bytes": file_size,
                        "size_mb": round(file_size / 1024 / 1024, 2),
                        "tables_count": len(tables),
                        "tables": tables,
                        "environment": self.config.environment.value,
                        "schema_version": self._get_current_schema_version()
                    }
                else:
                    # Generic database info
                    result = conn.execute(text("SELECT version()"))
                    version = result.scalar() if result else "Unknown"
                    
                    info = {
                        "type": "Generic",
                        "version": version,
                        "url": self._get_safe_url(),
                        "environment": self.config.environment.value,
                        "schema_version": self._get_current_schema_version()
                    }
                
                self.logger.info("Database info retrieved", **info)
                return info
                
        except Exception as e:
            self.logger.error("Error getting database info", error=str(e))
            return {"error": str(e)}
    
    def _get_current_schema_version(self) -> str:
        """Obtener versión actual del esquema"""
        try:
            with self.get_db_session() as session:
                record = session.query(DatabaseSchemaVersion).order_by(
                    DatabaseSchemaVersion.id.desc()
                ).first()
                return record.version if record else "unknown"
        except:
            return "unknown"
    
    def optimize_database(self):
        """Optimizar base de datos con logging estructurado"""
        start_time = time.time()
        
        try:
            if self.config.get_database_url().startswith("sqlite"):
                with self.engine.connect() as conn:
                    conn.execute(text("ANALYZE"))
                    conn.execute(text("PRAGMA optimize"))
                
                duration_ms = (time.time() - start_time) * 1000
                self.logger.info(
                    "Database optimization completed",
                    database_type="sqlite",
                    duration_ms=duration_ms
                )
            else:
                self.logger.info(
                    "Database optimization not implemented for non-SQLite databases",
                    database_type="non-sqlite"
                )
                
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.logger.error(
                "Error optimizing database",
                error=str(e),
                duration_ms=duration_ms
            )
            raise
    
    def backup_database(self, backup_path: str) -> bool:
        """Crear backup de la base de datos con logging estructurado"""
        start_time = time.time()
        
        try:
            if self.config.get_database_url().startswith("sqlite"):
                source_path = self.config.get_database_url().replace("sqlite:///", "")
                
                if source_path == ":memory:":
                    self.logger.warning("Cannot backup in-memory database")
                    return False
                
                # Ensure backup directory exists
                Path(backup_path).parent.mkdir(parents=True, exist_ok=True)
                
                # Create backup using sqlite3 backup API
                source_conn = sqlite3.connect(source_path)
                backup_conn = sqlite3.connect(backup_path)
                
                with backup_conn:
                    source_conn.backup(backup_conn)
                
                source_conn.close()
                backup_conn.close()
                
                # Get backup file size
                backup_size = os.path.getsize(backup_path)
                duration_ms = (time.time() - start_time) * 1000
                
                self.logger.info(
                    "Database backup created",
                    backup_path=backup_path,
                    backup_size_mb=round(backup_size / 1024 / 1024, 2),
                    duration_ms=duration_ms
                )
                return True
            else:
                self.logger.warning(
                    "Database backup not implemented for non-SQLite databases"
                )
                return False
                
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.logger.error(
                "Error creating database backup",
                error=str(e),
                backup_path=backup_path,
                duration_ms=duration_ms
            )
            return False
    
    def close(self):
        """Cerrar engine de base de datos"""
        if self.engine:
            self.engine.dispose()
            self.logger.info("Database engine closed")

# ==========================================
# DEPENDENCY INJECTION CONTAINER
# ==========================================

class DatabaseContainer:
    """Container para dependency injection de servicios de base de datos"""
    
    def __init__(self, config: Optional[DatabaseConfig] = None):
        self._config = config
        self._manager: Optional[DatabaseManager] = None
    
    def get_database_manager(self) -> DatabaseManager:
        """Obtener instancia del gestor de base de datos (singleton)"""
        if self._manager is None:
            self._manager = DatabaseManager(self._config)
        return self._manager
    
    def get_database_config(self) -> DatabaseConfig:
        """Obtener configuración de base de datos"""
        if self._config is None:
            self._config = get_config_for_environment()
        return self._config
    
    def reset(self):
        """Resetear container (útil para tests)"""
        if self._manager:
            self._manager.close()
        self._manager = None
        self._config = None

# ==========================================
# GLOBAL CONTAINER (para compatibilidad)
# ==========================================

# Container global para compatibilidad con código existente
_global_container = DatabaseContainer()

def get_database_manager() -> DatabaseManager:
    """Obtener gestor de base de datos (dependency injection friendly)"""
    return _global_container.get_database_manager()

def set_database_config(config: DatabaseConfig):
    """Establecer configuración de base de datos"""
    _global_container._config = config
    _global_container.reset()

# Instancia global para compatibilidad (DEPRECATED - usar dependency injection)
db_manager = get_database_manager()

# ==========================================
# DEPENDENCY FUNCTIONS FOR FASTAPI
# ==========================================

def get_db() -> Generator[Session, None, None]:
    """
    Dependency function para FastAPI
    Proporciona sesión de base de datos que se cierra automáticamente
    """
    manager = get_database_manager()
    session = manager.get_session()
    
    try:
        yield session
    except Exception as e:
        session.rollback()
        logger.error("Database session error in dependency", error=str(e))
        raise
    finally:
        session.close()

def get_db_manager() -> DatabaseManager:
    """Dependency function para obtener el gestor de base de datos"""
    return get_database_manager()

def get_db_config() -> DatabaseConfig:
    """Dependency function para obtener configuración de base de datos"""
    return _global_container.get_database_config()

# ==========================================
# HEALTH CHECK FUNCTIONS
# ==========================================

def check_database_health() -> dict:
    """
    Verificar salud de la base de datos con logging estructurado
    Usado por health check endpoints
    """
    start_time = time.time()
    manager = get_database_manager()
    
    try:
        is_connected = manager.check_connection()
        response_time_ms = (time.time() - start_time) * 1000
        
        if is_connected:
            db_info = manager.get_database_info()
            
            health_data = {
                "status": "healthy",
                "response_time_ms": round(response_time_ms, 2),
                "environment": manager.config.environment.value,
                "database_info": db_info,
                "schema_validated": manager._schema_validated
            }
            
            logger.info("Database health check passed", **health_data)
            return health_data
        else:
            health_data = {
                "status": "unhealthy", 
                "response_time_ms": round(response_time_ms, 2),
                "environment": manager.config.environment.value,
                "error": "Cannot connect to database"
            }
            
            logger.error("Database health check failed", **health_data)
            return health_data
            
    except Exception as e:
        response_time_ms = (time.time() - start_time) * 1000
        health_data = {
            "status": "unhealthy",
            "response_time_ms": round(response_time_ms, 2),
            "error": str(e)
        }
        
        logger.error("Database health check exception", **health_data)
        return health_data

# ==========================================
# UTILITY FUNCTIONS
# ==========================================

def measure_query_performance(func):
    """Decorator para medir performance de queries con logging estructurado"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        operation_logger = structlog.get_logger("database.query").bind(
            function=func.__name__,
            module=func.__module__
        )
        
        try:
            result = func(*args, **kwargs)
            execution_time_ms = (time.time() - start_time) * 1000
            
            operation_logger.info(
                "Query executed successfully",
                duration_ms=execution_time_ms,
                success=True
            )
            
            return result
        except Exception as e:
            execution_time_ms = (time.time() - start_time) * 1000
            
            operation_logger.error(
                "Query execution failed",
                duration_ms=execution_time_ms,
                error=str(e),
                success=False
            )
            raise
    return wrapper

def execute_with_retry(func, max_retries: int = 3, delay: float = 1.0):
    """Ejecutar función con reintentos en caso de error de base de datos"""
    retry_logger = structlog.get_logger("database.retry")
    
    for attempt in range(max_retries):
        try:
            result = func()
            if attempt > 0:
                retry_logger.info(
                    "Function succeeded after retry",
                    attempt=attempt + 1,
                    max_retries=max_retries
                )
            return result
        except Exception as e:
            if attempt == max_retries - 1:
                retry_logger.error(
                    "Function failed after all retries",
                    attempts=max_retries,
                    final_error=str(e)
                )
                raise
            else:
                retry_logger.warning(
                    "Function failed, retrying",
                    attempt=attempt + 1,
                    max_retries=max_retries,
                    error=str(e),
                    retry_delay=delay
                )
                time.sleep(delay)
                delay *= 2  # Exponential backoff

# ==========================================
# MIGRATION SUPPORT (sin dependencia circular)
# ==========================================

def run_migrations_safe():
    """Ejecutar migraciones de forma segura sin dependencias circulares"""
    try:
        # Import dinámico para evitar dependencias circulares
        import importlib.util
        
        migrations_path = Path(__file__).parent / "migrations.py"
        if migrations_path.exists():
            spec = importlib.util.spec_from_file_location("migrations", migrations_path)
            migrations_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(migrations_module)
            
            if hasattr(migrations_module, 'run_migrations'):
                migrations_module.run_migrations()
                logger.info("Migrations executed successfully")
            else:
                logger.warning("No run_migrations function found in migrations module")
        else:
            logger.info("No migrations module found, skipping")
            
    except Exception as e:
        logger.error("Error running migrations", error=str(e))
        # No re-raise para evitar fallos en inicialización

def create_initial_data_safe():
    """Crear datos iniciales de forma segura"""
    try:
        import importlib.util
        
        migrations_path = Path(__file__).parent / "migrations.py"
        if migrations_path.exists():
            spec = importlib.util.spec_from_file_location("migrations", migrations_path)
            migrations_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(migrations_module)
            
            if hasattr(migrations_module, 'create_initial_data'):
                migrations_module.create_initial_data()
                logger.info("Initial data created successfully")
            else:
                logger.info("No create_initial_data function found")
        else:
            logger.info("No migrations module found for initial data")
            
    except Exception as e:
        logger.error("Error creating initial data", error=str(e))

# ==========================================
# INITIALIZATION FUNCTIONS
# ==========================================

def initialize_database(create_tables: bool = True, run_migrations: bool = True):
    """
    Inicializar completamente la base de datos con logging estructurado
    """
    init_logger = structlog.get_logger("database.init")
    start_time = time.time()
    
    try:
        init_logger.info("Starting database initialization")
        
        manager = get_database_manager()
        
        # Check connection
        if not manager.check_connection():
            raise RuntimeError("Cannot connect to database")
        
        # Validate schema version
        manager.validate_schema_version()
        
        # Create tables if requested
        if create_tables:
            manager.create_all_tables()
        
        # Run migrations if requested
        if run_migrations:
            run_migrations_safe()
        
        # Create initial data
        create_initial_data_safe()
        
        duration_ms = (time.time() - start_time) * 1000
        init_logger.info(
            "Database initialization completed successfully",
            duration_ms=duration_ms,
            environment=manager.config.environment.value
        )
        
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        init_logger.error(
            "Database initialization failed",
            error=str(e),
            duration_ms=duration_ms
        )
        raise

def reset_database():
    """
    Resetear completamente la base de datos (¡CUIDADO!)
    Solo para desarrollo y testing
    """
    reset_logger = structlog.get_logger("database.reset")
    manager = get_database_manager()
    
    if manager.config.environment == Environment.PRODUCTION:
        raise ValueError("Cannot reset database in production environment")
    
    reset_logger.warning(
        "Resetting database - ALL DATA WILL BE LOST",
        environment=manager.config.environment.value
    )
    
    start_time = time.time()
    
    try:
        # Drop all tables
        manager.drop_all_tables()
        
        # Recreate tables
        manager.create_all_tables()
        
        # Create initial data
        create_initial_data_safe()
        
        duration_ms = (time.time() - start_time) * 1000
        reset_logger.info(
            "Database reset completed",
            duration_ms=duration_ms,
            environment=manager.config.environment.value
        )
        
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        reset_logger.error(
            "Database reset failed",
            error=str(e),
            duration_ms=duration_ms
        )
        raise

# ==========================================
# CONTEXT MANAGERS FOR TRANSACTIONS
# ==========================================

@contextmanager
def db_transaction() -> Generator[Session, None, None]:
    """
    Context manager para transacciones explícitas con logging estructurado
    """
    manager = get_database_manager()
    session = manager.get_session()
    tx_logger = structlog.get_logger("database.transaction")
    start_time = time.time()
    
    try:
        session.begin()
        tx_logger.debug("Transaction started")
        
        yield session
        
        session.commit()
        duration_ms = (time.time() - start_time) * 1000
        tx_logger.info("Transaction committed", duration_ms=duration_ms)
        
    except Exception as e:
        session.rollback()
        duration_ms = (time.time() - start_time) * 1000
        tx_logger.error(
            "Transaction rolled back",
            error=str(e),
            duration_ms=duration_ms
        )
        raise
    finally:
        session.close()

@contextmanager
def db_readonly_session() -> Generator[Session, None, None]:
    """
    Context manager para sesiones de solo lectura con logging estructurado
    """
    manager = get_database_manager()
    session = manager.get_session()
    ro_logger = structlog.get_logger("database.readonly")
    start_time = time.time()
    
    try:
        # Set session to read-only mode (if supported)
        if hasattr(session, 'connection'):
            session.connection().execute(text("BEGIN DEFERRED"))
        
        ro_logger.debug("Read-only session started")
        yield session
        
        duration_ms = (time.time() - start_time) * 1000
        ro_logger.debug("Read-only session completed", duration_ms=duration_ms)
        
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        ro_logger.error(
            "Read-only session error",
            error=str(e),
            duration_ms=duration_ms
        )
        raise
    finally:
        session.close()

# ==========================================
# STARTUP INITIALIZATION
# ==========================================

# Setup structured logging on import
setup_structured_logging()