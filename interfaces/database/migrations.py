# ==========================================
# INTERFACES/DATABASE/MIGRATIONS.PY
# Database Migrations & Initial Data Setup
# Sistema de Traducción Académica v2.2 - ENTERPRISE GRADE
# ==========================================

from sqlalchemy.orm import Session
from sqlalchemy import text, inspect
from typing import List, Dict, Any, Optional, Callable
import logging
import json
from datetime import datetime, date, timedelta
from packaging import version
from enum import Enum
import sqlite3

from .database import db_manager, get_db
from .models import (
    SystemConfig, ErrorPattern, UsageStatistic,
    PatternType, create_indexes
)
from .crud import system_config_crud, error_pattern_crud, usage_statistics_crud

logger = logging.getLogger(__name__)

# ==========================================
# ENTERPRISE CONSTANTS & ENUMS
# ==========================================

class MigrationOperation(Enum):
    """Operations supported by migration system"""
    FORWARD = "forward"
    ROLLBACK = "rollback"
    VALIDATE = "validate"

class InitialPatternDefinitions:
    """Centralized pattern definitions for maintainability"""
    
    PDF_ARTIFACTS = [
        {
            "content": "Page \\d+",
            "regex": r"^Page \d+$",
            "description": "Números de página simples",
            "effectiveness": 0.95
        },
        {
            "content": "Dewey_\\d+_\\d+pp",
            "regex": r"Dewey_\d+_\d+pp",
            "description": "Patrón específico detectado en auditoría",
            "effectiveness": 0.98
        },
        {
            "content": "© \\d{4}.*",
            "regex": r"© \d{4}.*",
            "description": "Avisos de copyright",
            "effectiveness": 0.92
        },
        {
            "content": "Chapter \\d+",
            "regex": r"^Chapter \d+$",
            "description": "Headers de capítulo simples",
            "effectiveness": 0.89
        }
    ]
    
    SEMANTIC_PATTERNS = [
        {
            "content": "low_similarity_academic",
            "description": "Similitud semántica baja en texto académico",
            "effectiveness": 0.85
        },
        {
            "content": "missing_terminology",
            "description": "Terminología académica faltante",
            "effectiveness": 0.78
        }
    ]
    
    FORMAT_PATTERNS = [
        {
            "content": "lost_italics",
            "description": "Itálicas perdidas en traducción",
            "effectiveness": 0.90
        },
        {
            "content": "lost_bold",
            "description": "Negritas perdidas en traducción",
            "effectiveness": 0.88
        }
    ]
    
    FOOTNOTE_PATTERNS = [
        {
            "content": "disconnected_footnote",
            "description": "Notas al pie desconectadas",
            "effectiveness": 0.82
        },
        {
            "content": "missing_footnote_link",
            "description": "Enlaces de notas al pie faltantes",
            "effectiveness": 0.87
        }
    ]
    
    TERMINOLOGY_PATTERNS = [
        {
            "content": "inconsistent_dasein",
            "regex": r"\bdasein\b|\bDasein\b",
            "description": "Inconsistencia en traducción de 'Dasein'",
            "effectiveness": 0.95
        },
        {
            "content": "inconsistent_sein",
            "regex": r"\bsein\b|\bSein\b",
            "description": "Inconsistencia en traducción de 'Sein'",
            "effectiveness": 0.93
        }
    ]

# ==========================================
# ENTERPRISE MIGRATION WRAPPER
# ==========================================

class MigrationCRUDWrapper:
    """
    Independent wrapper for CRUD operations in migrations
    Reduces coupling with main CRUD modules
    """
    
    @staticmethod
    def safe_upsert_config(db: Session, config_key: str, config_value: str,
                          config_type: str = "string", description: str = ""):
        """Safe upsert for system configuration"""
        try:
            existing = db.query(SystemConfig).filter(
                SystemConfig.config_key == config_key
            ).first()
            
            if existing:
                existing.config_value = config_value
                existing.config_type = config_type
                existing.description = description
                existing.updated_at = datetime.now()
            else:
                new_config = SystemConfig(
                    config_key=config_key,
                    config_value=config_value,
                    config_type=config_type,
                    description=description
                )
                db.add(new_config)
            
            db.flush()
            return True
            
        except Exception as e:
            logger.error(f"Failed to upsert config {config_key}: {e}")
            return False
    
    @staticmethod
    def safe_create_error_pattern(db: Session, pattern_data: Dict[str, Any]) -> bool:
        """Safe creation of error patterns"""
        try:
            existing = db.query(ErrorPattern).filter(
                ErrorPattern.pattern_content == pattern_data["pattern_content"]
            ).first()
            
            if existing:
                logger.debug(f"Pattern already exists: {pattern_data['pattern_content']}")
                return True
            
            new_pattern = ErrorPattern(**pattern_data)
            db.add(new_pattern)
            db.flush()
            return True
            
        except Exception as e:
            logger.error(f"Failed to create pattern {pattern_data.get('pattern_content', 'unknown')}: {e}")
            return False

# ==========================================
# ENTERPRISE MIGRATION SYSTEM
# ==========================================

class MigrationManager:
    """
    Enterprise-grade migration manager with rollback support
    and semantic versioning
    """
    
    def __init__(self):
        self.current_version = "2.2.0"
        self.migrations = {
            "2.2.0": {
                "forward": self._migration_2_2_0_forward,
                "rollback": self._migration_2_2_0_rollback
            },
            "2.2.1": {
                "forward": self._migration_2_2_1_forward,
                "rollback": self._migration_2_2_1_rollback
            },
        }
        self.crud_wrapper = MigrationCRUDWrapper()
    
    def get_database_version(self, db: Session) -> str:
        """Obtener versión actual de la base de datos"""
        try:
            config = db.query(SystemConfig).filter(
                SystemConfig.config_key == "database_version"
            ).first()
            return config.config_value if config else "0.0.0"
        except Exception:
            return "0.0.0"
    
    def set_database_version(self, db: Session, target_version: str):
        """Establecer versión de la base de datos"""
        self.crud_wrapper.safe_upsert_config(
            db,
            config_key="database_version",
            config_value=target_version,
            config_type="string",
            description="Current database schema version"
        )
    
    def needs_migration(self, db: Session) -> bool:
        """Verificar si se necesita migración usando comparación semántica"""
        current_db_version = self.get_database_version(db)
        try:
            return version.parse(current_db_version) < version.parse(self.current_version)
        except Exception as e:
            logger.warning(f"Version parsing failed, assuming migration needed: {e}")
            return True
    
    def run_migrations(self, db: Session, operation: MigrationOperation = MigrationOperation.FORWARD):
        """Ejecutar migraciones con soporte para rollback"""
        current_db_version = self.get_database_version(db)
        logger.info(f"Current database version: {current_db_version}")
        logger.info(f"Target version: {self.current_version}")
        logger.info(f"Operation: {operation.value}")
        
        if operation == MigrationOperation.FORWARD:
            if version.parse(current_db_version) >= version.parse(self.current_version):
                logger.info("Database is up to date")
                return
            
            self._run_forward_migrations(db, current_db_version)
        
        elif operation == MigrationOperation.ROLLBACK:
            self._run_rollback_migrations(db, current_db_version)
    
    def _run_forward_migrations(self, db: Session, current_db_version: str):
        """Ejecutar migraciones hacia adelante"""
        for version_str, migration_funcs in self.migrations.items():
            if self._should_run_migration(current_db_version, version_str):
                logger.info(f"Running forward migration to version {version_str}")
                try:
                    migration_funcs["forward"](db)
                    self.set_database_version(db, version_str)
                    db.commit()
                    logger.info(f"Migration to {version_str} completed successfully")
                except Exception as e:
                    db.rollback()
                    logger.error(f"Migration to {version_str} failed: {e}")
                    raise
    
    def _run_rollback_migrations(self, db: Session, current_db_version: str):
        """Ejecutar rollback de migraciones"""
        # Get sorted versions in reverse order for rollback
        sorted_versions = sorted(
            self.migrations.keys(),
            key=lambda v: version.parse(v),
            reverse=True
        )
        
        for version_str in sorted_versions:
            if version.parse(version_str) <= version.parse(current_db_version):
                logger.info(f"Running rollback for version {version_str}")
                try:
                    self.migrations[version_str]["rollback"](db)
                    # Set version to previous version or 0.0.0
                    prev_version = self._get_previous_version(version_str)
                    self.set_database_version(db, prev_version)
                    db.commit()
                    logger.info(f"Rollback of {version_str} completed successfully")
                except Exception as e:
                    db.rollback()
                    logger.error(f"Rollback of {version_str} failed: {e}")
                    raise
    
    def _should_run_migration(self, current_version: str, target_version: str) -> bool:
        """Determinar si se debe ejecutar una migración específica usando comparación semántica"""
        try:
            return version.parse(target_version) > version.parse(current_version)
        except Exception as e:
            logger.warning(f"Version comparison failed for {current_version} -> {target_version}: {e}")
            # Fallback to string comparison
            return target_version > current_version
    
    def _get_previous_version(self, current_version: str) -> str:
        """Obtener versión anterior para rollback"""
        sorted_versions = sorted(
            self.migrations.keys(),
            key=lambda v: version.parse(v)
        )
        
        try:
            current_index = sorted_versions.index(current_version)
            if current_index > 0:
                return sorted_versions[current_index - 1]
        except (ValueError, IndexError):
            pass
        
        return "0.0.0"
    
    def _safe_create_indexes(self, db: Session) -> int:
        """Crear índices con validación previa"""
        created_count = 0
        
        try:
            # Get existing indexes
            inspector = inspect(db.bind)
            existing_indexes = set()
            
            for table_name in inspector.get_table_names():
                table_indexes = inspector.get_indexes(table_name)
                for idx in table_indexes:
                    existing_indexes.add(idx['name'])
            
            # Create new indexes if they don't exist
            additional_indexes = create_indexes()
            for index_sql in additional_indexes:
                # Extract index name from SQL (basic parsing)
                index_name = self._extract_index_name(index_sql)
                
                if index_name and index_name not in existing_indexes:
                    try:
                        db.execute(text(index_sql))
                        created_count += 1
                        logger.debug(f"Created index: {index_name}")
                    except Exception as e:
                        logger.warning(f"Failed to create index {index_name}: {e}")
                else:
                    logger.debug(f"Index {index_name} already exists, skipping")
            
            return created_count
            
        except Exception as e:
            logger.error(f"Index creation validation failed: {e}")
            return 0
    
    def _extract_index_name(self, index_sql: str) -> Optional[str]:
        """Extract index name from CREATE INDEX SQL"""
        try:
            # Simple regex to extract index name from "CREATE INDEX idx_name ON..."
            import re
            match = re.search(r'CREATE\s+INDEX\s+(\w+)\s+ON', index_sql, re.IGNORECASE)
            return match.group(1) if match else None
        except Exception:
            return None
    
    # ==========================================
    # MIGRATION IMPLEMENTATIONS
    # ==========================================
    
    def _migration_2_2_0_forward(self, db: Session):
        """Migración inicial para versión 2.2.0"""
        logger.info("Running initial migration 2.2.0 (forward)")
        
        # Crear índices adicionales con validación
        created_indexes = self._safe_create_indexes(db)
        logger.info(f"Created {created_indexes} new indexes")
        
        # Configuraciones iniciales del sistema
        self._create_initial_system_config(db)
        
        # Patrones de error iniciales
        self._create_initial_error_patterns(db)
        
        logger.info("Migration 2.2.0 forward completed")
    
    def _migration_2_2_0_rollback(self, db: Session):
        """Rollback para migración 2.2.0"""
        logger.info("Running rollback for migration 2.2.0")
        
        # Remove system configs (optional - could be kept)
        try:
            db.query(SystemConfig).filter(
                SystemConfig.config_key.in_([
                    "system_name", "system_version", "max_concurrent_books",
                    "default_quality_threshold", "enable_automatic_retry"
                ])
            ).delete(synchronize_session=False)
            
            # Remove initial error patterns
            db.query(ErrorPattern).filter(
                ErrorPattern.created_by.in_(["system_init", "audit_finding", "philosophy_expert"])
            ).delete(synchronize_session=False)
            
            logger.info("Rollback 2.2.0 completed")
            
        except Exception as e:
            logger.warning(f"Partial rollback 2.2.0: {e}")
    
    def _migration_2_2_1_forward(self, db: Session):
        """Migración futura 2.2.1 (placeholder)"""
        logger.info("Running migration 2.2.1 (forward)")
        # Future schema changes would go here
        
        # Example: Add new system configs
        self.crud_wrapper.safe_upsert_config(
            db,
            config_key="enhanced_validation",
            config_value="true",
            config_type="boolean",
            description="Enhanced validation features enabled"
        )
        
        logger.info("Migration 2.2.1 forward completed")
    
    def _migration_2_2_1_rollback(self, db: Session):
        """Rollback para migración 2.2.1"""
        logger.info("Running rollback for migration 2.2.1")
        
        try:
            db.query(SystemConfig).filter(
                SystemConfig.config_key == "enhanced_validation"
            ).delete()
            
            logger.info("Rollback 2.2.1 completed")
            
        except Exception as e:
            logger.warning(f"Partial rollback 2.2.1: {e}")
    
    def _create_initial_system_config(self, db: Session):
        """Crear configuraciones iniciales del sistema"""
        initial_configs = [
            {
                "config_key": "system_name",
                "config_value": "Sistema de Traducción Académica",
                "config_type": "string",
                "description": "Nombre del sistema"
            },
            {
                "config_key": "system_version",
                "config_value": "2.2.0",
                "config_type": "string", 
                "description": "Versión actual del sistema"
            },
            {
                "config_key": "max_concurrent_books",
                "config_value": "5",
                "config_type": "integer",
                "description": "Máximo número de libros en procesamiento simultáneo"
            },
            {
                "config_key": "default_quality_threshold",
                "config_value": "0.85",
                "config_type": "float",
                "description": "Umbral mínimo de calidad semántica"
            },
            {
                "config_key": "enable_automatic_retry",
                "config_value": "true",
                "config_type": "boolean",
                "description": "Habilitar reintentos automáticos en errores"
            },
            {
                "config_key": "max_retry_attempts",
                "config_value": "3",
                "config_type": "integer",
                "description": "Número máximo de reintentos automáticos"
            },
            {
                "config_key": "api_rate_limits",
                "config_value": json.dumps({
                    "deepl": {"requests_per_minute": 100},
                    "claude": {"requests_per_minute": 60},
                    "abbyy": {"requests_per_minute": 50}
                }),
                "config_type": "json",
                "description": "Límites de tasa para APIs externas"
            },
            {
                "config_key": "supported_languages",
                "config_value": json.dumps({
                    "source": ["de", "en", "fr", "it", "nl"],
                    "target": ["es"]
                }),
                "config_type": "json",
                "description": "Idiomas soportados por el sistema"
            },
            {
                "config_key": "email_notifications",
                "config_value": "false",
                "config_type": "boolean",
                "description": "Habilitar notificaciones por email"
            },
            {
                "config_key": "backup_enabled",
                "config_value": "true",
                "config_type": "boolean",
                "description": "Habilitar backup automático de base de datos"
            }
        ]
        
        success_count = 0
        for config in initial_configs:
            if self.crud_wrapper.safe_upsert_config(
                db, 
                config["config_key"], 
                config["config_value"],
                config["config_type"],
                config["description"]
            ):
                success_count += 1
                logger.debug(f"Created system config: {config['config_key']}")
        
        logger.info(f"Created {success_count}/{len(initial_configs)} system configurations")
    
    def _create_initial_error_patterns(self, db: Session):
        """Crear patrones de error iniciales usando definiciones centralizadas"""
        patterns_created = 0
        
        # PDF Artifacts
        for pattern_def in InitialPatternDefinitions.PDF_ARTIFACTS:
            pattern_data = {
                "pattern_type": PatternType.PDF_ARTIFACT.value,
                "pattern_content": pattern_def["content"],
                "pattern_regex": pattern_def.get("regex"),
                "description": pattern_def["description"],
                "effectiveness_score": pattern_def["effectiveness"],
                "created_by": "system_init"
            }
            
            if self.crud_wrapper.safe_create_error_pattern(db, pattern_data):
                patterns_created += 1
        
        # Semantic patterns
        for pattern_def in InitialPatternDefinitions.SEMANTIC_PATTERNS:
            pattern_data = {
                "pattern_type": PatternType.SEMANTIC.value,
                "pattern_content": pattern_def["content"],
                "pattern_regex": pattern_def.get("regex"),
                "description": pattern_def["description"],
                "effectiveness_score": pattern_def["effectiveness"],
                "created_by": "system_init"
            }
            
            if self.crud_wrapper.safe_create_error_pattern(db, pattern_data):
                patterns_created += 1
        
        # Format preservation patterns
        for pattern_def in InitialPatternDefinitions.FORMAT_PATTERNS:
            pattern_data = {
                "pattern_type": PatternType.FORMAT.value,
                "pattern_content": pattern_def["content"],
                "pattern_regex": pattern_def.get("regex"),
                "description": pattern_def["description"],
                "effectiveness_score": pattern_def["effectiveness"],
                "created_by": "system_init"
            }
            
            if self.crud_wrapper.safe_create_error_pattern(db, pattern_data):
                patterns_created += 1
        
        # Footnote patterns
        for pattern_def in InitialPatternDefinitions.FOOTNOTE_PATTERNS:
            pattern_data = {
                "pattern_type": PatternType.FOOTNOTE.value,
                "pattern_content": pattern_def["content"],
                "pattern_regex": pattern_def.get("regex"),
                "description": pattern_def["description"],
                "effectiveness_score": pattern_def["effectiveness"],
                "created_by": "system_init"
            }
            
            if self.crud_wrapper.safe_create_error_pattern(db, pattern_data):
                patterns_created += 1
        
        # Terminology patterns
        for pattern_def in InitialPatternDefinitions.TERMINOLOGY_PATTERNS:
            pattern_data = {
                "pattern_type": PatternType.TERMINOLOGY.value,
                "pattern_content": pattern_def["content"],
                "pattern_regex": pattern_def.get("regex"),
                "description": pattern_def["description"],
                "effectiveness_score": pattern_def["effectiveness"],
                "created_by": "philosophy_expert"
            }
            
            if self.crud_wrapper.safe_create_error_pattern(db, pattern_data):
                patterns_created += 1
        
        logger.info(f"Created {patterns_created} error patterns")

# ==========================================
# ENHANCED INITIAL DATA CREATION
# ==========================================

def create_initial_data(skip_if_exists: bool = True, environment: str = "production"):
    """
    Crear datos iniciales para el sistema con control mejorado
    """
    logger.info(f"Creating initial data for environment: {environment}")
    
    with db_manager.get_db_session() as db:
        try:
            # Crear estadísticas de ejemplo según el entorno
            if environment in ["development", "testing"]:
                _create_sample_usage_statistics(db, skip_if_exists)
            
            # Crear configuraciones de desarrollo si no existen
            if environment == "development":
                _create_development_configs(db, skip_if_exists)
            
            logger.info("Initial data creation completed")
            
        except Exception as e:
            logger.error(f"Error creating initial data: {e}")
            raise

def _create_sample_usage_statistics(db: Session, skip_if_exists: bool = True):
    """Crear estadísticas de muestra con control de duplicados mejorado"""
    if skip_if_exists:
        # Check if any statistics exist
        existing_count = db.query(UsageStatistic).count()
        if existing_count > 0:
            logger.info(f"Skipping sample statistics creation - {existing_count} records exist")
            return
    
    today = date.today()
    created_count = 0
    
    for i in range(30):
        stat_date = today - timedelta(days=i)
        
        # Verificar si ya existe
        if skip_if_exists:
            existing = db.query(UsageStatistic).filter(
                UsageStatistic.date == stat_date
            ).first()
            if existing:
                continue
        
        # Generar datos de muestra realistas
        books_processed = max(0, int(5 + (i % 7) - 3))  # 2-8 books per day
        books_completed = int(books_processed * 0.85)  # 85% success rate
        books_failed = books_processed - books_completed
        
        sample_stats = {
            "date": stat_date,
            "books_processed": books_processed,
            "books_completed": books_completed,
            "books_failed": books_failed,
            "total_processing_time_hours": books_processed * 2.5,  # 2.5h average
            "average_processing_time_minutes": 150,  # 2.5h in minutes
            "average_quality_score": 0.87 + (i % 10) * 0.01,  # 0.87-0.96
            "average_semantic_score": 0.85 + (i % 8) * 0.015,  # 0.85-0.955
            "average_format_preservation": 0.92 + (i % 5) * 0.01,  # 0.92-0.96
            "total_errors": books_processed // 3,  # Some errors
            "errors_resolved": books_processed // 4,  # Most resolved
            "critical_errors": max(0, books_processed // 10),  # Few critical
            "api_calls_deepl": books_processed * 50,  # 50 calls per book
            "api_calls_claude": books_processed * 25,  # 25 calls per book
            "api_calls_abbyy": books_processed * 5,   # 5 calls per book
            "total_api_cost": books_processed * 12.50,  # $12.50 per book
            "reviews_generated": books_processed,
            "reviews_completed": int(books_processed * 0.90),  # 90% completion
            "average_review_time_minutes": 45  # 45 minutes average
        }
        
        try:
            new_stat = UsageStatistic(**sample_stats)
            db.add(new_stat)
            created_count += 1
        except Exception as e:
            logger.warning(f"Failed to create usage statistic for {stat_date}: {e}")
    
    if created_count > 0:
        db.commit()
        logger.info(f"Created {created_count} sample usage statistics")

def _create_development_configs(db: Session, skip_if_exists: bool = True):
    """Crear configuraciones específicas para desarrollo con control mejorado"""
    dev_configs = [
        {
            "config_key": "debug_mode",
            "config_value": "true",
            "config_type": "boolean",
            "description": "Modo de depuración habilitado"
        },
        {
            "config_key": "log_level",
            "config_value": "INFO",
            "config_type": "string",
            "description": "Nivel de logging del sistema"
        },
        {
            "config_key": "cache_enabled",
            "config_value": "true",
            "config_type": "boolean",
            "description": "Habilitar cache de embeddings y API responses"
        },
        {
            "config_key": "mock_apis",
            "config_value": "false",
            "config_type": "boolean", 
            "description": "Usar APIs simuladas para desarrollo"
        },
        {
            "config_key": "dashboard_refresh_interval",
            "config_value": "30",
            "config_type": "integer",
            "description": "Intervalo de actualización del dashboard en segundos"
        }
    ]
    
    wrapper = MigrationCRUDWrapper()
    created_count = 0
    
    for config in dev_configs:
        if skip_if_exists:
            existing = db.query(SystemConfig).filter(
                SystemConfig.config_key == config["config_key"]
            ).first()
            if existing:
                continue
        
        if wrapper.safe_upsert_config(
            db,
            config["config_key"],
            config["config_value"],
            config["config_type"],
            config["description"]
        ):
            created_count += 1
    
    if created_count > 0:
        db.commit()
        logger.info(f"Created {created_count} development configurations")

# ==========================================
# ENHANCED DATABASE VALIDATION
# ==========================================

def validate_database_schema(db: Session) -> Dict[str, Any]:
    """
    Validar que el esquema de base de datos sea correcto con validaciones mejoradas
    """
    logger.info("Validating database schema...")
    
    validation_results = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "table_count": 0,
        "index_count": 0,
        "schema_version": "unknown"
    }
    
    try:
        # Get current schema version
        try:
            config = db.query(SystemConfig).filter(
                SystemConfig.config_key == "database_version"
            ).first()
            if config:
                validation_results["schema_version"] = config.config_value
        except Exception:
            validation_results["warnings"].append("Could not determine schema version")
        
        # Get database inspector
        inspector = inspect(db.bind)
        
        # Expected tables from models
        expected_tables = [
            "system_config",
            "book_processing_history", 
            "error_patterns",
            "audit_logs",
            "terminology_suggestions",
            "editorial_reviews",
            "usage_statistics"
        ]
        
        # Check if all tables exist
        existing_tables = inspector.get_table_names()
        validation_results["table_count"] = len(existing_tables)
        
        for table in expected_tables:
            if table not in existing_tables:
                validation_results["errors"].append(f"Missing table: {table}")
                validation_results["valid"] = False
        
        # Check indexes for each existing table with improved validation
        total_indexes = 0
        for table in existing_tables:
            try:
                indexes = inspector.get_indexes(table)
                total_indexes += len(indexes)
                
                # Validate specific important indexes
                if table == "book_processing_history":
                    index_names = [idx["name"] for idx in indexes]
                    if not any("book_id" in name.lower() for name in index_names):
                        validation_results["warnings"].append(f"Missing book_id index on {table}")
                
                # Check for primary key
                pk_constraint = inspector.get_pk_constraint(table)
                if not pk_constraint.get("constrained_columns"):
                    validation_results["warnings"].append(f"No primary key found on {table}")
                        
            except Exception as e:
                validation_results["warnings"].append(f"Could not validate indexes for {table}: {e}")
        
        validation_results["index_count"] = total_indexes
        
        # Check foreign key constraints with improved detection
        for table in existing_tables:
            try:
                fks = inspector.get_foreign_keys(table)
                if table in ["audit_logs", "terminology_suggestions", "editorial_reviews"]:
                    book_id_fk_found = any(
                        "book_id" in fk.get("constrained_columns", []) 
                        for fk in fks
                    )
                    if not book_id_fk_found:
                        validation_results["warnings"].append(f"Missing book_id foreign key on {table}")
                        
            except Exception as e:
                validation_results["warnings"].append(f"Could not validate foreign keys for {table}: {e}")
        
        logger.info(f"Schema validation completed: {validation_results}")
        
    except Exception as e:
        validation_results["valid"] = False
        validation_results["errors"].append(f"Validation error: {str(e)}")
    
    return validation_results

# ==========================================
# ENHANCED CLEANUP AND MAINTENANCE
# ==========================================

def cleanup_old_data(db: Session, days_to_keep: int = 90):
    """
    Limpiar datos antiguos para mantener performance con validaciones mejoradas
    """
    logger.info(f"Cleaning up data older than {days_to_keep} days...")
    
    cutoff_date = datetime.now() - timedelta(days=days_to_keep)
    cleanup_stats = {
        "audits_deleted": 0,
        "stats_deleted": 0,
        "reviews_deleted": 0,
        "errors": []
    }
    
    try:
        # Import here to avoid circular imports
        from .models import AuditLog, EditorialReview
        
        # Cleanup old audit logs (keep only recent)
        try:
            cleanup_stats["audits_deleted"] = db.query(AuditLog).filter(
                AuditLog.created_at < cutoff_date
            ).delete()
        except Exception as e:
            cleanup_stats["errors"].append(f"Audit log cleanup failed: {e}")
        
        # Cleanup old usage statistics (keep last 365 days)
        if days_to_keep > 365:
            try:
                stats_cutoff = date.today() - timedelta(days=365)
                cleanup_stats["stats_deleted"] = db.query(UsageStatistic).filter(
                    UsageStatistic.date < stats_cutoff
                ).delete()
            except Exception as e:
                cleanup_stats["errors"].append(f"Usage statistics cleanup failed: {e}")
        
        # Cleanup resolved editorial reviews older than cutoff
        try:
            cleanup_stats["reviews_deleted"] = db.query(EditorialReview).filter(
                EditorialReview.resolved == True,
                EditorialReview.resolved_at < cutoff_date
            ).delete()
        except Exception as e:
            cleanup_stats["errors"].append(f"Editorial review cleanup failed: {e}")
        
        db.commit()
        
        logger.info(f"Cleanup completed: {cleanup_stats}")
        
    except Exception as e:
        db.rollback()
        logger.error(f"Cleanup failed: {e}")
        raise

def optimize_database_performance(db: Session):
    """
    Optimizar performance de la base de datos con validaciones mejoradas
    """
    logger.info("Optimizing database performance...")
    
    optimization_stats = {
        "analyze_completed": False,
        "patterns_deactivated": 0,
        "errors": []
    }
    
    try:
        # Run ANALYZE for query planner optimization
        try:
            db.execute(text("ANALYZE"))
            optimization_stats["analyze_completed"] = True
        except Exception as e:
            optimization_stats["errors"].append(f"ANALYZE failed: {e}")
        
        # Update statistics for inactive error patterns
        try:
            # Use direct query instead of CRUD to avoid dependency issues
            inactive_patterns = db.query(ErrorPattern).filter(
                ErrorPattern.effectiveness_score < 0.1,
                ErrorPattern.usage_count >= 10
            ).count()
            
            if inactive_patterns > 0:
                db.query(ErrorPattern).filter(
                    ErrorPattern.effectiveness_score < 0.1,
                    ErrorPattern.usage_count >= 10
                ).update({
                    "is_active": False,
                    "updated_at": datetime.now()
                })
                optimization_stats["patterns_deactivated"] = inactive_patterns
                
        except Exception as e:
            optimization_stats["errors"].append(f"Pattern optimization failed: {e}")
        
        db.commit()
        logger.info(f"Performance optimization completed: {optimization_stats}")
        
    except Exception as e:
        db.rollback()
        logger.error(f"Performance optimization failed: {e}")
        raise

# ==========================================
# ENHANCED MAIN MIGRATION FUNCTIONS
# ==========================================

def run_migrations(operation: str = "forward"):
    """
    Función principal para ejecutar migraciones con soporte de operaciones
    """
    logger.info(f"Starting database migrations (operation: {operation})...")
    
    try:
        migration_manager = MigrationManager()
        op_enum = MigrationOperation(operation.lower())
        
        with db_manager.get_db_session() as db:
            if op_enum == MigrationOperation.FORWARD:
                if migration_manager.needs_migration(db):
                    migration_manager.run_migrations(db, op_enum)
                else:
                    logger.info("Database is up to date")
            else:
                migration_manager.run_migrations(db, op_enum)
        
        logger.info("Database migrations completed successfully")
        
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        raise

def validate_and_fix_database():
    """
    Validar y reparar base de datos si es necesario con validaciones mejoradas
    """
    logger.info("Validating database...")
    
    try:
        with db_manager.get_db_session() as db:
            validation = validate_database_schema(db)
            
            if not validation["valid"]:
                logger.error(f"Database validation failed: {validation['errors']}")
                raise RuntimeError("Database schema validation failed")
            
            if validation["warnings"]:
                logger.warning(f"Database warnings: {validation['warnings']}")
            
            logger.info(f"Database validation passed (version: {validation['schema_version']})")
        
    except Exception as e:
        logger.error(f"Database validation failed: {e}")
        raise

# ==========================================
# EXPORT MAIN FUNCTIONS
# ==========================================

__all__ = [
    "run_migrations",
    "create_initial_data", 
    "validate_and_fix_database",
    "cleanup_old_data",
    "optimize_database_performance",
    "MigrationManager",
    "MigrationOperation",
    "InitialPatternDefinitions",
    "MigrationCRUDWrapper"
]