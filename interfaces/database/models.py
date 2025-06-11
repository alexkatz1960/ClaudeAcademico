# ==========================================
# INTERFACES/DATABASE/MODELS.PY
# SQLAlchemy Models - Enterprise Implementation
# Sistema de Traducción Académica v2.2 - ENTERPRISE GRADE
# ==========================================

from sqlalchemy import (
    Column, Integer, String, Text, Float, Boolean, 
    DateTime, Date, JSON, ForeignKey, Index, CheckConstraint, event
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, validates
from sqlalchemy.sql import func
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
import json
import logging
from decimal import Decimal, InvalidOperation

# Import enterprise enums
from .enums import (
    BookStatus, ReviewSeverity, PatternType, ProcessingPhase,
    ConfigType, ReviewDecision, TerminologyPriority, ErrorSeverity
)

logger = logging.getLogger(__name__)

Base = declarative_base()

# ==========================================
# ENTERPRISE MIXINS AND BASE CLASSES
# ==========================================

class TimestampMixin:
    """Mixin para timestamps automáticos en todas las tablas"""
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now(), nullable=False)

class TableNameMixin:
    """Mixin para centralizar naming convention de tablas"""
    @classmethod
    def __tablename__(cls):
        # Convert CamelCase to snake_case
        import re
        name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', cls.__name__)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()

class ValidationMixin:
    """Mixin para validaciones comunes enterprise"""
    
    def validate_json_field(self, field_name: str, value: Any) -> bool:
        """Valida que un campo JSON sea serializable y válido"""
        if value is None:
            return True
        
        try:
            # Ensure serializability
            json.dumps(value)
            
            # Additional validation based on field name
            if field_name == 'quality_scores':
                return self._validate_quality_scores_schema(value)
            elif field_name == 'phases_completed':
                return self._validate_phases_completed_schema(value)
            elif field_name == 'api_rate_limits':
                return self._validate_api_rate_limits_schema(value)
                
            return True
            
        except (TypeError, ValueError) as e:
            logger.warning(f"JSON validation failed for {field_name}: {e}")
            return False
    
    def _validate_quality_scores_schema(self, value: Any) -> bool:
        """Validate quality scores JSON schema"""
        if not isinstance(value, dict):
            return False
        
        valid_phases = {phase.value for phase in ProcessingPhase}
        
        for phase, score in value.items():
            if phase not in valid_phases:
                return False
            if not isinstance(score, (int, float)) or not (0 <= score <= 1):
                return False
        
        return True
    
    def _validate_phases_completed_schema(self, value: Any) -> bool:
        """Validate phases completed JSON schema"""
        if not isinstance(value, list):
            return False
        
        valid_phases = {phase.value for phase in ProcessingPhase}
        
        for phase in value:
            if not isinstance(phase, str) or phase not in valid_phases:
                return False
        
        return True
    
    def _validate_api_rate_limits_schema(self, value: Any) -> bool:
        """Validate API rate limits JSON schema"""
        if not isinstance(value, dict):
            return False
        
        expected_apis = {'deepl', 'claude', 'abbyy'}
        
        for api_name, limits in value.items():
            if api_name not in expected_apis:
                continue
            if not isinstance(limits, dict):
                return False
            if 'requests_per_minute' not in limits:
                return False
            if not isinstance(limits['requests_per_minute'], int) or limits['requests_per_minute'] <= 0:
                return False
        
        return True

# ==========================================
# ENTERPRISE MODELS
# ==========================================

class SystemConfig(Base, TimestampMixin, ValidationMixin):
    """
    Configuración y metadatos del sistema - Enterprise Grade
    
    Almacena configuraciones dinámicas del sistema que pueden
    cambiar sin requerir restart de la aplicación.
    Enhanced with robust validation and type safety.
    """
    __tablename__ = "system_config"

    id = Column(Integer, primary_key=True, index=True)
    config_key = Column(String(100), unique=True, nullable=False, index=True)
    config_value = Column(Text, nullable=False)
    config_type = Column(String(20), default=ConfigType.STRING.value, nullable=False)
    description = Column(Text, nullable=False)  # Made non-nullable for enterprise compliance
    is_active = Column(Boolean, default=True, nullable=False)
    
    # Enterprise metadata
    validation_regex = Column(String(500))  # Optional regex for value validation
    default_value = Column(Text)  # Fallback value
    is_sensitive = Column(Boolean, default=False)  # Mark sensitive configs (passwords, etc.)
    
    # Constraints with enum integration
    __table_args__ = (
        CheckConstraint(
            f"config_type IN {tuple(ct.value for ct in ConfigType)}",
            name='valid_config_type'
        ),
        CheckConstraint(
            "LENGTH(config_key) >= 3",
            name='min_config_key_length'
        ),
        CheckConstraint(
            "LENGTH(description) >= 10",
            name='min_description_length'
        ),
        Index('idx_config_key_active', config_key, is_active),
        Index('idx_config_type_active', config_type, is_active),
    )

    def __repr__(self):
        return f"<SystemConfig(key='{self.config_key}', type='{self.config_type}')>"

    @property
    def parsed_value(self) -> Any:
        """Convierte config_value al tipo apropiado con validación robusta"""
        try:
            config_type_enum = ConfigType(self.config_type)
            
            if config_type_enum == ConfigType.INTEGER:
                return int(self.config_value)
            elif config_type_enum == ConfigType.FLOAT:
                return float(self.config_value)
            elif config_type_enum == ConfigType.BOOLEAN:
                return self.config_value.lower() in ('true', '1', 'yes', 'on')
            elif config_type_enum == ConfigType.JSON:
                return json.loads(self.config_value)
            else:  # STRING
                return self.config_value
                
        except (ValueError, TypeError, json.JSONDecodeError) as e:
            logger.error(f"Failed to parse config {self.config_key}: {e}")
            
            # Return default value if available, otherwise raise
            if self.default_value is not None:
                logger.warning(f"Using default value for config {self.config_key}")
                return self.default_value
            
            raise ValueError(f"Invalid config value for {self.config_key}: {e}")
    
    @validates('config_value')
    def validate_config_value(self, key, value):
        """Validate config value before assignment"""
        if not value:
            if self.default_value:
                return self.default_value
            raise ValueError(f"Config value cannot be empty for {self.config_key}")
        
        # Validate against regex if provided
        if self.validation_regex:
            import re
            if not re.match(self.validation_regex, str(value)):
                raise ValueError(f"Config value {value} doesn't match pattern {self.validation_regex}")
        
        return value
    
    def update_value(self, new_value: Any, validate: bool = True) -> bool:
        """Safely update configuration value with validation"""
        try:
            if validate:
                # Test parsing with new value
                old_value = self.config_value
                self.config_value = str(new_value)
                
                try:
                    _ = self.parsed_value  # Test parsing
                except Exception:
                    self.config_value = old_value  # Rollback
                    raise
            else:
                self.config_value = str(new_value)
            
            self.updated_at = func.now()
            logger.info(f"Updated config {self.config_key} = {new_value}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update config {self.config_key}: {e}")
            return False

class BookProcessingHistory(Base, TimestampMixin, ValidationMixin):
    """
    Historial completo de procesamiento de libros - Enterprise Grade
    
    Registro detallado de cada libro procesado por el sistema,
    incluyendo métricas de calidad, tiempo y errores.
    Enhanced with comprehensive validation and metrics.
    """
    __tablename__ = "book_processing_history"

    id = Column(Integer, primary_key=True, index=True)
    book_id = Column(String(100), unique=True, nullable=False, index=True)
    title = Column(String(500), nullable=False)  # Made non-nullable
    source_lang = Column(String(5), nullable=False, index=True)
    target_lang = Column(String(5), default='es', nullable=False)
    
    # Status tracking with enum
    status = Column(String(20), nullable=False, index=True, default=BookStatus.QUEUED.value)
    current_phase = Column(String(50))
    progress_percentage = Column(Float, default=0.0, nullable=False)
    
    # Enhanced timestamps
    started_at = Column(DateTime, default=func.now(), nullable=False)
    completed_at = Column(DateTime)
    last_activity_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Processing details with validation
    total_phases = Column(Integer, default=6, nullable=False)
    phases_completed = Column(JSON)  # Validated JSON array
    quality_scores = Column(JSON)    # Validated JSON object
    error_count = Column(Integer, default=0, nullable=False)
    warning_count = Column(Integer, default=0, nullable=False)
    processing_time_seconds = Column(Integer)
    
    # File paths - enhanced validation
    input_file_path = Column(String(500), nullable=False)
    output_file_path = Column(String(500))
    backup_file_path = Column(String(500))
    
    # Enhanced metrics
    total_paragraphs = Column(Integer)
    total_footnotes = Column(Integer)
    total_words = Column(Integer)
    semantic_score_avg = Column(Float)
    format_preservation_score = Column(Float)
    footnote_preservation_score = Column(Float)
    
    # Processing metadata
    processing_node = Column(String(100))  # For distributed processing
    priority = Column(Integer, default=5)  # 1-10 priority scale
    estimated_completion = Column(DateTime)

    # Enhanced relationships
    audit_logs = relationship("AuditLog", back_populates="book", cascade="all, delete-orphan")
    editorial_reviews = relationship("EditorialReview", back_populates="book", cascade="all, delete-orphan")
    terminology_suggestions = relationship("TerminologySuggestion", back_populates="book", cascade="all, delete-orphan")

    # Enhanced constraints with enum integration
    __table_args__ = (
        CheckConstraint(
            f"status IN {tuple(status.value for status in BookStatus)}",
            name='valid_status'
        ),
        CheckConstraint(
            "progress_percentage >= 0 AND progress_percentage <= 100",
            name='valid_progress'
        ),
        CheckConstraint(
            "source_lang IN ('de', 'en', 'fr', 'it', 'nl')",
            name='valid_source_lang'
        ),
        CheckConstraint(
            "target_lang IN ('es', 'en')",  # Future support for multiple targets
            name='valid_target_lang'
        ),
        CheckConstraint(
            "priority >= 1 AND priority <= 10",
            name='valid_priority'
        ),
        CheckConstraint(
            "total_phases >= 1 AND total_phases <= 20",
            name='valid_total_phases'
        ),
        Index('idx_book_status_lang', status, source_lang),
        Index('idx_book_completed_at', completed_at),
        Index('idx_book_priority_status', priority.desc(), status),
    )

    def __repr__(self):
        return f"<BookProcessingHistory(book_id='{self.book_id}', status='{self.status}')>"

    @validates('phases_completed', 'quality_scores')
    def validate_json_fields(self, key, value):
        """Validate JSON fields against schemas"""
        if value is not None and not self.validate_json_field(key, value):
            raise ValueError(f"Invalid JSON schema for {key}")
        return value

    @property
    def status_enum(self) -> BookStatus:
        """Get status as enum"""
        return BookStatus(self.status)

    @property
    def duration_minutes(self) -> Optional[float]:
        """Duración total del procesamiento en minutos"""
        if self.processing_time_seconds:
            return self.processing_time_seconds / 60
        return None

    @property
    def duration_hours(self) -> Optional[float]:
        """Duración total en horas"""
        if self.processing_time_seconds:
            return self.processing_time_seconds / 3600
        return None

    @property
    def is_completed(self) -> bool:
        """Indica si el procesamiento está completado"""
        return self.status == BookStatus.COMPLETED.value

    @property
    def is_active(self) -> bool:
        """Indica si está en procesamiento activo"""
        return self.status in [BookStatus.QUEUED.value, BookStatus.PROCESSING.value]

    @property
    def completion_rate(self) -> float:
        """Tasa de completitud de fases"""
        if not self.phases_completed or self.total_phases == 0:
            return 0.0
        return len(self.phases_completed) / self.total_phases

    @property
    def average_quality_score(self) -> Optional[float]:
        """Score promedio de calidad de todas las fases"""
        if not self.quality_scores:
            return None
        
        scores = [score for score in self.quality_scores.values() if isinstance(score, (int, float))]
        return sum(scores) / len(scores) if scores else None

    @property
    def words_per_minute(self) -> Optional[float]:
        """Velocidad de procesamiento en palabras por minuto"""
        if self.total_words and self.duration_minutes:
            return self.total_words / self.duration_minutes
        return None
    
    def update_progress(self, phase: ProcessingPhase, quality_score: float, 
                       progress_percentage: float = None):
        """Update progress with enterprise logging"""
        try:
            # Update phases completed
            if not self.phases_completed:
                self.phases_completed = []
            
            if phase.value not in self.phases_completed:
                self.phases_completed.append(phase.value)
            
            # Update quality scores
            if not self.quality_scores:
                self.quality_scores = {}
            
            self.quality_scores[phase.value] = quality_score
            
            # Update progress percentage
            if progress_percentage is not None:
                self.progress_percentage = min(100.0, max(0.0, progress_percentage))
            else:
                # Calculate based on phases completed
                self.progress_percentage = (len(self.phases_completed) / self.total_phases) * 100
            
            # Update current phase
            self.current_phase = phase.value
            self.last_activity_at = func.now()
            
            logger.info(f"Updated progress for {self.book_id}: {phase.value} = {quality_score:.3f}")
            
        except Exception as e:
            logger.error(f"Failed to update progress for {self.book_id}: {e}")
            raise

class ErrorPattern(Base, TimestampMixin):
    """
    Patrones de errores para aprendizaje incremental - Enterprise Grade
    
    Sistema de machine learning que aprende de errores pasados
    para mejorar el procesamiento futuro.
    Enhanced with robust metrics and validation.
    """
    __tablename__ = "error_patterns"

    id = Column(Integer, primary_key=True, index=True)
    pattern_type = Column(String(50), nullable=False, index=True)
    pattern_content = Column(Text, nullable=False)
    pattern_regex = Column(Text)
    
    # Enhanced effectiveness metrics
    frequency = Column(Integer, default=1, nullable=False)
    success_rate = Column(Float, default=0.0, nullable=False)
    effectiveness_score = Column(Float, default=0.0, nullable=False)
    false_positive_rate = Column(Float, default=0.0, nullable=False)
    confidence_interval = Column(Float, default=0.0)  # Statistical confidence
    
    # Enhanced tracking
    first_seen = Column(DateTime, default=func.now(), nullable=False)
    last_seen = Column(DateTime, default=func.now(), nullable=False)
    usage_count = Column(Integer, default=0, nullable=False)
    success_count = Column(Integer, default=0, nullable=False)
    failure_count = Column(Integer, default=0, nullable=False)
    created_by = Column(String(50), default='system', nullable=False)
    
    # Enhanced metadata
    description = Column(Text, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    category = Column(String(50))  # Subcategory within pattern_type
    severity_weight = Column(Float, default=1.0)  # Weight for severity calculations
    
    # ML features
    feature_vector = Column(JSON)  # For ML model training
    model_version = Column(String(20))  # Track which model version created this

    # Enhanced constraints with enum integration
    __table_args__ = (
        CheckConstraint(
            f"pattern_type IN {tuple(pt.value for pt in PatternType)}",
            name='valid_pattern_type'
        ),
        CheckConstraint(
            "success_rate >= 0 AND success_rate <= 1",
            name='valid_success_rate'
        ),
        CheckConstraint(
            "effectiveness_score >= 0 AND effectiveness_score <= 1",
            name='valid_effectiveness'
        ),
        CheckConstraint(
            "false_positive_rate >= 0 AND false_positive_rate <= 1",
            name='valid_false_positive_rate'
        ),
        CheckConstraint(
            "usage_count = success_count + failure_count",
            name='valid_usage_count'
        ),
        Index('idx_pattern_type_active', pattern_type, is_active),
        Index('idx_pattern_effectiveness', effectiveness_score.desc()),
        Index('idx_pattern_category_type', category, pattern_type),
    )

    def __repr__(self):
        return f"<ErrorPattern(type='{self.pattern_type}', effectiveness={self.effectiveness_score:.3f})>"

    @property
    def pattern_type_enum(self) -> PatternType:
        """Get pattern type as enum"""
        return PatternType(self.pattern_type)

    def update_effectiveness(self, success: bool):
        """Actualiza métricas de efectividad basado en uso con logging enterprise"""
        try:
            # Update counters
            self.usage_count += 1
            self.last_seen = func.now()
            
            if success:
                self.success_count += 1
            else:
                self.failure_count += 1
            
            # Recalculate success rate
            self.success_rate = self.success_count / self.usage_count if self.usage_count > 0 else 0.0
            
            # Calculate confidence interval (simple binomial confidence)
            if self.usage_count >= 30:  # Sufficient sample size
                import math
                p = self.success_rate
                n = self.usage_count
                self.confidence_interval = 1.96 * math.sqrt((p * (1 - p)) / n)
            else:
                self.confidence_interval = 1.0  # Low confidence with small sample
            
            # Enhanced effectiveness score calculation
            # Combines success rate, frequency weight, and confidence
            frequency_weight = min(self.frequency / 100, 1.0)
            confidence_weight = max(0.1, 1.0 - self.confidence_interval)
            
            self.effectiveness_score = (
                self.success_rate * 0.6 +  # Primary factor: success rate
                frequency_weight * 0.2 +   # Secondary: frequency
                confidence_weight * 0.2    # Tertiary: confidence
            )
            
            logger.info(
                f"Updated pattern {self.id}: success_rate={self.success_rate:.3f}, "
                f"effectiveness={self.effectiveness_score:.3f}, confidence={confidence_weight:.3f}"
            )
            
        except Exception as e:
            logger.error(f"Failed to update effectiveness for pattern {self.id}: {e}")
            raise
    
    def deactivate_if_ineffective(self, min_usage: int = 20, min_effectiveness: float = 0.3) -> bool:
        """Deactivate pattern if proven ineffective"""
        if self.usage_count >= min_usage and self.effectiveness_score < min_effectiveness:
            self.is_active = False
            logger.warning(f"Deactivated ineffective pattern {self.id}: {self.effectiveness_score:.3f}")
            return True
        return False

class AuditLog(Base, TimestampMixin, ValidationMixin):
    """
    Registro detallado de auditorías por fase - Enterprise Grade
    
    Cada fase del pipeline genera un log de auditoría detallado
    con métricas de calidad y alertas identificadas.
    Enhanced with comprehensive metrics and validation.
    """
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    book_id = Column(String(100), ForeignKey('book_processing_history.book_id'), nullable=False, index=True)
    phase_name = Column(String(50), nullable=False, index=True)
    
    # Enhanced quality scores
    quality_score = Column(Float, nullable=False)
    integrity_score = Column(Float)
    format_preservation_score = Column(Float)
    footnote_preservation_score = Column(Float)
    semantic_similarity_score = Column(Float)
    terminology_consistency_score = Column(Float)
    
    # Enhanced performance metrics
    processing_time_seconds = Column(Integer)
    memory_usage_mb = Column(Float)
    cpu_usage_percent = Column(Float)
    api_calls_count = Column(Integer, default=0)
    
    # Enhanced alertas y problemas
    alerts_count = Column(Integer, default=0, nullable=False)
    critical_alerts_count = Column(Integer, default=0, nullable=False)
    alerts_detail = Column(JSON)  # Array de alertas detalladas - validado
    metrics_detail = Column(JSON)  # Métricas específicas por fase - validado
    improvements_applied = Column(JSON)  # Mejoras aplicadas automáticamente - validado
    
    # Enhanced metadata
    system_version = Column(String(20), nullable=False)
    api_versions = Column(JSON)  # Versiones de APIs externas usadas
    environment = Column(String(20), default='production')
    processing_node = Column(String(100))  # For distributed processing
    
    # Validation results
    validation_passed = Column(Boolean, default=True)
    validation_errors = Column(JSON)

    # Enhanced relationships
    book = relationship("BookProcessingHistory", back_populates="audit_logs")

    # Enhanced constraints with enum integration
    __table_args__ = (
        CheckConstraint(
            f"phase_name IN {tuple(phase.value for phase in ProcessingPhase)}",
            name='valid_phase_name'
        ),
        CheckConstraint(
            "quality_score >= 0 AND quality_score <= 1",
            name='valid_quality_score'
        ),
        CheckConstraint(
            "integrity_score IS NULL OR (integrity_score >= 0 AND integrity_score <= 1)",
            name='valid_integrity_score'
        ),
        CheckConstraint(
            "alerts_count >= critical_alerts_count",
            name='valid_alert_counts'
        ),
        CheckConstraint(
            "processing_time_seconds IS NULL OR processing_time_seconds >= 0",
            name='valid_processing_time'
        ),
        Index('idx_audit_book_phase', book_id, phase_name),
        Index('idx_audit_quality_score', quality_score.desc()),
        Index('idx_audit_critical_alerts', critical_alerts_count.desc()),
    )

    def __repr__(self):
        return f"<AuditLog(book_id='{self.book_id}', phase='{self.phase_name}', score={self.quality_score:.3f})>"

    @validates('alerts_detail', 'metrics_detail', 'improvements_applied', 'api_versions', 'validation_errors')
    def validate_json_fields(self, key, value):
        """Validate JSON fields against schemas"""
        if value is not None and not self.validate_json_field(key, value):
            raise ValueError(f"Invalid JSON schema for {key}")
        return value

    @property
    def phase_enum(self) -> ProcessingPhase:
        """Get phase as enum"""
        return ProcessingPhase(self.phase_name)

    @property
    def has_critical_alerts(self) -> bool:
        """Indica si hay alertas críticas en esta auditoría"""
        return self.critical_alerts_count > 0

    @property
    def has_warnings(self) -> bool:
        """Indica si hay alertas de advertencia"""
        return self.alerts_count > self.critical_alerts_count

    @property
    def performance_score(self) -> Optional[float]:
        """Calculate combined performance score"""
        if not all([self.processing_time_seconds, self.memory_usage_mb]):
            return None
        
        # Normalized performance score (lower is better, so invert)
        time_score = max(0, 1 - (self.processing_time_seconds / 3600))  # Normalize to 1 hour
        memory_score = max(0, 1 - (self.memory_usage_mb / 1024))  # Normalize to 1GB
        
        return (time_score + memory_score) / 2

    def add_alert(self, alert_type: str, severity: ErrorSeverity, message: str, 
                  context: Dict[str, Any] = None):
        """Add alert with enterprise validation"""
        try:
            if not self.alerts_detail:
                self.alerts_detail = []
            
            alert = {
                'type': alert_type,
                'severity': severity.value,
                'message': message,
                'context': context or {},
                'timestamp': datetime.now().isoformat()
            }
            
            self.alerts_detail.append(alert)
            self.alerts_count = len(self.alerts_detail)
            
            # Update critical alerts count
            self.critical_alerts_count = sum(
                1 for alert in self.alerts_detail 
                if alert.get('severity') == ErrorSeverity.CRITICAL.value
            )
            
            logger.warning(f"Added {severity.value} alert to {self.book_id}/{self.phase_name}: {message}")
            
        except Exception as e:
            logger.error(f"Failed to add alert: {e}")
            raise

class TerminologySuggestion(Base, TimestampMixin):
    """
    Sugerencias de términos generadas por Claude - Enterprise Grade
    
    Sistema de aprendizaje terminológico que captura y gestiona
    sugerencias de Claude para mejorar la consistencia.
    Enhanced with comprehensive tracking and validation.
    """
    __tablename__ = "terminology_suggestions"

    id = Column(Integer, primary_key=True, index=True)
    book_id = Column(String(100), ForeignKey('book_processing_history.book_id'), nullable=False, index=True)
    glossary_id = Column(String(100), nullable=False, index=True)
    
    # Enhanced terms
    source_term = Column(String(200), nullable=False, index=True)
    target_term = Column(String(200), nullable=False)
    alternative_terms = Column(JSON)  # Array of alternative translations
    context = Column(Text, nullable=False)
    justification = Column(Text, nullable=False)
    
    # Enhanced metrics
    confidence_score = Column(Float, nullable=False)
    priority = Column(String(10), default=TerminologyPriority.MEDIUM.value, nullable=False)
    frequency_estimate = Column(String(20), default='unknown')
    usage_frequency = Column(Integer, default=0)  # Actual usage count
    
    # Enhanced state tracking
    suggested_by = Column(String(50), default='claude', nullable=False)
    applied = Column(Boolean, default=False, nullable=False)
    reviewed = Column(Boolean, default=False, nullable=False)
    approved = Column(Boolean, default=False, nullable=False)
    
    # Enhanced feedback
    editor_feedback = Column(Text)
    rejection_reason = Column(Text)
    reviewer_id = Column(String(50))
    review_session_id = Column(String(100))
    
    # Enhanced metadata
    discipline = Column(String(50))  # Academic discipline
    source_language = Column(String(5), nullable=False)
    complexity_level = Column(Integer, default=5)  # 1-10 complexity scale
    
    # ML features
    semantic_features = Column(JSON)  # For ML analysis
    usage_patterns = Column(JSON)  # Usage pattern analysis

    # Enhanced relationships
    book = relationship("BookProcessingHistory", back_populates="terminology_suggestions")

    # Enhanced constraints with enum integration
    __table_args__ = (
        CheckConstraint(
            f"priority IN {tuple(tp.value for tp in TerminologyPriority)}",
            name='valid_priority'
        ),
        CheckConstraint(
            "confidence_score >= 0 AND confidence_score <= 1",
            name='valid_confidence'
        ),
        CheckConstraint(
            "complexity_level >= 1 AND complexity_level <= 10",
            name='valid_complexity'
        ),
        CheckConstraint(
            "usage_frequency >= 0",
            name='valid_usage_frequency'
        ),
        Index('idx_terminology_book_glossary', book_id, glossary_id),
        Index('idx_terminology_source_term', source_term),
        Index('idx_terminology_priority_reviewed', priority, reviewed),
        Index('idx_terminology_applied_approved', applied, approved),
    )

    def __repr__(self):
        return f"<TerminologySuggestion('{self.source_term}' -> '{self.target_term}', priority='{self.priority}')>"

    @property
    def priority_enum(self) -> TerminologyPriority:
        """Get priority as enum"""
        return TerminologyPriority(self.priority)

    def mark_applied(self, usage_count: int = 1):
        """Mark suggestion as applied with usage tracking"""
        try:
            self.applied = True
            self.usage_frequency += usage_count
            
            logger.info(f"Marked terminology suggestion {self.id} as applied: {self.source_term} -> {self.target_term}")
            
        except Exception as e:
            logger.error(f"Failed to mark suggestion {self.id} as applied: {e}")
            raise
    
    def update_confidence(self, new_confidence: float, reason: str = ""):
        """Update confidence score with logging"""
        try:
            old_confidence = self.confidence_score
            self.confidence_score = max(0.0, min(1.0, new_confidence))
            
            logger.info(
                f"Updated confidence for suggestion {self.id}: {old_confidence:.3f} -> {new_confidence:.3f} "
                f"({reason})"
            )
            
        except Exception as e:
            logger.error(f"Failed to update confidence for suggestion {self.id}: {e}")
            raise

class EditorialReview(Base, TimestampMixin):
    """
    Registro completo de revisiones editoriales - Enterprise Grade
    
    Sistema de workflow para revisión editorial de traducciones,
    incluyendo tracking de decisiones y tiempos.
    Enhanced with comprehensive workflow management.
    """
    __tablename__ = "editorial_reviews"

    id = Column(Integer, primary_key=True, index=True)
    book_id = Column(String(100), ForeignKey('book_processing_history.book_id'), nullable=False, index=True)
    item_id = Column(String(100), nullable=False)
    section_number = Column(Integer, index=True)
    
    # Enhanced content
    original_text = Column(Text, nullable=False)
    translated_text = Column(Text, nullable=False)
    location_info = Column(String(200), nullable=False)
    page_number = Column(Integer)
    paragraph_number = Column(Integer)
    
    # Enhanced analysis
    alert_type = Column(String(50), nullable=False)
    severity = Column(String(10), nullable=False)
    similarity_score = Column(Float, nullable=False)
    suggested_action = Column(Text, nullable=False)
    
    # Enhanced decision tracking
    editor_decision = Column(String(20), default=ReviewDecision.PENDING.value)
    editor_notes = Column(Text)
    corrected_text = Column(Text)
    correction_type = Column(String(50))  # Type of correction made
    
    # Enhanced state and tracking
    reviewed = Column(Boolean, default=False, nullable=False)
    resolved = Column(Boolean, default=False, nullable=False)
    resolution_time_minutes = Column(Integer)
    review_difficulty = Column(Integer, default=5)  # 1-10 difficulty scale
    
    # Enhanced timestamps
    reviewed_at = Column(DateTime)
    resolved_at = Column(DateTime)
    deadline = Column(DateTime)
    
    # Enhanced metadata
    reviewer_id = Column(String(50))
    review_session_id = Column(String(100))
    client_feedback = Column(Text)
    quality_assessment = Column(Float)  # Post-review quality score
    
    # Workflow management
    assigned_to = Column(String(50))
    escalated = Column(Boolean, default=False)
    escalation_reason = Column(Text)

    # Enhanced relationships
    book = relationship("BookProcessingHistory", back_populates="editorial_reviews")

    # Enhanced constraints with enum integration
    __table_args__ = (
        CheckConstraint(
            f"severity IN {tuple(rs.value for rs in ReviewSeverity)}",
            name='valid_severity'
        ),
        CheckConstraint(
            f"editor_decision IN {tuple(rd.value for rd in ReviewDecision)}",
            name='valid_editor_decision'
        ),
        CheckConstraint(
            "similarity_score >= 0 AND similarity_score <= 1",
            name='valid_similarity_score'
        ),
        CheckConstraint(
            "review_difficulty >= 1 AND review_difficulty <= 10",
            name='valid_review_difficulty'
        ),
        CheckConstraint(
            "quality_assessment IS NULL OR (quality_assessment >= 0 AND quality_assessment <= 1)",
            name='valid_quality_assessment'
        ),
        Index('idx_review_book_severity', book_id, severity),
        Index('idx_review_resolved_reviewed', resolved, reviewed),
        Index('idx_review_assigned_deadline', assigned_to, deadline),
    )

    def __repr__(self):
        return f"<EditorialReview(book_id='{self.book_id}', severity='{self.severity}', resolved={self.resolved})>"

    @property
    def severity_enum(self) -> ReviewSeverity:
        """Get severity as enum"""
        return ReviewSeverity(self.severity)

    @property
    def decision_enum(self) -> ReviewDecision:
        """Get decision as enum"""
        return ReviewDecision(self.editor_decision)

    def mark_reviewed(self, reviewer_id: str, decision: ReviewDecision, notes: str = ""):
        """Marca el item como revisado con logging enterprise"""
        try:
            self.reviewed = True
            self.reviewed_at = func.now()
            self.reviewer_id = reviewer_id
            self.editor_decision = decision.value
            self.editor_notes = notes
            
            logger.info(f"Marked review {self.id} as reviewed by {reviewer_id}: {decision.value}")
            
        except Exception as e:
            logger.error(f"Failed to mark review {self.id} as reviewed: {e}")
            raise

    def mark_resolved(self, quality_score: float = None):
        """Marca el item como resuelto con métricas enterprise"""
        try:
            self.resolved = True
            self.resolved_at = func.now()
            
            if quality_score is not None:
                self.quality_assessment = max(0.0, min(1.0, quality_score))
            
            if self.reviewed_at:
                # Calcular tiempo de resolución
                resolution_time = (self.resolved_at - self.reviewed_at).total_seconds() / 60
                self.resolution_time_minutes = int(resolution_time)
            
            logger.info(f"Marked review {self.id} as resolved in {self.resolution_time_minutes} minutes")
            
        except Exception as e:
            logger.error(f"Failed to mark review {self.id} as resolved: {e}")
            raise
    
    def escalate(self, reason: str, assigned_to: str = None):
        """Escalate review to higher level"""
        try:
            self.escalated = True
            self.escalation_reason = reason
            
            if assigned_to:
                self.assigned_to = assigned_to
            
            logger.warning(f"Escalated review {self.id}: {reason}")
            
        except Exception as e:
            logger.error(f"Failed to escalate review {self.id}: {e}")
            raise

class UsageStatistic(Base):
    """
    Estadísticas diarias de uso del sistema - Enterprise Grade
    
    Métricas agregadas por día para análisis de performance
    y uso del sistema a lo largo del tiempo.
    Enhanced with comprehensive metrics and calculations.
    """
    __tablename__ = "usage_statistics"

    id = Column(Integer, primary_key=True, index=True)
    date = Column(Date, unique=True, nullable=False, index=True)
    
    # Enhanced processing statistics
    books_processed = Column(Integer, default=0, nullable=False)
    books_completed = Column(Integer, default=0, nullable=False)
    books_failed = Column(Integer, default=0, nullable=False)
    books_cancelled = Column(Integer, default=0, nullable=False)
    
    # Enhanced time metrics
    total_processing_time_hours = Column(Float, default=0.0, nullable=False)
    average_processing_time_minutes = Column(Float, default=0.0)
    median_processing_time_minutes = Column(Float, default=0.0)
    min_processing_time_minutes = Column(Float)
    max_processing_time_minutes = Column(Float)
    
    # Enhanced quality metrics
    average_quality_score = Column(Float, default=0.0)
    average_semantic_score = Column(Float, default=0.0)
    average_format_preservation = Column(Float, default=0.0)
    quality_score_std_dev = Column(Float, default=0.0)  # Standard deviation
    
    # Enhanced error statistics
    total_errors = Column(Integer, default=0, nullable=False)
    errors_resolved = Column(Integer, default=0, nullable=False)
    critical_errors = Column(Integer, default=0, nullable=False)
    system_errors = Column(Integer, default=0, nullable=False)
    user_errors = Column(Integer, default=0, nullable=False)
    
    # Enhanced API usage
    api_calls_deepl = Column(Integer, default=0, nullable=False)
    api_calls_claude = Column(Integer, default=0, nullable=False)
    api_calls_abbyy = Column(Integer, default=0, nullable=False)
    total_api_cost = Column(Float, default=0.0, nullable=False)
    api_response_time_avg = Column(Float, default=0.0)
    
    # Enhanced editorial metrics
    reviews_generated = Column(Integer, default=0, nullable=False)
    reviews_completed = Column(Integer, default=0, nullable=False)
    average_review_time_minutes = Column(Float, default=0.0)
    reviews_escalated = Column(Integer, default=0, nullable=False)
    
    # Enhanced system metrics
    system_uptime_minutes = Column(Integer, default=0)
    peak_concurrent_books = Column(Integer, default=0)
    average_memory_usage_mb = Column(Float, default=0.0)
    average_cpu_usage_percent = Column(Float, default=0.0)
    
    # Language distribution
    books_de_es = Column(Integer, default=0)
    books_en_es = Column(Integer, default=0)
    books_fr_es = Column(Integer, default=0)
    books_it_es = Column(Integer, default=0)
    books_nl_es = Column(Integer, default=0)

    # Enhanced constraints
    __table_args__ = (
        CheckConstraint(
            "books_completed + books_failed + books_cancelled <= books_processed",
            name='valid_completion_count'
        ),
        CheckConstraint(
            "average_quality_score >= 0 AND average_quality_score <= 1",
            name='valid_avg_quality'
        ),
        CheckConstraint(
            "errors_resolved <= total_errors",
            name='valid_error_resolution'
        ),
        CheckConstraint(
            "reviews_completed <= reviews_generated",
            name='valid_review_completion'
        ),
        Index('idx_usage_date', date.desc()),
        Index('idx_usage_books_processed', books_processed.desc()),
    )

    def __repr__(self):
        return f"<UsageStatistic(date='{self.date}', processed={self.books_processed})>"

    @property
    def success_rate(self) -> float:
        """Tasa de éxito en procesamiento"""
        if self.books_processed == 0:
            return 0.0
        return self.books_completed / self.books_processed

    @property
    def failure_rate(self) -> float:
        """Tasa de fallos en procesamiento"""
        if self.books_processed == 0:
            return 0.0
        return self.books_failed / self.books_processed

    @property
    def error_resolution_rate(self) -> float:
        """Tasa de resolución de errores"""
        if self.total_errors == 0:
            return 1.0
        return self.errors_resolved / self.total_errors

    @property
    def review_completion_rate(self) -> float:
        """Tasa de completitud de revisiones"""
        if self.reviews_generated == 0:
            return 1.0
        return self.reviews_completed / self.reviews_generated

    @property
    def books_per_hour(self) -> float:
        """Libros procesados por hora"""
        if self.total_processing_time_hours == 0:
            return 0.0
        return self.books_processed / self.total_processing_time_hours

    @property
    def cost_per_book(self) -> float:
        """Costo promedio por libro"""
        if self.books_processed == 0:
            return 0.0
        return self.total_api_cost / self.books_processed

    @property
    def efficiency_score(self) -> float:
        """Score combinado de eficiencia"""
        if self.books_processed == 0:
            return 0.0
        
        # Combine multiple factors into efficiency score
        success_weight = self.success_rate * 0.4
        quality_weight = self.average_quality_score * 0.3 if self.average_quality_score else 0
        speed_weight = min(1.0, self.books_per_hour / 5.0) * 0.2  # Normalize to 5 books/hour max
        error_weight = self.error_resolution_rate * 0.1
        
        return success_weight + quality_weight + speed_weight + error_weight

# ==========================================
# ENTERPRISE UTILITIES AND FUNCTIONS
# ==========================================

def create_indexes():
    """Crear índices adicionales para optimización de queries enterprise"""
    additional_indexes = [
        # Core performance indexes
        "CREATE INDEX IF NOT EXISTS idx_book_status_created ON book_processing_history(status, created_at DESC)",
        "CREATE INDEX IF NOT EXISTS idx_audit_quality_phase ON audit_logs(quality_score DESC, phase_name)",
        "CREATE INDEX IF NOT EXISTS idx_review_severity_created ON editorial_reviews(severity, created_at DESC)",
        "CREATE INDEX IF NOT EXISTS idx_terminology_applied_priority ON terminology_suggestions(applied, priority)",
        "CREATE INDEX IF NOT EXISTS idx_error_pattern_effectiveness ON error_patterns(effectiveness_score DESC, is_active)",
        "CREATE INDEX IF NOT EXISTS idx_usage_stats_metrics ON usage_statistics(date DESC, books_processed, average_quality_score)",
        "CREATE INDEX IF NOT EXISTS idx_book_completion_time ON book_processing_history(completed_at, processing_time_seconds)",
        
        # Enterprise analytics indexes
        "CREATE INDEX IF NOT EXISTS idx_book_language_status ON book_processing_history(source_lang, target_lang, status)",
        "CREATE INDEX IF NOT EXISTS idx_audit_performance ON audit_logs(processing_time_seconds, memory_usage_mb)",
        "CREATE INDEX IF NOT EXISTS idx_review_workflow ON editorial_reviews(assigned_to, deadline, resolved)",
        "CREATE INDEX IF NOT EXISTS idx_terminology_usage ON terminology_suggestions(usage_frequency DESC, applied)",
        "CREATE INDEX IF NOT EXISTS idx_error_pattern_ml ON error_patterns(model_version, category, effectiveness_score)",
        
        # Dashboard optimization indexes
        "CREATE INDEX IF NOT EXISTS idx_book_active_priority ON book_processing_history(status, priority DESC) WHERE status IN ('queued', 'processing')",
        "CREATE INDEX IF NOT EXISTS idx_review_pending_severity ON editorial_reviews(severity, created_at) WHERE resolved = false",
        "CREATE INDEX IF NOT EXISTS idx_stats_recent ON usage_statistics(date DESC, books_processed) WHERE date >= date('now', '-30 days')",
        
        # ML and analytics indexes
        "CREATE INDEX IF NOT EXISTS idx_pattern_learning ON error_patterns(pattern_type, usage_count, success_rate)",
        "CREATE INDEX IF NOT EXISTS idx_terminology_semantic ON terminology_suggestions(discipline, confidence_score DESC)",
    ]
    return additional_indexes

def validate_language_code(lang_code: str) -> bool:
    """Valida códigos de idioma soportados con logging enterprise"""
    supported_languages = {'de', 'en', 'fr', 'it', 'nl', 'es'}
    is_valid = lang_code in supported_languages
    
    if not is_valid:
        logger.warning(f"Invalid language code: {lang_code}. Supported: {supported_languages}")
    
    return is_valid

def validate_json_field(json_data: Any, schema_name: str = None) -> bool:
    """Valida que un campo JSON sea serializable con validación específica"""
    if json_data is None:
        return True
        
    try:
        # Test serializability
        json.dumps(json_data)
        
        # Schema-specific validation
        if schema_name:
            return _validate_json_schema(json_data, schema_name)
        
        return True
        
    except (TypeError, ValueError) as e:
        logger.error(f"JSON validation failed for {schema_name}: {e}")
        return False

def _validate_json_schema(data: Any, schema_name: str) -> bool:
    """Internal JSON schema validation"""
    try:
        if schema_name == 'quality_scores':
            return isinstance(data, dict) and all(
                isinstance(k, str) and isinstance(v, (int, float)) and 0 <= v <= 1
                for k, v in data.items()
            )
        elif schema_name == 'phases_completed':
            valid_phases = {phase.value for phase in ProcessingPhase}
            return isinstance(data, list) and all(
                isinstance(phase, str) and phase in valid_phases
                for phase in data
            )
        elif schema_name == 'alerts_detail':
            required_keys = {'type', 'severity', 'message', 'timestamp'}
            return isinstance(data, list) and all(
                isinstance(alert, dict) and required_keys.issubset(alert.keys())
                for alert in data
            )
        
        return True
        
    except Exception as e:
        logger.error(f"Schema validation failed for {schema_name}: {e}")
        return False

def get_model_metadata() -> Dict[str, Any]:
    """Obtener metadatos de todos los modelos"""
    return {
        'version': '2.2.0',
        'created_by': 'ClaudeAcademico Development Team - Enterprise Grade',
        'schema_description': 'Sistema de Traducción Académica - Database Schema',
        'supports_migrations': True,
        'requires_indexes': True,
        'enterprise_features': [
            'enum_integration',
            'comprehensive_validation',
            'enterprise_logging',
            'ml_features',
            'workflow_management',
            'performance_metrics'
        ],
        'models': {
            'SystemConfig': 'Enhanced configuration management with robust parsing',
            'BookProcessingHistory': 'Comprehensive book processing tracking',
            'ErrorPattern': 'ML-enhanced error pattern learning',
            'AuditLog': 'Detailed audit logging with performance metrics',
            'TerminologySuggestion': 'Advanced terminology management',
            'EditorialReview': 'Complete editorial workflow management',
            'UsageStatistic': 'Comprehensive system analytics'
        }
    }

# Enterprise metadata configuration
Base.metadata.info = get_model_metadata()

# SQLAlchemy event listeners for enterprise features
@event.listens_for(BookProcessingHistory, 'before_update')
def update_last_activity(mapper, connection, target):
    """Update last_activity_at on any book update"""
    target.last_activity_at = func.now()

@event.listens_for(ErrorPattern, 'after_update')
def log_pattern_update(mapper, connection, target):
    """Log pattern effectiveness updates"""
    if hasattr(target, '_sa_instance_state') and target._sa_instance_state.modified:
        logger.info(f"Pattern {target.id} updated: effectiveness={target.effectiveness_score:.3f}")

# Export enterprise functions
__all__ = [
    # Base classes
    'Base', 'TimestampMixin', 'TableNameMixin', 'ValidationMixin',
    
    # Models
    'SystemConfig', 'BookProcessingHistory', 'ErrorPattern', 'AuditLog',
    'TerminologySuggestion', 'EditorialReview', 'UsageStatistic',
    
    # Utilities
    'create_indexes', 'validate_language_code', 'validate_json_field',
    'get_model_metadata'
]