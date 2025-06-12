"""
Enumeraciones para el Sistema de Traducción Académica v2.2
Centraliza todos los valores enum usados en el sistema
"""

from enum import Enum, auto

class BookStatus(str, Enum):
    """Estados de procesamiento de libros"""
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    REVIEW = "review"

class ReviewSeverity(str, Enum):
    """Severidad de items de revisión editorial"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class PatternType(str, Enum):
    """Tipos de patrones de error"""
    SEMANTIC = "semantic"
    FORMAT = "format"
    FOOTNOTE = "footnote"
    PDF_ARTIFACT = "pdf_artifact"
    TERMINOLOGY = "terminology"
    STRUCTURAL = "structural"

class ProcessingPhase(str, Enum):
    """Fases del pipeline de procesamiento"""
    PDF_CLEANUP = "pdf_cleanup"
    HTML_CONVERSION = "html_conversion"
    TRANSLATION = "translation"
    SEMANTIC_VALIDATION = "semantic_validation"
    FOOTNOTE_RECONNECTION = "footnote_reconnection"
    EDITORIAL_REVIEW = "editorial_review"
    FINAL_VALIDATION = "final_validation"

class ConfigType(str, Enum):
    """Tipos de configuración del sistema"""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    JSON = "json"

class ReviewDecision(str, Enum):
    """Decisiones de revisión editorial"""
    PENDING = "pending"
    ACCEPT = "accept"
    REJECT = "reject"
    MODIFY = "modify"
    ESCALATE = "escalate"

class TerminologyPriority(str, Enum):
    """Prioridad de sugerencias terminológicas"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ErrorSeverity(str, Enum):
    """Severidad de errores del sistema"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"