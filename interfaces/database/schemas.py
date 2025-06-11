# ==========================================
# INTERFACES/DATABASE/SCHEMAS.PY
# Pydantic Schemas - Enterprise API Validation Models
# Sistema de Traducción Académica v2.2
# ==========================================

from pydantic import BaseModel, Field, validator, root_validator
from typing import Optional, List, Dict, Any, Union
from datetime import datetime, date
import json
import re

# Importar enums centralizados
from .enums import (
    BookStatus, ProcessingPhase, LanguageCode, Severity, Priority, 
    EditorDecision, PatternType, ConfigType, AlertType, 
    ReviewStatus, QualityLevel, DocumentType, AcademicDiscipline
)

# ==========================================
# BASE SCHEMAS CON LÍMITES ENTERPRISE
# ==========================================

class TimestampMixin(BaseModel):
    """Mixin para timestamps en todos los schemas"""
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

class PaginationParams(BaseModel):
    """Parámetros estándar de paginación con límites enterprise"""
    page: int = Field(1, ge=1, le=10000, description="Número de página")
    size: int = Field(20, ge=1, le=1000, description="Elementos por página")
    
    @property
    def offset(self) -> int:
        return (self.page - 1) * self.size

class PaginatedResponse(BaseModel):
    """Respuesta paginada estándar con límites definidos"""
    items: List[Any] = Field(..., max_items=1000, description="Items de la página actual")
    total: int = Field(..., ge=0, description="Total de elementos")
    page: int = Field(..., ge=1, description="Página actual")
    size: int = Field(..., ge=1, le=1000, description="Tamaño de página")
    pages: int = Field(..., ge=0, description="Total de páginas")
    
    @root_validator
    def validate_pagination_consistency(cls, values):
        """Validar consistencia de datos de paginación"""
        total = values.get('total', 0)
        page = values.get('page', 1)
        size = values.get('size', 20)
        items = values.get('items', [])
        
        # Calcular páginas totales
        pages = (total + size - 1) // size if total > 0 else 0
        values['pages'] = pages
        
        # Validar que los items no excedan el tamaño de página
        if len(items) > size:
            raise ValueError(f"Items count ({len(items)}) exceeds page size ({size})")
        
        # Validar que la página actual sea válida
        if page > pages and total > 0:
            raise ValueError(f"Page {page} exceeds total pages {pages}")
        
        return values

# ==========================================
# SYSTEM CONFIG SCHEMAS ENTERPRISE
# ==========================================

class SystemConfigBase(BaseModel):
    """Schema base para configuración del sistema con validaciones estrictas"""
    config_key: str = Field(
        ..., 
        max_length=100, 
        regex=r'^[a-zA-Z0-9._-]+$',
        description="Clave de configuración (alfanumérico, puntos, guiones)"
    )
    config_value: str = Field(..., max_length=10000, description="Valor de configuración")
    config_type: ConfigType = Field(ConfigType.STRING, description="Tipo de dato")
    description: Optional[str] = Field(None, max_length=500, description="Descripción de la configuración")
    is_active: bool = Field(True, description="Configuración activa")

class SystemConfigCreate(SystemConfigBase):
    """Schema para crear configuración con validación por tipo"""
    
    @validator('config_value')
    def validate_value_type(cls, v, values):
        """Valida que el valor sea compatible con el tipo especificado"""
        config_type = values.get('config_type')
        
        if config_type == ConfigType.INTEGER:
            try:
                int(v)
            except ValueError:
                raise ValueError("Valor debe ser un entero válido")
        elif config_type == ConfigType.FLOAT:
            try:
                float(v)
            except ValueError:
                raise ValueError("Valor debe ser un número decimal válido")
        elif config_type == ConfigType.BOOLEAN:
            if v.lower() not in ('true', 'false', '1', '0', 'yes', 'no', 'on', 'off'):
                raise ValueError("Valor booleano inválido. Use: true/false, 1/0, yes/no, on/off")
        elif config_type == ConfigType.JSON:
            try:
                json.loads(v)
            except json.JSONDecodeError as e:
                raise ValueError(f"JSON inválido: {e}")
        elif config_type == ConfigType.LIST:
            try:
                parsed = json.loads(v)
                if not isinstance(parsed, list):
                    raise ValueError("Valor debe ser una lista JSON válida")
            except json.JSONDecodeError:
                raise ValueError("Lista JSON inválida")
        elif config_type == ConfigType.DICT:
            try:
                parsed = json.loads(v)
                if not isinstance(parsed, dict):
                    raise ValueError("Valor debe ser un objeto JSON válido")
            except json.JSONDecodeError:
                raise ValueError("Objeto JSON inválido")
        
        return v

class SystemConfigUpdate(BaseModel):
    """Schema para actualizar configuración"""
    config_value: Optional[str] = Field(None, max_length=10000)
    description: Optional[str] = Field(None, max_length=500)
    is_active: Optional[bool] = None

class SystemConfigResponse(SystemConfigBase, TimestampMixin):
    """Schema de respuesta para configuración con valor parseado"""
    id: int
    parsed_value: Any = Field(..., description="Valor parseado según tipo")
    
    class Config:
        orm_mode = True

# ==========================================
# BOOK PROCESSING SCHEMAS ENTERPRISE
# ==========================================

class BookProcessingBase(BaseModel):
    """Schema base para procesamiento de libros con validaciones estrictas"""
    book_id: str = Field(
        ..., 
        max_length=100,
        regex=r'^[a-zA-Z0-9_-]+$',
        description="ID único del libro (alfanumérico, guiones, guiones bajos)"
    )
    title: Optional[str] = Field(None, max_length=500, description="Título del libro")
    source_lang: LanguageCode = Field(..., description="Idioma origen")
    target_lang: LanguageCode = Field(LanguageCode.SPANISH, description="Idioma destino")
    document_type: Optional[DocumentType] = Field(DocumentType.BOOK, description="Tipo de documento")
    academic_discipline: Optional[AcademicDiscipline] = Field(None, description="Disciplina académica")

class BookProcessingCreate(BookProcessingBase):
    """Schema para iniciar procesamiento de libro"""
    input_file_path: str = Field(..., max_length=1000, description="Ruta del archivo de entrada")
    
    @validator('input_file_path')
    def validate_file_path(cls, v):
        """Validar que sea un archivo PDF válido"""
        if not v.lower().endswith('.pdf'):
            raise ValueError("Solo se aceptan archivos PDF")
        return v
    
    @validator('book_id')
    def validate_book_id_format(cls, v):
        """Validar formato específico del book_id"""
        if len(v) < 3:
            raise ValueError("book_id debe tener al menos 3 caracteres")
        if len(v) > 100:
            raise ValueError("book_id no puede exceder 100 caracteres")
        if not re.match(r'^[a-zA-Z0-9_-]+$', v):
            raise ValueError("book_id solo puede contener letras, números, guiones y guiones bajos")
        return v

class BookProcessingUpdate(BaseModel):
    """Schema para actualizar estado de procesamiento con validaciones cruzadas"""
    status: Optional[BookStatus] = None
    current_phase: Optional[ProcessingPhase] = None
    progress_percentage: Optional[float] = Field(None, ge=0, le=100)
    phases_completed: Optional[List[ProcessingPhase]] = Field(None, max_items=10)
    quality_scores: Optional[Dict[str, float]] = None
    error_count: Optional[int] = Field(None, ge=0, le=1000)
    processing_time_seconds: Optional[int] = Field(None, ge=0)
    output_file_path: Optional[str] = Field(None, max_length=1000)
    total_paragraphs: Optional[int] = Field(None, ge=0, le=100000)
    total_footnotes: Optional[int] = Field(None, ge=0, le=10000)
    semantic_score_avg: Optional[float] = Field(None, ge=0, le=1)
    format_preservation_score: Optional[float] = Field(None, ge=0, le=1)
    footnote_preservation_score: Optional[float] = Field(None, ge=0, le=1)
    
    @validator('quality_scores')
    def validate_quality_scores_values(cls, v):
        """Validar que todos los scores estén en rango válido"""
        if v:
            for key, score in v.items():
                if not isinstance(score, (int, float)):
                    raise ValueError(f"Score '{key}' debe ser numérico")
                if not 0 <= score <= 1:
                    raise ValueError(f"Score '{key}' debe estar entre 0 y 1, recibido: {score}")
                if len(key) > 50:
                    raise ValueError(f"Nombre de score muy largo: {key}")
            
            # Limitar número de scores
            if len(v) > 20:
                raise ValueError("Máximo 20 quality scores permitidos")
        
        return v
    
    @root_validator
    def validate_processing_consistency(cls, values):
        """CRÍTICO: Validar consistencia entre campos relacionados"""
        status = values.get('status')
        progress = values.get('progress_percentage')
        current_phase = values.get('current_phase')
        phases_completed = values.get('phases_completed', [])
        quality_scores = values.get('quality_scores', {})
        
        # Validar consistencia status-progress
        if status == BookStatus.COMPLETED and progress is not None and progress < 100:
            raise ValueError("Libro completado debe tener progreso 100%")
        
        if status == BookStatus.FAILED and progress is not None and progress > 95:
            raise ValueError("Libro fallido no puede tener progreso superior al 95%")
        
        # Validar consistencia de fases
        if current_phase and phases_completed:
            phase_order = [
                ProcessingPhase.PDF_CLEANUP,
                ProcessingPhase.HTML_CONVERSION,
                ProcessingPhase.TRANSLATION,
                ProcessingPhase.VALIDATION,
                ProcessingPhase.FOOTNOTE_RECONNECTION,
                ProcessingPhase.FINAL_AUDIT
            ]
            
            current_index = phase_order.index(current_phase) if current_phase in phase_order else -1
            
            for completed_phase in phases_completed:
                if completed_phase in phase_order:
                    completed_index = phase_order.index(completed_phase)
                    if completed_index > current_index:
                        raise ValueError(f"Fase completada '{completed_phase}' no puede estar después de fase actual '{current_phase}'")
        
        # Validar scores con contexto
        semantic_score = values.get('semantic_score_avg')
        format_score = values.get('format_preservation_score')
        footnote_score = values.get('footnote_preservation_score')
        total_paragraphs = values.get('total_paragraphs')
        total_footnotes = values.get('total_footnotes')
        
        # Si hay scores de preservación, debe haber totales correspondientes
        if format_score is not None and total_paragraphs is None:
            raise ValueError("format_preservation_score requiere total_paragraphs")
        
        if footnote_score is not None and total_footnotes is None:
            raise ValueError("footnote_preservation_score requiere total_footnotes")
        
        # Si el score es muy bajo, debería haber conteo de errores
        error_count = values.get('error_count', 0)
        if semantic_score is not None and semantic_score < 0.7 and error_count == 0:
            raise ValueError("Score semántico bajo (<0.7) debería tener error_count > 0")
        
        return values

class BookProcessingResponse(BookProcessingBase, TimestampMixin):
    """Schema de respuesta para procesamiento de libros"""
    id: int
    status: BookStatus
    current_phase: Optional[ProcessingPhase] = None
    progress_percentage: float = 0.0
    started_at: datetime
    completed_at: Optional[datetime] = None
    total_phases: int = 6
    phases_completed: Optional[List[ProcessingPhase]] = Field(None, max_items=10)
    quality_scores: Optional[Dict[str, float]] = None
    error_count: int = 0
    processing_time_seconds: Optional[int] = None
    input_file_path: Optional[str] = None
    output_file_path: Optional[str] = None
    total_paragraphs: Optional[int] = None
    total_footnotes: Optional[int] = None
    semantic_score_avg: Optional[float] = None
    format_preservation_score: Optional[float] = None
    footnote_preservation_score: Optional[float] = None
    duration_minutes: Optional[float] = None
    is_completed: bool = False
    
    class Config:
        orm_mode = True

class BookStatusSummary(BaseModel):
    """Resumen de estado de libro para dashboard"""
    book_id: str
    title: Optional[str]
    status: BookStatus
    progress_percentage: float
    current_phase: Optional[ProcessingPhase]
    started_at: datetime
    estimated_completion: Optional[datetime]
    quality_score: Optional[float]
    error_count: int

# ==========================================
# ERROR PATTERN SCHEMAS ENTERPRISE
# ==========================================

class ErrorPatternBase(BaseModel):
    """Schema base para patrones de error con validaciones mejoradas"""
    pattern_type: PatternType = Field(..., description="Tipo de patrón")
    pattern_content: str = Field(..., min_length=1, max_length=2000, description="Contenido del patrón")
    pattern_regex: Optional[str] = Field(None, max_length=1000, description="Expresión regular del patrón")
    description: Optional[str] = Field(None, max_length=1000, description="Descripción del patrón")
    
    @validator('pattern_regex')
    def validate_regex_syntax(cls, v):
        """Validar que la expresión regular sea válida"""
        if v:
            try:
                re.compile(v)
            except re.error as e:
                raise ValueError(f"Expresión regular inválida: {e}")
        return v

class ErrorPatternCreate(ErrorPatternBase):
    """Schema para crear patrón de error"""
    created_by: str = Field("system", max_length=100, description="Creado por")

class ErrorPatternUpdate(BaseModel):
    """Schema para actualizar patrón de error"""
    pattern_content: Optional[str] = Field(None, min_length=1, max_length=2000)
    pattern_regex: Optional[str] = Field(None, max_length=1000)
    description: Optional[str] = Field(None, max_length=1000)
    is_active: Optional[bool] = None

class ErrorPatternResponse(ErrorPatternBase, TimestampMixin):
    """Schema de respuesta para patrón de error"""
    id: int
    frequency: int = 1
    success_rate: float = 0.0
    effectiveness_score: float = 0.0
    false_positive_rate: float = 0.0
    first_seen: datetime
    last_seen: datetime
    usage_count: int = 0
    created_by: str = "system"
    is_active: bool = True
    
    class Config:
        orm_mode = True

class ErrorPatternStats(BaseModel):
    """Estadísticas de patrón de error"""
    pattern_id: int
    pattern_type: PatternType
    effectiveness_score: float = Field(..., ge=0, le=1)
    usage_count: int = Field(..., ge=0)
    success_rate: float = Field(..., ge=0, le=1)
    recommendation: str = Field(..., max_length=500)

# ==========================================
# AUDIT LOG SCHEMAS ENTERPRISE
# ==========================================

class AlertDetail(BaseModel):
    """Detalle de alerta en auditoría con validaciones estrictas"""
    alert_type: AlertType = Field(..., description="Tipo de alerta")
    severity: Severity = Field(..., description="Severidad de la alerta")
    message: str = Field(..., min_length=1, max_length=1000, description="Mensaje de la alerta")
    location: Optional[str] = Field(None, max_length=200, description="Ubicación del problema")
    suggested_action: Optional[str] = Field(None, max_length=500, description="Acción sugerida")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Metadata adicional")

class AuditLogBase(BaseModel):
    """Schema base para log de auditoría"""
    book_id: str = Field(..., max_length=100, description="ID del libro auditado")
    phase_name: ProcessingPhase = Field(..., description="Fase auditada")
    quality_score: float = Field(..., ge=0, le=1, description="Score de calidad general")
    integrity_score: Optional[float] = Field(None, ge=0, le=1, description="Score de integridad")
    format_preservation_score: Optional[float] = Field(None, ge=0, le=1, description="Score de preservación de formato")
    footnote_preservation_score: Optional[float] = Field(None, ge=0, le=1, description="Score de preservación de footnotes")

class AuditLogCreate(AuditLogBase):
    """Schema para crear log de auditoría con validación cruzada crítica"""
    processing_time_seconds: Optional[int] = Field(None, ge=0, le=86400)  # máx 24h
    memory_usage_mb: Optional[float] = Field(None, ge=0, le=32000)  # máx 32GB
    alerts_detail: Optional[List[AlertDetail]] = Field(None, max_items=100, description="Detalles de alertas")
    metrics_detail: Optional[Dict[str, Any]] = None
    improvements_applied: Optional[List[str]] = Field(None, max_items=50, description="Mejoras aplicadas")
    system_version: Optional[str] = Field("2.2.0", max_length=20)
    api_versions: Optional[Dict[str, str]] = None
    
    @root_validator
    def validate_quality_alerts_consistency(cls, values):
        """CRÍTICO: Validar coherencia entre quality_score y alerts_detail"""
        quality_score = values.get('quality_score')
        alerts_detail = values.get('alerts_detail', [])
        
        if quality_score is not None:
            # Contar alertas por severidad
            critical_alerts = sum(1 for alert in alerts_detail if alert.severity == Severity.CRITICAL)
            high_alerts = sum(1 for alert in alerts_detail if alert.severity == Severity.HIGH)
            
            # Si el quality_score es bajo, DEBE haber alertas críticas
            if quality_score < 0.5 and critical_alerts == 0:
                raise ValueError(
                    f"Quality score muy bajo ({quality_score:.2f}) requiere al menos una alerta crítica"
                )
            
            # Si el quality_score es medio-bajo, debería haber alertas de alta severidad
            if quality_score < 0.7 and critical_alerts == 0 and high_alerts == 0:
                raise ValueError(
                    f"Quality score bajo-medio ({quality_score:.2f}) requiere al menos una alerta de alta severidad"
                )
            
            # Si el quality_score es alto, no debería haber muchas alertas críticas
            if quality_score > 0.9 and critical_alerts > 2:
                raise ValueError(
                    f"Quality score alto ({quality_score:.2f}) no debería tener {critical_alerts} alertas críticas"
                )
            
            # Si el quality_score es perfecto, no debería haber alertas críticas
            if quality_score >= 0.95 and critical_alerts > 0:
                raise ValueError(
                    f"Quality score excelente ({quality_score:.2f}) no debería tener alertas críticas"
                )
        
        # Validar coherencia entre scores relacionados
        integrity_score = values.get('integrity_score')
        format_score = values.get('format_preservation_score')
        footnote_score = values.get('footnote_preservation_score')
        
        if integrity_score is not None and quality_score is not None:
            # El integrity_score no debería ser mucho mayor que quality_score
            if integrity_score > quality_score + 0.3:
                raise ValueError(
                    f"Integrity score ({integrity_score:.2f}) muy alto comparado con quality score ({quality_score:.2f})"
                )
        
        # Validar alerts_detail structure
        if alerts_detail:
            for i, alert in enumerate(alerts_detail):
                if not isinstance(alert, AlertDetail):
                    # Si es dict, validar campos mínimos
                    if isinstance(alert, dict):
                        required_fields = ['alert_type', 'severity', 'message']
                        missing = [f for f in required_fields if f not in alert]
                        if missing:
                            raise ValueError(f"Alert {i} missing required fields: {missing}")
        
        return values

class AuditLogResponse(AuditLogBase, TimestampMixin):
    """Schema de respuesta para log de auditoría"""
    id: int
    processing_time_seconds: Optional[int] = None
    memory_usage_mb: Optional[float] = None
    alerts_count: int = 0
    alerts_detail: Optional[List[AlertDetail]] = Field(None, max_items=100)
    metrics_detail: Optional[Dict[str, Any]] = None
    improvements_applied: Optional[List[str]] = Field(None, max_items=50)
    system_version: Optional[str] = None
    api_versions: Optional[Dict[str, str]] = None
    has_critical_alerts: bool = False
    
    class Config:
        orm_mode = True

# ==========================================
# TERMINOLOGY SUGGESTION SCHEMAS ENTERPRISE
# ==========================================

class TerminologySuggestionBase(BaseModel):
    """Schema base para sugerencia terminológica con validaciones mejoradas"""
    glossary_id: str = Field(..., max_length=100, description="ID del glosario")
    source_term: str = Field(..., min_length=1, max_length=200, description="Término origen")
    target_term: str = Field(..., min_length=1, max_length=200, description="Término traducido")
    context: Optional[str] = Field(None, max_length=2000, description="Contexto del término")
    justification: Optional[str] = Field(None, max_length=1000, description="Justificación de la traducción")

class TerminologySuggestionCreate(TerminologySuggestionBase):
    """Schema para crear sugerencia terminológica"""
    book_id: str = Field(..., max_length=100, description="ID del libro")
    confidence_score: Optional[float] = Field(None, ge=0, le=1)
    priority: Priority = Field(Priority.MEDIUM, description="Prioridad de la sugerencia")
    frequency_estimate: str = Field("unknown", max_length=50, description="Estimación de frecuencia")
    suggested_by: str = Field("claude", max_length=50, description="Sugerido por")

class TerminologySuggestionUpdate(BaseModel):
    """Schema para actualizar sugerencia terminológica"""
    applied: Optional[bool] = None
    reviewed: Optional[bool] = None
    approved: Optional[bool] = None
    editor_feedback: Optional[str] = Field(None, max_length=1000)
    rejection_reason: Optional[str] = Field(None, max_length=500)

class TerminologySuggestionResponse(TerminologySuggestionBase, TimestampMixin):
    """Schema de respuesta para sugerencia terminológica"""
    id: int
    book_id: str
    confidence_score: Optional[float] = None
    priority: Priority = Priority.MEDIUM
    frequency_estimate: str = "unknown"
    suggested_by: str = "claude"
    applied: bool = False
    reviewed: bool = False
    approved: bool = False
    editor_feedback: Optional[str] = None
    rejection_reason: Optional[str] = None
    
    class Config:
        orm_mode = True

class TerminologyBatch(BaseModel):
    """Schema para procesamiento en lote de términos con límites enterprise"""
    suggestions: List[TerminologySuggestionCreate] = Field(..., min_items=1, max_items=100)
    batch_id: str = Field(..., max_length=100, description="ID del lote")
    
    @validator('suggestions')
    def validate_suggestions_consistency(cls, v):
        """Validar consistencia del lote de sugerencias"""
        if not v:
            raise ValueError("Lista de sugerencias no puede estar vacía")
        
        # Verificar que no haya términos duplicados en el lote
        seen_terms = set()
        for suggestion in v:
            term_key = (suggestion.source_term.lower(), suggestion.target_term.lower())
            if term_key in seen_terms:
                raise ValueError(f"Término duplicado en lote: {suggestion.source_term} -> {suggestion.target_term}")
            seen_terms.add(term_key)
        
        return v

# ==========================================
# EDITORIAL REVIEW SCHEMAS ENTERPRISE
# ==========================================

class EditorialReviewBase(BaseModel):
    """Schema base para revisión editorial con validaciones mejoradas"""
    item_id: str = Field(..., max_length=100, description="ID único del item")
    section_number: Optional[int] = Field(None, ge=1, le=10000, description="Número de sección")
    location_info: Optional[str] = Field(None, max_length=200, description="Información de ubicación")
    alert_type: AlertType = Field(..., description="Tipo de alerta")
    severity: Severity = Field(..., description="Severidad del problema")
    similarity_score: Optional[float] = Field(None, ge=0, le=1, description="Score de similitud")
    suggested_action: Optional[str] = Field(None, max_length=500, description="Acción sugerida")

class EditorialReviewCreate(EditorialReviewBase):
    """Schema para crear revisión editorial"""
    book_id: str = Field(..., max_length=100, description="ID del libro")
    original_text: Optional[str] = Field(None, max_length=5000, description="Texto original")
    translated_text: Optional[str] = Field(None, max_length=5000, description="Texto traducido")

class EditorialReviewUpdate(BaseModel):
    """Schema para actualizar revisión editorial"""
    editor_decision: Optional[EditorDecision] = None
    editor_notes: Optional[str] = Field(None, max_length=2000)
    corrected_text: Optional[str] = Field(None, max_length=5000)
    reviewer_id: Optional[str] = Field(None, max_length=100)

class EditorialReviewResponse(EditorialReviewBase, TimestampMixin):
    """Schema de respuesta para revisión editorial"""
    id: int
    book_id: str
    original_text: Optional[str] = None
    translated_text: Optional[str] = None
    editor_decision: Optional[EditorDecision] = None
    editor_notes: Optional[str] = None
    corrected_text: Optional[str] = None
    reviewed: bool = False
    resolved: bool = False
    resolution_time_minutes: Optional[int] = None
    reviewed_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    reviewer_id: Optional[str] = None
    review_session_id: Optional[str] = None
    
    class Config:
        orm_mode = True

class ReviewSummary(BaseModel):
    """Resumen de revisión para dashboard con métricas enterprise"""
    book_id: str
    total_items: int = Field(..., ge=0)
    critical_items: int = Field(..., ge=0)
    high_priority_items: int = Field(..., ge=0)
    reviewed_items: int = Field(..., ge=0)
    resolved_items: int = Field(..., ge=0)
    pending_items: int = Field(..., ge=0)
    average_score: Optional[float] = Field(None, ge=0, le=1)
    estimated_time_remaining: Optional[int] = Field(None, ge=0, description="Minutos estimados")
    
    @root_validator
    def validate_summary_consistency(cls, values):
        """Validar consistencia de números en resumen"""
        total = values.get('total_items', 0)
        critical = values.get('critical_items', 0)
        high_priority = values.get('high_priority_items', 0)
        reviewed = values.get('reviewed_items', 0)
        resolved = values.get('resolved_items', 0)
        pending = values.get('pending_items', 0)
        
        # Validar que los números sean consistentes
        if critical + high_priority > total:
            raise ValueError("Suma de items críticos y alta prioridad no puede exceder total")
        
        if reviewed + pending != total:
            raise ValueError("Suma de items revisados y pendientes debe igualar total")
        
        if resolved > reviewed:
            raise ValueError("Items resueltos no puede exceder items revisados")
        
        return values

# ==========================================
# USAGE STATISTICS SCHEMAS ENTERPRISE
# ==========================================

class UsageStatisticBase(BaseModel):
    """Schema base para estadísticas de uso con validaciones enterprise"""
    date: date = Field(..., description="Fecha de las estadísticas")

class UsageStatisticCreate(UsageStatisticBase):
    """Schema para crear estadísticas de uso con validaciones de rango"""
    books_processed: int = Field(0, ge=0, le=10000)
    books_completed: int = Field(0, ge=0, le=10000)
    books_failed: int = Field(0, ge=0, le=10000)
    total_processing_time_hours: float = Field(0.0, ge=0, le=100000)
    average_processing_time_minutes: float = Field(0.0, ge=0, le=10080)  # máx 1 semana
    average_quality_score: float = Field(0.0, ge=0, le=1)
    average_semantic_score: float = Field(0.0, ge=0, le=1)
    average_format_preservation: float = Field(0.0, ge=0, le=1)
    total_errors: int = Field(0, ge=0, le=100000)
    errors_resolved: int = Field(0, ge=0, le=100000)
    critical_errors: int = Field(0, ge=0, le=10000)
    api_calls_deepl: int = Field(0, ge=0, le=1000000)
    api_calls_claude: int = Field(0, ge=0, le=1000000)
    api_calls_abbyy: int = Field(0, ge=0, le=1000000)
    total_api_cost: float = Field(0.0, ge=0, le=100000)
    reviews_generated: int = Field(0, ge=0, le=100000)
    reviews_completed: int = Field(0, ge=0, le=100000)
    average_review_time_minutes: float = Field(0.0, ge=0, le=1440)  # máx 24h
    
    @root_validator
    def validate_statistics_consistency(cls, values):
        """Validar consistencia de estadísticas de uso"""
        processed = values.get('books_processed', 0)
        completed = values.get('books_completed', 0)
        failed = values.get('books_failed', 0)
        total_errors = values.get('total_errors', 0)
        errors_resolved = values.get('errors_resolved', 0)
        reviews_generated = values.get('reviews_generated', 0)
        reviews_completed = values.get('reviews_completed', 0)
        
        # Validar consistencia de libros
        if completed + failed > processed:
            raise ValueError("Suma de libros completados y fallidos no puede exceder procesados")
        
        # Validar consistencia de errores
        if errors_resolved > total_errors:
            raise ValueError("Errores resueltos no puede exceder total de errores")
        
        # Validar consistencia de revisiones
        if reviews_completed > reviews_generated:
            raise ValueError("Revisiones completadas no puede exceder revisiones generadas")
        
        return values

class UsageStatisticResponse(UsageStatisticCreate):
    """Schema de respuesta para estadísticas de uso"""
    id: int
    success_rate: float = Field(..., ge=0, le=1, description="Tasa de éxito en procesamiento")
    error_resolution_rate: float = Field(..., ge=0, le=1, description="Tasa de resolución de errores")
    
    class Config:
        orm_mode = True

class DashboardMetrics(BaseModel):
    """Métricas principales para dashboard con validaciones enterprise"""
    total_books_processed: int = Field(..., ge=0)
    books_in_progress: int = Field(..., ge=0)
    books_completed_today: int = Field(..., ge=0)
    active_errors: int = Field(..., ge=0)
    average_quality_score: float = Field(..., ge=0, le=1)
    quality_trend: str = Field(..., regex="^(up|down|stable)$")
    average_processing_time: float = Field(..., ge=0)
    system_load: float = Field(..., ge=0, le=1)
    daily_api_cost: float = Field(..., ge=0)
    monthly_api_cost: float = Field(..., ge=0)
    critical_alerts: int = Field(..., ge=0)
    system_health: str = Field(..., regex="^(healthy|warning|critical)$")

class SystemHealthCheck(BaseModel):
    """Schema para health check del sistema con límites definidos"""
    status: str = Field(..., regex="^(healthy|degraded|unhealthy)$")
    timestamp: datetime
    services: Dict[str, str] = Field(..., max_items=50, description="Estado de servicios")
    metrics: Dict[str, float] = Field(..., max_items=100, description="Métricas del sistema")
    alerts: List[str] = Field(..., max_items=100, description="Alertas activas")
    uptime_seconds: float = Field(..., ge=0)

# ==========================================
# QUERY SCHEMAS PARA FILTERING ENTERPRISE
# ==========================================

class BookQueryParams(BaseModel):
    """Parámetros de consulta para libros con validaciones"""
    status: Optional[BookStatus] = None
    source_lang: Optional[LanguageCode] = None
    date_from: Optional[date] = None
    date_to: Optional[date] = None
    min_quality_score: Optional[float] = Field(None, ge=0, le=1)
    has_errors: Optional[bool] = None
    search: Optional[str] = Field(None, max_length=200)
    
    @root_validator
    def validate_date_range(cls, values):
        """Validar rango de fechas"""
        date_from = values.get('date_from')
        date_to = values.get('date_to')
        
        if date_from and date_to and date_from > date_to:
            raise ValueError("date_from no puede ser posterior a date_to")
        
        return values

class ReviewQueryParams(BaseModel):
    """Parámetros de consulta para revisiones con validaciones"""
    book_id: Optional[str] = Field(None, max_length=100)
    severity: Optional[Severity] = None
    reviewed: Optional[bool] = None
    resolved: Optional[bool] = None
    reviewer_id: Optional[str] = Field(None, max_length=100)
    date_from: Optional[date] = None
    date_to: Optional[date] = None
    
    @root_validator
    def validate_review_query_consistency(cls, values):
        """Validar consistencia de parámetros de consulta"""
        date_from = values.get('date_from')
        date_to = values.get('date_to')
        
        if date_from and date_to and date_from > date_to:
            raise ValueError("date_from no puede ser posterior a date_to")
        
        return values

class StatisticsQueryParams(BaseModel):
    """Parámetros de consulta para estadísticas con validaciones"""
    date_from: Optional[date] = None
    date_to: Optional[date] = None
    group_by: Optional[str] = Field(None, regex="^(day|week|month)$")
    
    @root_validator
    def validate_statistics_query(cls, values):
        """Validar parámetros de consulta de estadísticas"""
        date_from = values.get('date_from')
        date_to = values.get('date_to')
        
        if date_from and date_to:
            if date_from > date_to:
                raise ValueError("date_from no puede ser posterior a date_to")
            
            # Limitar rango máximo a 2 años
            delta = date_to - date_from
            if delta.days > 730:
                raise ValueError("Rango de fechas no puede exceder 2 años")
        
        return values

# ==========================================
# BULK OPERATIONS SCHEMAS ENTERPRISE
# ==========================================

class BulkUpdateRequest(BaseModel):
    """Schema para operaciones bulk con validaciones enterprise"""
    ids: List[int] = Field(..., min_items=1, max_items=1000, description="Lista de IDs a actualizar")
    updates: Dict[str, Any] = Field(..., min_items=1, max_items=20, description="Campos a actualizar")
    
    @validator('ids')
    def validate_ids_unique_and_positive(cls, v):
        """Validar que los IDs sean únicos y positivos"""
        if len(set(v)) != len(v):
            raise ValueError("IDs duplicados no permitidos")
        
        if any(id_val <= 0 for id_val in v):
            raise ValueError("Todos los IDs deben ser positivos")
        
        return v
    
    @validator('updates')
    def validate_updates_format(cls, v):
        """Validar formato de actualizaciones"""
        # No permitir actualizaciones de campos críticos
        forbidden_fields = {'id', 'created_at', 'book_id'}
        forbidden_in_updates = [field for field in v.keys() if field in forbidden_fields]
        
        if forbidden_in_updates:
            raise ValueError(f"No se pueden actualizar campos críticos: {forbidden_in_updates}")
        
        return v

class BulkUpdateResponse(BaseModel):
    """Respuesta de operación bulk con métricas detalladas"""
    updated_count: int = Field(..., ge=0)
    failed_count: int = Field(..., ge=0)
    errors: List[str] = Field(..., max_items=1000)
    execution_time_ms: float = Field(..., ge=0)
    success_rate: float = Field(..., ge=0, le=1)
    
    @root_validator
    def calculate_success_rate(cls, values):
        """Calcular tasa de éxito automáticamente"""
        updated = values.get('updated_count', 0)
        failed = values.get('failed_count', 0)
        total = updated + failed
        
        if total > 0:
            values['success_rate'] = updated / total
        else:
            values['success_rate'] = 0.0
        
        return values

# ==========================================
# API RESPONSE WRAPPERS ENTERPRISE
# ==========================================

class APIResponse(BaseModel):
    """Wrapper estándar para respuestas de API con metadatos enterprise"""
    success: bool = True
    message: str = "Operation completed successfully"
    data: Optional[Any] = None
    errors: Optional[List[str]] = Field(None, max_items=100)
    timestamp: datetime = Field(default_factory=datetime.now)
    request_id: Optional[str] = Field(None, description="ID único de la request")
    execution_time_ms: Optional[float] = Field(None, ge=0)
    
    @validator('message')
    def validate_message_length(cls, v):
        """Validar longitud del mensaje"""
        if len(v) > 500:
            raise ValueError("Mensaje de respuesta muy largo (máx 500 caracteres)")
        return v

class ErrorResponse(BaseModel):
    """Respuesta de error estándar con información detallada"""
    success: bool = False
    message: str = Field(..., max_length=500)
    error_code: Optional[str] = Field(None, max_length=50)
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    request_id: Optional[str] = None
    
    @validator('error_code')
    def validate_error_code_format(cls, v):
        """Validar formato del código de error"""
        if v and not re.match(r'^[A-Z0-9_]+$', v):
            raise ValueError("Código de error debe ser alfanumérico en mayúsculas con guiones bajos")
        return v