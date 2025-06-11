 # ==========================================
# INTERFACES/FASTAPI_BACKEND/API/DEPENDENCIES.PY
# Shared Dependencies - Enterprise Grade
# Sistema de Traducción Académica v2.2
# ==========================================

from fastapi import Depends, Query, Request, HTTPException, UploadFile, File, status
from fastapi.security import HTTPBearer
from sqlalchemy.orm import Session
from typing import Optional, List, Dict, Any, Union, Callable, Generator
from pydantic import BaseModel, Field, validator
from datetime import datetime, date
from enum import Enum
import structlog
import uuid
from pathlib import Path
import mimetypes

# Import our modules
from ..core.config import get_settings, Settings
from ..core.security import (
    get_current_user, require_authenticated_user, User, UserRole, PermissionLevel
)
from ..utils.responses import ResponseBuilder, create_pagination_info, PaginationInfo
from ...database.database import get_db, get_database_manager, DatabaseManager
from ...database.models import BookProcessingHistory, BookStatus, ProcessingPhase

logger = structlog.get_logger(__name__)

# ==========================================
# QUERY PARAMETER MODELS
# ==========================================

class SortOrder(str, Enum):
    ASC = "asc"
    DESC = "desc"

class BookFilterParams(BaseModel):
    """Parámetros de filtro para libros"""
    status: Optional[BookStatus] = Field(None, description="Filtrar por estado")
    source_lang: Optional[str] = Field(None, description="Filtrar por idioma origen")
    target_lang: Optional[str] = Field(None, description="Filtrar por idioma destino")
    date_from: Optional[date] = Field(None, description="Fecha desde (YYYY-MM-DD)")
    date_to: Optional[date] = Field(None, description="Fecha hasta (YYYY-MM-DD)")
    search: Optional[str] = Field(None, description="Búsqueda en título o book_id")
    has_errors: Optional[bool] = Field(None, description="Solo libros con errores")
    priority_min: Optional[int] = Field(None, ge=1, le=10, description="Prioridad mínima")
    priority_max: Optional[int] = Field(None, ge=1, le=10, description="Prioridad máxima")

    @validator('source_lang', 'target_lang')
    def validate_language_codes(cls, v):
        if v is not None:
            valid_langs = {'de', 'en', 'fr', 'it', 'nl', 'es'}
            if v not in valid_langs:
                raise ValueError(f'Invalid language code. Must be one of: {valid_langs}')
        return v

class PaginationParams(BaseModel):
    """Parámetros de paginación estándar"""
    page: int = Field(1, ge=1, description="Número de página")
    page_size: int = Field(20, ge=1, le=100, description="Elementos por página")
    
    @property
    def offset(self) -> int:
        return (self.page - 1) * self.page_size
    
    @property
    def limit(self) -> int:
        return self.page_size

class SortParams(BaseModel):
    """Parámetros de ordenamiento"""
    sort_by: str = Field("created_at", description="Campo por el cual ordenar")
    sort_order: SortOrder = Field(SortOrder.DESC, description="Orden ascendente o descendente")

class DateRangeParams(BaseModel):
    """Parámetros de rango de fechas"""
    date_from: Optional[datetime] = Field(None, description="Fecha desde")
    date_to: Optional[datetime] = Field(None, description="Fecha hasta")
    
    @validator('date_to')
    def validate_date_range(cls, v, values):
        if v is not None and 'date_from' in values and values['date_from'] is not None:
            if v < values['date_from']:
                raise ValueError('date_to must be after date_from')
        return v

# ==========================================
# FILE UPLOAD VALIDATION
# ==========================================

class FileValidationConfig(BaseModel):
    """Configuración para validación de archivos"""
    max_size_mb: int = Field(50, description="Tamaño máximo en MB")
    allowed_extensions: List[str] = Field(['.pdf', '.docx'], description="Extensiones permitidas")
    allowed_mime_types: List[str] = Field([
        'application/pdf',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
    ], description="Tipos MIME permitidos")

class ValidatedFile(BaseModel):
    """Archivo validado listo para procesamiento"""
    file: UploadFile
    filename: str
    size_mb: float
    extension: str
    mime_type: str
    safe_filename: str  # Sanitized filename

async def validate_upload_file(
    file: UploadFile = File(...),
    settings: Settings = Depends(get_settings)
) -> ValidatedFile:
    """
    Dependency para validar archivos subidos
    
    Valida tamaño, extensión, tipo MIME y sanitiza el nombre
    """
    
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No filename provided"
        )
    
    # Read file size
    content = await file.read()
    file_size_mb = len(content) / (1024 * 1024)
    
    # Reset file pointer
    await file.seek(0)
    
    # Validate file size
    if file_size_mb > settings.max_file_size_mb:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Maximum size: {settings.max_file_size_mb}MB"
        )
    
    # Get file extension
    file_path = Path(file.filename)
    extension = file_path.suffix.lower()
    
    # Validate extension
    if extension not in settings.allowed_file_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file extension. Allowed: {settings.allowed_file_extensions}"
        )
    
    # Validate MIME type
    mime_type, _ = mimetypes.guess_type(file.filename)
    expected_mime_types = {
        '.pdf': 'application/pdf',
        '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
    }
    
    expected_mime = expected_mime_types.get(extension)
    if expected_mime and mime_type != expected_mime:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type. Expected {expected_mime}, got {mime_type}"
        )
    
    # Sanitize filename
    safe_filename = sanitize_filename(file.filename)
    
    logger.info(
        "File upload validated",
        filename=file.filename,
        safe_filename=safe_filename,
        size_mb=file_size_mb,
        extension=extension,
        mime_type=mime_type
    )
    
    return ValidatedFile(
        file=file,
        filename=file.filename,
        size_mb=file_size_mb,
        extension=extension,
        mime_type=mime_type or "unknown",
        safe_filename=safe_filename
    )

def sanitize_filename(filename: str) -> str:
    """Sanitizar nombre de archivo para seguridad"""
    import re
    
    # Remove path separators and dangerous characters
    safe_name = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # Remove multiple underscores
    safe_name = re.sub(r'_{2,}', '_', safe_name)
    
    # Limit length
    if len(safe_name) > 200:
        name_part = safe_name[:180]
        ext_part = Path(safe_name).suffix
        safe_name = name_part + ext_part
    
    return safe_name

# ==========================================
# DATABASE DEPENDENCIES
# ==========================================

def get_db_session() -> Generator[Session, None, None]:
    """
    Dependency para obtener sesión de base de datos
    
    Wrapper sobre get_db para compatibilidad
    """
    yield from get_db()

def get_db_manager() -> DatabaseManager:
    """Dependency para obtener database manager"""
    return get_database_manager()

async def get_db_with_user_context(
    db: Session = Depends(get_db_session),
    current_user: Optional[User] = Depends(get_current_user)
) -> Session:
    """
    Database session con contexto de usuario para auditoría
    
    Agrega información del usuario actual a la sesión para logging
    """
    
    if current_user:
        # Add user context to session for audit logging
        db.info = db.info or {}
        db.info.update({
            'current_user': current_user.username,
            'user_role': current_user.role.value,
            'request_timestamp': datetime.now().isoformat()
        })
    
    return db

# ==========================================
# QUERY PARAMETER DEPENDENCIES
# ==========================================

def get_pagination_params(
    page: int = Query(1, ge=1, description="Número de página"),
    page_size: int = Query(20, ge=1, le=100, description="Elementos por página")
) -> PaginationParams:
    """Dependency para parámetros de paginación"""
    return PaginationParams(page=page, page_size=page_size)

def get_book_filter_params(
    status: Optional[str] = Query(None, description="Filtrar por estado del libro"),
    source_lang: Optional[str] = Query(None, description="Idioma origen"),
    target_lang: Optional[str] = Query(None, description="Idioma destino"),
    date_from: Optional[date] = Query(None, description="Fecha desde"),
    date_to: Optional[date] = Query(None, description="Fecha hasta"),
    search: Optional[str] = Query(None, description="Búsqueda en título"),
    has_errors: Optional[bool] = Query(None, description="Solo libros con errores"),
    priority_min: Optional[int] = Query(None, ge=1, le=10, description="Prioridad mínima"),
    priority_max: Optional[int] = Query(None, ge=1, le=10, description="Prioridad máxima")
) -> BookFilterParams:
    """Dependency para parámetros de filtro de libros"""
    
    # Convert status string to enum if provided
    status_enum = None
    if status:
        try:
            status_enum = BookStatus(status.lower())
        except ValueError:
            valid_statuses = [s.value for s in BookStatus]
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid status. Valid options: {valid_statuses}"
            )
    
    return BookFilterParams(
        status=status_enum,
        source_lang=source_lang,
        target_lang=target_lang,
        date_from=date_from,
        date_to=date_to,
        search=search,
        has_errors=has_errors,
        priority_min=priority_min,
        priority_max=priority_max
    )

def get_sort_params(
    sort_by: str = Query("created_at", description="Campo para ordenar"),
    sort_order: str = Query("desc", description="Orden: asc o desc")
) -> SortParams:
    """Dependency para parámetros de ordenamiento"""
    
    # Validate sort order
    try:
        order_enum = SortOrder(sort_order.lower())
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid sort order. Use 'asc' or 'desc'"
        )
    
    # Validate sort field (for books)
    valid_sort_fields = {
        'created_at', 'updated_at', 'title', 'status', 'source_lang', 
        'progress_percentage', 'priority', 'completed_at'
    }
    
    if sort_by not in valid_sort_fields:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid sort field. Valid options: {valid_sort_fields}"
        )
    
    return SortParams(sort_by=sort_by, sort_order=order_enum)

def get_date_range_params(
    date_from: Optional[datetime] = Query(None, description="Fecha desde"),
    date_to: Optional[datetime] = Query(None, description="Fecha hasta")
) -> DateRangeParams:
    """Dependency para parámetros de rango de fechas"""
    return DateRangeParams(date_from=date_from, date_to=date_to)

# ==========================================
# REQUEST CONTEXT DEPENDENCIES
# ==========================================

async def get_request_context(request: Request) -> Dict[str, Any]:
    """
    Dependency para obtener contexto de request con información útil
    """
    
    context = {
        'method': request.method,
        'url': str(request.url),
        'path': request.url.path,
        'query_params': dict(request.query_params),
        'headers': dict(request.headers),
        'client_ip': request.client.host if request.client else 'unknown',
        'user_agent': request.headers.get('user-agent', 'unknown'),
        'timestamp': datetime.now().isoformat(),
        'request_id': getattr(request.state, 'request_id', str(uuid.uuid4()))
    }
    
    # Add request ID to state if not present
    if not hasattr(request.state, 'request_id'):
        request.state.request_id = context['request_id']
    
    return context

async def get_response_builder(request: Request) -> ResponseBuilder:
    """Dependency para obtener ResponseBuilder con contexto de request"""
    return ResponseBuilder(request)

# ==========================================
# SERVICE DEPENDENCIES
# ==========================================

def get_logger_with_context(
    request_context: Dict[str, Any] = Depends(get_request_context),
    current_user: Optional[User] = Depends(get_current_user)
) -> structlog.BoundLogger:
    """
    Dependency para obtener logger con contexto enriquecido
    """
    
    log_context = {
        'request_id': request_context['request_id'],
        'method': request_context['method'],
        'path': request_context['path'],
        'client_ip': request_context['client_ip']
    }
    
    if current_user:
        log_context.update({
            'username': current_user.username,
            'user_role': current_user.role.value
        })
    
    return logger.bind(**log_context)

# ==========================================
# BUSINESS LOGIC DEPENDENCIES
# ==========================================

async def get_book_by_id(
    book_id: str,
    db: Session = Depends(get_db_session),
    current_user: User = Depends(require_authenticated_user)
) -> BookProcessingHistory:
    """
    Dependency para obtener libro por ID con validación de permisos
    """
    
    book = db.query(BookProcessingHistory).filter(
        BookProcessingHistory.book_id == book_id
    ).first()
    
    if not book:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Book with ID '{book_id}' not found"
        )
    
    # Basic permission check - viewers can see all books
    # More granular permissions can be added here
    if current_user.role == UserRole.VIEWER:
        # Viewers can only see completed books
        if book.status != BookStatus.COMPLETED.value:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Viewers can only access completed books"
            )
    
    logger.info(
        "Book accessed",
        book_id=book_id,
        username=current_user.username,
        book_status=book.status
    )
    
    return book

async def validate_book_access_for_modification(
    book: BookProcessingHistory = Depends(get_book_by_id),
    current_user: User = Depends(require_authenticated_user)
) -> BookProcessingHistory:
    """
    Dependency para validar acceso de modificación a un libro
    """
    
    # Only editors and admins can modify books
    if current_user.role not in [UserRole.EDITOR, UserRole.ADMIN]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Editor or Admin role required for modifications"
        )
    
    # Can't modify books that are currently processing
    if book.status == BookStatus.PROCESSING.value:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Cannot modify book while processing is in progress"
        )
    
    return book

# ==========================================
# QUERY BUILDER DEPENDENCIES
# ==========================================

def build_book_query(
    db: Session = Depends(get_db_session),
    filters: BookFilterParams = Depends(get_book_filter_params),
    sort_params: SortParams = Depends(get_sort_params)
):
    """
    Dependency para construir query de libros con filtros y ordenamiento
    """
    
    query = db.query(BookProcessingHistory)
    
    # Apply filters
    if filters.status:
        query = query.filter(BookProcessingHistory.status == filters.status.value)
    
    if filters.source_lang:
        query = query.filter(BookProcessingHistory.source_lang == filters.source_lang)
    
    if filters.target_lang:
        query = query.filter(BookProcessingHistory.target_lang == filters.target_lang)
    
    if filters.date_from:
        query = query.filter(BookProcessingHistory.created_at >= filters.date_from)
    
    if filters.date_to:
        query = query.filter(BookProcessingHistory.created_at <= filters.date_to)
    
    if filters.search:
        search_term = f"%{filters.search}%"
        query = query.filter(
            (BookProcessingHistory.title.ilike(search_term)) |
            (BookProcessingHistory.book_id.ilike(search_term))
        )
    
    if filters.has_errors is not None:
        if filters.has_errors:
            query = query.filter(BookProcessingHistory.error_count > 0)
        else:
            query = query.filter(BookProcessingHistory.error_count == 0)
    
    if filters.priority_min is not None:
        query = query.filter(BookProcessingHistory.priority >= filters.priority_min)
    
    if filters.priority_max is not None:
        query = query.filter(BookProcessingHistory.priority <= filters.priority_max)
    
    # Apply sorting
    sort_column = getattr(BookProcessingHistory, sort_params.sort_by, None)
    if sort_column is not None:
        if sort_params.sort_order == SortOrder.DESC:
            query = query.order_by(sort_column.desc())
        else:
            query = query.order_by(sort_column.asc())
    
    return query

# ==========================================
# PAGINATION HELPER
# ==========================================

async def paginate_query(
    query,
    pagination: PaginationParams = Depends(get_pagination_params)
) -> tuple[List[Any], PaginationInfo]:
    """
    Helper para paginar cualquier query
    
    Returns: (items, pagination_info)
    """
    
    # Get total count
    total_items = query.count()
    
    # Apply pagination
    items = query.offset(pagination.offset).limit(pagination.limit).all()
    
    # Create pagination info
    pagination_info = create_pagination_info(
        page=pagination.page,
        page_size=pagination.page_size,
        total_items=total_items
    )
    
    return items, pagination_info

# ==========================================
# VALIDATION DEPENDENCIES
# ==========================================

def validate_language_code(lang_code: str) -> str:
    """Validate language code"""
    valid_languages = {'de', 'en', 'fr', 'it', 'nl', 'es'}
    if lang_code not in valid_languages:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid language code. Valid options: {valid_languages}"
        )
    return lang_code

def validate_processing_phase(phase: str) -> ProcessingPhase:
    """Validate processing phase"""
    try:
        return ProcessingPhase(phase.lower())
    except ValueError:
        valid_phases = [p.value for p in ProcessingPhase]
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid processing phase. Valid options: {valid_phases}"
        )

# ==========================================
# MONITORING AND HEALTH DEPENDENCIES
# ==========================================

async def check_system_health(
    db_manager: DatabaseManager = Depends(get_db_manager),
    settings: Settings = Depends(get_settings)
) -> Dict[str, Any]:
    """
    Dependency para verificar salud del sistema
    """
    
    health_info = {
        'timestamp': datetime.now().isoformat(),
        'environment': settings.environment.value,
        'version': settings.app_version,
        'status': 'healthy'
    }
    
    # Check database
    try:
        db_healthy = db_manager.check_connection()
        health_info['database'] = {
            'status': 'healthy' if db_healthy else 'unhealthy',
            'connection_test': db_healthy
        }
        
        if not db_healthy:
            health_info['status'] = 'degraded'
            
    except Exception as e:
        health_info['database'] = {
            'status': 'unhealthy',
            'error': str(e)
        }
        health_info['status'] = 'unhealthy'
    
    # Check external APIs (mock check for now)
    health_info['external_apis'] = {
        'deepl': {'status': 'unknown', 'note': 'Health check not implemented'},
        'claude': {'status': 'unknown', 'note': 'Health check not implemented'},
        'abbyy': {'status': 'unknown', 'note': 'Health check not implemented'}
    }
    
    return health_info

# ==========================================
# EXPORT DEPENDENCIES
# ==========================================

__all__ = [
    # Models
    'BookFilterParams',
    'PaginationParams', 
    'SortParams',
    'DateRangeParams',
    'ValidatedFile',
    'FileValidationConfig',
    
    # Database dependencies
    'get_db_session',
    'get_db_manager',
    'get_db_with_user_context',
    
    # Query parameter dependencies
    'get_pagination_params',
    'get_book_filter_params',
    'get_sort_params',
    'get_date_range_params',
    
    # File validation
    'validate_upload_file',
    'sanitize_filename',
    
    # Request context
    'get_request_context',
    'get_response_builder',
    'get_logger_with_context',
    
    # Business logic
    'get_book_by_id',
    'validate_book_access_for_modification',
    'build_book_query',
    'paginate_query',
    
    # Validation
    'validate_language_code',
    'validate_processing_phase',
    
    # Monitoring
    'check_system_health',
    
    # Enums
    'SortOrder'
]
