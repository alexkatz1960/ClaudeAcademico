 # ==========================================
# INTERFACES/FASTAPI_BACKEND/UTILS/RESPONSES.PY
# Standardized API Responses - Enterprise Grade
# Sistema de Traducción Académica v2.2
# ==========================================

from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional, Union, Generic, TypeVar
from datetime import datetime
from enum import Enum
import structlog
import traceback
import uuid

logger = structlog.get_logger(__name__)

# ==========================================
# RESPONSE TYPES AND ENUMS
# ==========================================

class ResponseStatus(str, Enum):
    SUCCESS = "success"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"

class ErrorCode(str, Enum):
    # General errors
    INTERNAL_ERROR = "INTERNAL_ERROR"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    NOT_FOUND = "NOT_FOUND"
    PERMISSION_DENIED = "PERMISSION_DENIED"
    RATE_LIMITED = "RATE_LIMITED"
    
    # Authentication errors
    AUTHENTICATION_FAILED = "AUTHENTICATION_FAILED"
    TOKEN_EXPIRED = "TOKEN_EXPIRED"
    INVALID_CREDENTIALS = "INVALID_CREDENTIALS"
    
    # Business logic errors
    BOOK_NOT_FOUND = "BOOK_NOT_FOUND"
    PROCESSING_ERROR = "PROCESSING_ERROR"
    INVALID_FILE_FORMAT = "INVALID_FILE_FORMAT"
    FILE_TOO_LARGE = "FILE_TOO_LARGE"
    PROCESSING_TIMEOUT = "PROCESSING_TIMEOUT"
    
    # External API errors
    EXTERNAL_API_ERROR = "EXTERNAL_API_ERROR"
    DEEPL_API_ERROR = "DEEPL_API_ERROR"
    CLAUDE_API_ERROR = "CLAUDE_API_ERROR"
    ABBYY_API_ERROR = "ABBYY_API_ERROR"
    
    # Database errors
    DATABASE_ERROR = "DATABASE_ERROR"
    CONNECTION_ERROR = "CONNECTION_ERROR"
    CONSTRAINT_VIOLATION = "CONSTRAINT_VIOLATION"

class ErrorSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

# ==========================================
# GENERIC RESPONSE MODELS
# ==========================================

T = TypeVar('T')

class ResponseMetadata(BaseModel):
    """Metadatos estándar para todas las respuestas"""
    timestamp: datetime = Field(default_factory=datetime.now)
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    version: str = Field(default="2.2.0")
    environment: Optional[str] = None
    processing_time_ms: Optional[float] = None

class PaginationInfo(BaseModel):
    """Información de paginación"""
    page: int = Field(ge=1, description="Página actual")
    page_size: int = Field(ge=1, le=1000, description="Elementos por página")
    total_items: int = Field(ge=0, description="Total de elementos")
    total_pages: int = Field(ge=0, description="Total de páginas")
    has_next: bool = Field(description="Hay página siguiente")
    has_previous: bool = Field(description="Hay página anterior")

    @property
    def offset(self) -> int:
        """Calcular offset para queries de base de datos"""
        return (self.page - 1) * self.page_size

class ErrorDetail(BaseModel):
    """Detalle de error estructurado"""
    code: ErrorCode = Field(description="Código de error único")
    message: str = Field(description="Mensaje de error legible")
    severity: ErrorSeverity = Field(default=ErrorSeverity.MEDIUM)
    field: Optional[str] = Field(None, description="Campo específico con error")
    context: Optional[Dict[str, Any]] = Field(None, description="Contexto adicional")
    suggestion: Optional[str] = Field(None, description="Sugerencia para resolver")
    documentation_url: Optional[str] = Field(None, description="URL de documentación")

class BaseResponse(BaseModel, Generic[T]):
    """Respuesta base para todas las APIs"""
    status: ResponseStatus = Field(description="Estado de la respuesta")
    data: Optional[T] = Field(None, description="Datos de la respuesta")
    message: str = Field(description="Mensaje descriptivo")
    metadata: ResponseMetadata = Field(default_factory=ResponseMetadata)
    errors: Optional[List[ErrorDetail]] = Field(None, description="Lista de errores")

class SuccessResponse(BaseResponse[T]):
    """Respuesta exitosa estándar"""
    status: ResponseStatus = Field(default=ResponseStatus.SUCCESS)
    message: str = Field(default="Operation completed successfully")

class ErrorResponse(BaseResponse[None]):
    """Respuesta de error estándar"""
    status: ResponseStatus = Field(default=ResponseStatus.ERROR)
    data: None = Field(default=None)
    errors: List[ErrorDetail] = Field(description="Lista de errores")

class PaginatedResponse(BaseResponse[List[T]]):
    """Respuesta paginada estándar"""
    pagination: PaginationInfo = Field(description="Información de paginación")

# ==========================================
# SPECIFIC RESPONSE MODELS
# ==========================================

class HealthCheckResponse(BaseModel):
    """Respuesta para health checks"""
    status: str = Field(description="Estado del servicio")
    version: str = Field(description="Versión de la aplicación")
    timestamp: datetime = Field(default_factory=datetime.now)
    dependencies: Dict[str, Dict[str, Any]] = Field(description="Estado de dependencias")
    system_info: Dict[str, Any] = Field(description="Información del sistema")

class ValidationErrorResponse(ErrorResponse):
    """Respuesta específica para errores de validación"""
    
    @classmethod
    def from_validation_error(cls, validation_errors: List[Dict[str, Any]]) -> "ValidationErrorResponse":
        """Crear respuesta desde errores de validación de Pydantic"""
        error_details = []
        
        for error in validation_errors:
            error_detail = ErrorDetail(
                code=ErrorCode.VALIDATION_ERROR,
                message=error.get('msg', 'Validation error'),
                field='.'.join(str(loc) for loc in error.get('loc', [])),
                context={'input': error.get('input'), 'type': error.get('type')},
                suggestion="Please check the field format and try again"
            )
            error_details.append(error_detail)
        
        return cls(
            message="Validation failed",
            errors=error_details
        )

class ProcessingStatusResponse(SuccessResponse[Dict[str, Any]]):
    """Respuesta para estado de procesamiento"""
    
    @classmethod
    def from_book_status(cls, book_data: Dict[str, Any]) -> "ProcessingStatusResponse":
        """Crear respuesta desde datos de libro"""
        return cls(
            data=book_data,
            message=f"Book {book_data.get('book_id', 'unknown')} status retrieved"
        )

# ==========================================
# RESPONSE BUILDERS
# ==========================================

class ResponseBuilder:
    """Constructor de respuestas con logging automático"""
    
    def __init__(self, request: Optional[Request] = None):
        self.request = request
        self.start_time = datetime.now()
    
    def success(
        self, 
        data: Optional[T] = None, 
        message: str = "Operation completed successfully",
        status_code: int = status.HTTP_200_OK
    ) -> JSONResponse:
        """Crear respuesta de éxito"""
        
        response_data = SuccessResponse(
            data=data,
            message=message,
            metadata=self._create_metadata()
        )
        
        self._log_response("success", status_code, message)
        
        return JSONResponse(
            content=jsonable_encoder(response_data.dict()),
            status_code=status_code
        )
    
    def error(
        self,
        message: str,
        errors: Optional[List[ErrorDetail]] = None,
        status_code: int = status.HTTP_400_BAD_REQUEST
    ) -> JSONResponse:
        """Crear respuesta de error"""
        
        if errors is None:
            errors = [ErrorDetail(
                code=ErrorCode.INTERNAL_ERROR,
                message=message
            )]
        
        response_data = ErrorResponse(
            message=message,
            errors=errors,
            metadata=self._create_metadata()
        )
        
        self._log_response("error", status_code, message)
        
        return JSONResponse(
            content=jsonable_encoder(response_data.dict()),
            status_code=status_code
        )
    
    def not_found(
        self, 
        resource: str = "Resource", 
        resource_id: Optional[str] = None
    ) -> JSONResponse:
        """Respuesta estándar para recursos no encontrados"""
        
        message = f"{resource} not found"
        if resource_id:
            message += f" (ID: {resource_id})"
        
        error_detail = ErrorDetail(
            code=ErrorCode.NOT_FOUND,
            message=message,
            suggestion=f"Please verify the {resource.lower()} ID and try again"
        )
        
        return self.error(
            message=message,
            errors=[error_detail],
            status_code=status.HTTP_404_NOT_FOUND
        )
    
    def validation_error(
        self, 
        validation_errors: List[Dict[str, Any]]
    ) -> JSONResponse:
        """Respuesta para errores de validación"""
        
        validation_response = ValidationErrorResponse.from_validation_error(validation_errors)
        validation_response.metadata = self._create_metadata()
        
        self._log_response("validation_error", status.HTTP_422_UNPROCESSABLE_ENTITY, "Validation failed")
        
        return JSONResponse(
            content=jsonable_encoder(validation_response.dict()),
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY
        )
    
    def paginated(
        self,
        data: List[T],
        pagination: PaginationInfo,
        message: str = "Data retrieved successfully"
    ) -> JSONResponse:
        """Crear respuesta paginada"""
        
        response_data = PaginatedResponse(
            data=data,
            pagination=pagination,
            message=message,
            metadata=self._create_metadata()
        )
        
        self._log_response("success", status.HTTP_200_OK, f"Retrieved {len(data)} items (page {pagination.page})")
        
        return JSONResponse(
            content=jsonable_encoder(response_data.dict()),
            status_code=status.HTTP_200_OK
        )
    
    def created(
        self,
        data: Optional[T] = None,
        message: str = "Resource created successfully",
        location: Optional[str] = None
    ) -> JSONResponse:
        """Respuesta para recursos creados"""
        
        response_data = SuccessResponse(
            data=data,
            message=message,
            metadata=self._create_metadata()
        )
        
        headers = {}
        if location:
            headers["Location"] = location
        
        self._log_response("created", status.HTTP_201_CREATED, message)
        
        return JSONResponse(
            content=jsonable_encoder(response_data.dict()),
            status_code=status.HTTP_201_CREATED,
            headers=headers
        )
    
    def accepted(
        self,
        data: Optional[T] = None,
        message: str = "Request accepted for processing"
    ) -> JSONResponse:
        """Respuesta para operaciones asíncronas aceptadas"""
        
        response_data = SuccessResponse(
            data=data,
            message=message,
            metadata=self._create_metadata()
        )
        
        self._log_response("accepted", status.HTTP_202_ACCEPTED, message)
        
        return JSONResponse(
            content=jsonable_encoder(response_data.dict()),
            status_code=status.HTTP_202_ACCEPTED
        )
    
    def no_content(self, message: str = "Operation completed successfully") -> JSONResponse:
        """Respuesta sin contenido"""
        
        self._log_response("no_content", status.HTTP_204_NO_CONTENT, message)
        
        return JSONResponse(
            content=None,
            status_code=status.HTTP_204_NO_CONTENT
        )
    
    def _create_metadata(self) -> ResponseMetadata:
        """Crear metadatos de respuesta"""
        processing_time_ms = (datetime.now() - self.start_time).total_seconds() * 1000
        
        metadata = ResponseMetadata(processing_time_ms=processing_time_ms)
        
        # Add request ID from request if available
        if self.request and hasattr(self.request.state, 'request_id'):
            metadata.request_id = self.request.state.request_id
        
        return metadata
    
    def _log_response(self, response_type: str, status_code: int, message: str):
        """Log de respuesta con información estructurada"""
        processing_time_ms = (datetime.now() - self.start_time).total_seconds() * 1000
        
        log_data = {
            "response_type": response_type,
            "status_code": status_code,
            "message": message,
            "processing_time_ms": processing_time_ms
        }
        
        if self.request:
            log_data.update({
                "method": self.request.method,
                "url": str(self.request.url),
                "user_agent": self.request.headers.get("user-agent", "unknown")
            })
        
        if status_code >= 400:
            logger.warning("API response with error", **log_data)
        else:
            logger.info("API response success", **log_data)

# ==========================================
# EXCEPTION HANDLERS
# ==========================================

class APIException(HTTPException):
    """Excepción base para la API con información estructurada"""
    
    def __init__(
        self,
        status_code: int,
        error_code: ErrorCode,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        suggestion: Optional[str] = None
    ):
        self.error_code = error_code
        self.details = details or {}
        self.suggestion = suggestion
        
        super().__init__(status_code=status_code, detail=message)

class BookNotFoundException(APIException):
    """Excepción para libro no encontrado"""
    
    def __init__(self, book_id: str):
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            error_code=ErrorCode.BOOK_NOT_FOUND,
            message=f"Book with ID '{book_id}' not found",
            details={"book_id": book_id},
            suggestion="Please verify the book ID and try again"
        )

class ProcessingException(APIException):
    """Excepción para errores de procesamiento"""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            error_code=ErrorCode.PROCESSING_ERROR,
            message=f"Processing failed: {message}",
            details=details,
            suggestion="Please check the input format and try again"
        )

class ExternalAPIException(APIException):
    """Excepción para errores de APIs externas"""
    
    def __init__(self, api_name: str, message: str, details: Optional[Dict[str, Any]] = None):
        error_code_map = {
            "deepl": ErrorCode.DEEPL_API_ERROR,
            "claude": ErrorCode.CLAUDE_API_ERROR,
            "abbyy": ErrorCode.ABBYY_API_ERROR
        }
        
        super().__init__(
            status_code=status.HTTP_502_BAD_GATEWAY,
            error_code=error_code_map.get(api_name.lower(), ErrorCode.EXTERNAL_API_ERROR),
            message=f"{api_name} API error: {message}",
            details=details,
            suggestion="The external service is temporarily unavailable. Please try again later."
        )

# ==========================================
# UTILITY FUNCTIONS
# ==========================================

def create_error_detail(
    code: ErrorCode,
    message: str,
    field: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
    suggestion: Optional[str] = None
) -> ErrorDetail:
    """Función auxiliar para crear detalles de error"""
    return ErrorDetail(
        code=code,
        message=message,
        field=field,
        context=context,
        suggestion=suggestion
    )

def handle_database_error(e: Exception) -> ErrorDetail:
    """Manejar errores de base de datos de forma consistente"""
    error_message = str(e)
    
    # Identify specific database errors
    if "UNIQUE constraint failed" in error_message:
        return create_error_detail(
            code=ErrorCode.CONSTRAINT_VIOLATION,
            message="Duplicate entry detected",
            suggestion="Please use a unique identifier"
        )
    elif "FOREIGN KEY constraint failed" in error_message:
        return create_error_detail(
            code=ErrorCode.CONSTRAINT_VIOLATION,
            message="Referenced resource does not exist",
            suggestion="Please verify the referenced resource exists"
        )
    else:
        return create_error_detail(
            code=ErrorCode.DATABASE_ERROR,
            message="Database operation failed",
            context={"original_error": error_message},
            suggestion="Please try again or contact support"
        )

def create_pagination_info(
    page: int,
    page_size: int,
    total_items: int
) -> PaginationInfo:
    """Crear información de paginación"""
    total_pages = (total_items + page_size - 1) // page_size
    
    return PaginationInfo(
        page=page,
        page_size=page_size,
        total_items=total_items,
        total_pages=total_pages,
        has_next=page < total_pages,
        has_previous=page > 1
    )

def format_exception_for_response(e: Exception) -> ErrorDetail:
    """Formatear excepción para respuesta de API"""
    if isinstance(e, APIException):
        return ErrorDetail(
            code=e.error_code,
            message=e.detail,
            context=e.details,
            suggestion=e.suggestion
        )
    else:
        # Log full exception for debugging
        logger.error("Unhandled exception", exception=str(e), traceback=traceback.format_exc())
        
        return ErrorDetail(
            code=ErrorCode.INTERNAL_ERROR,
            message="An internal error occurred",
            suggestion="Please try again or contact support if the problem persists"
        )

# ==========================================
# RESPONSE DECORATORS
# ==========================================

def with_error_handling(func):
    """Decorator para manejo automático de errores en endpoints"""
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except APIException:
            # Re-raise API exceptions as they're already formatted
            raise
        except Exception as e:
            # Convert unexpected exceptions to API exceptions
            logger.error(f"Unexpected error in {func.__name__}", error=str(e))
            raise APIException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                error_code=ErrorCode.INTERNAL_ERROR,
                message="An internal error occurred",
                details={"function": func.__name__, "error": str(e)}
            )
    
    return wrapper

# ==========================================
# EXPORT CLASSES AND FUNCTIONS
# ==========================================

__all__ = [
    # Enums
    'ResponseStatus',
    'ErrorCode',
    'ErrorSeverity',
    
    # Models
    'ResponseMetadata',
    'PaginationInfo',
    'ErrorDetail',
    'BaseResponse',
    'SuccessResponse',
    'ErrorResponse',
    'PaginatedResponse',
    'HealthCheckResponse',
    'ValidationErrorResponse',
    'ProcessingStatusResponse',
    
    # Builders
    'ResponseBuilder',
    
    # Exceptions
    'APIException',
    'BookNotFoundException',
    'ProcessingException',
    'ExternalAPIException',
    
    # Utilities
    'create_error_detail',
    'handle_database_error',
    'create_pagination_info',
    'format_exception_for_response',
    'with_error_handling'
]
