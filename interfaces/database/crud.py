# ==========================================
# INTERFACES/DATABASE/CRUD.PY
# CRUD Operations - Enterprise Implementation
# Sistema de Traducción Académica v2.2
# ==========================================

from sqlalchemy.orm import Session, Query
from sqlalchemy import and_, or_, func, desc, asc, text
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from pydantic import BaseModel, ValidationError
from typing import List, Optional, Dict, Any, Tuple, Union, Type
from datetime import datetime, date, timedelta
import logging
import time
import json
from functools import wraps
from contextlib import contextmanager
from abc import ABC, abstractmethod
import threading

from .models import (
    SystemConfig, BookProcessingHistory, ErrorPattern, AuditLog,
    TerminologySuggestion, EditorialReview, UsageStatistic
)
from .schemas import (
    PaginationParams, BookStatus, ProcessingPhase, Severity,
    Priority, EditorDecision, PatternType, LanguageCode,
    SystemConfigCreate, SystemConfigUpdate, BookProcessingCreate,
    BookProcessingUpdate, ErrorPatternCreate, ErrorPatternUpdate,
    AuditLogCreate, TerminologySuggestionCreate, TerminologySuggestionUpdate,
    EditorialReviewCreate, EditorialReviewUpdate, UsageStatisticsCreate,
    UsageStatisticsUpdate, BulkUpdateRequest, BulkUpdateResponse
)

logger = logging.getLogger(__name__)

# ==========================================
# EXCEPCIONES PERSONALIZADAS
# ==========================================

class CRUDError(Exception):
    """Excepción base para errores CRUD"""
    pass

class CRUDValidationError(CRUDError):
    """Error de validación de datos"""
    pass

class ConcurrencyError(CRUDError):
    """Error de concurrencia"""
    pass

class NotFoundError(CRUDError):
    """Registro no encontrado"""
    pass

class DuplicateError(CRUDError):
    """Registro duplicado"""
    pass

# ==========================================
# MÉTRICAS DE PERFORMANCE
# ==========================================

class PerformanceMetrics:
    """Recolector de métricas de performance"""
    
    def __init__(self):
        self._metrics = {}
        self._lock = threading.Lock()
    
    def record_operation(self, operation_name: str, duration: float, record_count: int = 1):
        """Registrar métrica de operación"""
        with self._lock:
            if operation_name not in self._metrics:
                self._metrics[operation_name] = {
                    'total_calls': 0,
                    'total_duration': 0.0,
                    'total_records': 0,
                    'avg_duration': 0.0,
                    'avg_records_per_call': 0.0
                }
            
            metrics = self._metrics[operation_name]
            metrics['total_calls'] += 1
            metrics['total_duration'] += duration
            metrics['total_records'] += record_count
            metrics['avg_duration'] = metrics['total_duration'] / metrics['total_calls']
            metrics['avg_records_per_call'] = metrics['total_records'] / metrics['total_calls']
    
    def get_metrics(self) -> Dict[str, Any]:
        """Obtener todas las métricas"""
        with self._lock:
            return dict(self._metrics)
    
    def reset_metrics(self):
        """Resetear métricas"""
        with self._lock:
            self._metrics.clear()

# Instancia global de métricas
performance_metrics = PerformanceMetrics()

def measure_performance(operation_name: str = None):
    """Decorator para medir performance de operaciones"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            operation = operation_name or f"{func.__name__}"
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                # Determinar número de registros basado en el resultado
                record_count = 1
                if isinstance(result, (list, tuple)):
                    record_count = len(result)
                elif isinstance(result, tuple) and len(result) == 2:
                    # Caso de get_multi que retorna (items, total)
                    items, total = result
                    record_count = len(items) if isinstance(items, list) else 1
                
                performance_metrics.record_operation(operation, duration, record_count)
                
                # Log para operaciones lentas
                if duration > 1.0:  # > 1 segundo
                    logger.warning(f"Slow operation: {operation} took {duration:.3f}s for {record_count} records")
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                logger.error(f"Operation {operation} failed after {duration:.3f}s: {e}")
                raise
                
        return wrapper
    return decorator

# ==========================================
# BASE CRUD CLASS
# ==========================================

class BaseCRUD(ABC):
    """
    Base class para todas las operaciones CRUD
    Incluye validación Pydantic, control de concurrencia y métricas
    """
    
    def __init__(self, model, create_schema: Type[BaseModel], update_schema: Type[BaseModel]):
        self.model = model
        self.create_schema = create_schema
        self.update_schema = update_schema
        self._model_fields = set(column.name for column in model.__table__.columns)
    
    @contextmanager
    def _transaction_scope(self, db: Session):
        """Context manager para manejo seguro de transacciones"""
        try:
            yield db
            db.commit()
        except Exception as e:
            db.rollback()
            logger.error(f"Transaction rollback for {self.model.__name__}: {e}")
            raise
    
    def _validate_input(self, obj_in: Union[Dict[str, Any], BaseModel], schema: Type[BaseModel]) -> Dict[str, Any]:
        """Validar entrada usando schemas Pydantic"""
        try:
            if isinstance(obj_in, BaseModel):
                validated_data = obj_in.dict(exclude_unset=True)
            else:
                validated_obj = schema(**obj_in)
                validated_data = validated_obj.dict(exclude_unset=True)
            
            # Filtrar solo campos que existen en el modelo
            filtered_data = {
                key: value for key, value in validated_data.items()
                if key in self._model_fields and value is not None
            }
            
            return filtered_data
            
        except ValidationError as e:
            logger.error(f"Validation error for {self.model.__name__}: {e}")
            raise CRUDValidationError(f"Validation failed: {e}")
    
    @measure_performance()
    def get(self, db: Session, id: int, for_update: bool = False) -> Optional[Any]:
        """Obtener registro por ID con control de concurrencia opcional"""
        try:
            query = db.query(self.model).filter(self.model.id == id)
            if for_update:
                query = query.with_for_update()
            return query.first()
        except SQLAlchemyError as e:
            logger.error(f"Error getting {self.model.__name__} with id {id}: {e}")
            return None
    
    @measure_performance()
    def get_multi(
        self, 
        db: Session, 
        pagination: PaginationParams,
        filters: Optional[Dict[str, Any]] = None,
        order_by: Optional[str] = None
    ) -> Tuple[List[Any], int]:
        """Obtener múltiples registros con paginación optimizada"""
        try:
            query = db.query(self.model)
            
            # Aplicar filtros
            if filters:
                query = self._apply_filters(query, filters)
            
            # Obtener total antes de aplicar paginación
            total = query.count()
            
            # Aplicar ordenamiento
            if order_by:
                query = self._apply_ordering(query, order_by)
            
            # Aplicar paginación
            items = query.offset(pagination.offset).limit(pagination.size).all()
            
            return items, total
            
        except SQLAlchemyError as e:
            logger.error(f"Error getting multiple {self.model.__name__}: {e}")
            return [], 0
    
    @measure_performance()
    def create(self, db: Session, *, obj_in: Union[Dict[str, Any], BaseModel], user_id: str = "system") -> Optional[Any]:
        """Crear nuevo registro con validación completa"""
        try:
            # Validar entrada
            validated_data = self._validate_input(obj_in, self.create_schema)
            
            with self._transaction_scope(db):
                # Agregar metadatos de auditoría si el modelo los soporta
                if hasattr(self.model, 'created_by'):
                    validated_data['created_by'] = user_id
                if hasattr(self.model, 'created_at'):
                    validated_data['created_at'] = datetime.now()
                
                db_obj = self.model(**validated_data)
                db.add(db_obj)
                db.flush()  # Para obtener el ID sin commit
                
                logger.info(f"Created {self.model.__name__} with id {db_obj.id}")
                return db_obj
                
        except CRUDValidationError:
            raise
        except IntegrityError as e:
            logger.error(f"Integrity error creating {self.model.__name__}: {e}")
            raise DuplicateError(f"Record already exists or violates constraints: {e}")
        except SQLAlchemyError as e:
            logger.error(f"Database error creating {self.model.__name__}: {e}")
            raise CRUDError(f"Failed to create record: {e}")
    
    @measure_performance()
    def update(
        self, 
        db: Session, 
        *, 
        db_obj: Any, 
        obj_in: Union[Dict[str, Any], BaseModel],
        user_id: str = "system",
        use_lock: bool = False
    ) -> Optional[Any]:
        """Actualizar registro existente con validación y control de concurrencia"""
        try:
            # Validar entrada
            validated_data = self._validate_input(obj_in, self.update_schema)
            
            if not validated_data:
                logger.warning(f"No valid fields to update for {self.model.__name__}")
                return db_obj
            
            with self._transaction_scope(db):
                # Re-obtener objeto con lock si es necesario
                if use_lock:
                    db_obj = db.query(self.model).filter(
                        self.model.id == db_obj.id
                    ).with_for_update().first()
                    
                    if not db_obj:
                        raise NotFoundError(f"Record not found for update")
                
                # Aplicar cambios
                for field, value in validated_data.items():
                    if hasattr(db_obj, field):
                        setattr(db_obj, field, value)
                
                # Actualizar metadatos de auditoría
                if hasattr(db_obj, 'updated_at'):
                    db_obj.updated_at = datetime.now()
                if hasattr(db_obj, 'updated_by'):
                    db_obj.updated_by = user_id
                
                db.flush()
                logger.info(f"Updated {self.model.__name__} with id {db_obj.id}")
                return db_obj
                
        except CRUDValidationError:
            raise
        except SQLAlchemyError as e:
            logger.error(f"Error updating {self.model.__name__}: {e}")
            raise CRUDError(f"Failed to update record: {e}")
    
    @measure_performance()
    def delete(self, db: Session, *, id: int, user_id: str = "system") -> bool:
        """Eliminar registro por ID"""
        try:
            obj = db.query(self.model).get(id)
            if obj:
                db.delete(obj)
                db.commit()
                logger.info(f"Deleted {self.model.__name__} with id {id}")
                return True
            return False
        except SQLAlchemyError as e:
            db.rollback()
            logger.error(f"Error deleting {self.model.__name__} with id {id}: {e}")
            return False
    
    @measure_performance()
    def bulk_update(
        self,
        db: Session,
        *,
        request: BulkUpdateRequest,
        user_id: str = "system"
    ) -> BulkUpdateResponse:
        """Actualización en lote con validación"""
        errors = []
        updated_count = 0
        
        try:
            # Validar datos de actualización
            validated_updates = self._validate_input(request.updates, self.update_schema)
            
            if not validated_updates:
                return BulkUpdateResponse(
                    updated_count=0,
                    errors=["No valid fields to update"],
                    success=False
                )
            
            with self._transaction_scope(db):
                # Agregar metadatos de auditoría
                if hasattr(self.model, 'updated_at'):
                    validated_updates['updated_at'] = datetime.now()
                if hasattr(self.model, 'updated_by'):
                    validated_updates['updated_by'] = user_id
                
                # Ejecutar actualización en lote
                result = db.query(self.model).filter(
                    self.model.id.in_(request.ids)
                ).update(validated_updates, synchronize_session=False)
                
                updated_count = result
                
                logger.info(f"Bulk updated {updated_count} {self.model.__name__} records")
                
                return BulkUpdateResponse(
                    updated_count=updated_count,
                    errors=errors,
                    success=True
                )
                
        except CRUDValidationError as e:
            errors.append(f"Validation error: {e}")
        except SQLAlchemyError as e:
            logger.error(f"Error in bulk update: {e}")
            errors.append(f"Database error: {e}")
        
        return BulkUpdateResponse(
            updated_count=updated_count,
            errors=errors,
            success=len(errors) == 0
        )
    
    @abstractmethod
    def _apply_filters(self, query: Query, filters: Dict[str, Any]) -> Query:
        """Aplicar filtros específicos del modelo"""
        pass
    
    def _apply_ordering(self, query: Query, order_by: str) -> Query:
        """Aplicar ordenamiento"""
        if order_by.startswith('-'):
            field = order_by[1:]
            if hasattr(self.model, field):
                return query.order_by(desc(getattr(self.model, field)))
        else:
            if hasattr(self.model, order_by):
                return query.order_by(asc(getattr(self.model, order_by)))
        
        # Fallback: ordenar por id descendente
        return query.order_by(desc(self.model.id))

# ==========================================
# SYSTEM CONFIG CRUD
# ==========================================

class SystemConfigCRUD(BaseCRUD):
    """CRUD operations for SystemConfig"""
    
    def __init__(self):
        super().__init__(SystemConfig, SystemConfigCreate, SystemConfigUpdate)
    
    @measure_performance("get_config_by_key")
    def get_by_key(self, db: Session, *, config_key: str, for_update: bool = False) -> Optional[SystemConfig]:
        """Obtener configuración por clave"""
        try:
            query = db.query(SystemConfig).filter(
                and_(
                    SystemConfig.config_key == config_key,
                    SystemConfig.is_active == True
                )
            )
            if for_update:
                query = query.with_for_update()
            return query.first()
        except SQLAlchemyError as e:
            logger.error(f"Error getting config {config_key}: {e}")
            return None
    
    def get_active_configs(self, db: Session) -> List[SystemConfig]:
        """Obtener todas las configuraciones activas"""
        return db.query(SystemConfig).filter(
            SystemConfig.is_active == True
        ).order_by(SystemConfig.config_key).all()
    
    def upsert_config(
        self, 
        db: Session, 
        *, 
        config_key: str, 
        config_value: str,
        config_type: str = 'string',
        description: Optional[str] = None,
        user_id: str = "system"
    ) -> SystemConfig:
        """Crear o actualizar configuración"""
        existing = self.get_by_key(db, config_key=config_key, for_update=True)
        
        if existing:
            return self.update(
                db, 
                db_obj=existing, 
                obj_in={
                    'config_value': config_value,
                    'config_type': config_type,
                    'description': description
                },
                user_id=user_id,
                use_lock=True
            )
        else:
            return self.create(
                db, 
                obj_in={
                    'config_key': config_key,
                    'config_value': config_value,
                    'config_type': config_type,
                    'description': description
                },
                user_id=user_id
            )
    
    def _apply_filters(self, query: Query, filters: Dict[str, Any]) -> Query:
        """Aplicar filtros específicos para SystemConfig"""
        if 'config_type' in filters:
            query = query.filter(SystemConfig.config_type == filters['config_type'])
        if 'is_active' in filters:
            query = query.filter(SystemConfig.is_active == filters['is_active'])
        if 'search' in filters:
            search_term = f"%{filters['search']}%"
            query = query.filter(
                or_(
                    SystemConfig.config_key.ilike(search_term),
                    SystemConfig.description.ilike(search_term)
                )
            )
        return query

# ==========================================
# BOOK PROCESSING CRUD
# ==========================================

class BookCRUD(BaseCRUD):
    """CRUD operations for BookProcessingHistory"""
    
    def __init__(self):
        super().__init__(BookProcessingHistory, BookProcessingCreate, BookProcessingUpdate)
    
    @measure_performance("get_book_by_id")
    def get_by_book_id(self, db: Session, *, book_id: str, for_update: bool = False) -> Optional[BookProcessingHistory]:
        """Obtener libro por book_id con lock opcional"""
        try:
            query = db.query(BookProcessingHistory).filter(
                BookProcessingHistory.book_id == book_id
            )
            if for_update:
                query = query.with_for_update()
            return query.first()
        except SQLAlchemyError as e:
            logger.error(f"Error getting book {book_id}: {e}")
            return None
    
    def get_active_books(self, db: Session) -> List[BookProcessingHistory]:
        """Obtener libros en procesamiento activo"""
        return db.query(BookProcessingHistory).filter(
            BookProcessingHistory.status.in_(['queued', 'processing'])
        ).order_by(BookProcessingHistory.started_at).all()
    
    def get_recent_books(
        self, 
        db: Session, 
        *, 
        days: int = 7,
        limit: int = 50
    ) -> List[BookProcessingHistory]:
        """Obtener libros procesados recientemente"""
        cutoff_date = datetime.now() - timedelta(days=days)
        return db.query(BookProcessingHistory).filter(
            BookProcessingHistory.started_at >= cutoff_date
        ).order_by(desc(BookProcessingHistory.started_at)).limit(limit).all()
    
    @measure_performance("update_book_progress")
    def update_progress(
        self,
        db: Session,
        *,
        book_id: str,
        current_phase: ProcessingPhase,
        progress_percentage: float,
        quality_scores: Optional[Dict[str, float]] = None,
        user_id: str = "system"
    ) -> Optional[BookProcessingHistory]:
        """Actualizar progreso con control de concurrencia crítico"""
        try:
            # CRÍTICO: Usar lock para evitar race conditions
            book = self.get_by_book_id(db, book_id=book_id, for_update=True)
            if not book:
                raise NotFoundError(f"Book {book_id} not found")
            
            # Validar que el progreso sea lógico
            if progress_percentage < book.progress_percentage:
                logger.warning(f"Progress going backwards for {book_id}: {book.progress_percentage} -> {progress_percentage}")
            
            updates = {
                'current_phase': current_phase.value,
                'progress_percentage': progress_percentage
            }
            
            # Actualizar quality scores de forma segura
            if quality_scores:
                existing_scores = book.quality_scores or {}
                existing_scores.update(quality_scores)
                updates['quality_scores'] = existing_scores
            
            result = self.update(db, db_obj=book, obj_in=updates, user_id=user_id, use_lock=True)
            
            logger.info(f"Progress updated for {book_id}: {progress_percentage}% in phase {current_phase.value}")
            return result
            
        except NotFoundError:
            raise
        except Exception as e:
            logger.error(f"Failed to update progress for {book_id}: {e}")
            raise CRUDError(f"Progress update failed: {e}")
    
    def mark_completed(
        self,
        db: Session,
        *,
        book_id: str,
        output_file_path: str,
        processing_time_seconds: int,
        final_quality_scores: Dict[str, float],
        user_id: str = "system"
    ) -> Optional[BookProcessingHistory]:
        """Marcar libro como completado"""
        book = self.get_by_book_id(db, book_id=book_id, for_update=True)
        if not book:
            raise NotFoundError(f"Book {book_id} not found")
        
        updates = {
            'status': BookStatus.COMPLETED.value,
            'completed_at': datetime.now(),
            'progress_percentage': 100.0,
            'output_file_path': output_file_path,
            'processing_time_seconds': processing_time_seconds,
            'quality_scores': final_quality_scores
        }
        
        # Calcular scores promedio para reportes
        if final_quality_scores:
            semantic_score = final_quality_scores.get('semantic_integrity', 0)
            format_score = final_quality_scores.get('format_preservation', 0)
            footnote_score = final_quality_scores.get('footnote_preservation', 0)
            
            updates.update({
                'semantic_score_avg': semantic_score,
                'format_preservation_score': format_score,
                'footnote_preservation_score': footnote_score
            })
        
        result = self.update(db, db_obj=book, obj_in=updates, user_id=user_id, use_lock=True)
        
        logger.info(f"Book {book_id} completed in {processing_time_seconds}s with quality scores: {final_quality_scores}")
        return result
    
    def mark_failed(
        self,
        db: Session,
        *,
        book_id: str,
        error_message: str,
        user_id: str = "system"
    ) -> Optional[BookProcessingHistory]:
        """Marcar libro como fallido"""
        book = self.get_by_book_id(db, book_id=book_id, for_update=True)
        if not book:
            raise NotFoundError(f"Book {book_id} not found")
        
        error_count = book.error_count + 1
        
        updates = {
            'status': BookStatus.FAILED.value,
            'error_count': error_count,
            'completed_at': datetime.now()
        }
        
        result = self.update(db, db_obj=book, obj_in=updates, user_id=user_id, use_lock=True)
        
        logger.error(f"Book {book_id} failed with error: {error_message} (total errors: {error_count})")
        return result
    
    def get_statistics_summary(self, db: Session) -> Dict[str, Any]:
        """Obtener resumen estadístico optimizado"""
        try:
            # Usar una sola consulta con agregaciones
            stats = db.query(
                func.count(BookProcessingHistory.id).label('total'),
                func.sum(func.case([(BookProcessingHistory.status == BookStatus.COMPLETED, 1)], else_=0)).label('completed'),
                func.sum(func.case([(BookProcessingHistory.status.in_([BookStatus.QUEUED, BookStatus.PROCESSING]), 1)], else_=0)).label('in_progress'),
                func.sum(func.case([(BookProcessingHistory.status == BookStatus.FAILED, 1)], else_=0)).label('failed'),
                func.avg(BookProcessingHistory.semantic_score_avg).label('avg_quality')
            ).first()
            
            total = stats.total or 0
            completed = stats.completed or 0
            in_progress = stats.in_progress or 0
            failed = stats.failed or 0
            avg_quality = float(stats.avg_quality or 0.0)
            
            return {
                'total_books': total,
                'completed_books': completed,
                'in_progress_books': in_progress,
                'failed_books': failed,
                'success_rate': completed / total if total > 0 else 0,
                'average_quality_score': avg_quality
            }
            
        except SQLAlchemyError as e:
            logger.error(f"Error getting book statistics: {e}")
            return {}
    
    def _apply_filters(self, query: Query, filters: Dict[str, Any]) -> Query:
        """Aplicar filtros específicos para BookProcessingHistory"""
        if 'status' in filters:
            query = query.filter(BookProcessingHistory.status == filters['status'])
        if 'source_lang' in filters:
            query = query.filter(BookProcessingHistory.source_lang == filters['source_lang'])
        if 'date_from' in filters:
            query = query.filter(BookProcessingHistory.started_at >= filters['date_from'])
        if 'date_to' in filters:
            query = query.filter(BookProcessingHistory.started_at <= filters['date_to'])
        if 'min_quality_score' in filters:
            query = query.filter(BookProcessingHistory.semantic_score_avg >= filters['min_quality_score'])
        if 'has_errors' in filters:
            if filters['has_errors']:
                query = query.filter(BookProcessingHistory.error_count > 0)
            else:
                query = query.filter(BookProcessingHistory.error_count == 0)
        if 'search' in filters:
            search_term = f"%{filters['search']}%"
            query = query.filter(
                or_(
                    BookProcessingHistory.book_id.ilike(search_term),
                    BookProcessingHistory.title.ilike(search_term)
                )
            )
        return query

# ==========================================
# ERROR PATTERN CRUD
# ==========================================

class ErrorPatternCRUD(BaseCRUD):
    """CRUD operations for ErrorPattern"""
    
    def __init__(self):
        super().__init__(ErrorPattern, ErrorPatternCreate, ErrorPatternUpdate)
    
    def get_by_type(
        self, 
        db: Session, 
        *, 
        pattern_type: PatternType,
        active_only: bool = True
    ) -> List[ErrorPattern]:
        """Obtener patrones por tipo"""
        query = db.query(ErrorPattern).filter(
            ErrorPattern.pattern_type == pattern_type.value
        )
        
        if active_only:
            query = query.filter(ErrorPattern.is_active == True)
        
        return query.order_by(desc(ErrorPattern.effectiveness_score)).all()
    
    def update_effectiveness(
        self,
        db: Session,
        *,
        pattern_id: int,
        success: bool,
        user_id: str = "system"
    ) -> Optional[ErrorPattern]:
        """Actualizar efectividad de patrón de forma atómica"""
        try:
            with self._transaction_scope(db):
                pattern = self.get(db, pattern_id, for_update=True)
                if not pattern:
                    raise NotFoundError(f"Pattern {pattern_id} not found")
                
                # Actualizar estadísticas de forma atómica
                pattern.usage_count += 1
                pattern.last_seen = datetime.now()
                
                # Recalcular success rate
                if success:
                    total_successes = (pattern.success_rate * (pattern.usage_count - 1)) + 1
                    pattern.success_rate = total_successes / pattern.usage_count
                else:
                    total_successes = pattern.success_rate * (pattern.usage_count - 1)
                    pattern.success_rate = total_successes / pattern.usage_count
                
                # Recalcular effectiveness score (fórmula empresarial)
                pattern.effectiveness_score = (
                    pattern.success_rate * 0.7 +  # 70% peso al success rate
                    min(pattern.usage_count / 100, 1.0) * 0.2 +  # 20% peso a la experiencia
                    (1 - pattern.false_positive_rate) * 0.1  # 10% peso a baja tasa de falsos positivos
                )
                
                db.flush()
                
                logger.info(f"Pattern {pattern_id} effectiveness updated: {pattern.effectiveness_score:.3f}")
                
                return pattern
                
        except NotFoundError:
            raise
        except SQLAlchemyError as e:
            logger.error(f"Error updating pattern effectiveness: {e}")
            raise CRUDError(f"Failed to update pattern effectiveness: {e}")
    
    def _apply_filters(self, query: Query, filters: Dict[str, Any]) -> Query:
        """Aplicar filtros específicos para ErrorPattern"""
        if 'pattern_type' in filters:
            query = query.filter(ErrorPattern.pattern_type == filters['pattern_type'])
        if 'is_active' in filters:
            query = query.filter(ErrorPattern.is_active == filters['is_active'])
        if 'min_effectiveness' in filters:
            query = query.filter(ErrorPattern.effectiveness_score >= filters['min_effectiveness'])
        if 'created_by' in filters:
            query = query.filter(ErrorPattern.created_by == filters['created_by'])
        if 'search' in filters:
            search_term = f"%{filters['search']}%"
            query = query.filter(
                or_(
                    ErrorPattern.pattern_content.ilike(search_term),
                    ErrorPattern.description.ilike(search_term)
                )
            )
        return query

# ==========================================
# AUDIT LOG CRUD
# ==========================================

class AuditLogCRUD(BaseCRUD):
    """CRUD operations for AuditLog"""
    
    def __init__(self):
        super().__init__(AuditLog, AuditLogCreate, AuditLogCreate)  # No update schema for audit logs
    
    def get_by_book(
        self, 
        db: Session, 
        *, 
        book_id: str
    ) -> List[AuditLog]:
        """Obtener logs de auditoría por libro"""
        return db.query(AuditLog).filter(
            AuditLog.book_id == book_id
        ).order_by(AuditLog.created_at).all()
    
    def get_critical_alerts(
        self, 
        db: Session, 
        *, 
        days: int = 1
    ) -> List[AuditLog]:
        """Obtener auditorías con alertas críticas"""
        cutoff_date = datetime.now() - timedelta(days=days)
        return db.query(AuditLog).filter(
            and_(
                AuditLog.created_at >= cutoff_date,
                AuditLog.alerts_count > 0
            )
        ).order_by(desc(AuditLog.created_at)).all()
    
    def get_quality_trends(
        self, 
        db: Session, 
        *, 
        days: int = 30
    ) -> List[Dict[str, Any]]:
        """Obtener tendencias de calidad por día"""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        results = db.query(
            func.date(AuditLog.created_at).label('date'),
            func.avg(AuditLog.quality_score).label('avg_quality'),
            func.count(AuditLog.id).label('count')
        ).filter(
            AuditLog.created_at >= cutoff_date
        ).group_by(
            func.date(AuditLog.created_at)
        ).order_by('date').all()
        
        return [
            {
                'date': result.date,
                'average_quality': float(result.avg_quality),
                'count': result.count
            }
            for result in results
        ]
    
    def _apply_filters(self, query: Query, filters: Dict[str, Any]) -> Query:
        """Aplicar filtros específicos para AuditLog"""
        if 'book_id' in filters:
            query = query.filter(AuditLog.book_id == filters['book_id'])
        if 'phase_name' in filters:
            query = query.filter(AuditLog.phase_name == filters['phase_name'])
        if 'min_quality_score' in filters:
            query = query.filter(AuditLog.quality_score >= filters['min_quality_score'])
        if 'has_alerts' in filters:
            if filters['has_alerts']:
                query = query.filter(AuditLog.alerts_count > 0)
            else:
                query = query.filter(AuditLog.alerts_count == 0)
        if 'date_from' in filters:
            query = query.filter(AuditLog.created_at >= filters['date_from'])
        if 'date_to' in filters:
            query = query.filter(AuditLog.created_at <= filters['date_to'])
        return query

# ==========================================
# TERMINLOGY SUGGESTION CRUD
# ==========================================

class TerminologyCRUD(BaseCRUD):
    """CRUD operations for TerminologySuggestion"""
    
    def __init__(self):
        super().__init__(TerminologySuggestion, TerminologySuggestionCreate, TerminologySuggestionUpdate)
    
    def get_pending_review(
        self, 
        db: Session, 
        *, 
        limit: int = 50
    ) -> List[TerminologySuggestion]:
        """Obtener sugerencias pendientes de revisión"""
        return db.query(TerminologySuggestion).filter(
            TerminologySuggestion.reviewed == False
        ).order_by(
            desc(TerminologySuggestion.priority),
            desc(TerminologySuggestion.confidence_score)
        ).limit(limit).all()
    
    def _apply_filters(self, query: Query, filters: Dict[str, Any]) -> Query:
        """Aplicar filtros específicos para TerminologySuggestion"""
        if 'book_id' in filters:
            query = query.filter(TerminologySuggestion.book_id == filters['book_id'])
        if 'glossary_id' in filters:
            query = query.filter(TerminologySuggestion.glossary_id == filters['glossary_id'])
        if 'priority' in filters:
            query = query.filter(TerminologySuggestion.priority == filters['priority'])
        if 'reviewed' in filters:
            query = query.filter(TerminologySuggestion.reviewed == filters['reviewed'])
        if 'approved' in filters:
            query = query.filter(TerminologySuggestion.approved == filters['approved'])
        if 'applied' in filters:
            query = query.filter(TerminologySuggestion.applied == filters['applied'])
        if 'search' in filters:
            search_term = f"%{filters['search']}%"
            query = query.filter(
                or_(
                    TerminologySuggestion.source_term.ilike(search_term),
                    TerminologySuggestion.target_term.ilike(search_term),
                    TerminologySuggestion.context.ilike(search_term)
                )
            )
        return query

# ==========================================
# EDITORIAL REVIEW CRUD
# ==========================================

class EditorialReviewCRUD(BaseCRUD):
    """CRUD operations for EditorialReview"""
    
    def __init__(self):
        super().__init__(EditorialReview, EditorialReviewCreate, EditorialReviewUpdate)
    
    def get_by_book(
        self, 
        db: Session, 
        *, 
        book_id: str,
        severity_filter: Optional[Severity] = None
    ) -> List[EditorialReview]:
        """Obtener revisiones por libro"""
        query = db.query(EditorialReview).filter(
            EditorialReview.book_id == book_id
        )
        
        if severity_filter:
            query = query.filter(EditorialReview.severity == severity_filter.value)
        
        return query.order_by(
            EditorialReview.severity,
            EditorialReview.section_number
        ).all()
    
    def _apply_filters(self, query: Query, filters: Dict[str, Any]) -> Query:
        """Aplicar filtros específicos para EditorialReview"""
        if 'book_id' in filters:
            query = query.filter(EditorialReview.book_id == filters['book_id'])
        if 'severity' in filters:
            query = query.filter(EditorialReview.severity == filters['severity'])
        if 'reviewed' in filters:
            query = query.filter(EditorialReview.reviewed == filters['reviewed'])
        if 'resolved' in filters:
            query = query.filter(EditorialReview.resolved == filters['resolved'])
        return query

# ==========================================
# USAGE STATISTICS CRUD
# ==========================================

class UsageStatisticsCRUD(BaseCRUD):
    """CRUD operations for UsageStatistic"""
    
    def __init__(self):
        super().__init__(UsageStatistic, UsageStatisticsCreate, UsageStatisticsUpdate)
    
    def get_by_date(self, db: Session, *, target_date: date) -> Optional[UsageStatistic]:
        """Obtener estadísticas por fecha"""
        return db.query(UsageStatistic).filter(
            UsageStatistic.date == target_date
        ).first()
    
    def _apply_filters(self, query: Query, filters: Dict[str, Any]) -> Query:
        """Aplicar filtros específicos para UsageStatistic"""
        if 'date_from' in filters:
            query = query.filter(UsageStatistic.date >= filters['date_from'])
        if 'date_to' in filters:
            query = query.filter(UsageStatistic.date <= filters['date_to'])
        return query

# ==========================================
# CRUD FACTORY
# ==========================================

class CRUDFactory:
    """Factory para crear instancias CRUD con dependency injection"""
    
    def __init__(self):
        self._instances = {}
    
    def get_system_config_crud(self) -> SystemConfigCRUD:
        if 'system_config' not in self._instances:
            self._instances['system_config'] = SystemConfigCRUD()
        return self._instances['system_config']
    
    def get_book_crud(self) -> BookCRUD:
        if 'book' not in self._instances:
            self._instances['book'] = BookCRUD()
        return self._instances['book']
    
    def get_error_pattern_crud(self) -> ErrorPatternCRUD:
        if 'error_pattern' not in self._instances:
            self._instances['error_pattern'] = ErrorPatternCRUD()
        return self._instances['error_pattern']
    
    def get_audit_log_crud(self) -> AuditLogCRUD:
        if 'audit_log' not in self._instances:
            self._instances['audit_log'] = AuditLogCRUD()
        return self._instances['audit_log']
    
    def get_terminology_crud(self) -> TerminologyCRUD:
        if 'terminology' not in self._instances:
            self._instances['terminology'] = TerminologyCRUD()
        return self._instances['terminology']
    
    def get_editorial_review_crud(self) -> EditorialReviewCRUD:
        if 'editorial_review' not in self._instances:
            self._instances['editorial_review'] = EditorialReviewCRUD()
        return self._instances['editorial_review']
    
    def get_usage_statistics_crud(self) -> UsageStatisticsCRUD:
        if 'usage_statistics' not in self._instances:
            self._instances['usage_statistics'] = UsageStatisticsCRUD()
        return self._instances['usage_statistics']
    
    def get_performance_metrics(self) -> PerformanceMetrics:
        return performance_metrics
    
    def reset_metrics(self):
        """Resetear todas las métricas"""
        performance_metrics.reset_metrics()

# ==========================================
# INSTANCIA GLOBAL FACTORY (para compatibilidad con código existente)
# NOTA: Se recomienda usar dependency injection en lugar de estas instancias globales
# ==========================================

crud_factory = CRUDFactory()

# Instancias individuales para compatibilidad
system_config_crud = crud_factory.get_system_config_crud()
book_crud = crud_factory.get_book_crud()
error_pattern_crud = crud_factory.get_error_pattern_crud()
audit_log_crud = crud_factory.get_audit_log_crud()
terminology_crud = crud_factory.get_terminology_crud()
editorial_review_crud = crud_factory.get_editorial_review_crud()
usage_statistics_crud = crud_factory.get_usage_statistics_crud()