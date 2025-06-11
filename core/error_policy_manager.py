"""
ErrorPolicyManager - Sistema de Manejo Inteligente de Errores v2.2
===================================================================

Componente enterprise-grade para manejo de errores con circuit breakers,
políticas de reintento diferenciadas y escalación automática.

Características:
- Circuit breaker pattern para APIs externas
- Reintentos con backoff exponencial
- Escalación automática por severidad
- Métricas y monitoreo en tiempo real
- Configuración flexible y extensible

Autor: Sistema de Traducción Académica v2.2
"""

import logging
import time
import json
import asyncio
import hashlib
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any, Protocol, Union
from enum import Enum
from functools import wraps
import traceback
from contextlib import asynccontextmanager

# Standard library imports
import sqlite3
from pathlib import Path
import threading


class ErrorSeverity(Enum):
    """Niveles de severidad de errores"""
    LOW = "low"           # Similarity 0.75-0.85
    MEDIUM = "medium"     # Similarity 0.60-0.75  
    HIGH = "high"         # Similarity 0.40-0.60
    CRITICAL = "critical" # Similarity <0.40


class CircuitBreakerState(Enum):
    """Estados del circuit breaker"""
    CLOSED = "closed"         # Funcionamiento normal
    OPEN = "open"             # Bloqueando todas las llamadas
    HALF_OPEN = "half_open"   # Permitiendo llamadas de prueba


class RetryStrategy(Enum):
    """Estrategias de reintento disponibles"""
    IMMEDIATE = "immediate"                # Reintento inmediato
    LINEAR_BACKOFF = "linear_backoff"      # Incremento lineal
    EXPONENTIAL_BACKOFF = "exponential_backoff"  # Incremento exponencial
    CONSERVATIVE = "conservative"          # Parámetros más conservadores
    CHUNKED = "chunked"                   # Procesamiento por chunks


@dataclass
class ErrorContext:
    """Contexto detallado de un error"""
    error_id: str
    timestamp: datetime
    component: str
    function_name: str
    error_type: str
    error_message: str
    severity: ErrorSeverity
    similarity_score: Optional[float] = None
    book_id: Optional[str] = None
    phase: Optional[str] = None
    stack_trace: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetryAttempt:
    """Información de un intento de reintento"""
    attempt_number: int
    timestamp: datetime
    delay_seconds: float
    strategy: RetryStrategy
    success: bool
    error_message: Optional[str] = None


@dataclass
class CircuitBreakerMetrics:
    """Métricas del circuit breaker"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    state_changes: int = 0
    last_failure_time: Optional[datetime] = None
    consecutive_failures: int = 0
    average_response_time: float = 0.0


@dataclass
class ErrorPolicyConfig:
    """Configuración de políticas de error"""
    # Circuit Breaker Settings
    failure_threshold: int = 5
    recovery_timeout: int = 60
    half_open_max_calls: int = 3
    
    # Retry Settings
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    backoff_multiplier: float = 2.0
    
    # Severity Thresholds
    similarity_thresholds: Dict[ErrorSeverity, float] = field(default_factory=lambda: {
        ErrorSeverity.LOW: 0.75,
        ErrorSeverity.MEDIUM: 0.60,
        ErrorSeverity.HIGH: 0.40,
        ErrorSeverity.CRITICAL: 0.0
    })
    
    # Escalation Settings
    manual_review_threshold: ErrorSeverity = ErrorSeverity.CRITICAL
    auto_escalation_enabled: bool = True
    
    # Monitoring Settings
    metrics_retention_days: int = 30
    alert_cooldown_minutes: int = 15


class Logger(Protocol):
    """Protocol para logging dependency injection"""
    def info(self, msg: str) -> None: ...
    def warning(self, msg: str) -> None: ...
    def error(self, msg: str) -> None: ...
    def debug(self, msg: str) -> None: ...


class DatabaseConnector(Protocol):
    """Protocol para database dependency injection"""
    def execute_query(self, query: str, params: tuple = ()) -> List[Dict]: ...
    def execute_update(self, query: str, params: tuple = ()) -> int: ...


class AlertManager(Protocol):
    """Protocol para alertas dependency injection"""
    def send_alert(self, error_context: ErrorContext) -> None: ...


class CircuitBreakerError(Exception):
    """Exception lanzada cuando el circuit breaker está abierto"""
    pass


class ManualReviewRequiredError(Exception):
    """Exception lanzada cuando se requiere revisión manual"""
    pass


class CircuitBreaker:
    """
    Implementación enterprise-grade del Circuit Breaker pattern
    
    Características:
    - Estados bien definidos (Closed, Open, Half-Open)
    - Métricas detalladas y monitoreo
    - Configuración flexible
    - Thread-safe operations
    """
    
    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        half_open_max_calls: int = 3,
        logger: Optional[Logger] = None
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        self.logger = logger or logging.getLogger(__name__)
        
        # State management
        self._state = CircuitBreakerState.CLOSED
        self._lock = threading.RLock()
        
        # Metrics
        self.metrics = CircuitBreakerMetrics()
        
        # State tracking
        self._consecutive_failures = 0
        self._last_failure_time: Optional[datetime] = None
        self._half_open_calls = 0
        
        self.logger.info(f"🔧 Circuit Breaker '{name}' inicializado")
    
    @property
    def state(self) -> CircuitBreakerState:
        """Estado actual del circuit breaker"""
        with self._lock:
            return self._state
    
    def _should_attempt_reset(self) -> bool:
        """Determina si debería intentar reset desde estado OPEN"""
        if not self._last_failure_time:
            return False
        
        time_since_failure = datetime.now() - self._last_failure_time
        return time_since_failure.total_seconds() >= self.recovery_timeout
    
    def _record_success(self) -> None:
        """Registra una operación exitosa"""
        with self._lock:
            self.metrics.successful_requests += 1
            self.metrics.total_requests += 1
            self._consecutive_failures = 0
            
            if self._state == CircuitBreakerState.HALF_OPEN:
                self._half_open_calls += 1
                if self._half_open_calls >= self.half_open_max_calls:
                    self._transition_to_closed()
    
    def _record_failure(self, error: Exception) -> None:
        """Registra una operación fallida"""
        with self._lock:
            self.metrics.failed_requests += 1
            self.metrics.total_requests += 1
            self._consecutive_failures += 1
            self._last_failure_time = datetime.now()
            self.metrics.last_failure_time = self._last_failure_time
            
            if self._state == CircuitBreakerState.CLOSED:
                if self._consecutive_failures >= self.failure_threshold:
                    self._transition_to_open()
            elif self._state == CircuitBreakerState.HALF_OPEN:
                self._transition_to_open()
    
    def _transition_to_open(self) -> None:
        """Transición a estado OPEN"""
        previous_state = self._state
        self._state = CircuitBreakerState.OPEN
        self.metrics.state_changes += 1
        self._half_open_calls = 0
        
        self.logger.warning(
            f"🔴 Circuit Breaker '{self.name}' -> OPEN "
            f"(failures: {self._consecutive_failures}/{self.failure_threshold})"
        )
    
    def _transition_to_half_open(self) -> None:
        """Transición a estado HALF_OPEN"""
        previous_state = self._state
        self._state = CircuitBreakerState.HALF_OPEN
        self.metrics.state_changes += 1
        self._half_open_calls = 0
        
        self.logger.info(f"🟡 Circuit Breaker '{self.name}' -> HALF_OPEN (test mode)")
    
    def _transition_to_closed(self) -> None:
        """Transición a estado CLOSED"""
        previous_state = self._state
        self._state = CircuitBreakerState.CLOSED
        self.metrics.state_changes += 1
        self._consecutive_failures = 0
        
        self.logger.info(f"🟢 Circuit Breaker '{self.name}' -> CLOSED (recovered)")
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Ejecuta función protegida por circuit breaker
        
        Args:
            func: Función a ejecutar
            *args, **kwargs: Argumentos para la función
            
        Returns:
            Resultado de la función
            
        Raises:
            CircuitBreakerError: Si el circuit breaker está abierto
        """
        with self._lock:
            if self._state == CircuitBreakerState.OPEN:
                if self._should_attempt_reset():
                    self._transition_to_half_open()
                else:
                    raise CircuitBreakerError(
                        f"Circuit breaker '{self.name}' is OPEN. "
                        f"Next attempt in {self.recovery_timeout - (datetime.now() - self._last_failure_time).total_seconds():.1f}s"
                    )
            
            elif self._state == CircuitBreakerState.HALF_OPEN:
                if self._half_open_calls >= self.half_open_max_calls:
                    raise CircuitBreakerError(
                        f"Circuit breaker '{self.name}' HALF_OPEN max calls reached"
                    )
        
        # Execute protected function
        start_time = time.time()
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            execution_time = time.time() - start_time
            self._record_success()
            
            self.logger.debug(
                f"✅ Circuit Breaker '{self.name}' call succeeded "
                f"({execution_time:.3f}s)"
            )
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self._record_failure(e)
            
            self.logger.error(
                f"❌ Circuit Breaker '{self.name}' call failed "
                f"({execution_time:.3f}s): {str(e)}"
            )
            
            raise
    
    def get_metrics(self) -> CircuitBreakerMetrics:
        """Obtiene métricas actuales del circuit breaker"""
        with self._lock:
            return CircuitBreakerMetrics(
                total_requests=self.metrics.total_requests,
                successful_requests=self.metrics.successful_requests,
                failed_requests=self.metrics.failed_requests,
                state_changes=self.metrics.state_changes,
                last_failure_time=self.metrics.last_failure_time,
                consecutive_failures=self._consecutive_failures,
                average_response_time=self.metrics.average_response_time
            )


class RetryManager:
    """
    Gestor de políticas de reintento con backoff inteligente
    
    Características:
    - Múltiples estrategias de backoff
    - Jitter para evitar thundering herd
    - Límites configurables
    - Logging detallado
    """
    
    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        backoff_multiplier: float = 2.0,
        logger: Optional[Logger] = None
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_multiplier = backoff_multiplier
        self.logger = logger or logging.getLogger(__name__)
    
    def _calculate_delay(
        self,
        attempt: int,
        strategy: RetryStrategy,
        base_delay: Optional[float] = None
    ) -> float:
        """Calcula delay para reintento basado en estrategia"""
        base = base_delay or self.base_delay
        
        if strategy == RetryStrategy.IMMEDIATE:
            return 0.0
        elif strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = base * attempt
        elif strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            delay = base * (self.backoff_multiplier ** (attempt - 1))
        elif strategy == RetryStrategy.CONSERVATIVE:
            # Para errores semánticos: más conservador
            delay = base * (3.0 ** (attempt - 1))
        else:
            delay = base
        
        # Add jitter (±25%) to prevent thundering herd
        import random
        jitter_factor = 0.75 + (random.random() * 0.5)  # 0.75 to 1.25
        delay *= jitter_factor
        
        return min(delay, self.max_delay)
    
    async def retry_with_policy(
        self,
        func: Callable,
        error_context: ErrorContext,
        strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF,
        custom_max_retries: Optional[int] = None,
        *args,
        **kwargs
    ) -> Any:
        """
        Ejecuta función con política de reintento
        
        Args:
            func: Función a ejecutar
            error_context: Contexto del error
            strategy: Estrategia de reintento
            custom_max_retries: Override de max_retries
            *args, **kwargs: Argumentos para la función
            
        Returns:
            Resultado de la función
            
        Raises:
            Exception: Última excepción si todos los intentos fallan
        """
        max_attempts = custom_max_retries or self.max_retries
        attempts: List[RetryAttempt] = []
        last_exception = None
        
        for attempt in range(1, max_attempts + 2):  # +1 for initial attempt
            attempt_start = datetime.now()
            
            try:
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                # Success!
                attempts.append(RetryAttempt(
                    attempt_number=attempt,
                    timestamp=attempt_start,
                    delay_seconds=0.0,
                    strategy=strategy,
                    success=True
                ))
                
                if attempt > 1:
                    self.logger.info(
                        f"✅ Retry succeeded on attempt {attempt}/{max_attempts + 1} "
                        f"for {error_context.component}.{error_context.function_name}"
                    )
                
                return result
                
            except Exception as e:
                last_exception = e
                
                attempts.append(RetryAttempt(
                    attempt_number=attempt,
                    timestamp=attempt_start,
                    delay_seconds=0.0,
                    strategy=strategy,
                    success=False,
                    error_message=str(e)
                ))
                
                if attempt <= max_attempts:
                    delay = self._calculate_delay(attempt, strategy)
                    
                    self.logger.warning(
                        f"⚠️ Attempt {attempt}/{max_attempts + 1} failed for "
                        f"{error_context.component}.{error_context.function_name}: {str(e)}. "
                        f"Retrying in {delay:.2f}s"
                    )
                    
                    if delay > 0:
                        await asyncio.sleep(delay)
                else:
                    self.logger.error(
                        f"❌ All {max_attempts + 1} attempts failed for "
                        f"{error_context.component}.{error_context.function_name}"
                    )
        
        # Store retry history in error context
        error_context.metadata['retry_attempts'] = [
            {
                'attempt': a.attempt_number,
                'timestamp': a.timestamp.isoformat(),
                'strategy': a.strategy.value,
                'success': a.success,
                'error': a.error_message
            }
            for a in attempts
        ]
        
        raise last_exception


class ErrorPolicyManager:
    """
    Sistema Enterprise-Grade de Manejo Inteligente de Errores
    
    Características principales:
    - Circuit breakers para APIs externas  
    - Políticas de reintento diferenciadas por severidad
    - Escalación automática a revisión manual
    - Métricas y monitoreo en tiempo real
    - Configuración flexible y extensible
    
    Principios de diseño:
    - SOLID principles aplicados
    - Dependency injection limpia
    - Type hints completos
    - Logging estructurado
    - Error recovery automático
    """
    
    def __init__(
        self,
        config: Optional[ErrorPolicyConfig] = None,
        logger: Optional[Logger] = None,
        db_connector: Optional[DatabaseConnector] = None,
        alert_manager: Optional[AlertManager] = None
    ):
        """
        Inicializa el ErrorPolicyManager
        
        Args:
            config: Configuración de políticas de error
            logger: Logger para dependency injection
            db_connector: Conector de base de datos
            alert_manager: Gestor de alertas
        """
        self.config = config or ErrorPolicyConfig()
        self.logger = logger or logging.getLogger(__name__)
        self.db_connector = db_connector
        self.alert_manager = alert_manager
        
        # Initialize components
        self.retry_manager = RetryManager(
            max_retries=self.config.max_retries,
            base_delay=self.config.base_delay,
            max_delay=self.config.max_delay,
            backoff_multiplier=self.config.backoff_multiplier,
            logger=self.logger
        )
        
        # Circuit breakers for external APIs
        self.circuit_breakers: Dict[str, CircuitBreaker] = {
            'deepl_api': CircuitBreaker(
                name='DeepL API',
                failure_threshold=self.config.failure_threshold,
                recovery_timeout=self.config.recovery_timeout,
                logger=self.logger
            ),
            'claude_api': CircuitBreaker(
                name='Claude API', 
                failure_threshold=self.config.failure_threshold,
                recovery_timeout=self.config.recovery_timeout,
                logger=self.logger
            ),
            'abbyy_api': CircuitBreaker(
                name='ABBYY API',
                failure_threshold=self.config.failure_threshold,
                recovery_timeout=self.config.recovery_timeout,
                logger=self.logger
            )
        }
        
        # Error tracking
        self.error_history: List[ErrorContext] = []
        self._error_counts_by_type: Dict[str, int] = {}
        self._last_alert_times: Dict[str, datetime] = {}
        
        # Initialize database if connector provided
        if self.db_connector:
            self._initialize_error_tracking_tables()
        
        self.logger.info("🚀 ErrorPolicyManager inicializado con configuración enterprise")
    
    def _initialize_error_tracking_tables(self) -> None:
        """Inicializa tablas de tracking de errores en base de datos"""
        try:
            create_table_query = """
            CREATE TABLE IF NOT EXISTS error_tracking (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                error_id TEXT UNIQUE NOT NULL,
                timestamp TEXT NOT NULL,
                component TEXT NOT NULL,
                function_name TEXT NOT NULL,
                error_type TEXT NOT NULL,
                error_message TEXT NOT NULL,
                severity TEXT NOT NULL,
                similarity_score REAL,
                book_id TEXT,
                phase TEXT,
                stack_trace TEXT,
                metadata TEXT,
                resolved BOOLEAN DEFAULT FALSE,
                resolution_strategy TEXT,
                retry_attempts INTEGER DEFAULT 0,
                escalated BOOLEAN DEFAULT FALSE,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
            self.db_connector.execute_update(create_table_query)
            
            # Create indices for performance
            indices = [
                "CREATE INDEX IF NOT EXISTS idx_error_tracking_timestamp ON error_tracking(timestamp)",
                "CREATE INDEX IF NOT EXISTS idx_error_tracking_severity ON error_tracking(severity)",
                "CREATE INDEX IF NOT EXISTS idx_error_tracking_component ON error_tracking(component)",
                "CREATE INDEX IF NOT EXISTS idx_error_tracking_resolved ON error_tracking(resolved)"
            ]
            
            for index_query in indices:
                self.db_connector.execute_update(index_query)
                
            self.logger.info("✅ Tablas de tracking de errores inicializadas")
            
        except Exception as e:
            self.logger.error(f"❌ Error inicializando tablas de tracking: {e}")
    
    def _determine_error_severity(
        self,
        similarity_score: Optional[float] = None,
        error_type: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> ErrorSeverity:
        """
        Determina la severidad del error basado en múltiples factores
        
        Args:
            similarity_score: Score de similitud semántica
            error_type: Tipo de error
            context: Contexto adicional
            
        Returns:
            Severidad determinada del error
        """
        # Criterio principal: similarity score (como especifica la documentación)
        if similarity_score is not None:
            for severity, threshold in sorted(
                self.config.similarity_thresholds.items(),
                key=lambda x: x[1],
                reverse=True
            ):
                if similarity_score >= threshold:
                    return severity
            return ErrorSeverity.CRITICAL
        
        # Criterios secundarios basados en tipo de error
        critical_errors = [
            'ConnectionError', 'TimeoutError', 'CircuitBreakerError',
            'ManualReviewRequiredError', 'MemoryError'
        ]
        
        high_errors = [
            'ValueError', 'KeyError', 'AttributeError',
            'ProcessingError', 'ValidationError'
        ]
        
        if error_type in critical_errors:
            return ErrorSeverity.CRITICAL
        elif error_type in high_errors:
            return ErrorSeverity.HIGH
        else:
            return ErrorSeverity.MEDIUM
    
    def _generate_error_id(self, error_context: ErrorContext) -> str:
        """Genera ID único para el error"""
        content = f"{error_context.component}:{error_context.function_name}:{error_context.error_type}:{error_context.timestamp.isoformat()}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def _should_send_alert(self, error_context: ErrorContext) -> bool:
        """Determina si debe enviar alerta basado en cooldown y severidad"""
        alert_key = f"{error_context.component}:{error_context.error_type}"
        
        if alert_key in self._last_alert_times:
            time_since_last = datetime.now() - self._last_alert_times[alert_key]
            cooldown = timedelta(minutes=self.config.alert_cooldown_minutes)
            
            if time_since_last < cooldown:
                return False
        
        # Always alert for critical errors
        if error_context.severity == ErrorSeverity.CRITICAL:
            return True
        
        # Alert for high errors if they repeat
        if error_context.severity == ErrorSeverity.HIGH:
            error_count = self._error_counts_by_type.get(alert_key, 0)
            return error_count >= 3
        
        return False
    
    def _store_error_context(self, error_context: ErrorContext) -> None:
        """Almacena contexto de error en base de datos y memoria"""
        # Store in memory
        self.error_history.append(error_context)
        
        # Update counters
        error_key = f"{error_context.component}:{error_context.error_type}"
        self._error_counts_by_type[error_key] = self._error_counts_by_type.get(error_key, 0) + 1
        
        # Store in database if available
        if self.db_connector:
            try:
                insert_query = """
                INSERT INTO error_tracking 
                (error_id, timestamp, component, function_name, error_type, 
                 error_message, severity, similarity_score, book_id, phase, 
                 stack_trace, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """
                
                params = (
                    error_context.error_id,
                    error_context.timestamp.isoformat(),
                    error_context.component,
                    error_context.function_name,
                    error_context.error_type,
                    error_context.error_message,
                    error_context.severity.value,
                    error_context.similarity_score,
                    error_context.book_id,
                    error_context.phase,
                    error_context.stack_trace,
                    json.dumps(error_context.metadata) if error_context.metadata else None
                )
                
                self.db_connector.execute_update(insert_query, params)
                
            except Exception as e:
                self.logger.error(f"❌ Error almacenando contexto en BD: {e}")
    
    def _get_retry_strategy_for_severity(self, severity: ErrorSeverity) -> RetryStrategy:
        """
        Determina estrategia de reintento basada en severidad
        
        Según documentación:
        - LOW (0.75-0.85): Reintento directo hasta 3 veces
        - MEDIUM (0.60-0.75): Claude más conservador
        - HIGH (0.40-0.60): Procesamiento por chunks más pequeños  
        - CRITICAL (<0.40): Escalación a revisión manual
        """
        strategy_map = {
            ErrorSeverity.LOW: RetryStrategy.IMMEDIATE,
            ErrorSeverity.MEDIUM: RetryStrategy.CONSERVATIVE,
            ErrorSeverity.HIGH: RetryStrategy.CHUNKED,
            ErrorSeverity.CRITICAL: RetryStrategy.EXPONENTIAL_BACKOFF
        }
        
        return strategy_map.get(severity, RetryStrategy.EXPONENTIAL_BACKOFF)
    
    def _get_max_retries_for_severity(self, severity: ErrorSeverity) -> int:
        """Determina número máximo de reintentos por severidad"""
        retry_map = {
            ErrorSeverity.LOW: 3,
            ErrorSeverity.MEDIUM: 2,
            ErrorSeverity.HIGH: 1,
            ErrorSeverity.CRITICAL: 0  # No retry, escalate immediately
        }
        
        return retry_map.get(severity, 1)
    
    async def handle_error(
        self,
        error: Exception,
        component: str,
        function_name: str,
        similarity_score: Optional[float] = None,
        book_id: Optional[str] = None,
        phase: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ErrorContext:
        """
        Manejo principal de errores con políticas diferenciadas
        
        Args:
            error: Excepción capturada
            component: Nombre del componente que falló
            function_name: Función donde ocurrió el error
            similarity_score: Score de similitud semántica (si aplica)
            book_id: ID del libro siendo procesado
            phase: Fase del pipeline donde ocurrió el error
            metadata: Metadatos adicionales
            
        Returns:
            Contexto del error procesado
        """
        # Create error context
        error_context = ErrorContext(
            error_id="",  # Will be generated
            timestamp=datetime.now(),
            component=component,
            function_name=function_name,
            error_type=type(error).__name__,
            error_message=str(error),
            severity=self._determine_error_severity(similarity_score, type(error).__name__),
            similarity_score=similarity_score,
            book_id=book_id,
            phase=phase,
            stack_trace=traceback.format_exc(),
            metadata=metadata or {}
        )
        
        # Generate unique error ID
        error_context.error_id = self._generate_error_id(error_context)
        
        # Log error with appropriate level
        log_msg = (
            f"🚨 Error {error_context.severity.value.upper()} en "
            f"{component}.{function_name}: {str(error)}"
        )
        
        if book_id:
            log_msg += f" [Book: {book_id}]"
        if phase:
            log_msg += f" [Phase: {phase}]"
        if similarity_score is not None:
            log_msg += f" [Similarity: {similarity_score:.3f}]"
        
        if error_context.severity == ErrorSeverity.CRITICAL:
            self.logger.error(log_msg)
        elif error_context.severity == ErrorSeverity.HIGH:
            self.logger.warning(log_msg)
        else:
            self.logger.info(log_msg)
        
        # Store error context
        self._store_error_context(error_context)
        
        # Send alert if needed
        if self._should_send_alert(error_context) and self.alert_manager:
            try:
                self.alert_manager.send_alert(error_context)
                alert_key = f"{error_context.component}:{error_context.error_type}"
                self._last_alert_times[alert_key] = datetime.now()
            except Exception as alert_error:
                self.logger.error(f"❌ Error enviando alerta: {alert_error}")
        
        return error_context
    
    async def execute_with_policy(
        self,
        func: Callable,
        component: str,
        function_name: str,
        circuit_breaker_name: Optional[str] = None,
        similarity_score: Optional[float] = None,
        book_id: Optional[str] = None,
        phase: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        *args,
        **kwargs
    ) -> Any:
        """
        Ejecuta función con todas las políticas de error aplicadas
        
        Args:
            func: Función a ejecutar
            component: Nombre del componente
            function_name: Nombre de la función
            circuit_breaker_name: Nombre del circuit breaker a usar
            similarity_score: Score de similitud (si aplica)
            book_id: ID del libro
            phase: Fase del pipeline
            metadata: Metadatos adicionales
            *args, **kwargs: Argumentos para la función
            
        Returns:
            Resultado de la función
            
        Raises:
            ManualReviewRequiredError: Si se requiere revisión manual
            Exception: Última excepción si no se puede recuperar
        """
        error_context = None
        
        try:
            # Wrap function with circuit breaker if specified
            if circuit_breaker_name and circuit_breaker_name in self.circuit_breakers:
                circuit_breaker = self.circuit_breakers[circuit_breaker_name]
                return await circuit_breaker.call(func, *args, **kwargs)
            else:
                # Execute function directly
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)
                    
        except Exception as error:
            # Handle error with policy
            error_context = await self.handle_error(
                error=error,
                component=component,
                function_name=function_name,
                similarity_score=similarity_score,
                book_id=book_id,
                phase=phase,
                metadata=metadata
            )
            
            # Determine if should escalate or retry
            if error_context.severity >= self.config.manual_review_threshold:
                if self.config.auto_escalation_enabled:
                    await self._escalate_to_manual_review(error_context)
                    raise ManualReviewRequiredError(
                        f"Error crítico requiere revisión manual: {error_context.error_id}"
                    )
            
            # Attempt retry with appropriate strategy
            max_retries = self._get_max_retries_for_severity(error_context.severity)
            if max_retries > 0:
                retry_strategy = self._get_retry_strategy_for_severity(error_context.severity)
                
                try:
                    result = await self.retry_manager.retry_with_policy(
                        func=func,
                        error_context=error_context,
                        strategy=retry_strategy,
                        custom_max_retries=max_retries,
                        *args,
                        **kwargs
                    )
                    
                    # Mark as resolved if retry succeeded
                    await self._mark_error_as_resolved(
                        error_context.error_id,
                        f"Resolved via {retry_strategy.value} retry"
                    )
                    
                    return result
                    
                except Exception as retry_error:
                    # All retries failed, update error context
                    await self.handle_error(
                        error=retry_error,
                        component=component,
                        function_name=f"{function_name}_retry_failed",
                        similarity_score=similarity_score,
                        book_id=book_id,
                        phase=phase,
                        metadata={
                            'original_error_id': error_context.error_id,
                            'retries_attempted': max_retries
                        }
                    )
                    
                    # Escalate if retries failed and severity is high
                    if error_context.severity >= ErrorSeverity.HIGH:
                        await self._escalate_to_manual_review(error_context)
                        raise ManualReviewRequiredError(
                            f"Reintentos fallidos, requiere revisión manual: {error_context.error_id}"
                        )
                    
                    raise retry_error
            
            # No retries allowed for this severity, re-raise
            raise error
    
    async def _escalate_to_manual_review(self, error_context: ErrorContext) -> None:
        """
        Escalación automática a revisión manual
        
        Args:
            error_context: Contexto del error a escalar
        """
        try:
            # Update database to mark as escalated
            if self.db_connector:
                update_query = """
                UPDATE error_tracking 
                SET escalated = TRUE, 
                    resolution_strategy = 'manual_review_required'
                WHERE error_id = ?
                """
                self.db_connector.execute_update(update_query, (error_context.error_id,))
            
            # Generate manual review ticket/alert
            escalation_metadata = {
                'escalation_reason': 'automatic_severity_threshold',
                'severity': error_context.severity.value,
                'similarity_score': error_context.similarity_score,
                'escalated_at': datetime.now().isoformat(),
                'requires_human_intervention': True
            }
            
            # Send high-priority alert
            if self.alert_manager:
                escalation_context = ErrorContext(
                    error_id=f"ESCALATION_{error_context.error_id}",
                    timestamp=datetime.now(),
                    component="ErrorPolicyManager",
                    function_name="escalate_to_manual_review",
                    error_type="ManualReviewRequired",
                    error_message=f"Error {error_context.error_id} escalated for manual review",
                    severity=ErrorSeverity.CRITICAL,
                    book_id=error_context.book_id,
                    phase=error_context.phase,
                    metadata=escalation_metadata
                )
                
                self.alert_manager.send_alert(escalation_context)
            
            self.logger.critical(
                f"🆘 ERROR ESCALADO A REVISIÓN MANUAL: {error_context.error_id} "
                f"[{error_context.component}.{error_context.function_name}] "
                f"Severity: {error_context.severity.value}"
            )
            
        except Exception as e:
            self.logger.error(f"❌ Error en escalación automática: {e}")
    
    async def _mark_error_as_resolved(
        self,
        error_id: str,
        resolution_strategy: str
    ) -> None:
        """
        Marca error como resuelto en base de datos
        
        Args:
            error_id: ID del error a marcar como resuelto
            resolution_strategy: Estrategia usada para resolverlo
        """
        if self.db_connector:
            try:
                update_query = """
                UPDATE error_tracking 
                SET resolved = TRUE, 
                    resolution_strategy = ?
                WHERE error_id = ?
                """
                self.db_connector.execute_update(
                    update_query, 
                    (resolution_strategy, error_id)
                )
                
                self.logger.info(f"✅ Error {error_id} marcado como resuelto: {resolution_strategy}")
                
            except Exception as e:
                self.logger.error(f"❌ Error marcando como resuelto {error_id}: {e}")
    
    def get_circuit_breaker_status(self) -> Dict[str, Dict[str, Any]]:
        """
        Obtiene estado actual de todos los circuit breakers
        
        Returns:
            Estado de cada circuit breaker
        """
        status = {}
        
        for name, cb in self.circuit_breakers.items():
            metrics = cb.get_metrics()
            status[name] = {
                'state': cb.state.value,
                'total_requests': metrics.total_requests,
                'success_rate': (
                    metrics.successful_requests / metrics.total_requests 
                    if metrics.total_requests > 0 else 0.0
                ),
                'consecutive_failures': metrics.consecutive_failures,
                'last_failure_time': (
                    metrics.last_failure_time.isoformat() 
                    if metrics.last_failure_time else None
                ),
                'state_changes': metrics.state_changes
            }
        
        return status
    
    def get_error_statistics(
        self,
        hours_back: int = 24
    ) -> Dict[str, Any]:
        """
        Obtiene estadísticas de errores recientes
        
        Args:
            hours_back: Horas hacia atrás para el análisis
            
        Returns:
            Estadísticas de errores
        """
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        recent_errors = [
            error for error in self.error_history 
            if error.timestamp >= cutoff_time
        ]
        
        if not recent_errors:
            return {
                'total_errors': 0,
                'by_severity': {},
                'by_component': {},
                'resolution_rate': 0.0,
                'most_common_errors': []
            }
        
        # Statistics by severity
        by_severity = {}
        for severity in ErrorSeverity:
            count = sum(1 for e in recent_errors if e.severity == severity)
            by_severity[severity.value] = count
        
        # Statistics by component
        by_component = {}
        for error in recent_errors:
            component = error.component
            by_component[component] = by_component.get(component, 0) + 1
        
        # Most common error types
        error_types = {}
        for error in recent_errors:
            error_type = error.error_type
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        most_common = sorted(
            error_types.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:5]
        
        return {
            'total_errors': len(recent_errors),
            'by_severity': by_severity,
            'by_component': by_component,
            'most_common_errors': most_common,
            'time_window_hours': hours_back
        }
    
    @asynccontextmanager
    async def protected_execution(
        self,
        component: str,
        function_name: str,
        circuit_breaker_name: Optional[str] = None,
        **context_kwargs
    ):
        """
        Context manager para ejecución protegida con manejo automático de errores
        
        Args:
            component: Nombre del componente
            function_name: Nombre de la función
            circuit_breaker_name: Circuit breaker a usar
            **context_kwargs: Argumentos de contexto (book_id, phase, etc.)
        
        Usage:
            async with error_manager.protected_execution(
                "SemanticValidator", 
                "validate_integrity"
            ) as executor:
                result = await executor(some_function, arg1, arg2)
        """
        async def executor(func, *args, **kwargs):
            return await self.execute_with_policy(
                func=func,
                component=component,
                function_name=function_name,
                circuit_breaker_name=circuit_breaker_name,
                *args,
                **context_kwargs,
                **kwargs
            )
        
        try:
            yield executor
        except Exception as e:
            # Additional context-level error handling if needed
            await self.handle_error(
                error=e,
                component=component,
                function_name=f"{function_name}_context",
                **context_kwargs
            )
            raise


# ==========================================
# TESTS UNITARIOS EMBEBIDOS
# ==========================================

if __name__ == "__main__":
    import unittest
    from unittest.mock import Mock, AsyncMock
    import asyncio
    
    class TestErrorPolicyManager(unittest.TestCase):
        """Tests unitarios para ErrorPolicyManager"""
        
        def setUp(self):
            """Setup para cada test"""
            self.mock_logger = Mock()
            self.mock_db = Mock()
            self.mock_alerts = Mock()
            
            self.config = ErrorPolicyConfig(
                failure_threshold=3,
                max_retries=2,
                base_delay=0.1  # Fast tests
            )
            
            self.error_manager = ErrorPolicyManager(
                config=self.config,
                logger=self.mock_logger,
                db_connector=self.mock_db,
                alert_manager=self.mock_alerts
            )
        
        def test_error_severity_determination(self):
            """Test determinación de severidad por similarity score"""
            # Test LOW severity (0.75-0.85)
            severity = self.error_manager._determine_error_severity(similarity_score=0.80)
            self.assertEqual(severity, ErrorSeverity.LOW)
            
            # Test MEDIUM severity (0.60-0.75)
            severity = self.error_manager._determine_error_severity(similarity_score=0.65)
            self.assertEqual(severity, ErrorSeverity.MEDIUM)
            
            # Test HIGH severity (0.40-0.60)
            severity = self.error_manager._determine_error_severity(similarity_score=0.50)
            self.assertEqual(severity, ErrorSeverity.HIGH)
            
            # Test CRITICAL severity (<0.40)
            severity = self.error_manager._determine_error_severity(similarity_score=0.30)
            self.assertEqual(severity, ErrorSeverity.CRITICAL)
        
        def test_circuit_breaker_states(self):
            """Test estados del circuit breaker"""
            cb = self.error_manager.circuit_breakers['deepl_api']
            
            # Initial state should be CLOSED
            self.assertEqual(cb.state, CircuitBreakerState.CLOSED)
            
            # Simulate failures to trip circuit breaker
            for i in range(self.config.failure_threshold):
                cb._record_failure(Exception(f"Test error {i}"))
            
            # Should be OPEN now
            self.assertEqual(cb.state, CircuitBreakerState.OPEN)
        
        def test_retry_strategy_selection(self):
            """Test selección de estrategia de reintento por severidad"""
            # LOW errors should use IMMEDIATE retry
            strategy = self.error_manager._get_retry_strategy_for_severity(ErrorSeverity.LOW)
            self.assertEqual(strategy, RetryStrategy.IMMEDIATE)
            
            # MEDIUM errors should use CONSERVATIVE retry  
            strategy = self.error_manager._get_retry_strategy_for_severity(ErrorSeverity.MEDIUM)
            self.assertEqual(strategy, RetryStrategy.CONSERVATIVE)
            
            # HIGH errors should use CHUNKED processing
            strategy = self.error_manager._get_retry_strategy_for_severity(ErrorSeverity.HIGH)
            self.assertEqual(strategy, RetryStrategy.CHUNKED)
            
            # CRITICAL errors should use EXPONENTIAL_BACKOFF
            strategy = self.error_manager._get_retry_strategy_for_severity(ErrorSeverity.CRITICAL)
            self.assertEqual(strategy, RetryStrategy.EXPONENTIAL_BACKOFF)
        
        async def test_successful_execution(self):
            """Test ejecución exitosa sin errores"""
            async def successful_function():
                return "success"
            
            result = await self.error_manager.execute_with_policy(
                func=successful_function,
                component="TestComponent",
                function_name="test_function"
            )
            
            self.assertEqual(result, "success")
        
        async def test_error_handling_and_retry(self):
            """Test manejo de errores con reintento"""
            call_count = 0
            
            async def failing_function():
                nonlocal call_count
                call_count += 1
                if call_count < 2:
                    raise ValueError("Test error")
                return "success_after_retry"
            
            result = await self.error_manager.execute_with_policy(
                func=failing_function,
                component="TestComponent", 
                function_name="test_function",
                similarity_score=0.80  # LOW severity -> should retry
            )
            
            self.assertEqual(result, "success_after_retry")
            self.assertEqual(call_count, 2)
        
        async def test_critical_error_escalation(self):
            """Test escalación automática de errores críticos"""
            async def critical_function():
                raise ValueError("Critical error")
            
            with self.assertRaises(ManualReviewRequiredError):
                await self.error_manager.execute_with_policy(
                    func=critical_function,
                    component="TestComponent",
                    function_name="test_function", 
                    similarity_score=0.30  # CRITICAL severity
                )
        
        def test_error_statistics(self):
            """Test generación de estadísticas de errores"""
            # Add some test errors
            error1 = ErrorContext(
                error_id="test1",
                timestamp=datetime.now(),
                component="TestComponent",
                function_name="test_func",
                error_type="ValueError",
                error_message="Test error",
                severity=ErrorSeverity.LOW
            )
            
            error2 = ErrorContext(
                error_id="test2", 
                timestamp=datetime.now(),
                component="TestComponent",
                function_name="test_func",
                error_type="ConnectionError",
                error_message="Connection failed",
                severity=ErrorSeverity.CRITICAL
            )
            
            self.error_manager.error_history = [error1, error2]
            
            stats = self.error_manager.get_error_statistics()
            
            self.assertEqual(stats['total_errors'], 2)
            self.assertEqual(stats['by_severity']['low'], 1)
            self.assertEqual(stats['by_severity']['critical'], 1)
    
    # Run tests
    async def run_async_tests():
        """Run async tests"""
        test_instance = TestErrorPolicyManager()
        test_instance.setUp()
        
        print("🧪 Running ErrorPolicyManager tests...")
        
        # Test severity determination
        test_instance.test_error_severity_determination()
        print("✅ test_error_severity_determination passed")
        
        # Test circuit breaker states
        test_instance.test_circuit_breaker_states()
        print("✅ test_circuit_breaker_states passed")
        
        # Test retry strategy selection
        test_instance.test_retry_strategy_selection()
        print("✅ test_retry_strategy_selection passed")
        
        # Test async methods
        await test_instance.test_successful_execution()
        print("✅ test_successful_execution passed")
        
        await test_instance.test_error_handling_and_retry()
        print("✅ test_error_handling_and_retry passed")
        
        await test_instance.test_critical_error_escalation()
        print("✅ test_critical_error_escalation passed")
        
        # Test statistics
        test_instance.test_error_statistics()
        print("✅ test_error_statistics passed")
        
        print("🎉 All tests passed! ErrorPolicyManager is ready for production.")
    
    # Example usage demonstration
    async def demo_usage():
        """Demonstra uso típico del ErrorPolicyManager"""
        print("\n📋 DEMO: Uso típico del ErrorPolicyManager")
        
        # Initialize with custom configuration
        config = ErrorPolicyConfig(
            failure_threshold=3,
            recovery_timeout=30,
            max_retries=2
        )
        
        error_manager = ErrorPolicyManager(config=config)
        
        # Example 1: Successful function execution
        print("\n1. Ejecución exitosa:")
        async def successful_api_call():
            return {"status": "success", "data": "translated_text"}
        
        result = await error_manager.execute_with_policy(
            func=successful_api_call,
            component="DeepLTranslator",
            function_name="translate_document",
            circuit_breaker_name="deepl_api",
            book_id="book_123",
            phase="translation"
        )
        print(f"   Result: {result}")
        
        # Example 2: Function with retryable error
        print("\n2. Error con reintento exitoso:")
        attempt_count = 0
        
        async def flaky_api_call():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count == 1:
                raise ConnectionError("Network timeout")
            return {"status": "success", "retry_succeeded": True}
        
        try:
            result = await error_manager.execute_with_policy(
                func=flaky_api_call,
                component="ClaudeAPI",
                function_name="suggest_terminology",
                circuit_breaker_name="claude_api",
                similarity_score=0.80,  # LOW severity -> will retry
                book_id="book_123"
            )
            print(f"   Result after retry: {result}")
        except Exception as e:
            print(f"   Error: {e}")
        
        # Example 3: Using context manager
        print("\n3. Uso con context manager:")
        async with error_manager.protected_execution(
            "SemanticValidator",
            "validate_integrity",
            book_id="book_123",
            phase="validation"
        ) as executor:
            async def validation_function():
                return {"integrity_score": 0.95, "alerts": []}
            
            result = await executor(validation_function)
            print(f"   Validation result: {result}")
        
        # Show statistics
        print("\n📊 Estadísticas del ErrorPolicyManager:")
        stats = error_manager.get_error_statistics()
        print(f"   Total errores: {stats['total_errors']}")
        print(f"   Por severidad: {stats['by_severity']}")
        
        cb_status = error_manager.get_circuit_breaker_status()
        print(f"\n🔌 Estado de Circuit Breakers:")
        for name, status in cb_status.items():
            print(f"   {name}: {status['state']} (success_rate: {status['success_rate']:.2%})")
    
    if __name__ == "__main__":
        # Run tests and demo
        asyncio.run(run_async_tests())
        asyncio.run(demo_usage())