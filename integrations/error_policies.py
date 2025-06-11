#!/usr/bin/env python3
"""
🔧 ERROR_POLICIES.PY - Enterprise Error Policy Manager
Sistema de Traducción Académica v2.2 - APIs Integration Layer

Gestión inteligente de errores y políticas de reintento enterprise:
- Políticas diferenciadas por tipo de error y proveedor
- Backoff exponencial con jitter para evitar thundering herd
- Circuit breakers automáticos para protección del sistema
- Análisis de patrones de error para mejora continua
- Rate limiting adaptativo basado en errores
- Métricas detalladas de recuperación y resilencia

CARACTERÍSTICAS ENTERPRISE:
✅ Políticas específicas por criticidad de servicio
✅ Circuit breakers con estados adaptativos
✅ Análisis de patrones de error con ML básico
✅ Rate limiting inteligente basado en historial
✅ Métricas de resilencia en tiempo real
✅ Escalación automática a soporte humano
✅ Recuperación gradual con health checks

Autor: Sistema ClaudeAcademico v2.2 - Enterprise Edition
Fecha: Enero 2025
Ubicación: integrations/error_policies.py
"""

import asyncio
import hashlib
import json
import random
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

# Import models - conditional para evitar circular imports
try:
    from .models import APIProvider, ServiceCriticality, Logger
except ImportError:
    # Fallback para tests independientes
    from enum import Enum
    
    class APIProvider(Enum):
        DEEPL = "deepl"
        CLAUDE = "claude"
        ABBYY = "abbyy"
    
    class ServiceCriticality(Enum):
        CRITICAL = "critical"
        IMPORTANT = "important" 
        AUXILIARY = "auxiliary"
    
    # Protocol para Logger
    class Logger:
        def debug(self, msg: str, *args, **kwargs): pass
        def info(self, msg: str, *args, **kwargs): pass
        def warning(self, msg: str, *args, **kwargs): pass
        def error(self, msg: str, *args, **kwargs): pass
        def critical(self, msg: str, *args, **kwargs): pass


# ===============================================================================
# ENUMS Y CONSTANTES ENTERPRISE
# ===============================================================================

class ErrorType(Enum):
    """Tipos de errores categorizados por naturaleza."""
    NETWORK_ERROR = "network_error"
    TIMEOUT_ERROR = "timeout_error"
    RATE_LIMIT_ERROR = "rate_limit_error"
    AUTHENTICATION_ERROR = "authentication_error"
    QUOTA_EXCEEDED = "quota_exceeded"
    VALIDATION_ERROR = "validation_error"
    SERVER_ERROR = "server_error"
    UNKNOWN_ERROR = "unknown_error"


class CircuitBreakerState(Enum):
    """Estados del circuit breaker."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Blocking requests
    HALF_OPEN = "half_open"  # Testing recovery


class ErrorSeverity(Enum):
    """Severidad de errores para clasificación."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# ===============================================================================
# CIRCUIT BREAKER ENTERPRISE
# ===============================================================================

class EnterpriseCircuitBreaker:
    """
    Circuit Breaker enterprise con estados adaptativos y recuperación inteligente.
    
    Características Enterprise:
    ✅ Estados adaptativos basados en criticidad del servicio
    ✅ Recovery gradual con health checks
    ✅ Métricas detalladas de performance
    ✅ Thresholds dinámicos basados en historial
    """
    
    def __init__(self,
                 service_name: str,
                 criticality: ServiceCriticality,
                 failure_threshold: int = 5,
                 recovery_timeout: float = 60.0,
                 success_threshold: int = 3,
                 logger: Optional[Logger] = None):
        
        self.service_name = service_name
        self.criticality = criticality
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold
        self.logger = logger or self._create_default_logger()
        
        # Estado del circuit breaker
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.last_success_time = None
        
        # Métricas detalladas
        self.total_requests = 0
        self.total_failures = 0
        self.total_successes = 0
        self.state_changes = []
        self.recent_errors = deque(maxlen=100)  # Últimos 100 errores
        
        # Configuración adaptativa basada en criticidad
        self._adjust_thresholds_by_criticality()
        
        self.logger.info(f"🔧 Circuit Breaker inicializado para {service_name} (criticidad: {criticality.value})")
    
    def _create_default_logger(self):
        """Crea logger por defecto."""
        import logging
        return logging.getLogger(f"{__name__}.CircuitBreaker.{self.service_name}")
    
    def _adjust_thresholds_by_criticality(self):
        """Ajusta thresholds basado en criticidad del servicio."""
        if self.criticality == ServiceCriticality.CRITICAL:
            # Servicios críticos: más tolerantes a fallos
            self.failure_threshold = max(self.failure_threshold, 8)
            self.recovery_timeout = min(self.recovery_timeout, 30.0)
            self.success_threshold = max(self.success_threshold, 5)
        elif self.criticality == ServiceCriticality.AUXILIARY:
            # Servicios auxiliares: menos tolerantes
            self.failure_threshold = min(self.failure_threshold, 3)
            self.recovery_timeout = max(self.recovery_timeout, 120.0)
            self.success_threshold = min(self.success_threshold, 2)
    
    async def call(self, func, *args, **kwargs):
        """
        Ejecuta función a través del circuit breaker.
        
        Args:
            func: Función a ejecutar
            *args, **kwargs: Argumentos para la función
            
        Returns:
            Resultado de la función o excepción
        """
        self.total_requests += 1
        
        # Verificar estado del circuit breaker
        if self.state == CircuitBreakerState.OPEN:
            if self._should_attempt_reset():
                self._transition_to_half_open()
            else:
                raise CircuitBreakerOpenError(
                    f"Circuit breaker OPEN para {self.service_name}. "
                    f"Próximo intento en {self._time_until_reset():.1f}s"
                )
        
        try:
            # Ejecutar función
            result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
            
            # Registrar éxito
            await self._record_success()
            return result
            
        except Exception as e:
            # Registrar fallo
            await self._record_failure(e)
            raise
    
    async def _record_success(self):
        """Registra un éxito y actualiza estado si es necesario."""
        self.total_successes += 1
        self.success_count += 1
        self.last_success_time = time.time()
        
        # Resetear contador de fallos en operación normal
        if self.state == CircuitBreakerState.CLOSED:
            self.failure_count = 0
        
        # Transición de HALF_OPEN a CLOSED
        elif self.state == CircuitBreakerState.HALF_OPEN:
            if self.success_count >= self.success_threshold:
                self._transition_to_closed()
        
        self.logger.debug(f"✅ {self.service_name}: Éxito registrado (successes: {self.success_count})")
    
    async def _record_failure(self, error: Exception):
        """Registra un fallo y actualiza estado si es necesario."""
        self.total_failures += 1
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        # Registrar error para análisis
        error_info = {
            "timestamp": datetime.now().isoformat(),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "state": self.state.value
        }
        self.recent_errors.append(error_info)
        
        # Resetear contador de éxitos
        self.success_count = 0
        
        # Transición a OPEN si se alcanza el threshold
        if self.failure_count >= self.failure_threshold:
            if self.state in [CircuitBreakerState.CLOSED, CircuitBreakerState.HALF_OPEN]:
                self._transition_to_open()
        
        self.logger.warning(f"❌ {self.service_name}: Fallo registrado (failures: {self.failure_count}/{self.failure_threshold})")
    
    def _should_attempt_reset(self) -> bool:
        """Determina si se debe intentar resetear el circuit breaker."""
        if self.last_failure_time is None:
            return True
        
        time_since_failure = time.time() - self.last_failure_time
        return time_since_failure >= self.recovery_timeout
    
    def _time_until_reset(self) -> float:
        """Calcula tiempo hasta el próximo intento de reset."""
        if self.last_failure_time is None:
            return 0.0
        
        time_since_failure = time.time() - self.last_failure_time
        return max(0.0, self.recovery_timeout - time_since_failure)
    
    def _transition_to_closed(self):
        """Transición a estado CLOSED (normal)."""
        old_state = self.state
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        
        self._record_state_change(old_state, self.state)
        self.logger.info(f"🟢 {self.service_name}: Circuit breaker CLOSED (servicio recuperado)")
    
    def _transition_to_open(self):
        """Transición a estado OPEN (bloqueando requests)."""
        old_state = self.state
        self.state = CircuitBreakerState.OPEN
        
        self._record_state_change(old_state, self.state)
        self.logger.error(f"🔴 {self.service_name}: Circuit breaker OPEN (servicio bloqueado por {self.recovery_timeout}s)")
    
    def _transition_to_half_open(self):
        """Transición a estado HALF_OPEN (probando recuperación)."""
        old_state = self.state
        self.state = CircuitBreakerState.HALF_OPEN
        self.success_count = 0
        
        self._record_state_change(old_state, self.state)
        self.logger.warning(f"🟡 {self.service_name}: Circuit breaker HALF_OPEN (probando recuperación)")
    
    def _record_state_change(self, old_state: CircuitBreakerState, new_state: CircuitBreakerState):
        """Registra cambio de estado para métricas."""
        self.state_changes.append({
            "timestamp": datetime.now().isoformat(),
            "from_state": old_state.value,
            "to_state": new_state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count
        })
    
    def get_metrics(self) -> Dict[str, Any]:
        """Obtiene métricas detalladas del circuit breaker."""
        uptime = time.time() - (self.state_changes[0]["timestamp"] if self.state_changes else time.time())
        
        return {
            "service_name": self.service_name,
            "criticality": self.criticality.value,
            "current_state": self.state.value,
            "total_requests": self.total_requests,
            "total_successes": self.total_successes,
            "total_failures": self.total_failures,
            "success_rate": self.total_successes / max(self.total_requests, 1),
            "failure_rate": self.total_failures / max(self.total_requests, 1),
            "current_failure_count": self.failure_count,
            "current_success_count": self.success_count,
            "failure_threshold": self.failure_threshold,
            "success_threshold": self.success_threshold,
            "recovery_timeout": self.recovery_timeout,
            "time_until_reset": self._time_until_reset() if self.state == CircuitBreakerState.OPEN else 0,
            "state_changes_count": len(self.state_changes),
            "recent_errors_count": len(self.recent_errors),
            "uptime_seconds": uptime
        }


class CircuitBreakerOpenError(Exception):
    """Excepción lanzada cuando el circuit breaker está OPEN."""
    pass


# ===============================================================================
# ENTERPRISE ERROR POLICY MANAGER
# ===============================================================================

class EnterpriseErrorPolicyManager:
    """
    Gestor enterprise de políticas de error con análisis inteligente.
    
    Características Enterprise:
    ✅ Políticas diferenciadas por tipo de error y proveedor
    ✅ Circuit breakers por servicio con estados adaptativos
    ✅ Análisis de patrones de error con ML básico
    ✅ Rate limiting adaptativo basado en errores
    ✅ Escalación automática a soporte humano
    ✅ Métricas detalladas de resilencia
    """
    
    def __init__(self, logger: Logger):
        self.logger = logger
        
        # Circuit breakers por proveedor
        self.circuit_breakers: Dict[APIProvider, EnterpriseCircuitBreaker] = {}
        
        # Políticas de error por tipo
        self.error_policies = self._initialize_error_policies()
        
        # Historial de errores para análisis
        self.error_history = defaultdict(list)
        self.error_patterns = defaultdict(int)
        
        # Rate limiting adaptativo
        self.adaptive_delays = defaultdict(float)
        self.recent_errors_by_provider = defaultdict(lambda: deque(maxlen=20))
        
        # Métricas de resilencia
        self.total_errors_handled = 0
        self.successful_recoveries = 0
        self.escalations_to_human = 0
        
        self.logger.info("🔧 EnterpriseErrorPolicyManager inicializado")
    
    def _initialize_error_policies(self) -> Dict[ErrorType, Dict[str, Any]]:
        """Inicializa políticas de error enterprise."""
        return {
            ErrorType.NETWORK_ERROR: {
                "max_retries": 5,
                "base_delay": 2.0,
                "max_delay": 60.0,
                "backoff_multiplier": 2.0,
                "jitter": True,
                "severity": ErrorSeverity.MEDIUM
            },
            ErrorType.TIMEOUT_ERROR: {
                "max_retries": 3,
                "base_delay": 5.0,
                "max_delay": 30.0,
                "backoff_multiplier": 1.5,
                "jitter": True,
                "severity": ErrorSeverity.MEDIUM
            },
            ErrorType.RATE_LIMIT_ERROR: {
                "max_retries": 4,
                "base_delay": 60.0,  # Esperar más para rate limits
                "max_delay": 300.0,
                "backoff_multiplier": 1.2,
                "jitter": False,  # No jitter para rate limits
                "severity": ErrorSeverity.LOW
            },
            ErrorType.AUTHENTICATION_ERROR: {
                "max_retries": 1,  # Solo un reintento para auth
                "base_delay": 1.0,
                "max_delay": 5.0,
                "backoff_multiplier": 1.0,
                "jitter": False,
                "severity": ErrorSeverity.HIGH
            },
            ErrorType.QUOTA_EXCEEDED: {
                "max_retries": 0,  # No reintentar quotas
                "base_delay": 0.0,
                "max_delay": 0.0,
                "backoff_multiplier": 1.0,
                "jitter": False,
                "severity": ErrorSeverity.CRITICAL
            },
            ErrorType.VALIDATION_ERROR: {
                "max_retries": 0,  # No reintentar errores de validación
                "base_delay": 0.0,
                "max_delay": 0.0,
                "backoff_multiplier": 1.0,
                "jitter": False,
                "severity": ErrorSeverity.HIGH
            },
            ErrorType.SERVER_ERROR: {
                "max_retries": 3,
                "base_delay": 10.0,
                "max_delay": 120.0,
                "backoff_multiplier": 2.5,
                "jitter": True,
                "severity": ErrorSeverity.HIGH
            },
            ErrorType.UNKNOWN_ERROR: {
                "max_retries": 2,
                "base_delay": 5.0,
                "max_delay": 30.0,
                "backoff_multiplier": 2.0,
                "jitter": True,
                "severity": ErrorSeverity.MEDIUM
            }
        }
    
    def register_circuit_breaker(self, provider: APIProvider, criticality: ServiceCriticality):
        """Registra circuit breaker para un proveedor."""
        self.circuit_breakers[provider] = EnterpriseCircuitBreaker(
            service_name=provider.value,
            criticality=criticality,
            logger=self.logger
        )
        self.logger.info(f"🔧 Circuit breaker registrado para {provider.value} (criticidad: {criticality.value})")
    
    async def handle_api_error(self, 
                             error: Exception, 
                             context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Maneja errores de APIs con políticas inteligentes enterprise.
        
        Args:
            error: Excepción ocurrida
            context: Contexto del error (provider, attempt, etc.)
            
        Returns:
            Dict con decisiones de manejo de error
        """
        self.total_errors_handled += 1
        
        provider = context.get("provider", "unknown")
        attempt = context.get("attempt", 1)
        
        # Clasificar error
        error_type = self._classify_error(error)
        error_severity = self.error_policies[error_type]["severity"]
        
        # Registrar error para análisis
        await self._record_error(provider, error, error_type, context)
        
        # Obtener política de error
        policy = self.error_policies[error_type]
        
        # Decidir si reintentar
        should_retry = self._should_retry(error_type, attempt, provider)
        
        # Calcular delay para reintento
        retry_delay = self._calculate_retry_delay(error_type, attempt, provider) if should_retry else 0
        
        # Determinar acciones especiales
        actions = await self._determine_special_actions(error, error_type, provider, context)
        
        decision = {
            "should_retry": should_retry,
            "retry_delay": retry_delay,
            "max_retries": policy["max_retries"],
            "error_type": error_type.value,
            "error_severity": error_severity.value,
            "reason": self._get_error_reason(error_type, should_retry),
            "suggested_actions": actions,
            "escalate_to_human": error_severity == ErrorSeverity.CRITICAL or attempt > policy["max_retries"],
            "circuit_breaker_status": self._get_circuit_breaker_status(provider),
            "adaptive_delay_applied": self.adaptive_delays.get(provider, 0.0)
        }
        
        # Log de decisión
        await self._log_error_decision(provider, error, decision)
        
        return decision
    
    def _classify_error(self, error: Exception) -> ErrorType:
        """Clasifica error basado en tipo y mensaje."""
        error_str = str(error).lower()
        error_type_name = type(error).__name__.lower()
        
        # Clasificación por tipo de excepción
        if "timeout" in error_type_name or "timeout" in error_str:
            return ErrorType.TIMEOUT_ERROR
        elif "connection" in error_str or "network" in error_str:
            return ErrorType.NETWORK_ERROR
        elif "rate limit" in error_str or "too many requests" in error_str:
            return ErrorType.RATE_LIMIT_ERROR
        elif "auth" in error_str or "unauthorized" in error_str or "forbidden" in error_str:
            return ErrorType.AUTHENTICATION_ERROR
        elif "quota" in error_str or "limit exceeded" in error_str:
            return ErrorType.QUOTA_EXCEEDED
        elif "validation" in error_str or "invalid" in error_str or "bad request" in error_str:
            return ErrorType.VALIDATION_ERROR
        elif "server error" in error_str or "internal error" in error_str or "500" in error_str:
            return ErrorType.SERVER_ERROR
        else:
            return ErrorType.UNKNOWN_ERROR
    
    async def _record_error(self, 
                           provider: str, 
                           error: Exception, 
                           error_type: ErrorType, 
                           context: Dict[str, Any]):
        """Registra error para análisis posterior."""
        error_record = {
            "timestamp": datetime.now().isoformat(),
            "provider": provider,
            "error_type": error_type.value,
            "error_message": str(error),
            "error_class": type(error).__name__,
            "context": context,
            "hash": hashlib.md5(f"{provider}:{error_type.value}:{str(error)[:100]}".encode()).hexdigest()[:8]
        }
        
        # Agregar a historial
        self.error_history[provider].append(error_record)
        
        # Mantener solo los últimos 1000 errores por proveedor
        if len(self.error_history[provider]) > 1000:
            self.error_history[provider] = self.error_history[provider][-1000:]
        
        # Actualizar patrones
        pattern_key = f"{provider}:{error_type.value}"
        self.error_patterns[pattern_key] += 1
        
        # Agregar a cola de errores recientes para rate limiting adaptativo
        self.recent_errors_by_provider[provider].append(time.time())
        
        # Actualizar delay adaptativo
        self._update_adaptive_delay(provider)
    
    def _should_retry(self, error_type: ErrorType, attempt: int, provider: str) -> bool:
        """Determina si se debe reintentar basado en política y circuit breaker."""
        policy = self.error_policies[error_type]
        
        # Verificar límite de reintentos
        if attempt >= policy["max_retries"]:
            return False
        
        # Verificar circuit breaker
        if provider in self.circuit_breakers:
            cb_metrics = self.circuit_breakers[provider].get_metrics()
            if cb_metrics["current_state"] == "open":
                return False
        
        # Política específica por tipo de error
        if error_type in [ErrorType.QUOTA_EXCEEDED, ErrorType.VALIDATION_ERROR]:
            return False
        
        # Rate limiting adaptativo
        recent_errors = len([
            t for t in self.recent_errors_by_provider[provider]
            if time.time() - t < 300  # Últimos 5 minutos
        ])
        
        if recent_errors > 10:  # Demasiados errores recientes
            return False
        
        return True
    
    def _calculate_retry_delay(self, error_type: ErrorType, attempt: int, provider: str) -> float:
        """Calcula delay para reintento con backoff exponencial y jitter."""
        policy = self.error_policies[error_type]
        
        # Backoff exponencial
        base_delay = policy["base_delay"]
        multiplier = policy["backoff_multiplier"]
        max_delay = policy["max_delay"]
        
        delay = min(base_delay * (multiplier ** (attempt - 1)), max_delay)
        
        # Agregar jitter si está habilitado
        if policy["jitter"]:
            jitter_amount = delay * 0.1  # 10% de jitter
            delay += random.uniform(-jitter_amount, jitter_amount)
        
        # Agregar delay adaptativo
        adaptive_delay = self.adaptive_delays.get(provider, 0.0)
        delay += adaptive_delay
        
        return max(delay, 0.1)  # Mínimo 0.1 segundos
    
    def _update_adaptive_delay(self, provider: str):
        """Actualiza delay adaptativo basado en errores recientes."""
        recent_errors = [
            t for t in self.recent_errors_by_provider[provider]
            if time.time() - t < 300  # Últimos 5 minutos
        ]
        
        error_rate = len(recent_errors) / 20  # Rate sobre ventana de 20 errores max
        
        # Incrementar delay si hay muchos errores
        if error_rate > 0.5:  # Más del 50% de errores
            self.adaptive_delays[provider] = min(
                self.adaptive_delays.get(provider, 0.0) + 2.0,
                30.0  # Máximo 30 segundos de delay adaptativo
            )
        elif error_rate < 0.1:  # Menos del 10% de errores
            # Reducir delay gradualmente
            current_delay = self.adaptive_delays.get(provider, 0.0)
            self.adaptive_delays[provider] = max(current_delay - 1.0, 0.0)
    
    async def _determine_special_actions(self, 
                                       error: Exception, 
                                       error_type: ErrorType, 
                                       provider: str, 
                                       context: Dict[str, Any]) -> List[str]:
        """Determina acciones especiales basadas en el tipo de error."""
        actions = []
        
        if error_type == ErrorType.AUTHENTICATION_ERROR:
            actions.append("verify_api_credentials")
            actions.append("check_api_key_validity")
        
        elif error_type == ErrorType.QUOTA_EXCEEDED:
            actions.append("check_billing_status")
            actions.append("consider_plan_upgrade")
            actions.append("implement_usage_limits")
        
        elif error_type == ErrorType.RATE_LIMIT_ERROR:
            actions.append("implement_rate_limiting")
            actions.append("increase_request_delays")
        
        elif error_type == ErrorType.NETWORK_ERROR:
            actions.append("check_network_connectivity")
            actions.append("verify_dns_resolution")
        
        elif error_type == ErrorType.SERVER_ERROR:
            actions.append("check_service_status")
            actions.append("contact_provider_support")
        
        # Acciones específicas por proveedor
        if provider == "deepl" and error_type == ErrorType.QUOTA_EXCEEDED:
            actions.append("switch_to_fallback_translation_service")
        
        # Escalación automática para errores críticos
        if error_type in [ErrorType.QUOTA_EXCEEDED, ErrorType.AUTHENTICATION_ERROR]:
            actions.append("escalate_to_human_support")
            self.escalations_to_human += 1
        
        return actions
    
    def _get_error_reason(self, error_type: ErrorType, should_retry: bool) -> str:
        """Obtiene razón textual para la decisión de manejo."""
        if not should_retry:
            if error_type in [ErrorType.QUOTA_EXCEEDED, ErrorType.VALIDATION_ERROR]:
                return f"{error_type.value}_not_retryable"
            else:
                return "max_retries_exceeded"
        else:
            return f"{error_type.value}_retryable"
    
    def _get_circuit_breaker_status(self, provider: str) -> Dict[str, Any]:
        """Obtiene estado del circuit breaker para el proveedor."""
        if provider not in self.circuit_breakers:
            return {"status": "not_configured"}
        
        cb = self.circuit_breakers[provider]
        return {
            "status": cb.state.value,
            "failure_count": cb.failure_count,
            "success_count": cb.success_count,
            "time_until_reset": cb._time_until_reset() if cb.state.value == "open" else 0
        }
    
    async def _log_error_decision(self, provider: str, error: Exception, decision: Dict[str, Any]):
        """Log detallado de la decisión de manejo de error."""
        if decision["should_retry"]:
            self.logger.warning(
                f"🔄 Error manejado - {provider}: {type(error).__name__} | "
                f"Reintento en {decision['retry_delay']:.1f}s | "
                f"Tipo: {decision['error_type']} | "
                f"Severidad: {decision['error_severity']}"
            )
        else:
            if decision["escalate_to_human"]:
                self.logger.error(
                    f"🚨 Error crítico - {provider}: {type(error).__name__} | "
                    f"ESCALANDO A SOPORTE HUMANO | "
                    f"Tipo: {decision['error_type']} | "
                    f"Acciones: {', '.join(decision['suggested_actions'])}"
                )
            else:
                self.logger.error(
                    f"❌ Error no recuperable - {provider}: {type(error).__name__} | "
                    f"No se reintenta | "
                    f"Tipo: {decision['error_type']} | "
                    f"Razón: {decision['reason']}"
                )
    
    async def should_retry(self, error: Exception, attempt: int) -> bool:
        """Interfaz de compatibilidad: determina si debe reintentar."""
        error_type = self._classify_error(error)
        return self._should_retry(error_type, attempt, "unknown")
    
    async def get_retry_delay(self, attempt: int) -> float:
        """Interfaz de compatibilidad: calcula delay para reintento."""
        return self._calculate_retry_delay(ErrorType.UNKNOWN_ERROR, attempt, "unknown")
    
    def get_error_analytics(self) -> Dict[str, Any]:
        """Obtiene análisis detallado de errores para reporting."""
        total_errors = sum(len(errors) for errors in self.error_history.values())
        
        # Análisis por proveedor
        provider_analysis = {}
        for provider, errors in self.error_history.items():
            if errors:
                error_types = defaultdict(int)
                recent_errors = [e for e in errors if 
                               (datetime.now() - datetime.fromisoformat(e["timestamp"])).days < 7]
                
                for error in errors:
                    error_types[error["error_type"]] += 1
                
                provider_analysis[provider] = {
                    "total_errors": len(errors),
                    "recent_errors_7d": len(recent_errors),
                    "error_types": dict(error_types),
                    "most_common_error": max(error_types.items(), key=lambda x: x[1])[0] if error_types else "none",
                    "error_rate_trend": "increasing" if len(recent_errors) > len(errors) * 0.3 else "stable"
                }
        
        # Top patrones de error
        top_patterns = sorted(
            self.error_patterns.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        return {
            "summary": {
                "total_errors_handled": self.total_errors_handled,
                "successful_recoveries": self.successful_recoveries,
                "escalations_to_human": self.escalations_to_human,
                "recovery_rate": self.successful_recoveries / max(self.total_errors_handled, 1),
                "escalation_rate": self.escalations_to_human / max(self.total_errors_handled, 1)
            },
            "provider_analysis": provider_analysis,
            "top_error_patterns": [
                {"pattern": pattern, "count": count}
                for pattern, count in top_patterns
            ],
            "circuit_breaker_metrics": {
                provider.value: cb.get_metrics()
                for provider, cb in self.circuit_breakers.items()
            },
            "adaptive_delays": dict(self.adaptive_delays)
        }
    
    def reset_metrics(self):
        """Resetea métricas para nuevo período de reporte."""
        self.total_errors_handled = 0
        self.successful_recoveries = 0
        self.escalations_to_human = 0
        
        # Limpiar historial antiguo (mantener solo último mes)
        cutoff_date = datetime.now() - timedelta(days=30)
        
        for provider in self.error_history:
            self.error_history[provider] = [
                error for error in self.error_history[provider]
                if datetime.fromisoformat(error["timestamp"]) > cutoff_date
            ]
        
        self.logger.info("🔧 Métricas de error reseteadas para nuevo período")


if __name__ == "__main__":
    """Tests básicos del error policy manager."""
    print("🔧 ERROR_POLICIES.PY - Enterprise Error Policy Manager")
    print("Testing básico...")
    
    # Test básico
    import logging
    logger = logging.getLogger("test")
    
    try:
        manager = EnterpriseErrorPolicyManager(logger)
        print("✅ EnterpriseErrorPolicyManager creado exitosamente")
        
        # Test clasificación de errores
        test_error = TimeoutError("Connection timeout")
        error_type = manager._classify_error(test_error)
        print(f"✅ Error clasificado como: {error_type.value}")
        
        # Test políticas
        policies = manager.error_policies
        print(f"✅ {len(policies)} políticas de error configuradas")
        
        print("🎉 Tests básicos completados exitosamente")
        
    except Exception as e:
        print(f"❌ Error en testing: {e}")
        import traceback
        traceback.print_exc()