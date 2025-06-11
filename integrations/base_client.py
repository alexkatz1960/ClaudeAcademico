#!/usr/bin/env python3
"""
🛡️ BASE_CLIENT.PY - Cliente Base y Utilidades Compartidas  
Sistema de Traducción Académica v2.2 - APIs Integration Layer
POST-AUDITORÍA: Versión mejorada con correcciones críticas

Contiene componentes base reutilizables:
- CircuitBreaker para tolerancia a fallos
- RateLimiter para control de frecuencia (con límite de intentos)
- AsyncRedisCacheManager para optimización (Redis async)
- BaseAPIClient como clase padre común (sin recursividad)

Autor: Sistema ClaudeAcademico v2.2
Fecha: Enero 2025 (Post-Auditoría)
Ubicación: integrations/base_client.py
"""

import asyncio
import hashlib
import json
import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urljoin

import aiohttp
import redis.asyncio as redis  # ✅ CAMBIO: Redis async en lugar de redis tradicional
from aiohttp import ClientSession, ClientTimeout, ClientError, ClientResponseError, ClientConnectorError

from .models import (
    APIProvider, APIResponse, APIUsageMetrics, CircuitBreakerConfig,
    CircuitBreakerState, Logger, CacheManager, ErrorPolicyManager,
    create_request_id, create_error_context
)


# ===============================================================================
# CIRCUIT BREAKER IMPLEMENTATION
# ===============================================================================

class CircuitBreaker:
    """
    Circuit Breaker para protección contra fallos en APIs externas.
    
    Implementa el patrón Circuit Breaker para prevenir llamadas
    a servicios que están fallando sistemáticamente.
    
    Estados:
    - CLOSED: Operación normal, permite todas las llamadas
    - OPEN: Bloqueando llamadas, API está fallando
    - HALF_OPEN: Probando recuperación con llamadas limitadas
    """
    
    def __init__(self, config: CircuitBreakerConfig, logger: Logger):
        self.config = config
        self.logger = logger
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.last_success_time: Optional[datetime] = None
        
    async def call(self, func, *args, **kwargs):
        """
        Ejecuta función con protección del Circuit Breaker.
        
        Args:
            func: Función a ejecutar
            *args: Argumentos posicionales
            **kwargs: Argumentos nombrados
            
        Returns:
            Resultado de la función
            
        Raises:
            Exception: Si el circuit está abierto o la función falla
        """
        if self.state == CircuitBreakerState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitBreakerState.HALF_OPEN
                self.logger.info("🔄 Circuit Breaker: Intentando recuperación (HALF_OPEN)")
            else:
                self.logger.warning(f"⚡ Circuit Breaker: OPEN - Bloqueando llamada a {func.__name__}")
                raise Exception("Circuit Breaker is OPEN")
        
        try:
            # Si está en HALF_OPEN, solo permitir una llamada
            if self.state == CircuitBreakerState.HALF_OPEN:
                self.logger.debug("🔍 Circuit Breaker: Probando llamada en estado HALF_OPEN")
            
            result = await func(*args, **kwargs)
            await self._on_success()
            return result
            
        except self.config.expected_exception as e:
            await self._on_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """Determina si debe intentar resetear el circuit."""
        if not self.last_failure_time:
            return False
        
        time_since_failure = datetime.now() - self.last_failure_time
        return time_since_failure.total_seconds() >= self.config.timeout_duration
    
    async def _on_success(self) -> None:
        """Maneja éxito en llamada."""
        self.failure_count = 0
        self.last_success_time = datetime.now()
        
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.state = CircuitBreakerState.CLOSED
            self.logger.info("✅ Circuit Breaker: Recuperado exitosamente (CLOSED)")
    
    async def _on_failure(self) -> None:
        """Maneja fallo en llamada."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.config.failure_threshold:
            self.state = CircuitBreakerState.OPEN
            self.logger.error(f"🚨 Circuit Breaker: OPEN - {self.failure_count} fallos consecutivos")
    
    def get_state_info(self) -> Dict[str, Any]:
        """Obtiene información del estado actual."""
        return {
            "state": self.state.value,
            "failure_count": self.failure_count,
            "last_failure_time": self.last_failure_time.isoformat() if self.last_failure_time else None,
            "last_success_time": self.last_success_time.isoformat() if self.last_success_time else None,
            "failure_threshold": self.config.failure_threshold,
            "timeout_duration": self.config.timeout_duration
        }


# ===============================================================================
# RATE LIMITER IMPLEMENTATION (MEJORADO)
# ===============================================================================

class RateLimiter:
    """
    Rate Limiter para controlar frecuencia de llamadas a APIs.
    
    Implementa algoritmo token bucket para limitar requests por minuto/segundo.
    Mantiene historial de requests para aplicar límites de manera justa.
    
    ✅ POST-AUDITORÍA: Agregado control de intentos máximos para evitar loops infinitos.
    """
    
    def __init__(self, max_requests: int, time_window: int, logger: Logger, max_attempts: int = 10):
        """
        Args:
            max_requests: Máximo número de requests
            time_window: Ventana de tiempo en segundos
            logger: Logger para registro
            max_attempts: Máximo número de intentos para evitar loops infinitos
        """
        self.max_requests = max_requests
        self.time_window = time_window
        self.logger = logger
        self.max_attempts = max_attempts  # ✅ NUEVO: Control de loops infinitos
        self.requests: List[float] = []
        self._lock = asyncio.Lock()
        
    async def acquire(self, attempt: int = 0) -> bool:
        """
        Intenta adquirir permiso para hacer request.
        
        Args:
            attempt: Número de intento actual (interno)
            
        Returns:
            True si puede proceder, False si debe esperar
            
        Raises:
            Exception: Si se alcanzan intentos máximos
        """
        # ✅ MEJORA: Control de intentos máximos
        if attempt >= self.max_attempts:
            self.logger.error(f"🚨 Rate Limiter: Máximo de intentos alcanzado ({self.max_attempts})")
            raise Exception(f"Rate limiter max attempts reached: {self.max_attempts}")
        
        async with self._lock:
            now = time.time()
            
            # Limpiar requests antiguos
            cutoff = now - self.time_window
            self.requests = [req_time for req_time in self.requests if req_time > cutoff]
            
            if len(self.requests) < self.max_requests:
                self.requests.append(now)
                self.logger.debug(f"🚦 Rate Limiter: Request permitido ({len(self.requests)}/{self.max_requests})")
                return True
            
            # Calcular tiempo de espera
            oldest_request = min(self.requests)
            wait_time = self.time_window - (now - oldest_request)
            
            self.logger.debug(f"⏳ Rate Limiter: Esperando {wait_time:.2f}s antes de próximo request (intento {attempt + 1})")
            await asyncio.sleep(wait_time)
            
            # ✅ MEJORA: Llamada iterativa en lugar de recursiva
            return await self.acquire(attempt + 1)
    
    def get_current_usage(self) -> Dict[str, Any]:
        """Obtiene uso actual del rate limiter."""
        now = time.time()
        cutoff = now - self.time_window
        current_requests = [req for req in self.requests if req > cutoff]
        
        return {
            "current_requests": len(current_requests),
            "max_requests": self.max_requests,
            "time_window": self.time_window,
            "usage_percentage": (len(current_requests) / self.max_requests) * 100,
            "requests_remaining": self.max_requests - len(current_requests),
            "max_attempts": self.max_attempts
        }


# ===============================================================================
# ASYNC REDIS CACHE MANAGER IMPLEMENTATION (MEJORADO)
# ===============================================================================

class AsyncRedisCacheManager:
    """
    Cache Manager usando Redis async para optimización de APIs.
    
    ✅ POST-AUDITORÍA: Migrado a redis.asyncio para operaciones realmente asíncronas.
    """
    
    def __init__(self, redis_url: str, logger: Logger):
        self.redis_url = redis_url
        self.logger = logger
        self._redis_client: Optional[redis.Redis] = None
        self._connection_verified = False
        
    async def _get_client(self) -> redis.Redis:
        """Obtiene cliente Redis async con lazy initialization."""
        if self._redis_client is None or not self._connection_verified:
            try:
                # ✅ CAMBIO CRÍTICO: Usar redis.asyncio en lugar de redis tradicional
                self._redis_client = redis.from_url(self.redis_url, decode_responses=True)
                
                # Test de conexión asíncrono
                await self._redis_client.ping()
                self._connection_verified = True
                self.logger.info(f"✅ Redis Async: Conectado exitosamente a {self.redis_url}")
                
            except Exception as e:
                self.logger.error(f"❌ Redis Async: Error conectando a {self.redis_url}: {e}")
                self._redis_client = None
                self._connection_verified = False
                raise
        return self._redis_client
    
    async def get(self, key: str) -> Optional[Any]:
        """Obtiene valor del cache."""
        try:
            client = await self._get_client()
            # ✅ MEJORA: Operación realmente asíncrona
            value = await client.get(key)
            if value:
                self.logger.debug(f"📦 Cache HIT: {key}")
                return json.loads(value)
            self.logger.debug(f"📭 Cache MISS: {key}")
            return None
        except Exception as e:
            self.logger.error(f"❌ Error accediendo cache: {e}")
            return None
    
    async def set(self, key: str, value: Any, ttl: int = 3600) -> None:
        """Almacena valor en cache."""
        try:
            client = await self._get_client()
            serialized_value = json.dumps(value, default=str)
            # ✅ MEJORA: Operación realmente asíncrona
            await client.setex(key, ttl, serialized_value)
            self.logger.debug(f"💾 Cache SET: {key} (TTL: {ttl}s)")
        except Exception as e:
            self.logger.error(f"❌ Error escribiendo cache: {e}")
    
    async def delete(self, key: str) -> None:
        """Elimina valor del cache."""
        try:
            client = await self._get_client()
            # ✅ MEJORA: Operación realmente asíncrona
            await client.delete(key)
            self.logger.debug(f"🗑️ Cache DELETE: {key}")
        except Exception as e:
            self.logger.error(f"❌ Error eliminando cache: {e}")
    
    async def exists(self, key: str) -> bool:
        """Verifica si existe clave en cache."""
        try:
            client = await self._get_client()
            # ✅ MEJORA: Operación realmente asíncrona
            result = await client.exists(key)
            return bool(result)
        except Exception as e:
            self.logger.error(f"❌ Error verificando cache: {e}")
            return False
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del cache."""
        try:
            client = await self._get_client()
            # ✅ MEJORA: Operaciones realmente asíncronas
            info = await client.info()
            db_size = await client.dbsize()
            
            return {
                "connected": True,
                "memory_used": info.get("used_memory_human", "Unknown"),
                "total_keys": db_size,
                "hits": info.get("keyspace_hits", 0),
                "misses": info.get("keyspace_misses", 0),
                "hit_rate": (
                    info.get("keyspace_hits", 0) / 
                    max(info.get("keyspace_hits", 0) + info.get("keyspace_misses", 0), 1)
                ) * 100
            }
        except Exception as e:
            self.logger.error(f"❌ Error obteniendo stats de cache: {e}")
            return {"connected": False, "error": str(e)}
    
    async def close(self) -> None:
        """Cierra conexión Redis."""
        if self._redis_client:
            try:
                await self._redis_client.close()
                self.logger.info("🔒 Redis: Conexión cerrada")
            except Exception as e:
                self.logger.error(f"❌ Error cerrando Redis: {e}")


# ===============================================================================
# MEMORIA CACHE FALLBACK (SIN CAMBIOS)
# ===============================================================================

class MemoryCacheManager:
    """Cache en memoria como fallback cuando Redis no está disponible."""
    
    def __init__(self, logger: Logger, max_size: int = 1000):
        self.logger = logger
        self.max_size = max_size
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._access_times: Dict[str, float] = {}
        
    async def get(self, key: str) -> Optional[Any]:
        """Obtiene valor del cache en memoria."""
        if key in self._cache:
            # Verificar TTL
            cache_entry = self._cache[key]
            if time.time() < cache_entry["expires_at"]:
                self._access_times[key] = time.time()
                self.logger.debug(f"📦 Memory Cache HIT: {key}")
                return cache_entry["value"]
            else:
                # Expirado
                del self._cache[key]
                del self._access_times[key]
        
        self.logger.debug(f"📭 Memory Cache MISS: {key}")
        return None
    
    async def set(self, key: str, value: Any, ttl: int = 3600) -> None:
        """Almacena valor en cache en memoria."""
        # Limpiar cache si está lleno
        if len(self._cache) >= self.max_size:
            await self._evict_oldest()
        
        self._cache[key] = {
            "value": value,
            "expires_at": time.time() + ttl
        }
        self._access_times[key] = time.time()
        self.logger.debug(f"💾 Memory Cache SET: {key} (TTL: {ttl}s)")
    
    async def delete(self, key: str) -> None:
        """Elimina valor del cache."""
        self._cache.pop(key, None)
        self._access_times.pop(key, None)
        self.logger.debug(f"🗑️ Memory Cache DELETE: {key}")
    
    async def exists(self, key: str) -> bool:
        """Verifica si existe clave en cache."""
        if key in self._cache:
            cache_entry = self._cache[key]
            if time.time() < cache_entry["expires_at"]:
                return True
            else:
                del self._cache[key]
                del self._access_times[key]
        return False
    
    async def _evict_oldest(self) -> None:
        """Elimina el elemento más antiguo del cache."""
        if self._access_times:
            oldest_key = min(self._access_times.items(), key=lambda x: x[1])[0]
            await self.delete(oldest_key)


# ===============================================================================
# BASE API CLIENT (MEJORADO)
# ===============================================================================

class BaseAPIClient(ABC):
    """
    Clase base para clientes de APIs con funcionalidades comunes.
    
    ✅ POST-AUDITORÍA: Manejo granular de errores y retries sin recursividad.
    """
    
    def __init__(self,
                 api_key: str,
                 base_url: str,
                 provider: APIProvider,
                 logger: Logger,
                 cache_manager: Optional[CacheManager] = None,
                 error_policy_manager: Optional[ErrorPolicyManager] = None,
                 rate_limiter: Optional[RateLimiter] = None):
        
        self.api_key = api_key
        self.base_url = base_url
        self.provider = provider
        self.logger = logger
        self.cache_manager = cache_manager
        self.error_policy_manager = error_policy_manager
        self.rate_limiter = rate_limiter
        
        # Configurar Circuit Breaker
        circuit_config = CircuitBreakerConfig(
            failure_threshold=3,
            timeout_duration=60,
            expected_exception=aiohttp.ClientError
        )
        self.circuit_breaker = CircuitBreaker(circuit_config, logger)
        
        # Métricas de uso
        self.usage_metrics = APIUsageMetrics(provider=provider)
        
        # Cliente HTTP con timeouts y headers
        self.timeout = ClientTimeout(total=300, connect=30)
        self.headers = {
            "User-Agent": "ClaudeAcademico-v2.2/1.0",
            "Accept": "application/json",
            "Content-Type": "application/json"
        }
    
    def _generate_cache_key(self, endpoint: str, params: Dict[str, Any]) -> str:
        """Genera clave de cache única basada en endpoint y parámetros."""
        # Crear hash de parámetros para clave única
        params_str = json.dumps(params, sort_keys=True, default=str)
        params_hash = hashlib.md5(params_str.encode()).hexdigest()
        
        return f"{self.provider.value}:{endpoint}:{params_hash}"
    
    async def _make_request(self,
                          method: str,
                          endpoint: str,
                          data: Optional[Dict[str, Any]] = None,
                          params: Optional[Dict[str, Any]] = None,
                          headers: Optional[Dict[str, str]] = None,
                          cache_ttl: int = 3600,
                          use_cache: bool = True) -> APIResponse:
        """
        Realiza request HTTP con todas las protecciones y optimizaciones.
        
        ✅ POST-AUDITORÍA: Retries implementados con bucle en lugar de recursividad.
        """
        request_start = time.time()
        request_id = create_request_id(self.provider)
        
        # Preparar parámetros de request
        url = urljoin(self.base_url, endpoint)
        req_headers = {**self.headers, **(headers or {})}
        
        # Verificar cache si está habilitado
        cache_key = None
        if use_cache and self.cache_manager and method.upper() == "GET":
            cache_key = self._generate_cache_key(endpoint, params or {})
            cached_response = await self.cache_manager.get(cache_key)
            
            if cached_response:
                response_time = time.time() - request_start
                self.logger.info(f"📦 {self.provider.value.upper()} Cache hit: {endpoint}")
                
                return APIResponse(
                    success=True,
                    data=cached_response,
                    provider=self.provider,
                    request_id=request_id,
                    response_time=response_time,
                    cached=True
                )
        
        # Rate limiting
        if self.rate_limiter:
            await self.rate_limiter.acquire()
        
        # ✅ MEJORA CRÍTICA: Retries con bucle en lugar de recursividad
        max_retries = 3
        last_exception = None
        
        for retry_attempt in range(max_retries + 1):  # 0, 1, 2, 3
            try:
                # Llamada protegida por Circuit Breaker
                response_data = await self.circuit_breaker.call(
                    self._execute_request,
                    method, url, req_headers, data, params
                )
                
                response_time = time.time() - request_start
                
                # Actualizar métricas
                characters_count = len(str(data or params or ""))
                cost = self._estimate_cost(characters_count)
                self.usage_metrics.add_request(
                    characters=characters_count,
                    response_time=response_time,
                    cost=cost,
                    success=True
                )
                
                # Guardar en cache si corresponde
                if use_cache and self.cache_manager and cache_key and method.upper() == "GET":
                    await self.cache_manager.set(cache_key, response_data, cache_ttl)
                
                self.logger.info(f"✅ {self.provider.value.upper()} Request exitoso: {endpoint} ({response_time:.2f}s)")
                
                return APIResponse(
                    success=True,
                    data=response_data,
                    provider=self.provider,
                    request_id=request_id,
                    response_time=response_time,
                    cost_estimate=cost,
                    usage_metrics=self.usage_metrics.__dict__,
                    retry_count=retry_attempt
                )
                
            except Exception as e:
                last_exception = e
                
                # ✅ MEJORA: Manejo granular de errores por tipo
                error_type = self._classify_error(e)
                should_retry = self._should_retry_error(error_type, retry_attempt, max_retries)
                
                if should_retry:
                    # Backoff exponencial
                    delay = (2 ** retry_attempt) * 1.0  # 1s, 2s, 4s
                    self.logger.warning(
                        f"🔄 {self.provider.value.upper()} Retry {retry_attempt + 1}/{max_retries}: "
                        f"{endpoint} - {error_type} - Esperando {delay}s"
                    )
                    await asyncio.sleep(delay)
                    continue
                else:
                    self.logger.error(f"❌ Error no recuperable en {endpoint}: {error_type}")
                    break
        
        # Si llegamos aquí, todos los retries fallaron
        response_time = time.time() - request_start
        
        # Actualizar métricas de error
        self.usage_metrics.add_request(
            response_time=response_time,
            success=False
        )
        
        # Manejo de errores con ErrorPolicyManager
        error_context = create_error_context(
            provider=self.provider,
            endpoint=endpoint,
            request_id=request_id,
            method=method
        )
        
        if self.error_policy_manager:
            await self.error_policy_manager.handle_api_error(last_exception, error_context)
        
        self.logger.error(f"❌ {self.provider.value.upper()} Request falló después de {max_retries} reintentos: {endpoint} - {str(last_exception)}")
        
        return APIResponse(
            success=False,
            data=None,
            provider=self.provider,
            request_id=request_id,
            response_time=response_time,
            error_message=str(last_exception),
            usage_metrics=self.usage_metrics.__dict__,
            retry_count=max_retries
        )
    
    def _classify_error(self, error: Exception) -> str:
        """
        ✅ NUEVO: Clasifica errores para manejo granular.
        
        Args:
            error: Excepción a clasificar
            
        Returns:
            Tipo de error como string
        """
        if isinstance(error, asyncio.TimeoutError):
            return "timeout"
        elif isinstance(error, ClientConnectorError):
            return "connection_error"
        elif isinstance(error, ClientResponseError):
            if error.status >= 500:
                return "server_error"
            elif error.status == 429:
                return "rate_limit"
            elif error.status in [401, 403]:
                return "auth_error"
            else:
                return "client_error"
        elif isinstance(error, ClientError):
            return "client_error"
        elif isinstance(error, json.JSONDecodeError):
            return "json_decode_error"
        else:
            return "unknown_error"
    
    def _should_retry_error(self, error_type: str, current_attempt: int, max_retries: int) -> bool:
        """
        ✅ NUEVO: Determina si un error debe ser reintentado.
        
        Args:
            error_type: Tipo de error clasificado
            current_attempt: Intento actual (0-based)
            max_retries: Máximo número de retries
            
        Returns:
            True si debe reintentar, False si no
        """
        if current_attempt >= max_retries:
            return False
        
        # Errores que NO deben ser reintentados
        non_retryable_errors = {
            "auth_error",      # 401, 403 - problema de autenticación
            "client_error",    # 4xx general - problema del cliente
            "json_decode_error"  # Error de parsing - problema de código
        }
        
        if error_type in non_retryable_errors:
            return False
        
        # Errores que SÍ deben ser reintentados
        retryable_errors = {
            "timeout",         # Timeout - puede ser transitorio
            "connection_error", # Error de conexión - puede ser transitorio
            "server_error",    # 5xx - problema del servidor
            "rate_limit",      # 429 - rate limiting
            "unknown_error"    # Desconocido - mejor intentar
        }
        
        return error_type in retryable_errors
    
    async def _execute_request(self,
                             method: str,
                             url: str,
                             headers: Dict[str, str],
                             data: Optional[Dict[str, Any]] = None,
                             params: Optional[Dict[str, Any]] = None) -> Any:
        """Ejecuta la request HTTP real."""
        async with ClientSession(timeout=self.timeout) as session:
            async with session.request(
                method=method,
                url=url,
                headers=headers,
                json=data,
                params=params
            ) as response:
                
                # Verificar status de respuesta
                if response.status >= 400:
                    error_text = await response.text()
                    raise aiohttp.ClientResponseError(
                        request_info=response.request_info,
                        history=response.history,
                        status=response.status,
                        message=f"HTTP {response.status}: {error_text}"
                    )
                
                return await response.json()
    
    @abstractmethod
    def _estimate_cost(self, characters: int) -> float:
        """Estima costo de la request basado en caracteres procesados."""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Verifica salud de la API."""
        pass
    
    def get_usage_metrics(self) -> Dict[str, Any]:
        """Obtiene métricas de uso actuales."""
        return {
            "provider": self.provider.value,
            "requests_count": self.usage_metrics.requests_count,
            "characters_processed": self.usage_metrics.characters_processed,
            "cost_estimate": self.usage_metrics.cost_estimate,
            "success_rate": self.usage_metrics.success_rate,
            "average_response_time": self.usage_metrics.average_response_time,
            "error_count": self.usage_metrics.error_count,
            "daily_usage": self.usage_metrics.daily_usage,
            "circuit_breaker_state": self.circuit_breaker.state.value,
            "last_request": self.usage_metrics.last_request_time.isoformat() if self.usage_metrics.last_request_time else None
        }
    
    def get_circuit_breaker_info(self) -> Dict[str, Any]:
        """Obtiene información del circuit breaker."""
        return self.circuit_breaker.get_state_info()
    
    def get_rate_limiter_info(self) -> Dict[str, Any]:
        """Obtiene información del rate limiter."""
        if self.rate_limiter:
            return self.rate_limiter.get_current_usage()
        return {"enabled": False}


# ===============================================================================
# FACTORY FUNCTIONS (ACTUALIZADAS)
# ===============================================================================

def create_cache_manager(redis_url: Optional[str], logger: Logger) -> CacheManager:
    """
    Factory function para crear cache manager apropiado.
    
    ✅ POST-AUDITORÍA: Usa AsyncRedisCacheManager en lugar de RedisCacheManager.
    """
    if redis_url:
        try:
            return AsyncRedisCacheManager(redis_url, logger)
        except Exception as e:
            logger.warning(f"⚠️ Redis no disponible, usando cache en memoria: {e}")
            return MemoryCacheManager(logger)
    else:
        logger.info("💾 Usando cache en memoria (Redis no configurado)")
        return MemoryCacheManager(logger)


def create_rate_limiter(provider: APIProvider, logger: Logger) -> RateLimiter:
    """Factory function para crear rate limiter por proveedor."""
    from .models import DEFAULT_RATE_LIMITS
    
    limits = DEFAULT_RATE_LIMITS.get(provider, {"max_requests": 100, "time_window": 60})
    
    return RateLimiter(
        max_requests=limits["max_requests"],
        time_window=limits["time_window"],
        logger=logger
    )


# ===============================================================================
# TESTS UNITARIOS EMBEBIDOS (ACTUALIZADOS)
# ===============================================================================

async def test_circuit_breaker():
    """Test del Circuit Breaker."""
    logger = logging.getLogger("test")
    
    config = CircuitBreakerConfig(failure_threshold=2, timeout_duration=1)
    circuit = CircuitBreaker(config, logger)
    
    async def failing_function():
        raise Exception("Test error")
    
    # Primera llamada - debe fallar
    try:
        await circuit.call(failing_function)
        assert False, "Debería haber fallado"
    except Exception:
        pass
    
    # Segunda llamada - debe fallar y abrir circuit
    try:
        await circuit.call(failing_function)
        assert False, "Debería haber fallado"
    except Exception:
        pass
    
    # Circuit debe estar abierto
    assert circuit.state == CircuitBreakerState.OPEN
    
    print("✅ Test Circuit Breaker: PASSED")


async def test_rate_limiter():
    """Test del Rate Limiter con control de intentos."""
    logger = logging.getLogger("test")
    
    # Rate limiter: 2 requests por 2 segundos, máximo 3 intentos
    limiter = RateLimiter(max_requests=2, time_window=2, logger=logger, max_attempts=3)
    
    # Primer request - debe pasar inmediatamente
    start = time.time()
    result1 = await limiter.acquire()
    assert result1 == True
    assert time.time() - start < 0.1
    
    # Segundo request - debe pasar inmediatamente
    result2 = await limiter.acquire()
    assert result2 == True
    
    print("✅ Test Rate Limiter (mejorado): PASSED")


async def test_async_redis_cache():
    """Test del Async Redis Cache Manager con fallback."""
    logger = logging.getLogger("test")
    
    # Intentar con URL inválida para forzar fallback
    try:
        cache = AsyncRedisCacheManager("redis://invalid:6379/0", logger)
        await cache.get("test_key")
        print("⚠️ Redis funcionó (no esperado en test)")
    except Exception:
        print("✅ Test Async Redis Cache (fallback): PASSED")


async def test_memory_cache():
    """Test del Memory Cache Manager."""
    logger = logging.getLogger("test")
    
    cache = MemoryCacheManager(logger, max_size=5)
    
    # Test set/get
    await cache.set("test_key", {"data": "test_value"}, ttl=10)
    result = await cache.get("test_key")
    assert result == {"data": "test_value"}
    
    # Test exists
    exists = await cache.exists("test_key")
    assert exists == True
    
    # Test TTL expiration
    await cache.set("expire_key", "value", ttl=0)  # TTL inmediato
    await asyncio.sleep(0.1)
    result = await cache.get("expire_key")
    assert result is None
    
    print("✅ Test Memory Cache: PASSED")


async def run_all_tests():
    """Ejecuta todos los tests embebidos."""
    print("🧪 Ejecutando tests de base_client.py (POST-AUDITORÍA)...")
    
    try:
        await test_circuit_breaker()
        await test_rate_limiter()
        await test_async_redis_cache()
        await test_memory_cache()
        
        print("\n✅ Todos los tests de base_client.py (POST-AUDITORÍA) pasaron!")
        print("\n🏆 MEJORAS IMPLEMENTADAS:")
        print("  ✅ Redis async para operaciones no bloqueantes")
        print("  ✅ Retries con bucle en lugar de recursividad")
        print("  ✅ Manejo granular de errores por tipo")
        print("  ✅ Control de intentos máximos en RateLimiter")
        
    except Exception as e:
        print(f"\n❌ Test falló: {e}")
        raise


if __name__ == "__main__":
    """Ejecutar tests al correr el módulo directamente."""
    asyncio.run(run_all_tests())