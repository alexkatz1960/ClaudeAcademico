 # ==========================================
# INTERFACES/FASTAPI_BACKEND/API/MIDDLEWARE.PY
# Enterprise Middleware Stack - FastAPI
# Sistema de Traducción Académica v2.2
# ==========================================

from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.middleware.base import BaseHTTPMiddleware
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.sessions import SessionMiddleware
import structlog
import time
import uuid
import json
import traceback
from typing import Callable, Optional, Dict, Any, List
from datetime import datetime, timedelta
from pathlib import Path
import asyncio
from contextlib import asynccontextmanager
import gzip
import hashlib
from collections import defaultdict, deque

# Internal imports
from ..core.config import Settings, get_settings
from ..utils.responses import create_error_response, ErrorCode
from ...database.database import get_database_manager
from ...database.models import UsageStatistic

# Configure structured logger
logger = structlog.get_logger(__name__)

# ==========================================
# SECURITY HEADERS MIDDLEWARE
# ==========================================

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Enterprise-grade security headers middleware
    Implements OWASP security recommendations
    """
    
    def __init__(self, app, settings: Settings):
        super().__init__(app)
        self.settings = settings
        self.logger = structlog.get_logger("middleware.security")
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Apply security headers to all responses"""
        start_time = time.time()
        
        try:
            response = await call_next(request)
            
            # Apply security headers
            self._apply_security_headers(response, request)
            
            duration_ms = (time.time() - start_time) * 1000
            self.logger.debug(
                "Security headers applied",
                path=request.url.path,
                method=request.method,
                duration_ms=duration_ms
            )
            
            return response
            
        except Exception as e:
            self.logger.error(
                "Error in security middleware",
                error=str(e),
                path=request.url.path,
                method=request.method
            )
            raise
    
    def _apply_security_headers(self, response: Response, request: Request):
        """Apply comprehensive security headers"""
        
        # Content Security Policy
        csp_directives = [
            "default-src 'self'",
            "script-src 'self' 'unsafe-inline' https://cdnjs.cloudflare.com",
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com",
            "font-src 'self' https://fonts.gstatic.com",
            "img-src 'self' data: https:",
            "connect-src 'self'",
            "frame-ancestors 'none'",
            "base-uri 'self'",
            "form-action 'self'"
        ]
        
        if self.settings.ENVIRONMENT == "development":
            # More permissive CSP for development
            csp_directives = [
                "default-src 'self' 'unsafe-inline' 'unsafe-eval'",
                "script-src 'self' 'unsafe-inline' 'unsafe-eval'",
                "style-src 'self' 'unsafe-inline'",
                "img-src 'self' data: blob:",
                "connect-src 'self' ws: wss:",
                "frame-ancestors 'none'"
            ]
        
        # Security headers
        security_headers = {
            # Content Security Policy
            "Content-Security-Policy": "; ".join(csp_directives),
            
            # Prevent XSS attacks
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            
            # HTTPS enforcement (production only)
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains" if self.settings.ENVIRONMENT == "production" else "",
            
            # Referrer policy
            "Referrer-Policy": "strict-origin-when-cross-origin",
            
            # Feature policy / Permissions policy
            "Permissions-Policy": "camera=(), microphone=(), geolocation=(), payment=()",
            
            # Cache control for sensitive endpoints
            "Cache-Control": "no-store, no-cache, must-revalidate, private" if self._is_sensitive_endpoint(request) else "public, max-age=300",
            
            # Custom application headers
            "X-API-Version": "2.2.0",
            "X-Frame-Options": "DENY",
            "X-Powered-By": "",  # Remove server fingerprinting
        }
        
        # Apply headers to response
        for header_name, header_value in security_headers.items():
            if header_value:  # Only set non-empty headers
                response.headers[header_name] = header_value
    
    def _is_sensitive_endpoint(self, request: Request) -> bool:
        """Determine if endpoint contains sensitive data"""
        sensitive_paths = [
            "/api/auth/",
            "/api/users/",
            "/api/admin/",
            "/api/config/"
        ]
        
        return any(request.url.path.startswith(path) for path in sensitive_paths)

# ==========================================
# REQUEST LOGGING MIDDLEWARE
# ==========================================

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Enterprise request/response logging with structured logging
    Compatible with existing database logging infrastructure
    """
    
    def __init__(self, app, settings: Settings):
        super().__init__(app)
        self.settings = settings
        self.logger = structlog.get_logger("middleware.requests")
        
        # Request tracking
        self.request_counter = 0
        self.active_requests: Dict[str, Dict] = {}
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Log request and response with comprehensive metrics"""
        
        # Generate unique request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # Track request start
        start_time = time.time()
        self.request_counter += 1
        
        # Extract request information
        request_info = await self._extract_request_info(request)
        
        # Store active request
        self.active_requests[request_id] = {
            "start_time": start_time,
            "path": request.url.path,
            "method": request.method,
            "client_ip": request_info["client_ip"]
        }
        
        # Log request start
        self.logger.info(
            "Request started",
            request_id=request_id,
            **request_info,
            active_requests_count=len(self.active_requests)
        )
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate metrics
            duration_ms = (time.time() - start_time) * 1000
            
            # Extract response information
            response_info = self._extract_response_info(response)
            
            # Log successful request
            self.logger.info(
                "Request completed",
                request_id=request_id,
                duration_ms=round(duration_ms, 2),
                **request_info,
                **response_info,
                success=True
            )
            
            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id
            
            return response
            
        except Exception as e:
            # Calculate error metrics
            duration_ms = (time.time() - start_time) * 1000
            
            # Log error
            self.logger.error(
                "Request failed",
                request_id=request_id,
                duration_ms=round(duration_ms, 2),
                **request_info,
                error=str(e),
                error_type=type(e).__name__,
                traceback=traceback.format_exc() if self.settings.ENVIRONMENT == "development" else None,
                success=False
            )
            
            raise
            
        finally:
            # Clean up active request tracking
            self.active_requests.pop(request_id, None)
    
    async def _extract_request_info(self, request: Request) -> Dict[str, Any]:
        """Extract comprehensive request information"""
        
        # Get client IP (handle proxies)
        client_ip = request.client.host if request.client else "unknown"
        if "X-Forwarded-For" in request.headers:
            client_ip = request.headers["X-Forwarded-For"].split(",")[0].strip()
        elif "X-Real-IP" in request.headers:
            client_ip = request.headers["X-Real-IP"]
        
        # Extract request details
        request_info = {
            "method": request.method,
            "path": request.url.path,
            "query_params": dict(request.query_params),
            "client_ip": client_ip,
            "user_agent": request.headers.get("User-Agent", ""),
            "content_type": request.headers.get("Content-Type", ""),
            "content_length": request.headers.get("Content-Length", 0),
            "accept": request.headers.get("Accept", ""),
            "accept_encoding": request.headers.get("Accept-Encoding", ""),
            "accept_language": request.headers.get("Accept-Language", ""),
            "referer": request.headers.get("Referer", ""),
            "timestamp": datetime.now().isoformat()
        }
        
        # Extract authentication info if available
        if hasattr(request.state, "current_user") and request.state.current_user:
            request_info["user_id"] = getattr(request.state.current_user, "id", None)
            request_info["user_email"] = getattr(request.state.current_user, "email", None)
        
        # Extract API key info if available
        if "X-API-Key" in request.headers:
            api_key_hash = hashlib.sha256(
                request.headers["X-API-Key"].encode()
            ).hexdigest()[:8]
            request_info["api_key_hash"] = api_key_hash
        
        return request_info
    
    def _extract_response_info(self, response: Response) -> Dict[str, Any]:
        """Extract response information"""
        return {
            "status_code": response.status_code,
            "content_type": response.headers.get("Content-Type", ""),
            "content_length": response.headers.get("Content-Length", 0),
            "cache_control": response.headers.get("Cache-Control", "")
        }

# ==========================================
# RATE LIMITING MIDDLEWARE
# ==========================================

class RateLimitingMiddleware(BaseHTTPMiddleware):
    """
    Enterprise rate limiting with different limits per endpoint type
    Uses sliding window algorithm for accurate rate limiting
    """
    
    def __init__(self, app, settings: Settings):
        super().__init__(app)
        self.settings = settings
        self.logger = structlog.get_logger("middleware.ratelimit")
        
        # Rate limiting storage (in production, use Redis)
        self.request_counts: Dict[str, deque] = defaultdict(lambda: deque())
        self.cleanup_interval = 60  # seconds
        self.last_cleanup = time.time()
        
        # Define rate limits per endpoint type
        self.rate_limits = {
            # Authentication endpoints
            "/api/auth/login": {"requests": 5, "window": 60},  # 5 requests per minute
            "/api/auth/refresh": {"requests": 10, "window": 60},
            
            # Processing endpoints (more generous)
            "/api/books/": {"requests": 100, "window": 60},  # 100 requests per minute
            "/api/processing/": {"requests": 50, "window": 60},
            
            # Admin endpoints (restricted)
            "/api/admin/": {"requests": 20, "window": 60},
            
            # Default limit
            "default": {"requests": 200, "window": 60}  # 200 requests per minute
        }
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Apply rate limiting based on client IP and endpoint"""
        
        # Skip rate limiting for health checks
        if request.url.path in ["/health", "/metrics", "/docs", "/redoc"]:
            return await call_next(request)
        
        # Get client identifier
        client_id = self._get_client_identifier(request)
        
        # Get rate limit for this endpoint
        rate_limit = self._get_rate_limit_for_path(request.url.path)
        
        # Check rate limit
        if not self._check_rate_limit(client_id, request.url.path, rate_limit):
            self.logger.warning(
                "Rate limit exceeded",
                client_id=client_id,
                path=request.url.path,
                limit=rate_limit
            )
            
            return create_error_response(
                error_code=ErrorCode.RATE_LIMITED,
                message="Rate limit exceeded. Please try again later.",
                status_code=429,
                details={
                    "limit": rate_limit["requests"],
                    "window_seconds": rate_limit["window"],
                    "retry_after": rate_limit["window"]
                }
            )
        
        # Record this request
        self._record_request(client_id, request.url.path)
        
        # Periodic cleanup
        self._cleanup_old_requests()
        
        return await call_next(request)
    
    def _get_client_identifier(self, request: Request) -> str:
        """Get unique client identifier for rate limiting"""
        
        # Use authenticated user ID if available
        if hasattr(request.state, "current_user") and request.state.current_user:
            return f"user:{getattr(request.state.current_user, 'id', 'unknown')}"
        
        # Use API key if available
        if "X-API-Key" in request.headers:
            api_key_hash = hashlib.sha256(
                request.headers["X-API-Key"].encode()
            ).hexdigest()[:16]
            return f"api_key:{api_key_hash}"
        
        # Fall back to IP address
        client_ip = request.client.host if request.client else "unknown"
        if "X-Forwarded-For" in request.headers:
            client_ip = request.headers["X-Forwarded-For"].split(",")[0].strip()
        
        return f"ip:{client_ip}"
    
    def _get_rate_limit_for_path(self, path: str) -> Dict[str, int]:
        """Get rate limit configuration for specific path"""
        
        for pattern, limit in self.rate_limits.items():
            if pattern != "default" and path.startswith(pattern):
                return limit
        
        return self.rate_limits["default"]
    
    def _check_rate_limit(self, client_id: str, path: str, rate_limit: Dict[str, int]) -> bool:
        """Check if request is within rate limit using sliding window"""
        
        key = f"{client_id}:{path}"
        now = time.time()
        window_start = now - rate_limit["window"]
        
        # Get request timestamps for this client/path
        requests = self.request_counts[key]
        
        # Remove old requests outside the window
        while requests and requests[0] < window_start:
            requests.popleft()
        
        # Check if under limit
        return len(requests) < rate_limit["requests"]
    
    def _record_request(self, client_id: str, path: str):
        """Record a request timestamp"""
        key = f"{client_id}:{path}"
        self.request_counts[key].append(time.time())
    
    def _cleanup_old_requests(self):
        """Periodic cleanup of old request records"""
        now = time.time()
        
        if now - self.last_cleanup < self.cleanup_interval:
            return
        
        # Clean up old entries
        max_age = 300  # Keep records for 5 minutes
        cutoff_time = now - max_age
        
        keys_to_remove = []
        for key, requests in self.request_counts.items():
            # Remove old requests
            while requests and requests[0] < cutoff_time:
                requests.popleft()
            
            # Remove empty deques
            if not requests:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.request_counts[key]
        
        self.last_cleanup = now
        
        if keys_to_remove:
            self.logger.debug(
                "Rate limit cleanup completed",
                removed_keys=len(keys_to_remove),
                active_keys=len(self.request_counts)
            )

# ==========================================
# ERROR HANDLING MIDDLEWARE
# ==========================================

class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """
    Enterprise error handling middleware with structured logging
    Integrates with existing error response system
    """
    
    def __init__(self, app, settings: Settings):
        super().__init__(app)
        self.settings = settings
        self.logger = structlog.get_logger("middleware.errors")
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Handle all exceptions with structured logging and appropriate responses"""
        
        try:
            return await call_next(request)
            
        except HTTPException as e:
            # Handle FastAPI HTTP exceptions
            self.logger.info(
                "HTTP exception handled",
                status_code=e.status_code,
                detail=e.detail,
                path=request.url.path,
                method=request.method,
                request_id=getattr(request.state, "request_id", "unknown")
            )
            
            # Return standardized error response
            return create_error_response(
                error_code=self._map_http_status_to_error_code(e.status_code),
                message=str(e.detail),
                status_code=e.status_code,
                details={"original_error": "HTTPException"}
            )
            
        except ValueError as e:
            # Handle validation errors
            self.logger.warning(
                "Validation error",
                error=str(e),
                path=request.url.path,
                method=request.method,
                request_id=getattr(request.state, "request_id", "unknown")
            )
            
            return create_error_response(
                error_code=ErrorCode.VALIDATION_ERROR,
                message="Invalid input data",
                status_code=400,
                details={"validation_error": str(e)}
            )
            
        except PermissionError as e:
            # Handle permission errors
            self.logger.warning(
                "Permission denied",
                error=str(e),
                path=request.url.path,
                method=request.method,
                request_id=getattr(request.state, "request_id", "unknown")
            )
            
            return create_error_response(
                error_code=ErrorCode.PERMISSION_DENIED,
                message="Insufficient permissions",
                status_code=403,
                details={"permission_error": str(e)}
            )
            
        except Exception as e:
            # Handle unexpected errors
            error_id = str(uuid.uuid4())
            
            self.logger.error(
                "Unexpected error",
                error_id=error_id,
                error=str(e),
                error_type=type(e).__name__,
                path=request.url.path,
                method=request.method,
                request_id=getattr(request.state, "request_id", "unknown"),
                traceback=traceback.format_exc() if self.settings.ENVIRONMENT == "development" else None
            )
            
            # Return generic error response for security
            message = str(e) if self.settings.ENVIRONMENT == "development" else "An internal error occurred"
            
            return create_error_response(
                error_code=ErrorCode.INTERNAL_ERROR,
                message=message,
                status_code=500,
                details={
                    "error_id": error_id,
                    "support_message": "Please contact support with this error ID"
                }
            )
    
    def _map_http_status_to_error_code(self, status_code: int) -> ErrorCode:
        """Map HTTP status codes to internal error codes"""
        mapping = {
            400: ErrorCode.VALIDATION_ERROR,
            401: ErrorCode.AUTHENTICATION_FAILED,
            403: ErrorCode.PERMISSION_DENIED,
            404: ErrorCode.NOT_FOUND,
            409: ErrorCode.CONFLICT,
            422: ErrorCode.VALIDATION_ERROR,
            429: ErrorCode.RATE_LIMITED,
            500: ErrorCode.INTERNAL_ERROR,
            502: ErrorCode.EXTERNAL_SERVICE_ERROR,
            503: ErrorCode.SERVICE_UNAVAILABLE
        }
        
        return mapping.get(status_code, ErrorCode.INTERNAL_ERROR)

# ==========================================
# PERFORMANCE MONITORING MIDDLEWARE
# ==========================================

class PerformanceMonitoringMiddleware(BaseHTTPMiddleware):
    """
    Enterprise performance monitoring with metrics collection
    Integrates with database statistics system
    """
    
    def __init__(self, app, settings: Settings):
        super().__init__(app)
        self.settings = settings
        self.logger = structlog.get_logger("middleware.performance")
        
        # Performance metrics
        self.metrics = {
            "total_requests": 0,
            "total_errors": 0,
            "response_times": deque(maxlen=1000),  # Keep last 1000 response times
            "slow_requests": deque(maxlen=100),    # Keep last 100 slow requests
            "endpoint_stats": defaultdict(lambda: {
                "count": 0,
                "total_time": 0,
                "errors": 0
            })
        }
        
        self.slow_request_threshold = 2000  # ms
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Monitor request performance and collect metrics"""
        
        start_time = time.time()
        start_memory = await self._get_memory_usage()
        
        try:
            response = await call_next(request)
            
            # Calculate metrics
            duration_ms = (time.time() - start_time) * 1000
            end_memory = await self._get_memory_usage()
            memory_delta = end_memory - start_memory if end_memory and start_memory else 0
            
            # Update metrics
            self._update_metrics(request, duration_ms, memory_delta, success=True)
            
            # Log slow requests
            if duration_ms > self.slow_request_threshold:
                self._log_slow_request(request, duration_ms, memory_delta)
            
            # Add performance headers
            response.headers["X-Response-Time"] = f"{duration_ms:.2f}ms"
            if memory_delta > 0:
                response.headers["X-Memory-Delta"] = f"{memory_delta:.2f}MB"
            
            return response
            
        except Exception as e:
            # Calculate error metrics
            duration_ms = (time.time() - start_time) * 1000
            end_memory = await self._get_memory_usage()
            memory_delta = end_memory - start_memory if end_memory and start_memory else 0
            
            # Update error metrics
            self._update_metrics(request, duration_ms, memory_delta, success=False)
            
            raise
    
    async def _get_memory_usage(self) -> Optional[float]:
        """Get current memory usage in MB"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # Convert to MB
        except ImportError:
            return None
        except Exception as e:
            self.logger.debug("Error getting memory usage", error=str(e))
            return None
    
    def _update_metrics(self, request: Request, duration_ms: float, 
                       memory_delta: float, success: bool):
        """Update performance metrics"""
        
        endpoint = f"{request.method} {request.url.path}"
        
        # Update global metrics
        self.metrics["total_requests"] += 1
        if not success:
            self.metrics["total_errors"] += 1
        
        self.metrics["response_times"].append(duration_ms)
        
        # Update endpoint-specific metrics
        endpoint_stats = self.metrics["endpoint_stats"][endpoint]
        endpoint_stats["count"] += 1
        endpoint_stats["total_time"] += duration_ms
        if not success:
            endpoint_stats["errors"] += 1
        
        # Log performance metrics periodically
        if self.metrics["total_requests"] % 100 == 0:
            self._log_performance_summary()
    
    def _log_slow_request(self, request: Request, duration_ms: float, memory_delta: float):
        """Log details of slow requests"""
        
        slow_request_info = {
            "path": request.url.path,
            "method": request.method,
            "duration_ms": round(duration_ms, 2),
            "memory_delta_mb": round(memory_delta, 2) if memory_delta else None,
            "query_params": dict(request.query_params),
            "user_agent": request.headers.get("User-Agent", ""),
            "timestamp": datetime.now().isoformat()
        }
        
        self.metrics["slow_requests"].append(slow_request_info)
        
        self.logger.warning(
            "Slow request detected",
            **slow_request_info,
            threshold_ms=self.slow_request_threshold
        )
    
    def _log_performance_summary(self):
        """Log performance summary statistics"""
        
        if not self.metrics["response_times"]:
            return
        
        # Calculate statistics
        response_times = list(self.metrics["response_times"])
        avg_response_time = sum(response_times) / len(response_times)
        
        # Calculate percentiles
        sorted_times = sorted(response_times)
        p50 = sorted_times[int(len(sorted_times) * 0.5)]
        p95 = sorted_times[int(len(sorted_times) * 0.95)]
        p99 = sorted_times[int(len(sorted_times) * 0.99)]
        
        error_rate = (self.metrics["total_errors"] / self.metrics["total_requests"]) * 100
        
        self.logger.info(
            "Performance summary",
            total_requests=self.metrics["total_requests"],
            total_errors=self.metrics["total_errors"],
            error_rate_percent=round(error_rate, 2),
            avg_response_time_ms=round(avg_response_time, 2),
            p50_response_time_ms=round(p50, 2),
            p95_response_time_ms=round(p95, 2),
            p99_response_time_ms=round(p99, 2),
            slow_requests_count=len(self.metrics["slow_requests"])
        )
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get current metrics summary for health check endpoint"""
        
        if not self.metrics["response_times"]:
            return {"status": "no_data"}
        
        response_times = list(self.metrics["response_times"])
        avg_response_time = sum(response_times) / len(response_times)
        
        sorted_times = sorted(response_times)
        p95 = sorted_times[int(len(sorted_times) * 0.95)]
        
        error_rate = (self.metrics["total_errors"] / self.metrics["total_requests"]) * 100
        
        return {
            "total_requests": self.metrics["total_requests"],
            "total_errors": self.metrics["total_errors"],
            "error_rate_percent": round(error_rate, 2),
            "avg_response_time_ms": round(avg_response_time, 2),
            "p95_response_time_ms": round(p95, 2),
            "slow_requests_count": len(self.metrics["slow_requests"]),
            "endpoints_count": len(self.metrics["endpoint_stats"])
        }

# ==========================================
# MAINTENANCE MODE MIDDLEWARE
# ==========================================

class MaintenanceModeMiddleware(BaseHTTPMiddleware):
    """
    Enterprise maintenance mode with configurable bypass
    Allows graceful system maintenance
    """
    
    def __init__(self, app, settings: Settings):
        super().__init__(app)
        self.settings = settings
        self.logger = structlog.get_logger("middleware.maintenance")
        
        # Maintenance mode configuration
        self.maintenance_mode_file = Path("maintenance_mode.txt")
        self.bypass_paths = [
            "/health",
            "/metrics", 
            "/api/admin/maintenance",
            "/docs",
            "/redoc"
        ]
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Check maintenance mode and handle accordingly"""
        
        # Check if maintenance mode is enabled
        if not self._is_maintenance_mode_enabled():
            return await call_next(request)
        
        # Allow bypass paths
        if any(request.url.path.startswith(path) for path in self.bypass_paths):
            return await call_next(request)
        
        # Check for admin bypass
        if self._has_admin_bypass(request):
            return await call_next(request)
        
        # Return maintenance mode response
        maintenance_info = self._get_maintenance_info()
        
        self.logger.info(
            "Request blocked - maintenance mode",
            path=request.url.path,
            method=request.method,
            client_ip=request.client.host if request.client else "unknown"
        )
        
        return JSONResponse(
            status_code=503,
            content={
                "error": "Service Unavailable",
                "message": "System is currently under maintenance",
                "maintenance_info": maintenance_info,
                "retry_after": maintenance_info.get("estimated_duration", 3600)
            },
            headers={
                "Retry-After": str(maintenance_info.get("estimated_duration", 3600)),
                "X-Maintenance-Mode": "true"
            }
        )
    
    def _is_maintenance_mode_enabled(self) -> bool:
        """Check if maintenance mode is enabled"""
        return self.maintenance_mode_file.exists()
    
    def _has_admin_bypass(self, request: Request) -> bool:
        """Check if request has admin bypass token"""
        bypass_token = request.headers.get("X-Maintenance-Bypass")
        if not bypass_token:
            return False
        
        # Verify bypass token (in production, use secure token verification)
        expected_token = self.settings.MAINTENANCE_BYPASS_TOKEN
        return bypass_token == expected_token
    
    def _get_maintenance_info(self) -> Dict[str, Any]:
        """Get maintenance mode information"""
        try:
            if self.maintenance_mode_file.exists():
                content = self.maintenance_mode_file.read_text().strip()
                
                # Try to parse as JSON for structured info
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    # Fall back to simple message
                    return {
                        "message": content,
                        "started_at": datetime.now().isoformat(),
                        "estimated_duration": 3600
                    }
            
            return {
                "message": "System maintenance in progress",
                "started_at": datetime.now().isoformat(),
                "estimated_duration": 3600
            }
            
        except Exception as e:
            self.logger.error("Error reading maintenance info", error=str(e))
            return {
                "message": "System maintenance in progress",
                "error": "Could not read maintenance details"
            }

# ==========================================
# MIDDLEWARE FACTORY FUNCTIONS
# ==========================================

def setup_cors_middleware(app: FastAPI, settings: Settings):
    """Configure CORS middleware based on environment"""
    
    if settings.ENVIRONMENT == "development":
        # Permissive CORS for development
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["http://localhost:3000", "http://localhost:8501", "http://127.0.0.1:8501"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
            expose_headers=["X-Request-ID", "X-Response-Time"]
        )
    elif settings.ENVIRONMENT == "production":
        # Restrictive CORS for production
        allowed_origins = settings.CORS_ORIGINS.split(",") if settings.CORS_ORIGINS else []
        
        app.add_middleware(
            CORSMiddleware,
            allow_origins=allowed_origins,
            allow_credentials=True,
            allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
            allow_headers=["Content-Type", "Authorization", "X-API-Key", "X-Request-ID"],
            expose_headers=["X-Request-ID", "X-Response-Time"]
        )
    else:
        # Default CORS for staging/testing
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure based on staging environment
            allow_credentials=False,
            allow_methods=["GET", "POST", "PUT", "DELETE"],
            allow_headers=["Content-Type", "Authorization", "X-API-Key"]
        )

def setup_compression_middleware(app: FastAPI, settings: Settings):
    """Configure compression middleware"""
    
    # Add GZip compression for responses > 1KB
    app.add_middleware(
        GZipMiddleware,
        minimum_size=1024,
        compresslevel=6  # Good balance between compression and speed
    )

def setup_trusted_host_middleware(app: FastAPI, settings: Settings):
    """Configure trusted host middleware for security"""
    
    if settings.ENVIRONMENT == "production":
        allowed_hosts = settings.ALLOWED_HOSTS.split(",") if settings.ALLOWED_HOSTS else ["*"]
        
        if "*" not in allowed_hosts:
            app.add_middleware(
                TrustedHostMiddleware,
                allowed_hosts=allowed_hosts
            )

def setup_session_middleware(app: FastAPI, settings: Settings):
    """Configure session middleware"""
    
    if settings.SESSION_SECRET_KEY:
        app.add_middleware(
            SessionMiddleware,
            secret_key=settings.SESSION_SECRET_KEY,
            session_cookie="sessionid",
            max_age=settings.SESSION_TIMEOUT_MINUTES * 60,
            same_site="lax",
            https_only=settings.ENVIRONMENT == "production"
        )

def setup_all_middleware(app: FastAPI, settings: Settings) -> Dict[str, Any]:
    """
    Setup all middleware in correct order
    Returns middleware instances for health checks and monitoring
    """
    
    logger.info("Setting up middleware stack", environment=settings.ENVIRONMENT)
    
    # Store middleware instances for monitoring
    middleware_instances = {}
    
    # 1. Maintenance mode (first - blocks everything if enabled)
    maintenance_middleware = MaintenanceModeMiddleware(app, settings)
    app.add_middleware(type(maintenance_middleware), app=app, settings=settings)
    middleware_instances["maintenance"] = maintenance_middleware
    
    # 2. Trusted host (security)
    setup_trusted_host_middleware(app, settings)
    
    # 3. CORS (before other middleware that might modify headers)
    setup_cors_middleware(app, settings)
    
    # 4. Security headers
    security_middleware = SecurityHeadersMiddleware(app, settings)
    app.add_middleware(type(security_middleware), app=app, settings=settings)
    middleware_instances["security"] = security_middleware
    
    # 5. Performance monitoring
    performance_middleware = PerformanceMonitoringMiddleware(app, settings)
    app.add_middleware(type(performance_middleware), app=app, settings=settings)
    middleware_instances["performance"] = performance_middleware
    
    # 6. Request logging
    logging_middleware = RequestLoggingMiddleware(app, settings)
    app.add_middleware(type(logging_middleware), app=app, settings=settings)
    middleware_instances["logging"] = logging_middleware
    
    # 7. Rate limiting
    rate_limit_middleware = RateLimitingMiddleware(app, settings)
    app.add_middleware(type(rate_limit_middleware), app=app, settings=settings)
    middleware_instances["rate_limiting"] = rate_limit_middleware
    
    # 8. Error handling (should be last to catch all errors)
    error_middleware = ErrorHandlingMiddleware(app, settings)
    app.add_middleware(type(error_middleware), app=app, settings=settings)
    middleware_instances["error_handling"] = error_middleware
    
    # 9. Compression (FastAPI built-in)
    setup_compression_middleware(app, settings)
    
    # 10. Session middleware (if configured)
    setup_session_middleware(app, settings)
    
    logger.info(
        "Middleware stack setup completed",
        middleware_count=len(middleware_instances),
        middleware_types=list(middleware_instances.keys())
    )
    
    return middleware_instances

# ==========================================
# HEALTH CHECK INTEGRATION
# ==========================================

def get_middleware_health(middleware_instances: Dict[str, Any]) -> Dict[str, Any]:
    """Get health status of all middleware components"""
    
    health_status = {
        "status": "healthy",
        "middleware": {},
        "total_requests": 0,
        "total_errors": 0,
        "timestamp": datetime.now().isoformat()
    }
    
    try:
        # Performance middleware metrics
        if "performance" in middleware_instances:
            perf_metrics = middleware_instances["performance"].get_metrics_summary()
            health_status["middleware"]["performance"] = perf_metrics
            health_status["total_requests"] = perf_metrics.get("total_requests", 0)
            health_status["total_errors"] = perf_metrics.get("total_errors", 0)
        
        # Rate limiting status
        if "rate_limiting" in middleware_instances:
            rate_limit = middleware_instances["rate_limiting"]
            health_status["middleware"]["rate_limiting"] = {
                "active_limits": len(rate_limit.request_counts),
                "cleanup_interval": rate_limit.cleanup_interval
            }
        
        # Maintenance mode status
        if "maintenance" in middleware_instances:
            maintenance = middleware_instances["maintenance"]
            health_status["middleware"]["maintenance"] = {
                "enabled": maintenance._is_maintenance_mode_enabled(),
                "bypass_paths": maintenance.bypass_paths
            }
        
        # Determine overall health
        error_rate = 0
        if health_status["total_requests"] > 0:
            error_rate = (health_status["total_errors"] / health_status["total_requests"]) * 100
        
        if error_rate > 10:  # More than 10% error rate
            health_status["status"] = "degraded"
        elif error_rate > 25:  # More than 25% error rate
            health_status["status"] = "unhealthy"
        
    except Exception as e:
        logger.error("Error getting middleware health", error=str(e))
        health_status["status"] = "error"
        health_status["error"] = str(e)
    
    return health_status

# Export all middleware components
__all__ = [
    "SecurityHeadersMiddleware",
    "RequestLoggingMiddleware", 
    "RateLimitingMiddleware",
    "ErrorHandlingMiddleware",
    "PerformanceMonitoringMiddleware",
    "MaintenanceModeMiddleware",
    "setup_cors_middleware",
    "setup_compression_middleware",
    "setup_trusted_host_middleware", 
    "setup_session_middleware",
    "setup_all_middleware",
    "get_middleware_health"
]
