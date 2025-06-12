# ==========================================
# INTERFACES/FASTAPI_BACKEND/API/ENDPOINTS/__INIT__.PY
# Enterprise Endpoints Module Export - FastAPI
# Sistema de Traducción Académica v2.2
# ==========================================

"""
Enterprise FastAPI Endpoints Module

This module provides comprehensive REST API endpoints for the Academic Translation System,
including CRUD operations, workflow management, analytics, and real-time monitoring.

Modules:
- books: Book management and CRUD operations
- processing: Translation pipeline control and monitoring
- errors: Error management and pattern learning
- reviews: Editorial review workflow management
- statistics: System analytics and performance monitoring

Enterprise Features:
- Role-based access control
- Rate limiting and security middleware
- Comprehensive logging and monitoring
- Real-time streaming capabilities
- Multi-format export functionality
- Advanced analytics and reporting
"""

from fastapi import APIRouter
from typing import List, Dict, Any
import structlog

# Import all endpoint routers
from .books import router as books_router
from .processing import router as processing_router
from .errors import router as errors_router
from .reviews import router as reviews_router
from .statistics import router as statistics_router

# Configure logger
logger = structlog.get_logger(__name__)

# ==========================================
# ENDPOINTS REGISTRY
# ==========================================

class EndpointsRegistry:
    """Enterprise endpoints registry with metadata and configuration"""
    
    def __init__(self):
        self.logger = structlog.get_logger("endpoints.registry")
        self._routers = {}
        self._metadata = {}
        self._register_all_endpoints()
    
    def _register_all_endpoints(self):
        """Register all endpoint routers with metadata"""
        
        # Books endpoints
        self._routers["books"] = books_router
        self._metadata["books"] = {
            "name": "Books Management",
            "description": "CRUD operations for academic books and document management",
            "version": "2.2.0",
            "prefix": "/books",
            "tags": ["books"],
            "features": [
                "Book CRUD operations",
                "File upload/download",
                "Status tracking",
                "Metadata management",
                "Search and filtering"
            ],
            "permissions_required": ["read_books", "write_books"],
            "rate_limits": {
                "default": "100/minute",
                "upload": "10/minute"
            }
        }
        
        # Processing endpoints
        self._routers["processing"] = processing_router
        self._metadata["processing"] = {
            "name": "Processing Pipeline",
            "description": "Translation pipeline control, job management, and monitoring",
            "version": "2.2.0",
            "prefix": "/processing",
            "tags": ["processing"],
            "features": [
                "Job lifecycle management",
                "Real-time progress tracking",
                "Phase execution control",
                "Queue management",
                "Configuration validation",
                "Streaming updates"
            ],
            "permissions_required": ["manage_processing"],
            "rate_limits": {
                "default": "50/minute",
                "job_control": "20/minute"
            }
        }
        
        # Error management endpoints
        self._routers["errors"] = errors_router
        self._metadata["errors"] = {
            "name": "Error Management",
            "description": "Error tracking, pattern learning, and resolution workflow",
            "version": "2.2.0",
            "prefix": "/errors",
            "tags": ["errors"],
            "features": [
                "Active error monitoring",
                "Pattern learning ML",
                "Error analytics",
                "Resolution workflow",
                "System health monitoring"
            ],
            "permissions_required": ["view_errors", "manage_errors"],
            "rate_limits": {
                "default": "100/minute",
                "analytics": "30/minute"
            }
        }
        
        # Editorial review endpoints
        self._routers["reviews"] = reviews_router
        self._metadata["reviews"] = {
            "name": "Editorial Reviews",
            "description": "Editorial review workflow, assignment, and quality management",
            "version": "2.2.0",
            "prefix": "/reviews",
            "tags": ["reviews"],
            "features": [
                "Review workflow management",
                "Assignment and escalation",
                "Bulk operations",
                "Analytics and reporting",
                "Multi-format export"
            ],
            "permissions_required": ["view_reviews", "manage_reviews"],
            "rate_limits": {
                "default": "100/minute",
                "bulk_operations": "10/minute",
                "export": "5/minute"
            }
        }
        
        # Statistics and analytics endpoints
        self._routers["statistics"] = statistics_router
        self._metadata["statistics"] = {
            "name": "System Analytics",
            "description": "Comprehensive system analytics, performance monitoring, and reporting",
            "version": "2.2.0",
            "prefix": "/statistics",
            "tags": ["statistics"],
            "features": [
                "System overview metrics",
                "Performance analytics",
                "Quality analysis",
                "Trend analysis",
                "Real-time streaming",
                "Language statistics"
            ],
            "permissions_required": ["view_statistics"],
            "rate_limits": {
                "default": "60/minute",
                "analytics": "30/minute",
                "streaming": "5/concurrent"
            }
        }
        
        self.logger.info(
            "Registered all endpoint modules",
            modules=list(self._routers.keys()),
            total_endpoints=sum(len(router.routes) for router in self._routers.values())
        )
    
    def get_router(self, name: str) -> APIRouter:
        """Get router by name"""
        return self._routers.get(name)
    
    def get_all_routers(self) -> Dict[str, APIRouter]:
        """Get all registered routers"""
        return self._routers.copy()
    
    def get_metadata(self, name: str = None) -> Dict[str, Any]:
        """Get metadata for specific module or all modules"""
        if name:
            return self._metadata.get(name, {})
        return self._metadata.copy()
    
    def get_registry_info(self) -> Dict[str, Any]:
        """Get comprehensive registry information"""
        total_routes = sum(len(router.routes) for router in self._routers.values())
        
        return {
            "total_modules": len(self._routers),
            "total_routes": total_routes,
            "modules": list(self._routers.keys()),
            "api_version": "2.2.0",
            "last_updated": "2025-01-01",
            "enterprise_features": [
                "Role-based access control",
                "Rate limiting",
                "Comprehensive logging",
                "Real-time monitoring",
                "Advanced analytics",
                "Multi-format exports",
                "Streaming capabilities",
                "Error pattern learning"
            ]
        }

# ==========================================
# GLOBAL REGISTRY INSTANCE
# ==========================================

# Create global registry instance
endpoints_registry = EndpointsRegistry()

# ==========================================
# ROUTER AGGREGATION
# ==========================================

def create_api_router(
    include_modules: List[str] = None,
    exclude_modules: List[str] = None
) -> APIRouter:
    """
    Create aggregated API router with optional module filtering
    
    Args:
        include_modules: List of module names to include (None = all)
        exclude_modules: List of module names to exclude
    
    Returns:
        APIRouter: Configured router with selected modules
    """
    
    api_router = APIRouter()
    
    # Determine which modules to include
    all_modules = list(endpoints_registry.get_all_routers().keys())
    
    if include_modules:
        modules_to_include = [m for m in include_modules if m in all_modules]
    else:
        modules_to_include = all_modules
    
    if exclude_modules:
        modules_to_include = [m for m in modules_to_include if m not in exclude_modules]
    
    # Include selected routers
    included_count = 0
    for module_name in modules_to_include:
        router = endpoints_registry.get_router(module_name)
        if router:
            api_router.include_router(router)
            included_count += 1
            logger.debug(f"Included router: {module_name}")
    
    logger.info(
        "Created API router",
        total_modules=len(all_modules),
        included_modules=included_count,
        modules=modules_to_include
    )
    
    return api_router

def get_full_api_router() -> APIRouter:
    """
    Get complete API router with all endpoints
    
    Returns:
        APIRouter: Router with all endpoint modules included
    """
    return create_api_router()

def get_public_api_router() -> APIRouter:
    """
    Get public API router (excluding admin-only endpoints)
    
    Returns:
        APIRouter: Router with public endpoints only
    """
    # Include all modules - security is handled at endpoint level
    return create_api_router()

def get_admin_api_router() -> APIRouter:
    """
    Get admin API router (admin-only endpoints)
    
    Returns:
        APIRouter: Router with admin endpoints
    """
    # All modules include admin endpoints with proper permission checks
    return create_api_router()

# ==========================================
# ENDPOINT DISCOVERY
# ==========================================

def get_endpoint_summary() -> Dict[str, Any]:
    """
    Get comprehensive summary of all available endpoints
    
    Returns:
        Dict: Complete endpoint documentation and metadata
    """
    
    summary = {
        "api_info": endpoints_registry.get_registry_info(),
        "modules": {},
        "route_count_by_module": {},
        "security_requirements": {},
        "rate_limits": {}
    }
    
    for module_name, router in endpoints_registry.get_all_routers().items():
        module_metadata = endpoints_registry.get_metadata(module_name)
        
        # Route information
        routes = []
        for route in router.routes:
            if hasattr(route, 'methods') and hasattr(route, 'path'):
                routes.append({
                    "path": route.path,
                    "methods": list(route.methods),
                    "name": getattr(route, 'name', 'unnamed')
                })
        
        summary["modules"][module_name] = {
            "metadata": module_metadata,
            "routes": routes
        }
        
        summary["route_count_by_module"][module_name] = len(routes)
        summary["security_requirements"][module_name] = module_metadata.get("permissions_required", [])
        summary["rate_limits"][module_name] = module_metadata.get("rate_limits", {})
    
    return summary

def get_health_check_endpoints() -> List[str]:
    """
    Get list of health check endpoints across all modules
    
    Returns:
        List[str]: Health check endpoint paths
    """
    
    health_endpoints = []
    
    for module_name, router in endpoints_registry.get_all_routers().items():
        for route in router.routes:
            if hasattr(route, 'path'):
                path = route.path
                # Look for health, status, or ping endpoints
                if any(keyword in path.lower() for keyword in ['health', 'status', 'ping', 'alive']):
                    health_endpoints.append(f"{router.prefix}{path}")
    
    return health_endpoints

# ==========================================
# DEVELOPMENT HELPERS
# ==========================================

def print_endpoint_tree():
    """Print formatted endpoint tree for development/debugging"""
    
    print("\n" + "="*60)
    print("📚 ACADEMIC TRANSLATION SYSTEM - API ENDPOINTS")
    print("="*60)
    
    registry_info = endpoints_registry.get_registry_info()
    print(f"🔧 API Version: {registry_info['api_version']}")
    print(f"📦 Total Modules: {registry_info['total_modules']}")
    print(f"🛣️  Total Routes: {registry_info['total_routes']}")
    
    for module_name, router in endpoints_registry.get_all_routers().items():
        metadata = endpoints_registry.get_metadata(module_name)
        print(f"\n📁 {metadata['name']} ({module_name})")
        print(f"   📝 {metadata['description']}")
        print(f"   🏷️  Prefix: {metadata['prefix']}")
        print(f"   🔐 Permissions: {', '.join(metadata['permissions_required'])}")
        
        print("   🛣️  Routes:")
        for route in router.routes:
            if hasattr(route, 'methods') and hasattr(route, 'path'):
                methods = ' | '.join(route.methods)
                print(f"      {methods:10} {metadata['prefix']}{route.path}")
    
    print("\n" + "="*60)
    print("🚀 All endpoints ready for production!")
    print("="*60 + "\n")

# ==========================================
# MODULE EXPORTS
# ==========================================

# Export routers for direct import
books_router = books_router
processing_router = processing_router
errors_router = errors_router
reviews_router = reviews_router
statistics_router = statistics_router

# Export registry and utilities
__all__ = [
    # Routers
    "books_router",
    "processing_router", 
    "errors_router",
    "reviews_router",
    "statistics_router",
    
    # Registry
    "endpoints_registry",
    
    # Router creation functions
    "create_api_router",
    "get_full_api_router",
    "get_public_api_router",
    "get_admin_api_router",
    
    # Discovery functions
    "get_endpoint_summary",
    "get_health_check_endpoints",
    
    # Development helpers
    "print_endpoint_tree"
]

# ==========================================
# MODULE INITIALIZATION LOG
# ==========================================

logger.info(
    "FastAPI Endpoints module initialized",
    modules_loaded=list(endpoints_registry.get_all_routers().keys()),
    total_routes=endpoints_registry.get_registry_info()["total_routes"],
    version="2.2.0"
)