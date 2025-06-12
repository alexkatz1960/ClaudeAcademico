 # ==========================================
# INTERFACES/FASTAPI_BACKEND/API/ENDPOINTS/ERRORS.PY
# Enterprise Error Management Endpoints - FastAPI
# Sistema de Traducción Académica v2.2
# ==========================================

from fastapi import APIRouter, Depends, HTTPException, Query, Path
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc, func, text
from typing import List, Optional, Dict, Any, Union
from datetime import datetime, timedelta
from enum import Enum
import structlog
import json
from dataclasses import dataclass

# Internal imports
from ...database.database import get_db, db_transaction
from ...database.models import (
    BookProcessingHistory, AuditLog, ErrorPattern, 
    EditorialReview, UsageStatistic
)
from ...database.schemas import (
    ErrorPatternResponse, ErrorPatternCreate, ErrorPatternUpdate,
    AuditLogResponse, ErrorSummaryResponse, ErrorAnalyticsResponse,
    ErrorResolutionRequest, ErrorResolutionResponse,
    ErrorPatternAnalyticsResponse, SystemErrorResponse
)
from ...database.crud import (
    ErrorPatternCRUD, AuditCRUD, BookCRUD, EditorialReviewCRUD
)
from ...database.enums import (
    ErrorSeverity, PatternType, BookStatus, ReviewSeverity
)
from ..dependencies import (
    get_current_user, require_permissions, get_rate_limiter
)
from ...core.security import Permission
from ...utils.responses import (
    create_success_response, create_error_response,
    ErrorCode, SuccessResponse, ErrorResponse
)

# Configure logger
logger = structlog.get_logger(__name__)

# Create router
router = APIRouter(
    prefix="/errors",
    tags=["errors"],
    responses={
        404: {"description": "Error not found"},
        403: {"description": "Insufficient permissions"},
        422: {"description": "Validation error"}
    }
)

# ==========================================
# ERROR CLASSIFICATION ENUMS
# ==========================================

class ErrorTimeframe(str, Enum):
    """Timeframe for error analytics"""
    LAST_HOUR = "last_hour"
    LAST_24H = "last_24h"
    LAST_WEEK = "last_week"
    LAST_MONTH = "last_month"
    LAST_QUARTER = "last_quarter"

class ErrorGroupBy(str, Enum):
    """Grouping options for error analytics"""
    SEVERITY = "severity"
    PHASE = "phase"
    BOOK_ID = "book_id"
    PATTERN_TYPE = "pattern_type"
    DATE = "date"

@dataclass
class ErrorAnalytics:
    """Analytics data structure for errors"""
    total_errors: int
    critical_errors: int
    resolved_errors: int
    resolution_rate: float
    avg_resolution_time_minutes: float
    most_common_error_types: List[Dict[str, Any]]
    error_trends: List[Dict[str, Any]]

# ==========================================
# ERROR MANAGEMENT SERVICE
# ==========================================

class ErrorManagementService:
    """Enterprise error management service with comprehensive analytics"""
    
    def __init__(self):
        self.logger = structlog.get_logger("services.error_management")
    
    def get_active_errors(self, db: Session, limit: int = 50, 
                         severity_filter: Optional[ErrorSeverity] = None,
                         phase_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get active errors from audit logs with comprehensive filtering"""
        
        try:
            audit_crud = AuditCRUD(db)
            
            # Build query filters
            query_filters = {}
            
            if severity_filter:
                # Filter by critical alerts
                query_filters["critical_alerts_count__gt"] = 0
            
            if phase_filter:
                query_filters["phase_name"] = phase_filter
            
            # Get recent audit logs with alerts
            recent_logs = audit_crud.get_recent_with_alerts(
                filters=query_filters,
                limit=limit
            )
            
            active_errors = []
            
            for log in recent_logs:
                if log.alerts_detail:
                    for alert in log.alerts_detail:
                        error_item = {
                            "id": f"audit_{log.id}_{alert.get('type', 'unknown')}",
                            "book_id": log.book_id,
                            "phase": log.phase_name,
                            "error_type": alert.get("type", "unknown"),
                            "severity": alert.get("severity", "medium"),
                            "message": alert.get("message", ""),
                            "context": alert.get("context", {}),
                            "timestamp": alert.get("timestamp", log.created_at.isoformat()),
                            "quality_score": log.quality_score,
                            "resolved": False,  # Audit logs don't track resolution
                            "source": "audit_log"
                        }
                        active_errors.append(error_item)
            
            # Also get unresolved editorial reviews with high severity
            review_crud = EditorialReviewCRUD(db)
            unresolved_reviews = review_crud.get_unresolved_by_severity(
                min_severity=ReviewSeverity.HIGH,
                limit=limit//2
            )
            
            for review in unresolved_reviews:
                error_item = {
                    "id": f"review_{review.id}",
                    "book_id": review.book_id,
                    "phase": "editorial_review",
                    "error_type": review.alert_type,
                    "severity": review.severity,
                    "message": f"Editorial review required: {review.suggested_action}",
                    "context": {
                        "section_number": review.section_number,
                        "similarity_score": review.similarity_score,
                        "location": review.location_info
                    },
                    "timestamp": review.created_at.isoformat(),
                    "quality_score": review.similarity_score,
                    "resolved": review.resolved,
                    "source": "editorial_review"
                }
                active_errors.append(error_item)
            
            # Sort by timestamp (most recent first) and apply limit
            active_errors.sort(key=lambda x: x["timestamp"], reverse=True)
            
            self.logger.info(
                "Retrieved active errors",
                total_errors=len(active_errors),
                severity_filter=severity_filter.value if severity_filter else None,
                phase_filter=phase_filter
            )
            
            return active_errors[:limit]
            
        except Exception as e:
            self.logger.error("Error getting active errors", error=str(e))
            raise
    
    def get_error_analytics(self, db: Session, timeframe: ErrorTimeframe) -> ErrorAnalytics:
        """Get comprehensive error analytics for specified timeframe"""
        
        try:
            # Calculate date range
            now = datetime.now()
            if timeframe == ErrorTimeframe.LAST_HOUR:
                start_date = now - timedelta(hours=1)
            elif timeframe == ErrorTimeframe.LAST_24H:
                start_date = now - timedelta(days=1)
            elif timeframe == ErrorTimeframe.LAST_WEEK:
                start_date = now - timedelta(weeks=1)
            elif timeframe == ErrorTimeframe.LAST_MONTH:
                start_date = now - timedelta(days=30)
            else:  # LAST_QUARTER
                start_date = now - timedelta(days=90)
            
            audit_crud = AuditCRUD(db)
            review_crud = EditorialReviewCRUD(db)
            
            # Get audit logs with errors in timeframe
            error_logs = audit_crud.get_by_date_range(
                start_date=start_date,
                end_date=now,
                min_alerts=1
            )
            
            # Get editorial reviews in timeframe
            error_reviews = review_crud.get_by_date_range(
                start_date=start_date,
                end_date=now,
                resolved_only=False
            )
            
            # Calculate basic metrics
            total_errors = len(error_logs) + len(error_reviews)
            
            critical_errors = (
                len([log for log in error_logs if log.critical_alerts_count > 0]) +
                len([rev for rev in error_reviews if rev.severity == ReviewSeverity.CRITICAL.value])
            )
            
            resolved_errors = len([rev for rev in error_reviews if rev.resolved])
            
            resolution_rate = (resolved_errors / total_errors * 100) if total_errors > 0 else 0
            
            # Calculate average resolution time
            resolved_reviews_with_time = [
                rev for rev in error_reviews 
                if rev.resolved and rev.resolution_time_minutes
            ]
            
            avg_resolution_time = (
                sum(rev.resolution_time_minutes for rev in resolved_reviews_with_time) / 
                len(resolved_reviews_with_time)
            ) if resolved_reviews_with_time else 0
            
            # Get most common error types
            error_type_counts = {}
            
            for log in error_logs:
                if log.alerts_detail:
                    for alert in log.alerts_detail:
                        error_type = alert.get("type", "unknown")
                        error_type_counts[error_type] = error_type_counts.get(error_type, 0) + 1
            
            for review in error_reviews:
                error_type = review.alert_type
                error_type_counts[error_type] = error_type_counts.get(error_type, 0) + 1
            
            most_common_error_types = [
                {"type": error_type, "count": count}
                for error_type, count in sorted(error_type_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            ]
            
            # Calculate error trends by day
            error_trends = self._calculate_error_trends(error_logs, error_reviews, start_date, now)
            
            analytics = ErrorAnalytics(
                total_errors=total_errors,
                critical_errors=critical_errors,
                resolved_errors=resolved_errors,
                resolution_rate=resolution_rate,
                avg_resolution_time_minutes=avg_resolution_time,
                most_common_error_types=most_common_error_types,
                error_trends=error_trends
            )
            
            self.logger.info(
                "Generated error analytics",
                timeframe=timeframe.value,
                total_errors=total_errors,
                critical_errors=critical_errors,
                resolution_rate=resolution_rate
            )
            
            return analytics
            
        except Exception as e:
            self.logger.error("Error generating analytics", error=str(e))
            raise
    
    def _calculate_error_trends(self, error_logs: List[AuditLog], 
                               error_reviews: List[EditorialReview],
                               start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
        """Calculate daily error trends"""
        
        try:
            # Group errors by date
            daily_errors = {}
            
            current_date = start_date.date()
            end_date_only = end_date.date()
            
            # Initialize all dates with zero
            while current_date <= end_date_only:
                daily_errors[current_date.isoformat()] = {
                    "date": current_date.isoformat(),
                    "total_errors": 0,
                    "critical_errors": 0,
                    "resolved_errors": 0
                }
                current_date += timedelta(days=1)
            
            # Count errors from audit logs
            for log in error_logs:
                date_key = log.created_at.date().isoformat()
                if date_key in daily_errors:
                    daily_errors[date_key]["total_errors"] += log.alerts_count or 0
                    daily_errors[date_key]["critical_errors"] += log.critical_alerts_count or 0
            
            # Count errors from editorial reviews
            for review in error_reviews:
                date_key = review.created_at.date().isoformat()
                if date_key in daily_errors:
                    daily_errors[date_key]["total_errors"] += 1
                    if review.severity == ReviewSeverity.CRITICAL.value:
                        daily_errors[date_key]["critical_errors"] += 1
                    if review.resolved:
                        daily_errors[date_key]["resolved_errors"] += 1
            
            return list(daily_errors.values())
            
        except Exception as e:
            self.logger.error("Error calculating trends", error=str(e))
            return []
    
    def resolve_error(self, db: Session, error_id: str, resolution_data: Dict[str, Any],
                     user_id: str) -> bool:
        """Resolve error by ID with comprehensive tracking"""
        
        try:
            # Parse error ID to determine source
            if error_id.startswith("review_"):
                review_id = int(error_id.replace("review_", ""))
                review_crud = EditorialReviewCRUD(db)
                
                review = review_crud.get_by_id(review_id)
                if not review:
                    return False
                
                # Mark review as resolved
                update_data = {
                    "resolved": True,
                    "resolved_at": datetime.now(),
                    "editor_notes": resolution_data.get("notes", ""),
                    "reviewer_id": user_id
                }
                
                # Calculate resolution time if reviewed
                if review.reviewed_at:
                    resolution_time = (datetime.now() - review.reviewed_at).total_seconds() / 60
                    update_data["resolution_time_minutes"] = int(resolution_time)
                
                review_crud.update(review_id, update_data)
                
                self.logger.info(
                    "Resolved editorial review error",
                    error_id=error_id,
                    review_id=review_id,
                    resolved_by=user_id
                )
                
                return True
            
            elif error_id.startswith("audit_"):
                # Audit log errors can't be directly resolved, but we can log the resolution
                self.logger.info(
                    "Marked audit log error as acknowledged",
                    error_id=error_id,
                    acknowledged_by=user_id,
                    notes=resolution_data.get("notes", "")
                )
                
                return True
            
            else:
                self.logger.warning("Unknown error ID format", error_id=error_id)
                return False
                
        except Exception as e:
            self.logger.error("Error resolving error", error_id=error_id, error=str(e))
            return False

# ==========================================
# SERVICE INSTANCE
# ==========================================

error_service = ErrorManagementService()

# ==========================================
# ERROR TRACKING ENDPOINTS
# ==========================================

@router.get("/active",
    response_model=SuccessResponse[List[Dict[str, Any]]],
    summary="Get active errors",
    description="Retrieve all active errors from the system with optional filtering"
)
async def get_active_errors(
    limit: int = Query(50, ge=1, le=200, description="Maximum number of errors to return"),
    severity: Optional[ErrorSeverity] = Query(None, description="Filter by error severity"),
    phase: Optional[str] = Query(None, description="Filter by processing phase"),
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user),
    _rate_limiter = Depends(get_rate_limiter("errors"))
):
    """
    Get active system errors with comprehensive filtering
    
    Enterprise features:
    - Multi-source error aggregation (audit logs + editorial reviews)
    - Severity-based filtering
    - Phase-specific error tracking
    - Real-time error status
    """
    
    try:
        logger.info(
            "Getting active errors",
            limit=limit,
            severity=severity.value if severity else None,
            phase=phase,
            user_id=getattr(current_user, 'id', 'api_user')
        )
        
        active_errors = error_service.get_active_errors(
            db=db,
            limit=limit,
            severity_filter=severity,
            phase_filter=phase
        )
        
        return create_success_response(
            data=active_errors,
            message=f"Retrieved {len(active_errors)} active errors"
        )
        
    except Exception as e:
        logger.error("Error getting active errors", error=str(e))
        
        return create_error_response(
            error_code=ErrorCode.INTERNAL_ERROR,
            message="Failed to retrieve active errors",
            status_code=500,
            details={"error": str(e)}
        )

@router.get("/analytics",
    response_model=SuccessResponse[ErrorAnalyticsResponse],
    summary="Get error analytics",
    description="Get comprehensive error analytics and trends for specified timeframe"
)
async def get_error_analytics(
    timeframe: ErrorTimeframe = Query(ErrorTimeframe.LAST_24H, description="Analytics timeframe"),
    include_trends: bool = Query(True, description="Include daily error trends"),
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """
    Get comprehensive error analytics and trends
    
    Enterprise features:
    - Multi-timeframe analytics
    - Error trend analysis
    - Resolution rate tracking
    - Most common error types
    """
    
    try:
        logger.info(
            "Generating error analytics",
            timeframe=timeframe.value,
            user_id=getattr(current_user, 'id', 'api_user')
        )
        
        analytics = error_service.get_error_analytics(db, timeframe)
        
        response_data = ErrorAnalyticsResponse(
            timeframe=timeframe.value,
            total_errors=analytics.total_errors,
            critical_errors=analytics.critical_errors,
            resolved_errors=analytics.resolved_errors,
            resolution_rate_percentage=analytics.resolution_rate,
            avg_resolution_time_minutes=analytics.avg_resolution_time_minutes,
            most_common_error_types=analytics.most_common_error_types,
            error_trends=analytics.error_trends if include_trends else []
        )
        
        return create_success_response(
            data=response_data,
            message="Error analytics generated successfully"
        )
        
    except Exception as e:
        logger.error("Error generating analytics", error=str(e))
        
        return create_error_response(
            error_code=ErrorCode.INTERNAL_ERROR,
            message="Failed to generate error analytics",
            status_code=500,
            details={"error": str(e)}
        )

@router.post("/{error_id}/resolve",
    response_model=SuccessResponse[ErrorResolutionResponse],
    summary="Resolve error",
    description="Mark error as resolved with resolution notes and tracking"
)
async def resolve_error(
    error_id: str = Path(..., description="Error ID to resolve"),
    resolution: ErrorResolutionRequest = ...,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """
    Resolve error with comprehensive tracking
    
    Enterprise features:
    - Error resolution workflow
    - Resolution time tracking
    - User attribution
    - Audit trail
    """
    
    try:
        user_id = getattr(current_user, 'id', 'api_user')
        
        logger.info(
            "Resolving error",
            error_id=error_id,
            resolved_by=user_id,
            action=resolution.action
        )
        
        success = error_service.resolve_error(
            db=db,
            error_id=error_id,
            resolution_data={
                "action": resolution.action,
                "notes": resolution.notes,
                "resolution_type": resolution.resolution_type
            },
            user_id=user_id
        )
        
        if not success:
            return create_error_response(
                error_code=ErrorCode.NOT_FOUND,
                message=f"Error {error_id} not found or cannot be resolved",
                status_code=404
            )
        
        response_data = ErrorResolutionResponse(
            error_id=error_id,
            resolved=True,
            resolved_by=user_id,
            resolved_at=datetime.now(),
            resolution_action=resolution.action,
            resolution_notes=resolution.notes
        )
        
        return create_success_response(
            data=response_data,
            message="Error resolved successfully"
        )
        
    except Exception as e:
        logger.error("Error resolving error", error_id=error_id, error=str(e))
        
        return create_error_response(
            error_code=ErrorCode.INTERNAL_ERROR,
            message="Failed to resolve error",
            status_code=500,
            details={"error_id": error_id, "error": str(e)}
        )

@router.get("/summary",
    response_model=SuccessResponse[ErrorSummaryResponse],
    summary="Get error summary",
    description="Get high-level error summary for dashboard display"
)
async def get_error_summary(
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """
    Get high-level error summary for dashboard
    
    Enterprise features:
    - Real-time error counts
    - Critical error alerting
    - System health indicators
    """
    
    try:
        # Get 24-hour analytics for summary
        analytics = error_service.get_error_analytics(db, ErrorTimeframe.LAST_24H)
        
        # Get current active errors count
        active_errors = error_service.get_active_errors(db, limit=1000)
        current_active = len(active_errors)
        current_critical = len([e for e in active_errors if e["severity"] == "critical"])
        
        response_data = ErrorSummaryResponse(
            total_active_errors=current_active,
            critical_active_errors=current_critical,
            errors_last_24h=analytics.total_errors,
            critical_errors_last_24h=analytics.critical_errors,
            resolution_rate_last_24h=analytics.resolution_rate,
            avg_resolution_time_minutes=analytics.avg_resolution_time_minutes,
            system_health_status="healthy" if current_critical == 0 else "warning" if current_critical < 5 else "critical"
        )
        
        return create_success_response(
            data=response_data,
            message="Error summary retrieved successfully"
        )
        
    except Exception as e:
        logger.error("Error getting error summary", error=str(e))
        
        return create_error_response(
            error_code=ErrorCode.INTERNAL_ERROR,
            message="Failed to retrieve error summary",
            status_code=500,
            details={"error": str(e)}
        )

# ==========================================
# ERROR PATTERN MANAGEMENT ENDPOINTS
# ==========================================

@router.get("/patterns",
    response_model=SuccessResponse[List[ErrorPatternResponse]],
    summary="Get error patterns",
    description="Retrieve learned error patterns for ML analysis"
)
async def get_error_patterns(
    pattern_type: Optional[PatternType] = Query(None, description="Filter by pattern type"),
    min_effectiveness: float = Query(0.0, ge=0.0, le=1.0, description="Minimum effectiveness score"),
    limit: int = Query(50, ge=1, le=200, description="Maximum patterns to return"),
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """
    Get learned error patterns for analysis
    
    Enterprise features:
    - Pattern type filtering
    - Effectiveness-based sorting
    - ML integration ready
    """
    
    try:
        error_pattern_crud = ErrorPatternCRUD(db)
        
        # Build filters
        filters = {}
        if pattern_type:
            filters["pattern_type"] = pattern_type.value
        if min_effectiveness > 0:
            filters["effectiveness_score__gte"] = min_effectiveness
        
        patterns = error_pattern_crud.get_by_filters(filters, limit=limit)
        
        response_data = [
            ErrorPatternResponse(
                id=pattern.id,
                pattern_type=pattern.pattern_type,
                pattern_content=pattern.pattern_content,
                effectiveness_score=pattern.effectiveness_score,
                success_rate=pattern.success_rate,
                usage_count=pattern.usage_count,
                is_active=pattern.is_active,
                created_at=pattern.created_at,
                last_seen=pattern.last_seen
            )
            for pattern in patterns
        ]
        
        return create_success_response(
            data=response_data,
            message=f"Retrieved {len(response_data)} error patterns"
        )
        
    except Exception as e:
        logger.error("Error getting error patterns", error=str(e))
        
        return create_error_response(
            error_code=ErrorCode.INTERNAL_ERROR,
            message="Failed to retrieve error patterns",
            status_code=500,
            details={"error": str(e)}
        )

@router.post("/patterns",
    response_model=SuccessResponse[ErrorPatternResponse],
    summary="Create error pattern",
    description="Create new error pattern for ML learning"
)
async def create_error_pattern(
    pattern: ErrorPatternCreate,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user),
    _permissions = Depends(require_permissions([Permission.MANAGE_SYSTEM]))
):
    """
    Create new error pattern for ML learning
    
    Enterprise features:
    - Pattern validation
    - Effectiveness tracking
    - User attribution
    """
    
    try:
        user_id = getattr(current_user, 'id', 'api_user')
        
        logger.info(
            "Creating error pattern",
            pattern_type=pattern.pattern_type,
            created_by=user_id
        )
        
        error_pattern_crud = ErrorPatternCRUD(db)
        
        pattern_data = {
            **pattern.dict(),
            "created_by": user_id,
            "description": pattern.description or f"Pattern created by {user_id}"
        }
        
        new_pattern = error_pattern_crud.create(pattern_data)
        
        response_data = ErrorPatternResponse(
            id=new_pattern.id,
            pattern_type=new_pattern.pattern_type,
            pattern_content=new_pattern.pattern_content,
            effectiveness_score=new_pattern.effectiveness_score,
            success_rate=new_pattern.success_rate,
            usage_count=new_pattern.usage_count,
            is_active=new_pattern.is_active,
            created_at=new_pattern.created_at,
            last_seen=new_pattern.last_seen
        )
        
        return create_success_response(
            data=response_data,
            message="Error pattern created successfully",
            status_code=201
        )
        
    except Exception as e:
        logger.error("Error creating error pattern", error=str(e))
        
        return create_error_response(
            error_code=ErrorCode.INTERNAL_ERROR,
            message="Failed to create error pattern",
            status_code=500,
            details={"error": str(e)}
        )

@router.get("/patterns/analytics",
    response_model=SuccessResponse[ErrorPatternAnalyticsResponse],
    summary="Get error pattern analytics",
    description="Get analytics on error pattern effectiveness and usage"
)
async def get_error_pattern_analytics(
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """
    Get comprehensive error pattern analytics
    
    Enterprise features:
    - Pattern effectiveness analysis
    - Usage statistics
    - ML model performance tracking
    """
    
    try:
        error_pattern_crud = ErrorPatternCRUD(db)
        
        # Get all patterns for analysis
        all_patterns = error_pattern_crud.get_all()
        
        if not all_patterns:
            response_data = ErrorPatternAnalyticsResponse(
                total_patterns=0,
                active_patterns=0,
                avg_effectiveness_score=0.0,
                most_effective_patterns=[],
                pattern_usage_distribution={},
                effectiveness_distribution={}
            )
        else:
            active_patterns = [p for p in all_patterns if p.is_active]
            
            # Calculate averages
            avg_effectiveness = sum(p.effectiveness_score for p in all_patterns) / len(all_patterns)
            
            # Get most effective patterns
            most_effective = sorted(all_patterns, key=lambda x: x.effectiveness_score, reverse=True)[:5]
            most_effective_data = [
                {
                    "id": p.id,
                    "pattern_type": p.pattern_type,
                    "effectiveness_score": p.effectiveness_score,
                    "usage_count": p.usage_count
                }
                for p in most_effective
            ]
            
            # Pattern usage distribution by type
            usage_distribution = {}
            for pattern in all_patterns:
                pattern_type = pattern.pattern_type
                usage_distribution[pattern_type] = usage_distribution.get(pattern_type, 0) + pattern.usage_count
            
            # Effectiveness distribution
            effectiveness_ranges = {
                "0.0-0.2": 0, "0.2-0.4": 0, "0.4-0.6": 0,
                "0.6-0.8": 0, "0.8-1.0": 0
            }
            
            for pattern in all_patterns:
                score = pattern.effectiveness_score
                if score < 0.2:
                    effectiveness_ranges["0.0-0.2"] += 1
                elif score < 0.4:
                    effectiveness_ranges["0.2-0.4"] += 1
                elif score < 0.6:
                    effectiveness_ranges["0.4-0.6"] += 1
                elif score < 0.8:
                    effectiveness_ranges["0.6-0.8"] += 1
                else:
                    effectiveness_ranges["0.8-1.0"] += 1
            
            response_data = ErrorPatternAnalyticsResponse(
                total_patterns=len(all_patterns),
                active_patterns=len(active_patterns),
                avg_effectiveness_score=avg_effectiveness,
                most_effective_patterns=most_effective_data,
                pattern_usage_distribution=usage_distribution,
                effectiveness_distribution=effectiveness_ranges
            )
        
        return create_success_response(
            data=response_data,
            message="Error pattern analytics generated successfully"
        )
        
    except Exception as e:
        logger.error("Error getting pattern analytics", error=str(e))
        
        return create_error_response(
            error_code=ErrorCode.INTERNAL_ERROR,
            message="Failed to generate pattern analytics",
            status_code=500,
            details={"error": str(e)}
        )

# ==========================================
# SYSTEM ERROR ENDPOINTS
# ==========================================

@router.get("/system",
    response_model=SuccessResponse[List[SystemErrorResponse]],
    summary="Get system errors",
    description="Get system-level errors requiring immediate attention"
)
async def get_system_errors(
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user),
    _permissions = Depends(require_permissions([Permission.VIEW_SYSTEM_LOGS]))
):
    """
    Get critical system errors requiring immediate attention
    
    Enterprise features:
    - System health monitoring
    - Critical error detection
    - Admin-only access
    """
    
    try:
        # Get critical errors from audit logs
        audit_crud = AuditCRUD(db)
        
        # Get recent critical errors
        critical_logs = audit_crud.get_recent_with_alerts(
            filters={"critical_alerts_count__gt": 0},
            limit=20
        )
        
        system_errors = []
        
        for log in critical_logs:
            if log.alerts_detail:
                for alert in log.alerts_detail:
                    if alert.get("severity") == "critical":
                        system_error = SystemErrorResponse(
                            id=f"system_{log.id}_{alert.get('type', 'unknown')}",
                            error_type=alert.get("type", "unknown"),
                            message=alert.get("message", ""),
                            component=log.phase_name,
                            severity="critical",
                            timestamp=alert.get("timestamp", log.created_at.isoformat()),
                            book_id=log.book_id,
                            context=alert.get("context", {}),
                            requires_attention=True
                        )
                        system_errors.append(system_error)
        
        # Sort by timestamp (most recent first)
        system_errors.sort(key=lambda x: x.timestamp, reverse=True)
        
        return create_success_response(
            data=system_errors,
            message=f"Retrieved {len(system_errors)} system errors"
        )
        
    except Exception as e:
        logger.error("Error getting system errors", error=str(e))
        
        return create_error_response(
            error_code=ErrorCode.INTERNAL_ERROR,
            message="Failed to retrieve system errors",
            status_code=500,
            details={"error": str(e)}
        )

# Export all endpoint functions for testing
__all__ = [
    "router",
    "get_active_errors",
    "get_error_analytics", 
    "resolve_error",
    "get_error_summary",
    "get_error_patterns",
    "create_error_pattern",
    "get_error_pattern_analytics",
    "get_system_errors"
]
