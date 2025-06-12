 # ==========================================
# INTERFACES/FASTAPI_BACKEND/API/ENDPOINTS/PROCESSING.PY
# Enterprise Processing Pipeline Endpoints - FastAPI (REFINADO)
# Sistema de Traducción Académica v2.2
# ==========================================

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc, func
from typing import List, Optional, Dict, Any, Union, AsyncIterator
from datetime import datetime, timedelta
from pathlib import Path
import asyncio
import json
import structlog
import uuid
from dataclasses import dataclass, field
from enum import Enum
import time
from contextlib import asynccontextmanager

# Internal imports
from ...database.database import get_db, db_transaction
from ...database.models import BookProcessingHistory, AuditLog, ErrorPattern, TerminologySuggestion
from ...database.schemas import (
    ProcessingJobCreate, ProcessingJobResponse, ProcessingStatusResponse,
    ProcessingConfigUpdate, ProcessingMetricsResponse, ProcessingErrorResponse,
    PipelinePhaseResponse, ProcessingQueueResponse
)
from ...database.crud import BookCRUD, AuditCRUD, ErrorPatternCRUD
from ...database.enums import BookStatus, ProcessingPhase, ErrorSeverity, PatternType
from ..dependencies import (
    get_current_user, require_permissions, get_rate_limiter, get_optional_current_user
)
from ...core.security import Permission
from ...core.config import get_settings
from ...utils.responses import (
    create_success_response, create_error_response, create_streaming_response,
    ErrorCode, SuccessResponse, ErrorResponse
)

# Core processing imports (these would integrate with actual components)
# TODO: Implement actual integration with core processing components
# from ....core.advanced_pdf_cleaner import AdvancedPDFCleaner
# from ....core.semantic_validator import SemanticIntegrityValidator
# from ....core.html_to_docx_converter import HTMLtoDocxConverter
# from ....core.footnote_reconnection_engine import FootnoteReconnectionEngine

# Configure logger
logger = structlog.get_logger(__name__)

# Create router
router = APIRouter(
    prefix="/processing",
    tags=["processing"],
    responses={
        404: {"description": "Processing job not found"},
        403: {"description": "Insufficient permissions"},
        409: {"description": "Processing conflict"}
    }
)

settings = get_settings()

# ==========================================
# ENTERPRISE PROCESSING SERVICES
# ==========================================

class ProcessingJobStatus(str, Enum):
    """Processing job status enumeration"""
    QUEUED = "queued"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class ProcessingPriority(str, Enum):
    """Processing priority levels"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"

@dataclass
class ProcessingJobConfig:
    """Configuration for processing jobs"""
    source_lang: str
    target_lang: str = "es"
    quality_threshold: float = 0.85
    enable_claude_refinement: bool = True
    preserve_formatting: bool = True
    generate_review: bool = True
    skip_phases: List[str] = field(default_factory=list)
    custom_glossary: Optional[str] = None
    max_retries: int = 3
    timeout_minutes: int = 120
    priority: ProcessingPriority = ProcessingPriority.NORMAL

    def __post_init__(self):
        """Validate configuration after initialization"""
        # MEJORA 3: Validación de fases permitidas
        if self.skip_phases:
            valid_phases = {phase.value for phase in ProcessingPhase}
            invalid_phases = [phase for phase in self.skip_phases if phase not in valid_phases]
            if invalid_phases:
                raise ValueError(
                    f"Invalid phases in skip_phases: {invalid_phases}. "
                    f"Valid phases: {list(valid_phases)}"
                )

@dataclass 
class ProcessingJobMetrics:
    """Metrics for processing jobs"""
    started_at: datetime
    current_phase: str
    progress_percentage: float = 0.0
    phases_completed: List[str] = field(default_factory=list)
    quality_scores: Dict[str, float] = field(default_factory=dict)
    processing_time_seconds: int = 0
    memory_usage_mb: float = 0.0
    error_count: int = 0
    retry_count: int = 0
    estimated_completion: Optional[datetime] = None
    # MEJORA 1: Métricas de finalización
    completed_at: Optional[datetime] = None
    total_duration_seconds: Optional[int] = None
    final_quality_score: Optional[float] = None

# ==========================================
# ENTERPRISE UTILITY FUNCTIONS
# ==========================================

def is_processable_status(status: str) -> bool:
    """
    MEJORA 2: Función centralizada para validar estados procesables
    Evita repetición de validaciones de estado en múltiples endpoints
    """
    return status in [BookStatus.QUEUED.value, BookStatus.FAILED.value]

def calculate_safe_average(values: List[float]) -> float:
    """
    MEJORA 4: Cálculo seguro de promedio evitando división por cero
    """
    if not values or len(values) == 0:
        return 0.0
    return sum(values) / len(values)

class ProcessingPipelineService:
    """Enterprise processing pipeline service with comprehensive job management"""
    
    def __init__(self):
        self.logger = structlog.get_logger("services.processing_pipeline")
        self.active_jobs: Dict[str, Dict[str, Any]] = {}
        self.job_configs: Dict[str, ProcessingJobConfig] = {}
        self.job_metrics: Dict[str, ProcessingJobMetrics] = {}
        
        # Phase definitions with estimated durations
        self.pipeline_phases = {
            ProcessingPhase.PDF_CLEANUP: {
                "name": "PDF Cleanup",
                "estimated_duration_minutes": 5,
                "description": "Clean PDF artifacts and prepare for conversion",
                "weight": 0.15
            },
            ProcessingPhase.CONVERSION: {
                "name": "HTML to DOCX Conversion", 
                "estimated_duration_minutes": 8,
                "description": "Convert cleaned content to DOCX with format preservation",
                "weight": 0.20
            },
            ProcessingPhase.TRANSLATION: {
                "name": "Hybrid Translation",
                "estimated_duration_minutes": 45,
                "description": "DeepL translation with Claude refinement",
                "weight": 0.50
            },
            ProcessingPhase.VALIDATION: {
                "name": "Semantic Validation",
                "estimated_duration_minutes": 10,
                "description": "Validate translation integrity and quality",
                "weight": 0.10
            },
            ProcessingPhase.REVIEW_GENERATION: {
                "name": "Editorial Review Generation",
                "estimated_duration_minutes": 3,
                "description": "Generate editorial review materials",
                "weight": 0.05
            }
        }
    
    async def start_processing_job(self, book_id: str, config: ProcessingJobConfig,
                                 user_id: str, db: Session) -> str:
        """Start new processing job with comprehensive setup"""
        
        job_id = f"job_{uuid.uuid4().hex[:12]}"
        
        try:
            self.logger.info(
                "Starting processing job",
                job_id=job_id,
                book_id=book_id,
                config=config.__dict__,
                user_id=user_id
            )
            
            # Validate book exists and is in valid state
            book_crud = BookCRUD(db)
            book = book_crud.get_by_id(book_id)
            
            if not book:
                raise ValueError(f"Book {book_id} not found")
            
            # MEJORA 2: Usar función centralizada para validación de estado
            if not is_processable_status(book.status):
                raise ValueError(f"Book {book_id} not in processable state: {book.status}")
            
            # Initialize job tracking
            self.job_configs[job_id] = config
            self.job_metrics[job_id] = ProcessingJobMetrics(
                started_at=datetime.now(),
                current_phase=ProcessingPhase.PDF_CLEANUP.value
            )
            
            self.active_jobs[job_id] = {
                "book_id": book_id,
                "status": ProcessingJobStatus.STARTING,
                "started_by": user_id,
                "current_phase": ProcessingPhase.PDF_CLEANUP.value,
                "cancellation_requested": False
            }
            
            # Update book status
            book_crud.update(book_id, {
                "status": BookStatus.PROCESSING.value,
                "current_phase": ProcessingPhase.PDF_CLEANUP.value,
                "started_at": datetime.now(),
                "last_activity_at": datetime.now()
            })
            
            # Create initial audit log
            audit_crud = AuditCRUD(db)
            audit_crud.create({
                "book_id": book_id,
                "phase_name": "job_start",
                "quality_score": 1.0,
                "metrics_detail": {
                    "job_id": job_id,
                    "config": config.__dict__,
                    "started_by": user_id
                },
                "system_version": "2.2.0"
            })
            
            self.logger.info(
                "Processing job started successfully",
                job_id=job_id,
                book_id=book_id
            )
            
            return job_id
            
        except Exception as e:
            self.logger.error(
                "Error starting processing job",
                job_id=job_id,
                book_id=book_id,
                error=str(e)
            )
            
            # Cleanup on error
            self._cleanup_job(job_id)
            raise
    
    async def complete_processing_job(self, job_id: str, success: bool = True, 
                                    final_metrics: Optional[Dict[str, Any]] = None) -> bool:
        """
        MEJORA 1: Finalización explícita del job con métricas completas
        Almacena completed_at, total_duration y métricas finales para trazabilidad
        """
        
        try:
            if job_id not in self.active_jobs:
                raise ValueError(f"Job {job_id} not found")
            
            completion_time = datetime.now()
            job = self.active_jobs[job_id]
            metrics = self.job_metrics.get(job_id)
            
            # Calcular duración total
            if metrics and metrics.started_at:
                total_duration = (completion_time - metrics.started_at).total_seconds()
                metrics.completed_at = completion_time
                metrics.total_duration_seconds = int(total_duration)
                
                # Calcular score final de calidad
                if metrics.quality_scores:
                    metrics.final_quality_score = calculate_safe_average(
                        list(metrics.quality_scores.values())
                    )
            
            # Actualizar estado del job
            final_status = ProcessingJobStatus.COMPLETED if success else ProcessingJobStatus.FAILED
            job["status"] = final_status
            job["completed_at"] = completion_time
            job["final_metrics"] = final_metrics or {}
            
            self.logger.info(
                "Processing job completed",
                job_id=job_id,
                book_id=job["book_id"],
                success=success,
                total_duration_seconds=metrics.total_duration_seconds if metrics else None,
                final_quality_score=metrics.final_quality_score if metrics else None
            )
            
            return True
            
        except Exception as e:
            self.logger.error(
                "Error completing processing job",
                job_id=job_id,
                error=str(e)
            )
            return False
    
    async def pause_processing_job(self, job_id: str, user_id: str) -> bool:
        """Pause processing job gracefully"""
        
        try:
            if job_id not in self.active_jobs:
                raise ValueError(f"Job {job_id} not found")
            
            job = self.active_jobs[job_id]
            
            if job["status"] != ProcessingJobStatus.RUNNING:
                raise ValueError(f"Job {job_id} not in running state: {job['status']}")
            
            # Request pause (actual implementation would coordinate with running phases)
            job["status"] = ProcessingJobStatus.PAUSED
            job["paused_by"] = user_id
            job["paused_at"] = datetime.now()
            
            self.logger.info(
                "Processing job paused",
                job_id=job_id,
                book_id=job["book_id"],
                paused_by=user_id
            )
            
            return True
            
        except Exception as e:
            self.logger.error(
                "Error pausing processing job",
                job_id=job_id,
                error=str(e)
            )
            return False
    
    async def resume_processing_job(self, job_id: str, user_id: str) -> bool:
        """Resume paused processing job"""
        
        try:
            if job_id not in self.active_jobs:
                raise ValueError(f"Job {job_id} not found")
            
            job = self.active_jobs[job_id]
            
            if job["status"] != ProcessingJobStatus.PAUSED:
                raise ValueError(f"Job {job_id} not in paused state: {job['status']}")
            
            # Resume processing
            job["status"] = ProcessingJobStatus.RUNNING
            job["resumed_by"] = user_id
            job["resumed_at"] = datetime.now()
            
            self.logger.info(
                "Processing job resumed",
                job_id=job_id,
                book_id=job["book_id"],
                resumed_by=user_id
            )
            
            return True
            
        except Exception as e:
            self.logger.error(
                "Error resuming processing job",
                job_id=job_id,
                error=str(e)
            )
            return False
    
    async def cancel_processing_job(self, job_id: str, user_id: str, reason: str = "") -> bool:
        """Cancel processing job with cleanup"""
        
        try:
            if job_id not in self.active_jobs:
                raise ValueError(f"Job {job_id} not found")
            
            job = self.active_jobs[job_id]
            
            if job["status"] in [ProcessingJobStatus.COMPLETED, ProcessingJobStatus.CANCELLED]:
                raise ValueError(f"Job {job_id} already finished: {job['status']}")
            
            # Request cancellation
            job["cancellation_requested"] = True
            job["cancelled_by"] = user_id
            job["cancellation_reason"] = reason
            job["cancelled_at"] = datetime.now()
            
            # If job is running, mark for graceful shutdown
            if job["status"] == ProcessingJobStatus.RUNNING:
                job["status"] = ProcessingJobStatus.CANCELLED
            
            self.logger.warning(
                "Processing job cancelled",
                job_id=job_id,
                book_id=job["book_id"],
                cancelled_by=user_id,
                reason=reason
            )
            
            return True
            
        except Exception as e:
            self.logger.error(
                "Error cancelling processing job",
                job_id=job_id,
                error=str(e)
            )
            return False
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive job status"""
        
        if job_id not in self.active_jobs:
            return None
        
        job = self.active_jobs[job_id]
        config = self.job_configs.get(job_id)
        metrics = self.job_metrics.get(job_id)
        
        # Calculate progress
        progress = self._calculate_job_progress(job_id)
        
        return {
            "job_id": job_id,
            "book_id": job["book_id"],
            "status": job["status"],
            "current_phase": job["current_phase"],
            "progress_percentage": progress,
            "config": config.__dict__ if config else {},
            "metrics": metrics.__dict__ if metrics else {},
            "started_at": job.get("started_at"),
            "estimated_completion": self._calculate_eta(job_id)
        }
    
    def get_active_jobs(self) -> List[Dict[str, Any]]:
        """Get all active processing jobs"""
        
        active_jobs = []
        
        for job_id in self.active_jobs:
            job_status = self.get_job_status(job_id)
            if job_status and job_status["status"] in [
                ProcessingJobStatus.QUEUED,
                ProcessingJobStatus.STARTING, 
                ProcessingJobStatus.RUNNING,
                ProcessingJobStatus.PAUSED
            ]:
                active_jobs.append(job_status)
        
        return sorted(active_jobs, key=lambda x: x.get("started_at", datetime.min), reverse=True)
    
    def _calculate_job_progress(self, job_id: str) -> float:
        """Calculate job progress based on completed phases"""
        
        if job_id not in self.job_metrics:
            return 0.0
        
        metrics = self.job_metrics[job_id]
        total_weight = sum(phase["weight"] for phase in self.pipeline_phases.values())
        completed_weight = 0.0
        
        for phase_name in metrics.phases_completed:
            if phase_name in [p.value for p in ProcessingPhase]:
                phase_enum = ProcessingPhase(phase_name)
                if phase_enum in self.pipeline_phases:
                    completed_weight += self.pipeline_phases[phase_enum]["weight"]
        
        # Add partial progress for current phase if available
        current_phase = metrics.current_phase
        if current_phase and current_phase in [p.value for p in ProcessingPhase]:
            phase_enum = ProcessingPhase(current_phase)
            if phase_enum in self.pipeline_phases:
                # Estimate partial completion based on time spent
                phase_duration = self.pipeline_phases[phase_enum]["estimated_duration_minutes"]
                time_in_phase = (datetime.now() - metrics.started_at).total_seconds() / 60
                partial_completion = min(time_in_phase / phase_duration, 0.9)  # Max 90% for current phase
                completed_weight += self.pipeline_phases[phase_enum]["weight"] * partial_completion
        
        progress = (completed_weight / total_weight) * 100
        return min(progress, 99.9)  # Never show 100% until actually complete
    
    def _calculate_eta(self, job_id: str) -> Optional[datetime]:
        """Calculate estimated completion time"""
        
        if job_id not in self.job_metrics:
            return None
        
        metrics = self.job_metrics[job_id]
        progress = self._calculate_job_progress(job_id)
        
        if progress <= 0:
            return None
        
        elapsed_time = datetime.now() - metrics.started_at
        estimated_total_time = elapsed_time / (progress / 100)
        
        return metrics.started_at + estimated_total_time
    
    def _cleanup_job(self, job_id: str):
        """Clean up job tracking data"""
        
        self.active_jobs.pop(job_id, None)
        self.job_configs.pop(job_id, None)
        self.job_metrics.pop(job_id, None)

class ProcessingPhaseExecutor:
    """Enterprise processing phase executor with error handling and monitoring"""
    
    def __init__(self, pipeline_service: ProcessingPipelineService):
        self.pipeline_service = pipeline_service
        self.logger = structlog.get_logger("services.phase_executor")
    
    async def execute_phase(self, job_id: str, phase: ProcessingPhase, 
                          book_id: str, db: Session) -> Dict[str, Any]:
        """Execute specific processing phase with comprehensive monitoring"""
        
        start_time = time.time()
        
        try:
            self.logger.info(
                "Starting phase execution",
                job_id=job_id,
                phase=phase.value,
                book_id=book_id
            )
            
            # Update job metrics
            if job_id in self.pipeline_service.job_metrics:
                metrics = self.pipeline_service.job_metrics[job_id]
                metrics.current_phase = phase.value
            
            # Execute phase based on type
            phase_result = await self._execute_phase_logic(phase, book_id, job_id)
            
            # Calculate execution time
            execution_time = time.time() - start_time
            
            # Update metrics on success
            if job_id in self.pipeline_service.job_metrics:
                metrics = self.pipeline_service.job_metrics[job_id]
                metrics.phases_completed.append(phase.value)
                metrics.quality_scores[phase.value] = phase_result.get("quality_score", 0.0)
                metrics.processing_time_seconds += int(execution_time)
            
            # Create audit log
            audit_crud = AuditCRUD(db)
            audit_crud.create({
                "book_id": book_id,
                "phase_name": phase.value,
                "quality_score": phase_result.get("quality_score", 0.0),
                "processing_time_seconds": int(execution_time),
                "metrics_detail": phase_result.get("metrics", {}),
                "system_version": "2.2.0"
            })
            
            self.logger.info(
                "Phase execution completed",
                job_id=job_id,
                phase=phase.value,
                book_id=book_id,
                execution_time_seconds=execution_time,
                quality_score=phase_result.get("quality_score", 0.0)
            )
            
            return {
                "success": True,
                "phase": phase.value,
                "execution_time_seconds": execution_time,
                **phase_result
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            self.logger.error(
                "Phase execution failed",
                job_id=job_id,
                phase=phase.value,
                book_id=book_id,
                error=str(e),
                execution_time_seconds=execution_time
            )
            
            # Update error metrics
            if job_id in self.pipeline_service.job_metrics:
                metrics = self.pipeline_service.job_metrics[job_id]
                metrics.error_count += 1
            
            # Create error audit log
            audit_crud = AuditCRUD(db)
            audit_crud.create({
                "book_id": book_id,
                "phase_name": phase.value,
                "quality_score": 0.0,
                "processing_time_seconds": int(execution_time),
                "alerts_count": 1,
                "critical_alerts_count": 1,
                "alerts_detail": [{
                    "type": "phase_execution_error",
                    "severity": "critical",
                    "message": str(e),
                    "timestamp": datetime.now().isoformat()
                }],
                "system_version": "2.2.0"
            })
            
            return {
                "success": False,
                "phase": phase.value,
                "error": str(e),
                "execution_time_seconds": execution_time
            }
    
    async def _execute_phase_logic(self, phase: ProcessingPhase, book_id: str, 
                                 job_id: str) -> Dict[str, Any]:
        """Execute the actual logic for each phase"""
        
        # TODO: Implement actual integration with core processing components
        # This is a comprehensive stub that simulates the processing pipeline
        
        if phase == ProcessingPhase.PDF_CLEANUP:
            return await self._execute_pdf_cleanup(book_id, job_id)
        elif phase == ProcessingPhase.CONVERSION:
            return await self._execute_conversion(book_id, job_id)
        elif phase == ProcessingPhase.TRANSLATION:
            return await self._execute_translation(book_id, job_id)
        elif phase == ProcessingPhase.VALIDATION:
            return await self._execute_validation(book_id, job_id)
        elif phase == ProcessingPhase.REVIEW_GENERATION:
            return await self._execute_review_generation(book_id, job_id)
        else:
            raise ValueError(f"Unknown phase: {phase}")
    
    async def _execute_pdf_cleanup(self, book_id: str, job_id: str) -> Dict[str, Any]:
        """Execute PDF cleanup phase (STUB IMPLEMENTATION)"""
        
        self.logger.info(
            "Executing PDF cleanup phase",
            book_id=book_id,
            job_id=job_id,
            implementation_status="STUB - TO BE IMPLEMENTED"
        )
        
        # TODO: Implement actual PDF cleanup integration
        # from ....core.advanced_pdf_cleaner import AdvancedPDFCleaner
        # cleaner = AdvancedPDFCleaner()
        # result = await cleaner.clean_pdf(pdf_path)
        
        # Simulate processing delay
        await asyncio.sleep(2)
        
        # Simulate results
        return {
            "quality_score": 0.95,
            "artifacts_removed": 42,
            "pages_processed": 156,
            "metrics": {
                "headers_removed": 24,
                "footers_removed": 18,
                "watermarks_removed": 0,
                "processing_method": "heuristic_analysis"
            }
        }
    
    async def _execute_conversion(self, book_id: str, job_id: str) -> Dict[str, Any]:
        """Execute HTML to DOCX conversion phase (STUB IMPLEMENTATION)"""
        
        self.logger.info(
            "Executing conversion phase",
            book_id=book_id,
            job_id=job_id,
            implementation_status="STUB - TO BE IMPLEMENTED"
        )
        
        # TODO: Implement actual conversion integration
        # from ....core.html_to_docx_converter import HTMLtoDocxConverter
        # converter = HTMLtoDocxConverter()
        # result = await converter.convert_with_validation(html_content)
        
        # Simulate processing delay
        await asyncio.sleep(3)
        
        # Simulate results
        return {
            "quality_score": 0.92,
            "formats_preserved": 247,
            "footnotes_reconnected": 18,
            "metrics": {
                "bold_preserved": 156,
                "italic_preserved": 89,
                "footnote_links_functional": 18,
                "conversion_method": "pandoc_with_fallback"
            }
        }
    
    async def _execute_translation(self, book_id: str, job_id: str) -> Dict[str, Any]:
        """Execute hybrid translation phase (STUB IMPLEMENTATION)"""
        
        self.logger.info(
            "Executing translation phase",
            book_id=book_id,
            job_id=job_id,
            implementation_status="STUB - TO BE IMPLEMENTED"
        )
        
        # TODO: Implement actual translation integration
        # - DeepL API integration for base translation
        # - Claude API integration for terminology refinement
        # - Semantic validation during translation
        
        # Simulate longer processing delay for translation
        await asyncio.sleep(8)
        
        # Simulate results
        return {
            "quality_score": 0.89,
            "words_translated": 23456,
            "terminology_suggestions": 15,
            "metrics": {
                "deepl_api_calls": 234,
                "claude_api_calls": 45,
                "semantic_score": 0.87,
                "terminology_consistency": 0.91,
                "translation_method": "deepl_with_claude_refinement"
            }
        }
    
    async def _execute_validation(self, book_id: str, job_id: str) -> Dict[str, Any]:
        """Execute semantic validation phase (STUB IMPLEMENTATION)"""
        
        self.logger.info(
            "Executing validation phase",
            book_id=book_id,
            job_id=job_id,
            implementation_status="STUB - TO BE IMPLEMENTED"
        )
        
        # TODO: Implement actual validation integration
        # from ....core.semantic_validator import SemanticIntegrityValidator
        # validator = SemanticIntegrityValidator()
        # result = await validator.validate_semantic_integrity(original, translated)
        
        # Simulate processing delay
        await asyncio.sleep(2)
        
        # Simulate results
        return {
            "quality_score": 0.88,
            "integrity_score": 0.87,
            "sections_validated": 145,
            "metrics": {
                "semantic_similarity": 0.87,
                "content_preservation": 0.94,
                "format_preservation": 0.96,
                "validation_method": "embedding_cosine_similarity"
            }
        }
    
    async def _execute_review_generation(self, book_id: str, job_id: str) -> Dict[str, Any]:
        """Execute editorial review generation phase (STUB IMPLEMENTATION)"""
        
        self.logger.info(
            "Executing review generation phase",
            book_id=book_id,
            job_id=job_id,
            implementation_status="STUB - TO BE IMPLEMENTED"
        )
        
        # TODO: Implement actual review generation integration
        # - Generate editorial review sheets
        # - Create terminology suggestions
        # - Generate audit reports
        
        # Simulate processing delay
        await asyncio.sleep(1)
        
        # Simulate results
        return {
            "quality_score": 1.0,
            "review_items_generated": 23,
            "formats_generated": ["csv", "excel", "html", "json"],
            "metrics": {
                "critical_items": 3,
                "warning_items": 12,
                "info_items": 8,
                "generation_method": "automated_analysis"
            }
        }

class ProcessingQueueManager:
    """Enterprise processing queue manager with priority handling"""
    
    def __init__(self):
        self.logger = structlog.get_logger("services.processing_queue")
        self.queue_storage: Dict[str, List[Dict[str, Any]]] = {
            priority.value: [] for priority in ProcessingPriority
        }
        self.max_concurrent_jobs = 3
    
    def add_to_queue(self, book_id: str, config: ProcessingJobConfig, 
                    user_id: str) -> Dict[str, Any]:
        """Add processing job to priority queue"""
        
        queue_item = {
            "book_id": book_id,
            "config": config,
            "user_id": user_id,
            "queued_at": datetime.now(),
            "priority": config.priority,
            "estimated_duration": self._estimate_processing_duration(config)
        }
        
        # Add to appropriate priority queue
        self.queue_storage[config.priority.value].append(queue_item)
        
        self.logger.info(
            "Added job to processing queue",
            book_id=book_id,
            priority=config.priority.value,
            queue_position=len(self.queue_storage[config.priority.value])
        )
        
        return queue_item
    
    def get_next_job(self) -> Optional[Dict[str, Any]]:
        """Get next job from queue based on priority"""
        
        # Check priority queues in order: URGENT -> HIGH -> NORMAL -> LOW
        for priority in [ProcessingPriority.URGENT, ProcessingPriority.HIGH, 
                        ProcessingPriority.NORMAL, ProcessingPriority.LOW]:
            
            queue = self.queue_storage[priority.value]
            if queue:
                job = queue.pop(0)  # FIFO within same priority
                
                self.logger.info(
                    "Retrieved job from queue",
                    book_id=job["book_id"],
                    priority=priority.value,
                    wait_time_minutes=(datetime.now() - job["queued_at"]).total_seconds() / 60
                )
                
                return job
        
        return None
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get comprehensive queue status"""
        
        total_queued = sum(len(queue) for queue in self.queue_storage.values())
        
        queue_status = {
            "total_queued": total_queued,
            "max_concurrent": self.max_concurrent_jobs,
            "queues_by_priority": {}
        }
        
        for priority, queue in self.queue_storage.items():
            if queue:
                oldest_job = min(queue, key=lambda x: x["queued_at"])
                estimated_wait = self._calculate_estimated_wait(priority, len(queue))
                
                queue_status["queues_by_priority"][priority] = {
                    "count": len(queue),
                    "oldest_queued_at": oldest_job["queued_at"],
                    "estimated_wait_minutes": estimated_wait
                }
        
        return queue_status
    
    def remove_from_queue(self, book_id: str) -> bool:
        """Remove job from queue by book_id"""
        
        for priority, queue in self.queue_storage.items():
            for i, job in enumerate(queue):
                if job["book_id"] == book_id:
                    removed_job = queue.pop(i)
                    
                    self.logger.info(
                        "Removed job from queue",
                        book_id=book_id,
                        priority=priority,
                        wait_time_minutes=(datetime.now() - removed_job["queued_at"]).total_seconds() / 60
                    )
                    
                    return True
        
        return False
    
    def _estimate_processing_duration(self, config: ProcessingJobConfig) -> int:
        """Estimate processing duration in minutes based on configuration"""
        
        base_duration = 60  # Base 60 minutes
        
        # Adjust based on configuration
        if config.enable_claude_refinement:
            base_duration += 15
        
        if config.generate_review:
            base_duration += 5
        
        # Adjust for skipped phases
        phase_durations = {
            ProcessingPhase.PDF_CLEANUP.value: 5,
            ProcessingPhase.CONVERSION.value: 8,
            ProcessingPhase.TRANSLATION.value: 45,
            ProcessingPhase.VALIDATION.value: 10,
            ProcessingPhase.REVIEW_GENERATION.value: 3
        }
        
        for skipped_phase in config.skip_phases:
            if skipped_phase in phase_durations:
                base_duration -= phase_durations[skipped_phase]
        
        return max(base_duration, 10)  # Minimum 10 minutes
    
    def _calculate_estimated_wait(self, priority: str, position: int) -> int:
        """Calculate estimated wait time in minutes"""
        
        # Simple estimation based on position and average processing time
        avg_processing_time = 75  # Average 75 minutes per job
        concurrent_factor = self.max_concurrent_jobs
        
        estimated_wait = (position * avg_processing_time) / concurrent_factor
        
        return int(estimated_wait)

# ==========================================
# SERVICE INSTANCES (DEPENDENCY INJECTION)
# ==========================================

pipeline_service = ProcessingPipelineService()
phase_executor = ProcessingPhaseExecutor(pipeline_service)
queue_manager = ProcessingQueueManager()

# ==========================================
# PROCESSING CONTROL ENDPOINTS
# ==========================================

@router.post("/jobs/start",
    response_model=SuccessResponse[ProcessingJobResponse],
    summary="Start processing job",
    description="Start new processing job for a book with custom configuration"
)
async def start_processing_job(
    processing_job: ProcessingJobCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user),
    _rate_limiter = Depends(get_rate_limiter("processing"))
):
    """
    Start new processing job with comprehensive configuration
    
    Enterprise features:
    - Custom processing configuration
    - Priority queue management
    - Background job execution
    - Comprehensive validation
    """
    
    try:
        logger.info(
            "Starting processing job request",
            book_id=processing_job.book_id,
            user_id=getattr(current_user, 'id', 'api_user'),
            config=processing_job.config.dict() if processing_job.config else {}
        )
        
        # Validate book exists and is processable
        book_crud = BookCRUD(db)
        book = book_crud.get_by_id(processing_job.book_id)
        
        if not book:
            return create_error_response(
                error_code=ErrorCode.NOT_FOUND,
                message=f"Book {processing_job.book_id} not found",
                status_code=404
            )
        
        # MEJORA 2: Usar función centralizada para validación de estado
        if not is_processable_status(book.status):
            return create_error_response(
                error_code=ErrorCode.CONFLICT,
                message=f"Book {processing_job.book_id} not in processable state: {book.status}",
                status_code=409,
                details={"current_status": book.status, "allowed_statuses": [BookStatus.QUEUED.value, BookStatus.FAILED.value]}
            )
        
        # Create processing configuration (will validate skip_phases via __post_init__)
        try:
            config = processing_job.config or ProcessingJobConfig(
                source_lang=book.source_lang,
                target_lang=book.target_lang
            )
        except ValueError as ve:
            # MEJORA 3: Capturar errores de validación de fases
            return create_error_response(
                error_code=ErrorCode.VALIDATION_ERROR,
                message=f"Invalid processing configuration: {str(ve)}",
                status_code=422,
                details={"validation_error": str(ve)}
            )
        
        # Add to processing queue
        queue_item = queue_manager.add_to_queue(
            processing_job.book_id,
            config,
            getattr(current_user, 'id', 'api_user')
        )
        
        # Schedule background processing
        background_tasks.add_task(
            process_job_from_queue,
            processing_job.book_id,
            db
        )
        
        response_data = ProcessingJobResponse(
            book_id=processing_job.book_id,
            status=ProcessingJobStatus.QUEUED,
            config=config,
            queued_at=queue_item["queued_at"],
            estimated_duration_minutes=queue_item["estimated_duration"],
            queue_position=len(queue_manager.queue_storage[config.priority.value])
        )
        
        logger.info(
            "Processing job queued successfully",
            book_id=processing_job.book_id,
            priority=config.priority.value
        )
        
        return create_success_response(
            data=response_data,
            message="Processing job queued successfully",
            status_code=201
        )
        
    except Exception as e:
        logger.error(
            "Error starting processing job",
            book_id=processing_job.book_id if hasattr(processing_job, 'book_id') else 'unknown',
            error=str(e)
        )
        
        return create_error_response(
            error_code=ErrorCode.INTERNAL_ERROR,
            message="Failed to start processing job",
            status_code=500,
            details={"error": str(e)}
        )

@router.post("/jobs/{job_id}/pause",
    response_model=SuccessResponse[Dict[str, str]],
    summary="Pause processing job",
    description="Pause running processing job gracefully"
)
async def pause_processing_job(
    job_id: str,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """
    Pause running processing job
    
    Enterprise features:
    - Graceful pause handling
    - State preservation
    - Resume capability
    """
    
    try:
        user_id = getattr(current_user, 'id', 'api_user')
        
        success = await pipeline_service.pause_processing_job(job_id, user_id)
        
        if not success:
            return create_error_response(
                error_code=ErrorCode.NOT_FOUND,
                message=f"Failed to pause job {job_id}",
                status_code=404
            )
        
        return create_success_response(
            data={"job_id": job_id, "status": "paused", "paused_by": user_id},
            message="Processing job paused successfully"
        )
        
    except Exception as e:
        logger.error(
            "Error pausing processing job",
            job_id=job_id,
            error=str(e)
        )
        
        return create_error_response(
            error_code=ErrorCode.INTERNAL_ERROR,
            message="Failed to pause processing job",
            status_code=500,
            details={"job_id": job_id, "error": str(e)}
        )

@router.post("/jobs/{job_id}/resume",
    response_model=SuccessResponse[Dict[str, str]],
    summary="Resume processing job",
    description="Resume paused processing job"
)
async def resume_processing_job(
    job_id: str,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """
    Resume paused processing job
    
    Enterprise features:
    - State restoration
    - Progress continuation
    - Error recovery
    """
    
    try:
        user_id = getattr(current_user, 'id', 'api_user')
        
        success = await pipeline_service.resume_processing_job(job_id, user_id)
        
        if not success:
            return create_error_response(
                error_code=ErrorCode.NOT_FOUND,
                message=f"Failed to resume job {job_id}",
                status_code=404
            )
        
        return create_success_response(
            data={"job_id": job_id, "status": "running", "resumed_by": user_id},
            message="Processing job resumed successfully"
        )
        
    except Exception as e:
        logger.error(
            "Error resuming processing job",
            job_id=job_id,
            error=str(e)
        )
        
        return create_error_response(
            error_code=ErrorCode.INTERNAL_ERROR,
            message="Failed to resume processing job",
            status_code=500,
            details={"job_id": job_id, "error": str(e)}
        )

@router.post("/jobs/{job_id}/cancel",
    response_model=SuccessResponse[Dict[str, str]],
    summary="Cancel processing job",
    description="Cancel processing job with cleanup"
)
async def cancel_processing_job(
    job_id: str,
    reason: str = Query("", description="Cancellation reason"),
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """
    Cancel processing job with comprehensive cleanup
    
    Enterprise features:
    - Graceful cancellation
    - Resource cleanup
    - Audit trail
    """
    
    try:
        user_id = getattr(current_user, 'id', 'api_user')
        
        success = await pipeline_service.cancel_processing_job(job_id, user_id, reason)
        
        if not success:
            return create_error_response(
                error_code=ErrorCode.NOT_FOUND,
                message=f"Failed to cancel job {job_id}",
                status_code=404
            )
        
        return create_success_response(
            data={
                "job_id": job_id, 
                "status": "cancelled", 
                "cancelled_by": user_id,
                "reason": reason
            },
            message="Processing job cancelled successfully"
        )
        
    except Exception as e:
        logger.error(
            "Error cancelling processing job",
            job_id=job_id,
            error=str(e)
        )
        
        return create_error_response(
            error_code=ErrorCode.INTERNAL_ERROR,
            message="Failed to cancel processing job",
            status_code=500,
            details={"job_id": job_id, "error": str(e)}
        )

# ==========================================
# MONITORING AND STATUS ENDPOINTS
# ==========================================

@router.get("/jobs/{job_id}/status",
    response_model=SuccessResponse[ProcessingStatusResponse],
    summary="Get job status",
    description="Get comprehensive processing job status and metrics"
)
async def get_processing_job_status(
    job_id: str,
    include_metrics: bool = Query(True, description="Include detailed metrics"),
    include_logs: bool = Query(False, description="Include recent log entries"),
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """
    Get comprehensive processing job status
    
    Enterprise features:
    - Real-time status updates
    - Detailed metrics
    - Progress tracking
    - Error information
    """
    
    try:
        job_status = pipeline_service.get_job_status(job_id)
        
        if not job_status:
            return create_error_response(
                error_code=ErrorCode.NOT_FOUND,
                message=f"Processing job {job_id} not found",
                status_code=404
            )
        
        # Build comprehensive response
        response_data = ProcessingStatusResponse(
            job_id=job_id,
            book_id=job_status["book_id"],
            status=ProcessingJobStatus(job_status["status"]),
            current_phase=job_status["current_phase"],
            progress_percentage=job_status["progress_percentage"],
            started_at=job_status.get("started_at"),
            estimated_completion=job_status.get("estimated_completion")
        )
        
        # Add optional metrics
        if include_metrics and "metrics" in job_status:
            response_data.metrics = job_status["metrics"]
        
        # Add optional logs
        if include_logs:
            # Get recent audit logs for this job
            audit_crud = AuditCRUD(db)
            recent_logs = audit_crud.get_recent_by_book_id(job_status["book_id"], limit=10)
            response_data.recent_logs = [
                {
                    "phase": log.phase_name,
                    "timestamp": log.created_at,
                    "quality_score": log.quality_score,
                    "alerts_count": log.alerts_count
                }
                for log in recent_logs
            ]
        
        return create_success_response(
            data=response_data,
            message="Job status retrieved successfully"
        )
        
    except Exception as e:
        logger.error(
            "Error getting job status",
            job_id=job_id,
            error=str(e)
        )
        
        return create_error_response(
            error_code=ErrorCode.INTERNAL_ERROR,
            message="Failed to retrieve job status",
            status_code=500,
            details={"job_id": job_id, "error": str(e)}
        )

@router.get("/jobs/active",
    response_model=SuccessResponse[List[ProcessingStatusResponse]],
    summary="Get active jobs",
    description="Get all active processing jobs with their status"
)
async def get_active_processing_jobs(
    include_metrics: bool = Query(False, description="Include detailed metrics"),
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """
    Get all active processing jobs
    
    Enterprise features:
    - System-wide job monitoring
    - Resource utilization tracking
    - Performance overview
    """
    
    try:
        active_jobs = pipeline_service.get_active_jobs()
        
        response_data = []
        for job in active_jobs:
            job_response = ProcessingStatusResponse(
                job_id=job["job_id"],
                book_id=job["book_id"],
                status=ProcessingJobStatus(job["status"]),
                current_phase=job["current_phase"],
                progress_percentage=job["progress_percentage"],
                started_at=job.get("started_at"),
                estimated_completion=job.get("estimated_completion")
            )
            
            if include_metrics and "metrics" in job:
                job_response.metrics = job["metrics"]
            
            response_data.append(job_response)
        
        return create_success_response(
            data=response_data,
            message=f"Retrieved {len(response_data)} active jobs"
        )
        
    except Exception as e:
        logger.error(
            "Error getting active jobs",
            error=str(e)
        )
        
        return create_error_response(
            error_code=ErrorCode.INTERNAL_ERROR,
            message="Failed to retrieve active jobs",
            status_code=500,
            details={"error": str(e)}
        )

@router.get("/queue/status",
    response_model=SuccessResponse[ProcessingQueueResponse],
    summary="Get processing queue status",
    description="Get comprehensive processing queue status and statistics"
)
async def get_processing_queue_status(
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """
    Get comprehensive processing queue status
    
    Enterprise features:
    - Queue length by priority
    - Estimated wait times
    - System capacity monitoring
    """
    
    try:
        queue_status = queue_manager.get_queue_status()
        active_jobs = pipeline_service.get_active_jobs()
        
        response_data = ProcessingQueueResponse(
            total_queued=queue_status["total_queued"],
            active_jobs_count=len(active_jobs),
            max_concurrent_jobs=queue_status["max_concurrent"],
            queues_by_priority=queue_status["queues_by_priority"],
            system_capacity_percentage=(len(active_jobs) / queue_status["max_concurrent"]) * 100
        )
        
        return create_success_response(
            data=response_data,
            message="Queue status retrieved successfully"
        )
        
    except Exception as e:
        logger.error(
            "Error getting queue status",
            error=str(e)
        )
        
        return create_error_response(
            error_code=ErrorCode.INTERNAL_ERROR,
            message="Failed to retrieve queue status",
            status_code=500,
            details={"error": str(e)}
        )

@router.get("/phases/{phase_name}/status",
    response_model=SuccessResponse[PipelinePhaseResponse],
    summary="Get phase status",
    description="Get detailed status for specific processing phase"
)
async def get_processing_phase_status(
    phase_name: str,
    book_id: Optional[str] = Query(None, description="Filter by specific book"),
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """
    Get detailed status for specific processing phase
    
    Enterprise features:
    - Phase-specific monitoring
    - Performance analytics per phase
    - Error tracking by phase
    """
    
    try:
        # Validate phase name
        try:
            phase_enum = ProcessingPhase(phase_name)
        except ValueError:
            return create_error_response(
                error_code=ErrorCode.VALIDATION_ERROR,
                message=f"Invalid phase name: {phase_name}",
                status_code=422,
                details={"valid_phases": [p.value for p in ProcessingPhase]}
            )
        
        # Get phase information
        phase_info = pipeline_service.pipeline_phases.get(phase_enum, {})
        
        # Get recent audit logs for this phase
        audit_crud = AuditCRUD(db)
        query_filters = {"phase_name": phase_name}
        if book_id:
            query_filters["book_id"] = book_id
        
        recent_executions = audit_crud.get_recent_by_filters(query_filters, limit=20)
        
        # MEJORA 4: Cálculo seguro de estadísticas de fase evitando división por cero
        if recent_executions:
            durations = [log.processing_time_seconds or 0 for log in recent_executions]
            quality_scores = [log.quality_score for log in recent_executions]
            error_flags = [1 for log in recent_executions if log.critical_alerts_count > 0]
            
            avg_duration = calculate_safe_average(durations)
            avg_quality = calculate_safe_average(quality_scores)
            error_rate = calculate_safe_average(error_flags)
        else:
            avg_duration = 0.0
            avg_quality = 0.0
            error_rate = 0.0
        
        response_data = PipelinePhaseResponse(
            phase_name=phase_name,
            phase_description=phase_info.get("description", ""),
            estimated_duration_minutes=phase_info.get("estimated_duration_minutes", 0),
            recent_executions_count=len(recent_executions),
            average_duration_seconds=avg_duration,
            average_quality_score=avg_quality,
            error_rate_percentage=error_rate * 100,
            currently_running_count=sum(1 for job in pipeline_service.get_active_jobs() 
                                      if job.get("current_phase") == phase_name)
        )
        
        return create_success_response(
            data=response_data,
            message=f"Phase {phase_name} status retrieved successfully"
        )
        
    except Exception as e:
        logger.error(
            "Error getting phase status",
            phase_name=phase_name,
            error=str(e)
        )
        
        return create_error_response(
            error_code=ErrorCode.INTERNAL_ERROR,
            message="Failed to retrieve phase status",
            status_code=500,
            details={"phase_name": phase_name, "error": str(e)}
        )

# ==========================================
# CONFIGURATION ENDPOINTS
# ==========================================

@router.get("/config/default",
    response_model=SuccessResponse[ProcessingJobConfig],
    summary="Get default processing configuration",
    description="Get default processing configuration template"
)
async def get_default_processing_config(
    source_lang: str = Query(..., regex="^(de|en|fr|it|nl)$", description="Source language"),
    target_lang: str = Query("es", regex="^(es|en)$", description="Target language"),
    current_user = Depends(get_current_user)
):
    """
    Get default processing configuration
    
    Enterprise features:
    - Language-specific defaults
    - User preference integration
    - Configuration templates
    """
    
    try:
        # Create default configuration based on language pair
        default_config = ProcessingJobConfig(
            source_lang=source_lang,
            target_lang=target_lang,
            quality_threshold=0.85,
            enable_claude_refinement=True,
            preserve_formatting=True,
            generate_review=True,
            max_retries=3,
            timeout_minutes=120,
            priority=ProcessingPriority.NORMAL
        )
        
        # Adjust defaults based on language pair complexity
        if source_lang in ["de", "nl"]:  # More complex languages
            default_config.timeout_minutes = 150
            default_config.quality_threshold = 0.82
        
        return create_success_response(
            data=default_config,
            message="Default configuration retrieved successfully"
        )
        
    except Exception as e:
        logger.error(
            "Error getting default config",
            source_lang=source_lang,
            target_lang=target_lang,
            error=str(e)
        )
        
        return create_error_response(
            error_code=ErrorCode.INTERNAL_ERROR,
            message="Failed to retrieve default configuration",
            status_code=500,
            details={"error": str(e)}
        )

@router.post("/config/validate",
    response_model=SuccessResponse[Dict[str, Any]],
    summary="Validate processing configuration",
    description="Validate processing configuration before starting job"
)
async def validate_processing_config(
    config: ProcessingJobConfig,
    current_user = Depends(get_current_user)
):
    """
    Validate processing configuration
    
    Enterprise features:
    - Configuration validation
    - Resource estimation
    - Compatibility checking
    """
    
    try:
        validation_result = {
            "is_valid": True,
            "warnings": [],
            "estimated_duration_minutes": 0,
            "estimated_cost": 0.0,
            "compatibility_check": {}
        }
        
        # Validate configuration values
        if config.quality_threshold < 0.5 or config.quality_threshold > 1.0:
            validation_result["warnings"].append("Quality threshold should be between 0.5 and 1.0")
        
        if config.timeout_minutes < 30:
            validation_result["warnings"].append("Timeout less than 30 minutes may cause premature failures")
        
        if config.max_retries > 5:
            validation_result["warnings"].append("High retry count may delay error detection")
        
        # Estimate duration
        base_duration = 60
        if config.enable_claude_refinement:
            base_duration += 20
        if config.generate_review:
            base_duration += 5
        
        validation_result["estimated_duration_minutes"] = base_duration
        
        # Estimate cost (rough approximation)
        estimated_cost = 2.50  # Base cost
        if config.enable_claude_refinement:
            estimated_cost += 1.00
        
        validation_result["estimated_cost"] = estimated_cost
        
        # Compatibility check
        validation_result["compatibility_check"] = {
            "source_language_supported": config.source_lang in ["de", "en", "fr", "it", "nl"],
            "target_language_supported": config.target_lang in ["es", "en"],
            "claude_available": True,  # Would check actual API availability
            "deepl_available": True    # Would check actual API availability
        }
        
        return create_success_response(
            data=validation_result,
            message="Configuration validated successfully"
        )
        
    except Exception as e:
        logger.error(
            "Error validating config",
            config=config.dict(),
            error=str(e)
        )
        
        return create_error_response(
            error_code=ErrorCode.INTERNAL_ERROR,
            message="Failed to validate configuration",
            status_code=500,
            details={"error": str(e)}
        )

# ==========================================
# STREAMING ENDPOINTS
# ==========================================

@router.get("/jobs/{job_id}/stream",
    response_class=StreamingResponse,
    summary="Stream job progress",
    description="Stream real-time job progress updates via Server-Sent Events"
)
async def stream_job_progress(
    job_id: str,
    db: Session = Depends(get_db),
    # MEJORA 5: Autenticación opcional para SSE - permite API key para dashboards
    current_user = Depends(get_optional_current_user)
):
    """
    Stream real-time job progress updates
    
    Enterprise features:
    - Server-Sent Events streaming
    - Real-time progress updates
    - Connection management
    - Optional authentication (supports API key for dashboards)
    """
    
    async def generate_progress_stream():
        """Generate Server-Sent Events stream for job progress"""
        
        try:
            # Log quien está accediendo al stream
            user_info = getattr(current_user, 'id', 'anonymous') if current_user else 'anonymous'
            logger.info(
                "Starting progress stream",
                job_id=job_id,
                user=user_info
            )
            
            while True:
                # Get current job status
                job_status = pipeline_service.get_job_status(job_id)
                
                if not job_status:
                    yield f"data: {json.dumps({'error': 'Job not found'})}\n\n"
                    break
                
                # Send status update
                yield f"data: {json.dumps(job_status)}\n\n"
                
                # Check if job is finished
                if job_status["status"] in [ProcessingJobStatus.COMPLETED.value, 
                                          ProcessingJobStatus.FAILED.value,
                                          ProcessingJobStatus.CANCELLED.value]:
                    yield f"data: {json.dumps({'message': 'Job completed', 'final_status': job_status['status']})}\n\n"
                    break
                
                # Wait before next update
                await asyncio.sleep(2)
                
        except Exception as e:
            logger.error(
                "Error in progress stream",
                job_id=job_id,
                error=str(e)
            )
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(
        generate_progress_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Job-ID": job_id
        }
    )

# ==========================================
# HELPER FUNCTIONS
# ==========================================

async def process_job_from_queue(book_id: str, db: Session):
    """Process job from queue (background task)"""
    
    try:
        logger.info("Processing job from queue", book_id=book_id)
        
        # Get next job from queue
        next_job = queue_manager.get_next_job()
        
        if not next_job or next_job["book_id"] != book_id:
            logger.warning("Job not found in queue", book_id=book_id)
            return
        
        # Start processing job
        job_id = await pipeline_service.start_processing_job(
            book_id=next_job["book_id"],
            config=next_job["config"],
            user_id=next_job["user_id"],
            db=db
        )
        
        # Execute processing phases
        phases = [
            ProcessingPhase.PDF_CLEANUP,
            ProcessingPhase.CONVERSION,
            ProcessingPhase.TRANSLATION,
            ProcessingPhase.VALIDATION,
            ProcessingPhase.REVIEW_GENERATION
        ]
        
        # Filter out skipped phases
        config = next_job["config"]
        if config.skip_phases:
            phases = [phase for phase in phases if phase.value not in config.skip_phases]
        
        for phase in phases:
            # Check for cancellation
            if job_id in pipeline_service.active_jobs:
                job = pipeline_service.active_jobs[job_id]
                if job.get("cancellation_requested"):
                    logger.info("Job cancellation requested", job_id=job_id)
                    break
            
            # Execute phase
            phase_result = await phase_executor.execute_phase(job_id, phase, book_id, db)
            
            if not phase_result["success"]:
                logger.error(
                    "Phase execution failed",
                    job_id=job_id,
                    phase=phase.value,
                    error=phase_result.get("error")
                )
                
                # MEJORA 1: Usar método de finalización explícita del job
                await pipeline_service.complete_processing_job(
                    job_id, 
                    success=False,
                    final_metrics={"failed_phase": phase.value, "error": phase_result.get("error")}
                )
                
                # Update book status
                book_crud = BookCRUD(db)
                book_crud.update(book_id, {
                    "status": BookStatus.FAILED.value,
                    "error_count": func.coalesce(BookProcessingHistory.error_count, 0) + 1
                })
                
                return
        
        # MEJORA 1: Usar método de finalización explícita del job con métricas
        await pipeline_service.complete_processing_job(
            job_id, 
            success=True,
            final_metrics={"phases_executed": len(phases), "total_phases_available": len(phases)}
        )
        
        # Update book status
        book_crud = BookCRUD(db)
        book_crud.update(book_id, {
            "status": BookStatus.COMPLETED.value,
            "completed_at": datetime.now(),
            "progress_percentage": 100.0
        })
        
        logger.info("Job processing completed successfully", job_id=job_id, book_id=book_id)
        
    except Exception as e:
        logger.error(
            "Error processing job from queue",
            book_id=book_id,
            error=str(e)
        )

# Export all endpoint functions for testing
__all__ = [
    "router",
    "start_processing_job",
    "pause_processing_job",
    "resume_processing_job", 
    "cancel_processing_job",
    "get_processing_job_status",
    "get_active_processing_jobs",
    "get_processing_queue_status",
    "get_processing_phase_status",
    "get_default_processing_config",
    "validate_processing_config",
    "stream_job_progress",
    # Utility functions exported for testing
    "is_processable_status",
    "calculate_safe_average"
]
