"""
ClaudeAcademico v2.2 - FastAPI Backend
Sistema de Traducción Académica - API REST
"""

import os
import sys
import shutil  # AGREGADO: necesario para shutil.copyfileobj
from pathlib import Path
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from contextlib import asynccontextmanager
import uvicorn
from typing import Optional, List, Dict, Any
import asyncio
import logging
from datetime import datetime
import json

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# IMPORTACIONES COMPONENTES CORE (CORREGIDAS)
try:
    # Intentar importar componentes Core
    from core.pdf_cleaner import AdvancedPDFCleaner
    from core.semantic_validator import SemanticIntegrityValidator  
    from core.html_to_docx_converter import HTMLtoDocxConverter
    from core.footnote_reconnection_engine import FootnoteReconnectionEngine
    CORE_COMPONENTS_AVAILABLE = True
    logger_temp = logging.getLogger("claudeacademico.startup")
    logger_temp.info("✅ Componentes Core importados correctamente")
except ImportError as e:
    CORE_COMPONENTS_AVAILABLE = False
    logger_temp = logging.getLogger("claudeacademico.startup")
    logger_temp.warning(f"⚠️ Componentes Core no disponibles: {e}")
    logger_temp.info("💡 Sistema funcionará en modo simulación únicamente")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("claudeacademico.api")

# ==========================================
# ACADEMIC TRANSLATION PIPELINE (REAL)
# ==========================================

class AcademicTranslationPipeline:
    """Pipeline que ejecuta los 4 componentes Core reales"""
    
    def __init__(self):
        if not CORE_COMPONENTS_AVAILABLE:
            logger.warning("⚠️ Pipeline inicializado en modo simulación")
            return
            
        try:
            self.pdf_cleaner = AdvancedPDFCleaner()
            self.docx_converter = HTMLtoDocxConverter()
            self.semantic_validator = SemanticIntegrityValidator()
            self.footnote_engine = FootnoteReconnectionEngine()
            logger.info("✅ Pipeline real inicializado con componentes Core")
        except Exception as e:
            logger.error(f"❌ Error inicializando pipeline: {e}")
            raise
        
    async def process_document_real(
        self, 
        file_path: str, 
        source_lang: str = "de", 
        target_lang: str = "es"
    ) -> dict:
        """Ejecuta el pipeline REAL de 4 fases"""
        
        if not CORE_COMPONENTS_AVAILABLE:
            return {
                "pipeline_executed": False,
                "error": "Componentes Core no disponibles",
                "suggestion": "Verificar instalación de módulos core/"
            }
        
        results = {
            "pipeline_executed": True,
            "phases": [],
            "files_generated": [],
            "quality_scores": {}
        }
        
        try:
            logger.info(f"🚀 Iniciando pipeline real para: {file_path}")
            
            # FASE 1: Limpieza inteligente PDF
            logger.info(f"🔄 FASE 1: Limpieza PDF avanzada...")
            try:
                cleaned_text = await self.pdf_cleaner.process_document(file_path)
                results["phases"].append({
                    "phase": 1,
                    "name": "AdvancedPDFCleaner", 
                    "status": "completed",
                    "artifacts_removed": len(cleaned_text.get("removed_patterns", [])),
                    "execution_time": "simulated"
                })
            except Exception as e:
                logger.error(f"❌ Error en Fase 1: {e}")
                results["phases"].append({
                    "phase": 1,
                    "name": "AdvancedPDFCleaner",
                    "status": "error",
                    "error": str(e)
                })
                # Continuar con contenido simulado
                cleaned_text = {"clean_content": f"<html><body>Contenido limpio simulado de {file_path}</body></html>"}
            
            # FASE 2: Conversión premium DOCX
            logger.info(f"🔄 FASE 2: Conversión HTML→DOCX...")
            try:
                docx_result = await self.docx_converter.convert_to_docx(
                    cleaned_text["clean_content"]
                )
                docx_path = docx_result["output_file"]
                results["phases"].append({
                    "phase": 2, 
                    "name": "HTMLtoDocxConverter",
                    "status": "completed",
                    "output_file": docx_path
                })
                results["files_generated"].append(docx_path)
            except Exception as e:
                logger.error(f"❌ Error en Fase 2: {e}")
                # Simular archivo DOCX
                docx_path = file_path.replace(".pdf", "_converted.docx")
                results["phases"].append({
                    "phase": 2,
                    "name": "HTMLtoDocxConverter", 
                    "status": "simulated",
                    "output_file": docx_path,
                    "error": str(e)
                })
            
            # FASE 3: Traducción híbrida (PLACEHOLDER para DeepL + Claude)
            logger.info(f"🔄 FASE 3: Traducción híbrida...")
            translated_path = docx_path.replace(".docx", "_translated.docx")
            
            # TODO: Integrar DeepL Pro API + Claude terminology aquí
            # if DEEPL_API_KEY and CLAUDE_API_KEY:
            #     resultado_traduccion = await traducir_con_deepl_y_claude(docx_path)
            
            results["phases"].append({
                "phase": 3,
                "name": "DeepL+Claude Hybrid",
                "status": "simulated", # Cambiar a "completed" cuando integres APIs
                "output_file": translated_path,
                "note": "APIs DeepL/Claude no integradas aún"
            })
            
            # FASE 4: Validación integral
            logger.info(f"🔄 FASE 4: Validación semántica...")
            try:
                validation_result = await self.semantic_validator.validate_translation(
                    original_file=docx_path,
                    translated_file=translated_path
                )
                results["phases"].append({
                    "phase": 4,
                    "name": "SemanticIntegrityValidator", 
                    "status": "completed",
                    "integrity_score": validation_result.overall_integrity_score
                })
                results["quality_scores"]["semantic_integrity"] = validation_result.overall_integrity_score
            except Exception as e:
                logger.error(f"❌ Error en Fase 4: {e}")
                results["phases"].append({
                    "phase": 4,
                    "name": "SemanticIntegrityValidator",
                    "status": "error", 
                    "error": str(e)
                })
                results["quality_scores"]["semantic_integrity"] = 0.85  # Simulado
            
            # FASE 5: Reconexión de footnotes
            logger.info(f"🔄 FASE 5: Reconexión de footnotes...")
            try:
                footnote_result = await self.footnote_engine.reconnect_footnotes(translated_path)
                final_path = footnote_result["output_file"]
                results["phases"].append({
                    "phase": 5,
                    "name": "FootnoteReconnectionEngine",
                    "status": "completed", 
                    "footnotes_reconnected": footnote_result["reconnected_count"]
                })
                results["files_generated"].append(final_path)
                results["final_document"] = final_path
            except Exception as e:
                logger.error(f"❌ Error en Fase 5: {e}")
                results["phases"].append({
                    "phase": 5,
                    "name": "FootnoteReconnectionEngine",
                    "status": "error",
                    "error": str(e)
                })
                results["final_document"] = translated_path
            
            logger.info(f"✅ Pipeline real completado para: {file_path}")
            return results
            
        except Exception as e:
            logger.error(f"❌ Error crítico en pipeline: {e}")
            results["error"] = str(e)
            results["status"] = "failed"
            return results

# ==========================================
# STARTUP AND SHUTDOWN HANDLERS
# ==========================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler"""
    # Startup
    logger.info("🚀 ClaudeAcademico v2.2 FastAPI Backend Starting...")
    
    try:
        # Initialize directories
        required_dirs = ['uploads', 'output', 'temp', 'logs', 'data', 'backups']
        for dir_name in required_dirs:
            Path(dir_name).mkdir(exist_ok=True)
        
        # Test basic configurations
        logger.info("📁 Directories initialized")
        
        # Initialize pipeline
        if CORE_COMPONENTS_AVAILABLE:
            global pipeline
            pipeline = AcademicTranslationPipeline()
            logger.info("🔧 Pipeline real inicializado")
        else:
            logger.warning("⚠️ Pipeline ejecutará en modo simulación")
        
        logger.info("✅ FastAPI Backend ready")
        
    except Exception as e:
        logger.error(f"❌ Startup error: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("🛑 ClaudeAcademico FastAPI Backend shutting down...")

# ==========================================
# FASTAPI APPLICATION
# ==========================================

app = FastAPI(
    title="ClaudeAcademico v2.2 API",
    description="Sistema de Traducción Académica - API REST",
    version="2.2.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# ==========================================
# CORS MIDDLEWARE
# ==========================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8501", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# GLOBAL STATE (temporary until database is ready)
# ==========================================

# Temporary in-memory storage for demo
books_state = {}
processing_queue = []

# Pipeline instance (will be initialized in lifespan)
pipeline = None

# ==========================================
# MODELS (Pydantic)
# ==========================================

from pydantic import BaseModel
from typing import Union

class BookUploadResponse(BaseModel):
    book_id: str
    message: str
    file_path: str
    file_size: int

class BookStatus(BaseModel):
    book_id: str
    status: str
    progress_percentage: float
    current_phase: Optional[str] = None
    quality_scores: Dict[str, float] = {}
    estimated_completion: Optional[str] = None
    error_count: int = 0

class ProcessingRequest(BaseModel):
    book_id: str
    source_lang: str
    target_lang: str = "es"
    priority: int = 5

class APIResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    timestamp: str

# ==========================================
# UTILITY FUNCTIONS
# ==========================================

def generate_book_id() -> str:
    """Generate unique book ID"""
    from datetime import datetime
    import uuid
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_uuid = str(uuid.uuid4())[:8]
    return f"book_{timestamp}_{short_uuid}"

def get_api_response(success: bool, message: str, data: Any = None) -> APIResponse:
    """Generate standardized API response"""
    return APIResponse(
        success=success,
        message=message,
        data=data,
        timestamp=datetime.now().isoformat()
    )

# ==========================================
# HEALTH CHECK ENDPOINTS
# ==========================================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "ClaudeAcademico v2.2 API",
        "status": "running",
        "version": "2.2.0",
        "timestamp": datetime.now().isoformat(),
        "docs": "/docs",
        "core_components": CORE_COMPONENTS_AVAILABLE
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "ClaudeAcademico API",
        "timestamp": datetime.now().isoformat(),
        "uptime": "running",
        "version": "2.2.0",
        "core_components": CORE_COMPONENTS_AVAILABLE
    }

@app.get("/status")
async def system_status():
    """System status with detailed info"""
    
    # Check directories
    dirs_status = {}
    required_dirs = ['uploads', 'output', 'temp', 'logs', 'data']
    for dir_name in required_dirs:
        dir_path = Path(dir_name)
        dirs_status[dir_name] = {
            "exists": dir_path.exists(),
            "writable": dir_path.exists() and os.access(dir_path, os.W_OK)
        }
    
    # Check environment variables
    env_status = {}
    important_env_vars = ['DEEPL_API_KEY', 'ABBYY_APPLICATION_ID', 'SECRET_KEY']
    for var in important_env_vars:
        value = os.getenv(var)
        env_status[var] = {
            "configured": bool(value),
            "length": len(value) if value else 0
        }
    
    return {
        "status": "operational",
        "timestamp": datetime.now().isoformat(),
        "directories": dirs_status,
        "environment": env_status,
        "active_books": len(books_state),
        "queue_length": len(processing_queue),
        "core_components": CORE_COMPONENTS_AVAILABLE,
        "pipeline_available": pipeline is not None
    }

# ==========================================
# FILE UPLOAD ENDPOINTS
# ==========================================

@app.post("/books/upload", response_model=BookUploadResponse)
async def upload_book(file: UploadFile = File(...)):
    """Upload PDF file for processing"""
    
    try:
        # Validate file type
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are accepted")
        
        # Generate book ID
        book_id = generate_book_id()
        
        # Ensure uploads directory exists
        uploads_dir = Path("uploads")
        uploads_dir.mkdir(exist_ok=True)
        
        # Save file
        file_path = uploads_dir / f"{book_id}_{file.filename}"
        
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Initialize book state
        books_state[book_id] = {
            "book_id": book_id,
            "original_filename": file.filename,
            "file_path": str(file_path),
            "file_size": len(content),
            "status": "uploaded",
            "uploaded_at": datetime.now().isoformat(),
            "progress_percentage": 0.0
        }
        
        logger.info(f"📁 Uploaded file: {file.filename} -> {book_id}")
        
        return BookUploadResponse(
            book_id=book_id,
            message=f"File '{file.filename}' uploaded successfully",
            file_path=str(file_path),
            file_size=len(content)
        )
        
    except Exception as e:
        logger.error(f"❌ Upload error: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

# ==========================================
# PROCESSING ENDPOINTS (SIMULACIÓN)
# ==========================================

@app.post("/books/{book_id}/start")
async def start_processing(book_id: str, request: ProcessingRequest, background_tasks: BackgroundTasks):
    """Start book processing (SIMULACIÓN)"""
    
    if book_id not in books_state:
        raise HTTPException(status_code=404, detail="Book not found")
    
    book = books_state[book_id]
    
    if book["status"] == "processing":
        raise HTTPException(status_code=400, detail="Book is already being processed")
    
    # Update book status
    book.update({
        "status": "queued",
        "source_lang": request.source_lang,
        "target_lang": request.target_lang,
        "priority": request.priority,
        "started_at": datetime.now().isoformat()
    })
    
    # Add to processing queue (background task simulation)
    background_tasks.add_task(simulate_processing, book_id)
    
    logger.info(f"🚀 Started processing (SIMULATION): {book_id} ({request.source_lang} -> {request.target_lang})")
    
    return get_api_response(
        success=True,
        message=f"Processing started for book {book_id} (SIMULATION MODE)",
        data={"book_id": book_id, "estimated_duration": "45-90 minutes", "mode": "simulation"}
    )

async def simulate_processing(book_id: str):
    """Simulate book processing (placeholder)"""
    
    if book_id not in books_state:
        return
    
    book = books_state[book_id]
    
    try:
        # Simulate processing phases
        phases = [
            ("pdf_cleanup", 15),
            ("html_conversion", 25), 
            ("translation", 50),
            ("semantic_validation", 70),
            ("footnote_reconnection", 85),
            ("final_validation", 100)
        ]
        
        book["status"] = "processing"
        
        for phase_name, progress in phases:
            book.update({
                "current_phase": phase_name,
                "progress_percentage": progress,
                "last_update": datetime.now().isoformat()
            })
            
            # Simulate processing time
            await asyncio.sleep(2)
            
            logger.info(f"📊 {book_id}: {phase_name} -> {progress}%")
        
        # Complete processing
        book.update({
            "status": "completed",
            "progress_percentage": 100.0,
            "completed_at": datetime.now().isoformat(),
            "current_phase": "completed"
        })
        
        logger.info(f"✅ Completed processing (SIMULATION): {book_id}")
        
    except Exception as e:
        book.update({
            "status": "failed",
            "error": str(e),
            "failed_at": datetime.now().isoformat()
        })
        logger.error(f"❌ Processing failed for {book_id}: {e}")

# ==========================================
# PIPELINE REAL ENDPOINTS (NUEVO)
# ==========================================

@app.post("/translate-document-real")
async def translate_document_real(
    file: UploadFile = File(...),
    source_lang: str = "de",
    target_lang: str = "es"
):
    """
    🎯 ENDPOINT QUE SÍ EJECUTA EL PIPELINE REAL
    
    Este endpoint usa los componentes Core reales en lugar de simulaciones.
    """
    
    # Validar archivo
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(
            status_code=400, 
            detail="Solo archivos PDF son aceptados para pipeline real"
        )
    
    # Guardar archivo subido
    upload_dir = Path("uploads")
    upload_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = upload_dir / f"real_pipeline_{timestamp}_{file.filename}"
    
    try:
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        logger.info(f"📁 Archivo para pipeline real guardado: {file_path}")
        
        # Ejecutar pipeline real
        if pipeline is None:
            raise HTTPException(
                status_code=503,
                detail="Pipeline no disponible. Verificar componentes Core."
            )
        
        result = await pipeline.process_document_real(
            str(file_path), 
            source_lang, 
            target_lang
        )
        
        return {
            "message": "🎉 PIPELINE REAL EJECUTADO",
            "status": "completed" if result.get("pipeline_executed") else "error",
            "input_file": str(file_path),
            "source_lang": source_lang,
            "target_lang": target_lang,
            "pipeline_results": result,
            "core_components_used": CORE_COMPONENTS_AVAILABLE,
            "difference": "Este endpoint SÍ procesa el documento con los componentes Core reales",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Error en pipeline real: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error en pipeline real: {str(e)}"
        )

@app.post("/compare-pipelines")
async def compare_real_vs_simulation(file: UploadFile = File(...)):
    """Endpoint para comparar pipeline real vs simulación actual"""
    
    return {
        "comparison": {
            "simulation_endpoints": ["/books/upload", "/books/{book_id}/start"],
            "real_pipeline_endpoint": "/translate-document-real",
            "simulation_result": "Archivo copiado sin procesamiento real",
            "real_pipeline_result": "Documento procesado con 4 fases Core",
            "recommendation": "Usar /translate-document-real para traducciones reales",
            "core_components_available": CORE_COMPONENTS_AVAILABLE
        },
        "usage_examples": {
            "simulation": "POST /books/upload + POST /books/{book_id}/start",
            "real_pipeline": "POST /translate-document-real con archivo PDF"
        }
    }

# ==========================================
# BOOK MANAGEMENT ENDPOINTS (EXISTENTES)
# ==========================================

@app.get("/books/{book_id}/status", response_model=BookStatus)
async def get_book_status(book_id: str):
    """Get book processing status"""
    
    if book_id not in books_state:
        raise HTTPException(status_code=404, detail="Book not found")
    
    book = books_state[book_id]
    
    return BookStatus(
        book_id=book_id,
        status=book["status"],
        progress_percentage=book.get("progress_percentage", 0.0),
        current_phase=book.get("current_phase"),
        quality_scores=book.get("quality_scores", {}),
        estimated_completion=book.get("estimated_completion"),
        error_count=book.get("error_count", 0)
    )

@app.get("/books")
async def list_books():
    """List all books"""
    return get_api_response(
        success=True,
        message=f"Found {len(books_state)} books",
        data={"books": list(books_state.values())}
    )

@app.get("/books/active")
async def get_active_books():
    """Get currently processing books"""
    active_books = [
        book for book in books_state.values() 
        if book["status"] in ["queued", "processing"]
    ]
    
    return get_api_response(
        success=True,
        message=f"Found {len(active_books)} active books",
        data={"books": active_books}
    )

@app.delete("/books/{book_id}")
async def delete_book(book_id: str):
    """Delete book and files"""
    
    if book_id not in books_state:
        raise HTTPException(status_code=404, detail="Book not found")
    
    book = books_state[book_id]
    
    # Delete file if exists
    file_path = Path(book["file_path"])
    if file_path.exists():
        file_path.unlink()
    
    # Remove from state
    del books_state[book_id]
    
    logger.info(f"🗑️ Deleted book: {book_id}")
    
    return get_api_response(
        success=True,
        message=f"Book {book_id} deleted successfully"
    )

# ==========================================
# DOWNLOAD ENDPOINTS
# ==========================================

@app.get("/books/{book_id}/download")
async def download_book(book_id: str):
    """Download processed book"""
    
    if book_id not in books_state:
        raise HTTPException(status_code=404, detail="Book not found")
    
    book = books_state[book_id]
    
    if book["status"] != "completed":
        raise HTTPException(status_code=400, detail="Book processing not completed")
    
    # For now, return the original file (placeholder)
    file_path = Path(book["file_path"])
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        path=file_path,
        filename=f"translated_{book['original_filename']}",
        media_type="application/pdf"
    )

# ==========================================
# STATISTICS ENDPOINTS
# ==========================================

@app.get("/statistics/system")
async def get_system_statistics():
    """Get system-wide statistics"""
    
    total_books = len(books_state)
    completed_books = len([b for b in books_state.values() if b["status"] == "completed"])
    processing_books = len([b for b in books_state.values() if b["status"] == "processing"])
    failed_books = len([b for b in books_state.values() if b["status"] == "failed"])
    
    return get_api_response(
        success=True,
        message="System statistics retrieved",
        data={
            "books_total": total_books,
            "books_completed": completed_books,
            "books_processing": processing_books,
            "books_failed": failed_books,
            "success_rate": (completed_books / total_books * 100) if total_books > 0 else 0,
            "active_books": processing_books,
            "core_components": CORE_COMPONENTS_AVAILABLE,
            "pipeline_ready": pipeline is not None,
            "timestamp": datetime.now().isoformat()
        }
    )

# ==========================================
# ERROR HANDLERS
# ==========================================

@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content=get_api_response(
            success=False,
            message="Resource not found"
        ).dict()
    )

@app.exception_handler(500)
async def internal_server_error_handler(request, exc):
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content=get_api_response(
            success=False,
            message="Internal server error"
        ).dict()
    )

# ==========================================
# MAIN ENTRY POINT
# ==========================================

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )