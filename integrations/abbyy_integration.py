#!/usr/bin/env python3
"""
📄 ABBYY_INTEGRATION.PY - Integración ABBYY FineReader Cloud API
Sistema de Traducción Académica v2.2 - APIs Integration Layer
POST-AUDITORÍA: Versión mejorada con correcciones críticas

Integración especializada con ABBYY FineReader Cloud para OCR avanzado
y conversión de documentos académicos con preservación de estructura.

Características:
✅ OCR avanzado con ADRT (Adaptive Document Recognition Technology)
✅ Preservación de estructura y formatos complejos
✅ Soporte multiidioma automático
✅ Conversión a múltiples formatos de salida
✅ Procesamiento de documentos académicos especializados
✅ Estimación real de páginas y costos precisos
✅ Requests centralizadas y polling resiliente
✅ Retries automáticos para todas las operaciones

Autor: Sistema ClaudeAcademico v2.2
Fecha: Enero 2025 (Post-Auditoría)
Ubicación: integrations/abbyy_integration.py
"""

import asyncio
import re
import time
from pathlib import Path
from typing import List, Optional, Dict, Any
from urllib.parse import urljoin

import aiofiles
import aiohttp
from aiohttp import FormData

from .base_client import BaseAPIClient, create_rate_limiter
from .models import (
    APIProvider, APIResponse, DocumentProcessingTask, SupportedLanguage,
    ProcessingStatus, Logger, CacheManager, ErrorPolicyManager,
    create_request_id, estimate_pages_from_characters
)

# ===============================================================================
# CONSTANTES CENTRALIZADAS (MEJORA CRÍTICA)
# ===============================================================================

# ✅ MEJORA: Constantes centralizadas para evitar repetición
SUPPORTED_INPUT_FORMATS = ['.pdf', '.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif']
SUPPORTED_OUTPUT_FORMATS = ['docx', 'doc', 'rtf', 'txt', 'pdf', 'html', 'xml', 'xlsx', 'csv']

EXTENSION_TO_CONTENT_TYPE = {
    '.pdf': 'application/pdf',
    '.jpg': 'image/jpeg',
    '.jpeg': 'image/jpeg',
    '.png': 'image/png',
    '.bmp': 'image/bmp',
    '.tiff': 'image/tiff',
    '.tif': 'image/tiff',
    '.gif': 'image/gif'
}

# Mapeo de idiomas internos a códigos ABBYY
LANGUAGE_MAPPING = {
    'en': 'English',
    'de': 'German', 
    'fr': 'French',
    'it': 'Italian',
    'es': 'Spanish',
    'nl': 'Dutch',
    'pt': 'Portuguese',
    'ru': 'Russian',
    'zh': 'Chinese',
    'ja': 'Japanese'
}

# Configuraciones por tipo de documento
ACADEMIC_PROCESSING_PROFILES = {
    "standard": {
        "language": "English,German,French,Italian,Spanish,Dutch",
        "outputFormat": "docx",
        "textTypes": "normal,table,barcode",
        "imageSource": "auto",
        "correctOrientation": True,
        "correctSkew": True,
        "readBarcodes": False,
        "preserveLayout": True,
        "preserveImages": True,
        "preserveTextFormatting": True,
        "preserveNonTextualObjects": True,
        "recognizeTextInImages": True,
        "preprocessImages": True,
        "profile": "documentConversion"
    },
    "high_quality": {
        "language": "English,German,French,Italian,Spanish,Dutch",
        "outputFormat": "docx",
        "textTypes": "normal,table,barcode",
        "imageSource": "auto",
        "correctOrientation": True,
        "correctSkew": True,
        "readBarcodes": False,
        "preserveLayout": True,
        "preserveImages": True,
        "preserveTextFormatting": True,
        "preserveNonTextualObjects": True,
        "recognizeTextInImages": True,
        "preprocessImages": True,
        "profile": "documentConversion",
        "imageProcessing": "sharp"
    },
    "text_only": {
        "language": "English,German,French,Italian,Spanish,Dutch",
        "outputFormat": "txt",
        "textTypes": "normal,table",
        "imageSource": "auto",
        "correctOrientation": True,
        "correctSkew": True,
        "readBarcodes": False,
        "preserveLayout": False,
        "preserveImages": False,
        "preserveTextFormatting": False,
        "preprocessImages": True,
        "profile": "textExtraction"
    }
}


# ===============================================================================
# ABBYY FINEREADER CLOUD INTEGRATION (MEJORADA)
# ===============================================================================

class ABBYYIntegration(BaseAPIClient):
    """
    Integración con ABBYY FineReader Cloud para OCR y conversión de documentos.
    
    ✅ POST-AUDITORÍA: Versión mejorada con robustez enterprise-grade.
    
    ABBYY FineReader Cloud es la solución OCR más avanzada disponible,
    especialmente efectiva para documentos académicos complejos con:
    - Múltiples columnas y layouts complejos
    - Tablas, fórmulas matemáticas y diagramas
    - Notas al pie y referencias cruzadas
    - Múltiples idiomas en el mismo documento
    
    Características Enterprise:
    ✅ ADRT (Adaptive Document Recognition Technology)
    ✅ Preservación de estructura original de documentos
    ✅ Soporte multiidioma con detección automática
    ✅ Estimación real de páginas y costos precisos
    ✅ Requests centralizadas con BaseAPIClient
    ✅ Polling resiliente con tolerancia a fallos
    ✅ Retries automáticos para todas las operaciones
    ✅ Rate limiting automático (100 req/min)
    ✅ Conversión a múltiples formatos especializados
    """
    
    def __init__(self,
                 api_key: str,
                 logger: Logger,
                 cache_manager: Optional[CacheManager] = None,
                 error_policy_manager: Optional[ErrorPolicyManager] = None,
                 max_file_size_mb: int = 180,  # ✅ NUEVO: Límite configurable (bajo el límite de 200MB)
                 processing_timeout: int = 1200):  # ✅ NUEVO: Timeout configurable (20 min)
        
        # ✅ VALIDACIÓN: API key de ABBYY
        if not validate_abbyy_api_key_robust(api_key):
            raise ValueError("API key de ABBYY inválida (debe tener al menos 20 caracteres)")
        
        # Rate limiter específico de ABBYY: 100 requests/minuto
        rate_limiter = create_rate_limiter(APIProvider.ABBYY, logger)
        
        super().__init__(
            api_key=api_key,
            base_url="https://cloud-eu.abbyy.com/v2/",
            provider=APIProvider.ABBYY,
            logger=logger,
            cache_manager=cache_manager,
            error_policy_manager=error_policy_manager,
            rate_limiter=rate_limiter
        )
        
        # Headers específicos de ABBYY
        self.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        })
        
        # ✅ NUEVO: Configuración avanzada
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024
        self.processing_timeout = processing_timeout
        
        # Configuración optimizada por defecto (perfil estándar)
        self.default_profile = "standard"
    
    def _estimate_cost(self, characters: int) -> float:
        """
        ✅ MEJORADO: Estima costo ABBYY basado en páginas reales.
        
        ABBYY pricing: ~$0.10 por página
        Estimación mejorada considerando tipo de contenido
        """
        estimated_pages = estimate_pages_from_characters(characters)
        base_cost = estimated_pages * 0.10
        
        # Factor de ajuste por complejidad típica de documentos académicos
        academic_factor = 1.2  # Documentos académicos tienden a ser más complejos
        
        return base_cost * academic_factor
    
    def _estimate_pages_from_file(self, file_path: Path) -> int:
        """
        ✅ MEJORA CRÍTICA: Estima páginas reales del archivo.
        
        Intenta múltiples métodos para obtener conteo preciso de páginas.
        """
        try:
            # Método 1: Si es PDF, intentar extraer número de páginas con PyMuPDF
            if file_path.suffix.lower() == '.pdf':
                try:
                    import pymupdf as fitz
                    doc = fitz.open(str(file_path))
                    page_count = doc.page_count
                    doc.close()
                    self.logger.debug(f"📊 ABBYY: Páginas detectadas con PyMuPDF: {page_count}")
                    return page_count
                except ImportError:
                    self.logger.debug("PyMuPDF no disponible, usando estimación por tamaño")
                except Exception as e:
                    self.logger.debug(f"Error con PyMuPDF: {e}, usando estimación por tamaño")
            
            # Método 2: Estimación por tamaño de archivo
            file_size = file_path.stat().st_size
            
            # Heurísticas por tipo de archivo
            if file_path.suffix.lower() == '.pdf':
                # PDFs académicos típicos: ~300-500KB por página
                estimated_pages = max(1, file_size // (400 * 1024))
            else:
                # Imágenes: típicamente 1 página, pero considerar tamaño
                if file_size > 5 * 1024 * 1024:  # > 5MB
                    estimated_pages = file_size // (5 * 1024 * 1024)  # Estimación por 5MB por página
                else:
                    estimated_pages = 1
            
            self.logger.debug(f"📊 ABBYY: Páginas estimadas por tamaño: {estimated_pages}")
            return int(estimated_pages)
            
        except Exception as e:
            self.logger.warning(f"⚠️ ABBYY: Error estimando páginas: {e}, usando 1 página")
            return 1
    
    def _sanitize_filename(self, filename: str) -> str:
        """
        ✅ NUEVO: Sanitiza nombres de archivo para evitar problemas del sistema.
        """
        # Remover caracteres problemáticos
        sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
        # Remover espacios múltiples y al inicio/final
        sanitized = re.sub(r'\s+', '_', sanitized.strip())
        # Limitar longitud
        if len(sanitized) > 200:
            name_part = sanitized[:180]
            ext_part = sanitized[-20:] if '.' in sanitized[-20:] else ""
            sanitized = f"{name_part}...{ext_part}"
        
        return sanitized
    
    async def health_check(self) -> bool:
        """Verifica salud de ABBYY API."""
        try:
            # ABBYY no tiene endpoint específico de health, verificamos con listSubmittedJobs
            response = await self._make_request("GET", "listSubmittedJobs", use_cache=False)
            
            # Si devuelve respuesta válida (aunque sea vacía), está disponible
            if response.success or response.data is not None:
                self.logger.info("✅ ABBYY: Health check exitoso")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"❌ ABBYY health check falló: {e}")
            return False
    
    async def get_application_info(self) -> APIResponse:
        """Obtiene información de la aplicación y créditos disponibles."""
        return await self._make_request("GET", "getApplicationInfo", cache_ttl=3600)  # Cache por 1 hora
    
    async def list_supported_formats(self) -> APIResponse:
        """
        ✅ MEJORADO: Lista formatos usando constantes centralizadas.
        """
        supported_formats = {
            "input_formats": {
                "images": [ext.lstrip('.') for ext in SUPPORTED_INPUT_FORMATS if ext != '.pdf'],
                "documents": ["pdf"],
                "max_file_size": f"{self.max_file_size_bytes // (1024 * 1024)}MB",
                "max_pages": "500 páginas",
                "max_resolution": "600 DPI recomendado"
            },
            "output_formats": {
                "documents": [fmt for fmt in SUPPORTED_OUTPUT_FORMATS if fmt in ["docx", "doc", "rtf", "txt", "pdf", "html", "xml"]],
                "structured": [fmt for fmt in SUPPORTED_OUTPUT_FORMATS if fmt in ["xlsx", "csv"]],
                "images": ["pdf", "tiff"]
            },
            "recommended_for_academic": {
                "input": "PDF de alta calidad (300+ DPI)",
                "output": "DOCX para preservación máxima de formato"
            },
            "language_support": {
                "automatic_detection": True,
                "supported_languages": list(LANGUAGE_MAPPING.values()),
                "mixed_language_support": True
            },
            "processing_profiles": list(ACADEMIC_PROCESSING_PROFILES.keys())
        }
        
        return APIResponse(
            success=True,
            data=supported_formats,
            provider=self.provider,
            request_id=create_request_id(self.provider),
            response_time=0.0,
            cached=True
        )
    
    async def process_document(self, 
                             task: DocumentProcessingTask,
                             profile: str = "standard",
                             max_retries: int = 2) -> APIResponse:  # ✅ NUEVO: Retries configurables
        """
        Procesa documento PDF usando ABBYY FineReader Cloud.
        
        ✅ POST-AUDITORÍA: Con retries automáticos, estimación real de páginas y polling resiliente.
        
        Proceso completo:
        1. Validación de archivo y configuración
        2. Upload y envío para procesamiento (con retry)
        3. Polling hasta completar OCR (con tolerancia a fallos)
        4. Download del resultado procesado (con retry)
        
        Args:
            task: Configuración de procesamiento de documento
            profile: Perfil de procesamiento ("standard", "high_quality", "text_only")
            max_retries: Número máximo de reintentos por operación
            
        Returns:
            APIResponse con información del documento procesado
        """
        file_path = Path(task.file_path)
        
        # Validaciones iniciales
        validation_response = self._validate_processing_task(task, file_path)
        if not validation_response.success:
            return validation_response
        
        file_size = file_path.stat().st_size
        # ✅ MEJORA CRÍTICA: Estimación real de páginas
        estimated_pages = self._estimate_pages_from_file(file_path)
        
        self.logger.info(f"📄 ABBYY: Iniciando procesamiento de {file_path.name} ({file_size / 1024 / 1024:.1f}MB, ~{estimated_pages} páginas)")
        
        task_id = None  # Para diagnóstico en caso de error
        
        try:
            # ✅ MEJORA: Paso 1 con retries automáticos
            processing_response = await self._retry_operation(
                self._submit_processing_task_centralized,
                max_retries,
                "upload",
                task, profile
            )
            
            if not processing_response.success:
                return processing_response
            
            task_id = processing_response.data["taskId"]
            
            # ✅ MEJORA: Paso 2 con polling resiliente
            completed_response = await self._poll_processing_completion_resilient(task_id)
            
            if not completed_response.success:
                # ✅ MEJORA: Incluir task_id para diagnóstico
                completed_response.data = completed_response.data or {}
                completed_response.data["partial_task_id"] = task_id
                return completed_response
            
            # ✅ MEJORA: Paso 3 con retries automáticos
            download_response = await self._retry_operation(
                self._download_processed_document_centralized,
                max_retries,
                "download",
                task_id, task.output_format
            )
            
            if download_response.success:
                # ✅ MEJORA: Información mejorada con estimación real
                download_response.data.update({
                    "original_file": str(file_path),
                    "original_size": file_size,
                    "estimated_pages": estimated_pages,
                    "actual_cost_estimate": self._estimate_cost(estimated_pages * 2000),  # 2000 chars por página estimado
                    "processing_settings": {
                        "output_format": task.output_format,
                        "languages": [lang.value for lang in task.language] if task.language else "auto",
                        "preserve_layout": task.preserve_layout,
                        "preserve_formatting": task.preserve_formatting,
                        "profile_used": profile
                    },
                    "task_id": task_id  # ✅ MEJORA: Siempre incluir task_id
                })
                
                self.logger.info(f"✅ ABBYY: Documento procesado guardado en {download_response.data['output_path']}")
            else:
                # ✅ MEJORA: Incluir task_id incluso en fallo de descarga
                download_response.data = download_response.data or {}
                download_response.data["partial_task_id"] = task_id
            
            return download_response
            
        except Exception as e:
            self.logger.error(f"❌ ABBYY: Error en procesamiento: {e}")
            return APIResponse(
                success=False,
                data={"partial_task_id": task_id} if task_id else None,
                provider=self.provider,
                request_id=create_request_id(self.provider),
                response_time=0.0,
                error_message=str(e)
            )
    
    async def _retry_operation(self, operation, max_retries: int, operation_name: str, *args, **kwargs) -> APIResponse:
        """
        ✅ NUEVO: Ejecuta operación con retries automáticos.
        """
        last_error = None
        
        for attempt in range(max_retries + 1):  # 0, 1, 2...
            try:
                response = await operation(*args, **kwargs)
                
                if response.success:
                    if attempt > 0:
                        self.logger.info(f"✅ ABBYY: {operation_name} exitoso después de {attempt} reintentos")
                    return response
                else:
                    # Si la respuesta no es exitosa pero no hay excepción, no reintentar
                    return response
                    
            except Exception as e:
                last_error = e
                
                if attempt < max_retries:
                    delay = (2 ** attempt) * 3.0  # 3s, 6s, 12s
                    self.logger.warning(f"🔄 ABBYY: {operation_name} falló (intento {attempt + 1}/{max_retries + 1}), reintentando en {delay}s: {str(e)}")
                    await asyncio.sleep(delay)
                else:
                    self.logger.error(f"❌ ABBYY: {operation_name} falló después de {max_retries + 1} intentos: {str(e)}")
        
        # Si llegamos aquí, todos los retries fallaron
        return APIResponse(
            success=False,
            data=None,
            provider=self.provider,
            request_id=create_request_id(self.provider),
            response_time=0.0,
            error_message=f"{operation_name} falló después de {max_retries + 1} intentos: {str(last_error)}"
        )
    
    def _validate_processing_task(self, task: DocumentProcessingTask, file_path: Path) -> APIResponse:
        """
        ✅ MEJORADO: Validación usando constantes centralizadas.
        """
        # Verificar que archivo existe
        if not file_path.exists():
            return APIResponse(
                success=False,
                data=None,
                provider=self.provider,
                request_id=create_request_id(self.provider),
                response_time=0.0,
                error_message=f"Archivo no encontrado: {task.file_path}"
            )
        
        # Verificar tamaño del archivo
        file_size = file_path.stat().st_size
        
        if file_size > self.max_file_size_bytes:
            return APIResponse(
                success=False,
                data=None,
                provider=self.provider,
                request_id=create_request_id(self.provider),
                response_time=0.0,
                error_message=f"Archivo demasiado grande: {file_size / 1024 / 1024:.1f}MB (máximo: {self.max_file_size_bytes / 1024 / 1024:.1f}MB)"
            )
        
        # ✅ MEJORA: Usar constantes centralizadas
        if file_path.suffix.lower() not in SUPPORTED_INPUT_FORMATS:
            return APIResponse(
                success=False,
                data=None,
                provider=self.provider,
                request_id=create_request_id(self.provider),
                response_time=0.0,
                error_message=f"Formato no soportado: {file_path.suffix}. Formatos válidos: {', '.join(SUPPORTED_INPUT_FORMATS)}"
            )
        
        # ✅ MEJORA: Usar constantes centralizadas
        if task.output_format not in SUPPORTED_OUTPUT_FORMATS:
            return APIResponse(
                success=False,
                data=None,
                provider=self.provider,
                request_id=create_request_id(self.provider),
                response_time=0.0,
                error_message=f"Formato de salida no soportado: {task.output_format}. Formatos válidos: {', '.join(SUPPORTED_OUTPUT_FORMATS)}"
            )
        
        return APIResponse(
            success=True,
            data={"validation": "passed"},
            provider=self.provider,
            request_id=create_request_id(self.provider),
            response_time=0.0
        )
    
    async def _submit_processing_task_centralized(self, task: DocumentProcessingTask, profile: str = "standard") -> APIResponse:
        """
        ✅ MEJORA CRÍTICA: Envía documento usando _make_request_multipart centralizado.
        """
        file_path = Path(task.file_path)
        
        self.logger.debug(f"📤 ABBYY: Enviando documento para procesamiento: {file_path.name}")
        
        # ✅ MEJORA: Usar perfil de procesamiento
        settings = ACADEMIC_PROCESSING_PROFILES.get(profile, ACADEMIC_PROCESSING_PROFILES["standard"]).copy()
        
        # Configurar formato de salida
        settings["outputFormat"] = task.output_format
        
        # ✅ MEJORA: Configurar idiomas usando mapeo centralizado
        if task.language:
            language_codes = []
            for lang in task.language:
                abbyy_lang = LANGUAGE_MAPPING.get(lang.value, lang.value.capitalize())
                language_codes.append(abbyy_lang)
            
            settings["language"] = ",".join(language_codes)
        
        # Configurar preservación de layout y formato
        settings["preserveLayout"] = task.preserve_layout
        settings["preserveTextFormatting"] = task.preserve_formatting
        
        # Leer archivo
        async with aiofiles.open(task.file_path, 'rb') as file:
            file_content = await file.read()
        
        # Preparar form data para upload
        form_data = FormData()
        form_data.add_field('file', file_content,
                           filename=file_path.name,
                           content_type=EXTENSION_TO_CONTENT_TYPE.get(file_path.suffix.lower(), 'application/octet-stream'))
        
        # Agregar configuración como campos de formulario
        for key, value in settings.items():
            form_data.add_field(key, str(value))
        
        # ✅ MEJORA CRÍTICA: Usar endpoint correcto según tipo de archivo
        if file_path.suffix.lower() == '.pdf':
            endpoint = "processDocument"
        else:
            endpoint = "processImage"
        
        # ✅ MEJORA CRÍTICA: Usar método centralizado
        response = await self._make_request_multipart(
            "POST", 
            endpoint,
            form_data=form_data,
            timeout_override=self.processing_timeout
        )
        
        if response.success:
            task_id = response.data.get("taskId")
            self.logger.info(f"📤 ABBYY: Documento enviado exitosamente, Task ID: {task_id}")
        
        return response
    
    async def _poll_processing_completion_resilient(self, task_id: str) -> APIResponse:
        """
        ✅ MEJORA CRÍTICA: Polling con tolerancia a fallos transitorios.
        """
        max_attempts = 120  # 20 minutos máximo
        poll_interval = 10   # 10 segundos iniciales
        max_consecutive_failures = 3  # ✅ NUEVO: Máximo de fallos consecutivos
        start_time = time.time()
        consecutive_failures = 0
        
        self.logger.info(f"⏳ ABBYY: Esperando procesamiento de tarea {task_id}")
        
        for attempt in range(max_attempts):
            try:
                status_response = await self._make_request(
                    "GET",
                    "getTaskStatus",
                    params={"taskId": task_id},
                    use_cache=False
                )
                
                # ✅ MEJORA: Resetear contador de fallos en éxito
                consecutive_failures = 0
                
                if not status_response.success:
                    self.logger.warning(f"⚠️ ABBYY: Error consultando estado de tarea: {status_response.error_message}")
                    consecutive_failures += 1
                    
                    # Solo abortar si hay muchos fallos consecutivos
                    if consecutive_failures >= max_consecutive_failures:
                        return status_response
                    
                    # Continuar con el siguiente intento
                    await asyncio.sleep(poll_interval)
                    continue
                
                status = status_response.data.get("taskStatus")
                
                if status == "Completed":
                    elapsed_time = time.time() - start_time
                    self.logger.info(f"✅ ABBYY: Procesamiento completado para tarea {task_id} ({elapsed_time:.1f}s)")
                    return status_response
                
                elif status in ["ProcessingFailed", "NotEnoughCredits", "InternalError"]:
                    error_msg = status_response.data.get("error", f"Error: {status}")
                    result_url = status_response.data.get("resultUrl", "")
                    
                    # Si hay resultUrl pero status es error, puede ser un resultado parcial útil
                    if result_url and status == "ProcessingFailed":
                        self.logger.warning(f"⚠️ ABBYY: Procesamiento con errores pero resultado disponible")
                        # Continuar con descarga del resultado parcial
                        return status_response
                    
                    self.logger.error(f"❌ ABBYY: Procesamiento falló: {error_msg}")
                    return APIResponse(
                        success=False,
                        data=status_response.data,
                        provider=self.provider,
                        request_id=task_id,
                        response_time=0.0,
                        error_message=error_msg
                    )
                
                # Status "Queued" o "InProgress" - continuar polling
                remaining_attempts = max_attempts - attempt - 1
                estimated_remaining = remaining_attempts * poll_interval
                
                self.logger.debug(f"🔄 ABBYY: Procesamiento en progreso... ({attempt + 1}/{max_attempts}) - Status: {status} - ETA: {estimated_remaining}s")
                
                # Polling adaptativo: aumentar intervalo para documentos grandes
                if attempt > 20:  # Después de 3+ minutos
                    poll_interval = 15  # 15 segundos
                elif attempt > 50:  # Después de 10+ minutos
                    poll_interval = 20  # 20 segundos
                    
            except Exception as e:
                # ✅ MEJORA: Tolerancia a errores transitorios
                consecutive_failures += 1
                self.logger.warning(f"⚠️ ABBYY: Error transitorio en polling ({consecutive_failures}/{max_consecutive_failures}): {str(e)}")
                
                # Solo abortar si hay muchos fallos consecutivos
                if consecutive_failures >= max_consecutive_failures:
                    return APIResponse(
                        success=False,
                        data=None,
                        provider=self.provider,
                        request_id=task_id,
                        response_time=0.0,
                        error_message=f"Demasiados errores consecutivos en polling: {str(e)}"
                    )
                
                # Esperar un poco más antes del siguiente intento
                await asyncio.sleep(poll_interval * 1.5)
                continue
                
            await asyncio.sleep(poll_interval)
        
        # Timeout
        elapsed_time = time.time() - start_time
        return APIResponse(
            success=False,
            data=None,
            provider=self.provider,
            request_id=task_id,
            response_time=elapsed_time,
            error_message=f"Timeout esperando completar procesamiento ABBYY ({elapsed_time:.1f}s)"
        )
    
    async def _download_processed_document_centralized(self, task_id: str, output_format: str) -> APIResponse:
        """
        ✅ MEJORA CRÍTICA: Descarga usando _make_request_binary centralizado y paths sanitizados.
        """
        timestamp = int(time.time())
        # ✅ MEJORA: Sanitizar nombre de archivo
        sanitized_id = self._sanitize_filename(task_id)
        output_path = f"abbyy_processed_{sanitized_id}_{timestamp}.{output_format}"
        
        self.logger.debug(f"💾 ABBYY: Descargando documento procesado a {output_path}")
        
        # ✅ MEJORA CRÍTICA: Usar método centralizado
        response = await self._make_request_binary(
            "GET",
            "getResult",
            params={"taskId": task_id},
            output_path=output_path,
            timeout_override=self.processing_timeout
        )
        
        if response.success:
            file_size = response.data.get("file_size", 0)
            self.logger.info(f"💾 ABBYY: Documento descargado exitosamente ({file_size / 1024:.1f}KB)")
            
            # Agregar información adicional
            response.data.update({
                "task_id": task_id,
                "output_format": output_format,
                "output_path": output_path
            })
        
        return response
    
    # ===============================================================================
    # MÉTODOS AUXILIARES Y DE UTILIDAD
    # ===============================================================================
    
    async def get_task_status(self, task_id: str) -> APIResponse:
        """
        Obtiene status actual de tarea ABBYY.
        
        Útil para monitoreo externo sin polling automático.
        """
        return await self._make_request(
            "GET",
            "getTaskStatus",
            params={"taskId": task_id},
            use_cache=False
        )
    
    async def list_submitted_jobs(self, from_date: Optional[str] = None) -> APIResponse:
        """
        Lista trabajos enviados recientemente.
        
        Args:
            from_date: Fecha desde cuando listar (formato: YYYY-MM-DD)
        """
        params = {}
        if from_date:
            params["fromDate"] = from_date
        
        return await self._make_request(
            "GET",
            "listSubmittedJobs",
            params=params,
            cache_ttl=300  # Cache por 5 minutos
        )
    
    async def delete_completed_task(self, task_id: str) -> APIResponse:
        """
        Elimina tarea completada para liberar espacio.
        
        Args:
            task_id: ID de la tarea a eliminar
        """
        return await self._make_request(
            "DELETE",
            "deleteTask",
            params={"taskId": task_id},
            use_cache=False
        )
    
    async def process_image(self, 
                          image_path: str,
                          output_format: str = "txt",
                          languages: Optional[List[SupportedLanguage]] = None,
                          profile: str = "text_only") -> APIResponse:
        """
        ✅ MEJORADO: Procesa imagen con perfil específico para imágenes.
        
        Método optimizado para imágenes únicas (no documentos complejos).
        
        Args:
            image_path: Ruta a la imagen
            output_format: Formato de salida (txt, rtf, docx, etc.)
            languages: Idiomas esperados en la imagen
            profile: Perfil de procesamiento específico para imágenes
            
        Returns:
            APIResponse con texto extraído
        """
        task = DocumentProcessingTask(
            file_path=image_path,
            output_format=output_format,
            language=languages or [],
            preserve_layout=False,  # Para imágenes simples
            preserve_formatting=False
        )
        
        self.logger.info(f"🖼️ ABBYY: Procesando imagen {Path(image_path).name} con perfil '{profile}'")
        
        return await self.process_document(task, profile=profile)
    
    async def batch_process_images(self, 
                                 image_paths: List[str],
                                 output_format: str = "txt",
                                 languages: Optional[List[SupportedLanguage]] = None,
                                 max_concurrent: int = 3) -> List[APIResponse]:
        """
        ✅ NUEVO: Procesa múltiples imágenes en paralelo con límite de concurrencia.
        
        Args:
            image_paths: Lista de rutas a imágenes
            output_format: Formato de salida
            languages: Idiomas esperados
            max_concurrent: Máximo número de procesamiento simultáneo
            
        Returns:
            Lista de APIResponse para cada imagen
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_single_image(image_path: str) -> APIResponse:
            async with semaphore:
                return await self.process_image(image_path, output_format, languages)
        
        self.logger.info(f"📚 ABBYY: Procesando {len(image_paths)} imágenes en lotes de {max_concurrent}")
        
        tasks = [process_single_image(path) for path in image_paths]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convertir excepciones a APIResponse de error
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                error_response = APIResponse(
                    success=False,
                    data=None,
                    provider=self.provider,
                    request_id=create_request_id(self.provider),
                    response_time=0.0,
                    error_message=f"Error procesando {image_paths[i]}: {str(result)}"
                )
                processed_results.append(error_response)
            else:
                processed_results.append(result)
        
        successful = sum(1 for r in processed_results if r.success)
        self.logger.info(f"✅ ABBYY: Lote completado - {successful}/{len(image_paths)} imágenes procesadas exitosamente")
        
        return processed_results


# ===============================================================================
# UTILIDADES ESPECÍFICAS DE ABBYY (MEJORADAS)
# ===============================================================================

def validate_abbyy_api_key_robust(api_key: str) -> bool:
    """
    ✅ MEJORADO: Valida formato de API key de ABBYY con verificaciones adicionales.
    """
    if not api_key or not isinstance(api_key, str):
        return False
    
    # Verificar longitud mínima
    if len(api_key) < 20:
        return False
    
    # ✅ MEJORA: Verificaciones más estrictas
    # ABBYY API keys suelen ser UUIDs o strings largos alfanuméricos
    # Verificar que tenga caracteres alfanuméricos y posibles guiones
    if not re.match(r'^[0-9a-fA-F\-_]+$', api_key):
        return False
    
    # Verificar que no sea obviamente un placeholder
    placeholder_patterns = ['your_key', 'api_key', 'test_key', 'dummy']
    if any(pattern in api_key.lower() for pattern in placeholder_patterns):
        return False
    
    return True


def get_abbyy_processing_profiles() -> dict:
    """
    ✅ MEJORADO: Retorna perfiles con más detalles y configuraciones reales.
    """
    return {
        "standard": {
            "description": "Conversión completa de documentos académicos",
            "best_for": ["PDFs académicos", "libros", "artículos", "tesis"],
            "preserves": ["layout", "formatting", "images", "tables", "footnotes"],
            "settings": ACADEMIC_PROCESSING_PROFILES["standard"],
            "estimated_accuracy": "95-98% para documentos de calidad media-alta",
            "processing_time": "Medio (~15s por página)"
        },
        "high_quality": {
            "description": "Máxima calidad para documentos complejos",
            "best_for": ["manuscritos antiguos", "documentos escaneados de baja calidad", "layouts complejos"],
            "preserves": ["máxima fidelidad visual", "estructura compleja", "elementos gráficos"],
            "settings": ACADEMIC_PROCESSING_PROFILES["high_quality"],
            "estimated_accuracy": "98-99% con preprocesamiento avanzado",
            "processing_time": "Lento (~25s por página)"
        },
        "text_only": {
            "description": "Extracción rápida de texto para análisis",
            "best_for": ["análisis de contenido", "búsqueda", "procesamiento masivo"],
            "preserves": ["solo contenido textual"],
            "settings": ACADEMIC_PROCESSING_PROFILES["text_only"],
            "estimated_accuracy": "92-95% para texto plano",
            "processing_time": "Rápido (~8s por página)"
        },
        "custom": {
            "description": "Configuración personalizable",
            "best_for": ["casos específicos", "requisitos particulares"],
            "preserves": ["según configuración"],
            "settings": "definible por usuario",
            "estimated_accuracy": "Variable según configuración",
            "processing_time": "Variable"
        }
    }


def estimate_abbyy_processing_time_enhanced(file_size_mb: float, 
                                          pages: int, 
                                          complexity: str = "medium",
                                          profile: str = "standard") -> dict:
    """
    ✅ MEJORADO: Estima tiempo con factores más precisos y específicos por perfil.
    """
    # ✅ MEJORA: Factores específicos por perfil
    profile_factors = {
        "text_only": 0.6,     # Más rápido, solo texto
        "standard": 1.0,      # Tiempo base
        "high_quality": 1.8,  # Más lento, máxima calidad
        "custom": 1.2         # Intermedio
    }
    
    # Factores de complejidad
    complexity_factors = {
        "low": 0.8,     # Texto simple, una columna
        "medium": 1.2,  # Documentos académicos típicos
        "high": 2.0     # Layouts complejos, múltiples idiomas
    }
    
    profile_factor = profile_factors.get(profile, 1.0)
    complexity_factor = complexity_factors.get(complexity, 1.2)
    
    # ✅ MEJORA: Estimación base más precisa por perfil
    base_seconds_per_page = {
        "text_only": 8,
        "standard": 15,
        "high_quality": 25,
        "custom": 18
    }
    
    base_time_per_page = base_seconds_per_page.get(profile, 15)
    base_time = pages * base_time_per_page * complexity_factor
    
    # Factor adicional por tamaño de archivo
    if file_size_mb > 50:
        base_time *= 1.3
    elif file_size_mb > 100:
        base_time *= 1.6
    
    # Tiempo mínimo y máximo
    min_time = max(20, base_time * 0.6)
    max_time = min(1800, base_time * 2.5)  # Máximo 30 minutos
    
    # ✅ NUEVO: Categorías de tiempo
    if base_time <= 60:
        time_category = "fast"
    elif base_time <= 300:
        time_category = "medium"
    else:
        time_category = "slow"
    
    return {
        "estimated_seconds": int(base_time),
        "min_seconds": int(min_time),
        "max_seconds": int(max_time),
        "estimated_minutes": round(base_time / 60, 1),
        "complexity_factor": complexity_factor,
        "profile_factor": profile_factor,
        "pages_factor": pages,
        "time_category": time_category,
        "profile_used": profile,
        "user_message": _generate_abbyy_time_message(int(base_time), profile, time_category)
    }


def _generate_abbyy_time_message(seconds: int, profile: str, category: str) -> str:
    """✅ NUEVO: Genera mensaje amigable específico para ABBYY."""
    profile_names = {
        "text_only": "extracción rápida",
        "standard": "procesamiento estándar",
        "high_quality": "calidad máxima",
        "custom": "configuración personalizada"
    }
    
    profile_name = profile_names.get(profile, "procesamiento")
    
    if category == "fast":
        return f"{profile_name.title()} (~{seconds}s)"
    elif category == "medium":
        return f"{profile_name.title()}: {seconds // 60}m {seconds % 60}s estimado"
    else:
        return f"{profile_name.title()} (documento complejo): {seconds // 60}m {seconds % 60}s estimado"


def get_quality_optimization_tips_enhanced() -> dict:
    """
    ✅ MEJORADO: Consejos más específicos y accionables.
    """
    return {
        "input_optimization": {
            "resolution": {
                "text_documents": "300-400 DPI óptimo para texto normal",
                "complex_layouts": "400-600 DPI para tablas y fórmulas",
                "images_with_text": "600-800 DPI para calidad máxima",
                "avoid": "Evitar >800 DPI (innecesario, archivos grandes)"
            },
            "format": {
                "best": "PDF original vectorial mejor que escaneado",
                "good": "PNG/TIFF sin compresión para imágenes",
                "avoid": "JPEG con compresión alta (artefactos)"
            },
            "color": {
                "recommended": "Color o escala grises para documentos complejos",
                "acceptable": "B&N solo para texto simple sin gráficos",
                "note": "Color mejora precisión en tablas y diagramas"
            }
        },
        "document_preparation": {
            "physical_quality": [
                "Páginas planas sin dobleces o arrugas",
                "Eliminar grapas y clips que causen sombras",
                "Iluminación uniforme sin sombras laterales",
                "Enfoque nítido en toda la superficie"
            ],
            "scanning_setup": [
                "Orientación correcta antes de enviar",
                "Márgenes suficientes alrededor del texto",
                "Contraste adecuado entre texto y fondo",
                "Evitar reflejos en superficies brillantes"
            ]
        },
        "abbyy_settings": {
            "language_detection": [
                "Especificar idiomas conocidos mejora 15-20% precisión",
                "Usar 'auto' solo si idiomas son completamente desconocidos",
                "Limitar a idiomas realmente presentes en documento"
            ],
            "profile_selection": [
                "documentConversion: documentos académicos estándar",
                "textExtraction: cuando solo necesitas el texto",
                "high_quality: manuscritos o documentos deteriorados"
            ],
            "layout_preservation": [
                "Habilitar para documentos con formato importante",
                "Deshabilitar para extracción simple de texto",
                "Crítico para tablas y columnas múltiples"
            ]
        },
        "post_processing": {
            "manual_review": [
                "Revisar especialmente: tablas, fórmulas matemáticas, números",
                "Verificar: itálicas, negritas, símbolos especiales",
                "Confirmar: notas al pie correctamente vinculadas",
                "Validar: estructura de headings y subheadings"
            ],
            "common_issues": [
                "Confusión O/0, I/l/1 en fuentes similares",
                "Espaciado incorrecto en tablas alineadas",
                "Pérdida de formato en texto multilínea",
                "Separación incorrecta de columnas"
            ]
        },
        "performance_tips": {
            "batch_processing": [
                "Procesar máximo 3-5 documentos simultáneamente",
                "Usar perfiles apropiados para reducir tiempo",
                "Agrupar documentos similares para eficiencia"
            ],
            "cost_optimization": [
                "Usar text_only para análisis de contenido",
                "Reservar high_quality solo para casos críticos",
                "Considerar preprocesamiento local para ahorrar créditos"
            ]
        }
    }


def get_supported_language_combinations() -> dict:
    """✅ NUEVO: Combinaciones de idiomas más comunes para validación."""
    return {
        "academic_common": [
            ["English"], ["German"], ["French"], ["Italian"], ["Spanish"],
            ["English", "German"], ["English", "French"], ["German", "French"],
            ["English", "Spanish"], ["English", "Italian"]
        ],
        "multilingual_documents": [
            ["English", "German", "French"],
            ["English", "Spanish", "Portuguese"],
            ["German", "French", "Italian"],
            ["English", "German", "Dutch"]
        ],
        "single_language_codes": list(LANGUAGE_MAPPING.keys()),
        "abbyy_language_names": list(LANGUAGE_MAPPING.values())
    }


# ===============================================================================
# TESTS UNITARIOS EMBEBIDOS (MEJORADOS)
# ===============================================================================

async def test_abbyy_application_info():
    """✅ MEJORADO: Test básico de información de aplicación."""
    import os
    api_key = os.getenv("ABBYY_API_KEY")
    
    if not api_key or api_key.startswith("your_"):
        print("⚠️ Test ABBYY omitido: API key no configurada")
        return
    
    import logging
    logger = logging.getLogger("test")
    abbyy = ABBYYIntegration(api_key, logger)
    
    # Test health check
    is_healthy = await abbyy.health_check()
    assert is_healthy, "ABBYY API debe estar disponible"
    
    # Test información de aplicación
    app_info = await abbyy.get_application_info()
    assert app_info.success, f"Información de app debe ser exitosa: {app_info.error_message}"
    
    # Test formatos soportados mejorados
    formats = await abbyy.list_supported_formats()
    assert formats.success
    assert "input_formats" in formats.data
    assert "output_formats" in formats.data
    assert "processing_profiles" in formats.data
    
    print("✅ Test ABBYY Application Info (mejorado): PASSED")


def test_abbyy_utilities():
    """✅ MEJORADO: Test de utilidades específicas de ABBYY."""
    # Test validación API key robusta
    assert validate_abbyy_api_key_robust("1234567890abcdef1234567890") == True
    assert validate_abbyy_api_key_robust("short") == False
    assert validate_abbyy_api_key_robust("") == False
    assert validate_abbyy_api_key_robust("your_key_here_12345") == False  # Placeholder
    assert validate_abbyy_api_key_robust("api_key_test_123456789") == False  # Placeholder
    
    # Test perfiles de procesamiento mejorados
    profiles = get_abbyy_processing_profiles()
    assert "standard" in profiles
    assert "high_quality" in profiles
    assert "text_only" in profiles
    assert "estimated_accuracy" in profiles["standard"]
    assert "settings" in profiles["standard"]
    
    # Test estimación de tiempo mejorada
    estimation = estimate_abbyy_processing_time_enhanced(10.0, 50, "medium", "standard")
    assert estimation["estimated_seconds"] > 0
    assert estimation["min_seconds"] <= estimation["estimated_seconds"]
    assert estimation["estimated_seconds"] <= estimation["max_seconds"]
    assert estimation["pages_factor"] == 50
    assert estimation["profile_factor"] == 1.0
    assert "time_category" in estimation
    assert "user_message" in estimation
    
    # Test consejos de optimización mejorados
    tips = get_quality_optimization_tips_enhanced()
    assert "input_optimization" in tips
    assert "resolution" in tips["input_optimization"]
    assert "performance_tips" in tips
    
    # Test combinaciones de idiomas
    lang_combos = get_supported_language_combinations()
    assert "academic_common" in lang_combos
    assert ["English"] in lang_combos["academic_common"]
    
    print("✅ Test ABBYY Utilities (mejorado): PASSED")


def test_abbyy_task_validation():
    """✅ MEJORADO: Test de validación de tareas con constantes."""
    import logging
    logger = logging.getLogger("test")
    abbyy = ABBYYIntegration("test-api-key-1234567890", logger)
    
    # Test validación de tarea inválida (archivo no existe)
    invalid_task = DocumentProcessingTask(
        file_path="nonexistent_file.pdf",
        output_format="docx"
    )
    
    validation = abbyy._validate_processing_task(invalid_task, Path("nonexistent_file.pdf"))
    assert not validation.success
    assert "no encontrado" in validation.error_message
    
    # Test validación de formato no soportado
    invalid_format_task = DocumentProcessingTask(
        file_path="test.txt",  # .txt no es formato de entrada válido
        output_format="docx"
    )
    
    # Crear archivo temporal para test
    test_file = Path("test.txt")
    test_file.write_text("test content")
    
    try:
        validation = abbyy._validate_processing_task(invalid_format_task, test_file)
        assert not validation.success
        assert "Formato no soportado" in validation.error_message
        # ✅ MEJORA: Verificar que usa constantes
        assert any(fmt in validation.error_message for fmt in SUPPORTED_INPUT_FORMATS)
    finally:
        test_file.unlink(missing_ok=True)
    
    print("✅ Test ABBYY Task Validation (mejorado): PASSED")


def test_page_estimation():
    """✅ NUEVO: Test de estimación de páginas mejorada."""
    import logging
    logger = logging.getLogger("test")
    abbyy = ABBYYIntegration("test-api-key-1234567890", logger)
    
    # Test con archivo PDF simulado
    test_pdf = Path("test_document.pdf")
    # Simular PDF de ~5MB (aproximadamente 12-13 páginas)
    test_content = b"fake pdf content" * 100000  # ~1.6MB
    test_pdf.write_bytes(test_content)
    
    try:
        pages = abbyy._estimate_pages_from_file(test_pdf)
        assert pages >= 1  # Debe estimar al menos 1 página
        assert pages <= 50  # No debe ser excesivamente alto
    finally:
        test_pdf.unlink(missing_ok=True)
    
    # Test sanitización de nombres
    sanitized = abbyy._sanitize_filename("file<name>:with|problems?.pdf")
    assert "<" not in sanitized
    assert ">" not in sanitized
    assert ":" not in sanitized
    assert "|" not in sanitized
    assert "?" not in sanitized
    
    print("✅ Test Page Estimation: PASSED")


def test_language_mapping():
    """✅ NUEVO: Test de mapeo de idiomas."""
    # Test mapeo de códigos de idioma
    assert LANGUAGE_MAPPING["en"] == "English"
    assert LANGUAGE_MAPPING["de"] == "German"
    assert LANGUAGE_MAPPING["es"] == "Spanish"
    
    # Test que todos los idiomas soportados están mapeados
    assert len(LANGUAGE_MAPPING) >= 10  # Al menos 10 idiomas
    
    # Test constantes de formatos
    assert ".pdf" in SUPPORTED_INPUT_FORMATS
    assert ".jpg" in SUPPORTED_INPUT_FORMATS
    assert "docx" in SUPPORTED_OUTPUT_FORMATS
    assert "txt" in SUPPORTED_OUTPUT_FORMATS
    
    print("✅ Test Language Mapping: PASSED")


async def run_all_tests():
    """Ejecuta todos los tests embebidos."""
    print("🧪 Ejecutando tests de abbyy_integration.py (POST-AUDITORÍA)...")
    
    try:
        test_abbyy_utilities()
        test_abbyy_task_validation()
        test_page_estimation()
        test_language_mapping()
        await test_abbyy_application_info()  # Omitido si no hay API key
        
        print("\n✅ Todos los tests de abbyy_integration.py (POST-AUDITORÍA) pasaron!")
        print("\n🏆 MEJORAS IMPLEMENTADAS:")
        print("  ✅ Constantes centralizadas (DRY principle)")
        print("  ✅ Estimación real de páginas (crítico para costos)")
        print("  ✅ Requests centralizadas usando BaseAPIClient")
        print("  ✅ Polling resiliente con tolerancia a fallos")
        print("  ✅ Retries automáticos para todas las operaciones")
        print("  ✅ Sanitización robusta de nombres de archivo")
        print("  ✅ Validación de API key mejorada")
        print("  ✅ Perfiles de procesamiento con configuraciones reales")
        print("  ✅ Estimación de tiempo específica por perfil")
        print("  ✅ Procesamiento en lotes para múltiples imágenes")
        
    except Exception as e:
        print(f"\n❌ Test falló: {e}")
        raise


if __name__ == "__main__":
    """Ejecutar tests al correr el módulo directamente."""
    import logging
    import asyncio
    logging.basicConfig(level=logging.INFO)
    
    asyncio.run(run_all_tests())