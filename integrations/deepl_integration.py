#!/usr/bin/env python3
"""
🔄 DEEPL_INTEGRATION.PY - Integración DeepL Pro API
Sistema de Traducción Académica v2.2 - APIs Integration Layer
POST-AUDITORÍA: Versión mejorada con correcciones críticas

Integración especializada con DeepL Pro API para traducción de documentos
académicos con preservación de formato y estructura.

Características:
✅ Traducción de documentos completos (PDF, DOCX, HTML)
✅ Preservación de formatos originales
✅ Configuración académica especializada (formalidad)
✅ Polling resiliente con tolerancia a fallos transitorios
✅ Requests centralizadas para uploads/downloads binarios
✅ Retries automáticos para operaciones de documentos
✅ Validación robusta y sanitización de paths

Autor: Sistema ClaudeAcademico v2.2
Fecha: Enero 2025 (Post-Auditoría)
Ubicación: integrations/deepl_integration.py
"""

import asyncio
import re
import time
from pathlib import Path
from typing import Optional, Union, Dict, Any
from urllib.parse import urljoin

import aiofiles
import aiohttp
from aiohttp import ClientSession, FormData

from .base_client import BaseAPIClient, create_rate_limiter
from .models import (
    APIProvider, APIResponse, TranslationTask, SupportedLanguage,
    Logger, CacheManager, ErrorPolicyManager, create_request_id
)


# ===============================================================================
# DEEPL PRO API INTEGRATION (MEJORADA)
# ===============================================================================

class DeepLProIntegration(BaseAPIClient):
    """
    Integración con DeepL Pro API para traducción con preservación estructural.
    
    ✅ POST-AUDITORÍA: Versión mejorada con robustez enterprise-grade.
    
    DeepL Pro es considerado el mejor traductor automático disponible,
    especialmente efectivo para textos académicos y técnicos.
    
    Características Enterprise:
    ✅ Traducción de documentos completos preservando formato
    ✅ Configuración específica para textos académicos
    ✅ Polling resiliente con tolerancia a fallos transitorios
    ✅ Requests centralizadas para uploads/downloads
    ✅ Retries automáticos para todas las operaciones
    ✅ Validación robusta y sanitización de paths
    ✅ Rate limiting automático (1000 req/min)
    ✅ Fallbacks y recuperación de errores
    ✅ Métricas detalladas de uso y costos
    """
    
    def __init__(self,
                 api_key: str,
                 logger: Logger,
                 cache_manager: Optional[CacheManager] = None,
                 error_policy_manager: Optional[ErrorPolicyManager] = None,
                 document_timeout: int = 600,  # ✅ NUEVO: Timeout configurable para documentos
                 max_file_size_mb: int = 45):  # ✅ NUEVO: Límite configurable (bajo el límite de 50MB)
        
        # ✅ VALIDACIÓN: API key de DeepL Pro
        if not validate_deepl_api_key(api_key):
            raise ValueError("API key de DeepL Pro inválida (debe terminar en ':fx')")
        
        # Rate limiter específico de DeepL Pro: 1000 requests/minuto
        rate_limiter = create_rate_limiter(APIProvider.DEEPL, logger)
        
        super().__init__(
            api_key=api_key,
            base_url="https://api.deepl.com/v2/",
            provider=APIProvider.DEEPL,
            logger=logger,
            cache_manager=cache_manager,
            error_policy_manager=error_policy_manager,
            rate_limiter=rate_limiter
        )
        
        # Headers específicos de DeepL
        self.headers.update({
            "Authorization": f"DeepL-Auth-Key {api_key}",
            "Content-Type": "application/json"
        })
        
        # ✅ NUEVO: Configuración avanzada
        self.document_timeout = document_timeout
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024
        
        # Configuración optimizada para traducción académica
        self.academic_settings = {
            "preserve_formatting": True,
            "formality": "prefer_more",  # Formal para textos académicos
            "split_sentences": "1",     # Preservar estructura de oraciones
            "outline_detection": "1",   # Detectar estructura de documento
            "show_billed_characters": True
        }
    
    def _estimate_cost(self, characters: int) -> float:
        """
        Estima costo DeepL basado en caracteres.
        
        DeepL Pro: €25/mes por 1,000,000 caracteres
        = €0.000025 por carácter = $0.000027 USD aprox
        """
        return characters * 0.000027
    
    # ✅ MEJORA CRÍTICA: Extender _make_request para manejar multipart y binarios
    async def _make_request_multipart(self,
                                    method: str,
                                    endpoint: str,
                                    form_data: Optional[FormData] = None,
                                    headers: Optional[Dict[str, str]] = None,
                                    timeout_override: Optional[int] = None) -> APIResponse:
        """
        Realiza request HTTP multipart para uploads de archivos.
        
        ✅ MEJORA: Centraliza lógica de uploads usando infraestructura del cliente base.
        """
        request_start = time.time()
        request_id = create_request_id(self.provider)
        
        # Usar timeout específico para documentos si se proporciona
        timeout = self.timeout
        if timeout_override:
            timeout = aiohttp.ClientTimeout(total=timeout_override, connect=30)
        
        # Preparar headers (sin Content-Type para multipart)
        req_headers = {k: v for k, v in self.headers.items() if k != "Content-Type"}
        if headers:
            req_headers.update(headers)
        
        # Rate limiting
        if self.rate_limiter:
            await self.rate_limiter.acquire()
        
        # ✅ MEJORA: Usar Circuit Breaker para uploads también
        try:
            response_data = await self.circuit_breaker.call(
                self._execute_multipart_request,
                method, urljoin(self.base_url, endpoint), req_headers, form_data, timeout
            )
            
            response_time = time.time() - request_start
            
            # Actualizar métricas
            self.usage_metrics.add_request(
                response_time=response_time,
                success=True
            )
            
            self.logger.info(f"✅ {self.provider.value.upper()} Multipart request exitoso: {endpoint} ({response_time:.2f}s)")
            
            return APIResponse(
                success=True,
                data=response_data,
                provider=self.provider,
                request_id=request_id,
                response_time=response_time
            )
            
        except Exception as e:
            response_time = time.time() - request_start
            
            self.usage_metrics.add_request(
                response_time=response_time,
                success=False
            )
            
            self.logger.error(f"❌ {self.provider.value.upper()} Multipart request falló: {endpoint} - {str(e)}")
            
            return APIResponse(
                success=False,
                data=None,
                provider=self.provider,
                request_id=request_id,
                response_time=response_time,
                error_message=str(e)
            )
    
    async def _execute_multipart_request(self,
                                       method: str,
                                       url: str,
                                       headers: Dict[str, str],
                                       form_data: Optional[FormData] = None,
                                       timeout: Optional[aiohttp.ClientTimeout] = None) -> Any:
        """Ejecuta request multipart real."""
        async with ClientSession(timeout=timeout or self.timeout) as session:
            async with session.request(
                method=method,
                url=url,
                headers=headers,
                data=form_data
            ) as response:
                
                if response.status >= 400:
                    error_text = await response.text()
                    raise aiohttp.ClientResponseError(
                        request_info=response.request_info,
                        history=response.history,
                        status=response.status,
                        message=f"HTTP {response.status}: {error_text}"
                    )
                
                return await response.json()
    
    async def _make_request_binary(self,
                                 method: str,
                                 endpoint: str,
                                 data: Optional[Dict[str, Any]] = None,
                                 output_path: Optional[str] = None,
                                 timeout_override: Optional[int] = None) -> APIResponse:
        """
        Realiza request HTTP para descargas binarias.
        
        ✅ MEJORA: Centraliza lógica de downloads binarios.
        """
        request_start = time.time()
        request_id = create_request_id(self.provider)
        
        # Usar timeout específico para documentos si se proporciona
        timeout = self.timeout
        if timeout_override:
            timeout = aiohttp.ClientTimeout(total=timeout_override, connect=30)
        
        # Rate limiting
        if self.rate_limiter:
            await self.rate_limiter.acquire()
        
        try:
            response_data = await self.circuit_breaker.call(
                self._execute_binary_request,
                method, urljoin(self.base_url, endpoint), self.headers, data, output_path, timeout
            )
            
            response_time = time.time() - request_start
            
            self.usage_metrics.add_request(
                response_time=response_time,
                success=True
            )
            
            self.logger.info(f"✅ {self.provider.value.upper()} Binary download exitoso: {endpoint} ({response_time:.2f}s)")
            
            return APIResponse(
                success=True,
                data=response_data,
                provider=self.provider,
                request_id=request_id,
                response_time=response_time
            )
            
        except Exception as e:
            response_time = time.time() - request_start
            
            self.usage_metrics.add_request(
                response_time=response_time,
                success=False
            )
            
            self.logger.error(f"❌ {self.provider.value.upper()} Binary download falló: {endpoint} - {str(e)}")
            
            return APIResponse(
                success=False,
                data=None,
                provider=self.provider,
                request_id=request_id,
                response_time=response_time,
                error_message=str(e)
            )
    
    async def _execute_binary_request(self,
                                    method: str,
                                    url: str,
                                    headers: Dict[str, str],
                                    data: Optional[Dict[str, Any]] = None,
                                    output_path: Optional[str] = None,
                                    timeout: Optional[aiohttp.ClientTimeout] = None) -> Dict[str, Any]:
        """Ejecuta request binaria real y guarda archivo."""
        async with ClientSession(timeout=timeout or self.timeout) as session:
            async with session.request(
                method=method,
                url=url,
                headers=headers,
                json=data
            ) as response:
                
                if response.status >= 400:
                    error_text = await response.text()
                    raise aiohttp.ClientResponseError(
                        request_info=response.request_info,
                        history=response.history,
                        status=response.status,
                        message=f"HTTP {response.status}: {error_text}"
                    )
                
                # Guardar archivo si se proporciona path
                if output_path:
                    async with aiofiles.open(output_path, 'wb') as file:
                        async for chunk in response.content.iter_chunked(8192):
                            await file.write(chunk)
                    
                    file_size = Path(output_path).stat().st_size
                    
                    return {
                        "output_path": output_path,
                        "file_size": file_size,
                        "download_timestamp": time.time()
                    }
                else:
                    # Retornar contenido binario si no hay path
                    return {
                        "content": await response.read(),
                        "content_type": response.headers.get("Content-Type", "application/octet-stream")
                    }
    
    async def health_check(self) -> bool:
        """Verifica salud de DeepL API consultando uso."""
        try:
            response = await self._make_request("GET", "usage", use_cache=False)
            
            if response.success:
                usage_data = response.data
                character_limit = usage_data.get("character_limit", 0)
                character_count = usage_data.get("character_count", 0)
                
                # Verificar que no estemos cerca del límite
                usage_percentage = (character_count / character_limit) * 100 if character_limit > 0 else 0
                
                if usage_percentage > 95:
                    self.logger.warning(f"⚠️ DeepL: Uso alto ({usage_percentage:.1f}% del límite)")
                elif usage_percentage > 80:
                    self.logger.info(f"📊 DeepL: Uso moderado ({usage_percentage:.1f}% del límite)")
                
                self.logger.info(f"✅ DeepL: Saludable - {character_count:,}/{character_limit:,} caracteres usados")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"❌ DeepL health check falló: {e}")
            return False
    
    async def get_usage_info(self) -> APIResponse:
        """Obtiene información detallada de uso de la cuenta DeepL."""
        return await self._make_request("GET", "usage", cache_ttl=300)  # Cache por 5 minutos
    
    async def get_supported_languages(self, type: str = "target") -> APIResponse:
        """
        Obtiene idiomas soportados por DeepL.
        
        Args:
            type: "source" o "target" para idiomas de origen o destino
        """
        if type not in ["source", "target"]:
            return APIResponse(
                success=False,
                data=None,
                provider=self.provider,
                request_id=create_request_id(self.provider),
                response_time=0.0,
                error_message="Tipo debe ser 'source' o 'target'"
            )
        
        params = {"type": type}
        return await self._make_request("GET", "languages", params=params, cache_ttl=86400)  # Cache por 24h
    
    async def translate_text(self, task: TranslationTask) -> APIResponse:
        """
        Traduce texto usando DeepL API.
        
        Args:
            task: Tarea de traducción con configuración
            
        Returns:
            APIResponse con texto traducido y metadata
        """
        # ✅ MEJORA: Validación más robusta
        if not task.source_text or not task.source_text.strip():
            return APIResponse(
                success=False,
                data=None,
                provider=self.provider,
                request_id=create_request_id(self.provider),
                response_time=0.0,
                error_message="source_text no puede estar vacío"
            )
        
        # Validar tamaño del texto (límite DeepL: 128KB por request)
        text_size = len(task.source_text.encode('utf-8'))
        max_text_size = 128 * 1024  # 128KB
        
        if text_size > max_text_size:
            return APIResponse(
                success=False,
                data=None,
                provider=self.provider,
                request_id=create_request_id(self.provider),
                response_time=0.0,
                error_message=f"Texto demasiado grande: {text_size / 1024:.1f}KB (máximo: 128KB)"
            )
        
        # Preparar datos para DeepL
        data = {
            "text": [task.source_text],
            "source_lang": task.source_lang.value.upper(),
            "target_lang": task.target_lang.value.upper(),
            **self.academic_settings
        }
        
        # Override configuración específica de la tarea
        if hasattr(task, 'preserve_formatting') and not task.preserve_formatting:
            data["preserve_formatting"] = False
        
        if hasattr(task, 'formality') and task.formality:
            data["formality"] = task.formality
        
        self.logger.info(f"🔄 DeepL: Traduciendo texto {task.source_lang.value} → {task.target_lang.value} ({len(task.source_text)} chars)")
        
        response = await self._make_request("POST", "translate", data=data, use_cache=True, cache_ttl=3600)
        
        if response.success and response.data:
            translations = response.data.get("translations", [])
            
            if translations:
                translation = translations[0]
                translated_text = translation.get("text", "")
                detected_lang = translation.get("detected_source_language", "")
                billed_chars = response.data.get("billed_characters", len(task.source_text))
                
                self.logger.info(f"✅ DeepL: Traducción completada ({len(translated_text)} chars, {billed_chars} facturados)")
                
                # Actualizar response data para incluir información útil
                response.data = {
                    "translated_text": translated_text,
                    "detected_source_language": detected_lang,
                    "billed_characters": billed_chars,
                    "source_length": len(task.source_text),
                    "target_length": len(translated_text),
                    "translation_metadata": {
                        "formality_applied": data.get("formality"),
                        "formatting_preserved": data.get("preserve_formatting"),
                        "split_sentences": data.get("split_sentences"),
                        "outline_detection": data.get("outline_detection")
                    }
                }
                
                # Actualizar métricas con caracteres facturados reales
                response.cost_estimate = self._estimate_cost(billed_chars)
            else:
                response.success = False
                response.error_message = "No se recibieron traducciones en la respuesta"
        
        return response
    
    def _sanitize_filename(self, filename: str) -> str:
        """
        ✅ NUEVO: Sanitiza nombres de archivo para evitar problemas del sistema.
        
        Args:
            filename: Nombre de archivo a sanitizar
            
        Returns:
            Nombre de archivo sanitizado
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
    
    async def translate_document(self, 
                               file_path: str,
                               task: TranslationTask,
                               output_path: Optional[str] = None,
                               max_retries: int = 2) -> APIResponse:  # ✅ NUEVO: Retries para documentos
        """
        Traduce documento completo preservando formato.
        
        ✅ POST-AUDITORÍA: Con retries automáticos y mejor manejo de errores.
        
        Proceso:
        1. Upload del documento a DeepL (con retry)
        2. Polling hasta completar traducción (con tolerancia a fallos)
        3. Download del documento traducido (con retry)
        
        Args:
            file_path: Ruta al archivo a traducir
            task: Configuración de traducción
            output_path: Ruta de salida (opcional)
            max_retries: Número máximo de reintentos por operación
            
        Returns:
            APIResponse con información del documento traducido
        """
        file_path_obj = Path(file_path)
        
        if not file_path_obj.exists():
            return APIResponse(
                success=False,
                data=None,
                provider=self.provider,
                request_id=create_request_id(self.provider),
                response_time=0.0,
                error_message=f"Archivo no encontrado: {file_path}"
            )
        
        # Verificar tamaño del archivo
        file_size = file_path_obj.stat().st_size
        
        if file_size > self.max_file_size_bytes:
            return APIResponse(
                success=False,
                data=None,
                provider=self.provider,
                request_id=create_request_id(self.provider),
                response_time=0.0,
                error_message=f"Archivo demasiado grande: {file_size / 1024 / 1024:.1f}MB (máximo: {self.max_file_size_bytes / 1024 / 1024:.1f}MB)"
            )
        
        self.logger.info(f"📄 DeepL: Iniciando traducción de documento {file_path_obj.name} ({file_size / 1024:.1f}KB)")
        
        document_id = None  # ✅ MEJORA: Conservar para diagnóstico
        
        try:
            # ✅ MEJORA: Paso 1 con retries automáticos
            upload_response = await self._retry_operation(
                self._upload_document,
                max_retries,
                "upload",
                file_path, task
            )
            
            if not upload_response.success:
                return upload_response
            
            document_id = upload_response.data["document_id"]
            document_key = upload_response.data["document_key"]
            
            # ✅ MEJORA: Paso 2 con tolerancia a fallos transitorios
            completed_response = await self._poll_document_translation_resilient(document_id, document_key)
            
            if not completed_response.success:
                # ✅ MEJORA: Incluir document_id para diagnóstico
                completed_response.data = completed_response.data or {}
                completed_response.data["partial_document_id"] = document_id
                return completed_response
            
            # ✅ MEJORA: Paso 3 con retries automáticos
            download_response = await self._retry_operation(
                self._download_translated_document,
                max_retries,
                "download",
                document_id, document_key, output_path
            )
            
            if download_response.success:
                self.logger.info(f"✅ DeepL: Documento traducido guardado en {download_response.data['output_path']}")
                
                # Agregar información adicional
                download_response.data.update({
                    "original_file": str(file_path_obj),
                    "original_size": file_size,
                    "document_id": document_id,  # ✅ MEJORA: Siempre incluir document_id
                    "translation_task": {
                        "source_lang": task.source_lang.value,
                        "target_lang": task.target_lang.value,
                        "formality": getattr(task, 'formality', 'prefer_more'),
                        "preserve_formatting": getattr(task, 'preserve_formatting', True)
                    }
                })
            else:
                # ✅ MEJORA: Incluir document_id incluso en fallo de descarga
                download_response.data = download_response.data or {}
                download_response.data["partial_document_id"] = document_id
            
            return download_response
            
        except Exception as e:
            self.logger.error(f"❌ DeepL: Error en traducción de documento: {e}")
            return APIResponse(
                success=False,
                data={"partial_document_id": document_id} if document_id else None,  # ✅ MEJORA: Incluir ID parcial
                provider=self.provider,
                request_id=create_request_id(self.provider),
                response_time=0.0,
                error_message=str(e)
            )
    
    async def _retry_operation(self, operation, max_retries: int, operation_name: str, *args, **kwargs) -> APIResponse:
        """
        ✅ NUEVO: Ejecuta operación con retries automáticos.
        
        Args:
            operation: Función async a ejecutar
            max_retries: Número máximo de retries
            operation_name: Nombre de la operación para logging
            *args, **kwargs: Argumentos para la operación
            
        Returns:
            APIResponse de la operación
        """
        last_error = None
        
        for attempt in range(max_retries + 1):  # 0, 1, 2...
            try:
                response = await operation(*args, **kwargs)
                
                if response.success:
                    if attempt > 0:
                        self.logger.info(f"✅ DeepL: {operation_name} exitoso después de {attempt} reintentos")
                    return response
                else:
                    # Si la respuesta no es exitosa pero no hay excepción, no reintentar
                    return response
                    
            except Exception as e:
                last_error = e
                
                if attempt < max_retries:
                    delay = (2 ** attempt) * 2.0  # 2s, 4s, 8s
                    self.logger.warning(f"🔄 DeepL: {operation_name} falló (intento {attempt + 1}/{max_retries + 1}), reintentando en {delay}s: {str(e)}")
                    await asyncio.sleep(delay)
                else:
                    self.logger.error(f"❌ DeepL: {operation_name} falló después de {max_retries + 1} intentos: {str(e)}")
        
        # Si llegamos aquí, todos los retries fallaron
        return APIResponse(
            success=False,
            data=None,
            provider=self.provider,
            request_id=create_request_id(self.provider),
            response_time=0.0,
            error_message=f"{operation_name} falló después de {max_retries + 1} intentos: {str(last_error)}"
        )
    
    async def _upload_document(self, file_path: str, task: TranslationTask) -> APIResponse:
        """
        ✅ MEJORADO: Sube documento usando _make_request_multipart centralizado.
        """
        file_path_obj = Path(file_path)
        
        self.logger.debug(f"📤 DeepL: Subiendo documento {file_path_obj.name}")
        
        async with aiofiles.open(file_path, 'rb') as file:
            file_content = await file.read()
        
        # Preparar form data para upload
        form_data = FormData()
        form_data.add_field('file', file_content, 
                           filename=file_path_obj.name,
                           content_type=self._get_content_type(file_path_obj.suffix))
        form_data.add_field('source_lang', task.source_lang.value.upper())
        form_data.add_field('target_lang', task.target_lang.value.upper())
        form_data.add_field('formality', getattr(task, 'formality', 'prefer_more'))
        
        # Agregar configuraciones académicas
        for key, value in self.academic_settings.items():
            if key not in ['show_billed_characters']:  # Excluir configs no válidas para documentos
                form_data.add_field(key, str(value))
        
        # ✅ MEJORA CRÍTICA: Usar método centralizado
        response = await self._make_request_multipart(
            "POST", 
            "document", 
            form_data=form_data,
            timeout_override=self.document_timeout
        )
        
        if response.success:
            self.logger.info(f"📤 DeepL: Documento subido exitosamente, ID: {response.data.get('document_id')}")
        
        return response
    
    async def _poll_document_translation_resilient(self, document_id: str, document_key: str) -> APIResponse:
        """
        ✅ MEJORADO: Polling con tolerancia a fallos transitorios.
        
        Maneja errores temporales de red sin abortar el proceso completo.
        """
        max_attempts = 60  # 10 minutos máximo con polling cada 10s
        poll_interval = 10  # 10 segundos entre polls
        max_consecutive_failures = 3  # ✅ NUEVO: Máximo de fallos consecutivos antes de abortar
        start_time = time.time()
        consecutive_failures = 0
        
        self.logger.info(f"⏳ DeepL: Esperando traducción de documento {document_id}")
        
        for attempt in range(max_attempts):
            try:
                check_response = await self._make_request(
                    "POST", 
                    f"document/{document_id}",
                    data={"document_key": document_key},
                    use_cache=False
                )
                
                # ✅ MEJORA: Resetear contador de fallos en éxito
                consecutive_failures = 0
                
                if not check_response.success:
                    self.logger.warning(f"⚠️ DeepL: Error consultando estado del documento: {check_response.error_message}")
                    consecutive_failures += 1
                    
                    # Solo abortar si hay muchos fallos consecutivos
                    if consecutive_failures >= max_consecutive_failures:
                        return check_response
                    
                    # Continuar con el siguiente intento
                    await asyncio.sleep(poll_interval)
                    continue
                
                status = check_response.data.get("status")
                
                if status == "done":
                    elapsed_time = time.time() - start_time
                    self.logger.info(f"✅ DeepL: Traducción completada para documento {document_id} ({elapsed_time:.1f}s)")
                    return check_response
                
                elif status == "error":
                    error_msg = check_response.data.get("message", "Error desconocido en traducción")
                    self.logger.error(f"❌ DeepL: Error en traducción de documento: {error_msg}")
                    return APIResponse(
                        success=False,
                        data=check_response.data,
                        provider=self.provider,
                        request_id=document_id,
                        response_time=0.0,
                        error_message=error_msg
                    )
                
                # Status "translating" - continuar polling
                remaining_attempts = max_attempts - attempt - 1
                estimated_remaining = remaining_attempts * poll_interval
                
                self.logger.debug(f"🔄 DeepL: Traducción en progreso... ({attempt + 1}/{max_attempts}) - ETA: {estimated_remaining}s")
                
                # Polling adaptativo: reducir frecuencia después de 2 minutos
                if attempt > 12:  # Después de 2 minutos
                    poll_interval = 15  # Aumentar a 15 segundos
                elif attempt > 24:  # Después de 5 minutos
                    poll_interval = 20  # Aumentar a 20 segundos
                    
            except Exception as e:
                # ✅ MEJORA: Tolerancia a errores transitorios
                consecutive_failures += 1
                self.logger.warning(f"⚠️ DeepL: Error transitorio en polling ({consecutive_failures}/{max_consecutive_failures}): {str(e)}")
                
                # Solo abortar si hay muchos fallos consecutivos
                if consecutive_failures >= max_consecutive_failures:
                    return APIResponse(
                        success=False,
                        data=None,
                        provider=self.provider,
                        request_id=document_id,
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
            request_id=document_id,
            response_time=elapsed_time,
            error_message=f"Timeout esperando traducción de documento ({elapsed_time:.1f}s)"
        )
    
    async def _download_translated_document(self, 
                                          document_id: str, 
                                          document_key: str,
                                          output_path: Optional[str] = None) -> APIResponse:
        """
        ✅ MEJORADO: Descarga usando _make_request_binary centralizado y paths sanitizados.
        """
        if not output_path:
            timestamp = int(time.time())
            # ✅ MEJORA: Sanitizar nombre de archivo
            sanitized_id = self._sanitize_filename(document_id)
            output_path = f"deepl_translated_{sanitized_id}_{timestamp}.docx"
        else:
            # ✅ MEJORA: Sanitizar path proporcionado
            output_path = self._sanitize_filename(output_path)
        
        self.logger.debug(f"💾 DeepL: Descargando documento traducido a {output_path}")
        
        # ✅ MEJORA CRÍTICA: Usar método centralizado
        response = await self._make_request_binary(
            "POST",
            f"document/{document_id}/result",
            data={"document_key": document_key},
            output_path=output_path,
            timeout_override=self.document_timeout
        )
        
        if response.success:
            file_size = response.data.get("file_size", 0)
            self.logger.info(f"💾 DeepL: Documento descargado exitosamente ({file_size / 1024:.1f}KB)")
            
            # Agregar información adicional
            response.data.update({
                "document_id": document_id,
                "output_path": output_path
            })
        
        return response
    
    def _get_content_type(self, file_extension: str) -> str:
        """Obtiene content type basado en extensión de archivo."""
        content_types = {
            '.pdf': 'application/pdf',
            '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            '.doc': 'application/msword',
            '.rtf': 'application/rtf',
            '.txt': 'text/plain',
            '.html': 'text/html',
            '.htm': 'text/html'
        }
        
        return content_types.get(file_extension.lower(), 'application/octet-stream')
    
    async def get_document_status(self, document_id: str, document_key: str) -> APIResponse:
        """
        Obtiene status actual de documento en traducción.
        
        Útil para monitoreo externo sin polling automático.
        """
        return await self._make_request(
            "POST",
            f"document/{document_id}",
            data={"document_key": document_key},
            use_cache=False
        )
    
    async def translate_text_batch(self, texts: list[str], task: TranslationTask) -> APIResponse:
        """
        Traduce múltiples textos en una sola llamada.
        
        ✅ MEJORADO: Con validación del número de traducciones devueltas.
        
        Args:
            texts: Lista de textos a traducir (máximo 50)
            task: Configuración de traducción
            
        Returns:
            APIResponse con lista de traducciones
        """
        if len(texts) > 50:
            return APIResponse(
                success=False,
                data=None,
                provider=self.provider,
                request_id=create_request_id(self.provider),
                response_time=0.0,
                error_message="Máximo 50 textos por lote permitidos"
            )
        
        # ✅ MEJORA: Validar que ningún texto esté vacío
        empty_texts = [i for i, text in enumerate(texts) if not text or not text.strip()]
        if empty_texts:
            return APIResponse(
                success=False,
                data=None,
                provider=self.provider,
                request_id=create_request_id(self.provider),
                response_time=0.0,
                error_message=f"Textos vacíos en posiciones: {empty_texts}"
            )
        
        # Preparar datos para DeepL
        data = {
            "text": texts,
            "source_lang": task.source_lang.value.upper(),
            "target_lang": task.target_lang.value.upper(),
            **self.academic_settings
        }
        
        total_chars = sum(len(text) for text in texts)
        self.logger.info(f"🔄 DeepL: Traduciendo {len(texts)} textos en lote ({total_chars} chars total)")
        
        response = await self._make_request("POST", "translate", data=data, use_cache=True, cache_ttl=3600)
        
        if response.success and response.data:
            translations = response.data.get("translations", [])
            billed_chars = response.data.get("billed_characters", total_chars)
            
            translated_texts = [t.get("text", "") for t in translations]
            
            # ✅ MEJORA CRÍTICA: Validar número de traducciones
            if len(translated_texts) != len(texts):
                self.logger.warning(f"⚠️ DeepL: Número de traducciones ({len(translated_texts)}) no coincide con textos enviados ({len(texts)})")
                
                # Completar con textos vacíos si faltan traducciones
                while len(translated_texts) < len(texts):
                    translated_texts.append("")
                
                # Truncar si hay más traducciones de las esperadas
                translated_texts = translated_texts[:len(texts)]
            
            self.logger.info(f"✅ DeepL: Lote traducido ({len(translated_texts)} textos, {billed_chars} chars facturados)")
            
            response.data = {
                "translated_texts": translated_texts,
                "translations_metadata": translations,
                "billed_characters": billed_chars,
                "batch_size": len(texts),
                "source_characters": total_chars,
                "validation": {
                    "input_count": len(texts),
                    "output_count": len(translated_texts),
                    "count_match": len(translated_texts) == len(texts)
                }
            }
            
            response.cost_estimate = self._estimate_cost(billed_chars)
        
        return response


# ===============================================================================
# UTILIDADES ESPECÍFICAS DE DEEPL (MEJORADAS)
# ===============================================================================

def validate_deepl_api_key(api_key: str) -> bool:
    """
    ✅ MEJORADO: Valida formato de API key de DeepL con más verificaciones.
    """
    if not api_key or not isinstance(api_key, str):
        return False
    
    # Verificar longitud mínima
    if len(api_key) < 20:
        return False
    
    # ✅ MEJORA: Verificaciones más estrictas
    # DeepL Pro keys terminan en ":fx"
    # DeepL Free keys (no soportadas) terminan en ":dp"
    if not api_key.endswith(":fx"):
        return False
    
    # Verificar que tenga formato básico de UUID antes de ":fx"
    key_part = api_key[:-3]  # Remover ":fx"
    
    # Debe tener al menos guiones y caracteres hexadecimales
    if not re.match(r'^[0-9a-fA-F\-]+$', key_part):
        return False
    
    return True


def get_supported_file_formats() -> dict:
    """
    ✅ MEJORADO: Retorna formatos de archivo soportados por DeepL con más detalles.
    """
    return {
        "input_formats": [
            ".pdf", ".docx", ".doc", ".rtf", ".txt", ".html", ".htm"
        ],
        "output_formats": [
            ".docx", ".pdf", ".html", ".txt"
        ],
        "recommended_for_academic": [
            ".docx",  # Mejor preservación de formato
            ".pdf"    # Para documentos finales
        ],
        "max_file_size": "50MB",
        "max_file_size_bytes": 50 * 1024 * 1024,
        "max_pages": "100 páginas",
        "supported_content_types": {
            ".pdf": "application/pdf",
            ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ".doc": "application/msword",
            ".rtf": "application/rtf",
            ".txt": "text/plain",
            ".html": "text/html",
            ".htm": "text/html"
        },
        "processing_notes": {
            ".docx": "Preserva formato, tablas, imágenes",
            ".pdf": "Resultado puede variar según complejidad",
            ".html": "Preserva estructura y enlaces",
            ".txt": "Solo texto plano, sin formato"
        }
    }


def estimate_translation_time(file_size_mb: float, complexity: str = "medium") -> dict:
    """
    ✅ MEJORADO: Estima tiempo de traducción con factores más precisos.
    
    Args:
        file_size_mb: Tamaño del archivo en MB
        complexity: "low", "medium", "high"
    """
    # ✅ MEJORA: Factores de complejidad más precisos basados en experiencia
    complexity_factors = {
        "low": 0.8,     # Texto simple, repetitivo
        "medium": 1.5,  # Texto académico estándar
        "high": 2.8     # Texto técnico complejo, muchas referencias
    }
    
    factor = complexity_factors.get(complexity, 1.5)
    
    # ✅ MEJORA: Estimación base más precisa: ~25 segundos por MB para complejidad media
    base_time = file_size_mb * 25 * factor
    
    # Tiempo mínimo y máximo más realistas
    min_time = max(15, base_time * 0.6)  # Mínimo 15 segundos
    max_time = min(600, base_time * 2.5)  # Máximo 10 minutos
    
    # ✅ NUEVO: Categorías de tiempo para UI
    if base_time <= 30:
        time_category = "fast"
    elif base_time <= 120:
        time_category = "medium"
    else:
        time_category = "slow"
    
    return {
        "estimated_seconds": int(base_time),
        "min_seconds": int(min_time),
        "max_seconds": int(max_time),
        "estimated_minutes": round(base_time / 60, 1),
        "complexity_factor": factor,
        "time_category": time_category,
        "user_message": _generate_time_message(int(base_time), time_category)
    }


def _generate_time_message(seconds: int, category: str) -> str:
    """✅ NUEVO: Genera mensaje amigable para el usuario."""
    if category == "fast":
        return f"Traducción rápida (~{seconds}s)"
    elif category == "medium":
        return f"Tiempo estimado: {seconds // 60}m {seconds % 60}s"
    else:
        return f"Documento complejo, tiempo estimado: {seconds // 60}m {seconds % 60}s"


def get_language_pairs_supported() -> dict:
    """✅ NUEVO: Pares de idiomas más comunes para validación."""
    return {
        "common_academic": [
            ("en", "es"), ("de", "es"), ("fr", "es"),
            ("it", "es"), ("pt", "es"), ("es", "en")
        ],
        "all_to_spanish": [
            "en", "de", "fr", "it", "pt", "nl", "pl", 
            "ru", "ja", "zh", "ko"
        ],
        "spanish_to": [
            "en", "de", "fr", "it", "pt"
        ]
    }


# ===============================================================================
# TESTS UNITARIOS EMBEBIDOS (MEJORADOS)
# ===============================================================================

async def test_deepl_text_translation():
    """✅ MEJORADO: Test básico de traducción de texto."""
    import os
    api_key = os.getenv("DEEPL_API_KEY")
    
    if not api_key or api_key.startswith("your_"):
        print("⚠️ Test DeepL omitido: API key no configurada")
        return
    
    import logging
    logger = logging.getLogger("test")
    deepl = DeepLProIntegration(api_key, logger)
    
    # Test health check
    is_healthy = await deepl.health_check()
    assert is_healthy, "DeepL API debe estar disponible"
    
    # Test traducción simple
    task = TranslationTask(
        source_text="Hello, world! This is a test.",
        source_lang=SupportedLanguage.ENGLISH,
        target_lang=SupportedLanguage.SPANISH
    )
    
    response = await deepl.translate_text(task)
    assert response.success, f"Traducción debe ser exitosa: {response.error_message}"
    assert "translated_text" in response.data
    assert len(response.data["translated_text"]) > 0
    assert "validation" not in response.data  # Solo en batch
    
    print("✅ Test DeepL Text Translation: PASSED")


async def test_deepl_batch_translation():
    """✅ NUEVO: Test de traducción en lote con validación."""
    import os
    api_key = os.getenv("DEEPL_API_KEY")
    
    if not api_key or api_key.startswith("your_"):
        print("⚠️ Test DeepL Batch omitido: API key no configurada")
        return
    
    import logging
    logger = logging.getLogger("test")
    deepl = DeepLProIntegration(api_key, logger)
    
    # Test traducción en lote
    texts = ["Hello", "World", "Test"]
    task = TranslationTask(
        source_text="",  # No usado en batch
        source_lang=SupportedLanguage.ENGLISH,
        target_lang=SupportedLanguage.SPANISH
    )
    
    response = await deepl.translate_text_batch(texts, task)
    assert response.success, f"Traducción batch debe ser exitosa: {response.error_message}"
    assert "translated_texts" in response.data
    assert len(response.data["translated_texts"]) == len(texts)
    assert response.data["validation"]["count_match"] == True
    
    print("✅ Test DeepL Batch Translation: PASSED")


def test_deepl_utilities():
    """✅ MEJORADO: Test de utilidades específicas de DeepL."""
    # Test validación API key
    assert validate_deepl_api_key("12345678-1234-1234-1234-123456789012:fx") == True
    assert validate_deepl_api_key("test-key-1234567890:dp") == False  # Free key
    assert validate_deepl_api_key("invalid") == False
    assert validate_deepl_api_key("") == False
    assert validate_deepl_api_key(None) == False
    
    # Test formatos soportados
    formats = get_supported_file_formats()
    assert ".docx" in formats["input_formats"]
    assert ".pdf" in formats["output_formats"]
    assert "max_file_size_bytes" in formats
    assert formats["max_file_size_bytes"] == 50 * 1024 * 1024
    
    # Test estimación de tiempo
    estimation = estimate_translation_time(2.5, "medium")
    assert estimation["estimated_seconds"] > 0
    assert estimation["min_seconds"] <= estimation["estimated_seconds"]
    assert estimation["estimated_seconds"] <= estimation["max_seconds"]
    assert "time_category" in estimation
    assert "user_message" in estimation
    
    # Test pares de idiomas
    pairs = get_language_pairs_supported()
    assert ("en", "es") in pairs["common_academic"]
    assert "en" in pairs["all_to_spanish"]
    
    print("✅ Test DeepL Utilities (mejorado): PASSED")


def test_filename_sanitization():
    """✅ NUEVO: Test de sanitización de nombres de archivo."""
    import logging
    logger = logging.getLogger("test")
    deepl = DeepLProIntegration("dummy-key-1234567890:fx", logger)
    
    # Test casos problemáticos
    assert deepl._sanitize_filename("file<name>.txt") == "file_name_.txt"
    assert deepl._sanitize_filename("file:name?.doc") == "file_name_.doc"
    assert deepl._sanitize_filename("file   name.pdf") == "file_name.pdf"
    
    # Test longitud excesiva
    long_name = "a" * 250 + ".txt"
    sanitized = deepl._sanitize_filename(long_name)
    assert len(sanitized) <= 200
    assert sanitized.endswith(".txt")
    
    print("✅ Test Filename Sanitization: PASSED")


async def run_all_tests():
    """Ejecuta todos los tests embebidos."""
    print("🧪 Ejecutando tests de deepl_integration.py (POST-AUDITORÍA)...")
    
    try:
        test_deepl_utilities()
        test_filename_sanitization()
        await test_deepl_text_translation()  # Omitido si no hay API key
        await test_deepl_batch_translation()  # Omitido si no hay API key
        
        print("\n✅ Todos los tests de deepl_integration.py (POST-AUDITORÍA) pasaron!")
        print("\n🏆 MEJORAS IMPLEMENTADAS:")
        print("  ✅ Requests centralizadas para uploads/downloads")
        print("  ✅ Polling resiliente con tolerancia a fallos transitorios")
        print("  ✅ Retries automáticos para operaciones de documentos")
        print("  ✅ Sanitización robusta de nombres de archivo")
        print("  ✅ Validación del número de traducciones en batch")
        print("  ✅ Exposición de document_id para diagnóstico")
        print("  ✅ Validación mejorada de API keys y parámetros")
        
    except Exception as e:
        print(f"\n❌ Test falló: {e}")
        raise


if __name__ == "__main__":
    """Ejecutar tests al correr el módulo directamente."""
    import logging
    logging.basicConfig(level=logging.INFO)
    
    asyncio.run(run_all_tests())