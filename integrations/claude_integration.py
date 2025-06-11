#!/usr/bin/env python3
"""
🧠 CLAUDE_INTEGRATION.PY - Integración Claude (Anthropic) API
Sistema de Traducción Académica v2.2 - APIs Integration Layer
POST-AUDITORÍA: Versión mejorada con correcciones críticas

Integración especializada con Claude API para análisis terminológico
académico y refinamiento de traducciones especializadas.

Características:
✅ Análisis terminológico especializado por disciplina
✅ Refinamiento de traducciones técnicas y académicas
✅ Generación de glosarios bilingües estructurados
✅ Prompts optimizados para cada área de conocimiento
✅ Sugerencias contextuales inteligentes
✅ Parsing robusto con múltiples fallbacks
✅ Validación estructural estricta de respuestas JSON
✅ Modelo configurable con fallbacks automáticos

Autor: Sistema ClaudeAcademico v2.2
Fecha: Enero 2025 (Post-Auditoría)
Ubicación: integrations/claude_integration.py
"""

import json
import re
from typing import List, Optional, Dict, Any, Union
from dataclasses import asdict

from .base_client import BaseAPIClient, create_rate_limiter
from .models import (
    APIProvider, APIResponse, TerminologySuggestion, AcademicDiscipline,
    SupportedLanguage, Logger, CacheManager, ErrorPolicyManager,
    ACADEMIC_CONTEXTS, get_language_name, create_request_id
)


# ===============================================================================
# CLAUDE (ANTHROPIC) API INTEGRATION (MEJORADA)
# ===============================================================================

class ClaudeAPIIntegration(BaseAPIClient):
    """
    Integración con Claude (Anthropic) API para análisis terminológico académico.
    
    ✅ POST-AUDITORÍA: Versión mejorada con robustez enterprise-grade.
    
    Claude excela en análisis de texto académico y contextual, siendo ideal para:
    - Identificación de terminología técnica especializada
    - Análisis semántico profundo de conceptos académicos
    - Generación de glosarios bilingües contextualizados
    - Refinamiento de traducciones con precisión disciplinaria
    
    Características Enterprise:
    ✅ Prompts especializados por disciplina académica
    ✅ Análisis contextual profundo de terminología
    ✅ Generación de glosarios estructurados JSON
    ✅ Modelo configurable con fallbacks automáticos
    ✅ Parsing robusto con validación estructural
    ✅ Rate limiting automático (200 req/min)
    ✅ Fallbacks inteligentes y recuperación de errores
    ✅ Métricas detalladas de uso y performance
    """
    
    def __init__(self,
                 api_key: str,
                 logger: Logger,
                 cache_manager: Optional[CacheManager] = None,
                 error_policy_manager: Optional[ErrorPolicyManager] = None,
                 model: Optional[str] = None,  # ✅ NUEVO: Modelo configurable
                 fallback_model: Optional[str] = None):  # ✅ NUEVO: Modelo de fallback
        
        # ✅ VALIDACIÓN: API key de Claude
        if not validate_claude_api_key(api_key):
            raise ValueError("API key de Claude inválida (debe empezar con 'sk-ant-')")
        
        # Rate limiter específico de Claude: 200 requests/minuto
        rate_limiter = create_rate_limiter(APIProvider.CLAUDE, logger)
        
        super().__init__(
            api_key=api_key,
            base_url="https://api.anthropic.com/v1/",
            provider=APIProvider.CLAUDE,
            logger=logger,
            cache_manager=cache_manager,
            error_policy_manager=error_policy_manager,
            rate_limiter=rate_limiter
        )
        
        # Headers específicos de Anthropic
        self.headers.update({
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json"
        })
        
        # ✅ MEJORA CRÍTICA: Modelo configurable con fallbacks
        self.primary_model = model or "claude-3-sonnet-20240229"
        self.fallback_model = fallback_model or "claude-3-haiku-20240307"
        self.current_model = self.primary_model
        
        # Configuración para análisis académico
        self.academic_config = {
            "temperature": 0.3,  # Más conservador para terminología
            "max_tokens": 3000,
            "top_p": 0.9
        }
        
        # ✅ NUEVO: Configuraciones específicas por tarea
        self.task_configs = {
            "terminology": {"temperature": 0.3, "max_tokens": 3000},
            "refinement": {"temperature": 0.2, "max_tokens": 3000},
            "glossary": {"temperature": 0.1, "max_tokens": 4000},
            "quality": {"temperature": 0.2, "max_tokens": 3000}
        }
    
    def _estimate_cost(self, characters: int) -> float:
        """
        Estima costo Claude basado en tokens.
        
        Claude pricing: $3/1M tokens input, $15/1M tokens output
        Estimación: 4 chars ≈ 1 token, promedio input/output 50/50
        """
        estimated_tokens = characters / 4
        input_cost = (estimated_tokens / 1_000_000) * 3.0
        output_cost = (estimated_tokens * 0.5 / 1_000_000) * 15.0  # Output típicamente menor
        return input_cost + output_cost
    
    # ✅ MEJORA CRÍTICA: Método centralizado para llamadas a Claude
    async def _call_claude(self,
                         prompt: str,
                         task_type: str = "general",
                         use_fallback_on_error: bool = True,
                         cache_ttl: int = 3600,
                         use_cache: bool = True) -> APIResponse:
        """
        Método centralizado para todas las llamadas a Claude API.
        
        Args:
            prompt: Prompt a enviar a Claude
            task_type: Tipo de tarea ("terminology", "refinement", "glossary", "quality")
            use_fallback_on_error: Si usar modelo fallback en caso de error
            cache_ttl: TTL para cache
            use_cache: Si usar cache
            
        Returns:
            APIResponse con respuesta de Claude
        """
        # Obtener configuración específica de la tarea
        config = self.task_configs.get(task_type, self.academic_config)
        
        # Preparar datos para la request
        data = {
            "model": self.current_model,
            "messages": [{"role": "user", "content": prompt}],
            **config
        }
        
        self.logger.debug(f"🧠 Claude: Llamada {task_type} con modelo {self.current_model}")
        
        response = await self._make_request(
            "POST",
            "messages",
            data=data,
            cache_ttl=cache_ttl,
            use_cache=use_cache
        )
        
        # ✅ MEJORA: Si falla y tenemos fallback, intentar con modelo alternativo
        if not response.success and use_fallback_on_error and self.current_model != self.fallback_model:
            self.logger.warning(f"⚠️ Claude: Error con {self.current_model}, intentando fallback {self.fallback_model}")
            
            # Temporalmente cambiar a modelo fallback
            original_model = self.current_model
            self.current_model = self.fallback_model
            
            try:
                data["model"] = self.fallback_model
                response = await self._make_request(
                    "POST",
                    "messages",
                    data=data,
                    cache_ttl=cache_ttl,
                    use_cache=False  # No cachear fallbacks
                )
                
                if response.success:
                    self.logger.info(f"✅ Claude: Fallback exitoso con {self.fallback_model}")
                else:
                    self.logger.error(f"❌ Claude: Fallback también falló con {self.fallback_model}")
                
            finally:
                # Restaurar modelo original
                self.current_model = original_model
        
        return response
    
    def _extract_claude_content(self, response_data: Dict[str, Any]) -> str:
        """
        ✅ NUEVO: Extrae contenido de respuesta Claude de forma robusta.
        
        Args:
            response_data: Datos de respuesta de Claude API
            
        Returns:
            Contenido de texto extraído
        """
        try:
            # Formato estándar de Claude
            content = response_data.get("content", [])
            if content and isinstance(content, list) and len(content) > 0:
                return content[0].get("text", "")
            
            # Fallback: buscar texto directamente
            if "text" in response_data:
                return response_data["text"]
            
            # Último fallback: convertir a string
            return str(response_data)
            
        except Exception as e:
            self.logger.error(f"❌ Claude: Error extrayendo contenido: {e}")
            return ""
    
    async def health_check(self) -> bool:
        """Verifica salud de Claude API con test simple."""
        try:
            test_response = await self.analyze_terminology(
                text_sample="Test health check for academic terminology analysis.",
                discipline=AcademicDiscipline.GENERAL,
                source_lang=SupportedLanguage.ENGLISH,
                max_terms=1
            )
            
            # Considerar exitoso si Claude responde, incluso con errores de parsing
            if test_response.success or "analysis" in str(test_response.data):
                self.logger.info("✅ Claude: Health check exitoso")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"❌ Claude health check falló: {e}")
            return False
    
    async def analyze_terminology(self,
                                text_sample: str,
                                discipline: AcademicDiscipline,
                                source_lang: SupportedLanguage,
                                max_terms: int = 15) -> APIResponse:
        """
        Analiza texto académico para sugerir términos para glosario.
        
        ✅ POST-AUDITORÍA: Con validación estructural y parsing robusto.
        
        Args:
            text_sample: Muestra de texto para analizar (máx 3000 chars)
            discipline: Disciplina académica para especialización
            source_lang: Idioma del texto de origen
            max_terms: Máximo número de términos a sugerir
            
        Returns:
            APIResponse con lista de TerminologySuggestion
        """
        # ✅ MEJORA: Validación más robusta
        if not text_sample or not text_sample.strip():
            return APIResponse(
                success=False,
                data=None,
                provider=self.provider,
                request_id=create_request_id(self.provider),
                response_time=0.0,
                error_message="text_sample no puede estar vacío"
            )
        
        # Validar max_terms
        if max_terms <= 0 or max_terms > 50:
            max_terms = min(max(max_terms, 1), 50)
            self.logger.warning(f"⚠️ Claude: max_terms ajustado a {max_terms}")
        
        if len(text_sample) > 3000:
            text_sample = text_sample[:3000] + "..."
            self.logger.warning(f"⚠️ Claude: Texto truncado a 3000 caracteres")
        
        context = ACADEMIC_CONTEXTS.get(discipline, ACADEMIC_CONTEXTS[AcademicDiscipline.GENERAL])
        lang_name = get_language_name(source_lang)
        
        prompt = self._create_terminology_prompt(text_sample, context, lang_name, max_terms)
        
        self.logger.info(f"🔍 Claude: Analizando terminología {discipline.value} ({source_lang.value})")
        
        response = await self._call_claude(
            prompt=prompt,
            task_type="terminology",
            cache_ttl=7200  # Cache por 2 horas
        )
        
        if response.success:
            try:
                # Extraer contenido de la respuesta Claude
                claude_text = self._extract_claude_content(response.data)
                
                if not claude_text:
                    raise ValueError("Respuesta vacía de Claude")
                
                # ✅ MEJORA CRÍTICA: Parsear respuesta con validación estructural
                terminology_data = self._parse_terminology_response_robust(claude_text, discipline)
                
                # Crear análisis adicional
                analysis_summary = self._create_analysis_summary(terminology_data)
                
                self.logger.info(f"✅ Claude: {len(terminology_data)} términos sugeridos para {discipline.value}")
                
                response.data = {
                    "suggestions": [asdict(suggestion) for suggestion in terminology_data],
                    "discipline": discipline.value,
                    "source_language": source_lang.value,
                    "analysis_summary": analysis_summary,
                    "raw_response": claude_text,
                    "model_used": self.current_model,
                    "validation_passed": True
                }
                
            except Exception as e:
                self.logger.error(f"❌ Claude: Error parseando respuesta terminológica: {e}")
                
                # ✅ MEJORA: Fallback mejorado
                claude_text = self._extract_claude_content(response.data)
                fallback_terms = self._manual_parse_terminology_enhanced(claude_text, discipline)
                
                response.data = {
                    "suggestions": [asdict(term) for term in fallback_terms],
                    "discipline": discipline.value,
                    "source_language": source_lang.value,
                    "analysis_summary": {
                        "parsing_method": "enhanced_fallback", 
                        "total_terms": len(fallback_terms),
                        "confidence_average": sum(t.confidence for t in fallback_terms) / len(fallback_terms) if fallback_terms else 0.0
                    },
                    "raw_response": claude_text,
                    "parsing_error": str(e),
                    "validation_passed": False
                }
                
                if not fallback_terms:
                    response.success = False
                    response.error_message = f"Error parseando terminología: {e}"
        
        return response
    
    async def refine_translation(self,
                               original_text: str,
                               translated_text: str,
                               discipline: AcademicDiscipline,
                               source_lang: SupportedLanguage) -> APIResponse:
        """
        Refina traducción para mejorar precisión académica.
        
        ✅ POST-AUDITORÍA: Con método centralizado y validación robusta.
        """
        # Validar inputs
        if not original_text or not translated_text:
            return APIResponse(
                success=False,
                data=None,
                provider=self.provider,
                request_id=create_request_id(self.provider),
                response_time=0.0,
                error_message="original_text y translated_text son requeridos"
            )
        
        # Truncar textos si son muy largos
        if len(original_text) > 1500:
            original_text = original_text[:1500] + "..."
            self.logger.warning("⚠️ Claude: original_text truncado a 1500 caracteres")
        if len(translated_text) > 1500:
            translated_text = translated_text[:1500] + "..."
            self.logger.warning("⚠️ Claude: translated_text truncado a 1500 caracteres")
        
        context = ACADEMIC_CONTEXTS.get(discipline, ACADEMIC_CONTEXTS[AcademicDiscipline.GENERAL])
        lang_name = get_language_name(source_lang)
        
        prompt = self._create_refinement_prompt(original_text, translated_text, context, lang_name)
        
        self.logger.info(f"🔧 Claude: Refinando traducción {discipline.value}")
        
        response = await self._call_claude(
            prompt=prompt,
            task_type="refinement",
            use_cache=False  # No cachear refinamientos específicos
        )
        
        if response.success:
            try:
                claude_text = self._extract_claude_content(response.data)
                refinement_data = self._parse_refinement_response_robust(claude_text)
                
                improvements_count = len(refinement_data.get("mejoras_aplicadas", []))
                self.logger.info(f"✅ Claude: Traducción refinada con {improvements_count} mejoras")
                
                response.data = {
                    **refinement_data,
                    "original_length": len(original_text),
                    "translation_length": len(translated_text),
                    "discipline": discipline.value,
                    "model_used": self.current_model,
                    "raw_response": claude_text
                }
                
            except Exception as e:
                self.logger.error(f"❌ Claude: Error parseando refinamiento: {e}")
                response.success = False
                response.error_message = f"Error parseando refinamiento: {e}"
        
        return response
    
    async def generate_glossary_entries(self,
                                      terms_list: List[str],
                                      discipline: AcademicDiscipline,
                                      source_lang: SupportedLanguage) -> APIResponse:
        """
        Genera entradas de glosario para lista de términos.
        
        ✅ POST-AUDITORÍA: Con validación de entrada mejorada.
        """
        if not terms_list or len(terms_list) == 0:
            return APIResponse(
                success=False,
                data=None,
                provider=self.provider,
                request_id=create_request_id(self.provider),
                response_time=0.0,
                error_message="Lista de términos vacía"
            )
        
        # ✅ MEJORA: Filtrar términos vacíos y duplicados
        clean_terms = list(set([term.strip() for term in terms_list if term and term.strip()]))
        
        if not clean_terms:
            return APIResponse(
                success=False,
                data=None,
                provider=self.provider,
                request_id=create_request_id(self.provider),
                response_time=0.0,
                error_message="Todos los términos están vacíos después de limpieza"
            )
        
        # Limitar a 20 términos para evitar respuestas muy largas
        if len(clean_terms) > 20:
            clean_terms = clean_terms[:20]
            self.logger.warning(f"⚠️ Claude: Lista de términos truncada a 20 elementos")
        
        context = ACADEMIC_CONTEXTS.get(discipline, ACADEMIC_CONTEXTS[AcademicDiscipline.GENERAL])
        lang_name = get_language_name(source_lang)
        terms_str = "\n".join([f"- {term}" for term in clean_terms])
        
        prompt = self._create_glossary_prompt(terms_str, context, lang_name)
        
        self.logger.info(f"📖 Claude: Generando glosario {discipline.value} ({len(clean_terms)} términos)")
        
        response = await self._call_claude(
            prompt=prompt,
            task_type="glossary",
            cache_ttl=86400  # Cache glosarios por 24 horas
        )
        
        if response.success:
            try:
                claude_text = self._extract_claude_content(response.data)
                glossary_data = self._parse_glossary_response_robust(claude_text, discipline)
                
                entries_count = len(glossary_data.get("glosario", []))
                self.logger.info(f"✅ Claude: Glosario generado con {entries_count} entradas")
                
                response.data = {
                    **glossary_data,
                    "source_terms_count": len(clean_terms),
                    "source_terms_original": len(terms_list),
                    "discipline": discipline.value,
                    "source_language": source_lang.value,
                    "model_used": self.current_model,
                    "raw_response": claude_text
                }
                
            except Exception as e:
                self.logger.error(f"❌ Claude: Error parseando glosario: {e}")
                response.success = False
                response.error_message = f"Error parseando glosario: {e}"
        
        return response
    
    async def validate_translation_quality(self,
                                         original_text: str,
                                         translated_text: str,
                                         discipline: AcademicDiscipline) -> APIResponse:
        """
        Valida calidad de traducción académica y sugiere mejoras.
        
        ✅ POST-AUDITORÍA: Con método centralizado.
        """
        if not original_text or not translated_text:
            return APIResponse(
                success=False,
                data=None,
                provider=self.provider,
                request_id=create_request_id(self.provider),
                response_time=0.0,
                error_message="original_text y translated_text son requeridos"
            )
        
        context = ACADEMIC_CONTEXTS.get(discipline, ACADEMIC_CONTEXTS[AcademicDiscipline.GENERAL])
        
        prompt = self._create_quality_validation_prompt(original_text, translated_text, context, discipline)
        
        self.logger.info(f"📊 Claude: Validando calidad de traducción {discipline.value}")
        
        response = await self._call_claude(
            prompt=prompt,
            task_type="quality",
            use_cache=False
        )
        
        if response.success:
            try:
                claude_text = self._extract_claude_content(response.data)
                quality_data = self._parse_quality_response_robust(claude_text)
                
                quality_score = quality_data.get("calidad_general", 0.0)
                self.logger.info(f"✅ Claude: Análisis de calidad completado (score: {quality_score:.2f})")
                
                response.data = {
                    **quality_data,
                    "discipline": discipline.value,
                    "text_lengths": {
                        "original": len(original_text),
                        "translated": len(translated_text)
                    },
                    "model_used": self.current_model,
                    "raw_response": claude_text
                }
                
            except Exception as e:
                self.logger.error(f"❌ Claude: Error parseando análisis de calidad: {e}")
                response.success = False
                response.error_message = f"Error parseando análisis de calidad: {e}"
        
        return response
    
    # ===============================================================================
    # MÉTODOS DE CREACIÓN DE PROMPTS (SIN CAMBIOS SIGNIFICATIVOS)
    # ===============================================================================
    
    def _create_terminology_prompt(self, text: str, context: str, lang_name: str, max_terms: int) -> str:
        """Crea prompt especializado para análisis terminológico."""
        return f"""
Eres un experto terminólogo académico especializado en análisis de textos especializados.

TAREA: Analizar el siguiente texto en {lang_name} e identificar términos técnicos clave para crear un glosario bilingüe {lang_name}-español.

CONTEXTO DISCIPLINARIO: {context}

TEXTO A ANALIZAR:
{text}

CRITERIOS DE SELECCIÓN DE TÉRMINOS:
1. Términos técnicos específicos de la disciplina
2. Conceptos que requieren consistencia terminológica
3. Palabras que pueden tener múltiples traducciones según contexto
4. Términos frecuentes en textos académicos de este campo
5. Conceptos que pueden ser malinterpretados si se traducen incorrectamente

FORMATO DE RESPUESTA (JSON válido):
{{
    "terminos_sugeridos": [
        {{
            "termino_original": "Dasein",
            "traduccion_sugerida": "ser-ahí",
            "contexto": "filosofía heideggeriana, análisis existencial",
            "justificacion": "Término técnico fundamental que requiere consistencia",
            "confianza": 0.95,
            "prioridad": "alta",
            "frecuencia_estimada": "media",
            "alternativas": ["existencia", "ser-en-el-mundo"]
        }}
    ],
    "resumen_analisis": {{
        "total_terminos_identificados": {max_terms},
        "disciplinas_detectadas": ["filosofía", "ontología"],
        "nivel_especializacion": "alto",
        "recomendaciones": "Priorizar términos fenomenológicos y ontológicos"
    }}
}}

IMPORTANTE: 
- Máximo {max_terms} términos más relevantes
- Responder SOLO con JSON válido, sin texto adicional
- Incluir justificación académica para cada término
"""
    
    def _create_refinement_prompt(self, original: str, translation: str, context: str, lang_name: str) -> str:
        """Crea prompt para refinamiento de traducción."""
        return f"""
Eres un experto traductor académico especializado en {context}.

TAREA: Refinar la siguiente traducción académica de {lang_name} al español, mejorando la precisión terminológica y el estilo académico.

CONTEXTO DISCIPLINARIO: {context}

TEXTO ORIGINAL ({lang_name}):
{original}

TRADUCCIÓN INICIAL (Español):
{translation}

INSTRUCCIONES DE REFINAMIENTO:
1. Identificar términos técnicos que pueden mejorarse
2. Proponer alternativas más precisas para términos especializados
3. Mantener el estilo académico formal apropiado
4. Preservar el significado y estructura originales
5. Asegurar consistencia terminológica

FORMATO DE RESPUESTA (JSON válido):
{{
    "traduccion_refinada": "versión mejorada de la traducción completa",
    "mejoras_aplicadas": [
        {{
            "posicion": "párrafo 1",
            "original": "término original",
            "inicial": "traducción inicial",
            "refinada": "traducción mejorada", 
            "justificacion": "razón académica específica de la mejora",
            "tipo_mejora": "terminología|estilo|precisión|fluidez"
        }}
    ],
    "calidad_mejora": {{
        "confianza_general": 0.95,
        "areas_mejoradas": ["terminología especializada", "fluidez académica"],
        "riesgo_cambios": "bajo"
    }},
    "observaciones": "comentarios adicionales sobre la traducción y el proceso de refinamiento"
}}

IMPORTANTE: Responder SOLO con JSON válido, sin texto adicional.
"""
    
    def _create_glossary_prompt(self, terms_str: str, context: str, lang_name: str) -> str:
        """Crea prompt para generación de glosario."""
        return f"""
Eres un terminólogo experto en {context} especializado en glosarios académicos bilingües.

TAREA: Crear entradas de glosario completas para los siguientes términos de {lang_name} al español.

CONTEXTO DISCIPLINARIO: {context}

TÉRMINOS A PROCESAR:
{terms_str}

INSTRUCCIONES:
1. Para cada término, proporcionar la mejor traducción al español académico
2. Incluir contexto disciplinario específico y definición breve
3. Indicar nivel de confianza en la traducción
4. Sugerir términos relacionados cuando sea relevante
5. Identificar casos donde se requiere nota explicativa

FORMATO DE RESPUESTA (JSON válido):
{{
    "glosario": [
        {{
            "termino_original": "Begriff",
            "traduccion_principal": "concepto",
            "traducciones_alternativas": ["noción", "idea"],
            "contexto_disciplinario": "filosofía alemana, especialmente en Hegel y Kant",
            "definicion_breve": "unidad de pensamiento que representa algo universal",
            "nota_explicativa": "En Hegel, se refiere específicamente al concepto que se desarrolla dialécticamente",
            "confianza": 0.95,
            "frecuencia_estimada": "alta",
            "terminos_relacionados": ["Vorstellung", "Gedanke", "Idee"],
            "requiere_contexto": true
        }}
    ],
    "estadisticas": {{
        "total_terminos": 5,
        "confianza_promedio": 0.92,
        "terminos_alta_confianza": 4,
        "terminos_revision_necesaria": 1,
        "areas_tematicas": ["ontología", "epistemología"]
    }},
    "recomendaciones_uso": [
        "Mantener consistencia en traducción de términos técnicos",
        "Incluir notas explicativas para términos con múltiples acepciones"
    ]
}}

IMPORTANTE: Responder SOLO con JSON válido, sin texto adicional.
"""
    
    def _create_quality_validation_prompt(self, original: str, translated: str, context: str, discipline: AcademicDiscipline) -> str:
        """✅ NUEVO: Crea prompt optimizado para validación de calidad."""
        return f"""
Eres un experto en traducción académica especializado en {discipline.value}.

TAREA: Evaluar la calidad de la siguiente traducción académica y proporcionar análisis detallado.

CONTEXTO DISCIPLINARIO: {context}

TEXTO ORIGINAL:
{original[:1000]}

TRADUCCIÓN A EVALUAR:
{translated[:1000]}

CRITERIOS DE EVALUACIÓN:
1. Precisión terminológica especializada
2. Preservación del significado académico
3. Fluidez y naturalidad en español académico
4. Consistencia terminológica
5. Adecuación al registro académico

FORMATO DE RESPUESTA (JSON válido):
{{
    "calidad_general": 0.85,
    "criterios_evaluacion": {{
        "precision_terminologica": 0.90,
        "preservacion_significado": 0.85,
        "fluidez_academica": 0.80,
        "consistencia_terminologica": 0.85,
        "registro_academico": 0.90
    }},
    "fortalezas": [
        "Excelente manejo de terminología filosófica",
        "Preservación del registro académico formal"
    ],
    "areas_mejora": [
        "Inconsistencia en traducción de 'Begriff'",
        "Algunas construcciones poco naturales"
    ],
    "sugerencias_especificas": [
        {{
            "fragmento_original": "texto problemático",
            "traduccion_actual": "traducción actual", 
            "mejora_sugerida": "mejora propuesta",
            "justificacion": "razón de la mejora"
        }}
    ],
    "recomendacion_general": "Traducción de buena calidad que requiere ajustes menores en terminología específica"
}}

IMPORTANTE: Responder SOLO con JSON válido, sin texto adicional.
"""
    
    # ===============================================================================
    # MÉTODOS DE PARSING ROBUSTOS (MEJORADOS)
    # ===============================================================================
    
    def _parse_terminology_response_robust(self, claude_text: str, discipline: AcademicDiscipline) -> List[TerminologySuggestion]:
        """
        ✅ MEJORA CRÍTICA: Parsea respuesta con validación estructural estricta.
        """
        try:
            # Intentar extraer JSON de la respuesta
            json_text = self._extract_json_from_text_robust(claude_text)
            
            if json_text:
                data = json.loads(json_text)
                
                # ✅ VALIDACIÓN ESTRUCTURAL: Verificar claves requeridas
                if not self._validate_terminology_structure(data):
                    raise ValueError("Estructura JSON de terminología inválida")
                
                suggestions = []
                for term_data in data.get("terminos_sugeridos", []):
                    # ✅ VALIDACIÓN: Campos mínimos requeridos
                    if not all(key in term_data for key in ["termino_original", "traduccion_sugerida"]):
                        self.logger.warning(f"⚠️ Claude: Término incompleto ignorado: {term_data}")
                        continue
                    
                    suggestion = TerminologySuggestion(
                        source_term=term_data.get("termino_original", ""),
                        target_term=term_data.get("traduccion_sugerida", ""),
                        context=term_data.get("contexto", ""),
                        discipline=discipline,
                        confidence=float(term_data.get("confianza", 0.5)),
                        justification=term_data.get("justificacion", ""),
                        priority=term_data.get("prioridad", "medium"),
                        frequency_estimate=term_data.get("frecuencia_estimada", "unknown")
                    )
                    suggestions.append(suggestion)
                
                return suggestions
            
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            self.logger.warning(f"⚠️ Claude: Error parseando JSON terminológico, usando fallback: {e}")
        
        # Fallback: parsing manual mejorado
        return self._manual_parse_terminology_enhanced(claude_text, discipline)
    
    def _parse_refinement_response_robust(self, claude_text: str) -> dict:
        """✅ MEJORA CRÍTICA: Parsea refinamiento con validación."""
        try:
            json_text = self._extract_json_from_text_robust(claude_text)
            if json_text:
                data = json.loads(json_text)
                
                # ✅ VALIDACIÓN ESTRUCTURAL
                if not self._validate_refinement_structure(data):
                    raise ValueError("Estructura JSON de refinamiento inválida")
                
                return data
                
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            self.logger.warning(f"⚠️ Claude: Error parseando refinamiento: {e}")
        
        # Fallback básico mejorado
        return {
            "traduccion_refinada": "Error parseando refinamiento - revisar respuesta manual",
            "mejoras_aplicadas": [],
            "calidad_mejora": {"confianza_general": 0.0, "areas_mejoradas": [], "riesgo_cambios": "alto"},
            "observaciones": f"Error en parsing automático: {claude_text[:200]}..."
        }
    
    def _parse_glossary_response_robust(self, claude_text: str, discipline: AcademicDiscipline) -> dict:
        """✅ MEJORA CRÍTICA: Parsea glosario con validación."""
        try:
            json_text = self._extract_json_from_text_robust(claude_text)
            if json_text:
                data = json.loads(json_text)
                
                # ✅ VALIDACIÓN ESTRUCTURAL
                if not self._validate_glossary_structure(data):
                    raise ValueError("Estructura JSON de glosario inválida")
                
                return data
                
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            self.logger.warning(f"⚠️ Claude: Error parseando glosario: {e}")
        
        # Fallback básico mejorado
        return {
            "glosario": [],
            "estadisticas": {
                "total_terminos": 0,
                "confianza_promedio": 0.0,
                "terminos_alta_confianza": 0,
                "terminos_revision_necesaria": 0,
                "areas_tematicas": [discipline.value]
            },
            "recomendaciones_uso": ["Revisión manual necesaria debido a error de parsing"]
        }
    
    def _parse_quality_response_robust(self, claude_text: str) -> dict:
        """✅ MEJORA CRÍTICA: Parsea análisis de calidad con validación."""
        try:
            json_text = self._extract_json_from_text_robust(claude_text)
            if json_text:
                data = json.loads(json_text)
                
                # ✅ VALIDACIÓN ESTRUCTURAL
                if not self._validate_quality_structure(data):
                    raise ValueError("Estructura JSON de calidad inválida")
                
                return data
                
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            self.logger.warning(f"⚠️ Claude: Error parseando calidad: {e}")
        
        # Fallback básico
        return {
            "calidad_general": 0.0,
            "criterios_evaluacion": {
                "precision_terminologica": 0.0,
                "preservacion_significado": 0.0,
                "fluidez_academica": 0.0,
                "consistencia_terminologica": 0.0,
                "registro_academico": 0.0
            },
            "fortalezas": [],
            "areas_mejora": ["Error en análisis automático"],
            "sugerencias_especificas": [],
            "recomendacion_general": "Revisión manual necesaria debido a error de parsing"
        }
    
    # ===============================================================================
    # MÉTODOS DE VALIDACIÓN ESTRUCTURAL (NUEVOS)
    # ===============================================================================
    
    def _validate_terminology_structure(self, data: dict) -> bool:
        """✅ NUEVO: Valida estructura de respuesta terminológica."""
        required_keys = ["terminos_sugeridos"]
        
        if not all(key in data for key in required_keys):
            return False
        
        # Validar que terminos_sugeridos sea una lista
        if not isinstance(data["terminos_sugeridos"], list):
            return False
        
        # Validar estructura de cada término
        for term in data["terminos_sugeridos"]:
            if not isinstance(term, dict):
                return False
            if not all(key in term for key in ["termino_original", "traduccion_sugerida"]):
                return False
        
        return True
    
    def _validate_refinement_structure(self, data: dict) -> bool:
        """✅ NUEVO: Valida estructura de respuesta de refinamiento."""
        required_keys = ["traduccion_refinada", "mejoras_aplicadas", "calidad_mejora"]
        
        if not all(key in data for key in required_keys):
            return False
        
        # Validar tipos
        if not isinstance(data["mejoras_aplicadas"], list):
            return False
        if not isinstance(data["calidad_mejora"], dict):
            return False
        
        return True
    
    def _validate_glossary_structure(self, data: dict) -> bool:
        """✅ NUEVO: Valida estructura de respuesta de glosario."""
        required_keys = ["glosario", "estadisticas"]
        
        if not all(key in data for key in required_keys):
            return False
        
        # Validar tipos
        if not isinstance(data["glosario"], list):
            return False
        if not isinstance(data["estadisticas"], dict):
            return False
        
        return True
    
    def _validate_quality_structure(self, data: dict) -> bool:
        """✅ NUEVO: Valida estructura de respuesta de calidad."""
        required_keys = ["calidad_general", "criterios_evaluacion", "fortalezas", "areas_mejora"]
        
        if not all(key in data for key in required_keys):
            return False
        
        # Validar tipos y rangos
        if not isinstance(data["calidad_general"], (int, float)) or not (0 <= data["calidad_general"] <= 1):
            return False
        
        if not isinstance(data["criterios_evaluacion"], dict):
            return False
        
        if not isinstance(data["fortalezas"], list) or not isinstance(data["areas_mejora"], list):
            return False
        
        return True
    
    def _extract_json_from_text_robust(self, text: str) -> Optional[str]:
        """
        ✅ MEJORA CRÍTICA: Extrae JSON con mejor tolerancia a formatos.
        """
        if not text or not text.strip():
            return None
        
        # Limpiar texto
        text = text.strip()
        
        # Método 1: Buscar JSON entre llaves (más estricto)
        json_match = re.search(r'\{(?:[^{}]|{[^{}]*})*\}', text, re.DOTALL)
        if json_match:
            candidate = json_match.group(0)
            if self._is_valid_json(candidate):
                return candidate
        
        # Método 2: Buscar JSON entre marcadores de código
        code_patterns = [
            r'```(?:json)?\s*(\{.*?\})\s*```',
            r'```(\{.*?\})```',
            r'`(\{.*?\})`'
        ]
        
        for pattern in code_patterns:
            code_match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if code_match:
                candidate = code_match.group(1)
                if self._is_valid_json(candidate):
                    return candidate
        
        # Método 3: Intentar limpiar JSON común con errores
        cleaned = self._clean_malformed_json(text)
        if cleaned and self._is_valid_json(cleaned):
            return cleaned
        
        return None
    
    def _is_valid_json(self, text: str) -> bool:
        """✅ NUEVO: Verifica si un texto es JSON válido."""
        try:
            json.loads(text)
            return True
        except (json.JSONDecodeError, ValueError):
            return False
    
    def _clean_malformed_json(self, text: str) -> Optional[str]:
        """
        ✅ NUEVO: Intenta limpiar JSON malformado común.
        """
        # Buscar entre primera { y última }
        start = text.find('{')
        end = text.rfind('}')
        
        if start == -1 or end == -1 or start >= end:
            return None
        
        candidate = text[start:end+1]
        
        # Limpiezas comunes
        # Comillas simples a dobles
        candidate = re.sub(r"'([^']*)':", r'"\1":', candidate)
        
        # Claves sin comillas
        candidate = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', candidate)
        
        # Comas finales
        candidate = re.sub(r',(\s*[}\]])', r'\1', candidate)
        
        return candidate
    
    def _manual_parse_terminology_enhanced(self, text: str, discipline: AcademicDiscipline) -> List[TerminologySuggestion]:
        """
        ✅ MEJORA CRÍTICA: Parsing manual mejorado con heurísticas.
        """
        suggestions = []
        lines = text.split('\n')
        
        # Patrones mejorados para detectar términos
        patterns = [
            r'([^→:]+)[\s]*→[\s]*([^→:]+)',  # término → traducción
            r'([^:]+)[\s]*:[\s]*([^:]+)',   # término: traducción
            r'"([^"]+)"[\s]*→[\s]*"([^"]+)"',  # "término" → "traducción"
            r'- ([^:]+):[\s]*([^:\n]+)'     # - término: traducción
        ]
        
        for line in lines:
            line = line.strip()
            if len(line) < 3:  # Ignorar líneas muy cortas
                continue
            
            for pattern in patterns:
                matches = re.findall(pattern, line, re.IGNORECASE)
                for match in matches:
                    if len(match) >= 2:
                        source_term = match[0].strip().strip('"\'').strip('- ')
                        target_term = match[1].strip().strip('"\'').strip('- ')
                        
                        if (source_term and target_term and 
                            len(source_term) > 1 and len(target_term) > 1 and
                            len(source_term) < 100 and len(target_term) < 100):
                            
                            # ✅ MEJORA: Heurísticas para confidence y priority
                            confidence = self._estimate_term_confidence(source_term, target_term, line)
                            priority = self._estimate_term_priority(source_term, discipline)
                            
                            suggestion = TerminologySuggestion(
                                source_term=source_term,
                                target_term=target_term,
                                context="extraído automáticamente con heurísticas",
                                discipline=discipline,
                                confidence=confidence,
                                justification="parsing automático mejorado de respuesta Claude",
                                priority=priority,
                                frequency_estimate="unknown"
                            )
                            suggestions.append(suggestion)
        
        # Eliminar duplicados por source_term
        seen_terms = set()
        unique_suggestions = []
        for suggestion in suggestions:
            if suggestion.source_term not in seen_terms:
                seen_terms.add(suggestion.source_term)
                unique_suggestions.append(suggestion)
        
        return unique_suggestions[:15]  # Máximo 15 términos en fallback
    
    def _estimate_term_confidence(self, source_term: str, target_term: str, context_line: str) -> float:
        """✅ NUEVO: Estima confidence basado en heurísticas."""
        confidence = 0.5  # Base
        
        # Factores que aumentan confidence
        if len(source_term) > 3:  # Términos más largos tienden a ser más técnicos
            confidence += 0.1
        
        if any(word in context_line.lower() for word in ['concepto', 'término', 'technical', 'specialized']):
            confidence += 0.2
        
        if '"' in context_line:  # Términos en comillas tienden a ser más precisos
            confidence += 0.1
        
        # Factores que disminuyen confidence
        if len(source_term) < 3 or len(target_term) < 3:
            confidence -= 0.2
        
        if any(word in source_term.lower() for word in ['the', 'and', 'or', 'el', 'la', 'y', 'o']):
            confidence -= 0.1
        
        return max(0.1, min(0.9, confidence))
    
    def _estimate_term_priority(self, source_term: str, discipline: AcademicDiscipline) -> str:
        """✅ NUEVO: Estima priority basado en disciplina y término."""
        # Términos técnicos largos tienden a ser más importantes
        if len(source_term) > 8:
            return "alta"
        elif len(source_term) > 4:
            return "media"
        else:
            return "baja"
    
    def _create_analysis_summary(self, suggestions: List[TerminologySuggestion]) -> dict:
        """Crea resumen de análisis terminológico (sin cambios significativos)."""
        if not suggestions:
            return {
                "total_sugerencias": 0,
                "confianza_promedio": 0.0,
                "prioridades": {"alta": 0, "media": 0, "baja": 0}
            }
        
        total = len(suggestions)
        confianza_promedio = sum(s.confidence for s in suggestions) / total
        
        prioridades = {"alta": 0, "media": 0, "baja": 0}
        for suggestion in suggestions:
            priority_key = suggestion.priority if suggestion.priority in prioridades else "media"
            prioridades[priority_key] += 1
        
        return {
            "total_sugerencias": total,
            "confianza_promedio": confianza_promedio,
            "prioridades": prioridades,
            "terminos_alta_confianza": sum(1 for s in suggestions if s.confidence > 0.8),
            "disciplina": suggestions[0].discipline.value if suggestions else "unknown",
            "terminos_prioritarios": [
                s.source_term for s in suggestions 
                if s.priority == "alta" or s.confidence > 0.9
            ][:5]
        }


# ===============================================================================
# UTILIDADES ESPECÍFICAS DE CLAUDE (MEJORADAS)
# ===============================================================================

def validate_claude_api_key(api_key: str) -> bool:
    """
    ✅ MEJORADO: Valida formato de API key de Claude/Anthropic con verificaciones adicionales.
    """
    if not api_key or not isinstance(api_key, str):
        return False
    
    # Verificar longitud mínima
    if len(api_key) < 30:
        return False
    
    # Claude API keys empiezan con "sk-ant-"
    if not api_key.startswith("sk-ant-"):
        return False
    
    # ✅ MEJORA: Verificar que tenga formato básico después del prefijo
    key_part = api_key[7:]  # Remover "sk-ant-"
    
    # Debe tener al menos caracteres alfanuméricos y guiones
    if not re.match(r'^[0-9a-zA-Z\-_]+$', key_part):
        return False
    
    # Verificar longitud del key_part
    if len(key_part) < 20:
        return False
    
    return True


def get_supported_models() -> dict:
    """
    ✅ MEJORADO: Retorna modelos Claude soportados con más información.
    """
    return {
        "recommended": {
            "terminology": "claude-3-sonnet-20240229",
            "refinement": "claude-3-sonnet-20240229", 
            "glossary": "claude-3-sonnet-20240229",
            "quality": "claude-3-sonnet-20240229"
        },
        "alternatives": {
            "high_performance": "claude-3-opus-20240229",
            "cost_effective": "claude-3-haiku-20240307"
        },
        "characteristics": {
            "sonnet": "Balance óptimo calidad/velocidad para análisis académico",
            "opus": "Máxima calidad para análisis complejos",
            "haiku": "Rápido y económico para tareas simples"
        },
        "pricing": {
            "sonnet": {"input": 3.0, "output": 15.0, "unit": "per_million_tokens"},
            "opus": {"input": 15.0, "output": 75.0, "unit": "per_million_tokens"},
            "haiku": {"input": 0.25, "output": 1.25, "unit": "per_million_tokens"}
        },
        "context_windows": {
            "sonnet": 200000,
            "opus": 200000,
            "haiku": 200000
        }
    }


def estimate_claude_processing_time(text_length: int, complexity: str = "medium", task_type: str = "terminology") -> dict:
    """
    ✅ MEJORADO: Estima tiempo con factores más precisos y específicos por tarea.
    """
    # ✅ MEJORA: Factores específicos por tipo de tarea
    task_factors = {
        "terminology": 1.0,   # Análisis estándar
        "refinement": 1.5,    # Requiere más análisis
        "glossary": 2.0,      # Procesamiento más complejo
        "quality": 1.3        # Evaluación detallada
    }
    
    # Factores de complejidad
    complexity_factors = {
        "low": 0.7,     # Análisis simple
        "medium": 1.0,  # Análisis terminológico estándar
        "high": 1.8     # Análisis profundo + refinamiento
    }
    
    task_factor = task_factors.get(task_type, 1.0)
    complexity_factor = complexity_factors.get(complexity, 1.0)
    
    # ✅ MEJORA: Estimación base más precisa: ~3 segundos por 1000 caracteres
    base_time = (text_length / 1000) * 3 * complexity_factor * task_factor
    
    # Tiempo mínimo y máximo
    min_time = max(2, base_time * 0.6)
    max_time = min(120, base_time * 2.0)  # Máximo 2 minutos
    
    # ✅ NUEVO: Categorías de tiempo
    if base_time <= 10:
        time_category = "fast"
    elif base_time <= 30:
        time_category = "medium"
    else:
        time_category = "slow"
    
    return {
        "estimated_seconds": int(base_time),
        "min_seconds": int(min_time),
        "max_seconds": int(max_time),
        "complexity_factor": complexity_factor,
        "task_factor": task_factor,
        "time_category": time_category,
        "recommended_model": "claude-3-sonnet-20240229",
        "user_message": _generate_claude_time_message(int(base_time), task_type, time_category)
    }


def _generate_claude_time_message(seconds: int, task_type: str, category: str) -> str:
    """✅ NUEVO: Genera mensaje amigable específico para Claude."""
    task_names = {
        "terminology": "análisis terminológico",
        "refinement": "refinamiento de traducción",
        "glossary": "generación de glosario",
        "quality": "evaluación de calidad"
    }
    
    task_name = task_names.get(task_type, "procesamiento")
    
    if category == "fast":
        return f"{task_name.title()} rápido (~{seconds}s)"
    elif category == "medium":
        return f"{task_name.title()}: {seconds // 60}m {seconds % 60}s estimado"
    else:
        return f"{task_name.title()} complejo: {seconds // 60}m {seconds % 60}s estimado"


def get_academic_discipline_contexts() -> dict:
    """✅ NUEVO: Contextos académicos disponibles para validación."""
    return {
        discipline.value: context 
        for discipline, context in ACADEMIC_CONTEXTS.items()
    }


# ===============================================================================
# TESTS UNITARIOS EMBEBIDOS (MEJORADOS)
# ===============================================================================

async def test_claude_terminology_analysis():
    """✅ MEJORADO: Test básico de análisis terminológico."""
    import os
    api_key = os.getenv("CLAUDE_API_KEY")
    
    if not api_key or api_key.startswith("your_"):
        print("⚠️ Test Claude omitido: API key no configurada")
        return
    
    import logging
    logger = logging.getLogger("test")
    claude = ClaudeAPIIntegration(api_key, logger)
    
    # Test health check
    is_healthy = await claude.health_check()
    assert is_healthy, "Claude API debe estar disponible"
    
    # Test análisis terminológico
    sample_text = "The concept of Being in Heidegger's philosophy represents a fundamental way of understanding human existence."
    
    response = await claude.analyze_terminology(
        text_sample=sample_text,
        discipline=AcademicDiscipline.PHILOSOPHY,
        source_lang=SupportedLanguage.ENGLISH,
        max_terms=3
    )
    
    assert response.success, f"Análisis debe ser exitoso: {response.error_message}"
    assert "suggestions" in response.data
    assert "validation_passed" in response.data
    assert "model_used" in response.data
    
    print("✅ Test Claude Terminology Analysis (mejorado): PASSED")


def test_claude_utilities():
    """✅ MEJORADO: Test de utilidades específicas de Claude."""
    # Test validación API key
    assert validate_claude_api_key("sk-ant-1234567890abcdef1234567890") == True
    assert validate_claude_api_key("invalid-key") == False
    assert validate_claude_api_key("sk-wrong-prefix") == False
    assert validate_claude_api_key("sk-ant-short") == False
    assert validate_claude_api_key("") == False
    assert validate_claude_api_key(None) == False
    
    # Test modelos soportados
    models = get_supported_models()
    assert "recommended" in models
    assert "terminology" in models["recommended"]
    assert "pricing" in models
    assert "context_windows" in models
    
    # Test estimación de tiempo
    estimation = estimate_claude_processing_time(2000, "medium", "terminology")
    assert estimation["estimated_seconds"] > 0
    assert estimation["min_seconds"] <= estimation["estimated_seconds"]
    assert "task_factor" in estimation
    assert "time_category" in estimation
    assert "user_message" in estimation
    
    # Test contextos académicos
    contexts = get_academic_discipline_contexts()
    assert len(contexts) > 0
    assert "philosophy" in contexts or "filosofia" in contexts
    
    print("✅ Test Claude Utilities (mejorado): PASSED")


def test_json_parsing_robust():
    """✅ MEJORADO: Test de parsing robusto de respuestas JSON."""
    import logging
    logger = logging.getLogger("test")
    claude = ClaudeAPIIntegration("sk-ant-test12345678901234567890", logger)
    
    # Test extracción de JSON normal
    sample_text = 'Here is the analysis: {"terminos_sugeridos": [{"termino_original": "test"}]}'
    json_content = claude._extract_json_from_text_robust(sample_text)
    assert json_content is not None
    assert "terminos_sugeridos" in json_content
    
    # Test JSON malformado
    malformed_text = "Analysis: {terminos_sugeridos: [{termino_original: 'test',}]}"
    cleaned_json = claude._clean_malformed_json(malformed_text)
    assert cleaned_json is not None
    
    # Test validación estructural
    valid_term_data = {
        "terminos_sugeridos": [
            {"termino_original": "test", "traduccion_sugerida": "prueba"}
        ]
    }
    assert claude._validate_terminology_structure(valid_term_data) == True
    
    invalid_term_data = {"wrong_key": []}
    assert claude._validate_terminology_structure(invalid_term_data) == False
    
    # Test parsing manual mejorado
    fallback_text = """
    Term: Dasein → ser-ahí
    "Begriff": "concepto"
    - Vorstellung: representación
    """
    suggestions = claude._manual_parse_terminology_enhanced(fallback_text, AcademicDiscipline.PHILOSOPHY)
    assert len(suggestions) >= 2
    assert any(s.source_term == "Dasein" for s in suggestions)
    assert all(s.confidence > 0 for s in suggestions)  # Verificar heurísticas de confidence
    
    print("✅ Test JSON Parsing Robust (mejorado): PASSED")


def test_model_fallback():
    """✅ NUEVO: Test de funcionalidad de modelo fallback."""
    import logging
    logger = logging.getLogger("test")
    
    # Test configuración de modelos
    claude = ClaudeAPIIntegration(
        "sk-ant-test12345678901234567890", 
        logger,
        model="claude-3-opus-20240229",
        fallback_model="claude-3-haiku-20240307"
    )
    
    assert claude.primary_model == "claude-3-opus-20240229"
    assert claude.fallback_model == "claude-3-haiku-20240307"
    assert claude.current_model == "claude-3-opus-20240229"
    
    print("✅ Test Model Fallback: PASSED")


def test_confidence_estimation():
    """✅ NUEVO: Test de estimación de confidence por heurísticas."""
    import logging
    logger = logging.getLogger("test")
    claude = ClaudeAPIIntegration("sk-ant-test12345678901234567890", logger)
    
    # Test confidence con término técnico largo
    confidence_high = claude._estimate_term_confidence("Phenomenology", "fenomenología", 'Technical term: "Phenomenology"')
    assert confidence_high > 0.6
    
    # Test confidence con término corto común
    confidence_low = claude._estimate_term_confidence("the", "el", "the → el")
    assert confidence_low < 0.5
    
    # Test priority estimation
    priority_high = claude._estimate_term_priority("Phenomenological", AcademicDiscipline.PHILOSOPHY)
    assert priority_high == "alta"
    
    priority_low = claude._estimate_term_priority("and", AcademicDiscipline.PHILOSOPHY)
    assert priority_low == "baja"
    
    print("✅ Test Confidence Estimation: PASSED")


async def run_all_tests():
    """Ejecuta todos los tests embebidos."""
    print("🧪 Ejecutando tests de claude_integration.py (POST-AUDITORÍA)...")
    
    try:
        test_claude_utilities()
        test_json_parsing_robust()
        test_model_fallback()
        test_confidence_estimation()
        await test_claude_terminology_analysis()  # Omitido si no hay API key
        
        print("\n✅ Todos los tests de claude_integration.py (POST-AUDITORÍA) pasaron!")
        print("\n🏆 MEJORAS IMPLEMENTADAS:")
        print("  ✅ Método centralizado para llamadas a Claude (_call_claude)")
        print("  ✅ Modelo configurable con fallback automático")
        print("  ✅ Validación estructural estricta de respuestas JSON")
        print("  ✅ Parsing robusto con múltiples estrategias de limpieza")
        print("  ✅ Fallback manual mejorado con heurísticas de confidence")
        print("  ✅ Extracción robusta de contenido de respuestas")
        print("  ✅ Validación exhaustiva de entrada y limpieza de datos")
        print("  ✅ Estimación de tiempo específica por tipo de tarea")
        
    except Exception as e:
        print(f"\n❌ Test falló: {e}")
        raise


if __name__ == "__main__":
    """Ejecutar tests al correr el módulo directamente."""
    import logging
    import asyncio
    logging.basicConfig(level=logging.INFO)
    
    asyncio.run(run_all_tests())