"""
Sistema de Traducción Académica v2.2
SemanticIntegrityValidator - Validación de Integridad Semántica

Valida que las traducciones preserven el contenido semántico original
usando embeddings y análisis de similitud coseno.
"""

import re
import json
import hashlib
import logging
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    print("⚠️  WARNING: sentence-transformers y/o scikit-learn no disponibles")
    print("📦 Instalar con: pip install sentence-transformers scikit-learn")


@dataclass
class AlignedSection:
    """Sección alineada entre original y traducción"""
    original_index: int
    translated_index: int
    similarity_score: float
    alignment_confidence: float
    original_text: str
    translated_text: str


@dataclass
class BiDirectionalAlignment:
    """Resultado del alineamiento bidireccional"""
    section_similarities: List[SectionSimilarity]
    aligned_pairs: List[AlignedSection]
    orphaned_original: List[int]  # Índices de secciones originales sin match
    orphaned_translated: List[int]  # Índices de secciones traducidas sin match
    alignment_quality_score: float


@dataclass
class SectionSimilarity:
    """Métricas de similitud para una sección específica"""
    section_index: int
    original_text: str
    translated_text: str
    similarity_score: float
    best_match_index: int
    content_length_original: int
    content_length_translated: int
    is_critical_loss: bool
    is_semantic_drift: bool


@dataclass
class IntegrityAlert:
    """Alerta de problema de integridad detectado"""
    alert_id: str
    alert_type: str  # 'missing_content', 'semantic_drift', 'structural_mismatch'
    severity: str    # 'critical', 'high', 'medium', 'low'
    section_index: int
    similarity_score: float
    description: str
    suggested_action: str
    original_excerpt: str
    translated_excerpt: str


@dataclass
class IntegrityReport:
    """Reporte completo de integridad semántica"""
    book_id: str
    phase_name: str
    timestamp: datetime
    overall_integrity_score: float
    section_similarities: List[SectionSimilarity]
    missing_content_alerts: List[IntegrityAlert]
    semantic_drift_alerts: List[IntegrityAlert]
    structural_analysis: Dict[str, Any]
    detailed_analysis: Dict[str, Any]
    recommendations: List[str]
    processing_time_seconds: float


class SemanticIntegrityValidator:
    """
    Validador de integridad semántica para traducciones académicas.
    
    Utiliza embeddings de sentence-transformers para detectar:
    - Contenido perdido (similarity < threshold crítico)
    - Deriva semántica (cambios significativos de significado)
    - Inconsistencias estructurales
    """
    
    def __init__(self, 
                 model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2',
                 similarity_threshold_critical: float = 0.85,
                 similarity_threshold_warning: float = 0.90,
                 cache_dir: Optional[str] = None,
                 alignment_penalty_weight: float = 0.1,
                 db_connector: Optional[Any] = None):
        """
        Inicializa el validador semántico.
        
        Args:
            model_name: Modelo de sentence-transformers a usar
            similarity_threshold_critical: Umbral para alertas críticas
            similarity_threshold_warning: Umbral para alertas de advertencia
            cache_dir: Directorio para cache de embeddings
        """
        self.logger = logging.getLogger(__name__)
        
        # Umbrales de calidad
        self.similarity_threshold_critical = similarity_threshold_critical
        self.similarity_threshold_warning = similarity_threshold_warning
        
        # Configuración de alineamiento secuencial
        self.alignment_penalty_weight = alignment_penalty_weight
        
        # Configuración de cache
        self.cache_dir = Path(cache_dir) if cache_dir else Path("cache/embeddings")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Integración con base de datos
        self.db_connector = db_connector
        
        # Inicializar modelo de embeddings
        if EMBEDDINGS_AVAILABLE:
            try:
                self.logger.info(f"Cargando modelo de embeddings: {model_name}")
                self.embeddings_model = SentenceTransformer(model_name)
                self.logger.info("✅ Modelo de embeddings cargado exitosamente")
            except Exception as e:
                self.logger.error(f"❌ Error cargando modelo de embeddings: {e}")
                self.embeddings_model = None
        else:
            self.embeddings_model = None
            self.logger.warning("⚠️  Embeddings no disponibles - modo simulación")
        
        # Patrones para división de secciones
        self.section_patterns = [
            r'\n\s*\n',  # Doble salto de línea
            r'\.\s+[A-Z]',  # Punto seguido de mayúscula
            r'\n\d+\.',  # Línea que empieza con número y punto
            r'\n[A-Z][a-z]+:',  # Línea que empieza con palabra y dos puntos
        ]
        
        # Métricas de procesamiento
        self.processing_stats = {
            'texts_processed': 0,
            'embeddings_generated': 0,
            'cache_hits': 0,
            'total_processing_time': 0.0
        }
    
    def validate_semantic_integrity(self, 
                                   original_text: str, 
                                   translated_text: str,
                                   book_id: str = "unknown",
                                   phase_name: str = "translation") -> IntegrityReport:
        """
        Valida la integridad semántica entre texto original y traducido.
        
        Args:
            original_text: Texto en idioma original
            translated_text: Texto traducido
            book_id: Identificador del libro
            phase_name: Fase del procesamiento
            
        Returns:
            IntegrityReport con análisis completo
        """
        start_time = datetime.now()
        
        self.logger.info(f"🔍 Iniciando validación semántica para {book_id}")
        
        # 1. Análisis estructural básico
        structural_analysis = self._analyze_text_structure(original_text, translated_text)
        
        if not self.embeddings_model:
            # Modo simulación sin embeddings
            return self._create_simulated_report(
                book_id, phase_name, start_time, original_text, translated_text,
                structural_analysis
            )
        
        # 2. Dividir en secciones semánticas
        original_sections = self.split_into_semantic_sections(original_text)
        translated_sections = self.split_into_semantic_sections(translated_text)
        
        self.logger.info(f"📄 Secciones detectadas - Original: {len(original_sections)}, Traducido: {len(translated_sections)}")
        
        # 3. Generar embeddings con cache
        original_embeddings = self._get_cached_embeddings(original_sections, "original")
        translated_embeddings = self._get_cached_embeddings(translated_sections, "translated")
        
        # 4. Calcular matriz de similitudes con penalización por distancia
        similarity_matrix = self._calculate_enhanced_similarity_matrix(
            original_embeddings, translated_embeddings
        )
        
        # 5. Realizar alineamiento bidireccional
        aligned_sections = self._perform_bidirectional_alignment(
            original_sections, translated_sections, similarity_matrix
        )
        
        # 6. Detectar problemas de integridad
        missing_content_alerts = self._detect_missing_content(aligned_sections.section_similarities)
        semantic_drift_alerts = self._detect_semantic_drift(aligned_sections.section_similarities)
        orphaned_sections_alerts = self._detect_orphaned_sections(aligned_sections.orphaned_translated)
        
        # 7. Calcular score general de integridad
        overall_score = self._calculate_enhanced_integrity_score(aligned_sections)
        
        # 8. Generar análisis detallado
        detailed_analysis = self._generate_detailed_analysis(
            aligned_sections.section_similarities, similarity_matrix, 
            original_sections, translated_sections, aligned_sections
        )
        
        # 9. Generar recomendaciones
        recommendations = self._generate_recommendations(
            missing_content_alerts, semantic_drift_alerts, orphaned_sections_alerts, overall_score
        )
        
        # 10. Persistir en base de datos si está disponible
        if self.db_connector:
            self._persist_audit_result(book_id, phase_name, overall_score, 
                                     missing_content_alerts, semantic_drift_alerts)
        
        # Calcular tiempo de procesamiento
        processing_time = (datetime.now() - start_time).total_seconds()
        self.processing_stats['total_processing_time'] += processing_time
        self.processing_stats['texts_processed'] += 1
        
        # Crear reporte final
        report = IntegrityReport(
            book_id=book_id,
            phase_name=phase_name,
            timestamp=start_time,
            overall_integrity_score=overall_score,
            section_similarities=aligned_sections.section_similarities,
            missing_content_alerts=missing_content_alerts,
            semantic_drift_alerts=semantic_drift_alerts + orphaned_sections_alerts,  # Combinar alertas
            structural_analysis=structural_analysis,
            detailed_analysis=detailed_analysis,
            recommendations=recommendations,
            processing_time_seconds=processing_time
        )
        
        self.logger.info(f"✅ Validación completada - Score: {overall_score:.3f}, Tiempo: {processing_time:.2f}s")
        
        return report
    
    def split_into_semantic_sections(self, text: str) -> List[str]:
        """
        Divide el texto en secciones semánticamente coherentes.
        
        Args:
            text: Texto a dividir
            
        Returns:
            Lista de secciones de texto
        """
        if not text.strip():
            return []
        
        # Limpiar texto
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Dividir por párrafos primero
        paragraphs = re.split(r'\n\s*\n', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        sections = []
        current_section = ""
        
        for paragraph in paragraphs:
            # Si el párrafo es muy corto, agregarlo a la sección actual
            if len(paragraph) < 100 and current_section:
                current_section += f"\n\n{paragraph}"
            else:
                # Si hay sección acumulada, agregarla
                if current_section:
                    sections.append(current_section.strip())
                current_section = paragraph
        
        # Agregar última sección
        if current_section:
            sections.append(current_section.strip())
        
        # Filtrar secciones muy cortas (menos de 50 caracteres)
        sections = [s for s in sections if len(s) >= 50]
        
        # Si no hay secciones suficientes, dividir por oraciones
        if len(sections) < 2 and len(text) > 200:
            sentences = re.split(r'[.!?]+\s+', text)
            sentences = [s.strip() for s in sentences if len(s.strip()) > 30]
            
            # Agrupar oraciones en chunks de tamaño razonable
            chunk_size = max(2, len(sentences) // 4)
            sections = []
            for i in range(0, len(sentences), chunk_size):
                chunk = '. '.join(sentences[i:i + chunk_size])
                if chunk and len(chunk) > 50:
                    sections.append(chunk)
        
        return sections if sections else [text]
    
    def _get_cached_embeddings(self, sections: List[str], section_type: str) -> np.ndarray:
        """
        Obtiene embeddings con sistema de cache.
        
        Args:
            sections: Lista de secciones de texto
            section_type: Tipo de sección ('original' o 'translated')
            
        Returns:
            Array numpy con embeddings
        """
        embeddings = []
        
        for i, section in enumerate(sections):
            # Generar hash del texto para cache
            text_hash = hashlib.md5(section.encode('utf-8')).hexdigest()
            cache_file = self.cache_dir / f"{text_hash}.npy"
            
            if cache_file.exists():
                # Cargar desde cache
                try:
                    embedding = np.load(cache_file)
                    embeddings.append(embedding)
                    self.processing_stats['cache_hits'] += 1
                    continue
                except Exception as e:
                    self.logger.warning(f"Error cargando cache {cache_file}: {e}")
            
            # Generar embedding
            try:
                embedding = self.embeddings_model.encode([section])[0]
                embeddings.append(embedding)
                self.processing_stats['embeddings_generated'] += 1
                
                # Guardar en cache
                try:
                    np.save(cache_file, embedding)
                except Exception as e:
                    self.logger.warning(f"Error guardando cache {cache_file}: {e}")
                    
            except Exception as e:
                self.logger.error(f"Error generando embedding para sección {i}: {e}")
                # Usar embedding vacío como fallback
                embedding = np.zeros(384)  # Tamaño típico de all-MiniLM-L6-v2
                embeddings.append(embedding)
        
        return np.array(embeddings)
    
    def _calculate_enhanced_similarity_matrix(self, 
                                            original_embeddings: np.ndarray,
                                            translated_embeddings: np.ndarray) -> np.ndarray:
        """
        Calcula matriz de similitud con penalización por distancia secuencial.
        
        Args:
            original_embeddings: Embeddings del texto original
            translated_embeddings: Embeddings del texto traducido
            
        Returns:
            Matriz de similitud ajustada por distancia
        """
        # Similitud coseno base
        base_similarity = cosine_similarity(original_embeddings, translated_embeddings)
        
        # Crear matriz de penalización por distancia
        rows, cols = base_similarity.shape
        distance_penalty = np.zeros((rows, cols))
        
        for i in range(rows):
            for j in range(cols):
                # Penalización basada en distancia de índice
                distance = abs(i - j)
                # Normalizar por el tamaño máximo para evitar penalizaciones excesivas
                max_distance = max(rows, cols)
                normalized_distance = distance / max_distance
                distance_penalty[i, j] = normalized_distance * self.alignment_penalty_weight
        
        # Aplicar penalización
        enhanced_similarity = base_similarity - distance_penalty
        
        # Asegurar que los valores estén en rango válido [0, 1]
        enhanced_similarity = np.clip(enhanced_similarity, 0, 1)
        
        return enhanced_similarity
    
    def _perform_bidirectional_alignment(self,
                                       original_sections: List[str],
                                       translated_sections: List[str],
                                       similarity_matrix: np.ndarray) -> BiDirectionalAlignment:
        """
        Realiza alineamiento bidireccional entre secciones originales y traducidas.
        
        Args:
            original_sections: Secciones del texto original
            translated_sections: Secciones del texto traducido
            similarity_matrix: Matriz de similitud mejorada
            
        Returns:
            Resultado del alineamiento bidireccional
        """
        rows, cols = similarity_matrix.shape
        
        # Alineamiento greedy con prevención de duplicados
        aligned_pairs = []
        used_original = set()
        used_translated = set()
        
        # Crear lista de candidatos ordenados por similitud
        candidates = []
        for i in range(rows):
            for j in range(cols):
                candidates.append((similarity_matrix[i, j], i, j))
        
        # Ordenar por similitud descendente
        candidates.sort(reverse=True)
        
        # Asignar pares evitando duplicados
        for similarity, orig_idx, trans_idx in candidates:
            if orig_idx not in used_original and trans_idx not in used_translated:
                if similarity >= self.similarity_threshold_critical * 0.7:  # Umbral mínimo para alineamiento
                    aligned_pairs.append(AlignedSection(
                        original_index=orig_idx,
                        translated_index=trans_idx,
                        similarity_score=similarity,
                        alignment_confidence=similarity,
                        original_text=original_sections[orig_idx],
                        translated_text=translated_sections[trans_idx]
                    ))
                    used_original.add(orig_idx)
                    used_translated.add(trans_idx)
        
        # Identificar secciones huérfanas
        orphaned_original = [i for i in range(len(original_sections)) if i not in used_original]
        orphaned_translated = [i for i in range(len(translated_sections)) if i not in used_translated]
        
        # Convertir alineamientos a SectionSimilarity para compatibilidad
        section_similarities = []
        for aligned in aligned_pairs:
            similarity = SectionSimilarity(
                section_index=aligned.original_index,
                original_text=aligned.original_text,
                translated_text=aligned.translated_text,
                similarity_score=aligned.similarity_score,
                best_match_index=aligned.translated_index,
                content_length_original=len(aligned.original_text),
                content_length_translated=len(aligned.translated_text),
                is_critical_loss=aligned.similarity_score < self.similarity_threshold_critical,
                is_semantic_drift=(self.similarity_threshold_critical <= aligned.similarity_score < 
                                 self.similarity_threshold_warning)
            )
            section_similarities.append(similarity)
        
        # Agregar secciones huérfanas del original como pérdidas críticas
        for orig_idx in orphaned_original:
            similarity = SectionSimilarity(
                section_index=orig_idx,
                original_text=original_sections[orig_idx],
                translated_text="",
                similarity_score=0.0,
                best_match_index=-1,
                content_length_original=len(original_sections[orig_idx]),
                content_length_translated=0,
                is_critical_loss=True,
                is_semantic_drift=False
            )
            section_similarities.append(similarity)
        
        # Calcular calidad del alineamiento
        alignment_quality = len(aligned_pairs) / max(len(original_sections), len(translated_sections))
        
        return BiDirectionalAlignment(
            section_similarities=section_similarities,
            aligned_pairs=aligned_pairs,
            orphaned_original=orphaned_original,
            orphaned_translated=orphaned_translated,
            alignment_quality_score=alignment_quality
        )
    
    def _detect_orphaned_sections(self, orphaned_translated_indices: List[int]) -> List[IntegrityAlert]:
        """
        Detecta secciones huérfanas en la traducción (contenido agregado).
        
        Args:
            orphaned_translated_indices: Índices de secciones sin match en original
            
        Returns:
            Lista de alertas de contenido agregado
        """
        orphaned_alerts = []
        
        for trans_idx in orphaned_translated_indices:
            alert = IntegrityAlert(
                alert_id=f"orphaned_content_{trans_idx}",
                alert_type="orphaned_content",
                severity="medium",
                section_index=trans_idx,
                similarity_score=0.0,
                description=f"Sección traducida #{trans_idx} sin correspondencia en original. "
                          f"Posible contenido agregado o mal alineado.",
                suggested_action="Revisar si es contenido agregado intencionalmente o error de alineamiento.",
                original_excerpt="[Sin correspondencia]",
                translated_excerpt="[Sección huérfana en traducción]"
            )
            orphaned_alerts.append(alert)
        
        return orphaned_alerts
    
    def _calculate_enhanced_integrity_score(self, aligned_sections: BiDirectionalAlignment) -> float:
        """
        Calcula score de integridad mejorado considerando alineamiento.
        
        Args:
            aligned_sections: Resultado del alineamiento bidireccional
            
        Returns:
            Score de integridad mejorado (0.0 - 1.0)
        """
        if not aligned_sections.section_similarities:
            return 0.0
        
        # Peso por longitud de contenido (como antes)
        total_weighted_score = 0.0
        total_weight = 0.0
        
        # Filtrar embeddings inválidos (score = 0 y texto vacío)
        valid_similarities = [
            s for s in aligned_sections.section_similarities 
            if not (s.similarity_score == 0.0 and not s.translated_text.strip())
        ]
        
        for similarity in valid_similarities:
            weight = similarity.content_length_original
            total_weighted_score += similarity.similarity_score * weight
            total_weight += weight
        
        base_score = total_weighted_score / total_weight if total_weight > 0 else 0.0
        
        # Penalizar por calidad de alineamiento
        alignment_penalty = (1.0 - aligned_sections.alignment_quality_score) * 0.2
        
        # Penalizar por secciones huérfanas
        orphan_penalty = len(aligned_sections.orphaned_translated) * 0.05
        
        final_score = base_score - alignment_penalty - orphan_penalty
        
        return max(0.0, min(1.0, final_score))
    
    def _persist_audit_result(self, book_id: str, phase_name: str, quality_score: float,
                            missing_alerts: List[IntegrityAlert], 
                            semantic_alerts: List[IntegrityAlert]):
        """
        Persiste resultado de auditoría en base de datos.
        
        Args:
            book_id: ID del libro
            phase_name: Nombre de la fase
            quality_score: Score de calidad
            missing_alerts: Alertas de contenido perdido
            semantic_alerts: Alertas de deriva semántica
        """
        try:
            if hasattr(self.db_connector, 'log_audit_result'):
                audit_data = {
                    'book_id': book_id,
                    'phase_name': phase_name,
                    'quality_score': quality_score,
                    'integrity_score': quality_score,
                    'alerts': [asdict(alert) for alert in missing_alerts + semantic_alerts],
                    'metrics': self.processing_stats.copy()
                }
                self.db_connector.log_audit_result(audit_data)
                self.logger.info(f"✅ Resultado de auditoría persistido para {book_id}")
        except Exception as e:
            self.logger.error(f"❌ Error persistiendo auditoría: {e}")
    
    def _analyze_section_similarities(self, 
                                    original_sections: List[str],
                                    translated_sections: List[str],
                                    similarity_matrix: np.ndarray) -> List[SectionSimilarity]:
        """
        Analiza similitudes entre secciones original y traducida.
        
        Args:
            original_sections: Secciones del texto original
            translated_sections: Secciones del texto traducido
            similarity_matrix: Matriz de similitudes coseno
            
        Returns:
            Lista de análisis de similitud por sección
        """
        section_similarities = []
        
        for i, original_section in enumerate(original_sections):
            if i >= len(similarity_matrix):
                # No hay suficientes secciones traducidas
                similarity = SectionSimilarity(
                    section_index=i,
                    original_text=original_section,
                    translated_text="",
                    similarity_score=0.0,
                    best_match_index=-1,
                    content_length_original=len(original_section),
                    content_length_translated=0,
                    is_critical_loss=True,
                    is_semantic_drift=False
                )
                section_similarities.append(similarity)
                continue
            
            # Encontrar la mejor coincidencia en las secciones traducidas
            best_match_index = np.argmax(similarity_matrix[i])
            best_similarity = similarity_matrix[i][best_match_index]
            
            translated_section = ""
            if best_match_index < len(translated_sections):
                translated_section = translated_sections[best_match_index]
            
            # Determinar tipos de problemas
            is_critical_loss = best_similarity < self.similarity_threshold_critical
            is_semantic_drift = (self.similarity_threshold_critical <= best_similarity < 
                               self.similarity_threshold_warning)
            
            similarity = SectionSimilarity(
                section_index=i,
                original_text=original_section,
                translated_text=translated_section,
                similarity_score=best_similarity,
                best_match_index=best_match_index,
                content_length_original=len(original_section),
                content_length_translated=len(translated_section),
                is_critical_loss=is_critical_loss,
                is_semantic_drift=is_semantic_drift
            )
            
            section_similarities.append(similarity)
        
        return section_similarities
    
    def _detect_missing_content(self, section_similarities: List[SectionSimilarity]) -> List[IntegrityAlert]:
        """
        Detecta contenido perdido basado en similitudes bajas.
        
        Args:
            section_similarities: Análisis de similitudes por sección
            
        Returns:
            Lista de alertas de contenido perdido
        """
        missing_content_alerts = []
        
        for similarity in section_similarities:
            if similarity.is_critical_loss:
                alert = IntegrityAlert(
                    alert_id=f"missing_content_{similarity.section_index}",
                    alert_type="missing_content",
                    severity="critical" if similarity.similarity_score < 0.5 else "high",
                    section_index=similarity.section_index,
                    similarity_score=similarity.similarity_score,
                    description=f"Contenido posiblemente perdido en sección {similarity.section_index}. "
                              f"Similitud semántica muy baja: {similarity.similarity_score:.3f}",
                    suggested_action="Revisar manualmente la traducción de esta sección. "
                                   "Considerar retraducir si es contenido crítico.",
                    original_excerpt=self._truncate_text(similarity.original_text, 200),
                    translated_excerpt=self._truncate_text(similarity.translated_text, 200)
                )
                missing_content_alerts.append(alert)
        
        return missing_content_alerts
    
    def _detect_semantic_drift(self, section_similarities: List[SectionSimilarity]) -> List[IntegrityAlert]:
        """
        Detecta deriva semántica en las traducciones.
        
        Args:
            section_similarities: Análisis de similitudes por sección
            
        Returns:
            Lista de alertas de deriva semántica
        """
        semantic_drift_alerts = []
        
        for similarity in section_similarities:
            if similarity.is_semantic_drift:
                alert = IntegrityAlert(
                    alert_id=f"semantic_drift_{similarity.section_index}",
                    alert_type="semantic_drift",
                    severity="medium",
                    section_index=similarity.section_index,
                    similarity_score=similarity.similarity_score,
                    description=f"Deriva semántica detectada en sección {similarity.section_index}. "
                              f"Similitud: {similarity.similarity_score:.3f}",
                    suggested_action="Revisar terminología y contexto. "
                                   "Verificar que se preserve el significado académico.",
                    original_excerpt=self._truncate_text(similarity.original_text, 200),
                    translated_excerpt=self._truncate_text(similarity.translated_text, 200)
                )
                semantic_drift_alerts.append(alert)
        
        return semantic_drift_alerts
    
    def _calculate_overall_integrity_score(self, section_similarities: List[SectionSimilarity]) -> float:
        """
        Calcula score general de integridad semántica.
        
        Args:
            section_similarities: Análisis de similitudes por sección
            
        Returns:
            Score de integridad (0.0 - 1.0)
        """
        if not section_similarities:
            return 0.0
        
        # Peso por longitud de contenido
        total_weighted_score = 0.0
        total_weight = 0.0
        
        for similarity in section_similarities:
            weight = similarity.content_length_original
            total_weighted_score += similarity.similarity_score * weight
            total_weight += weight
        
        return total_weighted_score / total_weight if total_weight > 0 else 0.0
    
    def _analyze_text_structure(self, original: str, translated: str) -> Dict[str, Any]:
        """
        Analiza la estructura de los textos.
        
        Args:
            original: Texto original
            translated: Texto traducido
            
        Returns:
            Diccionario con análisis estructural
        """
        return {
            'original_length': len(original),
            'translated_length': len(translated),
            'length_ratio': len(translated) / len(original) if len(original) > 0 else 0,
            'original_words': len(original.split()),
            'translated_words': len(translated.split()),
            'words_ratio': len(translated.split()) / len(original.split()) if len(original.split()) > 0 else 0,
            'original_sentences': len(re.findall(r'[.!?]+', original)),
            'translated_sentences': len(re.findall(r'[.!?]+', translated)),
            'structural_similarity': self._calculate_structural_similarity(original, translated)
        }
    
    def _calculate_structural_similarity(self, original: str, translated: str) -> float:
        """
        Calcula similitud estructural básica.
        
        Args:
            original: Texto original
            translated: Texto traducido
            
        Returns:
            Score de similitud estructural (0.0 - 1.0)
        """
        # Análisis muy básico - en producción se podría expandir
        length_similarity = 1.0 - abs(len(original) - len(translated)) / max(len(original), len(translated), 1)
        
        orig_sentences = len(re.findall(r'[.!?]+', original))
        trans_sentences = len(re.findall(r'[.!?]+', translated))
        sentence_similarity = 1.0 - abs(orig_sentences - trans_sentences) / max(orig_sentences, trans_sentences, 1)
        
        return (length_similarity + sentence_similarity) / 2
    
    def _generate_detailed_analysis(self, 
                                  section_similarities: List[SectionSimilarity],
                                  similarity_matrix: np.ndarray,
                                  original_sections: List[str],
                                  translated_sections: List[str],
                                  aligned_sections: BiDirectionalAlignment) -> Dict[str, Any]:
        """
        Genera análisis detallado de métricas.
        
        Args:
            section_similarities: Similitudes por sección
            similarity_matrix: Matriz completa de similitudes
            original_sections: Secciones originales
            translated_sections: Secciones traducidas
            
        Returns:
            Diccionario con análisis detallado
        """
        similarities_scores = [s.similarity_score for s in section_similarities]
        
        return {
            'total_sections_original': len(original_sections),
            'total_sections_translated': len(translated_sections),
            'sections_analyzed': len(section_similarities),
            'sections_aligned': len(aligned_sections.aligned_pairs),
            'orphaned_original': len(aligned_sections.orphaned_original),
            'orphaned_translated': len(aligned_sections.orphaned_translated),
            'alignment_quality_score': aligned_sections.alignment_quality_score,
            'average_similarity': np.mean(similarities_scores) if similarities_scores else 0.0,
            'median_similarity': np.median(similarities_scores) if similarities_scores else 0.0,
            'min_similarity': np.min(similarities_scores) if similarities_scores else 0.0,
            'max_similarity': np.max(similarities_scores) if similarities_scores else 0.0,
            'std_similarity': np.std(similarities_scores) if similarities_scores else 0.0,
            'sections_below_critical': sum(1 for s in similarities_scores if s < self.similarity_threshold_critical),
            'sections_below_warning': sum(1 for s in similarities_scores if s < self.similarity_threshold_warning),
            'valid_embeddings_ratio': sum(1 for s in similarities_scores if s > 0.0) / len(similarities_scores) if similarities_scores else 0.0,
            'processing_stats': self.processing_stats.copy()
        }
    
    def _generate_recommendations(self, 
                                missing_content: List[IntegrityAlert],
                                semantic_drift: List[IntegrityAlert],
                                orphaned_content: List[IntegrityAlert],
                                overall_score: float) -> List[str]:
        """
        Genera recomendaciones basadas en el análisis.
        
        Args:
            missing_content: Alertas de contenido perdido
            semantic_drift: Alertas de deriva semántica
            overall_score: Score general de integridad
            
        Returns:
            Lista de recomendaciones
        """
        recommendations = []
        
        if overall_score < 0.70:
            recommendations.append("🚨 CRÍTICO: Score de integridad muy bajo. Considerar retraducir completamente.")
        elif overall_score < 0.85:
            recommendations.append("⚠️  ALERTA: Score de integridad bajo. Revisar y corregir secciones problemáticas.")
        elif overall_score < 0.90:
            recommendations.append("⚡ MEJORAR: Score aceptable pero mejorable. Revisar términos específicos.")
        else:
            recommendations.append("✅ EXCELENTE: Alta integridad semántica. Revisar alertas menores si las hay.")
        
        if missing_content:
            recommendations.append(f"📄 Revisar {len(missing_content)} secciones con posible contenido perdido")
        
        if semantic_drift:
            recommendations.append(f"🎯 Ajustar terminología en {len(semantic_drift)} secciones con deriva semántica")
        
        if orphaned_content:
            recommendations.append(f"👤 Revisar {len(orphaned_content)} secciones huérfanas (contenido agregado o mal alineado)")
        
        if len(missing_content) > len(semantic_drift) * 2:
            recommendations.append("🔍 PATRÓN: Más contenido perdido que deriva. Revisar proceso de traducción.")
        
        if len(orphaned_content) > len(missing_content):
            recommendations.append("📝 PATRÓN: Más contenido agregado que perdido. Revisar fidelidad al original.")
        
        return recommendations
    
    def _create_simulated_report(self, 
                               book_id: str,
                               phase_name: str,
                               start_time: datetime,
                               original_text: str,
                               translated_text: str,
                               structural_analysis: Dict[str, Any]) -> IntegrityReport:
        """
        Crea un reporte simulado cuando los embeddings no están disponibles.
        """
        self.logger.warning("⚠️  Creando reporte simulado - embeddings no disponibles")
        
        # Análisis estructural básico como proxy
        structural_score = structural_analysis.get('structural_similarity', 0.8)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return IntegrityReport(
            book_id=book_id,
            phase_name=phase_name,
            timestamp=start_time,
            overall_integrity_score=structural_score,
            section_similarities=[],
            missing_content_alerts=[],
            semantic_drift_alerts=[],
            structural_analysis=structural_analysis,
            detailed_analysis={'mode': 'simulated', 'reason': 'embeddings_not_available'},
            recommendations=["⚠️  Instalar sentence-transformers para análisis semántico completo"],
            processing_time_seconds=processing_time
        )
    
    def _truncate_text(self, text: str, max_length: int) -> str:
        """Trunca texto a longitud máxima."""
        if len(text) <= max_length:
            return text
        return text[:max_length-3] + "..."
    
    def generate_integrity_report(self, report: IntegrityReport) -> str:
        """
        Genera reporte de integridad en formato legible.
        
        Args:
            report: Reporte de integridad
            
        Returns:
            Reporte formateado como string
        """
        lines = []
        lines.append("=" * 80)
        lines.append("📊 REPORTE DE INTEGRIDAD SEMÁNTICA")
        lines.append("=" * 80)
        lines.append(f"📚 Libro: {report.book_id}")
        lines.append(f"🔄 Fase: {report.phase_name}")
        lines.append(f"⏰ Timestamp: {report.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"⚡ Tiempo procesamiento: {report.processing_time_seconds:.2f}s")
        lines.append("")
        
        # Score general
        score_emoji = "🟢" if report.overall_integrity_score > 0.9 else "🟡" if report.overall_integrity_score > 0.85 else "🔴"
        lines.append(f"📈 SCORE GENERAL DE INTEGRIDAD: {score_emoji} {report.overall_integrity_score:.3f}")
        lines.append("")
        
        # Análisis estructural
        if report.structural_analysis:
            lines.append("📋 ANÁLISIS ESTRUCTURAL:")
            struct = report.structural_analysis
            lines.append(f"   • Longitud original: {struct.get('original_length', 0):,} caracteres")
            lines.append(f"   • Longitud traducida: {struct.get('translated_length', 0):,} caracteres")
            lines.append(f"   • Ratio de longitud: {struct.get('length_ratio', 0):.2f}")
            lines.append(f"   • Palabras original: {struct.get('original_words', 0):,}")
            lines.append(f"   • Palabras traducidas: {struct.get('translated_words', 0):,}")
            lines.append("")
        
        # Análisis detallado
        if report.detailed_analysis and 'mode' not in report.detailed_analysis:
            lines.append("🔍 ANÁLISIS DETALLADO:")
            detail = report.detailed_analysis
            lines.append(f"   • Secciones analizadas: {detail.get('sections_analyzed', 0)}")
            lines.append(f"   • Secciones alineadas: {detail.get('sections_aligned', 0)}")
            lines.append(f"   • Calidad de alineamiento: {detail.get('alignment_quality_score', 0):.3f}")
            lines.append(f"   • Similitud promedio: {detail.get('average_similarity', 0):.3f}")
            lines.append(f"   • Similitud mínima: {detail.get('min_similarity', 0):.3f}")
            lines.append(f"   • Similitud máxima: {detail.get('max_similarity', 0):.3f}")
            lines.append(f"   • Secciones bajo umbral crítico: {detail.get('sections_below_critical', 0)}")
            lines.append(f"   • Secciones bajo umbral advertencia: {detail.get('sections_below_warning', 0)}")
            
            # Información sobre secciones huérfanas
            orphaned_orig = detail.get('orphaned_original', 0)
            orphaned_trans = detail.get('orphaned_translated', 0)
            if orphaned_orig > 0 or orphaned_trans > 0:
                lines.append(f"   • Secciones huérfanas originales: {orphaned_orig}")
                lines.append(f"   • Secciones huérfanas traducidas: {orphaned_trans}")
            
            lines.append(f"   • Ratio embeddings válidos: {detail.get('valid_embeddings_ratio', 0):.3f}")
            lines.append("")
        
        # Alertas de contenido perdido
        if report.missing_content_alerts:
            lines.append("🚨 ALERTAS DE CONTENIDO PERDIDO:")
            for alert in report.missing_content_alerts:
                lines.append(f"   • Sección {alert.section_index}: {alert.description}")
                lines.append(f"     Similitud: {alert.similarity_score:.3f} | Severidad: {alert.severity}")
                lines.append(f"     Acción: {alert.suggested_action}")
            lines.append("")
        
        # Alertas de deriva semántica
        semantic_alerts = [alert for alert in report.semantic_drift_alerts if alert.alert_type == 'semantic_drift']
        orphaned_alerts = [alert for alert in report.semantic_drift_alerts if alert.alert_type == 'orphaned_content']
        
        if semantic_alerts:
            lines.append("⚠️  ALERTAS DE DERIVA SEMÁNTICA:")
            for alert in semantic_alerts:
                lines.append(f"   • Sección {alert.section_index}: {alert.description}")
                lines.append(f"     Similitud: {alert.similarity_score:.3f}")
            lines.append("")
        
        # Alertas de contenido huérfano
        if orphaned_alerts:
            lines.append("👤 ALERTAS DE CONTENIDO HUÉRFANO:")
            for alert in orphaned_alerts:
                lines.append(f"   • Sección {alert.section_index}: {alert.description}")
                lines.append(f"     Acción: {alert.suggested_action}")
            lines.append("")
        
        # Recomendaciones
        if report.recommendations:
            lines.append("💡 RECOMENDACIONES:")
            for rec in report.recommendations:
                lines.append(f"   • {rec}")
            lines.append("")
        
        lines.append("=" * 80)
        
        return "\n".join(lines)
    
    def export_report_json(self, report: IntegrityReport, output_path: str) -> bool:
        """
        Exporta reporte en formato JSON.
        
        Args:
            report: Reporte de integridad
            output_path: Ruta del archivo de salida
            
        Returns:
            True si se exportó exitosamente
        """
        try:
            # Convertir a diccionario serializable
            report_dict = asdict(report)
            
            # Convertir datetime a string
            report_dict['timestamp'] = report.timestamp.isoformat()
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report_dict, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"✅ Reporte exportado a {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error exportando reporte: {e}")
            return False


def demo_semantic_validator():
    """Demostración del SemanticIntegrityValidator mejorado"""
    print("🧪 Demo: SemanticIntegrityValidator v2.2 (Mejorado)")
    print("=" * 60)
    
    # Crear validador con modelo multilingüe mejorado
    print("🔧 Inicializando validador con modelo multilingüe...")
    validator = SemanticIntegrityValidator(
        model_name='paraphrase-multilingual-MiniLM-L12-v2',
        alignment_penalty_weight=0.15
    )
    
    # Textos de ejemplo más complejos
    original_text = """
    El concepto de Dasein en la filosofía de Martin Heidegger representa una manera
    fundamental de entender la existencia humana. Dasein, literalmente "ser-ahí",
    se refiere a la forma específica en que los seres humanos existen en el mundo.
    
    A diferencia de otros entes, el Dasein tiene la característica peculiar de
    preguntarse por el ser mismo. Esta capacidad de cuestionamiento ontológico
    es lo que distingue al Dasein de otros modos de ser.
    
    La temporalidad es un aspecto central del Dasein. Heidegger argumenta que
    el Dasein existe siempre en una relación temporal específica, proyectándose
    hacia el futuro mientras retiene su pasado y actúa en el presente.
    
    El concepto de "arrojamiento" (Geworfenheit) describe cómo el Dasein se encuentra
    siempre ya situado en un mundo que no eligió, pero en el cual debe hacer
    elecciones auténticas para realizar su potencial más propio.
    """
    
    # Traducción con contenido agregado y reordenado
    problematic_translation = """
    Heidegger's philosophy introduces the concept of Dasein as a fundamental
    way to understand human existence. Being-there, literally translated,
    refers to how humans specifically exist in the world.
    
    This is a completely new paragraph that wasn't in the original text,
    demonstrating how the validator detects orphaned content in translations.
    
    Temporality represents a central aspect of Dasein. Heidegger argues that
    Dasein always exists in specific temporal relations, projecting toward
    the future while retaining the past and acting in the present.
    
    Unlike other entities, Dasein has the peculiar characteristic of questioning
    being itself. This ontological questioning capacity distinguishes Dasein
    from other modes of being.
    
    Another added section that demonstrates content drift and misalignment
    in the translation process that the enhanced validator should detect.
    """
    
    print("📝 Validando traducción con problemas de alineamiento...")
    print("   ✓ Contenido reordenado")
    print("   ✓ Secciones agregadas") 
    print("   ✓ Posible contenido perdido")
    print()
    
    report = validator.validate_semantic_integrity(
        original_text, problematic_translation, "demo_complex", "translation"
    )
    
    print(f"📊 RESULTADOS DE VALIDACIÓN:")
    print(f"   • Score de integridad: {report.overall_integrity_score:.3f}")
    print(f"   • Alertas de contenido perdido: {len(report.missing_content_alerts)}")
    
    # Separar alertas por tipo
    semantic_alerts = [a for a in report.semantic_drift_alerts if a.alert_type == 'semantic_drift']
    orphaned_alerts = [a for a in report.semantic_drift_alerts if a.alert_type == 'orphaned_content']
    
    print(f"   • Alertas de deriva semántica: {len(semantic_alerts)}")
    print(f"   • Alertas de contenido huérfano: {len(orphaned_alerts)}")
    
    if report.detailed_analysis:
        detail = report.detailed_analysis
        print(f"   • Calidad de alineamiento: {detail.get('alignment_quality_score', 0):.3f}")
        print(f"   • Secciones alineadas: {detail.get('sections_aligned', 0)}")
        print(f"   • Secciones huérfanas: {detail.get('orphaned_translated', 0)}")
    
    print()
    
    # Mostrar reporte detallado
    print("📋 REPORTE DETALLADO:")
    print(validator.generate_integrity_report(report))
    
    # Demo de comparación con traducciones buenas vs malas
    print("\n" + "="*60)
    print("🔍 COMPARACIÓN: Traducción Buena vs Problemática")
    print("="*60)
    
    good_translation = """
    The concept of Dasein in Martin Heidegger's philosophy represents a fundamental
    way of understanding human existence. Dasein, literally "being-there",
    refers to the specific way in which human beings exist in the world.
    
    Unlike other entities, Dasein has the peculiar characteristic of
    questioning being itself. This capacity for ontological questioning
    is what distinguishes Dasein from other modes of being.
    
    Temporality is a central aspect of Dasein. Heidegger argues that
    Dasein always exists in a specific temporal relation, projecting itself
    toward the future while retaining its past and acting in the present.
    
    The concept of "thrownness" (Geworfenheit) describes how Dasein always finds
    itself already situated in a world it did not choose, but in which it must
    make authentic choices to realize its ownmost potential.
    """
    
    good_report = validator.validate_semantic_integrity(
        original_text, good_translation, "demo_good", "translation"
    )
    
    print(f"Traducción BUENA    - Score: {good_report.overall_integrity_score:.3f}")
    print(f"Traducción PROBLEMÁTICA - Score: {report.overall_integrity_score:.3f}")
    print(f"Diferencia: {good_report.overall_integrity_score - report.overall_integrity_score:.3f}")
    
    print("\n🎯 Demo completado!")
    print("💡 El validador mejorado detectó exitosamente:")
    print("   ✓ Contenido reordenado y mal alineado")
    print("   ✓ Secciones agregadas sin correspondencia") 
    print("   ✓ Diferencias de calidad entre traducciones")
    print("   ✓ Métricas de alineamiento bidireccional")


if __name__ == "__main__":
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    demo_semantic_validator()