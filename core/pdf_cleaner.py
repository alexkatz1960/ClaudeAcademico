"""
AdvancedPDFCleaner - Sistema inteligente de limpieza de PDFs
Detecta y remueve artifacts que ABBYY/Adobe no identifican
"""

import re
import logging
from typing import List, Dict, Tuple, Any, Optional
from collections import Counter
from dataclasses import dataclass
from datetime import datetime

@dataclass
class CleanupReport:
    """Reporte detallado del proceso de limpieza"""
    original_lines: int
    removed_lines: int
    artifacts_detected: List[str]
    patterns_found: Dict[str, int]
    effectiveness_score: float
    processing_time_seconds: float
    timestamp: str

class PatternFrequencyAnalyzer:
    """Analiza frecuencia de patrones para detectar elementos repetitivos"""
    
    def __init__(self, min_frequency: int = 3):
        self.min_frequency = min_frequency
        self.logger = logging.getLogger(__name__)
    
    def find_repetitive_elements(self, text_lines: List[str]) -> List[str]:
        """Identifica elementos que se repiten en múltiples páginas"""
        # Tolerancia a fallos: filtrar inputs inválidos
        valid_lines = []
        for line in text_lines:
            if isinstance(line, str) and line.strip():
                valid_lines.append(line.strip())
        
        # Filtrar líneas cortas que podrían ser headers/footers
        short_lines = [
            line for line in valid_lines 
            if 3 <= len(line) <= 80
        ]
        
        # Contar frecuencias
        line_counts = Counter(short_lines)
        
        # Retornar líneas que aparecen frecuentemente
        repetitive = [
            line for line, count in line_counts.items() 
            if count >= self.min_frequency
        ]
        
        self.logger.info(f"Detectados {len(repetitive)} elementos repetitivos")
        return repetitive

class HeuristicLineClassifier:
    """
    Clasificador heurístico para identificar headers, footers y artifacts
    Nota: Usa heurísticas ponderadas, no machine learning tradicional
    """
    
    def __init__(self):
        self.academic_patterns = [
            r'^Page \d+$',                    # "Page 23"
            r'^\d{1,3}$',                     # Números de página aislados
            r'^Chapter \d+',                  # "Chapter 5"
            r'^\d+\s+[A-Z\s]{10,50}$',      # "12 INTRODUCTION TO THEORY"
            r'^[A-Z\s]{20,}$',               # Headers en mayúsculas
            r'.*\d{4}.*',                    # Líneas con años
            r'^[A-Za-z\s]+\|\s*\d+$',       # "Author Name | 45"
            r'^\s*\d+\s*$',                  # Solo números (folios)
            r'^.*©.*\d{4}.*$',               # Copyright notices
            r'^.*Press$|^.*University$',     # Nombres de editoriales
            r'Dewey_\d+_\d+pp',              # Patrón específico detectado en auditoría
        ]
        
        self.watermark_patterns = [
            r'^\s*watermark\s*$',
            r'^\s*draft\s*$',
            r'^\s*confidential\s*$',
        ]
        
        self.logger = logging.getLogger(__name__)
    
    def classify_line(self, line: str) -> Dict[str, float]:
        """Clasifica una línea como artifact o contenido"""
        # Tolerancia a fallos: validar input
        if not isinstance(line, str):
            return {'is_artifact': 0.0, 'is_content': 0.0, 'confidence': 0.0}
        
        line_clean = line.strip()
        if not line_clean:
            return {'is_artifact': 1.0, 'is_content': 0.0, 'confidence': 1.0}
        
        artifact_score = 0.0
        content_score = 0.0
        
        # Verificar patrones académicos específicos
        for pattern in self.academic_patterns:
            if re.match(pattern, line_clean, re.IGNORECASE):
                artifact_score += 0.3
                break
        
        # Verificar watermarks
        for pattern in self.watermark_patterns:
            if re.match(pattern, line_clean, re.IGNORECASE):
                artifact_score += 0.4
                break
        
        # Características estructurales
        if len(line_clean) < 10:
            artifact_score += 0.2
        elif len(line_clean) > 200:
            content_score += 0.2
        
        # Ratio de números vs texto
        number_ratio = len(re.findall(r'\d', line_clean)) / max(len(line_clean), 1)
        if number_ratio > 0.3:
            artifact_score += 0.2
        
        # Palabras de contenido académico
        academic_words = ['analysis', 'theory', 'concept', 'research', 'study', 'method']
        if any(word in line_clean.lower() for word in academic_words):
            content_score += 0.3
        
        # Puntuación y estructura de oración
        if re.search(r'[.!?]', line_clean) and len(line_clean.split()) > 5:
            content_score += 0.2
        
        # Normalizar scores
        total_score = artifact_score + content_score
        if total_score > 0:
            artifact_score /= total_score
            content_score /= total_score
        
        return {
            'is_artifact': artifact_score,
            'is_content': content_score,
            'confidence': abs(artifact_score - content_score)
        }

class AdvancedPDFCleaner:
    """
    Sistema avanzado de limpieza de PDFs con análisis heurístico
    Detecta y remueve artifacts que ABBYY/Adobe no identifican
    """
    
    def __init__(self, artifact_threshold: float = 0.6):
        self.pattern_detector = PatternFrequencyAnalyzer()
        self.heuristic_classifier = HeuristicLineClassifier()  # Nombre actualizado
        self.artifact_threshold = artifact_threshold
        self.logger = logging.getLogger(__name__)
        
        # Configurar logging
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def heuristic_cleanup(self, text_lines: List[str]) -> Tuple[List[str], CleanupReport]:
        """
        Limpieza heurística principal del documento
        
        Args:
            text_lines: Lista de líneas del documento
            
        Returns:
            Tuple con líneas limpias y reporte detallado
        """
        start_time = datetime.now()
        
        # Tolerancia a fallos: filtrar inputs inválidos
        valid_lines = []
        for line in text_lines:
            if isinstance(line, str):
                valid_lines.append(line)
            elif line is not None:
                # Convertir a string si no es None
                valid_lines.append(str(line))
        
        self.logger.info(f"Iniciando limpieza de {len(valid_lines)} líneas válidas (de {len(text_lines)} originales)")
        
        # 1. Encontrar elementos repetitivos
        repetitive_elements = self.pattern_detector.find_repetitive_elements(valid_lines)
        
        # 2. Clasificar cada línea
        cleaned_lines = []
        removed_artifacts = []
        patterns_found = {}
        
        for i, line in enumerate(valid_lines):
            # Tolerancia a fallos adicional
            try:
                line_clean = line.strip()
            except AttributeError:
                continue
            
            # Saltar líneas vacías
            if not line_clean:
                continue
            
            # Verificar si es elemento repetitivo
            if line_clean in repetitive_elements:
                removed_artifacts.append(f"Repetitive: {line_clean}")
                patterns_found['repetitive'] = patterns_found.get('repetitive', 0) + 1
                continue
            
            # Clasificar con heurísticas
            classification = self.heuristic_classifier.classify_line(line_clean)
            
            if classification['is_artifact'] > self.artifact_threshold:
                removed_artifacts.append(f"Heuristic Artifact: {line_clean}")
                patterns_found['heuristic_artifact'] = patterns_found.get('heuristic_artifact', 0) + 1
                self.logger.debug(f"Removido artifact (score: {classification['is_artifact']:.3f}): {line_clean[:50]}...")
            else:
                cleaned_lines.append(line)
                patterns_found['content'] = patterns_found.get('content', 0) + 1
        
        # 3. Generar reporte
        processing_time = (datetime.now() - start_time).total_seconds()
        
        report = CleanupReport(
            original_lines=len(text_lines),
            removed_lines=len(removed_artifacts),
            artifacts_detected=removed_artifacts,
            patterns_found=patterns_found,
            effectiveness_score=len(removed_artifacts) / max(len(text_lines), 1),
            processing_time_seconds=processing_time,
            timestamp=datetime.now().isoformat()
        )
        
        self.logger.info(f"Limpieza completada: {len(cleaned_lines)} líneas conservadas, "
                        f"{len(removed_artifacts)} artifacts removidos "
                        f"({report.effectiveness_score:.1%} efectividad)")
        
        return cleaned_lines, report
    
    def analyze_document_structure(self, text_lines: List[str]) -> Dict[str, Any]:
        """Analiza la estructura del documento para optimizar limpieza"""
        
        # Tolerancia a fallos: filtrar inputs inválidos
        valid_lines = [line for line in text_lines if isinstance(line, str)]
        
        structure_analysis = {
            'total_lines': len(text_lines),
            'valid_lines': len(valid_lines),
            'empty_lines': sum(1 for line in valid_lines if not line.strip()),
            'short_lines': sum(1 for line in valid_lines if len(line.strip()) < 20),
            'long_lines': sum(1 for line in valid_lines if len(line.strip()) > 100),
            'numeric_lines': sum(1 for line in valid_lines if re.match(r'^\s*\d+\s*$', line)),
            'uppercase_lines': sum(1 for line in valid_lines if line.strip().isupper() and len(line.strip()) > 5),
        }
        
        # Detectar patrones de paginación
        page_patterns = [
            r'page\s+\d+',
            r'^\s*\d+\s*$',
            r'chapter\s+\d+',
        ]
        
        structure_analysis['pagination_indicators'] = sum(
            1 for line in valid_lines 
            for pattern in page_patterns 
            if re.search(pattern, line, re.IGNORECASE)
        )
        
        return structure_analysis
    
    def get_cleanup_data(self, report: CleanupReport) -> Dict[str, Any]:
        """
        Genera datos estructurados del reporte de limpieza
        Separado del formateo para permitir diferentes tipos de export
        """
        return {
            'metadata': {
                'timestamp': report.timestamp,
                'processing_time_seconds': report.processing_time_seconds,
                'effectiveness_percentage': round(report.effectiveness_score * 100, 1)
            },
            'statistics': {
                'original_lines': report.original_lines,
                'removed_lines': report.removed_lines,
                'preserved_lines': report.original_lines - report.removed_lines,
                'patterns_found': report.patterns_found
            },
            'artifacts': {
                'total_detected': len(report.artifacts_detected),
                'sample_artifacts': report.artifacts_detected[:10],  # Primeros 10
                'has_more': len(report.artifacts_detected) > 10,
                'additional_count': max(0, len(report.artifacts_detected) - 10)
            }
        }
    
    def format_cleanup_summary(self, cleanup_data: Dict[str, Any]) -> str:
        """
        Formatea los datos de limpieza como string legible
        Recibe output de get_cleanup_data()
        """
        metadata = cleanup_data['metadata']
        stats = cleanup_data['statistics']
        artifacts = cleanup_data['artifacts']
        
        summary = f"""
=== REPORTE DE LIMPIEZA PDF ===
Timestamp: {metadata['timestamp']}
Tiempo de procesamiento: {metadata['processing_time_seconds']:.2f} segundos

ESTADÍSTICAS:
- Líneas originales: {stats['original_lines']}
- Líneas removidas: {stats['removed_lines']}
- Líneas conservadas: {stats['preserved_lines']}
- Efectividad: {metadata['effectiveness_percentage']}%

PATRONES DETECTADOS:
"""
        
        for pattern_type, count in stats['patterns_found'].items():
            summary += f"- {pattern_type}: {count} líneas\n"
        
        if artifacts['sample_artifacts']:
            summary += f"\nPRIMEROS ARTIFACTS REMOVIDOS (máximo 10):\n"
            for artifact in artifacts['sample_artifacts']:
                summary += f"- {artifact}\n"
            
            if artifacts['has_more']:
                summary += f"... y {artifacts['additional_count']} más\n"
        
        return summary
    
    def generate_cleanup_summary(self, report: CleanupReport) -> str:
        """
        Método de compatibilidad que mantiene la interfaz original
        Internamente usa la nueva separación de concerns
        """
        cleanup_data = self.get_cleanup_data(report)
        return self.format_cleanup_summary(cleanup_data)


# Función de utilidad para testing rápido
def test_pdf_cleaner():
    """Test básico del PDF Cleaner con casos edge incluidos"""
    
    # Datos de prueba con casos edge
    sample_text = [
        "Page 23",  # Header
        "Esta es una línea de contenido real con sustancia académica que debe preservarse",
        None,  # Input inválido
        "23 INTRODUCTION TO PHILOSOPHY",  # Header académico
        "",  # Línea vacía
        "El concepto de Dasein en Heidegger se refiere a la forma específica en que los seres humanos existen",
        123,  # Input no-string (será convertido)
        "© 2024 Academic Press",  # Copyright
        "Otro párrafo con contenido académico valioso para el análisis filosófico",
        "Page 24",  # Header repetido
        "Dewey_7102_1pp",  # Marca específica detectada en auditoría
        "Este es contenido genuino que debe mantenerse en el documento final",
    ]
    
    # Ejecutar limpieza
    cleaner = AdvancedPDFCleaner()
    cleaned_lines, report = cleaner.heuristic_cleanup(sample_text)
    
    # Mostrar resultados usando nueva separación de concerns
    print("=== TEST PDF CLEANER MEJORADO ===")
    
    # Usar nuevo método estructurado
    cleanup_data = cleaner.get_cleanup_data(report)
    print(f"Líneas originales: {cleanup_data['statistics']['original_lines']}")
    print(f"Líneas limpias: {len(cleaned_lines)}")
    print(f"Artifacts removidos: {cleanup_data['statistics']['removed_lines']}")
    print(f"Efectividad: {cleanup_data['metadata']['effectiveness_percentage']}%")
    
    print("\nCONTENIDO LIMPIO:")
    for i, line in enumerate(cleaned_lines, 1):
        print(f"{i}. {line}")
    
    # Mostrar reporte formateado
    print("\n" + cleaner.format_cleanup_summary(cleanup_data))
    
    # Test de análisis de estructura
    print("\n=== ANÁLISIS DE ESTRUCTURA ===")
    structure = cleaner.analyze_document_structure(sample_text)
    for key, value in structure.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    test_pdf_cleaner()