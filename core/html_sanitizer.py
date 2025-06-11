"""
Sanitizador HTML Enterprise-Grade para Translation System v2.2
Prevención de ataques XSS y validación de contenido HTML
"""

import bleach
import logging
from typing import List, Dict, Optional
import re

class HTMLSanitizer:
    """Sanitizador HTML seguro para contenido académico"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Tags HTML permitidos para contenido académico
        self.ALLOWED_TAGS = [
            'p', 'div', 'span', 'br',                    # Párrafos y contenedores
            'strong', 'b', 'em', 'i', 'u',              # Formato de texto
            'h1', 'h2', 'h3', 'h4', 'h5', 'h6',        # Encabezados
            'blockquote', 'cite',                        # Citas académicas
            'ul', 'ol', 'li',                           # Listas
            'table', 'thead', 'tbody', 'tr', 'td', 'th', # Tablas académicas
            'sup', 'sub',                               # Superíndices/subíndices
            'a'                                         # Enlaces (con restricciones)
        ]
        
        # Atributos permitidos por tag
        self.ALLOWED_ATTRIBUTES = {
            'a': ['href', 'title'],
            'blockquote': ['cite'],
            'table': ['class'],
            'td': ['colspan', 'rowspan'],
            'th': ['colspan', 'rowspan'],
            '*': ['class', 'id']  # Atributos globales permitidos
        }
        
        # Protocolos seguros para enlaces
        self.ALLOWED_PROTOCOLS = ['http', 'https', 'mailto']
    
    def sanitize_html(self, html_content: str) -> str:
        """
        Sanitizar contenido HTML removiendo elementos peligrosos
        
        Args:
            html_content: HTML sin sanitizar
            
        Returns:
            HTML sanitizado y seguro
        """
        if not html_content or not isinstance(html_content, str):
            return ""
        
        try:
            # Sanitización con bleach
            clean_html = bleach.clean(
                html_content,
                tags=self.ALLOWED_TAGS,
                attributes=self.ALLOWED_ATTRIBUTES,
                protocols=self.ALLOWED_PROTOCOLS,
                strip=True,  # Remover tags no permitidos en lugar de escaparlos
                strip_comments=True  # Remover comentarios HTML
            )
            
            # Validaciones adicionales
            clean_html = self._additional_security_checks(clean_html)
            
            self.logger.debug(f"HTML sanitizado: {len(html_content)} -> {len(clean_html)} chars")
            return clean_html
            
        except Exception as e:
            self.logger.error(f"Error sanitizando HTML: {e}")
            # En caso de error, devolver texto plano sin HTML
            return bleach.clean(html_content, tags=[], strip=True)
    
    def _additional_security_checks(self, html: str) -> str:
        """Validaciones de seguridad adicionales"""
        
        # Remover javascript: URLs que pudieran escapar la sanitización inicial
        html = re.sub(r'javascript\s*:', '', html, flags=re.IGNORECASE)
        
        # Remover data: URLs excepto imágenes simples
        html = re.sub(r'data:(?!image/(png|jpg|jpeg|gif|svg))', '', html, flags=re.IGNORECASE)
        
        # Remover event handlers que pudieran quedar
        dangerous_patterns = [
            r'on\w+\s*=',  # onclick, onload, etc.
            r'style\s*=.*expression',  # CSS expressions
            r'style\s*=.*javascript'   # JavaScript en CSS
        ]
        
        for pattern in dangerous_patterns:
            html = re.sub(pattern, '', html, flags=re.IGNORECASE)
        
        return html
    
    def validate_html_safety(self, html_content: str) -> Dict[str, any]:
        """
        Validar si el HTML contiene elementos potencialmente peligrosos
        
        Returns:
            Dict con información de seguridad
        """
        if not html_content:
            return {"safe": True, "issues": []}
        
        issues = []
        
        # Detectar scripts
        if re.search(r'<script', html_content, re.IGNORECASE):
            issues.append("Scripts JavaScript detectados")
        
        # Detectar event handlers
        if re.search(r'on\w+\s*=', html_content, re.IGNORECASE):
            issues.append("Event handlers JavaScript detectados")
        
        # Detectar iframes
        if re.search(r'<iframe', html_content, re.IGNORECASE):
            issues.append("iframes detectados")
        
        # Detectar object/embed
        if re.search(r'<(object|embed)', html_content, re.IGNORECASE):
            issues.append("Objetos embebidos detectados")
        
        return {
            "safe": len(issues) == 0,
            "issues": issues,
            "original_length": len(html_content)
        }

# Instancia global del sanitizador
html_sanitizer = HTMLSanitizer()