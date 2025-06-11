"""
Sistema de Configuración Segura para Translation System v2.2
Manejo de variables de entorno y configuración enterprise-grade
"""

import os
import logging
from typing import Optional
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

class ConfigurationError(Exception):
    """Error de configuración crítica del sistema"""
    pass

class Config:
    """Configuración centralizada y segura del sistema"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._validate_critical_config()
    
    # APIs Externas - CRÍTICAS
    @property
    def DEEPL_API_KEY(self) -> str:
        key = os.getenv("DEEPL_API_KEY")
        if not key:
            raise ConfigurationError("DEEPL_API_KEY no encontrada en variables de entorno")
        return key
    
    @property
    def CLAUDE_API_KEY(self) -> str:
        key = os.getenv("CLAUDE_API_KEY") 
        if not key:
            raise ConfigurationError("CLAUDE_API_KEY no encontrada en variables de entorno")
        return key
    
    @property
    def ABBYY_API_KEY(self) -> str:
        key = os.getenv("ABBYY_API_KEY")
        if not key:
            raise ConfigurationError("ABBYY_API_KEY no encontrada en variables de entorno")
        return key
    
    # Configuración de Base de Datos
    @property
    def DATABASE_PATH(self) -> str:
        return os.getenv("DATABASE_PATH", "translation_system.db")
    
    # Configuración de Cache
    @property
    def REDIS_URL(self) -> str:
        return os.getenv("REDIS_URL", "redis://localhost:6379/0")
    
    # Directorios de trabajo
    @property
    def WORKSPACE_DIR(self) -> str:
        return os.getenv("WORKSPACE_DIR", "./workspace")
    
    @property
    def TEMP_DIR(self) -> str:
        return os.getenv("TEMP_DIR", "./temp")
    
    def _validate_critical_config(self):
        """Validar configuración crítica al inicio"""
        try:
            # Solo validar APIs que tienen valores configurados
            if os.getenv("CLAUDE_API_KEY"):
                _ = self.CLAUDE_API_KEY
                self.logger.info("✅ CLAUDE_API_KEY configurada")
            
            if os.getenv("DEEPL_API_KEY"):
                _ = self.DEEPL_API_KEY
                self.logger.info("✅ DEEPL_API_KEY configurada")
            else:
                self.logger.warning("⚠️ DEEPL_API_KEY no configurada (se configurará más adelante)")
                
            if os.getenv("ABBYY_API_KEY"):
                _ = self.ABBYY_API_KEY
                self.logger.info("✅ ABBYY_API_KEY configurada")
            else:
                self.logger.warning("⚠️ ABBYY_API_KEY no configurada (se configurará más adelante)")
                
            self.logger.info("✅ Configuración validada exitosamente")
        except ConfigurationError as e:
            self.logger.critical(f"❌ Error de configuración crítica: {e}")
            raise

# Instancia global de configuración
config = Config()