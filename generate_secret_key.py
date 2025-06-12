#!/usr/bin/env python3
"""
Generador de SECRET_KEY para ClaudeAcademico v2.2
Genera claves criptográficamente seguras
"""

import secrets
import string
import hashlib
from datetime import datetime

def generate_secret_key(length=64):
    """Generar SECRET_KEY criptográficamente seguro"""
    # Usar caracteres seguros para la clave
    alphabet = string.ascii_letters + string.digits + "!@#$%^&*()_+-=[]{}|;:,.<>?"
    
    # Generar clave aleatoria
    secret_key = ''.join(secrets.choice(alphabet) for _ in range(length))
    
    return secret_key

def generate_multiple_keys():
    """Generar múltiples opciones de SECRET_KEY"""
    print("🔐 ClaudeAcademico v2.2 - SECRET_KEY Generator")
    print("=" * 50)
    print()
    
    print("🎲 Generando claves seguras...")
    print()
    
    # Opción 1: Clave estándar (64 caracteres)
    key1 = generate_secret_key(64)
    print("✅ OPCIÓN 1 (Recomendada para desarrollo):")
    print(f"SECRET_KEY={key1}")
    print()
    
    # Opción 2: Clave extra segura (128 caracteres)
    key2 = generate_secret_key(128)
    print("🔒 OPCIÓN 2 (Extra segura para producción):")
    print(f"SECRET_KEY={key2}")
    print()
    
    # Opción 3: Basada en tiempo + hash
    timestamp = str(datetime.now().timestamp()).replace('.', '')
    random_part = secrets.token_urlsafe(32)
    key3 = hashlib.sha256(f"{timestamp}_{random_part}_claudeacademico".encode()).hexdigest()
    print("⏰ OPCIÓN 3 (Basada en timestamp):")
    print(f"SECRET_KEY={key3}")
    print()
    
    # Opción 4: Solo letras y números (compatible con todos los sistemas)
    alphabet_simple = string.ascii_letters + string.digits
    key4 = ''.join(secrets.choice(alphabet_simple) for _ in range(64))
    print("📱 OPCIÓN 4 (Solo alfanumérico - máxima compatibilidad):")
    print(f"SECRET_KEY={key4}")
    print()
    
    print("💡 INSTRUCCIONES:")
    print("1. Copia UNA de las claves de arriba")
    print("2. Pégala en tu archivo .env reemplazando 'tu_secret_key_seguro_aqui_cambiar_en_produccion'")
    print("3. ¡Guarda el archivo .env!")
    print()
    print("⚠️  IMPORTANTE:")
    print("- NUNCA compartas esta clave")
    print("- Usa claves diferentes para desarrollo y producción")
    print("- Cambia la clave si crees que fue comprometida")
    print()
    
    return key1  # Retornar la primera opción como default

if __name__ == "__main__":
    generate_multiple_keys()