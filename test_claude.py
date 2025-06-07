import os
from dotenv import load_dotenv
from anthropic import Anthropic

def test_claude_api():
    """Test básico de conexión con Claude API"""
    
    # Cargar variables de entorno
    load_dotenv()
    
    # Verificar que existe la API key
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key or api_key == "tu_api_key_aqui":
        print("❌ ERROR: API key no configurada en .env")
        return False
    
    try:
        # Inicializar cliente Claude
        print("🔄 Probando conexión con Claude...")
        client = Anthropic(api_key=api_key)
        
        # Test simple
        response = client.messages.create(
           model="claude-3-5-sonnet-20241022",
            max_tokens=50,
            messages=[{"role": "user", "content": "Responde solo: 'API funcionando correctamente'"}]
        )
        
        print("✅ Claude API funcionando!")
        print(f"📝 Respuesta: {response.content[0].text}")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    test_claude_api()