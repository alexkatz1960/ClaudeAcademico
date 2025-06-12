from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil
from pathlib import Path
import os
import requests
from datetime import datetime

app = FastAPI(title="ClaudeAcademico API", version="2.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure directories exist
for directory in ["uploads", "output", "temp"]:
    Path(f"/app/{directory}").mkdir(exist_ok=True)

@app.get("/")
def root():
    return {
        "message": "ClaudeAcademico API v2.2", 
        "status": "running",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
def health():
    return {"status": "healthy", "service": "ClaudeAcademico"}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Solo archivos PDF")
    
    upload_dir = Path("/app/uploads")
    file_path = upload_dir / file.filename
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    return {
        "message": "Archivo subido exitosamente",
        "filename": file.filename,
        "size": file.size,
        "path": str(file_path)
    }

@app.post("/translate")
async def translate_text(text: str, source_lang: str = "de", target_lang: str = "es"):
    deepl_key = os.getenv("DEEPL_API_KEY")
    
    if not deepl_key:
        raise HTTPException(status_code=500, detail="DeepL API key no configurada")
    
    try:
        url = "https://api.deepl.com/v2/translate"
        data = {
            'auth_key': deepl_key,
            'text': text,
            'source_lang': source_lang.upper(),
            'target_lang': target_lang.upper()
        }
        
        response = requests.post(url, data=data)
        
        if response.status_code == 200:
            result = response.json()
            translated_text = result['translations'][0]['text']
            
            return {
                "original": text,
                "translated": translated_text,
                "source_lang": source_lang,
                "target_lang": target_lang,
                "status": "success"
            }
        else:
            raise HTTPException(status_code=500, detail=f"Error de DeepL: {response.text}")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error de traducción: {str(e)}")

@app.get("/files")
def list_files():
    upload_dir = Path("/app/uploads")
    files = []
    
    if upload_dir.exists():
        for file_path in upload_dir.iterdir():
            if file_path.is_file():
                files.append({
                    "name": file_path.name,
                    "size": file_path.stat().st_size,
                    "modified": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                })
    
    return {"files": files}

@app.get("/test-deepl")
def test_deepl():
    deepl_key = os.getenv("DEEPL_API_KEY")
    
    if not deepl_key:
        return {"status": "error", "message": "DeepL API key no configurada"}
    
    try:
        url = "https://api.deepl.com/v2/usage"
        headers = {"Authorization": f"DeepL-Auth-Key {deepl_key}"}
        
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            usage = response.json()
            return {
                "status": "connected",
                "usage": f"{usage['character_count']:,}/{usage['character_limit']:,}",
                "percentage": round((usage['character_count'] / usage['character_limit']) * 100, 2)
            }
        else:
            return {"status": "error", "message": f"Error: {response.status_code}"}
            
    except Exception as e:
        return {"status": "error", "message": str(e)}