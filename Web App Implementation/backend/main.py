from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from engine import run_gemini_rag_pipeline

app = FastAPI()
executor = ThreadPoolExecutor(max_workers=4)

# Enable CORS for frontend access
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Track the last uploaded image path per session (simple in-memory for single-user dev)
_last_image_path: dict = {}

@app.post("/analyze")
async def analyze_palm(
    image: UploadFile = File(None),
    question: str = Form(...),
):
    file_path = _last_image_path.get("path", "")

    if image and image.filename:
        content = await image.read()
        if len(content) > 100 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="File too large")
        file_path = os.path.join(UPLOAD_DIR, image.filename)
        with open(file_path, "wb") as f:
            f.write(content)
        _last_image_path["path"] = file_path

    if not file_path or not os.path.exists(file_path):
        raise HTTPException(
            status_code=400,
            detail="No palm image found. Please upload a palm image first."
        )

    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            executor,
            run_gemini_rag_pipeline,
            file_path,
            question,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Serve frontend files
app.mount("/", StaticFiles(directory="../frontend", html=True), name="frontend")