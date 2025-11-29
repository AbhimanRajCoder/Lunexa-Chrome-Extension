import asyncio
import logging
import os
import random
from typing import List, Optional

import httpx
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Load environment variables
load_dotenv()

# Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
REALITY_API_KEY = os.getenv("REALITY_API_KEY")
REALITY_BASE = os.getenv("REALITY_BASE")

HEADERS = {
    "X-API-KEY": REALITY_API_KEY,
    "Content-Type": "application/json"
}

# Initialize Logger
logger = logging.getLogger("uvicorn.error")

# Initialize FastAPI App
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from chat import router as chat_router
app.include_router(chat_router)

from extenstion import router as extension_router
app.include_router(extension_router)



class TestInput(BaseModel):
    text: str


class ConversationItem(BaseModel):
    text: str
    sender: str


class QueryInput(BaseModel):
    message: str
    conversation_history: List[ConversationItem]


class PresignResponse(BaseModel):
    signedUrl: Optional[str] = None
    requestId: Optional[str] = None


# ================================
# 🔹 TEXT VERIFICATION ENDPOINT
# ================================

@app.post("/testai/{model_name}")
async def test_ai_model(model_name: str, data: TestInput):
    """
    Simulates AI model testing for text verification.
    """
    if not data.text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty")

    hallucination_score = round(random.uniform(0.0, 1.0), 2)
    metrics = {
        "accuracy": random.uniform(0.7, 1.0),
        "reliability": random.uniform(0.6, 1.0),
        "response_time": random.uniform(0.2, 1.5),
        "fact_check_confidence": random.uniform(0.5, 1.0),
    }
    
    hallucination_level = (
        'low' if hallucination_score < 0.3 
        else 'moderate' if hallucination_score < 0.7 
        else 'high'
    )
    
    analysis = (
        f"The model '{model_name}' performed moderately well. "
        f"Based on analysis, its hallucination score is {hallucination_score}, "
        f"indicating {hallucination_level} hallucination tendency."
    )

    return {
        "model": model_name,
        "hallucination_score": hallucination_score,
        "metrics": metrics,
        "analysis": analysis
    }


# ================================
# 🎥 VIDEO VERIFICATION ENDPOINT
# ================================

@app.post("/testvideo")
async def test_video(video: UploadFile = File(...)):
    """
    Simulates video verification for deepfakes.
    """
    if not video.filename:
        raise HTTPException(status_code=400, detail="No video uploaded")

    is_authentic = random.choice([True, False])
    confidence_score = round(random.uniform(0.5, 0.98), 2)

    manipulated_segments = []
    if not is_authentic:
        num_segments = random.randint(1, 3)
        for _ in range(num_segments):
            start = round(random.uniform(5, 50), 1)
            duration = round(random.uniform(2, 6), 1)
            manipulated_segments.append({
                "type": random.choice(["Deepfake Face", "Synthetic Audio", "Spliced Frame"]),
                "start_time": start,
                "end_time": start + duration,
                "confidence": round(random.uniform(0.7, 0.95), 2)
            })

    analysis = (
        f"The video '{video.filename}' is likely "
        f"{'authentic' if is_authentic else 'manipulated'} with a confidence of {confidence_score * 100:.1f}%."
    )

    return {
        "filename": video.filename,
        "authentic": is_authentic,
        "confidence_score": confidence_score,
        "analysis": analysis,
        "manipulated_segments": manipulated_segments
    }


# ================================
# 📤 FILE UPLOAD ENDPOINT
# ================================

@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    try_fetch_result: bool = Query(False, description="If true, attempt to fetch analysis result after upload"),
    fetch_timeout: int = Query(5, description="Seconds to wait when try_fetch_result=True (total polling time)"),
    debug: bool = Query(False, description="If true, return raw presign JSON for debugging")
):
    """
    Handles file upload via presigned URL and optionally fetches analysis results.
    """
    filename = file.filename or "upload.bin"
    presign_payload = {"fileName": filename}
    presign_url = f"{REALITY_BASE}/api/files/aws-presigned"

    async with httpx.AsyncClient(timeout=15) as client:
        # STEP 1: Request presigned URL
        presign_resp = await client.post(presign_url, json=presign_payload, headers=HEADERS)

        # Accept any 2xx as success
        if presign_resp.status_code < 200 or presign_resp.status_code >= 300:
            raise HTTPException(
                status_code=502, 
                detail=f"Failed to get presigned URL: status={presign_resp.status_code} body={presign_resp.text}"
            )

        # Parse JSON safely
        try:
            presign_full = presign_resp.json()
        except Exception as e:
            raise HTTPException(
                status_code=502, 
                detail=f"Presign response not JSON: {e} body={presign_resp.text}"
            )

        # Optional: expose raw presign body in debug mode
        if debug:
            return {"presign_status": presign_resp.status_code, "presign_body": presign_full}

        logger.info("Presign response JSON: %s", presign_full)

        # Common candidate containers to search for fields
        candidates = []
        if isinstance(presign_full, dict):
            candidates.append(presign_full)
            for key in ("response", "data", "result"):
                v = presign_full.get(key)
                if isinstance(v, dict):
                    candidates.append(v)

        signed_url = None
        request_id = None

        # Try common key names and casings
        for c in candidates:
            for key in ("signedUrl", "signed_url", "signedURL", "uploadUrl", "url"):
                if key in c:
                    signed_url = c.get(key)
                    break
            for key in ("requestId", "request_id", "requestID", "id"):
                if key in c:
                    request_id = c.get(key)
                    break
            if signed_url and request_id:
                break

        # Helpful error if missing
        if not signed_url or not request_id:
            raise HTTPException(
                status_code=502,
                detail={
                    "message": "Presign response missing signedUrl or requestId (checked common keys).",
                    "checked_candidates": candidates,
                    "full_response": presign_full
                }
            )

        # STEP 2: Upload file bytes
        file_bytes = await file.read()
        upload_resp = await client.put(signed_url, content=file_bytes, timeout=20)

        if upload_resp.status_code not in (200, 201, 204):
            raise HTTPException(
                status_code=502, 
                detail=f"Upload failed: status {upload_resp.status_code} body: {upload_resp.text}"
            )

        result_payload = {"requestId": request_id, "uploadStatus": upload_resp.status_code}

        # STEP 3: Optionally attempt to fetch analysis result (short poll)
        if try_fetch_result:
            check_url = f"{REALITY_BASE}/api/media/users/{request_id}"
            deadline = asyncio.get_event_loop().time() + max(0, fetch_timeout)
            analysis = None
            while asyncio.get_event_loop().time() < deadline:
                get_resp = await client.get(check_url, headers=HEADERS)
                if get_resp.status_code == 200:
                    analysis = get_resp.json()
                    break
                await asyncio.sleep(1)
            result_payload["analysis"] = analysis

        return result_payload


# ================================
# 🔍 DEEPFAKE RESULT ENDPOINT
# ================================

@app.get("/result/{request_id}")
async def get_deepfake_result(request_id: str):
    """
    Fetch ONLY deepfake-related analysis results for an image.
    """
    url = f"{REALITY_BASE}/api/media/users/{request_id}"

    async with httpx.AsyncClient(timeout=20) as client:
        resp = await client.get(url, headers=HEADERS)

        if resp.status_code != 200:
            raise HTTPException(
                status_code=resp.status_code,
                detail=f"Failed to fetch result: {resp.text}"
            )

        data = resp.json()

        # Filter ONLY deepfake image models
        DEEPFAKE_IMAGE_MODELS = {
            "rd-context-img",
            "rd-pine-img",
            "rd-img-ensemble",
            "rd-cedar-img",
            "rd-elm-img",
            "rd-oak-img"
        }

        deepfake_models = []
        for model in data.get("models", []):
            if model.get("name") in DEEPFAKE_IMAGE_MODELS:
                deepfake_models.append({
                    "name": model.get("name"),
                    "status": model.get("status"),
                    "score": model.get("predictionNumber"),
                    "finalScore": model.get("finalScore"),
                    "normalizedScore": model.get("normalizedPredictionNumber"),
                })

        return {
            "requestId": data.get("requestId"),
            "overallStatus": data.get("overallStatus"),
            "resultsSummary": data.get("resultsSummary"),
            "deepfakeModels": deepfake_models
        }


# ================================
# 🤖 MISBOT ENDPOINT
# ================================

@app.post("/misbot")
def misbot(input: QueryInput):
    """
    Chatbot endpoint using Gemini to check statement accuracy.
    """
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"
        headers = {"Content-Type": "application/json"}

        # Prepare conversation context for Gemini
        history = []
        for msg in input.conversation_history:
            role = "user" if msg.sender == "user" else "model"
            history.append({"role": role, "parts": [{"text": msg.text}]})

        # Add the latest message
        history.append({
            "role": "user",
            "parts": [{"text": f"Check the accuracy and authenticity of this statement:\n\n{input.message}"}]
        })

        payload = {"contents": history}

        response = requests.post(url, headers=headers, json=payload)
        data = response.json()

        # Extract the Gemini model’s text response safely
        ai_reply = (
            data.get("candidates", [{}])[0]
            .get("content", {})
            .get("parts", [{}])[0]
            .get("text", "No response from Gemini.")
        )

        return {"result": ai_reply}

    except Exception as e:
        return {"error": str(e)}


# ================================
# 🌍 ROOT ENDPOINT
# ================================

@app.get("/")
async def root():
    return {"message": "AI Fact-Checker Backend is running successfully!"}