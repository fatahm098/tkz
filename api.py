"""
🚀 REST API untuk SMS Spam Detector Indonesia
Menggunakan FastAPI + IndoBERT

Fitur:
✅ Single text prediction
✅ Batch prediction
✅ Health check endpoint
✅ Model info endpoint
✅ Static file serving untuk frontend
✅ Auto-generated API docs (Swagger)

Install:
pip install fastapi uvicorn torch transformers pydantic pandas

Run:
uvicorn api:app --host 0.0.0.0 --port 7860

Akses Docs:
http://localhost:7860/docs (Swagger UI)
http://localhost:7860/redoc (ReDoc)
"""

from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import List, Optional
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re
from datetime import datetime
import pandas as pd
from io import StringIO
import logging
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# FASTAPI APP INITIALIZATION
# ============================================================================

app = FastAPI(
    title="SMS Spam Detector API",
    description="API untuk deteksi spam SMS Bahasa Indonesia menggunakan IndoBERT",
    version="1.0.0",
    contact={
        "name": "SMS Spam Detector Team",
        "email": "your.email@example.com",
    },
    license_info={
        "name": "MIT License",
    }
)

# CORS Middleware (untuk akses dari frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Untuk HF Spaces, allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# PYDANTIC MODELS (Request/Response Schema)
# ============================================================================

class SMSInput(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000, description="Teks SMS yang akan dideteksi")
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "SELAMAT! Anda menang 10 juta rupiah. Klik link berikut untuk klaim hadiah."
            }
        }

class SMSPrediction(BaseModel):
    text: str = Field(..., description="Teks SMS original")
    prediction: str = Field(..., description="Hasil prediksi: SPAM atau NON-SPAM")
    prediction_code: int = Field(..., description="Kode prediksi: 0=NON-SPAM, 1=SPAM")
    confidence: float = Field(..., description="Tingkat keyakinan model (0.0 - 1.0)")
    confidence_percentage: str = Field(..., description="Confidence dalam bentuk persentase")
    cleaned_text: str = Field(..., description="Teks setelah preprocessing")
    timestamp: str = Field(..., description="Waktu prediksi")
    warning: Optional[str] = Field(None, description="Peringatan jika terdeteksi SPAM")

class BatchSMSInput(BaseModel):
    texts: List[str] = Field(..., min_items=1, max_items=1000, description="List SMS yang akan dideteksi")
    
    class Config:
        json_schema_extra = {
            "example": {
                "texts": [
                    "Rapat hari ini jam 2 siang",
                    "PROMO GAJIAN! Diskon 50%",
                    "Terima kasih sudah berbelanja"
                ]
            }
        }

class BatchSMSPrediction(BaseModel):
    total: int = Field(..., description="Total SMS yang diproses")
    spam_count: int = Field(..., description="Jumlah SMS yang terdeteksi SPAM")
    non_spam_count: int = Field(..., description="Jumlah SMS yang terdeteksi NON-SPAM")
    spam_rate: float = Field(..., description="Persentase SPAM")
    results: List[SMSPrediction] = Field(..., description="Detail hasil untuk setiap SMS")
    timestamp: str = Field(..., description="Waktu pemrosesan")

class HealthCheck(BaseModel):
    status: str = Field(..., description="Status API")
    model_loaded: bool = Field(..., description="Status model")
    timestamp: str = Field(..., description="Waktu check")

class ModelInfo(BaseModel):
    model_name: str = Field(..., description="Nama model")
    model_type: str = Field(..., description="Tipe model")
    num_labels: int = Field(..., description="Jumlah label/kelas")
    max_length: int = Field(..., description="Panjang maksimal input")
    classes: dict = Field(..., description="Mapping label")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def clean_text(text: str) -> str:
    """Preprocessing text sesuai training"""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'\d{10,}', ' nomorpanjang ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ============================================================================
# LOAD MODEL (at startup)
# ============================================================================

tokenizer = None
model = None
MODEL_REPO = "fatahm0987/indobert-spam-sms"

@app.on_event("startup")
async def load_model():
    """Load model dari HuggingFace Hub saat startup"""
    global tokenizer, model
    
    try:
        logger.info(f"🚀 Loading model from HuggingFace: {MODEL_REPO}")
        
        tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_REPO)
        
        # Set to eval mode and move to CPU (HF Spaces biasanya CPU)
        model.eval()
        
        # Jika ada GPU, gunakan GPU
        if torch.cuda.is_available():
            model = model.cuda()
            logger.info("✅ Model loaded on GPU")
        else:
            logger.info("✅ Model loaded on CPU")
        
        logger.info("✅ Model successfully loaded from HuggingFace Hub")
    
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        raise RuntimeError(f"Model loading failed: {e}")

# ============================================================================
# PREDICTION FUNCTION
# ============================================================================

def predict_spam(text: str) -> tuple:
    """
    Prediksi spam menggunakan IndoBERT
    Returns: (prediction_code, confidence)
    """
    if tokenizer is None or model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet. Please wait...")
    
    cleaned = clean_text(text)
    
    if not cleaned or len(cleaned.strip()) == 0:
        raise HTTPException(status_code=400, detail="Text kosong setelah preprocessing")
    
    inputs = tokenizer(
        cleaned,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=128
    )
    
    # Move to same device as model
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred].item()
    
    return pred, confidence

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/", response_class=HTMLResponse, tags=["Root"])
async def read_root():
    """
    Serve the main HTML page
    """
    # Check if index.html exists
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    
    # Fallback HTML if index.html not found
    return HTMLResponse(content="""
    <!DOCTYPE html>
    <html>
    <head>
        <title>SMS Spam Detector API</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 800px;
                margin: 50px auto;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
            }
            .container {
                background: rgba(255,255,255,0.1);
                padding: 30px;
                border-radius: 15px;
                backdrop-filter: blur(10px);
            }
            h1 { font-size: 2.5em; margin-bottom: 10px; }
            .links { margin-top: 30px; }
            .links a {
                display: inline-block;
                margin: 10px;
                padding: 15px 30px;
                background: white;
                color: #667eea;
                text-decoration: none;
                border-radius: 25px;
                font-weight: bold;
            }
            .links a:hover {
                background: #f0f0f0;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚨 SMS Spam Detector API</h1>
            <p>API untuk deteksi spam SMS Bahasa Indonesia menggunakan IndoBERT</p>
            <div class="links">
                <a href="/docs">📚 API Documentation (Swagger)</a>
                <a href="/redoc">📖 ReDoc</a>
                <a href="/health">💚 Health Check</a>
            </div>
        </div>
    </body>
    </html>
    """)

@app.get("/health", response_model=HealthCheck, tags=["Health"])
async def health_check():
    """Check API health dan status model"""
    return HealthCheck(
        status="healthy" if model is not None else "loading",
        model_loaded=model is not None,
        timestamp=datetime.now().isoformat()
    )

@app.get("/model/info", response_model=ModelInfo, tags=["Model"])
async def model_info():
    """Informasi tentang model yang digunakan"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    
    return ModelInfo(
        model_name="IndoBERT Base",
        model_type="BERT Transformer",
        num_labels=2,
        max_length=128,
        classes={
            "0": "NON-SPAM (Normal)",
            "1": "SPAM (Fraud/Promo)"
        }
    )

@app.post("/predict", response_model=SMSPrediction, tags=["Prediction"])
async def predict(sms: SMSInput):
    """
    Deteksi spam untuk single SMS
    
    **Input:**
    - text: Teks SMS yang akan dideteksi
    
    **Output:**
    - prediction: SPAM atau NON-SPAM
    - confidence: Tingkat keyakinan (0.0 - 1.0)
    - warning: Peringatan jika terdeteksi SPAM
    """
    try:
        pred_code, confidence = predict_spam(sms.text)
        
        prediction_label = "SPAM" if pred_code == 1 else "NON-SPAM"
        warning = None
        
        if pred_code == 1:
            warning = "⚠️ PERINGATAN: SMS ini terdeteksi sebagai SPAM. Jangan klik link atau berikan informasi pribadi!"
        
        return SMSPrediction(
            text=sms.text,
            prediction=prediction_label,
            prediction_code=pred_code,
            confidence=confidence,
            confidence_percentage=f"{confidence*100:.2f}%",
            cleaned_text=clean_text(sms.text),
            timestamp=datetime.now().isoformat(),
            warning=warning
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/batch", response_model=BatchSMSPrediction, tags=["Prediction"])
async def batch_predict(batch: BatchSMSInput):
    """
    Deteksi spam untuk multiple SMS sekaligus
    
    **Input:**
    - texts: List berisi SMS yang akan dideteksi (max 1000)
    
    **Output:**
    - Statistik keseluruhan
    - Detail hasil untuk setiap SMS
    """
    try:
        results = []
        spam_count = 0
        
        for text in batch.texts:
            try:
                pred_code, confidence = predict_spam(text)
                prediction_label = "SPAM" if pred_code == 1 else "NON-SPAM"
                
                if pred_code == 1:
                    spam_count += 1
                
                warning = None
                if pred_code == 1:
                    warning = "⚠️ SPAM terdeteksi!"
                
                results.append(SMSPrediction(
                    text=text,
                    prediction=prediction_label,
                    prediction_code=pred_code,
                    confidence=confidence,
                    confidence_percentage=f"{confidence*100:.2f}%",
                    cleaned_text=clean_text(text),
                    timestamp=datetime.now().isoformat(),
                    warning=warning
                ))
            except Exception as e:
                logger.error(f"Error processing text: {text[:50]}... - {e}")
                continue
        
        total = len(results)
        non_spam_count = total - spam_count
        spam_rate = (spam_count / total * 100) if total > 0 else 0
        
        return BatchSMSPrediction(
            total=total,
            spam_count=spam_count,
            non_spam_count=non_spam_count,
            spam_rate=spam_rate,
            results=results,
            timestamp=datetime.now().isoformat()
        )
    
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.post("/predict/csv", tags=["Prediction"])
async def predict_csv(file: UploadFile = File(...)):
    """
    Upload CSV file untuk batch prediction
    
    **Input:**
    - CSV file dengan kolom 'text' yang berisi SMS
    
    **Output:**
    - CSV file dengan hasil prediksi
    """
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File harus berformat CSV")
    
    try:
        contents = await file.read()
        df = pd.read_csv(StringIO(contents.decode('utf-8')))
        
        if 'text' not in df.columns:
            raise HTTPException(
                status_code=400,
                detail="CSV harus memiliki kolom 'text'"
            )
        
        results = []
        for text in df['text']:
            try:
                pred_code, confidence = predict_spam(text)
                results.append({
                    'original_text': text,
                    'prediction': 'SPAM' if pred_code == 1 else 'NON-SPAM',
                    'prediction_code': pred_code,
                    'confidence': f"{confidence*100:.2f}%"
                })
            except:
                results.append({
                    'original_text': text,
                    'prediction': 'ERROR',
                    'prediction_code': -1,
                    'confidence': 'N/A'
                })
        
        results_df = pd.DataFrame(results)
        
        return {
            "message": "Prediction completed",
            "total": len(results_df),
            "spam_count": int((results_df['prediction_code'] == 1).sum()),
            "results": results_df.to_dict(orient='records')
        }
    
    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="CSV file kosong")
    except Exception as e:
        logger.error(f"CSV prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"CSV processing failed: {str(e)}")

# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=7860,  # HF Spaces default port
        reload=False,  # Disable reload in production
        log_level="info"
    )