# 🚀 SMS Spam Detector API

REST API untuk deteksi spam SMS Bahasa Indonesia menggunakan **IndoBERT**.

## 📋 Endpoints

### 1. Root
```
GET /
```
Welcome message dan list endpoints

### 2. Health Check
```
GET /health
```
Check status API dan model

### 3. Model Info
```
GET /model/info
```
Informasi detail tentang model

### 4. Single Prediction
```
POST /predict
```

**Request Body:**
```json
{
  "text": "SELAMAT! Anda menang 10 juta rupiah"
}
```

**Response:**
```json
{
  "text": "SELAMAT! Anda menang 10 juta rupiah",
  "prediction": "SPAM",
  "prediction_code": 1,
  "confidence": 0.9876,
  "confidence_percentage": "98.76%",
  "cleaned_text": "selamat anda menang juta rupiah",
  "timestamp": "2025-01-08T10:30:00",
  "warning": "⚠️ PERINGATAN: SMS ini terdeteksi sebagai SPAM..."
}
```

### 5. Batch Prediction
```
POST /predict/batch
```

**Request Body:**
```json
{
  "texts": [
    "Rapat hari ini jam 2 siang",
    "PROMO GAJIAN! Diskon 50%",
    "Terima kasih sudah berbelanja"
  ]
}
```

**Response:**
```json
{
  "total": 3,
  "spam_count": 1,
  "non_spam_count": 2,
  "spam_rate": 33.33,
  "results": [
    {
      "text": "Rapat hari ini jam 2 siang",
      "prediction": "NON-SPAM",
      "confidence": 0.95,
      ...
    },
    ...
  ],
  "timestamp": "2025-01-08T10:30:00"
}
```

### 6. CSV Upload
```
POST /predict/csv
```

Upload file CSV dengan kolom `text` untuk batch prediction.

---

## 🚀 Quick Start

### Option 1: Direct Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run server
uvicorn api:app --reload --host 0.0.0.0 --port 8000

# 3. Access API docs
# Swagger UI: http://localhost:8000/docs
# ReDoc: http://localhost:8000/redoc
```

### Option 2: Docker

```bash
# 1. Build image
docker build -t spam-detector-api .

# 2. Run container
docker run -p 8000:8000 spam-detector-api

# 3. Access API
curl http://localhost:8000/health
```

### Option 3: Docker Compose

```bash
# Run all services (API + Streamlit)
docker-compose up -d

# Check logs
docker-compose logs -f

# Stop
docker-compose down
```

---

## 📝 Usage Examples

### Python (requests)

```python
import requests

# Single prediction
response = requests.post(
    "http://localhost:8000/predict",
    json={"text": "PROMO GAJIAN! Diskon 50%"}
)
print(response.json())

# Batch prediction
response = requests.post(
    "http://localhost:8000/predict/batch",
    json={
        "texts": [
            "Rapat hari ini jam 2",
            "MENANG UNDIAN 10 JUTA!"
        ]
    }
)
print(response.json())
```

### cURL

```bash
# Health check
curl http://localhost:8000/health

# Single prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text":"SELAMAT! Anda menang hadiah"}'

# Batch prediction
curl -X POST "http://localhost:8000/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{"texts":["Rapat hari ini","PROMO 50%"]}'
```

### JavaScript (fetch)

```javascript
// Single prediction
fetch('http://localhost:8000/predict', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    text: 'PROMO GAJIAN! Diskon 50%'
  })
})
.then(res => res.json())
.then(data => console.log(data));

// Batch prediction
fetch('http://localhost:8000/predict/batch', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    texts: [
      'Rapat hari ini jam 2 siang',
      'MENANG UNDIAN 10 JUTA!'
    ]
  })
})
.then(res => res.json())
.then(data => console.log(data));
```

---

## 🧪 Testing

```bash
# Run test script
python test_api.py

# Test dengan pytest (jika ada)
pytest tests/ -v

# Load testing dengan locust
locust -f locustfile.py --host=http://localhost:8000
```

---

## 📊 API Documentation

Setelah server berjalan, akses:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

Interactive documentation dengan try-it-out feature!

---

## 🔒 Security

### Rate Limiting (Recommended)

Install slowapi:
```bash
pip install slowapi
```

Add to `api.py`:
```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/predict")
@limiter.limit("10/minute")
async def predict(request: Request, sms: SMSInput):
    ...
```

### HTTPS

Use reverse proxy (nginx) for HTTPS:
```nginx
server {
    listen 443 ssl;
    server_name api.example.com;
    
    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;
    
    location / {
        proxy_pass http://localhost:8000;
    }
}
```

---

## 📦 Deployment

### Heroku

```bash
# 1. Create Procfile
echo "web: uvicorn api:app --host 0.0.0.0 --port $PORT" > Procfile

# 2. Deploy
heroku create spam-detector-api
git push heroku main
```

### Railway

1. Connect GitHub repo
2. Add environment variables
3. Deploy automatically

### AWS Lambda (with Mangum)

```bash
pip install mangum

# In api.py
from mangum import Mangum
handler = Mangum(app)
```

### Google Cloud Run

```bash
gcloud run deploy spam-detector \
  --source . \
  --platform managed \
  --region asia-southeast1 \
  --allow-unauthenticated
```

---

## 🔧 Configuration

### Environment Variables

Create `.env` file:
```
MODEL_PATH=bert_output_final
MAX_BATCH_SIZE=1000
LOG_LEVEL=INFO
CORS_ORIGINS=*
```

Load in `api.py`:
```python
from dotenv import load_dotenv
import os

load_dotenv()
MODEL_PATH = os.getenv("MODEL_PATH", "bert_output_final")
```

---

## 📈 Monitoring

### Prometheus Metrics

```bash
pip install prometheus-fastapi-instrumentator

# In api.py
from prometheus_fastapi_instrumentator import Instrumentator

Instrumentator().instrument(app).expose(app)
```

Access metrics: http://localhost:8000/metrics

### Logging

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api.log'),
        logging.StreamHandler()
    ]
)
```

---

## 🐛 Troubleshooting

### Model Not Loading
```
Error: Model not loaded
Solution: Check MODEL_PATH dan pastikan folder bert_output_final/ ada
```

### Port Already in Use
```bash
# Kill process on port 8000
lsof -ti:8000 | xargs kill -9

# Or use different port
uvicorn api:app --port 8001
```

### CORS Error
```python
# In api.py, update CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Specific origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## 📚 Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [IndoBERT Paper](https://arxiv.org)
- [Transformers Docs](https://huggingface.co/docs/transformers)

---

## 📞 Support

- GitHub Issues: [Link]
- Email: your.email@example.com
- Discord: [Link]

---

## 📄 License

MIT License

---

**Made with ❤️ using FastAPI & IndoBERT**