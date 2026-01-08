---
title: SMS Spam Detector API
emoji: 🚨
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

# SMS Spam Detector Indonesia

API for SMS spam detection using IndoBERT.

**Endpoints:**
- GET /health
- POST /predict
- POST /predict/batch
```

---

### **STEP 4: Buat Space di Hugging Face**

1. **Login** ke https://huggingface.co
2. Click **"New Space"** (https://huggingface.co/new-space)
3. **Isi form:**
```
   Space name: sms-spam-detector-api
   License: MIT
   Select SDK: 
     - Untuk Streamlit: Streamlit
     - Untuk FastAPI: Docker
   Space hardware: CPU (basic) - FREE