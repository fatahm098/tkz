"""
Script untuk testing SMS Spam Detector API

Install:
pip install requests

Usage:
python test_api.py
"""

import requests
import json
from pprint import pprint

# Base URL (sesuaikan dengan server Anda)
BASE_URL = "http://localhost:8000"

def print_header(text):
    print("\n" + "="*80)
    print(f"  {text}")
    print("="*80)

def test_root():
    """Test root endpoint"""
    print_header("TEST 1: Root Endpoint")
    response = requests.get(f"{BASE_URL}/")
    print(f"Status Code: {response.status_code}")
    pprint(response.json())

def test_health():
    """Test health check"""
    print_header("TEST 2: Health Check")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    pprint(response.json())

def test_model_info():
    """Test model info"""
    print_header("TEST 3: Model Info")
    response = requests.get(f"{BASE_URL}/model/info")
    print(f"Status Code: {response.status_code}")
    pprint(response.json())

def test_single_prediction():
    """Test single SMS prediction"""
    print_header("TEST 4: Single Prediction - Normal SMS")
    
    data = {
        "text": "Rapat hari ini jam 2 siang di ruang meeting lantai 3"
    }
    
    response = requests.post(
        f"{BASE_URL}/predict",
        json=data,
        headers={"Content-Type": "application/json"}
    )
    
    print(f"Status Code: {response.status_code}")
    print(f"\nInput: {data['text']}")
    print("\nResponse:")
    pprint(response.json())

def test_spam_prediction():
    """Test SPAM detection"""
    print_header("TEST 5: Single Prediction - SPAM SMS")
    
    data = {
        "text": "SELAMAT! Anda menang 10 juta rupiah. Klik link berikut untuk klaim hadiah segera!"
    }
    
    response = requests.post(
        f"{BASE_URL}/predict",
        json=data,
        headers={"Content-Type": "application/json"}
    )
    
    print(f"Status Code: {response.status_code}")
    print(f"\nInput: {data['text']}")
    print("\nResponse:")
    result = response.json()
    pprint(result)
    
    # Highlight warning
    if result.get('warning'):
        print(f"\n⚠️  WARNING: {result['warning']}")

def test_batch_prediction():
    """Test batch prediction"""
    print_header("TEST 6: Batch Prediction")
    
    data = {
        "texts": [
            "Rapat hari ini jam 2 siang",
            "PROMO GAJIAN! Diskon 50% semua produk hari ini saja!",
            "Terima kasih sudah berbelanja di toko kami",
            "URGENT! Rekening anda diblokir. Hubungi 08123456789",
            "Meeting zoom link: zoom.us/j/123456789",
            "Cashback 100rb untuk transaksi minimal 500rb"
        ]
    }
    
    response = requests.post(
        f"{BASE_URL}/predict/batch",
        json=data,
        headers={"Content-Type": "application/json"}
    )
    
    print(f"Status Code: {response.status_code}")
    result = response.json()
    
    print(f"\nTotal SMS: {result['total']}")
    print(f"SPAM: {result['spam_count']} ({result['spam_rate']:.1f}%)")
    print(f"NON-SPAM: {result['non_spam_count']}")
    
    print("\nDetail Results:")
    for i, res in enumerate(result['results'], 1):
        print(f"\n{i}. {res['text'][:50]}...")
        print(f"   Prediction: {res['prediction']} (Confidence: {res['confidence_percentage']})")

def test_empty_text():
    """Test dengan text kosong"""
    print_header("TEST 7: Error Handling - Empty Text")
    
    data = {"text": ""}
    
    response = requests.post(
        f"{BASE_URL}/predict",
        json=data,
        headers={"Content-Type": "application/json"}
    )
    
    print(f"Status Code: {response.status_code}")
    print("Response:")
    pprint(response.json())

def test_long_text():
    """Test dengan text yang panjang"""
    print_header("TEST 8: Long Text Handling")
    
    long_text = "Promo diskon " * 100  # Text yang sangat panjang
    
    data = {"text": long_text}
    
    response = requests.post(
        f"{BASE_URL}/predict",
        json=data,
        headers={"Content-Type": "application/json"}
    )
    
    print(f"Status Code: {response.status_code}")
    print(f"Text Length: {len(long_text)} characters")
    result = response.json()
    print(f"Prediction: {result['prediction']} (Confidence: {result['confidence_percentage']})")

def test_mixed_language():
    """Test dengan mixed language"""
    print_header("TEST 9: Mixed Language")
    
    data = {
        "text": "Hello! Selamat anda dapat FREE GIFT worth 1000000 rupiah! Click here now!"
    }
    
    response = requests.post(
        f"{BASE_URL}/predict",
        json=data,
        headers={"Content-Type": "application/json"}
    )
    
    print(f"Status Code: {response.status_code}")
    print(f"\nInput: {data['text']}")
    result = response.json()
    print(f"Prediction: {result['prediction']} (Confidence: {result['confidence_percentage']})")
    print(f"Cleaned: {result['cleaned_text']}")

def test_special_characters():
    """Test dengan special characters"""
    print_header("TEST 10: Special Characters")
    
    data = {
        "text": "🚨🎉 PROMO!!! 💰💰💰 Diskon 99% $$$ Klik >>> www.promo.com/free <<<"
    }
    
    response = requests.post(
        f"{BASE_URL}/predict",
        json=data,
        headers={"Content-Type": "application/json"}
    )
    
    print(f"Status Code: {response.status_code}")
    result = response.json()
    print(f"Original: {result['text']}")
    print(f"Cleaned: {result['cleaned_text']}")
    print(f"Prediction: {result['prediction']} (Confidence: {result['confidence_percentage']})")

def performance_test():
    """Test performa dengan multiple requests"""
    print_header("TEST 11: Performance Test (10 requests)")
    
    import time
    
    texts = [
        "Rapat penting besok jam 9 pagi",
        "PROMO BESAR! Diskon hingga 70%",
        "Terima kasih atas pesanan Anda",
        "URGENT! Transfer sekarang atau akun diblokir",
        "Meeting link: meet.google.com/xyz",
    ] * 2  # 10 texts
    
    start_time = time.time()
    
    for i, text in enumerate(texts, 1):
        response = requests.post(
            f"{BASE_URL}/predict",
            json={"text": text},
            headers={"Content-Type": "application/json"}
        )
        print(f"Request {i}: {response.status_code}")
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    print(f"\nTotal Time: {elapsed:.2f} seconds")
    print(f"Average Time per Request: {elapsed/len(texts):.3f} seconds")
    print(f"Requests per Second: {len(texts)/elapsed:.2f} req/s")

# ============================================================================
# RUN ALL TESTS
# ============================================================================

def run_all_tests():
    """Run semua test"""
    try:
        print("\n🚀 Starting API Tests...")
        print(f"Base URL: {BASE_URL}")
        
        test_root()
        test_health()
        test_model_info()
        test_single_prediction()
        test_spam_prediction()
        test_batch_prediction()
        test_empty_text()
        test_long_text()
        test_mixed_language()
        test_special_characters()
        performance_test()
        
        print_header("✅ ALL TESTS COMPLETED")
        
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Cannot connect to API server")
        print(f"Make sure server is running at {BASE_URL}")
        print("Run: uvicorn api:app --reload")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")

if __name__ == "__main__":
    run_all_tests()