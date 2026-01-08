// SpamDetector.jsx - React Component untuk integrasi dengan API

import React, { useState } from 'react';
import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000';

function SpamDetector() {
  const [text, setText] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Single SMS Detection
  const handleDetect = async () => {
    if (!text.trim()) {
      setError('Masukkan teks SMS terlebih dahulu');
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await axios.post(`${API_BASE_URL}/predict`, {
        text: text
      });

      setResult(response.data);
    } catch (err) {
      setError(err.response?.data?.detail || 'Terjadi kesalahan');
    } finally {
      setLoading(false);
    }
  };

  // Example SMS
  const loadExample = (type) => {
    const examples = {
      normal: 'Rapat hari ini jam 2 siang di ruang meeting lantai 3. Jangan lupa bawa laptop.',
      fraud: 'URGENT! Rekening anda diblokir karena aktivitas mencurigakan. Segera hubungi 08123456789 untuk verifikasi.',
      promo: 'PROMO GAJIAN! Dapatkan cashback 50% untuk semua produk hari ini saja! Klik: bit.ly/promo123'
    };
    setText(examples[type]);
  };

  return (
    <div className="container mx-auto p-6 max-w-4xl">
      <div className="bg-white shadow-lg rounded-lg p-8">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold bg-gradient-to-r from-purple-600 to-blue-600 bg-clip-text text-transparent">
            🚨 SMS Spam Detector
          </h1>
          <p className="text-gray-600 mt-2">
            Powered by IndoBERT API
          </p>
        </div>

        {/* Input Area */}
        <div className="mb-6">
          <label className="block text-gray-700 font-semibold mb-2">
            Masukkan SMS:
          </label>
          <textarea
            className="w-full p-4 border-2 border-gray-300 rounded-lg focus:border-purple-500 focus:outline-none"
            rows="5"
            placeholder="Contoh: SELAMAT! Anda menang 10 juta rupiah..."
            value={text}
            onChange={(e) => setText(e.target.value)}
          />
        </div>

        {/* Example Buttons */}
        <div className="flex gap-2 mb-6">
          <button
            onClick={() => loadExample('normal')}
            className="px-4 py-2 bg-green-500 text-white rounded-lg hover:bg-green-600 transition"
          >
            📝 Normal
          </button>
          <button
            onClick={() => loadExample('fraud')}
            className="px-4 py-2 bg-red-500 text-white rounded-lg hover:bg-red-600 transition"
          >
            🚨 Fraud
          </button>
          <button
            onClick={() => loadExample('promo')}
            className="px-4 py-2 bg-yellow-500 text-white rounded-lg hover:bg-yellow-600 transition"
          >
            🎁 Promo
          </button>
        </div>

        {/* Detect Button */}
        <button
          onClick={handleDetect}
          disabled={loading || !text.trim()}
          className="w-full py-3 bg-gradient-to-r from-purple-600 to-blue-600 text-white font-semibold rounded-lg hover:shadow-lg transition disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {loading ? '🔄 Menganalisis...' : '🔍 Deteksi Sekarang'}
        </button>

        {/* Error Message */}
        {error && (
          <div className="mt-6 p-4 bg-red-100 border border-red-400 text-red-700 rounded-lg">
            ❌ {error}
          </div>
        )}

        {/* Result */}
        {result && (
          <div className="mt-8 space-y-4">
            {/* Prediction Badge */}
            <div className="text-center">
              {result.prediction === 'SPAM' ? (
                <div className="inline-block px-6 py-3 bg-red-500 text-white text-xl font-bold rounded-full shadow-lg">
                  🚨 SPAM (BERBAHAYA)
                </div>
              ) : (
                <div className="inline-block px-6 py-3 bg-green-500 text-white text-xl font-bold rounded-full shadow-lg">
                  ✅ NON-SPAM (AMAN)
                </div>
              )}
            </div>

            {/* Confidence */}
            <div className="bg-gray-100 p-6 rounded-lg">
              <div className="flex justify-between items-center mb-2">
                <span className="font-semibold">Confidence Score:</span>
                <span className="text-2xl font-bold text-purple-600">
                  {result.confidence_percentage}
                </span>
              </div>
              <div className="w-full bg-gray-300 rounded-full h-4 overflow-hidden">
                <div
                  className={`h-full ${
                    result.prediction === 'SPAM' ? 'bg-red-500' : 'bg-green-500'
                  } transition-all duration-500`}
                  style={{ width: result.confidence_percentage }}
                />
              </div>
            </div>

            {/* Warning */}
            {result.warning && (
              <div className="bg-yellow-100 border-l-4 border-yellow-500 p-4 rounded">
                <p className="font-semibold text-yellow-800">
                  ⚠️ Peringatan:
                </p>
                <p className="text-yellow-700 mt-2">
                  {result.warning}
                </p>
                <ul className="mt-3 text-sm text-yellow-700 list-disc list-inside">
                  <li>Jangan klik link apapun</li>
                  <li>Jangan berikan informasi pribadi</li>
                  <li>Laporkan sebagai spam</li>
                </ul>
              </div>
            )}

            {/* Details */}
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div className="bg-gray-50 p-3 rounded">
                <span className="text-gray-600">Text Length:</span>
                <span className="float-right font-semibold">
                  {result.text.length} chars
                </span>
              </div>
              <div className="bg-gray-50 p-3 rounded">
                <span className="text-gray-600">Timestamp:</span>
                <span className="float-right font-semibold">
                  {new Date(result.timestamp).toLocaleTimeString()}
                </span>
              </div>
            </div>

            {/* Cleaned Text */}
            <details className="bg-gray-50 p-4 rounded cursor-pointer">
              <summary className="font-semibold text-gray-700">
                🔍 Lihat Teks yang Diproses
              </summary>
              <pre className="mt-2 text-sm text-gray-600 whitespace-pre-wrap">
                {result.cleaned_text}
              </pre>
            </details>
          </div>
        )}
      </div>

      {/* API Info */}
      <div className="mt-6 text-center text-sm text-gray-500">
        API Endpoint: <code className="bg-gray-100 px-2 py-1 rounded">{API_BASE_URL}</code>
        <br />
        <a href={`${API_BASE_URL}/docs`} target="_blank" rel="noopener noreferrer" className="text-purple-600 hover:underline">
          📚 View API Documentation
        </a>
      </div>
    </div>
  );
}

export default SpamDetector;


// ============================================================================
// USAGE EXAMPLE
// ============================================================================

/*
// 1. Install dependencies
npm install axios

// 2. Import component
import SpamDetector from './SpamDetector';

// 3. Use in App.js
function App() {
  return (
    <div className="min-h-screen bg-gray-100 py-8">
      <SpamDetector />
    </div>
  );
}

// 4. Make sure Tailwind CSS is configured
// tailwind.config.js
module.exports = {
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  theme: {
    extend: {},
  },
  plugins: [],
}
*/


// ============================================================================
// BATCH DETECTION COMPONENT (BONUS)
// ============================================================================

function BatchSpamDetector() {
  const [texts, setTexts] = useState(['', '', '']);
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleBatchDetect = async () => {
    const validTexts = texts.filter(t => t.trim());
    
    if (validTexts.length === 0) {
      alert('Masukkan minimal 1 SMS');
      return;
    }

    setLoading(true);

    try {
      const response = await axios.post(`${API_BASE_URL}/predict/batch`, {
        texts: validTexts
      });

      setResults(response.data);
    } catch (err) {
      alert(err.response?.data?.detail || 'Terjadi kesalahan');
    } finally {
      setLoading(false);
    }
  };

  const addTextField = () => {
    setTexts([...texts, '']);
  };

  return (
    <div className="container mx-auto p-6 max-w-4xl">
      <div className="bg-white shadow-lg rounded-lg p-8">
        <h2 className="text-3xl font-bold mb-6">📂 Batch Detection</h2>

        {texts.map((text, index) => (
          <div key={index} className="mb-4">
            <label className="block text-gray-700 mb-2">SMS #{index + 1}</label>
            <textarea
              className="w-full p-3 border rounded"
              rows="2"
              value={text}
              onChange={(e) => {
                const newTexts = [...texts];
                newTexts[index] = e.target.value;
                setTexts(newTexts);
              }}
            />
          </div>
        ))}

        <div className="flex gap-4 mb-6">
          <button
            onClick={addTextField}
            className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
          >
            ➕ Tambah SMS
          </button>
          
          <button
            onClick={handleBatchDetect}
            disabled={loading}
            className="px-6 py-2 bg-purple-600 text-white rounded hover:bg-purple-700 disabled:opacity-50"
          >
            {loading ? '🔄 Processing...' : '🚀 Deteksi Semua'}
          </button>
        </div>

        {results && (
          <div className="mt-8">
            <div className="grid grid-cols-4 gap-4 mb-6">
              <div className="bg-blue-100 p-4 rounded text-center">
                <div className="text-3xl font-bold">{results.total}</div>
                <div className="text-sm text-gray-600">Total SMS</div>
              </div>
              <div className="bg-red-100 p-4 rounded text-center">
                <div className="text-3xl font-bold text-red-600">{results.spam_count}</div>
                <div className="text-sm text-gray-600">SPAM</div>
              </div>
              <div className="bg-green-100 p-4 rounded text-center">
                <div className="text-3xl font-bold text-green-600">{results.non_spam_count}</div>
                <div className="text-sm text-gray-600">Non-Spam</div>
              </div>
              <div className="bg-yellow-100 p-4 rounded text-center">
                <div className="text-3xl font-bold text-yellow-600">{results.spam_rate.toFixed(1)}%</div>
                <div className="text-sm text-gray-600">Spam Rate</div>
              </div>
            </div>

            <div className="space-y-4">
              {results.results.map((result, index) => (
                <div
                  key={index}
                  className={`p-4 rounded-lg border-l-4 ${
                    result.prediction === 'SPAM'
                      ? 'bg-red-50 border-red-500'
                      : 'bg-green-50 border-green-500'
                  }`}
                >
                  <div className="flex justify-between items-start">
                    <div className="flex-1">
                      <p className="font-semibold text-gray-800">
                        {result.text.substring(0, 80)}...
                      </p>
                    </div>
                    <div className="text-right ml-4">
                      <span
                        className={`px-3 py-1 rounded-full text-sm font-bold ${
                          result.prediction === 'SPAM'
                            ? 'bg-red-500 text-white'
                            : 'bg-green-500 text-white'
                        }`}
                      >
                        {result.prediction}
                      </span>
                      <p className="text-sm text-gray-600 mt-1">
                        {result.confidence_percentage}
                      </p>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export { BatchSpamDetector };