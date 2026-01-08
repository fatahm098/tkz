"""
🚀 APLIKASI WEB DETEKSI SPAM SMS BAHASA INDONESIA
Menggunakan IndoBERT (BERT for Indonesian Language)

FITUR:
✅ Deteksi Real-time
✅ Batch Processing (Upload CSV)
✅ Visualisasi Confidence Score
✅ History & Export Results
✅ Statistik Real-time

Cara Install:
pip install streamlit torch transformers pandas plotly

Cara Run:
streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import re
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# ============================================================================
# KONFIGURASI PAGE
# ============================================================================

st.set_page_config(
    page_title="SMS Spam Detector - IndoBERT",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-top: 0;
        margin-bottom: 2rem;
    }
    .spam-badge {
        background-color: #ff4444;
        color: white;
        padding: 0.5rem 1.5rem;
        border-radius: 25px;
        font-weight: bold;
        font-size: 1.2rem;
        display: inline-block;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .safe-badge {
        background-color: #00C851;
        color: white;
        padding: 0.5rem 1.5rem;
        border-radius: 25px;
        font-weight: bold;
        font-size: 1.2rem;
        display: inline-block;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem;
        font-weight: bold;
        font-size: 1.1rem;
        border-radius: 10px;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    .info-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# FUNGSI PREPROCESSING
# ============================================================================

def clean_text(text):
    """Cleaning text sesuai training"""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'\d{10,}', ' nomorpanjang ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ============================================================================
# LOAD MODEL (CACHE)
# ============================================================================

@st.cache_resource
def load_indobert_model():
    """Load IndoBERT model"""
    try:
        model_path = "bert_output_final"  # Sesuaikan dengan path model Anda
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model.eval()
        return tokenizer, model, None
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.info("""
        **Solusi:**
        1. Pastikan folder `bert_output_final/` ada di direktori yang sama dengan app.py
        2. Atau ubah `model_path` di kode sesuai lokasi model Anda
        3. Download model dari Google Colab jika belum
        """)
        return None, None, str(e)

# ============================================================================
# FUNGSI PREDIKSI
# ============================================================================

def predict_spam(text, tokenizer, model):
    """Prediksi menggunakan IndoBERT"""
    if tokenizer is None or model is None:
        return None, None
    
    clean = clean_text(text)
    inputs = tokenizer(clean, return_tensors='pt', padding=True, 
                      truncation=True, max_length=128)
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred].item()
    
    return pred, confidence

# ============================================================================
# VISUALISASI
# ============================================================================

def create_confidence_gauge(confidence, label):
    """Create gauge chart untuk confidence"""
    color = "#00C851" if label == 0 else "#ff4444"
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = confidence * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Confidence Score", 'font': {'size': 24, 'color': '#333'}},
        number = {'suffix': "%", 'font': {'size': 48, 'color': color}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 2, 'tickcolor': color},
            'bar': {'color': color, 'thickness': 0.75},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#ccc",
            'steps': [
                {'range': [0, 50], 'color': '#f8f9fa'},
                {'range': [50, 75], 'color': '#e9ecef'},
                {'range': [75, 90], 'color': '#dee2e6'},
                {'range': [90, 100], 'color': '#ced4da'}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(
        height=350,
        margin=dict(l=20, r=20, t=80, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'family': 'Arial'}
    )
    return fig

# ============================================================================
# INITIALIZE SESSION STATE
# ============================================================================

if 'history' not in st.session_state:
    st.session_state.history = []

if 'total_spam' not in st.session_state:
    st.session_state.total_spam = 0

if 'total_safe' not in st.session_state:
    st.session_state.total_safe = 0

# ============================================================================
# LOAD MODEL
# ============================================================================

with st.spinner("🔄 Loading IndoBERT model..."):
    tokenizer, model, error = load_indobert_model()

if error:
    st.stop()

# ============================================================================
# HEADER
# ============================================================================

st.markdown('<h1 class="main-header">🚨 SMS Spam Detector Indonesia</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Powered by IndoBERT - Transformer Model for Indonesian Language</p>', unsafe_allow_html=True)

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.image("https://raw.githubusercontent.com/huggingface/transformers/main/docs/source/en/imgs/transformers_logo_name.png", width=200)
    
    st.markdown("---")
    
    st.header("📊 Statistik Session")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total", len(st.session_state.history))
    with col2:
        spam_rate = (st.session_state.total_spam / len(st.session_state.history) * 100) if st.session_state.history else 0
        st.metric("Spam Rate", f"{spam_rate:.1f}%")
    
    st.metric("🚨 Spam", st.session_state.total_spam, delta=None, delta_color="inverse")
    st.metric("✅ Aman", st.session_state.total_safe, delta=None, delta_color="normal")
    
    st.markdown("---")
    
    st.header("ℹ️ Informasi Model")
    st.info("""
    **Model**: IndoBERT Base
    
    **Architecture**: BERT Transformer
    
    **Training**: Fine-tuned on Indonesian SMS Spam Dataset
    
    **Classes**:
    - 0: Normal (Non-Spam)
    - 1: Spam (Fraud + Promo)
    
    **Dataset**: 1,143 SMS
    - Normal: 569
    - Fraud: 335
    - Promo: 239
    """)
    
    st.markdown("---")
    
    if st.session_state.history:
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.history = []
            st.session_state.total_spam = 0
            st.session_state.total_safe = 0
            st.rerun()
    
    st.markdown("---")
    st.markdown("### 🔗 Links")
    st.markdown("- [IndoBERT Paper](https://arxiv.org)")
    st.markdown("- [GitHub Repo](#)")
    st.markdown("- [Dataset Info](#)")

# ============================================================================
# MODE SELECTION
# ============================================================================

tab1, tab2 = st.tabs(["📱 Single Detection", "📂 Batch Processing"])

# ============================================================================
# TAB 1: SINGLE DETECTION
# ============================================================================

with tab1:
    st.header("🔍 Single SMS Detection")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        text_input = st.text_area(
            "Masukkan SMS yang ingin dideteksi:",
            height=150,
            placeholder="Contoh: Selamat! Anda menang 10 juta rupiah. Klik link berikut untuk klaim hadiah...",
            help="Masukkan teks SMS dalam Bahasa Indonesia"
        )
        
        col_btn1, col_btn2 = st.columns([1, 1])
        
        with col_btn1:
            detect_btn = st.button("🔍 Deteksi Sekarang", type="primary", use_container_width=True)
        
        with col_btn2:
            if st.button("🗑️ Clear", use_container_width=True):
                st.rerun()
    
    with col2:
        st.markdown("### 💡 Contoh SMS")
        
        examples = {
            "📝 Normal": "Rapat hari ini jam 2 siang di ruang meeting lantai 3. Jangan lupa bawa laptop.",
            "🚨 Fraud": "URGENT! Rekening anda diblokir karena aktivitas mencurigakan. Segera hubungi 08123456789 untuk verifikasi.",
            "🎁 Promo": "PROMO GAJIAN! Dapatkan cashback 50% untuk semua produk hari ini saja! Klik link: bit.ly/promo123"
        }
        
        for label, example in examples.items():
            if st.button(label, use_container_width=True):
                text_input = example
                st.rerun()
    
    # DETECTION
    if detect_btn and text_input:
        st.markdown("---")
        
        with st.spinner("🔄 Menganalisis SMS..."):
            pred, conf = predict_spam(text_input, tokenizer, model)
        
        if pred is not None:
            # Update statistics
            if pred == 1:
                st.session_state.total_spam += 1
            else:
                st.session_state.total_safe += 1
            
            # Display hasil
            col_res1, col_res2 = st.columns([1.5, 1])
            
            with col_res1:
                st.markdown("### 🎯 Hasil Deteksi")
                
                if pred == 0:
                    st.markdown('<div class="safe-badge">✅ NON-SPAM (AMAN)</div>', 
                              unsafe_allow_html=True)
                    st.success("✅ Pesan ini terdeteksi sebagai pesan normal/legitimate")
                    st.info("""
                    **Analisis:**
                    - Tidak ada indikasi penipuan
                    - Tidak mengandung promosi mencurigakan
                    - Aman untuk dibaca dan ditanggapi
                    """)
                else:
                    st.markdown('<div class="spam-badge">🚨 SPAM (BERBAHAYA)</div>', 
                              unsafe_allow_html=True)
                    st.error("⚠️ **PERINGATAN:** Pesan ini terdeteksi sebagai SPAM/Penipuan!")
                    st.warning("""
                    **🛡️ Tindakan yang Disarankan:**
                    - ❌ JANGAN klik link/tautan apapun
                    - ❌ JANGAN berikan informasi pribadi (PIN, password, OTP)
                    - ❌ JANGAN transfer uang
                    - ✅ Laporkan sebagai spam
                    - ✅ Blokir nomor pengirim
                    - ✅ Hapus pesan
                    """)
                
                st.markdown("---")
                st.markdown("### 📊 Detail Analisis")
                
                detail_col1, detail_col2 = st.columns(2)
                with detail_col1:
                    st.metric("Confidence Level", f"{conf*100:.2f}%")
                    st.metric("Text Length", f"{len(text_input)} karakter")
                with detail_col2:
                    st.metric("Word Count", f"{len(text_input.split())} kata")
                    st.metric("Timestamp", datetime.now().strftime('%H:%M:%S'))
                
                # Cleaned text
                with st.expander("🔍 Lihat Teks yang Diproses"):
                    st.code(clean_text(text_input))
            
            with col_res2:
                st.markdown("### 📊 Confidence Score")
                st.plotly_chart(create_confidence_gauge(conf, pred), 
                              use_container_width=True)
                
                # Interpretation
                if conf >= 0.9:
                    st.success("🎯 **Sangat Yakin**")
                elif conf >= 0.7:
                    st.info("✅ **Cukup Yakin**")
                else:
                    st.warning("⚠️ **Kurang Yakin**")
            
            # Save to history
            st.session_state.history.append({
                'timestamp': datetime.now(),
                'text': text_input[:50] + "..." if len(text_input) > 50 else text_input,
                'full_text': text_input,
                'prediction': 'SPAM' if pred == 1 else 'NON-SPAM',
                'prediction_code': pred,
                'confidence': f"{conf*100:.2f}%"
            })

# ============================================================================
# TAB 2: BATCH PROCESSING
# ============================================================================

with tab2:
    st.header("📂 Batch Processing - Deteksi Massal")
    
    st.info("""
    📄 **Format CSV yang didukung:**
    - Harus memiliki kolom yang berisi teks SMS
    - Encoding: UTF-8
    - Separator: koma (,)
    
    **Contoh format:**
    ```
    text
    "Rapat hari ini jam 2 siang"
    "PROMO GAJIAN! Diskon 50%"
    "Terima kasih sudah berbelanja"
    ```
    """)
    
    uploaded_file = st.file_uploader(
        "📤 Upload File CSV",
        type=['csv'],
        help="Upload file CSV dengan kolom berisi teks SMS"
    )
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            
            st.success(f"✅ File berhasil diupload! Total: {len(df)} baris")
            
            st.markdown("### 📄 Preview Data (10 baris pertama)")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Pilih kolom teks
            text_column = st.selectbox(
                "📌 Pilih kolom yang berisi teks SMS:",
                df.columns,
                help="Pilih kolom yang berisi teks SMS yang akan dideteksi"
            )
            
            st.markdown("---")
            
            col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])
            
            with col_btn1:
                process_btn = st.button("🚀 Proses Semua Data", type="primary", use_container_width=True)
            
            if process_btn:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                results = []
                total = len(df)
                
                for idx, row in df.iterrows():
                    text = row[text_column]
                    pred, conf = predict_spam(text, tokenizer, model)
                    
                    results.append({
                        'original_text': text,
                        'prediction': 'SPAM' if pred == 1 else 'NON-SPAM',
                        'prediction_code': pred,
                        'confidence': f"{conf*100:.2f}%",
                        'confidence_value': conf
                    })
                    
                    progress = (idx + 1) / total
                    progress_bar.progress(progress)
                    status_text.text(f"⏳ Processing: {idx + 1}/{total} ({progress*100:.1f}%)")
                
                results_df = pd.DataFrame(results)
                
                progress_bar.empty()
                status_text.empty()
                
                st.success(f"✅ **Selesai!** Total {total} SMS berhasil diproses")
                
                st.markdown("---")
                
                # Statistics
                st.markdown("### 📊 Ringkasan Hasil")
                
                col1, col2, col3, col4 = st.columns(4)
                
                spam_count = (results_df['prediction_code'] == 1).sum()
                safe_count = (results_df['prediction_code'] == 0).sum()
                avg_conf = results_df['confidence_value'].mean() * 100
                
                with col1:
                    st.metric("📧 Total SMS", total)
                with col2:
                    st.metric("🚨 SPAM", spam_count, delta=f"{spam_count/total*100:.1f}%")
                with col3:
                    st.metric("✅ Non-Spam", safe_count, delta=f"{safe_count/total*100:.1f}%")
                with col4:
                    st.metric("📊 Avg Confidence", f"{avg_conf:.1f}%")
                
                st.markdown("---")
                
                # Results table
                st.markdown("### 📋 Hasil Detail")
                
                # Filter
                filter_option = st.selectbox(
                    "Filter berdasarkan:",
                    ["Semua", "SPAM saja", "NON-SPAM saja"]
                )
                
                if filter_option == "SPAM saja":
                    display_df = results_df[results_df['prediction'] == 'SPAM']
                elif filter_option == "NON-SPAM saja":
                    display_df = results_df[results_df['prediction'] == 'NON-SPAM']
                else:
                    display_df = results_df
                
                st.dataframe(
                    display_df[['original_text', 'prediction', 'confidence']],
                    use_container_width=True,
                    height=400
                )
                
                # Download button
                csv = results_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 Download Hasil (CSV)",
                    csv,
                    f"spam_detection_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    "text/csv",
                    key='download-csv',
                    use_container_width=True
                )
                
                st.markdown("---")
                
                # Visualizations
                col_viz1, col_viz2 = st.columns(2)
                
                with col_viz1:
                    st.markdown("#### 📊 Distribusi Hasil")
                    fig1 = px.pie(
                        results_df,
                        names='prediction',
                        title='Proporsi SPAM vs NON-SPAM',
                        color='prediction',
                        color_discrete_map={'SPAM': '#ff4444', 'NON-SPAM': '#00C851'},
                        hole=0.4
                    )
                    fig1.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(fig1, use_container_width=True)
                
                with col_viz2:
                    st.markdown("#### 📈 Distribusi Confidence")
                    fig2 = px.histogram(
                        results_df,
                        x='confidence_value',
                        nbins=20,
                        title='Distribusi Confidence Score',
                        labels={'confidence_value': 'Confidence Score'},
                        color='prediction',
                        color_discrete_map={'SPAM': '#ff4444', 'NON-SPAM': '#00C851'}
                    )
                    fig2.update_xaxis(tickformat='.0%')
                    st.plotly_chart(fig2, use_container_width=True)
        
        except Exception as e:
            st.error(f"❌ Error processing file: {e}")
            st.info("Pastikan file CSV memiliki format yang benar dan encoding UTF-8")

# ============================================================================
# HISTORY
# ============================================================================

if st.session_state.history:
    st.markdown("---")
    with st.expander("📜 History Deteksi", expanded=False):
        history_df = pd.DataFrame(st.session_state.history)
        
        # Sort by timestamp descending
        history_df = history_df.sort_values('timestamp', ascending=False)
        
        st.dataframe(
            history_df[['timestamp', 'text', 'prediction', 'confidence']],
            use_container_width=True,
            height=300
        )

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")

footer_col1, footer_col2, footer_col3 = st.columns(3)

with footer_col1:
    st.markdown("### 📚 Dataset")
    st.write("SMS Spam Indonesia v1.0")
    st.write("CC BY-SA 4.0 License")

with footer_col2:
    st.markdown("### 🤖 Model")
    st.write("**IndoBERT Base**")
    st.write("BERT for Indonesian")

with footer_col3:
    st.markdown("### 🛠️ Tech Stack")
    st.write("Streamlit + Transformers")
    st.write("PyTorch + HuggingFace")

st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray;'>© 2025 SMS Spam Detector Indonesia | Built with ❤️ using Streamlit & IndoBERT</p>",
    unsafe_allow_html=True
)