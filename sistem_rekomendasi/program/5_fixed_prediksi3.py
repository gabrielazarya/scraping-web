import os
import time
import re
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import requests
from bs4 import BeautifulSoup
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import gradio as gr

# ----------------------------
# Konfigurasi Path
# ----------------------------
BASE_DIR = r"D:\TA\TokPed\sistem_rekomendasi"
RESULT_DIR = os.path.join(BASE_DIR, "hasil_rekomendasi")
MODEL_DIR = os.path.join(BASE_DIR, "model_terbaik")

MODEL_PATH = os.path.join(MODEL_DIR, "model_K10_F10_E20_B16_D0.3.keras")
TOKENIZER_PATH = os.path.join(MODEL_DIR, "tokenizer.pkl")
MAXLEN_PATH = os.path.join(MODEL_DIR, "maxlen.pkl")

os.makedirs(RESULT_DIR, exist_ok=True)

# ----------------------------
# 1. Scraping Komentar Tokopedia (TANPA HEADLESS)
# ----------------------------
class TokopediaScraper:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
    
    def scrape_reviews(self, url):
        if not url:
            return None, "URL kosong."

        try:
            # Format URL untuk halaman review
            if '/review' not in url:
                url = url.rstrip('/') + '/review'
            
            print(f"Scraping dari: {url}")
            response = requests.get(url, headers=self.headers, timeout=15)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Multiple selector untuk mencari review
            review_selectors = [
                'span[data-testid="lblItemComment"]',
                'span[data-testid="lblItemUlasan"]',
                'div.review-content',
                'p.review-text',
                '.item-comment',
                '.review-item'
            ]
            
            reviews = []
            for selector in review_selectors:
                review_elements = soup.select(selector)
                if review_elements:
                    print(f"Ditemukan {len(review_elements)} review dengan selector: {selector}")
                    for review in review_elements:
                        review_text = review.get_text().strip()
                        if review_text and len(review_text) > 5:
                            reviews.append(review_text)
                    break
            
            if not reviews:
                # Fallback: cari semua teks yang panjang
                all_texts = soup.find_all(text=True)
                reviews = [text.strip() for text in all_texts if len(text.strip()) > 20 and len(text.strip()) < 500]
                print(f"Fallback: ditemukan {len(reviews)} teks panjang")
            
            if not reviews:
                return None, "Tidak ada komentar ditemukan."
            
            print(f"Berhasil mengambil {len(reviews)} review")
            return pd.DataFrame(reviews, columns=["komentar"]), f"Berhasil mengambil {len(reviews)} komentar."
            
        except Exception as e:
            print(f"Error scraping: {e}")
            return None, f"Error scraping: {str(e)}"

# ----------------------------
# 2. Text Preprocessing
# ----------------------------
class TextPreprocessor:
    def __init__(self):
        pass
        
    def clean_text(self, text):
        if pd.isna(text):
            return ""
        text = str(text).lower()
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#\w+', '', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def normalize_text(self, text):
        slang_dict = {
            'bgus': 'bagus', 'baguz': 'bagus', 'ori': 'original',
            'asl': 'asli', 'palsu': 'palsu', 'kw': 'palsu',
            'rekom': 'rekomendasi', 'recomended': 'rekomendasi',
            'jelek': 'jelek', 'jeleq': 'jelek', 'baguss': 'bagus',
            'sekalee': 'sekali', 'aslii': 'asli', 'kwalitas': 'kualitas'
        }
        words = text.split()
        normalized_words = [slang_dict.get(word, word) for word in words]
        return ' '.join(normalized_words)
    
    def preprocess(self, text):
        text = self.clean_text(text)
        text = self.normalize_text(text)
        return text

# ----------------------------
# 3. Hybrid Classification System
# ----------------------------
class HybridClassifier:
    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.model = None
        self.tokenizer = None
        self.MAX_LEN = 100
        
        # Kamus untuk rule-based classification
        self.palsu_keywords = [
            'palsu', 'kw', 'jelek', 'buruk', 'rusak', 'cacat', 'retak', 'sobek',
            'pecah', 'patah', 'mengecewakan', 'kecewa', 'tipu', 'bodong', 'scam',
            'tidak sesuai', 'beda foto', 'kualitas murah', 'murahan', 'aspal'
        ]
        
        self.asli_keywords = [
            'asli', 'ori', 'original', 'bagus', 'puas', 'recommended', 'rekomendasi',
            'mantap', 'oke', 'sesuai', 'memuaskan', 'terima kasih', 'kualitas bagus'
        ]
        
        self.load_resources()
    
    def load_resources(self):
        """Load model dan tokenizer"""
        try:
            if os.path.exists(MODEL_PATH):
                self.model = load_model(MODEL_PATH)
                print("Model LSTM loaded")
            else:
                print("Model LSTM tidak ditemukan")
                
            if os.path.exists(TOKENIZER_PATH):
                with open(TOKENIZER_PATH, "rb") as f:
                    self.tokenizer = pickle.load(f)
                print("Tokenizer loaded")
                
            if os.path.exists(MAXLEN_PATH):
                with open(MAXLEN_PATH, "rb") as f:
                    self.MAX_LEN = pickle.load(f)
                print(f"Max length: {self.MAX_LEN}")
                
        except Exception as e:
            print(f"Error loading resources: {e}")
    
    def rule_based_prediction(self, text):
        """Prediksi berbasis kamus keywords"""
        text_lower = text.lower()
        
        palsu_score = sum(1 for keyword in self.palsu_keywords if keyword in text_lower)
        asli_score = sum(1 for keyword in self.asli_keywords if keyword in text_lower)
        
        if palsu_score == 0 and asli_score == 0:
            return 0.5  # Netral
        
        total_score = palsu_score + asli_score
        prob_asli = asli_score / total_score
        
        # Adjust probability based on dominance
        if palsu_score > asli_score * 2:
            prob_asli = max(0.1, prob_asli - 0.3)
        elif asli_score > palsu_score * 2:
            prob_asli = min(0.9, prob_asli + 0.3)
        
        return prob_asli
    
    def ml_prediction(self, texts):
        """Prediksi menggunakan model LSTM"""
        if self.model is None or self.tokenizer is None:
            return None
            
        try:
            processed_texts = [self.preprocessor.preprocess(text) for text in texts]
            sequences = self.tokenizer.texts_to_sequences(processed_texts)
            X = pad_sequences(sequences, maxlen=self.MAX_LEN, padding="post", truncating="post")
            preds = self.model.predict(X, verbose=0)
            
            if preds.ndim == 2:
                if preds.shape[1] == 1:
                    return preds.flatten()
                elif preds.shape[1] == 2:
                    return preds[:, 1]
            return preds.flatten()
            
        except Exception as e:
            print(f"ML prediction error: {e}")
            return None
    
    def hybrid_classify(self, df):
        """Klasifikasi hybrid dengan bobot 3x untuk komentar palsu"""
        comments = df["komentar"].tolist()
        ml_probs = self.ml_prediction(comments)
        
        results = []
        for i, comment in enumerate(comments):
            # Rule-based prediction
            rule_prob = self.rule_based_prediction(comment)
            
            # Jika ML prediction available, gabungkan
            if ml_probs is not None:
                ml_prob = ml_probs[i]
                # 70% rule-based, 30% ML
                final_prob = 0.7 * rule_prob + 0.3 * ml_prob
            else:
                final_prob = rule_prob
            
            final_label = "Asli" if final_prob >= 0.5 else "Palsu"
            
            results.append({
                'komentar': comment,
                'Prob_Asli': final_prob,
                'Label_Pred': final_label,
                'Rule_Prob': rule_prob
            })
        
        return pd.DataFrame(results)

# ----------------------------
# 4. Initialize Classifier
# ----------------------------
hybrid_classifier = HybridClassifier()
scraper = TokopediaScraper()

# ----------------------------
# 5. Main Processing Function
# ----------------------------
def process_comments(df):
    """Process comments dengan hybrid approach"""
    print(f"Memproses {len(df)} komentar...")
    result_df = hybrid_classifier.hybrid_classify(df)
    
    # Hitung confidence
    confidence_scores = []
    for _, row in result_df.iterrows():
        prob = row['Prob_Asli']
        confidence = 2 * abs(prob - 0.5)  # 0-1 scale, 1 = most confident
        confidence_scores.append(confidence)
    
    result_df['Confidence'] = confidence_scores
    
    return result_df

# ----------------------------
# 6. Evaluation dengan Bobot 3x untuk Palsu
# ----------------------------
def evaluate_result(df):
    """Evaluasi hasil prediksi dengan bobot 3x untuk komentar palsu"""
    total_asli = (df["Label_Pred"] == "Asli").sum()
    total_palsu = (df["Label_Pred"] == "Palsu").sum()
    total = len(df)
    
    if total == 0:
        return "<div style='color:#d32f2f;text-align:center;'>Tidak ada data untuk dianalisis</div>", "", 0, 0
    
    # Hitung skor dengan bobot 3x untuk komentar palsu
    weighted_score = (total_asli * 1 + total_palsu * 3) / (total * 3) * 100
    persentase_asli = total_asli / total * 100
    persentase_palsu = total_palsu / total * 100
    avg_confidence = df["Confidence"].mean()
    
    # Decision logic dengan threshold 30% (dengan bobot)
    if weighted_score <= 30:
        hasil = "BARANG PALSU"
        warna = "#e63946"
        alasan = f"Skor keaslian {weighted_score:.1f}% (dengan bobot 3x untuk komentar palsu)"
    elif weighted_score <= 50:
        hasil = "BARANG DICURIGAI PALSU"
        warna = "#ff9800"
        alasan = f"Skor keaslian {weighted_score:.1f}% - perlu pemeriksaan lebih lanjut"
    else:
        hasil = "BARANG ASLI"
        warna = "#2a9d8f"
        alasan = f"Skor keaslian {weighted_score:.1f}% - barang terindikasi asli"
    
    hasil_html = f"""
    <div style='text-align:center; margin-top:10px; padding:20px; border-radius:10px; background-color:#f8f9fa; border:3px solid {warna};'>
        <h2 style='color:{warna}; margin-bottom:20px;'>{hasil}</h2>
        <div style='display:flex; justify-content:center; gap:40px; margin-bottom:15px;'>
            <div style='text-align:center;'>
                <div style='font-size:28px; color:#2a9d8f; font-weight:bold;'>{persentase_asli:.1f}%</div>
                <div style='color:#333333; font-size:14px;'>Asli ({total_asli})</div>
            </div>
            <div style='text-align:center;'>
                <div style='font-size:28px; color:#e63946; font-weight:bold;'>{persentase_palsu:.1f}%</div>
                <div style='color:#333333; font-size:14px;'>Palsu ({total_palsu})</div>
            </div>
        </div>
        <p style='color:#333333; margin:8px 0; font-size:14px;'><b>Total Komentar:</b> {total}</p>
        <p style='color:#333333; margin:8px 0; font-size:14px;'><b>Confidence Rata-rata:</b> {avg_confidence:.3f}</p>
        <p style='color:#333333; margin:8px 0; font-size:14px;'><b>Alasan:</b> {alasan}</p>
        <p style='color:#666666; margin:8px 0; font-size:12px;'>*Komentar palsu diberi bobot 3x lebih berat</p>
    </div>
    """
    
    summary_html = f"<div style='text-align:center; color:#333333; margin-top:10px; font-size:14px;'><b>Summary:</b> {total_asli} Asli | {total_palsu} Palsu | Confidence: {avg_confidence:.3f} | Skor: {weighted_score:.1f}%</div>"
    
    return hasil_html, summary_html, total_asli, total_palsu

def generate_bar_chart(asli, palsu):
    """Generate bar chart untuk visualisasi"""
    labels = ["Asli", "Palsu"]
    values = [asli, palsu]
    colors = ["#2a9d8f", "#e63946"]

    plt.figure(figsize=(6, 4))
    bars = plt.bar(labels, values, color=colors, alpha=0.8)
    plt.title("Distribusi Prediksi Komentar", fontsize=14, fontweight='bold', color='#333333')
    plt.ylabel("Jumlah Komentar", fontsize=12, color='#333333')
    plt.xticks(color='#333333')
    plt.yticks(color='#333333')
    
    # Tambah nilai di atas bar
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{value}', ha='center', va='bottom', fontweight='bold', color='#333333')
    
    plt.tight_layout()

    # Convert to HTML
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=100, bbox_inches="tight", facecolor='white')
    plt.close()
    buf.seek(0)
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"<img src='data:image/png;base64,{encoded}' width='500' style='display:block; margin:auto;'/>"

# ----------------------------
# 7. Main UI Function
# ----------------------------
def sistem_rekomendasi_ui(url):
    """Main function untuk UI"""
    status_msg = "Memulai proses..."
    
    try:
        # Step 1: Scraping
        status_msg = "Sedang scraping komentar dari Tokopedia..."
        df_scrape, msg = scraper.scrape_reviews(url)
        if df_scrape is None:
            return msg, "<div style='text-align:center;color:#d32f2f;'>Gagal: tidak ada komentar ditemukan</div>", None, "", ""
        
        # Step 2: Processing
        status_msg = "Menganalisis komentar dengan sistem hybrid..."
        df_result = process_comments(df_scrape)
        
        # Save results
        csv_path = os.path.join(RESULT_DIR, "ulasan_prediksi.csv")
        df_result[["komentar", "Prob_Asli", "Label_Pred", "Confidence"]].to_csv(csv_path, index=False, encoding="utf-8-sig")
        
        # Step 3: Evaluation
        status_msg = "Menyusun hasil analisis..."
        hasil_html, summary_html, total_asli, total_palsu = evaluate_result(df_result)
        chart_html = generate_bar_chart(total_asli, total_palsu)
        
        # Prepare display dataframe
        df_display = df_result[["komentar", "Prob_Asli", "Label_Pred", "Confidence"]].copy()
        df_display["Prob_Asli"] = df_display["Prob_Asli"].round(3)
        df_display["Confidence"] = df_display["Confidence"].round(3)
        
        status_msg = f"Proses selesai! {len(df_result)} komentar dianalisis."
        
        return status_msg, hasil_html, df_display, summary_html, chart_html

    except Exception as e:
        error_msg = f"Error: {str(e)}"
        print(f"ERROR: {error_msg}")
        return error_msg, "<div style='color:#d32f2f;text-align:center;'>Terjadi kesalahan saat proses</div>", None, "", ""

# ----------------------------
# 8. Gradio UI
# ----------------------------
custom_css = """
body { background-color: #ffffff !important; }
.gradio-container { background-color: #ffffff !important; }
.gr-button { 
    background-color: #1976d2 !important; 
    color: white !important; 
    border-radius: 8px !important; 
    border: none !important;
    padding: 12px 24px !important;
    font-weight: bold !important;
}
.gr-button:hover { background-color: #1565c0 !important; }
textarea, input[type="text"] { 
    background-color: #fafafa !important;
    color: #333333 !important;
    border: 2px solid #e0e0e0 !important;
    border-radius: 8px !important;
    padding: 12px !important;
}
.output-dataframe { 
    background-color: #ffffff !important;
    border: 1px solid #e0e0e0 !important;
    border-radius: 8px !important;
}
"""

with gr.Blocks(theme=gr.themes.Soft(), css=custom_css, title="Sistem Deteksi Keaslian Produk Tokopedia") as demo:
    
    # Header
    gr.HTML("""
    <div style='text-align:center; padding:20px; background:linear-gradient(135deg, #667eea 0%, #764ba2 100%); color:white; border-radius:10px; margin-bottom:20px;'>
        <h1 style='margin:0; font-size:2.5em;'>🔍 Sistem Deteksi Keaslian Produk</h1>
        <p style='margin:10px 0 0 0; font-size:1.2em;'>Analisis ulasan produk Tokopedia dengan AI dan Rule-Based</p>
        <p style='margin:5px 0 0 0; font-size:0.9em;'>Komentar palsu diberi bobot 3x lebih berat dalam perhitungan</p>
    </div>
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            url_input = gr.Textbox(
                label="📝 URL Review Tokopedia",
                placeholder="Contoh: https://www.tokopedia.com/nama-toko/nama-produk",
                lines=2,
                max_lines=2
            )
            analyze_btn = gr.Button("🚀 Analisis Keaslian Produk", variant="primary", size="lg")
            status = gr.Textbox(
                label="📊 Status Proses",
                interactive=False,
                show_label=True
            )
    
    with gr.Row():
        with gr.Column(scale=1):
            hasil_pred = gr.HTML(label="🎯 Hasil Analisis Akhir")
        with gr.Column(scale=1):
            chart_output = gr.HTML(label="📈 Visualisasi Hasil")
    
    total_info = gr.HTML(label="📋 Ringkasan Analisis")
    
    output_table = gr.DataFrame(
        headers=["Komentar", "Probabilitas Asli", "Label Prediksi", "Confidence"], 
        wrap=True,
        label="📄 Detail Prediksi per Komentar",
        elem_id="results-table"
    )

    # Event handler
    analyze_btn.click(
        sistem_rekomendasi_ui,
        inputs=[url_input],
        outputs=[status, hasil_pred, output_table, total_info, chart_output]
    )
    
    # Footer
    gr.HTML("""
    <div style='text-align:center; margin-top:30px; padding:15px; background-color:#f8f9fa; border-radius:8px; color:#666;'>
        <p style='margin:5px;'><b>Fitur Sistem:</b> Scraping Otomatis • Hybrid AI + Rule-Based • Bobot 3x untuk Komentar Palsu • Analisis Real-time</p>
        <p style='margin:5px; font-size:0.9em;'>Sistem akan otomatis mengambil semua komentar yang tersedia tanpa batasan jumlah</p>
    </div>
    """)

# ----------------------------
# 9. Run Application
# ----------------------------
if __name__ == "__main__":
    print("🚀 Menjalankan Sistem Deteksi Keaslian Produk Tokopedia...")
    print("📝 Fitur: Scraping tanpa headless • Hybrid classification • Bobot 3x untuk komentar palsu")
    print("🌐 Buka browser dan akses localhost yang ditampilkan...")
    
    demo.launch(
        share=False, 
        inbrowser=True,
        server_name="0.0.0.0",
        server_port=7860
    )