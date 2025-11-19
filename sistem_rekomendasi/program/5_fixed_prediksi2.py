# 5_improved_prediksi.py
import os
import time
import re
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver import ActionChains
from webdriver_manager.chrome import ChromeDriverManager

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

import gradio as gr

# ----------------------------
# Konfigurasi Path Utama
# ----------------------------
BASE_DIR = r"D:\TA\TokPed\sistem_rekomendasi"
RESULT_DIR = os.path.join(BASE_DIR, "hasil_rekomendasi")
MODEL_DIR = os.path.join(BASE_DIR, "model_terbaik")

MODEL_PATH = os.path.join(MODEL_DIR, "model_K10_F10_E20_B16_D0.3.keras")
TOKENIZER_PATH = os.path.join(MODEL_DIR, "tokenizer.pkl")
MAXLEN_PATH = os.path.join(MODEL_DIR, "maxlen.pkl")

os.makedirs(RESULT_DIR, exist_ok=True)

# ----------------------------
# 1. Scraping Komentar Tokopedia (HEADLESS)
# ----------------------------
def scrape_tokopedia(url):
    if not url:
        return None, "URL kosong."

    options = webdriver.ChromeOptions()
    options.add_argument("--disable-gpu")
    # ✅ HEADLESS MODE DI AKTIFKAN
    options.add_argument("--headless=new")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    # ✅ Tambahan arguments untuk headless yang lebih stabil
    options.add_argument("--disable-extensions")
    options.add_argument("--disable-browser-side-navigation")
    options.add_argument("--disable-features=VizDisplayCompositor")
    options.add_argument("--disable-background-timer-throttling")
    options.add_argument("--disable-backgrounding-occluded-windows")
    options.add_argument("--disable-renderer-backgrounding")
    # ✅ User agent untuk menghindari deteksi bot
    options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    
    try:
        print("Memulai scraping dalam mode headless...")
        driver.get(url)

        # Tunggu halaman loading
        WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "article.css-15m2bcr"))
        )
        time.sleep(2)  # Tambah waktu tunggu untuk headless
        
        # Click untuk memastikan halaman aktif
        actions = ActionChains(driver)
        actions.move_by_offset(10, 10).click().perform()
        
    except Exception as e:
        driver.quit()
        return None, f"Gagal memuat halaman ulasan: {e}"

    data = []
    page_count = 0
    max_pages = 10  # Batasi halaman untuk menghindari infinite loop
    
    try:
        while page_count < max_pages:
            time.sleep(2)  # Tunggu lebih lama di headless
            
            # Klik semua "Selengkapnya"
            buttons = driver.find_elements(By.XPATH, "//button[contains(., 'Selengkapnya')]")
            for btn in buttons:
                try:
                    driver.execute_script("arguments[0].click();", btn)
                    time.sleep(0.5)
                except:
                    continue

            # Ambil komentar
            reviews = driver.find_elements(By.CSS_SELECTOR, "span[data-testid='lblItemUlasan']")
            new_data_count = 0
            
            for r in reviews:
                text = r.text.strip()
                if text and text not in data:
                    data.append(text)
                    new_data_count += 1

            print(f"Halaman {page_count + 1}: {new_data_count} komentar baru")

            # Coba klik next page
            try:
                next_button = WebDriverWait(driver, 5).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, "button[aria-label^='Laman berikutnya']"))
                )
                driver.execute_script("arguments[0].click();", next_button)
                page_count += 1
                
                # Tunggu loading halaman baru
                WebDriverWait(driver, 10).until(
                    EC.presence_of_all_elements_located((By.CSS_SELECTOR, "article.css-15m2bcr"))
                )
                
            except Exception:
                print("Tidak ada halaman berikutnya atau sudah di halaman terakhir")
                break
                
    except Exception as e:
        print(f"Error selama scraping: {e}")
    finally:
        driver.quit()

    if not data:
        return None, "Tidak ada komentar ditemukan."

    csv_path = os.path.join(RESULT_DIR, "ulasan.csv")
    df = pd.DataFrame(data, columns=["komentar"])
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    
    print(f"Scraping selesai. Total {len(df)} komentar ditemukan.")
    return df, f"Berhasil mengambil {len(df)} komentar."

# ----------------------------
# 2. COMPREHENSIVE DICTIONARY & RULE-BASED CLASSIFICATION
# ----------------------------
class ComprehensiveProductClassifier:
    def __init__(self):
        # ========== KAMUS PALSU (KELUHAN KUALITAS BARANG) ==========
        self.palsu_categories = {
            # Kategori 1: Kata langsung menyebut palsu
            'explicit_palsu': [
                'palsu', 'kw', 'imitasi', 'aspal', 'replika', 'copy', 'fake', 
                'barang aspal', 'kelas kw', 'super kw', 'kw1', 'kw2', 'kw3',
                'quality kw', 'barang tiruan', 'barang abal-abal', 'barang murahan'
            ],
            
            # Kategori 2: Keluhan kualitas parah
            'quality_severe': [
                'jelek banget', 'sangat jelek', 'buruk sekali', 'sangat buruk',
                'parah banget', 'sangat parah', 'kacau banget', 'sangat kacau',
                'mengecewakan banget', 'sangat mengecewakan', 'kecewa berat',
                'sangat kecewa', 'penipuan', 'tipu-tipu', 'ditipu', 'tertipu',
                'bodong', 'scam', 'penipu', 'toko penipu'
            ],
            
            # Kategori 3: Kerusakan fisik
            'physical_damage': [
                'rusak', 'cacat', 'retak', 'sobek', 'pecah', 'patah', 'lecet',
                'penyok', 'penyok', 'bekas', 'bekas pakai', 'tidak baru',
                'kotor', 'kumal', 'kusam', 'berjamur', 'berdebu', 'kualitas rongsokan',
                'seperti bekas', 'seperti second', 'seperti barang lama'
            ],
            
            # Kategori 4: Tidak sesuai ekspektasi
            'not_as_expected': [
                'tidak sesuai', 'beda jauh', 'sangat berbeda', 'tidak sama',
                'beda foto', 'tidak seperti foto', 'tidak sesuai gambar',
                'tidak sesuai deskripsi', 'ukuran beda', 'warna beda',
                'bahan beda', 'model beda', 'kualitas beda', 'bohong',
                'membohongi', 'deskripsi tidak sesuai'
            ],
            
            # Kategori 5: Material jelek
            'bad_material': [
                'bahan jelek', 'material jelek', 'kain jelek', 'kulit jelek',
                'plastik jelek', 'kualitas murah', 'kelas rendah', 'murahan',
                'kualitas pasaran', 'kualitas abal', 'bahan tipis', 'kualitas tipis',
                'bahan kasar', 'kualitas kampungan', 'seperti plastik mainan'
            ],
            
            # Kategori 6: Fungsi tidak bekerja
            'not_working': [
                'tidak bisa dipakai', 'tidak berfungsi', 'rusak saat datang',
                'mati', 'error', 'hang', 'blank', 'tidak nyala', 'tidak jalan',
                'gagal fungsi', 'cacat produksi'
            ]
        }
        
        # ========== KAMUS ASLI (NON-KUALITAS) ==========
        self.asli_categories = {
            # Kategori 1: Eksplisit menyebut asli
            'explicit_asli': [
                'asli', 'ori', 'original', 'authentic', 'genuine', 'resmi',
                'orisinal', 'barang asli', 'produk asli', 'original banget',
                'asli pabrik', 'garansi resmi'
            ],
            
            # Kategori 2: Pengiriman & packaging
            'delivery_packaging': [
                'ongkir', 'ongkos kirim', 'biaya kirim', 'pengiriman', 'kurir',
                'jne', 'jnt', 'tiki', 'pos indonesia', 'gosend', 'grab express',
                'packing', 'bungkus', 'bubble wrap', 'dus', 'kardus', 'pembungkus',
                'packing aman', 'packing bagus', 'packing rapi', 'dikirim cepat',
                'pengiriman cepat', 'pengiriman lambat', 'pengiriman lama'
            ],
            
            # Kategori 3: Kepuasan tinggi
            'high_satisfaction': [
                'puas banget', 'sangat puas', 'puas sekali', 'sangat bagus',
                'bagus banget', 'sangat baik', 'mantap banget', 'keren banget',
                'wow', 'memuaskan', 'sangat memuaskan', 'recommended banget',
                'rekomendasi banget', 'worth it', 'layak', 'sepadan', 'pas',
                'cocok', 'sesuai ekspektasi', 'melebihi ekspektasi'
            ],
            
            # Kategori 4: Pelayanan toko
            'store_service': [
                'penjual ramah', 'respon cepat', 'pelayanan bagus', 'admin ramah',
                'seller baik', 'toko recommended', 'reliable', 'terpercaya',
                'profesional', 'fast respon', 'respon ramah'
            ],
            
            # Kategori 5: Harga & value
            'price_value': [
                'harga terjangkau', 'harga murah', 'harga pas', 'harga sesuai',
                'harga worth it', 'harga bersaing', 'harga promo', 'diskon',
                'murah meriah', 'harga pantas'
            ]
        }

    def calculate_detailed_scores(self, text):
        """Hitung skor detail untuk setiap kategori"""
        text_lower = text.lower()
        
        scores = {
            'palsu_total': 0,
            'asli_total': 0,
            'category_details': {}
        }
        
        # Hitung skor PALSU
        palsu_scores = {}
        for category, keywords in self.palsu_categories.items():
            score = 0
            for keyword in keywords:
                if keyword in text_lower:
                    # Beri bobot lebih untuk kata yang lebih spesifik
                    if len(keyword.split()) > 1:  # multi-word phrases
                        score += 3
                    else:  # single word
                        score += 2
            palsu_scores[category] = score
            scores['palsu_total'] += score
        
        # Hitung skor ASLI
        asli_scores = {}
        for category, keywords in self.asli_categories.items():
            score = 0
            for keyword in keywords:
                if keyword in text_lower:
                    if len(keyword.split()) > 1:
                        score += 3
                    else:
                        score += 2
            asli_scores[category] = score
            scores['asli_total'] += score
        
        scores['category_details'] = {
            'palsu': palsu_scores,
            'asli': asli_scores
        }
        
        return scores

    def rule_based_classification(self, text):
        """Klasifikasi berbasis rule dengan kamus komprehensif"""
        scores = self.calculate_detailed_scores(text)
        
        palsu_total = scores['palsu_total']
        asli_total = scores['asli_total']
        
        # Debug information
        print(f"Text: {text[:50]}...")
        print(f"Palsu score: {palsu_total}, Asli score: {asli_total}")
        
        # Decision logic dengan threshold
        if palsu_total == 0 and asli_total == 0:
            return "Netral", 0.5, scores
        
        # Jika ada strong indicator palsu
        strong_palsu_indicators = scores['category_details']['palsu']['explicit_palsu'] > 0
        strong_asli_indicators = scores['category_details']['asli']['explicit_asli'] > 0
        
        if strong_palsu_indicators and not strong_asli_indicators:
            return "Palsu", 0.1, scores
        elif strong_asli_indicators and not strong_palsu_indicators:
            return "Asli", 0.9, scores
        
        # Normal probability calculation
        total_score = palsu_total + asli_total
        prob_asli = asli_total / total_score if total_score > 0 else 0.5
        
        # Adjust probabilities based on category dominance
        if palsu_total > asli_total * 2:
            prob_asli = max(0.1, prob_asli - 0.3)
        elif asli_total > palsu_total * 2:
            prob_asli = min(0.9, prob_asli + 0.3)
        
        label = "Asli" if prob_asli >= 0.5 else "Palsu"
        
        return label, prob_asli, scores

# ----------------------------
# 3. Hybrid Classification System
# ----------------------------
class HybridClassifier:
    def __init__(self):
        self.rule_classifier = ComprehensiveProductClassifier()
        self.stemmer = StemmerFactory().create_stemmer()
        self.model = None
        self.tokenizer = None
        self.MAX_LEN = 100
        
        self.load_resources()
    
    def load_resources(self):
        """Load model dan tokenizer"""
        try:
            if os.path.exists(MODEL_PATH):
                self.model = load_model(MODEL_PATH)
                print("Model LSTM loaded")
            else:
                print("Model LSTM tidak ditemukan, menggunakan rule-based only")
                
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
    
    def preprocess_text(self, text):
        """Preprocessing sederhana"""
        if not isinstance(text, str):
            return ''
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        text = text.lower()
        return self.stemmer.stem(text)
    
    def ml_prediction(self, texts):
        """Prediksi menggunakan model ML"""
        if self.model is None or self.tokenizer is None:
            return None
            
        try:
            processed_texts = [self.preprocess_text(text) for text in texts]
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
        """Klasifikasi hybrid dengan kamus komprehensif"""
        comments = df["komentar"].tolist()
        ml_probs = self.ml_prediction(comments)
        
        results = []
        for i, comment in enumerate(comments):
            # Rule-based classification dengan kamus detail
            rule_label, rule_prob, scores = self.rule_classifier.rule_based_classification(comment)
            
            # Jika ML prediction available, gabungkan dengan bobot lebih ke rule-based
            if ml_probs is not None:
                ml_prob = ml_probs[i]
                # 80% rule-based, 20% ML (karena rule-based lebih reliable)
                final_prob = 0.8 * rule_prob + 0.2 * ml_prob
            else:
                final_prob = rule_prob
            
            final_label = "Asli" if final_prob >= 0.5 else "Palsu"
            
            results.append({
                'komentar': comment,
                'Prob_Asli': final_prob,
                'Label_Pred': final_label,
                'Rule_Prob': rule_prob,
                'Palsu_Score': scores['palsu_total'],
                'Asli_Score': scores['asli_total'],
                'Category_Details': scores['category_details']
            })
        
        return pd.DataFrame(results)

# ----------------------------
# 4. Initialize Hybrid Classifier
# ----------------------------
hybrid_classifier = HybridClassifier()

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
        confidence = 2 * abs(prob - 0.5)
        confidence_scores.append(confidence)
    
    result_df['Confidence'] = confidence_scores
    
    return result_df

# ----------------------------
# 6. Evaluation & Visualization
# ----------------------------
def evaluate_result(df):
    """Evaluasi hasil prediksi"""
    total_asli = int((df["Label_Pred"] == "Asli").sum())
    total_palsu = int((df["Label_Pred"] == "Palsu").sum())
    total = int(len(df))
    
    if total == 0:
        persentase_asli = persentase_palsu = 0.0
        avg_confidence = 0.0
        total_palsu_score = total_asli_score = 0
    else:
        persentase_asli = total_asli / total * 100
        persentase_palsu = total_palsu / total * 100
        avg_confidence = df["Confidence"].mean()
        total_palsu_score = df['Palsu_Score'].sum()
        total_asli_score = df['Asli_Score'].sum()
    
    # Decision logic yang lebih robust
    if total < 3:
        hasil = "DATA TERLALU SEDIKIT - tidak dapat disimpulkan"
        warna = "#ff9800"
    elif avg_confidence < 0.3:
        hasil = "HASIL TIDAK PASTI - confidence rendah"
        warna = "#ff9800"
    elif total_palsu > total_asli * 2 and total_palsu_score > total_asli_score * 2:
        hasil = "BARANG DIDUGA PALSU - dominasi keluhan kualitas"
        warna = "#e63946"
    elif total_asli > total_palsu * 2 and total_asli_score > total_palsu_score * 2:
        hasil = "BARANG DIDUGA ASLI - sedikit keluhan kualitas"
        warna = "#2a9d8f"
    elif total_palsu_score > total_asli_score * 3:
        hasil = "BARANG DIDUGA PALSU - skor keluhan sangat tinggi"
        warna = "#e63946"
    elif total_asli_score > total_palsu_score * 3:
        hasil = "BARANG DIDUGA ASLI - skor kepuasan sangat tinggi"
        warna = "#2a9d8f"
    else:
        hasil = "HASIL CENDERUNG NETRAL - data berimbang"
        warna = "#ff9800"
    
    hasil_html = f"""
    <div style='text-align:center; margin-top:10px; padding:15px; border-radius:10px; background-color:#f8f9fa; border:2px solid {warna};'>
        <h3 style='color:{warna}; margin-bottom:15px;'>{hasil}</h3>
        <div style='display:flex; justify-content:center; gap:30px; margin-bottom:10px;'>
            <div style='text-align:center;'>
                <div style='font-size:24px; color:#2a9d8f; font-weight:bold;'>{persentase_asli:.1f}%</div>
                <div style='color:#333333;'>Asli ({total_asli})</div>
            </div>
            <div style='text-align:center;'>
                <div style='font-size:24px; color:#e63946; font-weight:bold;'>{persentase_palsu:.1f}%</div>
                <div style='color:#333333;'>Palsu ({total_palsu})</div>
            </div>
        </div>
        <p style='color:#333333; margin:5px;'><b>Total Komentar:</b> {total}</p>
        <p style='color:#333333; margin:5px;'><b>Confidence:</b> {avg_confidence:.3f}</p>
        <p style='color:#333333; margin:5px;'><b>Skor Kualitas:</b> {total_palsu_score} | <b>Skor Kepuasan:</b> {total_asli_score}</p>
    </div>
    """
    
    summary_html = f"<div style='text-align:center; color:#333333; margin-top:8px;'><b>Summary:</b> {total_asli} Asli | {total_palsu} Palsu | Confidence: {avg_confidence:.3f}</div>"
    
    return hasil_html, summary_html, total_asli, total_palsu

def generate_bar_chart(asli, palsu):
    """Generate bar chart untuk visualisasi"""
    labels = ["Asli", "Palsu"]
    values = [asli, palsu]
    colors = ["#2a9d8f", "#e63946"]

    plt.figure(figsize=(5, 4))
    bars = plt.bar(labels, values, color=colors, alpha=0.8)
    plt.title("Distribusi Prediksi Komentar", fontsize=12, fontweight='bold', color='#333333')
    plt.ylabel("Jumlah Komentar", fontsize=10, color='#333333')
    plt.xticks(color='#333333')
    plt.yticks(color='#333333')
    
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{value}', ha='center', va='bottom', fontweight='bold', color='#333333')
    
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=100, bbox_inches="tight", facecolor='white')
    plt.close()
    buf.seek(0)
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"<img src='data:image/png;base64,{encoded}' width='450' style='display:block; margin:auto;'/>"

# ----------------------------
# 7. Main UI Function
# ----------------------------
def sistem_rekomendasi_ui(url):
    """Main function untuk UI"""
    status_msg = "Memulai proses..."
    
    try:
        status_msg = "Sedang scraping komentar dari Tokopedia (HEADLESS MODE)..."
        df_scrape, msg = scrape_tokopedia(url)
        if df_scrape is None:
            return msg, "<div style='text-align:center;color:#d32f2f;'>Gagal: tidak ada komentar</div>", None, "", ""
        
        status_msg = "Menganalisis komentar dengan sistem hybrid..."
        df_result = process_comments(df_scrape)
        
        csv_path = os.path.join(RESULT_DIR, "ulasan_prediksi.csv")
        df_result[["komentar", "Prob_Asli", "Label_Pred"]].to_csv(csv_path, index=False, encoding="utf-8-sig")
        
        status_msg = "Menyusun hasil analisis..."
        hasil_html, summary_html, total_asli, total_palsu = evaluate_result(df_result)
        chart_html = generate_bar_chart(total_asli, total_palsu)
        
        df_display = df_result[["komentar", "Prob_Asli", "Label_Pred"]].copy()
        df_display["Prob_Asli"] = df_display["Prob_Asli"].round(3)
        
        status_msg = f"Proses selesai! {len(df_result)} komentar dianalisis."
        
        return status_msg, hasil_html, df_display, summary_html, chart_html

    except Exception as e:
        error_msg = f"Error: {str(e)}"
        print(f"ERROR: {error_msg}")
        return error_msg, "<div style='color:#d32f2f;text-align:center;'>Terjadi kesalahan saat proses. Cek log terminal.</div>", None, "", ""

# ----------------------------
# 8. Gradio UI dengan Tema Light
# ----------------------------
custom_css = """
body { 
    background-color: #ffffff !important; 
    color: #333333 !important;
}
.gradio-container { 
    background-color: #ffffff !important; 
    color: #333333 !important;
}
.gr-button { 
    background-color: #1976d2 !important; 
    color: white !important; 
    border-radius: 8px !important; 
    border: none !important;
}
.gr-button:hover { 
    background-color: #1565c0 !important; 
}
textarea, input[type="text"], input[type="url"] { 
    background-color: #fafafa !important;
    color: #333333 !important;
    border: 2px solid #e0e0e0 !important;
    border-radius: 8px !important;
    padding: 12px !important;
}
.gr-box {
    background-color: #fafafa !important;
    color: #333333 !important;
    border: 1px solid #e0e0e0 !important;
}
.gr-textbox label, .gr-input label {
    color: #333333 !important;
    font-weight: 500 !important;
}
.output-dataframe { 
    background-color: #ffffff !important;
    color: #333333 !important;
    border: 1px solid #e0e0e0 !important;
    border-radius: 8px !important;
}
.output-dataframe table {
    background-color: #ffffff !important;
    color: #333333 !important;
}
.output-dataframe th {
    background-color: #f5f5f5 !important;
    color: #333333 !important;
    font-weight: 600 !important;
}
.output-dataframe td {
    background-color: #ffffff !important;
    color: #333333 !important;
    border-color: #e0e0e0 !important;
}
.gr-markdown {
    color: #333333 !important;
}
.gr-label {
    color: #333333 !important;
    font-weight: 600 !important;
}
"""

with gr.Blocks(theme=gr.themes.Soft(), css=custom_css) as demo:
    gr.HTML("<div style='height:20px;'></div>")
    gr.HTML("<h1 style='text-align:center; color:#333333;'>Sistem Deteksi Keaslian Produk Tokopedia</h1>")
    gr.HTML("<p style='text-align:center; color:#666666;'>Analisis ulasan produk dengan sistem hybrid AI dan rule-based</p>")
    
    with gr.Row():
        with gr.Column(scale=1):
            url_input = gr.Textbox(
                label="URL Review Tokopedia",
                placeholder="https://www.tokopedia.com/nama-toko/nama-produk/ulasan",
                lines=1
            )
            analyze_btn = gr.Button("Analisis Keaslian Produk", variant="primary")
            status = gr.Textbox(label="Status Proses", interactive=False)
    
    with gr.Row():
        with gr.Column(scale=1):
            hasil_pred = gr.HTML(label="Hasil Analisis")
        with gr.Column(scale=1):
            chart_output = gr.HTML(label="Visualisasi")
    
    total_info = gr.HTML(label="Ringkasan")
    output_table = gr.DataFrame(
        headers=["Komentar", "Probabilitas Asli", "Label Prediksi"], 
        wrap=True,
        label="Detail Prediksi Komentar",
        elem_id="results-table"
    )

    analyze_btn.click(
        sistem_rekomendasi_ui,
        inputs=[url_input],
        outputs=[status, hasil_pred, output_table, total_info, chart_output]
    )

# ----------------------------
# 9. Run Application
# ----------------------------
if __name__ == "__main__":
    print("Menjalankan Sistem Deteksi Keaslian Produk...")
    print("Mode: HEADLESS - tidak ada window browser yang terbuka")
    print("Menggunakan kamus komprehensif dan sistem hybrid")
    
    demo.launch(share=False, inbrowser=True)