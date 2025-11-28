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
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

from gensim.models import Word2Vec
from tensorflow.keras.models import load_model

import gradio as gr

BASE_DIR = "sistem_rekomendasi"
RESULT_DIR = os.path.join(BASE_DIR, "hasil_rekomendasi")
MODEL_DIR = os.path.join(BASE_DIR, "model_terbaik")

EMBEDDINGS_PATH = os.path.join(BASE_DIR, "model_word2vec_balanced", "word2vec_tokopedia_balanced.model")
LABELS_PATH = os.path.join(BASE_DIR, "model_word2vec_balanced", "labels.npy")
MODEL_PATH = os.path.join(MODEL_DIR, "model_K10_F10_E20_B16_D0.3.keras")

os.makedirs(RESULT_DIR, exist_ok=True)

def scrape_tokopedia(url):
    if not url:
        return None, "URL kosong."

    if '?' in url:
        url = url.split('?')[0]
    if not url.endswith('/review'):
        url = url + '/review'

    options = webdriver.ChromeOptions()
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver.get(url)

    try:
        WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "article.css-15m2bcr"))
        )
        time.sleep(1)
        actions = ActionChains(driver)
        actions.move_by_offset(10, 10).click().perform()
    except Exception as e:
        try:
            driver.quit()
        except:
            pass
        return None, f"Gagal memuat halaman ulasan: {e}"

    data = []
    while True:
        time.sleep(1)
        buttons = driver.find_elements(By.XPATH, "//button[contains(., 'Selengkapnya')]")
        for btn in buttons:
            try:
                driver.execute_script("arguments[0].click();", btn)
            except:
                continue

        reviews = driver.find_elements(By.CSS_SELECTOR, "span[data-testid='lblItemUlasan']")
        for r in reviews:
            text = r.text.strip()
            if text and text not in data:
                data.append(text)

        try:
            next_button = WebDriverWait(driver, 5).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, "button[aria-label^='Laman berikutnya']"))
            )
            driver.execute_script("arguments[0].click();", next_button)
            WebDriverWait(driver, 10).until(
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, "article.css-15m2bcr"))
            )
        except:
            break

    driver.quit()

    if not data:
        return None, "Tidak ada komentar ditemukan."

    csv_path = os.path.join(RESULT_DIR, "ulasan.csv")
    df = pd.DataFrame(data, columns=["komentar"])
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    return df, f"Berhasil mengambil {len(df)} komentar."

stem_factory = StemmerFactory()
stemmer = stem_factory.create_stemmer()
stop_factory = StopWordRemoverFactory()
stopwords = stop_factory.get_stop_words()

normalisasi_dict = {
    "bgt": "banget", "gk": "tidak", "ga": "tidak", "gak": "tidak",
    "nggak": "tidak", "ngga": "tidak", "tp": "tapi", "yg": "yang",
    "brg": "barang", "bgs": "bagus", "rekomen": "direkomendasikan",
    "rek": "rekomendasi", "trmksh": "terima kasih", "mksh": "makasih",
    "udh": "sudah", "sdh": "sudah", "blm": "belum", "sm": "sama",
    "aj": "saja", "nyah": "nya", "ny": "nya", "bhn": "bahan",
    "expetasi": "ekspektasi", "ok": "oke", "mantul": "mantap betul",
    "kw": "palsu", "ori": "original", "asl": "asli", "jeleg": "jelek",
    "baguz": "bagus", "lumayan": "cukup", "sip": "baik", "nice": "bagus"
}

def clean_text(text):
    if not isinstance(text, str):
        return ''
    text = re.sub(r'#|@', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

def preprocess_df(df):
    df = df.drop_duplicates(subset=["komentar"]).reset_index(drop=True)
    df["cleaned"] = df["komentar"].apply(clean_text)
    df["tokens"] = df["cleaned"].apply(lambda x: x.split())
    df["normalized"] = df["tokens"].apply(lambda x: [normalisasi_dict.get(w, w) for w in x])
    df["stemmed"] = df["normalized"].apply(lambda x: [stemmer.stem(w) for w in x if w not in stopwords])
    df["cleaned_final"] = df["stemmed"].apply(lambda x: ' '.join(x))
    return df

model = load_model(MODEL_PATH)

w2v_model = None
if os.path.exists(EMBEDDINGS_PATH):
    try:
        w2v_model = Word2Vec.load(EMBEDDINGS_PATH)
    except Exception:
        w2v_model = None

MAX_LEN = 50

def get_comment_embedding(tokens, model, vector_size=150):
    vectors = []
    for t in tokens:
        if t in model.wv.key_to_index:
            vectors.append(model.wv[t])
    if len(vectors) == 0:
        return np.zeros(vector_size)
    return np.mean(vectors, axis=0)

def prepare_embeddings(df_clean):
    if w2v_model is None:
        raise ValueError("Model Word2Vec tidak tersedia")
    
    tokens_list = [text.split() for text in df_clean["cleaned_final"].astype(str)]
    
    embeddings = np.array([get_comment_embedding(tokens, w2v_model) for tokens in tokens_list])
    
    return embeddings

def predict_with_embeddings(df):
    if "cleaned_final" not in df.columns:
        df["cleaned_final"] = df["komentar"].astype(str).apply(clean_text)

    X_embeddings = prepare_embeddings(df)
    
    preds = model.predict(X_embeddings, verbose=0)
    if preds.ndim == 2 and preds.shape[1] == 1:
        probs = preds.flatten()
    elif preds.ndim == 2 and preds.shape[1] == 2:
        probs = preds[:, 1]
    else:
        probs = preds.flatten()

    df["Prob_Asli"] = probs
    df["Label_Pred"] = np.where(probs >= 0.5, "Asli", "Palsu")
    return df

def contextual_analysis(comment, ml_prob):
    """
    Analisis konteks untuk meningkatkan pemahaman makna teks
    """
    text_lower = comment.lower()
    
    # Kata kunci yang sangat kuat menunjukkan barang palsu
    strong_fake_indicators = [
        'palsu', 'kw', 'tipu', 'scam', 'bodong', 'aspal', 'penipuan', 
        'fake', 'imitasi', 'jangan beli', 'menyesal beli', 'kapok',
        'kelas kw', 'kw1', 'kw2', 'kw3', 'super kw', 'barang aspal'
    ]
    
    # Kata kunci yang menunjukkan ketidaksesuaian
    mismatch_indicators = [
        'tidak sesuai', 'beda foto', 'tidak seperti foto', 'sangat berbeda',
        'tidak cocok', 'beda deskripsi', 'tidak sama'
    ]
    
    # Kata kunci kualitas buruk
    quality_indicators = [
        'jelek', 'buruk', 'rusak', 'cacat', 'retak', 'sobek', 'pecah',
        'patah', 'kualitas murah', 'murahan', 'kualitas rendah'
    ]
    
    # Kata kunci yang sangat kuat menunjukkan barang asli
    strong_genuine_indicators = [
        'asli', 'ori', 'original', 'resmi', 'garansi', 'terpercaya'
    ]
    
    # Kata kunci kepuasan
    satisfaction_indicators = [
        'puas', 'bagus', 'recommended', 'rekomendasi', 'mantap', 'sesuai',
        'memuaskan', 'kualitas bagus', 'sangat puas', 'sangat bagus'
    ]
    
    final_prob = ml_prob
    
    # Deteksi indikator palsu yang kuat
    fake_count = sum(1 for indicator in strong_fake_indicators if indicator in text_lower)
    mismatch_count = sum(1 for indicator in mismatch_indicators if indicator in text_lower)
    quality_count = sum(1 for indicator in quality_indicators if indicator in text_lower)
    
    # Deteksi indikator asli yang kuat
    genuine_count = sum(1 for indicator in strong_genuine_indicators if indicator in text_lower)
    satisfaction_count = sum(1 for indicator in satisfaction_indicators if indicator in text_lower)
    
    # Analisis sentimen berdasarkan kata kunci
    total_negative = fake_count + mismatch_count + quality_count
    total_positive = genuine_count + satisfaction_count
    
    # Penyesuaian probability berdasarkan konteks
    if total_negative > 0:
        penalty = min(0.7, total_negative * 0.15)
        final_prob = max(0.05, final_prob - penalty)
    
    if total_positive > 0:
        boost = min(0.3, total_positive * 0.1)
        final_prob = min(0.95, final_prob + boost)
    
    # Kasus khusus: jika ada kata kunci palsu yang sangat kuat
    strong_fake_found = any(indicator in text_lower for indicator in ['palsu', 'kw', 'scam', 'tipu'])
    if strong_fake_found:
        final_prob = max(0.02, final_prob * 0.3)
    
    # Kasus khusus: jika ada kata kunci asli yang sangat kuat
    strong_genuine_found = any(indicator in text_lower for indicator in ['asli', 'ori', 'original'])
    if strong_genuine_found and total_negative == 0:
        final_prob = min(0.98, final_prob + 0.2)
    
    return final_prob

def classify_with_model(df):
    df_result = predict_with_embeddings(df)
    
    results = []
    for i, row in df_result.iterrows():
        comment = row["komentar"]
        ml_prob = row["Prob_Asli"]
        
        # Analisis konteks untuk meningkatkan akurasi
        final_prob = contextual_analysis(comment, ml_prob)
        final_label = "Asli" if final_prob >= 0.5 else "Palsu"
        
        results.append({
            'komentar': comment,
            'Prob_Asli': final_prob,
            'Label_Pred': final_label
        })
    
    return pd.DataFrame(results)

def evaluate_result(df):
    total_asli = (df["Label_Pred"] == "Asli").sum()
    total_palsu = (df["Label_Pred"] == "Palsu").sum()
    total = len(df)
    
    if total == 0:
        return "<div style='color:#d32f2f;text-align:center;'>Tidak ada data untuk dianalisis</div>", "", 0, 0
    
    persentase_asli = total_asli / total * 100
    persentase_palsu = total_palsu / total * 100
    
    if persentase_palsu >= 30:
        hasil = "BARANG PALSU"
        warna = "#e63946"
        alasan = f"Dari total {total} komentar, hasil presentase barang palsu mencapai {persentase_palsu:.1f}% dan asli {persentase_asli:.1f}%."
        rekomendasi = f"Produk kemungkinan palsu, hindari pembelian."
    else:
        hasil = "BARANG ASLI"
        warna = "#2a9d8f"
        alasan = f"Dari total {total} komentar, hasil presentase barang asli mencapai {persentase_asli:.1f}% dan palsu {persentase_palsu:.1f}%."
        rekomendasi = f"Produk kemungkinan asli, namun pengguna disarankan untuk memeriksa kembali."
    
    hasil_html = f"""
    <div style='text-align:center; margin-top:10px; padding:20px; border-radius:10px; background-color:#ffffff; border:3px solid {warna};'>
        <h2 style='color:{warna}; margin-bottom:20px;'>{hasil}</h2>
        <div style='display:flex; justify-content:center; gap:40px; margin-bottom:15px;'>
            <div style='text-align:center;'>
                <div style='font-size:28px; color:#2a9d8f; font-weight:bold;'>{persentase_asli:.1f}%</div>
                <div style='color:#000000; font-size:14px;'>Asli ({total_asli})</div>
            </div>
            <div style='text-align:center;'>
                <div style='font-size:28px; color:#e63946; font-weight:bold;'>{persentase_palsu:.1f}%</div>
                <div style='color:#000000; font-size:14px;'>Palsu ({total_palsu})</div>
            </div>
        </div>
        <p style='color:#000000; margin:8px 0; font-size:14px;'>Total Komentar: {total}</p>
        <p style='color:#000000; margin:8px 0; font-size:14px;'>Analisis: {alasan}</p>
        <p style='color:#000000; margin:8px 0; font-size:14px;'>Rekomendasi: {rekomendasi}</p>
    </div>
    """
    
    summary_html = f"<div style='text-align:center; color:#000000; margin-top:10px; font-size:14px;'>Summary: {total_asli} Asli | {total_palsu} Palsu</div>"
    
    return hasil_html, summary_html, total_asli, total_palsu

def generate_bar_chart(asli, palsu):
    labels = ["Asli", "Palsu"]
    values = [asli, palsu]
    colors = ["#2a9d8f", "#e63946"]

    plt.figure(figsize=(6, 4))
    bars = plt.bar(labels, values, color=colors, alpha=0.8)
    plt.title("Distribusi Prediksi Komentar", fontsize=14, fontweight='bold', color='black')
    plt.ylabel("Jumlah Komentar", fontsize=12, color='black')
    plt.xticks(color='black')
    plt.yticks(color='black')
    
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{value}', ha='center', va='bottom', fontweight='bold', color='black')
    
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=100, bbox_inches="tight", facecolor='white')
    plt.close()
    buf.seek(0)
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"<img src='data:image/png;base64,{encoded}' width='500' style='display:block; margin:auto;'/>"

def sistem_rekomendasi_ui(url):
    status_msg = "Memulai proses..."
    
    try:
        status_msg = "Sedang scraping komentar dari Tokopedia..."
        df_scrape, msg = scrape_tokopedia(url)
        if df_scrape is None:
            return msg, "<div style='text-align:center;color:#d32f2f;'>Gagal: tidak ada komentar ditemukan</div>", None, "", ""
        
        status_msg = "Melakukan preprocessing data..."
        df_clean = preprocess_df(df_scrape)
        
        status_msg = "Menganalisis komentar dengan AI (Word2Vec + Neural Network)..."
        df_result = classify_with_model(df_clean)
        
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
        return error_msg, "<div style='color:#d32f2f;text-align:center;'>Terjadi kesalahan saat proses</div>", None, "", ""

custom_css = """
body {
    background-color: white !important;
    color: black !important;
}

.gradio-container {
    background-color: white !important;
    color: black !important;
}

#analyze-btn {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    color: white !important;
    border-radius: 8px !important;
    border: none !important;
    padding: 12px 24px !important;
    font-weight: bold !important;
}

#analyze-btn:hover {
    background: linear-gradient(135deg, #5a6fd8 0%, #6a4190 100%) !important;
}

.gr-button { 
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important; 
    color: white !important; 
    border-radius: 8px !important; 
    border: none !important;
    padding: 12px 24px !important;
    font-weight: bold !important;
}

.gr-button:hover { 
    background: linear-gradient(135deg, #5a6fd8 0%, #6a4190 100%) !important; 
}

textarea, input[type="text"], .gr-textbox, .gr-input {
    background-color: #000000 !important;
    color: white !important;
    border: 2px solid #333333 !important;
    border-radius: 8px !important;
    padding: 12px !important;
}

.gr-textbox label, .gr-input label {
    color: black !important;
    font-weight: bold !important;
}

.output-dataframe, .gr-box {
    background-color: #000000 !important;
    color: white !important;
    border: 1px solid #333333 !important;
    border-radius: 8px !important;
}

.gr-markdown, .gr-label {
    color: black !important;
}

.gr-header {
    color: black !important;
}

.svelte-1vd8eap {
    background-color: #000000 !important;
    color: white !important;
}

.svelte-1vd8eap label {
    color: black !important;
}

.gr-form {
    background-color: #000000 !important;
}

.gr-component {
    background-color: #000000 !important;
    color: white !important;
}

.gr-panel {
    background-color: #000000 !important;
    color: white !important;
}

.gr-textbox[data-testid="textbox"] {
    background-color: #000000 !important;
    color: white !important;
}

.gr-form > div {
    background-color: #000000 !important;
}

footer {
    color: black !important;
}

footer p {
    color: black !important;
}

.gr-text-sm {
    color: black !important;
}

.gr-prose {
    color: black !important;
}

.gr-block {
    background-color: #000000 !important;
}

.gr-block-title {
    color: black !important;
}

.gr-block-description {
    color: black !important;
}
"""

with gr.Blocks(css=custom_css, title="Sistem Deteksi Keaslian Produk Tokopedia") as demo:
    
    gr.HTML("""
    <div style='text-align:center; padding:20px; background:linear-gradient(135deg, #667eea 0%, #764ba2 100%); color:white; border-radius:10px; margin-bottom:20px;'>
        <h1 style='margin:0; font-size:2.5em; color:white;'>Sistem Deteksi Keaslian Produk</h1>
        <p style='margin:10px 0 0 0; font-size:1.2em; color:white;'>Analisis ulasan produk Tokopedia dengan AI dan Pemahaman Konteks</p>
        <p style='margin:5px 0 0 0; font-size:0.9em; color:white;'>Threshold: 30% komentar palsu = produk palsu</p>
    </div>
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            url_input = gr.Textbox(
                label="URL Review Tokopedia",
                placeholder="Contoh: https://www.tokopedia.com/nama-toko/nama-produk",
                lines=2,
                max_lines=2
            )
            analyze_btn = gr.Button("Analisis Keaslian Produk", variant="primary", size="lg", elem_id="analyze-btn")
            status = gr.Textbox(
                label="Status Proses",
                interactive=False,
                show_label=True
            )
    
    with gr.Row():
        with gr.Column(scale=1):
            hasil_pred = gr.HTML(label="Hasil Analisis Akhir")
        with gr.Column(scale=1):
            chart_output = gr.HTML(label="Visualisasi Hasil")
    
    total_info = gr.HTML(label="Ringkasan Analisis")
    
    output_table = gr.DataFrame(
        headers=["Komentar", "Probabilitas Asli", "Label Prediksi"], 
        wrap=True,
        label="Detail Prediksi per Komentar",
        elem_id="results-table"
    )

    analyze_btn.click(
        sistem_rekomendasi_ui,
        inputs=[url_input],
        outputs=[status, hasil_pred, output_table, total_info, chart_output]
    )
    
    gr.HTML("""
    <div style='text-align:center; margin-top:30px; padding:15px; background-color:#333333; border-radius:8px; color:black;'>
        <p style='margin:5px; color:white;'><b>Fitur Sistem:</b> Scraping Otomatis • AI dengan Word2Vec • Analisis Konteks • Threshold 30%</p>
        <p style='margin:5px; font-size:0.9em; color:white;'>Sistem akan otomatis mengambil semua komentar yang tersedia tanpa batasan jumlah</p>
    </div>
    """)

if __name__ == "__main__":
    print("Menjalankan Sistem Deteksi Keaslian Produk Tokopedia...")
    print("Fitur: Scraping Selenium • AI dengan Word2Vec • Analisis Konteks • Threshold 30% untuk komentar palsu")
    print("Buka browser dan akses localhost yang ditampilkan...")
    
    demo.launch(
        share=False, 
        inbrowser=True,
        server_name="0.0.0.0",
        server_port=7860
    )