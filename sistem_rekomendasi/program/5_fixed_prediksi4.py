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
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

import gradio as gr

BASE_DIR = r"D:\TA\TokPed\sistem_rekomendasi"
RESULT_DIR = os.path.join(BASE_DIR, "hasil_rekomendasi")
MODEL_DIR = os.path.join(BASE_DIR, "model_terbaik")

EMBEDDINGS_PATH = os.path.join(BASE_DIR, "model_word2vec_balanced", "word2vec_tokopedia_balanced.model")
LABELS_PATH = os.path.join(BASE_DIR, "model_word2vec_balanced", "labels.npy")
MODEL_PATH = os.path.join(MODEL_DIR, "model_K10_F10_E20_B16_D0.3.keras")

TOKENIZER_PATH = os.path.join(MODEL_DIR, "tokenizer.pkl")
MAXLEN_PATH = os.path.join(MODEL_DIR, "maxlen.pkl")

os.makedirs(RESULT_DIR, exist_ok=True)

def scrape_tokopedia(url):
    if not url:
        return None, "URL kosong."

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
    "expetasi": "ekspektasi", "ok": "oke", "mantul": "mantap betul"
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

def try_load_tokenizer():
    if os.path.exists(TOKENIZER_PATH):
        try:
            with open(TOKENIZER_PATH, "rb") as f:
                tok = pickle.load(f)
            return tok, "loaded_from_file"
        except Exception:
            pass
    return None, None

def try_reconstruct_tokenizer_from_embedding_and_w2v(model, w2v_model, max_words=40000):
    try:
        for layer in model.layers:
            if layer.__class__.__name__.lower() == "embedding":
                emb_weights = layer.get_weights()[0]
                break
        else:
            return None, "no_embedding_layer"

        if w2v_model is None:
            return None, "no_w2v"

        w2v_vocab = list(w2v_model.wv.index_to_key)
        w2v_mat = np.array([w2v_model.wv[w] for w in w2v_vocab])

        emb_norm = emb_weights / (np.linalg.norm(emb_weights, axis=1, keepdims=True) + 1e-9)
        w2v_norm = w2v_mat / (np.linalg.norm(w2v_mat, axis=1, keepdims=True) + 1e-9)

        mapping = {}
        for i in range(min(len(emb_norm), len(emb_weights))):
            sims = np.dot(w2v_norm, emb_norm[i])
            best_idx = np.argmax(sims)
            mapping[w2v_vocab[best_idx]] = i

        tokenizer = Tokenizer(oov_token="<OOV>")
        ordered = sorted(mapping.items(), key=lambda x: mapping[x[0]])
        tokenizer.word_index = {word: idx+1 for idx, (word, _) in enumerate(ordered)}
        return tokenizer, "reconstructed_from_w2v"
    except Exception as e:
        return None, f"failed_reconstruct:{e}"

def create_tokenizer_from_texts(texts, num_words=None):
    tokenizer = Tokenizer(oov_token="<OOV>", num_words=num_words)
    tokenizer.fit_on_texts(texts)
    try:
        with open(TOKENIZER_PATH, "wb") as f:
            pickle.dump(tokenizer, f)
    except Exception:
        pass
    return tokenizer

if os.path.exists(MAXLEN_PATH):
    try:
        with open(MAXLEN_PATH, "rb") as f:
            MAX_LEN = pickle.load(f)
    except Exception:
        MAX_LEN = 100
else:
    MAX_LEN = 100

tokenizer, tid = try_load_tokenizer()

class HybridClassifier:
    def __init__(self):
        self.palsu_keywords = [
            'palsu', 'kw', 'jelek', 'buruk', 'rusak', 'cacat', 'retak', 'sobek',
            'pecah', 'patah', 'mengecewakan', 'kecewa', 'tipu', 'bodong', 'scam',
            'tidak sesuai', 'beda foto', 'kualitas murah', 'murahan', 'aspal'
        ]
        
        self.asli_keywords = [
            'asli', 'ori', 'original', 'bagus', 'puas', 'recommended', 'rekomendasi',
            'mantap', 'oke', 'sesuai', 'memuaskan', 'terima kasih', 'kualitas bagus'
        ]
    
    def rule_based_prediction(self, text):
        text_lower = text.lower()
        
        palsu_score = sum(1 for keyword in self.palsu_keywords if keyword in text_lower)
        asli_score = sum(1 for keyword in self.asli_keywords if keyword in text_lower)
        
        if palsu_score == 0 and asli_score == 0:
            return 0.5
        
        total_score = palsu_score + asli_score
        prob_asli = asli_score / total_score
        
        if palsu_score > asli_score * 2:
            prob_asli = max(0.1, prob_asli - 0.3)
        elif asli_score > palsu_score * 2:
            prob_asli = min(0.9, prob_asli + 0.3)
        
        return prob_asli

hybrid_classifier = HybridClassifier()

def prepare_tokenizer_if_needed(df_clean):
    global tokenizer, tid
    if tokenizer is not None:
        return "already_have"

    tok, reason = try_reconstruct_tokenizer_from_embedding_and_w2v(model, w2v_model)
    if tok is not None:
        tokenizer = tok
        tid = reason
        try:
            with open(TOKENIZER_PATH, "wb") as f:
                pickle.dump(tokenizer, f)
        except:
            pass
        return reason

    texts = df_clean["cleaned_final"].astype(str).tolist()
    tokenizer = create_tokenizer_from_texts(texts, num_words=40000)
    tid = "fitted_from_texts"
    return tid

def texts_to_padded_sequences(texts):
    seq = tokenizer.texts_to_sequences(texts)
    X = pad_sequences(seq, maxlen=MAX_LEN, padding="post")
    return X

def predict_lstm(df):
    if "cleaned_final" not in df.columns:
        df["cleaned_final"] = df["komentar"].astype(str).apply(clean_text)

    prep = prepare_tokenizer_if_needed(df)
    X = texts_to_padded_sequences(df["cleaned_final"].astype(str).tolist())

    preds = model.predict(X, verbose=0)
    if preds.ndim == 2 and preds.shape[1] == 1:
        probs = preds.flatten()
    elif preds.ndim == 2 and preds.shape[1] == 2:
        probs = preds[:, 1]
    else:
        probs = preds.flatten()

    df["Prob_Asli"] = probs
    df["Label_Pred"] = np.where(probs >= 0.5, "Asli", "Palsu")
    return df

def hybrid_classify(df):
    df_result = predict_lstm(df)
    
    results = []
    for i, row in df_result.iterrows():
        comment = row["komentar"]
        ml_prob = row["Prob_Asli"]
        
        rule_prob = hybrid_classifier.rule_based_prediction(comment)
        
        final_prob = 0.7 * rule_prob + 0.3 * ml_prob
        final_label = "Asli" if final_prob >= 0.5 else "Palsu"
        
        results.append({
            'komentar': comment,
            'Prob_Asli': final_prob,
            'Label_Pred': final_label,
            'Rule_Prob': rule_prob,
            'ML_Prob': ml_prob
        })
    
    return pd.DataFrame(results)

def evaluate_result(df):
    total_asli = (df["Label_Pred"] == "Asli").sum()
    total_palsu = (df["Label_Pred"] == "Palsu").sum()
    total = len(df)
    
    if total == 0:
        return "<div style='color:#d32f2f;text-align:center;'>Tidak ada data untuk dianalisis</div>", "", 0, 0
    
    weighted_score = (total_asli * 1 + total_palsu * 3) / (total * 3) * 100
    persentase_asli = total_asli / total * 100
    persentase_palsu = total_palsu / total * 100
    
    confidence_scores = []
    for _, row in df.iterrows():
        prob = row['Prob_Asli']
        confidence = 2 * abs(prob - 0.5)
        confidence_scores.append(confidence)
    
    avg_confidence = np.mean(confidence_scores) if confidence_scores else 0
    
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
        <p style='color:#666666; margin:8px 0; font-size:14px;'><b>Total Komentar:</b> {total}</p>
        <p style='color:#666666; margin:8px 0; font-size:14px;'><b>Confidence Rata-rata:</b> {avg_confidence:.3f}</p>
        <p style='color:#666666; margin:8px 0; font-size:14px;'><b>Alasan:</b> {alasan}</p>
        <p style='color:#666666; margin:8px 0; font-size:12px;'>*Komentar palsu diberi bobot 3x lebih berat</p>
    </div>
    """
    
    summary_html = f"<div style='text-align:center; color:#666666; margin-top:10px; font-size:14px;'><b>Summary:</b> {total_asli} Asli | {total_palsu} Palsu | Confidence: {avg_confidence:.3f} | Skor: {weighted_score:.1f}%</div>"
    
    return hasil_html, summary_html, total_asli, total_palsu

def generate_bar_chart(asli, palsu):
    labels = ["Asli", "Palsu"]
    values = [asli, palsu]
    colors = ["#2a9d8f", "#e63946"]

    plt.figure(figsize=(6, 4))
    bars = plt.bar(labels, values, color=colors, alpha=0.8)
    plt.title("Distribusi Prediksi Komentar", fontsize=14, fontweight='bold', color='#333333')
    plt.ylabel("Jumlah Komentar", fontsize=12, color='#333333')
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
        
        status_msg = "Menganalisis komentar dengan sistem hybrid..."
        df_result = hybrid_classify(df_clean)
        
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
    
    gr.HTML("""
    <div style='text-align:center; padding:20px; background:linear-gradient(135deg, #667eea 0%, #764ba2 100%); color:white; border-radius:10px; margin-bottom:20px;'>
        <h1 style='margin:0; font-size:2.5em;'>Sistem Deteksi Keaslian Produk</h1>
        <p style='margin:10px 0 0 0; font-size:1.2em;'>Analisis ulasan produk Tokopedia dengan AI dan Rule-Based</p>
        <p style='margin:5px 0 0 0; font-size:0.9em;'>Komentar palsu diberi bobot 3x lebih berat dalam perhitungan</p>
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
            analyze_btn = gr.Button("Analisis Keaslian Produk", variant="primary", size="lg")
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
    <div style='text-align:center; margin-top:30px; padding:15px; background-color:#f8f9fa; border-radius:8px; color:#666;'>
        <p style='margin:5px;'><b>Fitur Sistem:</b> Scraping Otomatis • Hybrid AI + Rule-Based • Bobot 3x untuk Komentar Palsu • Analisis Real-time</p>
        <p style='margin:5px; font-size:0.9em;'>Sistem akan otomatis mengambil semua komentar yang tersedia tanpa batasan jumlah</p>
    </div>
    """)

if __name__ == "__main__":
    print("Menjalankan Sistem Deteksi Keaslian Produk Tokopedia...")
    print("Fitur: Scraping Selenium • Hybrid classification • Bobot 3x untuk komentar palsu")
    print("Buka browser dan akses localhost yang ditampilkan...")
    
    demo.launch(
        share=False, 
        inbrowser=True,
        server_name="0.0.0.0",
        server_port=7860
    )