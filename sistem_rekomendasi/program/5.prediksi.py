import os
import time
import re
import emoji
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
from webdriver_manager.chrome import ChromeDriverManager

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

from gensim.models import Word2Vec
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

import gradio as gr

# Konfigurasi Path Utama
BASE_DIR = r"D:\TA\TokPed\sistem_rekomendasi"
RESULT_DIR = os.path.join(BASE_DIR, "hasil_rekomendasi")

EMBEDDINGS_PATH = os.path.join(BASE_DIR, "model_word2vec_balanced", "embeddings_comments.npy")
LABELS_PATH = os.path.join(BASE_DIR, "model_word2vec_balanced", "labels.npy")
MODEL_EMBEDDINGS = os.path.join(BASE_DIR, "model_word2vec_balanced", "word2vec_tokopedia_balanced.model")

MODEL_PATHS = [
    os.path.join(BASE_DIR, "model_terbaik", "model_K10_F10_E20_B16_D0.3.keras")
]

os.makedirs(RESULT_DIR, exist_ok=True)

# 1. Scraping Komentar Tokopedia
def scrape_tokopedia(url):
    if not url:
        return None, "URL kosong."

    options = webdriver.ChromeOptions()
    options.add_argument("--disable-gpu")
    # options.add_argument("--headless=new")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver.get(url)

    try:
        WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "article.css-15m2bcr"))
        )

        # Tunggu sebentar dan klik di pojok kiri atas
        time.sleep(1)
        from selenium.webdriver import ActionChains
        actions = ActionChains(driver)
        actions.move_by_offset(10, 10).click().perform()

    except:
        driver.quit()
        return None, "Gagal memuat halaman ulasan."

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


# 2. Preprocessing
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
    text = emoji.replace_emoji(text, replace='')
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

# 3. Load Model dan Tokenizer
models = [load_model(path) for path in MODEL_PATHS]
labels = np.load(LABELS_PATH, allow_pickle=True)
tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(labels)
w2v_model = Word2Vec.load(MODEL_EMBEDDINGS)

# 4. Update Tokenizer & Word2Vec (belajar kata baru)
def update_tokenizer_with_new_words(df):
    new_texts = df["cleaned_final"].tolist()
    existing_vocab = set(tokenizer.word_index.keys())

    temp_tokenizer = Tokenizer(oov_token="<OOV>")
    temp_tokenizer.fit_on_texts(new_texts)
    new_vocab = set(temp_tokenizer.word_index.keys())

    added_words = new_vocab - existing_vocab
    for w in added_words:
        tokenizer.word_index[w] = len(tokenizer.word_index) + 1

def extend_word2vec_model(w2v_model, tokenizer):
    vocab = w2v_model.wv.key_to_index
    for word in tokenizer.word_index.keys():
        if word not in vocab:
            w2v_model.wv[word] = np.random.normal(0, 0.01, w2v_model.vector_size)

# 5. Prediksi (gabung 3 model terbaik)
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

def predict_lstm(df):
    # Pastikan kolom teks tersedia
    if "cleaned_final" not in df.columns:
        df["cleaned_final"] = df["komentar"].astype(str).str.lower()

    # Bersihkan teks
    df["cleaned_final"] = (
        df["cleaned_final"]
        .str.replace(r"[^a-zA-Z\s]", "", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    # Tokenisasi (pastikan tokenizer sama dengan training)
    sequences = tokenizer.texts_to_sequences(df["cleaned_final"])
    X = pad_sequences(sequences, maxlen=100, padding="post")

    # Debug: tampilkan panjang data
    print(f"Jumlah komentar: {len(df)}, jumlah sequence: {len(X)}")

    # Prediksi dengan semua model (pastikan jumlah hasil sama)
    preds = [m.predict(X, verbose=0).flatten() for m in models]

    # Cegah duplikasi: ambil hanya sebanyak jumlah baris df
    preds = [p[:len(df)] for p in preds]

    # Ensemble rata-rata berbobot
    weights = np.array([0.5, 0.3, 0.2])
    weights = weights[:len(preds)] / np.sum(weights[:len(preds)])
    avg_probs = np.average(preds, axis=0, weights=weights)

    # Tambahkan bias 30% ke arah palsu
    adjusted_probs = avg_probs * 0.7

    # Tangani mismatch panjang (jika masih terjadi)
    if len(avg_probs) != len(df):
        print(f"Panjang tidak cocok! Menyesuaikan dari {len(avg_probs)} ke {len(df)}")
        min_len = min(len(avg_probs), len(df))
        avg_probs = avg_probs[:min_len]
        adjusted_probs = adjusted_probs[:min_len]
        df = df.head(min_len).copy()

    # Simpan hasil
    df["Prob_Asli"] = avg_probs
    df["Label_Pred"] = np.where(adjusted_probs >= 0.5, "Asli", "Palsu")

    return df


# === 6. Evaluasi & Hasil Akhir ===
def evaluate_result(df):
    total_asli = (df["Label_Pred"] == "Asli").sum()
    total_palsu = (df["Label_Pred"] == "Palsu").sum()
    total = len(df)

    if total == 0:
        persentase_asli = persentase_palsu = 0
    else:
        persentase_asli = (total_asli / total) * 100
        persentase_palsu = (total_palsu / total) * 100

    hasil = "Barang yang dijual Asli" if total_asli > total_palsu else "Barang yang dijual Palsu"
    warna = "#2a9d8f" if hasil.endswith("Asli") else "#e63946"

    # 🖋️ Tulisan prediksi & total sekarang warna hitam
    hasil_html = f"""
    <div style='text-align:center; margin-top:20px;'>
        <h3 style='color:{warna};'>{hasil}</h3>
        <p style='color:black;'><b>Prediksi Asli:</b> {persentase_asli:.2f}%</p>
        <p style='color:black;'><b>Prediksi Palsu:</b> {persentase_palsu:.2f}%</p>
        <p style='color:black;'><i>Total komentar dianalisis:</i> {total}</p>
    </div>
    """

    # Tambahkan total komentar per label (warna tetap hitam)
    total_summary = f"""
    <div style='text-align:center; margin-top:10px; color:black;'>
        <b>Total komentar Asli:</b> {total_asli} &nbsp; | &nbsp;
        <b>Total komentar Palsu:</b> {total_palsu}
    </div>
    """
    return hasil_html, total_summary

# === 6b. Tambahkan Diagram Batang ===
def generate_bar_chart(df):
    total_asli = (df["Label_Pred"] == "Asli").sum()
    total_palsu = (df["Label_Pred"] == "Palsu").sum()

    labels = ["Asli", "Palsu"]
    values = [total_asli, total_palsu]
    colors = ["#2a9d8f", "#e63946"]

    plt.figure(figsize=(4, 3))
    plt.bar(labels, values, color=colors)
    plt.title("Distribusi Prediksi", fontsize=12)
    plt.ylabel("Jumlah Komentar")
    plt.tight_layout()

    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    plt.close()

    return f"<img src='data:image/png;base64,{img_base64}'/>"



# === 7. UI Gradio ===
def sistem_rekomendasi_ui(url):
    df_scrape, msg = scrape_tokopedia(url)
    if df_scrape is None:
        return msg, None, None, None

    df_clean = preprocess_df(df_scrape)
    preprocessed_path = os.path.join(RESULT_DIR, "ulasan_preprocessed.csv")
    df_clean.to_csv(preprocessed_path, index=False, encoding="utf-8-sig")

    df_pred = predict_lstm(df_clean)
    df_display = df_pred[["komentar", "Prob_Asli", "Label_Pred"]]

    csv_path = os.path.join(RESULT_DIR, "ulasan_prediksi.csv")
    df_display.to_csv(csv_path, index=False, encoding="utf-8-sig")

    hasil_html, total_summary = evaluate_result(df_pred)
    chart_html = generate_bar_chart(df_pred)
    return msg, hasil_html, df_display, total_summary, chart_html


# === 8. Gradio Blocks (layout diperbarui) ===
custom_css = """
body {
    background-color: #d6d6d6 !important;  
}
h1 {
    color: black;
}
.gradio-container {
    background-color: #f0f0f0 !important;
}
textarea, input[type="text"], input[type="url"], .gr-textbox, .gr-textbox input {
    background-color: #ffffff !important;  /* Textbox putih */
    color: #000000 !important;
    border: 1px solid #ccc !important;
    border-radius: 8px !important;
}
button {
    border-radius: 10px !important;
    font-weight: 600 !important;
}
p, b, i{
    color: black;
}
"""

with gr.Blocks(theme=gr.themes.Soft(), css=custom_css) as demo:
    gr.HTML("<div style='height:40px;'></div>")
    gr.HTML("<h1 style='text-align:center;'>Sistem Rekomendasi Keaslian Produk Tokopedia</h1>")

    url_input = gr.Textbox(label="Masukkan URL review Tokopedia", placeholder="https://www.tokopedia.com/.../ulasan")
    analyze_btn = gr.Button("Mulai Analisis")
    status = gr.Textbox(label="Status", interactive=False)
    hasil_pred = gr.HTML(label="Hasil Akhir Prediksi")
    output_table = gr.DataFrame(headers=["komentar", "Prob_Asli", "Label_Pred"], wrap=True)
    total_info = gr.HTML(label="Total Komentar")
    chart_output = gr.HTML(label="Visualisasi")

    analyze_btn.click(
        sistem_rekomendasi_ui,
        inputs=[url_input],
        outputs=[status, hasil_pred, output_table, total_info, chart_output]
    )


    gr.HTML("<div style='height:40px;'></div>")

demo.launch()
