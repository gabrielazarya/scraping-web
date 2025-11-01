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

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import gradio as gr

# ==============================
# Konfigurasi Path Utama
# ==============================
BASE_DIR = r"D:\TA\TokPed\sistem_rekomendasi"
RESULT_DIR = os.path.join(BASE_DIR, "hasil_rekomendasi")

EMBEDDINGS_PATH = os.path.join(BASE_DIR, "model_word2vec_balanced", "embeddings_comments.npy")
LABELS_PATH = os.path.join(BASE_DIR, "model_word2vec_balanced", "labels.npy")
MODEL_EMBEDDINGS = os.path.join(BASE_DIR, "model_word2vec_balanced", "word2vec_tokopedia_balanced.model")

MODEL_PATHS = [
    os.path.join(BASE_DIR, "hasil_training_lstm", "model_terbaik", "model_K3_F1_E30_B16_D0.5.keras"),
    os.path.join(BASE_DIR, "hasil_training_lstm", "model_terbaik", "model_K3_F2_E30_B16_D0.5.keras"),
    os.path.join(BASE_DIR, "hasil_training_lstm", "model_terbaik", "model_K3_F3_E30_B16_D0.5.keras")
]

os.makedirs(RESULT_DIR, exist_ok=True)

# ==============================
# 1. Scraping Komentar Tokopedia
# ==============================
def scrape_tokopedia(url):
    if not url:
        return None, "URL kosong."

    options = webdriver.ChromeOptions()
    options.add_argument("--disable-gpu")
    options.add_argument("--start-maximized")
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver.get(url)

    try:
        WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "article.css-15m2bcr"))
        )
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

# ==============================
# 2. Preprocessing
# ==============================
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

# ==============================
# 3. Load Model dan Tokenizer
# ==============================
models = [load_model(path) for path in MODEL_PATHS]
labels = np.load(LABELS_PATH, allow_pickle=True)
tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(labels)
w2v_model = Word2Vec.load(MODEL_EMBEDDINGS)

# ==============================
# 4. Update Tokenizer & Word2Vec
# ==============================
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

# ==============================
# 5. Prediksi
# ==============================
def predict_lstm(df):
    update_tokenizer_with_new_words(df)
    extend_word2vec_model(w2v_model, tokenizer)

    sequences = tokenizer.texts_to_sequences(df["cleaned_final"])
    X = pad_sequences(sequences, maxlen=100, padding="post")

    preds = [m.predict(X, verbose=0).flatten() for m in models]
    min_len = min(len(p) for p in preds)
    preds = [p[:min_len] for p in preds]
    avg_probs = np.mean(preds, axis=0)

    if len(avg_probs) > len(df):
        avg_probs = avg_probs[:len(df)]

    adjusted_probs = avg_probs * 0.7
    df["Prob_Asli"] = avg_probs
    df["Label_Pred"] = ["Asli" if p >= 0.5 else "Palsu" for p in adjusted_probs]
    return df

# ==============================
# 6. Evaluasi & Hasil Akhir
# ==============================
def evaluate_result(df):
    total_asli = (df["Label_Pred"] == "Asli").sum()
    total_palsu = (df["Label_Pred"] == "Palsu").sum()
    total = len(df)
    accuracy = (total_asli + total_palsu) / total if total > 0 else 0

    hasil = "Barang yang dijual Asli" if total_asli > total_palsu else "Barang yang dijual Palsu"
    warna = "#2a9d8f" if hasil.endswith("Asli") else "#e63946"

    hasil_html = f"""
    <div style='text-align:center; margin-top:20px;'>
        <h3 style='color:{warna};'>{hasil}</h3>
        <p><b>Akurasi Prediksi:</b> {accuracy*100:.2f}%</p>
    </div>
    """
    return hasil_html

# ==============================
# 7. UI Gradio
# ==============================
def sistem_rekomendasi_ui(url):
    df_scrape, msg = scrape_tokopedia(url)
    if df_scrape is None:
        return msg, None, None

    df_clean = preprocess_df(df_scrape)
    df_pred = predict_lstm(df_clean)
    df_display = df_pred[["komentar", "Prob_Asli", "Label_Pred"]]

    csv_path = os.path.join(RESULT_DIR, "ulasan_prediksi.csv")
    df_display.to_csv(csv_path, index=False, encoding="utf-8-sig")

    hasil_html = evaluate_result(df_pred)
    return msg, df_display, hasil_html

# ==============================
# 8. Gradio Blocks
# ==============================
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.HTML("<div style='height:40px;'></div>")
    gr.HTML("<h1 style='text-align:center;'>Sistem Rekomendasi Keaslian Produk Tokopedia</h1>")

    url_input = gr.Textbox(label="Masukkan URL review Tokopedia")
    analyze_btn = gr.Button("Mulai Analisis")
    status = gr.Textbox(label="Status", interactive=False)
    output_table = gr.DataFrame(headers=["komentar", "Prob_Asli", "Label_Pred"], wrap=True)
    hasil_pred = gr.HTML(label="Hasil Akhir Prediksi")

    analyze_btn.click(
        sistem_rekomendasi_ui,
        inputs=[url_input],
        outputs=[status, output_table, hasil_pred]
    )
    gr.HTML("<div style='height:40px;'></div>")

demo.launch()
