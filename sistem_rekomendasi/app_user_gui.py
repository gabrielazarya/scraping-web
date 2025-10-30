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
# 1. Scraping Komentar Tokopedia
# ==============================
def scrape_tokopedia(url):
    if not url:
        return None, "URL kosong."

    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--start-maximized")
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver.get(url)

    try:
        WebDriverWait(driver, 10).until(
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

    os.makedirs("hasil_rekomendasi", exist_ok=True)
    csv_path = os.path.join("hasil_rekomendasi", "ulasan.csv")
    df = pd.DataFrame(data, columns=["komentar"])
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    return df, f"Berhasil mengambil {len(df)} komentar. Disimpan di {csv_path}"

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
model_path = "model_lstm/lstm_tokopedia_final.keras"
w2v_path = "model_word2vec/word2vec_tokopedia.model"
train_data_path = "hasil_preprocessing/all_data_labeled.xlsx"

model_lstm = load_model(model_path)
model_w2v = Word2Vec.load(w2v_path)
train_data = pd.read_excel(train_data_path)
tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(train_data["cleaned_final"])

# ==============================
# 4. Prediksi LSTM dengan bobot lebih ke "Palsu"
# ==============================
def predict_lstm(df):
    sequences = tokenizer.texts_to_sequences(df["cleaned_final"])
    X = pad_sequences(sequences, maxlen=100, padding="post")
    probs = model_lstm.predict(X, verbose=0).flatten()
    
    # Terapkan bobot 30% lebih untuk "Palsu"
    adjusted_probs = probs * 0.7  # Asli dikurangi bobot
    df["Prob_Asli"] = probs
    df["Label_Pred"] = ["Asli" if p >= 0.5 else "Palsu" for p in adjusted_probs]
    return df

# ==============================
# 5. Evaluasi jika ada label asli
# ==============================
def evaluate(df):
    if "label" in df.columns:
        y_true = df["label"].map({"Asli":1, "Palsu":0}).values
        y_pred = df["Label_Pred"].map({"Asli":1, "Palsu":0}).values
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        return acc, prec, rec, f1
    else:
        return None, None, None, None

# ==============================
# 6. Visualisasi Chart
# ==============================
def generate_chart(df):
    asli = (df["Label_Pred"]=="Asli").sum()
    palsu = (df["Label_Pred"]=="Palsu").sum()
    fig, ax = plt.subplots(figsize=(5,3))
    bars = ax.bar(["Palsu","Asli"], [palsu,asli], color=["#e63946","#2a9d8f"])
    for bar in bars:
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1,
                int(bar.get_height()), ha='center', fontsize=10)
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"<img src='data:image/png;base64,{img_b64}' width='100%'/>"

# ==============================
# 7. Fungsi Utama UI
# ==============================
def sistem_rekomendasi_ui(url, label_file=None):
    df_scrape, msg = scrape_tokopedia(url)
    if df_scrape is None:
        return msg, None, None, None, None, None

    df_clean = preprocess_df(df_scrape)
    df_pred = predict_lstm(df_clean)

    # Jika user upload label asli
    if label_file:
        df_label = pd.read_csv(label_file)
        if "label" in df_label.columns:
            df_pred["label"] = df_label["label"]

    os.makedirs("hasil_rekomendasi", exist_ok=True)
    csv_path = os.path.join("hasil_rekomendasi","ulasan_prediksi.csv")
    df_pred.to_csv(csv_path, index=False, encoding="utf-8-sig")

    acc, prec, rec, f1 = evaluate(df_pred)
    chart = generate_chart(df_pred)

    summary_text = f"<div style='text-align:center;'><p>{msg}</p>"
    if acc is not None:
        summary_text += f"<p>Akurasi: {acc:.2f}, Presisi: {prec:.2f}, Recall: {rec:.2f}, F1-Score: {f1:.2f}</p>"
    summary_text += "</div>"

    return msg, df_pred.head(50), chart, summary_text, acc, csv_path

# ==============================
# 8. Gradio GUI
# ==============================
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.HTML("<h1 style='text-align:center'>Sistem Rekomendasi Keaslian Produk Tokopedia</h1>")
    url_input = gr.Textbox(label="Masukkan URL review Tokopedia")
    label_input = gr.File(label="Upload CSV Label (Opsional)")
    analyze_btn = gr.Button("Mulai Analisis")
    status = gr.Textbox(label="Status", interactive=False)
    output_table = gr.DataFrame(headers=["komentar","cleaned_final","Label_Pred","Prob_Asli"], wrap=True)
    output_chart = gr.HTML(label="Chart Hasil Prediksi")
    output_summary = gr.HTML(label="Ringkasan Evaluasi")
    download_btn = gr.File(label="Unduh CSV")

    analyze_btn.click(
        sistem_rekomendasi_ui,
        inputs=[url_input, label_input],
        outputs=[status, output_table, output_chart, output_summary, gr.Textbox(), download_btn]
    )

demo.launch()
