import gradio as gr
import pandas as pd
import numpy as np
import re
import requests
import emoji
import os
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from time import sleep

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from gensim.models import Word2Vec
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# ============================================================== #
# 1️⃣ Utility - Clean & Preprocess Text
# ============================================================== #
def clean_text(teks):
    teks = emoji.replace_emoji(str(teks), replace='')
    teks = re.sub(r'#|@', '', teks)
    teks = re.sub(r'[^\w\s]', ' ', teks)
    teks = re.sub(r'\s+', ' ', teks).strip()
    return teks.lower()

stem_factory = StemmerFactory()
stemmer = stem_factory.create_stemmer()

stop_factory = StopWordRemoverFactory()
stopwords = stop_factory.get_stop_words()

def preprocess_text(df):
    df["cleaned"] = df["komentar"].fillna("").apply(clean_text)
    df["cleaned"] = df["cleaned"].apply(
        lambda x: ' '.join([stemmer.stem(w) for w in x.split() if w not in stopwords])
    )
    return df

# ============================================================== #
# 2️⃣ Scraping Komentar dari URL Tokopedia (lebih fleksibel)
# ============================================================== #
def extract_product_id(url):
    # Cari productID di URL (paling belakang)
    match = re.search(r'/([a-zA-Z0-9_-]+)$', url.split("?")[0])
    return match.group(1) if match else None

def scrape_tokopedia_reviews(url, max_pages=5):
    product_id = extract_product_id(url)
    if not product_id:
        return None, "❌ URL tidak valid atau tidak bisa mengekstrak product ID."

    headers = {
        "accept": "*/*",
        "accept-language": "en-US,en;q=0.9",
        "content-type": "application/json",
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
    }

    session = requests.Session()
    all_reviews = []

    for page in range(1, max_pages+1):
        query = {
            "operationName": "ReviewListShopProduct",
            "variables": {
                "page": page,
                "perPage": 20,
                "productID": product_id,
                "sort": 1,
                "filter": {"withContent": True}
            },
            "query": """query ReviewListShopProduct($page: Int!, $perPage: Int!, $productID: String!, $sort: Int!, $filter: ReviewFilterInput) {
                reviewListShopProduct(page: $page, perPage: $perPage, productID: $productID, sort: $sort, filter: $filter) {
                    data { content }
                }
            }"""
        }

        try:
            res = session.post(
                "https://gql.tokopedia.com/graphql/ReviewListShopProduct",
                headers=headers, json=query, timeout=30
            )
            if res.status_code != 200:
                break
            data = res.json()
            reviews = data.get("data", {}).get("reviewListShopProduct", {}).get("data", [])
            if not reviews:
                break
            for r in reviews:
                content = r.get("content", "").strip()
                if content:
                    all_reviews.append(content)
            sleep(1)
        except Exception as e:
            print(f"⚠️ Error halaman {page}: {e}")
            break

    if not all_reviews:
        return None, "⚠️ Tidak ada komentar ditemukan."
    
    df = pd.DataFrame(all_reviews, columns=["komentar"])
    return df, f"✅ Berhasil mengambil {len(df)} komentar."

# ============================================================== #
# 3️⃣ Load Model LSTM + Tokenizer
# ============================================================== #
model_path = "model_lstm/lstm_tokopedia_final.h5"
w2v_path = "model_word2vec/word2vec_tokopedia.model"
train_data_path = "hasil_preprocessing/all_data_labeled.xlsx"

model_lstm = load_model(model_path)
model_w2v = Word2Vec.load(w2v_path)
train_data = pd.read_excel(train_data_path)

tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(train_data["cleaned"])

# ============================================================== #
# 4️⃣ Prediksi Komentar dengan probabilitas
# ============================================================== #
def predict_sentiment(df):
    sequences = tokenizer.texts_to_sequences(df["cleaned"])
    max_length = 100
    X = pad_sequences(sequences, maxlen=max_length, padding="post")
    probs = model_lstm.predict(X, verbose=0).flatten()
    df["Prob_Asli"] = probs
    df["Label"] = ["Asli" if p >= 0.5 else "Palsu" for p in probs]
    return df

# ============================================================== #
# 5️⃣ Generate Grafik & Ringkasan
# ============================================================== #
def generate_chart(df):
    asli = (df["Label"] == "Asli").sum()
    palsu = (df["Label"] == "Palsu").sum()

    fig, ax = plt.subplots(figsize=(5, 3))
    bars = ax.bar(["Palsu", "Asli"], [palsu, asli], color=["#e63946", "#2a9d8f"])
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                int(bar.get_height()), ha='center', fontsize=10)
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"<img src='data:image/png;base64,{img_b64}' width='100%'/>", asli, palsu

# ============================================================== #
# 6️⃣ Fungsi Utama Analisis
# ============================================================== #
def analisis_produk(url):
    df_raw, msg = scrape_tokopedia_reviews(url)
    if df_raw is None:
        return msg, None, None, None

    df_clean = preprocess_text(df_raw)
    df_pred = predict_sentiment(df_clean)
    chart, asli, palsu = generate_chart(df_pred)
    total = len(df_pred)
    verdict = "Asli" if asli >= palsu else "Palsu"

    summary_html = f"""
    <div style='text-align:center;'>
        <p>Dari <b>{total}</b> komentar:</p>
        <p style='color:#e63946;'>Palsu: {palsu}</p>
        <p style='color:#2a9d8f;'>Asli: {asli}</p>
        <hr style='width:60%; margin:12px auto;' />
        <h2 style='font-size:32px; color:{'#2a9d8f' if verdict=='Asli' else '#e63946'};'>Barang {verdict}</h2>
    </div>
    """
    return msg, df_pred, chart, summary_html

# ============================================================== #
# 7️⃣ UI Gradio
# ============================================================== #
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.HTML("""
    <div style='text-align:center; margin-top:20px;'>
        <h1 style='font-size:36px; font-weight:800;'>🔍 Sistem Analisis Keaslian Produk Tokopedia</h1>
        <p style='color:#6b7280;'>Masukkan URL produk Tokopedia untuk memeriksa keaslian melalui analisis komentar.</p>
    </div>
    """)

    url_input = gr.Textbox(label="Masukkan URL Review Tokopedia")
    analyze_btn = gr.Button("Mulai Analisis", variant="primary")
    status = gr.Textbox(label="Status", interactive=False)
    output_table = gr.DataFrame(headers=["Komentar", "Label", "Prob_Asli"], label="Hasil Prediksi", wrap=True)
    output_chart = gr.HTML(label="Grafik Hasil")
    output_summary = gr.HTML(label="Kesimpulan")
    download_btn = gr.File(label="Unduh Hasil CSV")

    def analisis_produk_ui(url):
        status_msg, df_pred, chart, summary = analisis_produk(url)
        file_path = None
        if df_pred is not None:
            file_path = "/tmp/hasil_prediksi.csv"
            df_pred.to_csv(file_path, index=False)
        return status_msg, df_pred.head(50) if df_pred is not None else None, chart, summary, file_path

    analyze_btn.click(
        analisis_produk_ui,
        inputs=url_input,
        outputs=[status, output_table, output_chart, output_summary, download_btn],
    )

demo.launch()
