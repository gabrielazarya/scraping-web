# 5_fixed_prediksi.py
import os
import time
import re
import emoji
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

# ----------------------------
# Konfigurasi Path Utama
# ----------------------------
BASE_DIR = r"D:\TA\TokPed\sistem_rekomendasi"
RESULT_DIR = os.path.join(BASE_DIR, "hasil_rekomendasi")
MODEL_DIR = os.path.join(BASE_DIR, "model_terbaik")

EMBEDDINGS_PATH = os.path.join(BASE_DIR, "model_word2vec_balanced", "word2vec_tokopedia_balanced.model")
LABELS_PATH = os.path.join(BASE_DIR, "model_word2vec_balanced", "labels.npy")
MODEL_PATH = os.path.join(MODEL_DIR, "model_K10_F10_E20_B16_D0.3.keras")

TOKENIZER_PATH = os.path.join(MODEL_DIR, "tokenizer.pkl")   # kalau punya, simpan di sini
MAXLEN_PATH = os.path.join(MODEL_DIR, "maxlen.pkl")         # kalau pernah simpan maxlen

os.makedirs(RESULT_DIR, exist_ok=True)

# ----------------------------
# 1. Scraping Komentar Tokopedia
# ----------------------------
def scrape_tokopedia(url):
    if not url:
        return None, "URL kosong."

    options = webdriver.ChromeOptions()
    options.add_argument("--disable-gpu")
    # options.add_argument("--headless=new")  # aktifkan kalau ingin headless
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
        # klik semua "Selengkapnya"
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

# ----------------------------
# 2. Preprocessing
# ----------------------------
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

# ----------------------------
# 3. Load Model & (optional) Word2Vec
# ----------------------------
# Muat model keras
model = load_model(MODEL_PATH)

# Muat Word2Vec jika ada (opsional)
w2v_model = None
if os.path.exists(EMBEDDINGS_PATH):
    try:
        w2v_model = Word2Vec.load(EMBEDDINGS_PATH)
    except Exception:
        w2v_model = None

# ----------------------------
# 4. Tokenizer handling (BEST-EFFORT)
# ----------------------------
# Prioritas:
# 1) load tokenizer.pkl kalau ada
# 2) coba reconstruct dari embedding layer & w2v (cocokkan vektor)
# 3) fallback: fit tokenizer di teks (dan simpan)

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
    """
    Upaya: ambil matriks embedding dari model (jika layer pertama adalah Embedding)
    dan bandingkan baris embedding dengan vektor w2v untuk menemukan kata yang paling mirip.
    Kembalikan tokenizer sederhana yang mapping word->index.
    Nota: proses ini mahal (O(V * vocab_w2v)) dan tidak sempurna.
    """
    try:
        # ambil layer embedding pertama
        for layer in model.layers:
            if layer.__class__.__name__.lower() == "embedding":
                emb_weights = layer.get_weights()[0]  # shape (vocab_model, dim)
                break
        else:
            return None, "no_embedding_layer"

        if w2v_model is None:
            return None, "no_w2v"

        w2v_vocab = list(w2v_model.wv.index_to_key)
        w2v_mat = np.array([w2v_model.wv[w] for w in w2v_vocab])

        # normalize
        emb_norm = emb_weights / (np.linalg.norm(emb_weights, axis=1, keepdims=True) + 1e-9)
        w2v_norm = w2v_mat / (np.linalg.norm(w2v_mat, axis=1, keepdims=True) + 1e-9)

        # untuk performa, gunakan k-NN via dot product
        mapping = {}
        # iterate through rows of emb_norm and cari kata w2v paling mirip
        for i in range(min(len(emb_norm), len(emb_weights))):
            sims = np.dot(w2v_norm, emb_norm[i])
            best_idx = np.argmax(sims)
            mapping[w2v_vocab[best_idx]] = i  # map word -> index_in_embedding
            # optional: you could set threshold to ensure quality

        # buat tokenizer-like object: word_index where indices are +1 (keras style)
        tokenizer = Tokenizer(oov_token="<OOV>")
        # Build word_index based on mapping, but Keras expects indices start at 1
        ordered = sorted(mapping.items(), key=lambda x: mapping[x[0]])
        tokenizer.word_index = {word: idx+1 for idx, (word, _) in enumerate(ordered)}
        return tokenizer, "reconstructed_from_w2v"
    except Exception as e:
        return None, f"failed_reconstruct:{e}"

def create_tokenizer_from_texts(texts, num_words=None):
    tokenizer = Tokenizer(oov_token="<OOV>", num_words=num_words)
    tokenizer.fit_on_texts(texts)
    # simpan untuk ke depannya
    try:
        with open(TOKENIZER_PATH, "wb") as f:
            pickle.dump(tokenizer, f)
    except Exception:
        pass
    return tokenizer

# muat maxlen kalau ada
if os.path.exists(MAXLEN_PATH):
    try:
        with open(MAXLEN_PATH, "rb") as f:
            MAX_LEN = pickle.load(f)
    except Exception:
        MAX_LEN = 100
else:
    MAX_LEN = 100

# coba load tokenizer dulu
tokenizer, tid = try_load_tokenizer()

# ----------------------------
# 5. Predict pipeline
# ----------------------------
def prepare_tokenizer_if_needed(df_clean):
    global tokenizer, tid
    if tokenizer is not None:
        return "already_have"

    # 1) try reconstruct via embedding & w2v
    tok, reason = try_reconstruct_tokenizer_from_embedding_and_w2v(model, w2v_model)
    if tok is not None:
        tokenizer = tok
        tid = reason
        # simpan
        try:
            with open(TOKENIZER_PATH, "wb") as f:
                pickle.dump(tokenizer, f)
        except:
            pass
        return reason

    # 2) fallback: fit tokenizer on current texts (save for future)
    texts = df_clean["cleaned_final"].astype(str).tolist()
    tokenizer = create_tokenizer_from_texts(texts, num_words=40000)
    tid = "fitted_from_texts"
    return tid

def texts_to_padded_sequences(texts):
    seq = tokenizer.texts_to_sequences(texts)
    X = pad_sequences(seq, maxlen=MAX_LEN, padding="post")
    return X

def predict_lstm(df):
    # pastikan preprocessing tersedia
    if "cleaned_final" not in df.columns:
        df["cleaned_final"] = df["komentar"].astype(str).apply(clean_text)

    # siapkan tokenizer jika belum ada
    prep = prepare_tokenizer_if_needed(df)

    # tokenisasi & pad
    X = texts_to_padded_sequences(df["cleaned_final"].astype(str).tolist())

    # prediksi
    preds = model.predict(X, verbose=0)
    # jika model keluaran probabilitas single dim, flatten
    if preds.ndim == 2 and preds.shape[1] == 1:
        probs = preds.flatten()
    elif preds.ndim == 2 and preds.shape[1] == 2:
        # ambil prob kelas 'asli' pada indeks 1 asumsi kelas [palsu, asli]
        probs = preds[:, 1]
    else:
        # fallback: jika multi-dim, ambil kolom 1 jika ada
        probs = preds.flatten()

    df["Prob_Asli"] = probs
    df["Label_Pred"] = np.where(probs >= 0.5, "Asli", "Palsu")
    return df

# ----------------------------
# 6. Evaluasi & Visual
# ----------------------------
def evaluate_result(df):
    total_asli = int((df["Label_Pred"] == "Asli").sum())
    total_palsu = int((df["Label_Pred"] == "Palsu").sum())
    total = int(len(df))
    if total == 0:
        persentase_asli = persentase_palsu = 0.0
    else:
        persentase_asli = total_asli / total * 100
        persentase_palsu = total_palsu / total * 100

    hasil = "Barang yang dijual Asli" if total_asli > total_palsu else "Barang yang dijual Palsu"
    warna = "#2a9d8f" if hasil.endswith("Asli") else "#e63946"

    hasil_html = f"""
    <div style='text-align:center; margin-top:10px;'>
        <h3 style='color:{warna};'>{hasil}</h3>
        <p style='color:black;'><b>Prediksi Asli:</b> {persentase_asli:.2f}%</p>
        <p style='color:black;'><b>Prediksi Palsu:</b> {persentase_palsu:.2f}%</p>
        <p style='color:black;'><i>Total komentar dianalisis:</i> {total}</p>
    </div>
    """
    summary_html = f"<div style='text-align:center; color:black; margin-top:8px;'>Total Asli: {total_asli} &nbsp;|&nbsp; Total Palsu: {total_palsu}</div>"
    return hasil_html, summary_html, total_asli, total_palsu

def generate_bar_chart(asli, palsu):
    labels = ["Asli", "Palsu"]
    values = [asli, palsu]
    colors = ["#2a9d8f", "#e63946"]

    plt.figure(figsize=(4, 3))
    plt.bar(labels, values, color=colors)
    plt.title("Distribusi Prediksi", fontsize=11)
    plt.ylabel("Jumlah Komentar")
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close()
    buf.seek(0)
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"<img src='data:image/png;base64,{encoded}' width='420'/>"

# ----------------------------
# 7. UI Gradio
# ----------------------------
def sistem_rekomendasi_ui(url):
    # status awal
    status_msg = "Memulai..."
    try:
        df_scrape, msg = scrape_tokopedia(url)
        if df_scrape is None:
            return msg, "<div style='text-align:center;color:red;'>Gagal: tidak ada komentar</div>", None, "", ""
        status_msg = "Selesai scraping. Preprocessing..."
        df_clean = preprocess_df(df_scrape)
        # simpan preprocessed
        preprocessed_path = os.path.join(RESULT_DIR, "ulasan_preprocessed.csv")
        df_clean.to_csv(preprocessed_path, index=False, encoding="utf-8-sig")

        status_msg = "Menyiapkan tokenizer dan melakukan prediksi..."
        df_pred = predict_lstm(df_clean)

        # simpan hasil prediksi ringkas
        df_display = df_pred[["komentar", "Prob_Asli", "Label_Pred"]]
        csv_path = os.path.join(RESULT_DIR, "ulasan_prediksi.csv")
        df_display.to_csv(csv_path, index=False, encoding="utf-8-sig")

        # evaluasi & chart
        hasil_html, summary_html, total_asli, total_palsu = evaluate_result(df_pred)
        chart_html = generate_bar_chart(total_asli, total_palsu)

        status_msg = f"Selesai. Tokenizer mode: {tid if 'tid' in globals() else 'unknown'}"
        # return: status, hasil_html (text), tabel (dataframe), summary, chart
        return status_msg, hasil_html, df_display, summary_html, chart_html

    except Exception as e:
        # kembalikan error ke UI
        return f"Error: {e}", "<div style='color:red;text-align:center;'>Terjadi kesalahan saat proses. Cek log terminal.</div>", None, "", ""

# Gradio UI (tetap sesuai style kamu; CSS minimal)
custom_css = """
body { background-color: #d6d6d6 !important; }
.gradio-container { background-color: #f0f0f0 !important; }
textarea, input[type="text"], input[type="url"], .gr-textbox, .gr-textbox input {
    background-color: #ffffff !important;
    color: #000000 !important;
    border: 1px solid #ccc !important;
    border-radius: 8px !important;
}
p, b, i { color: black; }
"""

with gr.Blocks(theme=gr.themes.Soft(), css=custom_css) as demo:
    gr.HTML("<div style='height:20px;'></div>")
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

    gr.HTML("<div style='height:30px;'></div>")

# Luncurkan app (tanpa reload arg supaya kompatibel dengan versi Gradio lama)
if __name__ == "__main__":
    demo.launch()
