import requests
import pandas as pd
import re
import os
import numpy as np
import emoji
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from gensim.models import Word2Vec
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def extract_product_id(url):
    match = re.search(r'tokopedia\.com/.+?/([a-zA-Z0-9\-]+)(?:/review|$)', url)
    return match.group(1) if match else None

url = input("Masukkan URL ulasan Tokopedia (yang /review): ")
product_id = extract_product_id(url)
if not product_id:
    print("Gagal mendeteksi ID produk dari URL.")
    exit()

all_reviews = []
for page in range(1, 11):
    res = requests.get(f"https://www.tokopedia.com/review-api/product/v2/review/{product_id}?page={page}&sort=0&limit=100")
    if res.status_code != 200:
        break
    data = res.json()
    reviews = data.get("data", {}).get("list", [])
    if not reviews:
        break
    for r in reviews:
        text = r.get("content", "").strip()
        if text:
            all_reviews.append(text)

if not all_reviews:
    print("Tidak ada komentar ditemukan.")
    exit()

df = pd.DataFrame(all_reviews, columns=["komentar"])
os.makedirs("sistem_rekomendasi/hasil", exist_ok=True)
raw_path = "sistem_rekomendasi/hasil/ulasan_api.csv"
df.to_csv(raw_path, index=False, encoding="utf-8-sig")

def clean_text(teks):
    teks = emoji.replace_emoji(teks, replace='')
    teks = re.sub(r'#(\w+)', r'\1', teks)
    teks = re.sub(r'@(\w+)', r'\1', teks)
    teks = re.sub(r'[^\w\s]', ' ', teks)
    teks = re.sub(r'\s+', ' ', teks).strip()
    return teks

df['cleaned'] = df['komentar'].astype(str).apply(clean_text).str.lower()
df = df[df['cleaned'].str.strip().astype(bool)].reset_index(drop=True)

stem_factory = StemmerFactory()
stemmer = stem_factory.create_stemmer()
df['stemmed'] = df['cleaned'].apply(lambda x: ' '.join([stemmer.stem(t) for t in x.split()]))

w2v_model = Word2Vec.load('sistem_rekomendasi/model_word2vec/word2vec_tokopedia.model')
lstm_model = load_model('sistem_rekomendasi/model_lstm/lstm_tokopedia_final.h5')

tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['stemmed'])
sequences = tokenizer.texts_to_sequences(df['stemmed'])
maxlen = 100
X = pad_sequences(sequences, maxlen=maxlen, padding='post')

pred = (lstm_model.predict(X) > 0.5).astype("int32").flatten()
df['prediksi'] = ['asli' if p == 1 else 'palsu' for p in pred]

output_path = "sistem_rekomendasi/hasil/hasil_prediksi.csv"
df.to_csv(output_path, index=False, encoding="utf-8-sig")

print(f"\nTotal ulasan: {len(df)}")
print("Distribusi hasil prediksi:")
print(df['prediksi'].value_counts())
print(f"\nFile hasil disimpan di: {output_path}")