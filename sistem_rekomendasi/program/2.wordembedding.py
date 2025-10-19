# === Import Library ===
import pandas as pd
from gensim.models import Word2Vec
import os

# ==============================
# 1️⃣ BACA DATA LABELED
# ==============================
input_path = 'sistem_rekomendasi/hasil_preprocessing/all_data_labeled.xlsx'
df = pd.read_excel(input_path)

# Pastikan kolom 'cleaned' dan 'label' ada
df = df[['cleaned', 'label']].dropna().reset_index(drop=True)

# Ubah teks jadi token list
sentences = [str(text).split() for text in df['cleaned'] if isinstance(text, str)]

print(f"Jumlah komentar untuk Word2Vec: {len(sentences)}")

# ==============================
# 2️⃣ LATIH WORD2VEC MODEL
# ==============================
model = Word2Vec(
    sentences=sentences,
    vector_size=100,   # dimensi vektor kata
    window=5,          # konteks kata
    min_count=2,       # abaikan kata yang muncul <2 kali
    sg=1,              # 1 = Skip-gram (lebih akurat), 0 = CBOW (lebih cepat)
    epochs=20,
    workers=4
)

# ==============================
# 3️⃣ SIMPAN MODEL
# ==============================
output_dir = 'sistem_rekomendasi/model_word2vec'
os.makedirs(output_dir, exist_ok=True)

model_path = os.path.join(output_dir, 'word2vec_tokopedia.model')
model.save(model_path)

print(f"✅ Model Word2Vec berhasil disimpan di: {model_path}")

# ==============================
# 4️⃣ UJI MODEL (opsional)
# ==============================
try:
    print("\nKata yang paling mirip dengan 'sepatu':")
    print(model.wv.most_similar('sepatu', topn=5))
except KeyError:
    print("\nKata 'sepatu' tidak ditemukan di vocab (mungkin terlalu jarang).")
