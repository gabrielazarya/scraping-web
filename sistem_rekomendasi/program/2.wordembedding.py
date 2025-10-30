# === Import Library ===
import pandas as pd
from gensim.models import Word2Vec
from sklearn.utils import shuffle
import numpy as np
import os

# ==============================
# 1️⃣ BACA DATA LABELED
# ==============================
input_path = 'sistem_rekomendasi/hasil_preprocessing/all_data_labeled.xlsx'
df = pd.read_excel(input_path)

# Gunakan kolom 'cleaned' dan 'label' saja
df = df[['cleaned', 'label']].dropna().reset_index(drop=True)

# Hilangkan duplikat agar tidak bias
df = df.drop_duplicates(subset=['cleaned'])
df = shuffle(df, random_state=42).reset_index(drop=True)

# Fungsi tokenisasi
def text_to_tokens(text):
    if isinstance(text, str):
        return text.split()
    elif isinstance(text, list):
        return text
    else:
        return []

# Tokenisasi
sentences = [text_to_tokens(t) for t in df['cleaned']]
print(f"Jumlah komentar untuk Word2Vec: {len(sentences)}")

# ==============================
# 2️⃣ LATIH MODEL WORD2VEC
# ==============================
vector_size = 150

model = Word2Vec(
    sentences=sentences,
    vector_size=vector_size,
    window=7,
    min_count=3,
    sg=1,
    epochs=35,
    negative=10,
    workers=4,
    sample=1e-4,
    seed=42
)

# Simpan model
output_dir = 'sistem_rekomendasi/model_word2vec_balanced'
os.makedirs(output_dir, exist_ok=True)
model_path = os.path.join(output_dir, 'word2vec_tokopedia_balanced.model')
model.save(model_path)
print(f"Model Word2Vec disimpan di: {model_path}")

# ==============================
# 3️⃣ BANGUN EMBEDDINGS KOMENTAR
# ==============================
def get_comment_vector(tokens, model, vector_size):
    """Hitung rata-rata embedding kata dalam satu komentar"""
    vectors = []
    for t in tokens:
        if t in model.wv.key_to_index:
            vectors.append(model.wv[t])
    if len(vectors) == 0:
        # Jika tidak ada kata di vocab, kembalikan vektor nol
        return np.zeros(vector_size)
    return np.mean(vectors, axis=0)

# Hasil embeddings per komentar
embeddings = np.array([get_comment_vector(t, model, vector_size) for t in sentences])
print(f"Shape embeddings: {embeddings.shape}")  # (jumlah_komentar, vector_size)

# ==============================
# 4️⃣ SIMPAN EMBEDDINGS & LABEL
# ==============================
embeddings_path = os.path.join(output_dir, 'embeddings_comments.npy')
labels_path = os.path.join(output_dir, 'labels.npy')

np.save(embeddings_path, embeddings)
np.save(labels_path, df['label'].to_numpy())

print(f"Embeddings disimpan di: {embeddings_path}")
print(f"Label disimpan di: {labels_path}")
