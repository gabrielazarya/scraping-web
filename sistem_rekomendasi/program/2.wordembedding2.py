# ============================================================
# IMPROVED WORD2VEC TRAINING DARI DATA EXCEL ASLI
# ============================================================

import pandas as pd
from gensim.models import Word2Vec
from sklearn.utils import resample
import os, random

# 1. Load dataset
input_path = 'sistem_rekomendasi/hasil_preprocessing/all_data_labeled.xlsx'
df = pd.read_excel(input_path)
df = df[['cleaned', 'label']].dropna()

print("Distribusi label sebelum balancing:")
print(df['label'].value_counts())

# 2. Balancing data (jika perlu)
df_0 = df[df['label'] == 0]
df_1 = df[df['label'] == 1]

if len(df_0) > 0 and len(df_1) > 0:
    min_samples = min(len(df_0), len(df_1))
    df_0_balanced = resample(df_0, n_samples=min_samples, random_state=42)
    df_1_balanced = resample(df_1, n_samples=min_samples, random_state=42)
    df_balanced = pd.concat([df_0_balanced, df_1_balanced])
else:
    df_balanced = df

print("\nDistribusi label setelah balancing:")
print(df_balanced['label'].value_counts())

# 3. Tokenisasi teks
def text_to_tokens(text):
    if isinstance(text, str):
        return text.split()
    return []

sentences = [text_to_tokens(t) for t in df_balanced['cleaned']]
sentences = [s for s in sentences if len(s) > 0]

# Acak urutan kalimat agar pelatihan lebih stabil
random.seed(42)
random.shuffle(sentences)

print(f"\nJumlah kalimat untuk training: {len(sentences)}")

# 4. Train model Word2Vec dengan pengaturan optimal
vector_size = 200  # lebih besar agar representasi lebih kaya
model = Word2Vec(
    sentences=sentences,
    vector_size=vector_size,
    window=7,
    min_count=1,        # semua kata ikut dilatih
    sg=1,               # skip-gram, cocok untuk dataset kecil
    epochs=50,          # tambah epochs agar lebih matang
    negative=10,
    workers=4,
    sample=1e-3,
    seed=42
)

# 5. Simpan model dan vocabulary
output_dir = 'model_word2vec_custom'
os.makedirs(output_dir, exist_ok=True)
model_path = os.path.join(output_dir, 'word2vec_tokopedia_custom.model')
txt_path = os.path.join(output_dir, 'word_vectors.txt')

model.save(model_path)
model.wv.save_word2vec_format(txt_path, binary=False)

print(f"\nModel Word2Vec disimpan di: {model_path}")
print(f"Vocabulary size: {len(model.wv.key_to_index)}")

# 6. Uji kata
test_words = ['asli', 'palsu', 'bagus', 'jelek', 'ori', 'kw']
print("\nUji kemiripan kata:")
for word in test_words:
    if word in model.wv.key_to_index:
        similar = model.wv.most_similar(word, topn=5)
        print(f"Kata mirip '{word}': {similar}")
    else:
        print(f"Kata '{word}' tidak ada di vocabulary")
