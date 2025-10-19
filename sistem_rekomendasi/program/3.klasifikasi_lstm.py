# === Import Library ===
import pandas as pd
import numpy as np
import os
from gensim.models import Word2Vec
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

# ==============================
# 1️⃣ LOAD DATA DAN MODEL
# ==============================
data_path = 'sistem_rekomendasi/hasil_preprocessing/all_data_labeled.xlsx'
w2v_path = 'sistem_rekomendasi/model_word2vec/word2vec_tokopedia.model'

df = pd.read_excel(data_path)
df = df[['cleaned', 'label']].dropna()

# Pastikan label berupa angka (1=Asli, 0=Palsu)
df['label'] = df['label'].map({'asli': 1, 'palsu': 0})
sentences = [str(text).split() for text in df['cleaned']]

# Load model Word2Vec
model_w2v = Word2Vec.load(w2v_path)
embedding_dim = model_w2v.vector_size

print(f"Data & Word2Vec berhasil dimuat — total {len(sentences)} data")

# ==============================
# 2️⃣ TOKENIZER & PADDING
# ==============================
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['cleaned'])
sequences = tokenizer.texts_to_sequences(df['cleaned'])
word_index = tokenizer.word_index

max_length = max(len(seq) for seq in sequences)
X = pad_sequences(sequences, maxlen=max_length, padding='post')
y = np.array(df['label'])

print(f"Jumlah kata unik: {len(word_index)}")
print(f"Panjang urutan maksimum: {max_length}")

# ==============================
# 3️⃣ BUAT EMBEDDING MATRIX DARI WORD2VEC
# ==============================
embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
for word, i in word_index.items():
    if word in model_w2v.wv:
        embedding_matrix[i] = model_w2v.wv[word]

# ==============================
# 4️⃣ FUNGSI MEMBANGUN MODEL LSTM
# ==============================
def build_lstm_model(input_length, embedding_matrix):
    model = Sequential([
        Embedding(
            input_dim=embedding_matrix.shape[0],
            output_dim=embedding_matrix.shape[1],
            weights=[embedding_matrix],
            input_length=input_length,
            trainable=False
        ),
        LSTM(128, dropout=0.3, recurrent_dropout=0.3),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# ==============================
# 5️⃣ UJI DENGAN K-FOLD VARIASI (3, 5, 10)
# ==============================
k_values = [3, 5, 10]
results_summary = {}

for k in k_values:
    print(f"\n==============================")
    print(f"Pengujian dengan K-Fold = {k}")
    print(f"==============================")

    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    acc_list, prec_list, rec_list, f1_list = [], [], [], []

    fold = 1
    for train_idx, test_idx in kf.split(X):
        print(f"\nFold ke-{fold}/{k}")
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = build_lstm_model(max_length, embedding_matrix)
        model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

        y_pred = (model.predict(X_test) > 0.5).astype("int32")

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        acc_list.append(acc)
        prec_list.append(prec)
        rec_list.append(rec)
        f1_list.append(f1)

        print(f"Fold {fold} — Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}")
        fold += 1

    # Simpan hasil rata-rata
    results_summary[k] = {
        'Akurasi': np.mean(acc_list),
        'Presisi': np.mean(prec_list),
        'Recall': np.mean(rec_list),
        'F1-Score': np.mean(f1_list)
    }

# ==============================
# 6️⃣ CETAK HASIL RATA-RATA SEMUA K
# ==============================
print("\nHASIL K-FOLD")
for k, metrics in results_summary.items():
    print(f"\nK = {k}")
    print(f"  Akurasi : {metrics['Akurasi']:.4f}")
    print(f"  Presisi : {metrics['Presisi']:.4f}")
    print(f"  Recall  : {metrics['Recall']:.4f}")
    print(f"  F1-Score: {metrics['F1-Score']:.4f}")

# ==============================
# 7️⃣ SIMPAN MODEL TERAKHIR & RANGKUMAN HASIL
# ==============================
output_dir = 'sistem_rekomendasi/model_lstm'
os.makedirs(output_dir, exist_ok=True)

model.save(os.path.join(output_dir, 'lstm_tokopedia_final.h5'))

result_path = os.path.join(output_dir, 'hasil_kfold_variatif.csv')
pd.DataFrame(results_summary).T.to_csv(result_path)

print(f"\nModel LSTM terakhir berhasil disimpan di: {output_dir}/lstm_tokopedia_final.h5")
print(f"Hasil pengujian (3, 5, 10 Fold) disimpan ke: {result_path}")
