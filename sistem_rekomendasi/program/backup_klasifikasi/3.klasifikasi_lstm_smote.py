import os
import signal
import time
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import Callback

# ======================================================
# KONFIGURASI DASAR
# ======================================================
DATA_PATH = 'sistem_rekomendasi/hasil_preprocessing/all_data_labeled.xlsx'
WORD2VEC_MODEL = 'sistem_rekomendasi/model_word2vec_balanced/word2vec_tokopedia_balanced.model'
OUTPUT_DIR = 'sistem_rekomendasi/hasil_training_lstm'
os.makedirs(OUTPUT_DIR, exist_ok=True)
CSV_LOG_PATH = os.path.join(OUTPUT_DIR, 'hasil_training_resume.csv')

# ======================================================
# HANDLER UNTUK GRACEFUL STOP
# ======================================================
stop_training = False

def signal_handler(sig, frame):
    global stop_training
    stop_training = True
    print("\n[INFO] Perintah berhenti diterima. Akan menghentikan setelah batch ini...")

signal.signal(signal.SIGINT, signal_handler)

class GracefulStopCallback(Callback):
    def on_batch_end(self, batch, logs=None):
        if stop_training:
            print("\n[INFO] Training dihentikan dengan aman di tengah batch.")
            self.model.stop_training = True

# ======================================================
# 1️⃣ PERSIAPAN DATA
# ======================================================
df = pd.read_excel(DATA_PATH)[['cleaned', 'label']].dropna()
df = df.drop_duplicates(subset=['cleaned'])
df = shuffle(df, random_state=42).reset_index(drop=True)

texts = df['cleaned'].astype(str).tolist()
labels = df['label'].tolist()

# Label encoding
label_encoder = LabelEncoder()
y_raw = label_encoder.fit_transform(labels)
class_names = label_encoder.classes_
print("[INFO] Kelas terdeteksi:", class_names)

# Tokenisasi
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
X_seq = tokenizer.texts_to_sequences(texts)
max_len = max(len(x) for x in X_seq)
X = pad_sequences(X_seq, maxlen=max_len, padding='post')

# ======================================================
# 2️⃣ BALANCING DATA DENGAN SMOTE
# ======================================================
print("[INFO] Sebelum SMOTE (count per class):", np.bincount(y_raw))
smote = SMOTE(random_state=42, k_neighbors=3)
X_flat = X.reshape(X.shape[0], -1)
X_res_flat, y_res = smote.fit_resample(X_flat, y_raw)
X_res = X_res_flat.reshape(X_res_flat.shape[0], max_len)
print("[INFO] Setelah SMOTE (count per class):", np.bincount(y_res))

# One-hot encode labels untuk Keras
y = to_categorical(y_res)

# ======================================================
# 3️⃣ LOAD WORD2VEC
# ======================================================
w2v = Word2Vec.load(WORD2VEC_MODEL)
embedding_dim = w2v.vector_size

embedding_matrix = np.zeros((len(tokenizer.word_index) + 1, embedding_dim))
for word, i in tokenizer.word_index.items():
    if word in w2v.wv:
        embedding_matrix[i] = w2v.wv[word]

# ======================================================
# 4️⃣ GRID PARAMETER & K
# ======================================================
k_values = [3, 5, 10]
epoch_list = [10, 20, 30]
batch_list = [16, 32, 64]
dropout_list = [0.2, 0.3, 0.5]

param_grid = []
for K in k_values:
    for e in epoch_list:
        for b in batch_list:
            for d in dropout_list:
                param_grid.append({'k': K, 'epoch': e, 'batch': b, 'dropout': d})

# ======================================================
# 5️⃣ FUNGSI PEMBUATAN MODEL
# ======================================================
def build_model(dropout):
    model = Sequential([
        Embedding(input_dim=len(tokenizer.word_index) + 1,
                  output_dim=embedding_dim,
                  weights=[embedding_matrix],
                  input_length=max_len,
                  trainable=True),
        LSTM(128),
        Dropout(dropout),
        Dense(y.shape[1], activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# ======================================================
# 6️⃣ RESUME HANDLER
# ======================================================
if os.path.exists(CSV_LOG_PATH):
    resume = input("Lanjut dari resume terakhir? (y/n): ").strip().lower()
    if resume == 'y':
        df_log = pd.read_csv(CSV_LOG_PATH)
        start_index = len(df_log)
        print(f"Melanjutkan dari kombinasi ke-{start_index}...")
    else:
        df_log = pd.DataFrame(columns=[
            'K', 'Fold', 'Epoch', 'Batch', 'Dropout',
            'Train Accuracy', 'Val Accuracy', 'Train Loss', 'Val Loss', 'Durasi (detik)'
        ])
        start_index = 0
else:
    df_log = pd.DataFrame(columns=[
        'K', 'Fold', 'Epoch', 'Batch', 'Dropout',
        'Train Accuracy', 'Val Accuracy', 'Train Loss', 'Val Loss', 'Durasi (detik)'
    ])
    start_index = 0

# ======================================================
# 7️⃣ PROSES TRAINING DENGAN MULTI-KFOLD + CONFUSION MATRIX
# ======================================================
for idx, params in enumerate(param_grid, start=1):
    if stop_training:
        print("[INFO] Training dihentikan oleh pengguna.")
        break

    if idx <= start_index:
        continue

    k = params['k']
    epoch = params['epoch']
    batch = params['batch']
    dropout = params['dropout']

    print(f"\n{'='*70}")
    print(f"Mulai kombinasi ke-{idx}/{len(param_grid)} | K={k} | Epoch={epoch} | Batch={batch} | Dropout={dropout}")
    print(f"{'='*70}")

    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    fold_idx = 0

    for train_idx, val_idx in kf.split(X_res):
        fold_idx += 1
        if stop_training:
            print("[INFO] Training dihentikan saat dalam fold.")
            break

        print(f"\nMemulai Fold {fold_idx}/{k} untuk kombinasi ke-{idx}")
        X_train, X_val = X_res[train_idx], X_res[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = build_model(dropout)

        start_time = time.time()
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epoch,
            batch_size=batch,
            verbose=1,
            callbacks=[GracefulStopCallback()]
        )
        duration = time.time() - start_time

        # Ambil hasil akhir
        train_acc = history.history['accuracy'][-1]
        val_acc = history.history['val_accuracy'][-1]
        train_loss = history.history['loss'][-1]
        val_loss = history.history['val_loss'][-1]

        # ✅ Tambahan evaluasi: Confusion Matrix dan Classification Report
        y_pred_probs = model.predict(X_val)
        y_pred = np.argmax(y_pred_probs, axis=1)
        y_true = np.argmax(y_val, axis=1)

        cm = confusion_matrix(y_true, y_pred)
        report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)

        print("\n=== CONFUSION MATRIX ===")
        print(cm)
        print("\n=== CLASSIFICATION REPORT ===")
        print(pd.DataFrame(report).transpose())

        # Simpan juga confusion matrix ke CSV per kombinasi
        cm_path = os.path.join(OUTPUT_DIR, f'confusion_K{k}_F{fold_idx}_E{epoch}_B{batch}_D{dropout}.csv')
        pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(cm_path)
        print(f"Confusion matrix disimpan: {cm_path}")

        # Simpan model
        model_path = os.path.join(OUTPUT_DIR, f'model_K{k}_F{fold_idx}_E{epoch}_B{batch}_D{dropout}.keras')
        model.save(model_path)
        print(f"Model disimpan: {model_path}")

        # Simpan hasil ke CSV real-time
        df_log.loc[len(df_log)] = [
            k, fold_idx, epoch, batch, dropout,
            round(train_acc, 4), round(val_acc, 4),
            round(train_loss, 4), round(val_loss, 4),
            round(duration, 2)
        ]
        df_log.to_csv(CSV_LOG_PATH, index=False)
        print("Hasil disimpan ke CSV (real-time).")

    if stop_training:
        print("[INFO] Training dihentikan sepenuhnya oleh pengguna.")
        break

print("\n[INFO] Semua kombinasi selesai dijalankan.")
