import os
import signal
import time
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score  # 🔹 Tambahan
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

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

label_encoder = LabelEncoder()
y = to_categorical(label_encoder.fit_transform(labels))

tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
X = tokenizer.texts_to_sequences(texts)
max_len = max(len(x) for x in X)
X = pad_sequences(X, maxlen=max_len, padding='post')

# ======================================================
# 2️⃣ LOAD WORD2VEC
# ======================================================
w2v = Word2Vec.load(WORD2VEC_MODEL)
embedding_dim = w2v.vector_size

embedding_matrix = np.zeros((len(tokenizer.word_index) + 1, embedding_dim))
for word, i in tokenizer.word_index.items():
    if word in w2v.wv:
        embedding_matrix[i] = w2v.wv[word]

# ======================================================
# 3️⃣ GRID PARAMETER & K
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
# 4️⃣ FUNGSI PEMBUATAN MODEL
# ======================================================
def build_model(dropout):
    model = Sequential([
        Embedding(input_dim=len(tokenizer.word_index) + 1,
                  output_dim=embedding_dim,
                  weights=[embedding_matrix],
                  input_length=max_len,
                  trainable=False),
        LSTM(128, dropout=dropout, recurrent_dropout=0.3),  # 🔹 Tambahan: recurrent_dropout agar lebih stabil
        Dropout(dropout),
        Dense(y.shape[1], activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# ======================================================
# 5️⃣ RESUME HANDLER
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
            'Train Accuracy', 'Val Accuracy', 'Train Loss', 'Val Loss',
            'Balanced Accuracy',  # 🔹 Tambahan
            'Durasi (detik)', 'Precision_Asli', 'Recall_Asli', 'F1_Asli',
            'Precision_Palsu', 'Recall_Palsu', 'F1_Palsu'
        ])
        start_index = 0
else:
    df_log = pd.DataFrame(columns=[
        'K', 'Fold', 'Epoch', 'Batch', 'Dropout',
        'Train Accuracy', 'Val Accuracy', 'Train Loss', 'Val Loss',
        'Balanced Accuracy',  # 🔹 Tambahan
        'Durasi (detik)', 'Precision_Asli', 'Recall_Asli', 'F1_Asli',
        'Precision_Palsu', 'Recall_Palsu', 'F1_Palsu'
    ])
    start_index = 0

# ======================================================
# 6️⃣ PROSES TRAINING DENGAN MULTI-KFOLD
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

    for train_idx, val_idx in kf.split(X):
        fold_idx += 1
        if stop_training:
            print("[INFO] Training dihentikan saat dalam fold.")
            break

        print(f"\nMemulai Fold {fold_idx}/{k} untuk kombinasi ke-{idx}")
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # 🔹 Class Weight
        y_train_labels = np.argmax(y_train, axis=1)
        classes = np.unique(y_train_labels)
        weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train_labels)
        class_weight = dict(zip(classes, weights))
        print(f"Class weight untuk fold ini: {class_weight}")

        model = build_model(dropout)

        # Tambahan callbacks
        model_path_best = os.path.join(OUTPUT_DIR, f'best_K{k}_F{fold_idx}_E{epoch}_B{batch}_D{dropout}.keras')
        callbacks = [
            GracefulStopCallback(),
            ModelCheckpoint(model_path_best, monitor='val_loss', save_best_only=True, verbose=1),
            EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)
        ]

        start_time = time.time()
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epoch,
            batch_size=batch,
            verbose=1,
            class_weight=class_weight,  # 🔹 Sudah benar ditempatkan di sini
            callbacks=callbacks
        )
        duration = time.time() - start_time

        # 🔹 Evaluasi tambahan: Balanced Accuracy
        y_pred = model.predict(X_val)
        y_pred_labels = np.argmax(y_pred, axis=1)
        y_true_labels = np.argmax(y_val, axis=1)
        report = classification_report(y_true_labels, y_pred_labels, target_names=label_encoder.classes_, output_dict=True)
        bal_acc = balanced_accuracy_score(y_true_labels, y_pred_labels)

        train_acc = history.history['accuracy'][-1]
        val_acc = history.history['val_accuracy'][-1]
        train_loss = history.history['loss'][-1]
        val_loss = history.history['val_loss'][-1]

        model_final_path = os.path.join(OUTPUT_DIR, f'model_K{k}_F{fold_idx}_E{epoch}_B{batch}_D{dropout}.keras')
        model.save(model_final_path)
        print(f"Model disimpan: {model_final_path} (format .keras)")

        df_log.loc[len(df_log)] = [
            k, fold_idx, epoch, batch, dropout,
            round(train_acc, 4), round(val_acc, 4),
            round(train_loss, 4), round(val_loss, 4),
            round(bal_acc, 4),  # 🔹 Balanced Accuracy
            round(duration, 2),
            round(report['asli']['precision'], 4), round(report['asli']['recall'], 4), round(report['asli']['f1-score'], 4),
            round(report['palsu']['precision'], 4), round(report['palsu']['recall'], 4), round(report['palsu']['f1-score'], 4)
        ]
        df_log.to_csv(CSV_LOG_PATH, index=False)
        print("Hasil disimpan ke CSV (real-time).")

        cm = confusion_matrix(y_true_labels, y_pred_labels)
        print("Confusion Matrix:\n", cm)

    if stop_training:
        print("[INFO] Training dihentikan sepenuhnya oleh pengguna.")
        break

print("\n[INFO] Semua kombinasi selesai dijalankan.")
