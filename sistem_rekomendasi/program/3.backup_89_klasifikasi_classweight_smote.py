import os, time, json, itertools, signal, warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle, class_weight
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import Callback, ReduceLROnPlateau

warnings.filterwarnings("ignore")

# ====================== Paths ======================
DATA_PATH = 'sistem_rekomendasi/hasil_preprocessing/all_data_labeled.xlsx'
EMBEDDINGS_PATH = 'sistem_rekomendasi/model_word2vec_balanced/embeddings_comments.npy'
LABELS_PATH = 'sistem_rekomendasi/model_word2vec_balanced/labels.npy'
OUTPUT_DIR = 'sistem_rekomendasi/hasil_training_lstm'
os.makedirs(OUTPUT_DIR, exist_ok=True)

CSV_LOG_PATH = os.path.join(OUTPUT_DIR, 'hasil_training_resume.csv')
CSV_REPORT_PATH = os.path.join(OUTPUT_DIR, 'hasil_classification_report.csv')
CSV_CONFUSION_PATH = os.path.join(OUTPUT_DIR, 'hasil_confusion_matrix.csv')

BALANCE_METHOD = 'oversample_per_fold'
stop_training = False

# ====================== Graceful Stop ======================
def signal_handler(sig, frame):
    global stop_training
    stop_training = True
    print("\n[INFO] Perintah berhenti diterima. Training akan dihentikan setelah batch ini...")

signal.signal(signal.SIGINT, signal_handler)

class GracefulStopCallback(Callback):
    def on_batch_end(self, batch, logs=None):
        if stop_training:
            print("\n[INFO] Training dihentikan oleh pengguna.")
            self.model.stop_training = True

# ====================== Load Embeddings ======================
X = np.load(EMBEDDINGS_PATH)
labels = np.load(LABELS_PATH, allow_pickle=True)

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(labels)
n_classes = len(le.classes_)
y = to_categorical(y_encoded)

print(f"[INFO] Data shape: {X.shape}, Classes: {list(le.classes_)}")

# ====================== Parameter Grid ======================
k_values = [3, 5, 10]
epochs_values = [10, 20, 30]
batch_values = [16, 32, 64]
dropout_values = [0.2, 0.3, 0.5]

param_grid = [
    {'k': k, 'epoch': e, 'batch': b, 'dropout': d}
    for k, e, b, d in itertools.product(k_values, epochs_values, batch_values, dropout_values)
]

print(f"[INFO] Total kombinasi parameter: {len(param_grid)}")  # 81

# ====================== CSV Init ======================
def init_csv(path, cols):
    if os.path.exists(path):
        return pd.read_csv(path)
    else:
        return pd.DataFrame(columns=cols)

df_log = init_csv(CSV_LOG_PATH, [
    'K', 'Fold', 'Epoch', 'Batch', 'Dropout',
    'Train Accuracy', 'Val Accuracy', 'Train Loss', 'Val Loss',
    'Precision', 'Recall', 'Durasi (detik)'
])
df_report = init_csv(CSV_REPORT_PATH, [
    'K', 'Fold', 'Epoch', 'Batch', 'Dropout', 'Label', 'Precision', 'Recall', 'F1-Score', 'Support'
])
df_conf = init_csv(CSV_CONFUSION_PATH, [
    'K', 'Fold', 'Epoch', 'Batch', 'Dropout', 'Confusion_Flat'
])

# ====================== Model Builder ======================
def build_model(input_dim, dropout=0.3):
    model = Sequential([
        LSTM(128, input_shape=(input_dim, 1)),
        Dropout(dropout),
        Dense(n_classes, activation='softmax')
    ])
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# ====================== TRAINING LOOP ======================
combo_counter = 0
for params in param_grid:
    if stop_training:
        break
    combo_counter += 1
    k = params['k']
    epoch = params['epoch']
    batch = params['batch']
    dropout = params['dropout']

    print(f"\nKombinasi ke {combo_counter}/{len(param_grid)} | K={k}, Epoch={epoch}, Batch={batch}, Dropout={dropout}")

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    fold_idx = 0

    for train_idx, val_idx in skf.split(X, np.argmax(y, axis=1)):
        if stop_training:
            break
        fold_idx += 1
        print(f"\n{'='*30} Fold {fold_idx}/{k} {'='*30}\n")  # Pemisah fold

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Reshape untuk LSTM: (samples, timesteps=1, features)
        X_train_lstm = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_val_lstm = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))

        # ====================== Oversample per fold ======================
        if BALANCE_METHOD == 'oversample_per_fold':
            counts = np.bincount(np.argmax(y_train, axis=1))
            max_count = counts.max()
            X_list, y_list = [], []
            for cls in range(n_classes):
                idx_cls = np.where(np.argmax(y_train, axis=1) == cls)[0]
                n_needed = max_count - len(idx_cls)
                if n_needed > 0:
                    idx_upsample = np.random.choice(idx_cls, n_needed, replace=True)
                    idx_total = np.concatenate([idx_cls, idx_upsample])
                else:
                    idx_total = idx_cls
                X_list.append(X_train_lstm[idx_total])
                y_list.append(y_train[idx_total])
            X_train_lstm = np.vstack(X_list)
            y_train = np.vstack(y_list)
            shuffle_idx = np.arange(len(X_train_lstm))
            np.random.shuffle(shuffle_idx)
            X_train_lstm = X_train_lstm[shuffle_idx]
            y_train = y_train[shuffle_idx]

        # Class weights
        class_weights_dict = dict(enumerate(
            class_weight.compute_class_weight('balanced', classes=np.arange(n_classes), y=np.argmax(y_train, axis=1))
        ))

        # Build model
        model = build_model(X_train_lstm.shape[1], dropout=dropout)
        callbacks = [GracefulStopCallback()]  # EarlyStopping dihapus agar semua epoch selesai

        # Fit model
        start_time = time.time()
        hist = model.fit(
            X_train_lstm, y_train,
            validation_data=(X_val_lstm, y_val),
            epochs=epoch,
            batch_size=batch,
            class_weight=class_weights_dict,
            verbose=1,  # hijau progress bar asli
            callbacks=callbacks
        )
        duration = time.time() - start_time

        # Predict & evaluate
        y_pred = np.argmax(model.predict(X_val_lstm, verbose=0), axis=1)
        y_true = np.argmax(y_val, axis=1)
        report = classification_report(y_true, y_pred, target_names=le.classes_, output_dict=True, zero_division=0)
        cm = confusion_matrix(y_true, y_pred)

        # Save per-class report
        for label, vals in report.items():
            if label in le.classes_:
                df_report.loc[len(df_report)] = [
                    k, fold_idx, epoch, batch, dropout,
                    label, vals['precision'], vals['recall'], vals['f1-score'], vals['support']
                ]

        # Save confusion matrix
        df_conf.loc[len(df_conf)] = [
            k, fold_idx, epoch, batch, dropout, json.dumps(cm.tolist())
        ]

        # Save model
        model_path = os.path.join(OUTPUT_DIR, f'model_K{k}_F{fold_idx}_E{epoch}_B{batch}_D{dropout}.keras')
        model.save(model_path)

        # Save log
        df_log.loc[len(df_log)] = [
            k, fold_idx, epoch, batch, dropout,
            round(hist.history['accuracy'][-1], 4),
            round(hist.history['val_accuracy'][-1], 4),
            round(hist.history['loss'][-1], 4),
            round(hist.history['val_loss'][-1], 4),
            report['weighted avg']['precision'],
            report['weighted avg']['recall'],
            round(duration, 2)
        ]

        # Update CSV setiap fold
        df_log.to_csv(CSV_LOG_PATH, index=False)
        df_report.to_csv(CSV_REPORT_PATH, index=False)
        df_conf.to_csv(CSV_CONFUSION_PATH, index=False)

print("\n[INFO] Training selesai sepenuhnya.")
