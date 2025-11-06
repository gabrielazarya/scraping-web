import os, time, json, itertools, signal, warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle, class_weight
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import Callback

warnings.filterwarnings("ignore")

# ====================== Paths ======================
DATA_PATH = 'sistem_rekomendasi/hasil_preprocessing/all_data_labeled.xlsx'
EMBEDDINGS_PATH = 'sistem_rekomendasi/model_word2vec_balanced/embeddings_comments.npy'
LABELS_PATH = 'sistem_rekomendasi/model_word2vec_balanced/labels.npy'
OUTPUT_DIR = 'sistem_rekomendasi/yang_kurang_hasil_training_lstm'
os.makedirs(OUTPUT_DIR, exist_ok=True)

CSV_LOG_PATH = os.path.join(OUTPUT_DIR, 'hasil_training_resume.csv')
CSV_REPORT_PATH = os.path.join(OUTPUT_DIR, 'hasil_classification_report.csv')
CSV_CONFUSION_PATH = os.path.join(OUTPUT_DIR, 'hasil_confusion_matrix.csv')
TEXT_LOG_PATH = os.path.join(OUTPUT_DIR, 'yang_kurang_training_detail_log.txt')

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

# ====================== Custom Logging Callback ======================
class EpochLoggingCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        with open(TEXT_LOG_PATH, "a", encoding="utf-8") as f:
            text = (
                f"Epoch {epoch+1} - "
                f"loss: {logs.get('loss', 0):.4f} - "
                f"accuracy: {logs.get('accuracy', 0):.4f} - "
                f"val_loss: {logs.get('val_loss', 0):.4f} - "
                f"val_accuracy: {logs.get('val_accuracy', 0):.4f}\n"
            )
            f.write(text)

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
k_values = [10]
epochs_values = [20]
batch_values = [64]
dropout_values = [0.3]

param_grid = [
    {'k': k, 'epoch': e, 'batch': b, 'dropout': d}
    for k, e, b, d in itertools.product(k_values, epochs_values, batch_values, dropout_values)
]

print(f"[INFO] Total kombinasi parameter: {len(param_grid)}")

# ====================== CSV Init ======================
def init_csv(path, cols):
    if os.path.exists(path):
        return pd.read_csv(path)
    else:
        return pd.DataFrame(columns=cols)

df_log = init_csv(CSV_LOG_PATH, [
    'K', 'Fold', 'Epoch', 'Batch', 'Dropout',
    'Train Accuracy', 'Val Accuracy', 'Test Accuracy',
    'Train Loss', 'Val Loss',
    'Precision', 'Recall', 'Durasi (detik)'
])
df_report = init_csv(CSV_REPORT_PATH, [
    'K', 'Fold', 'Epoch', 'Batch', 'Dropout', 'Label', 'Precision', 'Recall', 'F1-Score', 'Support'
])
df_conf = init_csv(CSV_CONFUSION_PATH, [
    'K', 'Fold', 'Epoch', 'Batch', 'Dropout', 'Confusion_Flat'
])

# ====================== Log File Init ======================
with open(TEXT_LOG_PATH, "a", encoding="utf-8") as f:
    f.write("\n" + "="*80 + "\n")
    f.write("LOG TRAINING DETAIL\n")
    f.write("="*80 + "\n")

# ====================== Resume Support ======================
def fold_done_before(k, fold, epoch, batch, dropout):
    existing = df_log[
        (df_log['K'] == k) &
        (df_log['Fold'] == fold) &
        (df_log['Epoch'] == epoch) &
        (df_log['Batch'] == batch) &
        (df_log['Dropout'] == dropout)
    ]
    return not existing.empty

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
    epoch_total = params['epoch']
    batch = params['batch']
    dropout = params['dropout']

    header_text = f"\nKombinasi ke {combo_counter}/{len(param_grid)} | K={k}, Epoch={epoch_total}, Batch={batch}, Dropout={dropout}\n"
    print(header_text)
    with open(TEXT_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(header_text)

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    fold_idx = 0

    for train_idx, val_idx in skf.split(X, np.argmax(y, axis=1)):
        if stop_training:
            break
        fold_idx += 1

        fold_text = f"\n{'='*30} Fold {fold_idx}/{k} {'='*30}\n"
        print(fold_text)
        with open(TEXT_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(fold_text)

        if fold_done_before(k, fold_idx, epoch_total, batch, dropout):
            print(f"[INFO] Fold {fold_idx} sudah pernah diselesaikan. Melewati fold ini.")
            continue

        X_train_full, X_val = X[train_idx], X[val_idx]
        y_train_full, y_val = y[train_idx], y[val_idx]

        X_train, X_test, y_train, y_test = train_test_split(
            X_train_full, y_train_full, test_size=0.1, random_state=42, stratify=y_train_full
        )

        X_train_lstm = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_val_lstm = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
        X_test_lstm = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

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

        class_weights_dict = dict(enumerate(
            class_weight.compute_class_weight('balanced', classes=np.arange(n_classes), y=np.argmax(y_train, axis=1))
        ))

        model = build_model(X_train_lstm.shape[1], dropout=dropout)
        callbacks = [GracefulStopCallback(), EpochLoggingCallback()]

        start_time = time.time()
        hist = model.fit(
            X_train_lstm, y_train,
            validation_data=(X_val_lstm, y_val),
            epochs=epoch_total,
            batch_size=batch,
            class_weight=class_weights_dict,
            verbose=1,
            callbacks=callbacks
        )
        duration = time.time() - start_time

        y_pred_val = np.argmax(model.predict(X_val_lstm, verbose=0), axis=1)
        y_true_val = np.argmax(y_val, axis=1)
        report = classification_report(y_true_val, y_pred_val, target_names=le.classes_, output_dict=True, zero_division=0)
        cm = confusion_matrix(y_true_val, y_pred_val)

        # Evaluate test set
        test_loss, test_acc = model.evaluate(X_test_lstm, y_test, verbose=0)
        test_text = f"[TEST] Fold {fold_idx}/{k} - Test Accuracy: {test_acc:.4f} - Test Loss: {test_loss:.4f}\n"
        print(test_text)
        with open(TEXT_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(test_text + "\n")

        for label, vals in report.items():
            if label in le.classes_:
                df_report.loc[len(df_report)] = [
                    k, fold_idx, epoch_total, batch, dropout,
                    label, vals['precision'], vals['recall'], vals['f1-score'], vals['support']
                ]

        df_conf.loc[len(df_conf)] = [
            k, fold_idx, epoch_total, batch, dropout, json.dumps(cm.tolist())
        ]

        model_path = os.path.join(OUTPUT_DIR, f'model_K{k}_F{fold_idx}_E{epoch_total}_B{batch}_D{dropout}.keras')
        model.save(model_path)

        df_log.loc[len(df_log)] = [
            k, fold_idx, epoch_total, batch, dropout,
            hist.history['accuracy'][-1],
            hist.history['val_accuracy'][-1],
            test_acc,
            hist.history['loss'][-1],
            hist.history['val_loss'][-1],
            report['weighted avg']['precision'],
            report['weighted avg']['recall'],
            duration
        ]

        df_log.to_csv(CSV_LOG_PATH, index=False)
        df_report.to_csv(CSV_REPORT_PATH, index=False)
        df_conf.to_csv(CSV_CONFUSION_PATH, index=False)

print("\n[INFO] Training selesai sepenuhnya.")
with open(TEXT_LOG_PATH, "a", encoding="utf-8") as f:
    f.write("\n[INFO] Training selesai sepenuhnya.\n")
