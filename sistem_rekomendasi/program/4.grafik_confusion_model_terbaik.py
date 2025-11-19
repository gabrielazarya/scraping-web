import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import ast

# 1️⃣ Baca file CSV
df = pd.read_csv("sistem_rekomendasi/model_terbaik/model_terbaik.csv")

# 2️⃣ Konversi string confusion_flat menjadi array numpy
def parse_confusion(conf_str):
    try:
        # Konversi string seperti "[[1951, 483], [85, 973]]" menjadi array numpy
        return np.array(ast.literal_eval(conf_str))
    except:
        return np.array([[0, 0], [0, 0]])

df['confusion_matrix'] = df['Confusion_Flat'].apply(parse_confusion)

# 3️⃣ Cari model terbaik berdasarkan Test Accuracy
best_model = df.loc[df['Test Accuracy'].idxmax()]

print("=" * 60)
print("MODEL TERBAIK")
print("=" * 60)
print(f"K-Fold           : {int(best_model['K'])}")
print(f"Epoch            : {int(best_model['Epoch'])}")
print(f"Batch Size       : {int(best_model['Batch'])}")
print(f"Dropout Rate     : {best_model['Dropout']}")
print(f"Fold             : {int(best_model['Fold'])}")
print(f"Test Accuracy    : {best_model['Test Accuracy']:.4f}")
print(f"Validation Accuracy: {best_model['Val Accuracy']:.4f}")
print(f"Train Accuracy   : {best_model['Train Accuracy']:.4f}")
print(f"F1 Score Macro   : {best_model['F1_Macro']:.4f}")
print(f"Precision        : {best_model['Precision']:.4f}")
print(f"Recall           : {best_model['Recall']:.4f}")

# 4️⃣ Ambil confusion matrix model terbaik
cm = best_model['confusion_matrix'].astype(int)

# Hitung metrik dari confusion matrix
accuracy = (cm[0,0] + cm[1,1]) / cm.sum()
precision = cm[1,1] / (cm[1,1] + cm[0,1])  # TP / (TP + FP)
recall = cm[1,1] / (cm[1,1] + cm[1,0])     # TP / (TP + FN)
f1_score = 2 * (precision * recall) / (precision + recall)

print(f"\nConfusion Matrix:")
print(f"[[{cm[0,0]:>4} (TN), {cm[0,1]:>4} (FP)]")
print(f" [{cm[1,0]:>4} (FN), {cm[1,1]:>4} (TP)]]")

print(f"\nDetailed Metrics from Confusion Matrix:")
print(f"Akurasi    : {accuracy:.4f}")
print(f"Precision  : {precision:.4f}")
print(f"Recall     : {recall:.4f}")
print(f"F1-Score   : {f1_score:.4f}")

# 5️⃣ Visualisasi confusion matrix model terbaik
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=True,
            xticklabels=["Pred: Asli", "Pred: Palsu"],
            yticklabels=["Aktual: Asli", "Aktual: Palsu"],
            annot_kws={"size": 12})

plt.title(f"Confusion Matrix - Model Terbaik\n"
          f"K={int(best_model['K'])}, Epoch={int(best_model['Epoch'])}, "
          f"Batch={int(best_model['Batch'])}, Dropout={best_model['Dropout']}\n"
          f"Test Accuracy: {best_model['Test Accuracy']:.4f}", 
          fontsize=12, pad=20)
plt.tight_layout()
plt.show()

# 6️⃣ Optional: Tampilkan top 3 model terbaik
print("\n" + "=" * 60)
print("TOP 3 MODEL TERBAIK")
print("=" * 60)

top_3_models = df.nlargest(3, 'Test Accuracy')[['K', 'Fold', 'Epoch', 'Batch', 'Dropout', 'Test Accuracy', 'F1_Macro']]

for i, (idx, model) in enumerate(top_3_models.iterrows(), 1):
    print(f"\n#{i} - Test Accuracy: {model['Test Accuracy']:.4f}")
    print(f"   K: {int(model['K'])}, Fold: {int(model['Fold'])}, Epoch: {int(model['Epoch'])}, "
          f"Batch: {int(model['Batch'])}, Dropout: {model['Dropout']}, "
          f"F1 Macro: {model['F1_Macro']:.4f}")