import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import ast

# 1️ Baca file CSV
df = pd.read_csv("sistem_rekomendasi/model_terbaik/gabungan_lengkap.csv")

# 2️ Konversi string confusion_flat menjadi array numpy
def parse_confusion(conf_str):
    try:
        # Konversi string seperti "[[1951, 483], [85, 973]]" menjadi array numpy
        return np.array(ast.literal_eval(conf_str))
    except:
        return np.array([[0, 0], [0, 0]])

df['confusion_matrix'] = df['Confusion_Flat'].apply(parse_confusion)

# 3️ Kelompokkan berdasarkan Epoch dan rata-ratakan confusion matrix
epoch_confusion_mean = df.groupby('Epoch')['confusion_matrix'].apply(
    lambda x: np.mean(np.stack(x), axis=0)  # Rata-rata semua matrix dalam grup epoch
).reset_index()

# 4️ Visualisasi confusion matrix rata-rata untuk setiap epoch
for _, row in epoch_confusion_mean.iterrows():
    epoch = int(row["Epoch"])
    cm = row["confusion_matrix"]
    
    # Konversi ke integer untuk annotation
    cm_int = cm.astype(int)
    
    # Hitung akurasi dari confusion matrix
    accuracy = (cm_int[0,0] + cm_int[1,1]) / cm_int.sum()
    
    # 5️⃣ Visualisasi confusion matrix
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm_int, annot=True, fmt="d", cmap="Blues", cbar=True,
                xticklabels=["Pred: Asli", "Pred: Palsu"],
                yticklabels=["Aktual: Asli", "Aktual: Palsu"])
    plt.title(f"Rata-rata Confusion Matrix (Epoch {epoch})\nAkurasi: {accuracy:.3f}")
    plt.tight_layout()
    plt.show()

# Tampilkan juga nilai numeriknya
print("\nRata-rata Confusion Matrix per Epoch:")
for _, row in epoch_confusion_mean.iterrows():
    epoch = int(row["Epoch"])
    cm = row["confusion_matrix"].astype(int)
    accuracy = (cm[0,0] + cm[1,1]) / cm.sum()
    print(f"\nEpoch {epoch}:")
    print(f"[[{cm[0,0]:>4}, {cm[0,1]:>4}]")
    print(f" [{cm[1,0]:>4}, {cm[1,1]:>4}]]")
    print(f"Akurasi: {accuracy:.3f}")