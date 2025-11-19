import pandas as pd
import matplotlib.pyplot as plt

# 1️ Baca file CSV
df = pd.read_csv("sistem_rekomendasi\model_terbaik\gabungan_lengkap.csv")

# 2️ Kelompokkan berdasarkan jumlah epoch
epoch_groups = df.groupby("Epoch").mean(numeric_only=True)

# 3️ Plot hubungan Epoch dengan akurasi
plt.figure(figsize=(10,6))
plt.plot(epoch_groups.index, epoch_groups["Train Accuracy"], marker='o', label="Train Accuracy")
plt.plot(epoch_groups.index, epoch_groups["Val Accuracy"], marker='o', label="Val Accuracy")
plt.plot(epoch_groups.index, epoch_groups["Test Accuracy"], marker='o', label="Test Accuracy")

# 4️ Tambahkan detail
plt.title("Pengaruh Jumlah Epoch terhadap Akurasi Model")
plt.xlabel("Jumlah Epoch")
plt.ylabel("Akurasi (0–1)")
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.show()
