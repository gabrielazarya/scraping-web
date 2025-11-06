import pandas as pd

# === 1. Baca semua file ===
file1 = pd.read_csv('sistem_rekomendasi/model_terbaik/hasil_training_resume.csv')  
file2 = pd.read_csv('sistem_rekomendasi/model_terbaik/hasil_classification_report.csv')
file3 = pd.read_csv('sistem_rekomendasi/model_terbaik/hasil_confusion_matrix.csv')

# === 2. Hapus kolom 'Durasi (detik)' dari file1 ===
if 'Durasi (detik)' in file1.columns:
    file1 = file1.drop(columns=['Durasi (detik)'])

# === 3. Olah F1 Score per label ===
f1_pivot = file2.pivot_table(
    index=['K', 'Fold', 'Epoch', 'Batch', 'Dropout'],
    columns='Label',
    values='F1-Score'
).reset_index()

# Ubah nama kolom agar lebih jelas
f1_pivot = f1_pivot.rename(
    columns={'asli': 'F1_Score_Asli', 'palsu': 'F1_Score_Palsu'}
)

# Tambahkan kolom F1 Macro
f1_pivot['F1_Macro'] = f1_pivot[['F1_Score_Asli', 'F1_Score_Palsu']].mean(axis=1)

# === 4. Gabungkan file1 dan F1 Score ===
merged = pd.merge(
    file1,
    f1_pivot,
    on=['K', 'Fold', 'Epoch', 'Batch', 'Dropout'],
    how='left'
)

# === 5. Gabungkan juga dengan Confusion Matrix (file3) ===
merged = pd.merge(
    merged,
    file3[['K', 'Fold', 'Epoch', 'Batch', 'Dropout', 'Confusion_Flat']],
    on=['K', 'Fold', 'Epoch', 'Batch', 'Dropout'],
    how='left'
)

# === 6. Simpan hasil akhir ===
merged.to_csv('sistem_rekomendasi/model_terbaik/gabungan_lengkap.csv', index=False)

print("File gabungan lengkap berhasil dibuat")
print(merged)

# === 7. Cari model terbaik berdasarkan Test Accuracy tertinggi ===
# Pastikan kolomnya benar (case-sensitive)
best_model = merged.loc[merged['Test Accuracy'].idxmax()]

# Tampilkan model terbaik
print("\nModel Terbaik Berdasarkan Test Accuracy")
print(best_model)

# Simpan model terbaik ke file terpisah
best_model.to_frame().T.to_csv('sistem_rekomendasi/model_terbaik/model_terbaik.csv', index=False)

print("\nModel terbaik berhasil disimpan")
