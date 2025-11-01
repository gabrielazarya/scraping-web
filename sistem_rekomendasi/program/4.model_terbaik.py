import pandas as pd

# =============================
# 1. Baca hasil eksperimen
# =============================
df = pd.read_csv("sistem_rekomendasi/hasil_training_lstm/hasil_training_resume.csv")

# Pastikan kolom Train Accuracy ada
if 'Train Accuracy' not in df.columns:
    print("Kolom 'Train Accuracy' tidak ditemukan. Pastikan disimpan saat training.")
else:
    # =============================
    # 2. Hitung rata-rata per kombinasi
    # =============================
    grouped = df.groupby(['K', 'Epoch', 'Batch', 'Dropout']).agg({
        'Train Accuracy': 'mean',
        'Val Accuracy': 'mean',
        'Precision': 'mean',
        'Recall': 'mean',
        'Val Loss': 'mean',
        'Durasi (detik)': 'mean'
    }).reset_index()

    # =============================
    # 3. Hitung selisih (indikasi overfitting)
    # =============================
    grouped['Delta (Train-Val)'] = grouped['Train Accuracy'] - grouped['Val Accuracy']
    grouped['Status'] = grouped['Delta (Train-Val)'].apply(
        lambda x: 'Seimbang' if abs(x) < 0.03 else ('Overfitting' if x > 0.03 else 'Underfitting')
    )

    # =============================
    # 4. Urutkan dari model terbaik
    # =============================
    best_models = grouped.sort_values(by='Val Accuracy', ascending=False)

    # =============================
    # 5. Tampilkan hasil
    # =============================
    print("\n5 Kombinasi Terbaik berdasarkan Val Accuracy:\n")
    print(best_models.head(5).to_string(index=False))

    best_model = best_models.iloc[0]
    print("\nModel Terbaik (Val Accuracy Tertinggi):")
    print(best_model)

    # =============================
    # 6. Simpan hasil analisis ke CSV
    # =============================
    output_path = "sistem_rekomendasi/hasil_training_lstm/urutan_model_terbaik.csv"
    best_models.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\nHasil analisis disimpan ke: {output_path}")
