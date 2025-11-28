import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 1. Baca file CSV
df = pd.read_csv("sistem_rekomendasi\model_terbaik\gabungan_lengkap.csv")

# Cari titik dengan Test Accuracy tertinggi secara global
max_accuracy = df['Test Accuracy'].max()
best_model_global = df[df['Test Accuracy'] == max_accuracy].iloc[0]

# 2. BUAT DIAGRAM BAR - MODEL TERBAIK PER FOLD UNTUK SETIAP KOMBINASI K DAN EPOCH
k_values = sorted(df['K'].unique())
epochs = sorted(df['Epoch'].unique())

print("=" * 70)
print("DIAGRAM BAR - MODEL TERBAIK PER FOLD UNTUK SETIAP K DAN EPOCH")
print("=" * 70)

# Loop melalui setiap kombinasi K dan Epoch
for k in k_values:
    for epoch in epochs:
        # Buat figure baru untuk diagram bar
        plt.figure(figsize=(14, 7))
        
        # Filter data untuk K dan Epoch tertentu
        combo_data = df[(df['K'] == k) & (df['Epoch'] == epoch)]
        
        # Dapatkan fold yang tersedia untuk K ini
        available_folds = sorted(combo_data['Fold'].unique())
        
        # Cari model terbaik untuk setiap fold dalam kombinasi ini
        best_per_fold = []
        best_values = []
        batch_sizes = []
        dropouts = []
        
        for fold in available_folds:
            fold_data = combo_data[combo_data['Fold'] == fold]
            if len(fold_data) > 0:
                best_in_fold = fold_data.loc[fold_data['Test Accuracy'].idxmax()]
                best_per_fold.append(best_in_fold)
                best_values.append(best_in_fold['Test Accuracy'])
                batch_sizes.append(best_in_fold['Batch'])
                dropouts.append(best_in_fold['Dropout'])
        
        # Warna untuk setiap bar
        colors = plt.cm.Set3(np.linspace(0, 1, len(available_folds)))
        
        # Buat diagram bar
        bars = plt.bar(available_folds, best_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Tambahkan nilai di atas bar
        for i, (bar, value, batch, dropout) in enumerate(zip(bars, best_values, batch_sizes, dropouts)):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{value:.4f}\nB:{batch}, D:{dropout}', 
                    ha='center', va='bottom', fontweight='bold', fontsize=14)
        
        # Tandai model terbaik GLOBAL jika ada di kombinasi ini
        if (best_model_global['K'] == k and best_model_global['Epoch'] == epoch and best_model_global['Fold'] in available_folds):
            global_fold_idx = available_folds.index(best_model_global['Fold'])
            bars[global_fold_idx].set_edgecolor('red')
            bars[global_fold_idx].set_linewidth(4)
            bars[global_fold_idx].set_facecolor('gold')
        
        # Detail grafik - PERBESAR FONT DI SINI
        plt.title(f'Model Terbaik per Fold - K = {k}, Epoch = {epoch}\n(Test Accuracy Tertinggi untuk Setiap Fold)', 
                  fontsize=18, fontweight='bold', pad=25)
        plt.xlabel('Fold', fontsize=16, fontweight='bold')  # Perbesar font sumbu x
        plt.ylabel('Test Accuracy', fontsize=16, fontweight='bold')  # Perbesar font sumbu y
        
        # Perbesar font angka pada sumbu x dan y
        plt.xticks(available_folds, fontsize=14)
        plt.yticks(fontsize=14)
        
        plt.grid(True, linestyle='--', alpha=0.4, axis='y')
        
        # Atur batas y-axis
        if len(best_values) > 0:
            plt.ylim(min(best_values) - 0.05, max(best_values) + 0.1)
        
        # Tambahkan informasi di bawah grafik
        plt.figtext(0.5, 0.01, 
                    f"Keterangan: B = Batch Size, D = Dropout Rate | Setiap bar menunjukkan konfigurasi terbaik untuk fold tersebut",
                    ha="center", fontsize=11, style='italic', bbox={"facecolor":"lightgray", "alpha":0.5, "pad":5})
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)
        plt.show()
        
        # Print informasi untuk kombinasi ini
        print(f"\nK={k}, EPOCH {epoch}:")
        print(f"  Model terbaik per fold:")
        for i, fold in enumerate(available_folds):
            print(f"    Fold {fold}: Test Acc = {best_values[i]:.6f}, Batch = {batch_sizes[i]}, Dropout = {dropouts[i]}")
        
        # Cari model terbaik dalam kombinasi ini
        if len(best_values) > 0:
            best_in_combo_idx = np.argmax(best_values)
            print(f"  Model terbaik dalam kombinasi ini: Fold {available_folds[best_in_combo_idx]} dengan Test Accuracy {best_values[best_in_combo_idx]:.6f}")

# 3. Print informasi model terbaik global
print("\n" + "=" * 70)
print("INFORMASI MODEL TERBAIK GLOBAL:")
print("=" * 70)
print(f"Test Accuracy : {best_model_global['Test Accuracy']:.6f}")
print(f"K             : {best_model_global['K']}")
print(f"Fold          : {best_model_global['Fold']}")
print(f"Epoch         : {best_model_global['Epoch']}")
print(f"Batch Size    : {best_model_global['Batch']}")
print(f"Dropout       : {best_model_global['Dropout']}")
print(f"Train Accuracy: {best_model_global['Train Accuracy']:.6f}")
print(f"Val Accuracy  : {best_model_global['Val Accuracy']:.6f}")
print(f"F1 Score Macro: {best_model_global['F1_Macro']:.6f}")