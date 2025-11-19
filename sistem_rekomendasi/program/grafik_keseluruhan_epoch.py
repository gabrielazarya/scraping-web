import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 1. Baca file CSV
df = pd.read_csv("sistem_rekomendasi\model_terbaik\gabungan_lengkap.csv")

# Cari titik dengan Test Accuracy tertinggi secara global
max_accuracy = df['Test Accuracy'].max()
best_model_global = df[df['Test Accuracy'] == max_accuracy].iloc[0]

# Warna untuk setiap fold
folds = sorted(df['Fold'].unique())
colors = plt.cm.tab10(np.linspace(0, 1, len(folds)))

# 2. BUAT GRAFIK SCATTER untuk setiap epoch terpisah (TANPA GARIS)
epochs = sorted(df['Epoch'].unique())

print("=" * 70)
print("GRAFIK SCATTER - TEST ACCURACY PER FOLD")
print("=" * 70)

for epoch in epochs:
    # Buat figure baru untuk setiap epoch
    plt.figure(figsize=(12, 8))
    
    epoch_data = df[df['Epoch'] == epoch]
    
    # Cari model terbaik untuk epoch ini
    best_epoch_accuracy = epoch_data['Test Accuracy'].max()
    best_model_epoch = epoch_data[epoch_data['Test Accuracy'] == best_epoch_accuracy].iloc[0]
    
    # Plot untuk setiap fold dalam epoch ini (TANPA GARIS)
    for fold in folds:
        fold_data = epoch_data[epoch_data['Fold'] == fold]
        
        # Plot titik data untuk fold ini saja (tanpa garis)
        plt.scatter([fold] * len(fold_data), fold_data['Test Accuracy'],
                   alpha=0.8, s=100, color=colors[fold-1], label=f"Fold {fold}")
    
    # Tandai model terbaik GLOBAL jika ada di epoch ini (dengan bintang)
    if best_model_global['Epoch'] == epoch:
        plt.scatter(best_model_global['Fold'], best_model_global['Test Accuracy'], 
                  color='red', s=400, marker='*', edgecolors='black', linewidth=3, 
                  zorder=5, label='Model Terbaik Global')
        
        # Anotasi untuk model terbaik global
        plt.annotate(f'TERBAIK GLOBAL\nAcc: {max_accuracy:.4f}\n'
                   f'Batch: {best_model_global["Batch"]}, Dropout: {best_model_global["Dropout"]}', 
                   xy=(best_model_global['Fold'], best_model_global['Test Accuracy']),
                   xytext=(20, 20), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='gold', alpha=0.9),
                   arrowprops=dict(arrowstyle='->', color='red', lw=2))
    
    # Tandai model terbaik untuk EPOCH INI (jika bukan yang global)
    elif best_epoch_accuracy < max_accuracy:
        plt.scatter(best_model_epoch['Fold'], best_model_epoch['Test Accuracy'], 
                  color='orange', s=200, marker='D', edgecolors='black', linewidth=2, 
                  zorder=4, label='Terbaik di Epoch Ini')
        
        # Anotasi untuk model terbaik epoch ini
        plt.annotate(f'TERBAIK EPOCH {epoch}\nAcc: {best_epoch_accuracy:.4f}', 
                   xy=(best_model_epoch['Fold'], best_model_epoch['Test Accuracy']),
                   xytext=(20, 20), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='orange', alpha=0.7),
                   arrowprops=dict(arrowstyle='->', color='orange'))
    
    # Detail grafik
    plt.title(f'Grafik Scatter - Test Accuracy per Fold (Epoch {epoch})', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Fold', fontsize=12)
    plt.ylabel('Test Accuracy', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xticks(folds)
    
    # Atur batas y-axis
    y_min = df['Test Accuracy'].min() - 0.05
    y_max = df['Test Accuracy'].max() + 0.05
    plt.ylim(y_min, y_max)
    
    # Legend
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Tambahkan garis horizontal di nilai tertentu untuk referensi
    for acc in [0.7, 0.8, 0.9]:
        plt.axhline(y=acc, color='gray', linestyle=':', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print info untuk epoch ini
    print(f"\nEPOCH {epoch}:")
    print(f"  Test Accuracy Tertinggi: {best_epoch_accuracy:.6f}")
    if best_model_global['Epoch'] == epoch:
        print("  MODEL TERBAIK GLOBAL")
    elif best_epoch_accuracy < max_accuracy:
        print("  MODEL TERBAIK DI EPOCH INI")
    print(f"  Rata-rata Test Accuracy: {epoch_data['Test Accuracy'].mean():.6f}")
    print(f"  Jumlah Eksperimen     : {len(epoch_data)}")

# 3. BUAT DIAGRAM BAR untuk setiap epoch (TANPA ERROR BAR/STICK)
print("\n" + "=" * 70)
print("DIAGRAM BAR - RATA-RATA TEST ACCURACY PER FOLD")
print("=" * 70)

for epoch in epochs:
    # Buat figure baru untuk diagram bar
    plt.figure(figsize=(12, 6))
    
    epoch_data = df[df['Epoch'] == epoch]
    
    # Hitung rata-rata test accuracy per fold
    fold_means = []
    for fold in folds:
        fold_data = epoch_data[epoch_data['Fold'] == fold]
        fold_means.append(fold_data['Test Accuracy'].mean())
    
    # Buat diagram bar TANPA ERROR BAR (seperti candle tanpa stick)
    bars = plt.bar(folds, fold_means, 
                   color=colors, alpha=0.7, edgecolor='black', linewidth=1)
    
    # Tambahkan nilai di atas bar
    for i, (bar, mean_val) in enumerate(zip(bars, fold_means)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{mean_val:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # Tandai fold dengan rata-rata tertinggi di epoch ini
    best_fold_mean = max(fold_means)
    best_fold_idx = fold_means.index(best_fold_mean)
    bars[best_fold_idx].set_edgecolor('red')
    bars[best_fold_idx].set_linewidth(3)
    
    # Tandai fold yang mengandung model terbaik GLOBAL
    if best_model_global['Epoch'] == epoch:
        global_fold_idx = best_model_global['Fold'] - 1  # -1 karena index dimulai dari 0
        bars[global_fold_idx].set_facecolor('gold')
        bars[global_fold_idx].set_alpha(1.0)
        bars[global_fold_idx].set_edgecolor('red')
        bars[global_fold_idx].set_linewidth(4)
        
        # Anotasi untuk model terbaik global
        plt.annotate(f'TERBAIK GLOBAL\nAcc: {max_accuracy:.4f}', 
                    xy=(global_fold_idx + 1, fold_means[global_fold_idx]),
                    xytext=(20, 25), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.9),
                    arrowprops=dict(arrowstyle='->', color='red', lw=2),
                    fontweight='bold')
    else:
        # Anotasi untuk fold terbaik di epoch ini saja
        plt.annotate(f'TERBAIK EPOCH\n{best_fold_mean:.4f}', 
                    xy=(best_fold_idx + 1, best_fold_mean),
                    xytext=(15, 20), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='orange', alpha=0.8),
                    arrowprops=dict(arrowstyle='->', color='red'))
    
    # Detail grafik
    plt.title(f'Diagram Bar - Test Accuracy per Fold (Epoch {epoch})', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Fold', fontsize=12)
    plt.ylabel('Test Accuracy', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.3, axis='y')
    plt.xticks(folds)
    
    # Atur batas y-axis
    plt.ylim(min(fold_means) - 0.05, max(fold_means) + 0.1)
    
    plt.tight_layout()
    plt.show()
    
    # Print statistik untuk diagram bar
    print(f"\nEPOCH {epoch} - Diagram Bar:")
    print(f"  Rata-rata Tertinggi: Fold {best_fold_idx + 1} = {best_fold_mean:.6f}")
    if best_model_global['Epoch'] == epoch:
        print(f"  Fold {best_model_global['Fold']} mengandung MODEL TERBAIK GLOBAL")

# 4. Print informasi model terbaik global
print("\n" + "=" * 70)
print("INFORMASI MODEL TERBAIK GLOBAL:")
print("=" * 70)
print(f"Test Accuracy : {best_model_global['Test Accuracy']:.6f}")
print(f"Fold          : {best_model_global['Fold']}")
print(f"Epoch         : {best_model_global['Epoch']}")
print(f"Batch Size    : {best_model_global['Batch']}")
print(f"Dropout       : {best_model_global['Dropout']}")
print(f"Train Accuracy: {best_model_global['Train Accuracy']:.6f}")
print(f"Val Accuracy  : {best_model_global['Val Accuracy']:.6f}")
print(f"F1 Score Macro: {best_model_global['F1_Macro']:.6f}")