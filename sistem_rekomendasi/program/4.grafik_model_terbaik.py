import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import ast

# 1. Baca file CSV
df = pd.read_csv("sistem_rekomendasi/model_terbaik/model_terbaik.csv")

# 2. Konversi string confusion_flat menjadi array numpy
def parse_confusion(conf_str):
    try:
        return np.array(ast.literal_eval(conf_str))
    except:
        return np.array([[0, 0], [0, 0]])

df['confusion_matrix'] = df['Confusion_Flat'].apply(parse_confusion)

# 3. Cari model terbaik berdasarkan Test Accuracy
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

# 4. Ambil confusion matrix model terbaik
cm = best_model['confusion_matrix'].astype(int)

# Hitung metrik dari confusion matrix
accuracy = (cm[0,0] + cm[1,1]) / cm.sum()
precision = cm[1,1] / (cm[1,1] + cm[0,1]) if (cm[1,1] + cm[0,1]) > 0 else 0
recall = cm[1,1] / (cm[1,1] + cm[1,0]) if (cm[1,1] + cm[1,0]) > 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(f"\nConfusion Matrix:")
print(f"[[{cm[0,0]:>4} (TN), {cm[0,1]:>4} (FP)]")
print(f" [{cm[1,0]:>4} (FN), {cm[1,1]:>4} (TP)]]")

print(f"\nDetailed Metrics from Confusion Matrix:")
print(f"Akurasi    : {accuracy:.4f}")
print(f"Precision  : {precision:.4f}")
print(f"Recall     : {recall:.4f}")
print(f"F1-Score   : {f1_score:.4f}")

# 5. Buat figure dengan multiple subplots
fig = plt.figure(figsize=(18, 12))

# 5.1 Confusion Matrix
plt.subplot(2, 3, 1)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=True,
            xticklabels=["Pred: Asli", "Pred: Palsu"],
            yticklabels=["Aktual: Asli", "Aktual: Palsu"],
            annot_kws={"size": 12})

plt.title(f"Confusion Matrix - Model Terbaik\n"
          f"K={int(best_model['K'])}, Epoch={int(best_model['Epoch'])}, "
          f"Batch={int(best_model['Batch'])}, Dropout={best_model['Dropout']}\n"
          f"Test Accuracy: {best_model['Test Accuracy']:.4f}", 
          fontsize=12, pad=20)

# 5.2 Perbandingan Akurasi
plt.subplot(2, 3, 2)
accuracies = ['Train', 'Validation', 'Test']
values = [best_model['Train Accuracy'], best_model['Val Accuracy'], best_model['Test Accuracy']]
colors = ['blue', 'orange', 'green']

bars = plt.bar(accuracies, values, color=colors, alpha=0.7)
plt.ylabel('Accuracy')
plt.title('Perbandingan Akurasi Model Terbaik')
plt.ylim(0, 1.0)

# Tambahkan nilai di atas bar
for bar, value in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{value:.4f}', ha='center', va='bottom', fontsize=10)

# 5.3 Perbandingan Metrik Klasifikasi
plt.subplot(2, 3, 3)
metrics = ['Precision', 'Recall', 'F1-Score']
metric_values = [best_model['Precision'], best_model['Recall'], best_model['F1_Macro']]
colors_metrics = ['red', 'purple', 'brown']

bars_metrics = plt.bar(metrics, metric_values, color=colors_metrics, alpha=0.7)
plt.ylabel('Score')
plt.title('Metrik Klasifikasi Model Terbaik')
plt.ylim(0, 1.0)

# Tambahkan nilai di atas bar
for bar, value in zip(bars_metrics, metric_values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{value:.4f}', ha='center', va='bottom', fontsize=10)

# 5.4 Distribusi Test Accuracy untuk setiap K
plt.subplot(2, 3, 4)
k_values = df['K'].unique()
k_values.sort()

data_by_k = [df[df['K'] == k]['Test Accuracy'] for k in k_values]

plt.boxplot(data_by_k, labels=k_values)
plt.xlabel('Nilai K')
plt.ylabel('Test Accuracy')
plt.title('Distribusi Test Accuracy per Nilai K')

# 5.5 Scatter Plot: Train vs Test Accuracy
plt.subplot(2, 3, 5)
plt.scatter(df['Train Accuracy'], df['Test Accuracy'], alpha=0.6, c=df['K'], cmap='viridis')
plt.xlabel('Train Accuracy')
plt.ylabel('Test Accuracy')
plt.title('Train Accuracy vs Test Accuracy')
plt.colorbar(label='Nilai K')

# Garis y=x untuk referensi
min_acc = min(df['Train Accuracy'].min(), df['Test Accuracy'].min())
max_acc = max(df['Train Accuracy'].max(), df['Test Accuracy'].max())
plt.plot([min_acc, max_acc], [min_acc, max_acc], 'r--', alpha=0.5, label='y=x')
plt.legend()

# 5.6 Top 5 Model Terbaik
plt.subplot(2, 3, 6)
top_5 = df.nlargest(5, 'Test Accuracy')
models_labels = [f"K={int(row['K'])}\nE{int(row['Epoch'])}B{int(row['Batch'])}" 
                for _, row in top_5.iterrows()]

plt.barh(models_labels, top_5['Test Accuracy'], color='teal', alpha=0.7)
plt.xlabel('Test Accuracy')
plt.title('Top 5 Model Terbaik')
plt.xlim(0.8, 1.0)

# Tambahkan nilai di ujung bar
for i, (label, acc) in enumerate(zip(models_labels, top_5['Test Accuracy'])):
    plt.text(acc + 0.005, i, f'{acc:.4f}', va='center', fontsize=9)

plt.tight_layout(pad=3.0)
plt.show()

# 6. Tampilkan top 3 model terbaik
print("\n" + "=" * 60)
print("TOP 3 MODEL TERBAIK")
print("=" * 60)

top_3_models = df.nlargest(3, 'Test Accuracy')[['K', 'Fold', 'Epoch', 'Batch', 'Dropout', 'Test Accuracy', 'F1_Macro']]

for i, (idx, model) in enumerate(top_3_models.iterrows(), 1):
    print(f"\n#{i} - Test Accuracy: {model['Test Accuracy']:.4f}")
    print(f"   K: {int(model['K'])}, Fold: {int(model['Fold'])}, Epoch: {int(model['Epoch'])}, "
          f"Batch: {int(model['Batch'])}, Dropout: {model['Dropout']}, "
          f"F1 Macro: {model['F1_Macro']:.4f}")

# 7. Analisis tambahan: Pengaruh Dropout terhadap Performance
print("\n" + "=" * 60)
print("ANALISIS PENGARUH DROPOUT")
print("=" * 60)

dropout_values = df['Dropout'].unique()
dropout_values.sort()

for dropout in dropout_values:
    dropout_data = df[df['Dropout'] == dropout]
    print(f"Dropout {dropout}: {len(dropout_data)} model, Test Accuracy rata-rata: {dropout_data['Test Accuracy'].mean():.4f}")

# 8. Grafik performa berdasarkan Dropout Rate
plt.figure(figsize=(10, 6))

dropout_stats = df.groupby('Dropout')['Test Accuracy'].agg(['mean', 'std', 'count']).reset_index()

plt.errorbar(dropout_stats['Dropout'], dropout_stats['mean'], 
             yerr=dropout_stats['std'], fmt='o-', capsize=5, 
             label='Rata-rata ± Std Dev', linewidth=2, markersize=8)

plt.xlabel('Dropout Rate')
plt.ylabel('Test Accuracy')
plt.title('Pengaruh Dropout Rate terhadap Test Accuracy')
plt.grid(True, alpha=0.3)
plt.legend()

# Tambahkan jumlah sampel di setiap titik
for i, row in dropout_stats.iterrows():
    plt.text(row['Dropout'], row['mean'] + 0.01, f"n={int(row['count'])}", 
             ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()