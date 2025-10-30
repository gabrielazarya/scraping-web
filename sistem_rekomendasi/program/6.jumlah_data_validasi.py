import pandas as pd

# === 1. Baca file Excel ===
file_path = "sistem_rekomendasi/validasi_data/all_produk_data_labeled.xlsx"
df = pd.read_excel(file_path)

# === 2. Isi ulang nilai produk yang kosong akibat merge cell ===
df['produk'] = df['produk'].ffill()  # forward fill agar semua baris diisi produk sebelumnya

# === 3. Hitung jumlah label keseluruhan ===
total_labels = df['label'].value_counts()

# === 4. Hitung jumlah label per produk ===
summary_per_produk = df.groupby('produk')['label'].value_counts().unstack(fill_value=0)

# Tambahkan kolom total data per produk
summary_per_produk['total'] = summary_per_produk.sum(axis=1)

# === 5. Urutkan produk berdasarkan angka setelah kata 'produk' ===
# (agar produk 1, produk 2, ... berurutan meskipun Excel baca acak)
summary_per_produk = summary_per_produk.sort_index(key=lambda x: x.str.extract(r'(\d+)').astype(float)[0])

# === 6. Buat tabel ringkasan total keseluruhan ===
total_summary = pd.DataFrame({
    'label': ['asli', 'palsu', 'total_semua_data'],
    'jumlah': [total_labels.get('asli', 0),
               total_labels.get('palsu', 0),
               len(df)]
})

# === 7. Simpan hasil ke file Excel ===
output_path = "sistem_rekomendasi/validasi_data/hasil_perhitungan_label.xlsx"
with pd.ExcelWriter(output_path) as writer:
    summary_per_produk.to_excel(writer, sheet_name="Per Produk")
    total_summary.to_excel(writer, sheet_name="Total Keseluruhan", index=False)

print("Hasil perhitungan berhasil disimpan ke:", output_path)
print("\n--- Ringkasan per Produk ---")
print(summary_per_produk.head(100))
print("\n--- Total Keseluruhan ---")
print(total_summary)
