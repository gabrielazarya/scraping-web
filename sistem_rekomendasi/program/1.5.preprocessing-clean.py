import pandas as pd
import re

# --- 1. Baca file Excel ---
df = pd.read_excel("sistem_rekomendasi/hasil_preprocessing/coba_all_data_labeled.xlsx")

# --- 2. Tentukan kolom teks komentar (ubah sesuai nama kolommu) ---
text_col = "cleaned"

# --- 3. Fungsi untuk deteksi noise ---
def is_noise(text):
    if not isinstance(text, str):
        return True

    t = text.strip().lower()

    # Hanya berisi tanda baca, angka, atau emoji
    if re.fullmatch(r"[\W_]+", t):
        return True

    # Hanya huruf acak tanpa vokal (contoh: sjsjsj, yyy, tttt)
    if not re.search(r"[aiueo]", t) and re.match(r"^[a-z]+$", t):
        return True

    # Terlalu banyak huruf sama (contoh: yyyyy, kkkkk)
    if re.fullmatch(r"(.)\1{3,}", t):
        return True

    return False

# --- 4. Deteksi noise ---
df["is_noise"] = df[text_col].apply(is_noise)

# --- 5. Ambil komentar yang dianggap noise ---
noise_comments = df[df["is_noise"]][text_col].tolist()

# --- 6. Tampilkan semua komentar yang dihapus ---
print("\n=== Komentar yang dihapus (noise) ===")
if len(noise_comments) == 0:
    print("Tidak ada komentar noise ditemukan")
else:
    for i, comment in enumerate(noise_comments, 1):
        print(f"{i}. {comment}")

print(f"\nTotal komentar noise: {len(noise_comments)}")

# --- 7. Hapus komentar noise ---
df_clean = df[~df["is_noise"]].drop(columns=["is_noise"])

# --- 8. Simpan hasil bersih ---

output_file = "sistem_rekomendasi/hasil_preprocessing/all_data_komentar_cleaned.xlsx"
df_clean.to_excel(output_file, index=False)

print(f"\nData bersih berhasil disimpan ke: {output_file}")
print(f"Total komentar setelah dibersihkan: {len(df_clean)}")
