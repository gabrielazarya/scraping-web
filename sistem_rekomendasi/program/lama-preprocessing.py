# === Import Library Utama ===
import pandas as pd
import re
import emoji
import os
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# ==============================
# 1️⃣ BACA DATA DARI CSV
# ==============================
input_path = 'sistem_rekomendasi/hasil_ulasan/50_csv_ulasan.csv'
df = pd.read_csv(input_path)

kolom_komentar = 'komentar'
df[kolom_komentar] = df[kolom_komentar].astype(str)
print(f"Jumlah data awal: {len(df)}")

# ==============================
# 🧹 PEMBERSIHAN AWAL (Versi Lebih Kuat)
# ==============================
def clean_text(teks):
    if not isinstance(teks, str):
        return ''
    # hapus karakter tak terlihat & whitespace berlebih
    teks = re.sub(r'[\n\r\t\xa0\u200b\ufeff]+', ' ', teks)
    teks = teks.strip()
    # hapus tanda baca yang berdiri sendiri
    if re.fullmatch(r'[\W_]+', teks):
        return ''
    return teks

df[kolom_komentar] = df[kolom_komentar].apply(clean_text)

# Hapus komentar kosong setelah dibersihkan
before = len(df)
df = df[df[kolom_komentar].astype(bool)].reset_index(drop=True)
after = len(df)
print(f"Hapus {before - after} baris kosong/tak valid (tersisa {after})")

# ==============================
# 2️⃣ HAPUS KOMENTAR DUPLIKAT
# ==============================
df = df.drop_duplicates(subset=[kolom_komentar], keep='first').reset_index(drop=True)
print(f"Setelah hapus duplikat: {len(df)}")

# ==============================
# 3️⃣ HAPUS KOMENTAR YANG HANYA BERISI EMOJI
# ==============================
def hanya_emoji(teks):
    teks_strip = teks.strip()
    teks_strip = re.sub(r"\s+", "", teks_strip)
    # Jika semua karakter emoji (dan tidak ada huruf/angka)
    return all(char in emoji.EMOJI_DATA for char in teks_strip) and len(teks_strip) > 0

before = len(df)
df = df[~df[kolom_komentar].apply(hanya_emoji)].reset_index(drop=True)
after = len(df)
print(f"Hapus {before - after} komentar full emoji (tersisa {after})")

# ==============================
# 4️⃣ CASE FOLDING
# ==============================
df[kolom_komentar] = df[kolom_komentar].str.lower()

# ==============================
# 5️⃣ NORMALISASI KATA TIDAK BAKU
# ==============================
normalisasi_dict = {
    "bgt": "banget", "gk": "tidak", "ga": "tidak", "gak": "tidak",
    "nggak": "tidak", "ngga": "tidak", "tp": "tapi", "yg": "yang",
    "brg": "barang", "bgs": "bagus", "rekomen": "direkomendasikan",
    "rek": "rekomendasi", "trmksh": "terima kasih", "mksh": "makasih",
    "udh": "sudah", "sdh": "sudah", "blm": "belum", "sm": "sama", "aj": "saja",
    "nyah": "nya", "ny": "nya", "bhn": "bahan", "expetasi": "ekspektasi"
}

def normalisasi(teks):
    kata = teks.split()
    hasil = [normalisasi_dict.get(k, k) for k in kata]
    return ' '.join(hasil)

df[kolom_komentar] = df[kolom_komentar].apply(normalisasi)

# ==============================
# 6️⃣ TOKENISASI
# ==============================
df['tokens'] = df[kolom_komentar].apply(lambda x: x.split())

# ==============================
# 7️⃣ STOPWORD REMOVAL
# ==============================
stop_factory = StopWordRemoverFactory()
stopwords = set(stop_factory.get_stop_words())

def hapus_stopword(tokens):
    return [token for token in tokens if token not in stopwords]

df['tokens'] = df['tokens'].apply(hapus_stopword)

# ==============================
# 8️⃣ STEMMING (SASTRAWI)
# ==============================
stem_factory = StemmerFactory()
stemmer = stem_factory.create_stemmer()

def stemming_list(tokens):
    return [stemmer.stem(t) for t in tokens]

df['tokens'] = df['tokens'].apply(stemming_list)

# ==============================
# 9️⃣ HASIL PREPROCESSING AKHIR
# ==============================
df['cleaned'] = df['tokens'].apply(lambda x: ' '.join(x))

# Hapus baris yang hasil cleaned-nya kosong
before = len(df)
df = df[df['cleaned'].str.strip().astype(bool)].reset_index(drop=True)
print(f"Hapus {before - len(df)} baris dengan hasil cleaned kosong")

print("\nContoh hasil preprocessing:")
print(df[['cleaned']].head(10))

# ==============================
# 🔟 SIMPAN KE EXCEL (HANYA KOLOM CLEANED)
# ==============================
output_dir = 'sistem_rekomendasi/hasil_preprocessing'
os.makedirs(output_dir, exist_ok=True)

output_path = os.path.join(output_dir, 'data_komentar_cleaned.xlsx')
df[['cleaned']].to_excel(output_path, index=False, engine='openpyxl')

print(f"\n✅ File hasil preprocessing disimpan sebagai '{output_path}' (hanya kolom cleaned)")
