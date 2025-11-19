import pandas as pd
import re
import emoji
import os
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

input_path = 'sistem_rekomendasi/hasil_ulasan/50_csv_ulasan.csv'
df = pd.read_csv(input_path)
kolom_komentar = 'komentar'
df[kolom_komentar] = df[kolom_komentar].astype(str)

print(f"Jumlah data awal: {len(df)}")

df = df.drop_duplicates(subset=[kolom_komentar], keep='first').reset_index(drop=True)
print(f"Setelah hapus duplikat: {len(df)} data unik")

def clean_text(teks):
    if not isinstance(teks, str):
        return ''
    teks = emoji.replace_emoji(teks, replace='')  # hapus emoji penuh
    teks = re.sub(r'#(\w+)', r'\1', teks)         # hapus tanda pagar #
    teks = re.sub(r'@(\w+)', r'\1', teks)         # hapus mention @
    teks = re.sub(r'[^\w\s]', ' ', teks)          # hapus simbol selain huruf/angka/underscore/spasi
    teks = re.sub(r'\s+', ' ', teks).strip()      # hapus spasi berlebih
    return teks

df['cleaned_text'] = df[kolom_komentar].apply(clean_text)
df = df[df['cleaned_text'].astype(bool)].reset_index(drop=True)

df['casefolded'] = df['cleaned_text'].str.lower()

df['tokens'] = df['casefolded'].apply(lambda x: x.split())

normalisasi_dict = {
    "bgt": "banget", "gk": "tidak", "ga": "tidak", "gak": "tidak",
    "nggak": "tidak", "ngga": "tidak", "tp": "tapi", "yg": "yang",
    "brg": "barang", "bgs": "bagus", "rekomen": "direkomendasikan",
    "rek": "rekomendasi", "trmksh": "terima kasih", "mksh": "makasih",
    "udh": "sudah", "sdh": "sudah", "blm": "belum", "sm": "sama",
    "aj": "saja", "nyah": "nya", "ny": "nya", "bhn": "bahan",
    "expetasi": "ekspektasi", "ok": "oke", "mantul": "mantap betul"
}

def normalisasi(tokens):
    return [normalisasi_dict.get(k, k) for k in tokens]

df['normalized'] = df['tokens'].apply(normalisasi)

stem_factory = StemmerFactory()
stemmer = stem_factory.create_stemmer()

def stemming_list(tokens):
    return [stemmer.stem(t) for t in tokens]

df['stemmed'] = df['normalized'].apply(stemming_list)

df['cleaned'] = df['stemmed'].apply(lambda x: ' '.join(x))
df = df[df['cleaned'].str.strip().astype(bool)].reset_index(drop=True)

df['label'] = ''

print("\nContoh hasil preprocessing:")
print(df[['casefolded', 'cleaned', 'label']].head(10))

output_dir = 'sistem_rekomendasi/hasil_preprocessing'
os.makedirs(output_dir, exist_ok=True)

output_full = os.path.join(output_dir, 'data_komentar_full_labeled.xlsx')

df[['komentar', 'cleaned_text', 'casefolded', 'tokens',
    'normalized', 'stemmed', 'cleaned', 'label']].to_excel(output_full, index=False, engine='openpyxl')

print(f"\nFile disimpan sebagai: {output_full}")
print("   Kolom 'label' siap diisi manual (asli/palsu).")
