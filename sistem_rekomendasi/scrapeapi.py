import os
import requests
import pandas as pd

# Ganti product_id sesuai produk
product_id = "2209384079"  # contoh ID produk (bisa diambil dari URL atau inspect network)

reviews = []
page = 1

while True:
    api_url = f"https://tokopedia.com/review-api/review?product_id={product_id}&page={page}&per_page=20"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.7339.186 Safari/537.36",
        "Accept": "application/json",
    }

    r = requests.get(api_url, headers=headers)
    if r.status_code != 200:
        print("Gagal ambil data:", r.status_code)
        break

    data = r.json()
    if not data.get("data") or not data["data"]["reviews"]:
        print("Tidak ada review lagi.")
        break

    for rev in data["data"]["reviews"]:
        reviews.append(rev["content"])

    print(f"Ambil halaman {page}, total sekarang {len(reviews)} ulasan.")
    page += 1

# Simpan ke CSV
os.makedirs("sistem_rekomendasi/ulasan", exist_ok=True)
df = pd.DataFrame(reviews, columns=["Ulasan"])
df.to_csv("sistem_rekomendasi/ulasan/ulasan_api.csv", index=False, encoding="utf-8-sig")
print(f"✅ Data berhasil disimpan, total: {len(reviews)} ulasan.")
