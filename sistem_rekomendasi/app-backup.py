import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import base64

# Mock data (nanti bisa diganti hasil prediksi model)
mock_comments = [
    {"Komentar": "Barangnya jelek banget, beda dari foto", "Label": "Palsu"},
    {"Komentar": "Kualitas bagus, pengiriman cepat", "Label": "Asli"},
    {"Komentar": "Mantab", "Label": "Asli"},
    {"Komentar": "Ori punya, recommended!", "Label": "Asli"},
    {"Komentar": "Palsu! Jangan beli disini", "Label": "Palsu"},
]

def analisis_produk(url):
    if not url.strip():
        return "Masukkan URL terlebih dahulu.", None, None, None

    # --- simulasi hasil klasifikasi ---
    df = pd.DataFrame(mock_comments)
    asli_count = df[df["Label"] == "Asli"].shape[0]
    palsu_count = df[df["Label"] == "Palsu"].shape[0]
    total = len(df)
    verdict = "Palsu" if palsu_count > asli_count else "Asli"

    # --- buat grafik batang ---
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(["Palsu", "Asli"], [palsu_count, asli_count], color=["#e63946", "#2a9d8f"])
    ax.set_title("Perbandingan Label Asli dan Palsu")
    ax.set_ylabel("Jumlah Komentar")
    plt.tight_layout()

    # ubah grafik ke base64 agar tampil di gradio
    buf = BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    img_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    img_html = f'<img src="data:image/png;base64,{img_base64}" width="100%">'

    # ringkasan hasil
    summary = f"""
    <div style='text-align:center;'>
      <h4>Dari {total} komentar:</h4>
      <p><b>{asli_count}</b> Asli | <b>{palsu_count}</b> Palsu</p>
      <h2 style='color:{'#e63946' if verdict=='Palsu' else '#2a9d8f'}'>Barang {verdict}</h2>
    </div>
    """

    return df, img_html, summary, f"✅ Analisis selesai untuk {url}"

# === Desain UI ===
with gr.Blocks(theme=gr.themes.Soft(), title="Sistem Rekomendasi Keaslian Barang") as demo:
    gr.Markdown("<h1 style='text-align:center;'>🛍️ Periksa Keaslian Produk Tokopedia</h1>")
    gr.Markdown("<p style='text-align:center;color:gray;'>Masukkan link produk Tokopedia dengan /review di akhir URL.</p>")
    
    url_input = gr.Textbox(label="Masukkan URL Review Tokopedia", placeholder="https://www.tokopedia.com/nama-toko/nama-produk/review")
    analyze_btn = gr.Button("Mulai Analisis 🔍")

    output_status = gr.Textbox(label="Status", interactive=False)
    output_table = gr.DataFrame(headers=["Komentar", "Label"], label="Hasil Komentar")
    output_plot = gr.HTML(label="Grafik Hasil")
    output_summary = gr.HTML(label="Ringkasan")

    analyze_btn.click(analisis_produk, inputs=url_input, outputs=[output_table, output_plot, output_summary, output_status])

# === Jalankan ===
if __name__ == "__main__":
    demo.launch()
