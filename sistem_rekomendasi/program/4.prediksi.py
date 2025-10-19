# from flask import Flask, render_template, request, jsonify
# import pandas as pd
# import re, os, emoji
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# from gensim.models import Word2Vec
# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.models import load_model
# from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
# import requests
# from io import BytesIO
# import base64

# app = Flask(__name__)

# stemmer = StemmerFactory().create_stemmer()
# model_w2v = Word2Vec.load('sistem_rekomendasi/model_word2vec/word2vec_tokopedia.model')
# model_lstm = load_model('sistem_rekomendasi/model_lstm/lstm_tokopedia_final.h5')
# train_data = pd.read_excel('sistem_rekomendasi/hasil_preprocessing/5199_data_komentar_labeled.xlsx')
# tokenizer = Tokenizer()
# tokenizer.fit_on_texts(train_data['cleaned'])

# def clean_text(teks):
#     teks = emoji.replace_emoji(str(teks), replace='')
#     teks = re.sub(r'#|@', '', teks)
#     teks = re.sub(r'[^\w\s]', ' ', teks)
#     teks = re.sub(r'\s+', ' ', teks).strip()
#     teks = teks.lower()
#     return ' '.join([stemmer.stem(w) for w in teks.split()])

# def scrape_tokopedia(url):
#     match = re.search(r'/([^/]+)/review', url)
#     if not match:
#         return []
#     product_id = match.group(1)
#     reviews = []
#     headers = {
#         "accept": "application/json, text/plain, */*",
#         "content-type": "application/json",
#         "origin": "https://www.tokopedia.com",
#         "referer": "https://www.tokopedia.com/",
#         "user-agent": "Mozilla/5.0"
#     }
#     for page in range(1, 4):
#         query = {
#             "operationName": "ReviewListShopProduct",
#             "variables": {
#                 "page": page,
#                 "perPage": 10,
#                 "productID": product_id,
#                 "sort": 1,
#                 "filter": {"rating": [], "media": [], "withContent": True}
#             },
#             "query": """query ReviewListShopProduct($page: Int!, $perPage: Int!, $productID: String!, $sort: Int!, $filter: ReviewFilterInput) {
#               reviewListShopProduct(page: $page, perPage: $perPage, productID: $productID, sort: $sort, filter: $filter) {
#                 data { content }
#               }
#             }"""
#         }
#         res = requests.post("https://gql.tokopedia.com/graphql/ReviewListShopProduct", headers=headers, json=query)
#         if res.status_code != 200:
#             break
#         data = res.json()
#         items = data.get("data", {}).get("reviewListShopProduct", {}).get("data", [])
#         for i in items:
#             content = i.get("content", "").strip()
#             if content:
#                 reviews.append(content)
#     return reviews

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     url = request.form['url']
#     comments = scrape_tokopedia(url)
#     if not comments:
#         return jsonify({'error': 'Tidak ada komentar ditemukan.'})

#     df = pd.DataFrame(comments, columns=['komentar'])
#     df['cleaned'] = df['komentar'].apply(clean_text)
#     seq = tokenizer.texts_to_sequences(df['cleaned'])
#     X = pad_sequences(seq, maxlen=100, padding='post')
#     preds = (model_lstm.predict(X) > 0.5).astype("int32")
#     df['label'] = ['Asli' if p == 1 else 'Palsu' for p in preds]

#     asli, palsu = (df['label'] == 'Asli').sum(), (df['label'] == 'Palsu').sum()
#     total = len(df)
#     hasil = "Barang Asli" if asli > palsu else "Barang Palsu"

#     plt.figure(figsize=(5,4))
#     plt.bar(['Asli', 'Palsu'], [asli, palsu], color=['green','red'])
#     plt.title('Perbandingan Hasil Prediksi')
#     plt.tight_layout()
#     img = BytesIO()
#     plt.savefig(img, format='png')
#     img.seek(0)
#     grafik_url = base64.b64encode(img.getvalue()).decode()

#     return jsonify({
#         'data': df.to_dict(orient='records'),
#         'asli': asli, 'palsu': palsu,
#         'hasil': hasil,
#         'grafik': grafik_url
#     })

# if __name__ == '__main__':
#     app.run(debug=True)
