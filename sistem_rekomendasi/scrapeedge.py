from selenium import webdriver
import time

url = "https://www.tokopedia.com/hokaofficialstore/sneaker-hoka-kawana-running-shoes-black-copper-original-black-40-b71d9/review"

driver = webdriver.Edge()
driver.get(url)

time.sleep(10)  # kasih waktu agar JS load

html = driver.page_source
with open("tokopedia_debug.html", "w", encoding="utf-8") as f:
    f.write(html)

driver.quit()

print("✅ HTML halaman disimpan ke tokopedia_debug.html")
