import os
import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

# fungsi auto scroll
def scroll_page(driver, pause_time=1):
    last_height = driver.execute_script("return document.body.scrollHeight")
    while True:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(pause_time)
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height

# input URL
url = input("Masukkan URL ulasan Tokopedia (yang /review): ")

if url:
    options = webdriver.ChromeOptions()
    options.add_argument("--start-maximized")
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver.get(url)

    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "article.css-15m2bcr"))
    )

    data = []

    while True:
        time.sleep(2)

        # scroll supaya review muncul
        scroll_page(driver, pause_time=2)

        # klik semua tombol "Selengkapnya"
        buttons = driver.find_elements(By.XPATH, "//button[contains(text(), 'Selengkapnya')]")
        for btn in buttons:
            try:
                driver.execute_script("arguments[0].click();", btn)
            except:
                continue

        # ambil semua ulasan di halaman ini
        reviews = driver.find_elements(By.CSS_SELECTOR, "p[data-testid='lblItemUlasan']")
        for r in reviews:
            text = r.text.strip()
            if text and text not in data:
                data.append(text)

        # klik tombol "Laman berikutnya" kalau ada
        try:
            next_button = WebDriverWait(driver, 5).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, "button[aria-label^='Laman berikutnya']"))
            )
            driver.execute_script("arguments[0].click();", next_button)

            # tunggu sampai halaman review berikutnya muncul
            WebDriverWait(driver, 10).until(
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, "article.css-15m2bcr"))
            )
        except:
            print("Tidak ada halaman berikutnya atau gagal klik.")
            break

    driver.quit()

    # simpan hasil ke CSV
    os.makedirs("sistem_rekomendasi/ulasan", exist_ok=True)
    df = pd.DataFrame(data, columns=["Ulasan"])
    df.to_csv("sistem_rekomendasi/ulasan/ulasan1.csv", index=False, encoding="utf-8-sig")

    print("Data berhasil disimpan, total:", len(data))
