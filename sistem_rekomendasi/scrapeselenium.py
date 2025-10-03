import os
import time
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

url = input("Masukkan URL ulasan Tokopedia (yang /review): ")

if url:
    # Setup Chrome dengan User-Agent
    options = webdriver.ChromeOptions()
    options.add_argument("--start-maximized")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.7339.186 Safari/537.36"
    )

    # ✅ Tidak usah pakai version="..."
    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=options
    )
    driver.get(url)

    # Tunggu sampai review muncul
    WebDriverWait(driver, 15).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "article.css-15m2bcr"))
    )

    data = []

    while True:
        time.sleep(2)

        # Scroll biar semua review di halaman kebuka
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)

        # Klik semua tombol "Selengkapnya"
        buttons = driver.find_elements(By.XPATH, "//button[contains(text(), 'Selengkapnya')]")
        for btn in buttons:
            try:
                driver.execute_script("arguments[0].click();", btn)
                time.sleep(0.3)
            except:
                continue

        # Ambil review dari halaman
        soup = BeautifulSoup(driver.page_source, "html.parser")
        containers = soup.find_all("article", class_="css-15m2bcr")

        for container in containers:
            try:
                review = container.find("p", {"data-testid": "lblItemUlasan"}).get_text(strip=True)
                if review and review not in data:  # hindari duplikat
                    data.append(review)
            except:
                continue

        # Coba klik tombol "Laman berikutnya"
        try:
            next_button = WebDriverWait(driver, 5).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, "button[aria-label^='Laman berikutnya']"))
            )
            driver.execute_script("arguments[0].click();", next_button)
            time.sleep(2)
        except:
            print("Tidak ada halaman berikutnya atau gagal klik.")
            break

    driver.quit()

    # Simpan hasil ke CSV
    os.makedirs("sistem_rekomendasi/ulasan", exist_ok=True)
    df = pd.DataFrame(data, columns=["Ulasan"])
    df.to_csv("sistem_rekomendasi/ulasan/ulasan1.csv", index=False, encoding="utf-8-sig")
    print(f"Data berhasil disimpan, total: {len(data)} ulasan.")
