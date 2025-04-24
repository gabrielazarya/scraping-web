import os
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import time
import pandas as pd

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

    for page in range(3):
        time.sleep(3)

        # Klik semua "Selengkapnya"
        buttons = driver.find_elements(By.XPATH, "//button[contains(text(), 'Selengkapnya')]")
        for btn in buttons:
            try:
                driver.execute_script("arguments[0].click();", btn)
            except:
                continue

        soup = BeautifulSoup(driver.page_source, "html.parser")
        containers = soup.find_all("article", class_="css-15m2bcr")  # class sesuai hasil inspect

        for container in containers:
            try:
                review = container.find("p", class_="css-cvmev1-unf-heading e1qvo2ff8").text.strip()
                if review:
                    data.append(review)
            except:
                continue

        # Klik "Laman berikutnya"
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

    # Simpan CSV
    df = pd.DataFrame(data, columns=["Ulasan"])
    df.to_csv("ulasan/Ulasan_Lengkap_Tokopedia.csv", index=False)
    print("✅ Data berhasil disimpan ke ulasan/Ulasan_Lengkap_Tokopedia.csv")
