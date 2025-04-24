from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import time
import pandas as pd

url = input("Masukkan URL produk Tokopedia: ")

if url:
    options = webdriver.ChromeOptions()
    options.add_argument("--start-maximized")
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

    driver.get(url)

    try:
        # Tunggu tab 'Ulasan' muncul, lalu klik
        WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, "//button[contains(., 'Ulasan')]"))
        ).click()
        print("✅ Tab Ulasan diklik.")
    except Exception as e:
        print("❌ Gagal klik tab Ulasan:", e)
        driver.quit()

    # Tunggu konten ulasan muncul
    time.sleep(5)

    soup = BeautifulSoup(driver.page_source, "html.parser")
    review_elements = soup.find_all("span", attrs={"data-testid": "lblItemUlasan"})

    data = []
    for review in review_elements:
        data.append(review.text.strip())

    driver.quit()

    print(data)
    df = pd.DataFrame(data, columns=["Ulasan"])
    df.to_csv("1Ulasan_Produk_Tokopedia.csv", index=False)
