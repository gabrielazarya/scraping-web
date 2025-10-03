import os
import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from selenium.common.exceptions import TimeoutException, NoSuchElementException, WebDriverException

def scrape_tokopedia_reviews(url):
    """
    Scrapes product reviews from a given Tokopedia URL.
    
    Args:
        url (str): The Tokopedia review URL (must end with /review).
    
    Returns:
        pd.DataFrame: A DataFrame containing the scraped reviews, or an empty DataFrame on failure.
    """
    options = webdriver.ChromeOptions()
    options.add_argument("--start-maximized")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
    
    driver = None
    try:
        # Ganti dengan jalur yang benar ke file chromedriver.exe yang Anda unduh
        path_to_chromedriver = "D:\\TA\\TokPed\\chromedriver.exe" 
        driver = webdriver.Chrome(service=Service(path_to_chromedriver), options=options)
        driver.get(url)
        
        print("Waiting for page to load...")
        # Use a more robust selector for the review container
        wait = WebDriverWait(driver, 20)
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "div[data-testid='divProductReview']")))
        
        data = set() # Use a set to automatically handle duplicates

        while True:
            # Scroll down to ensure all reviews are loaded
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)

            # Click all "Selengkapnya" buttons
            try:
                buttons = driver.find_elements(By.XPATH, "//button[contains(text(), 'Selengkapnya')]")
                for btn in buttons:
                    try:
                        driver.execute_script("arguments[0].click();", btn)
                        time.sleep(0.5) # Add a small delay for each click
                    except WebDriverException:
                        continue
            except NoSuchElementException:
                print("No 'Selengkapnya' buttons found.")
            
            # Scrape reviews
            reviews = driver.find_elements(By.CSS_SELECTOR, "p[data-testid^='lblItemUlasan']")
            if not reviews:
                print("No reviews found on this page.")
                break

            for r in reviews:
                text = r.text.strip()
                if text:
                    data.add(text)
            
            print(f"Total reviews scraped so far: {len(data)}")

            # Check for "Next Page" button
            try:
                next_button = wait.until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, "button[aria-label^='Laman berikutnya']"))
                )
                driver.execute_script("arguments[0].click();", next_button)
                print("Navigating to the next page...")
                
                # Wait for new reviews to load on the next page
                time.sleep(3) 

            except (TimeoutException, NoSuchElementException):
                print("No more pages or failed to find next page button.")
                break

        return pd.DataFrame(list(data), columns=["Ulasan"])

    except Exception as e:
        print(f"An error occurred: {e}")
        return pd.DataFrame(columns=["Ulasan"])

    finally:
        if driver:
            driver.quit()

if __name__ == "__main__":
    url = input("Masukkan URL ulasan Tokopedia (yang /review): ")
    if url:
        df_reviews = scrape_tokopedia_reviews(url)
        
        if not df_reviews.empty:
            output_dir = "sistem_rekomendasi/ulasan"
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, "ulasan.csv")
            
            df_reviews.to_csv(output_path, index=False, encoding="utf-8-sig")
            print(f"Data berhasil disimpan ke '{output_path}'. Total: {len(df_reviews)} ulasan.")
        else:
            print("Gagal mendapatkan ulasan. Silakan periksa URL atau coba lagi nanti.")