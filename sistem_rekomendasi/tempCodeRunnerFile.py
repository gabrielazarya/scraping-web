    elements = driver.find_elements(By.CSS_SELECTOR, '[data-testid="lblItemUlasan"]')
    for el in elements:
        text = el.text.strip()
        if text and text not in reviews:
            reviews.append(text)