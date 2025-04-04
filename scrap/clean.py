from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
import time
import json
import pandas as pd
import os
import re
import unicodedata

# === Fonctions utilitaires ===

# Nettoyage des noms de fichiers (accents, caractères spéciaux, espaces)
def clean_filename(text):
    text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode()
    text = re.sub(r'[^a-zA-Z0-9]+', '_', text)
    return text.strip('_')

# Nettoyage du contenu HTML pour l'utiliser dans le LLM
def clean_html(html):
    if not html:
        return ""
    soup = BeautifulSoup(html, 'html.parser')

    # Supprimer les scripts, styles et balises inutiles
    for tag in soup(['script', 'style', 'meta', 'noscript', 'svg']):
        tag.decompose()

    # Nettoyer les attributs superflus
    for tag in soup.find_all(True):
        tag.attrs = {k: v for k, v in tag.attrs.items() if k in ['href', 'src', 'alt']}

    return soup.prettify()

# === Chemins de sauvegarde ===

base_dir = "/Users/michel/Documents/Data Analyst/STAGE/scrap/content"
output_dir_json = os.path.join(base_dir, "json/cleaned")
output_dir_csv = os.path.join(base_dir, "csv/cleaned")

# Crée les dossiers s'ils n'existent pas
os.makedirs(output_dir_json, exist_ok=True)
os.makedirs(output_dir_csv, exist_ok=True)

# === URLs des pages de formation à scraper ===
urls = [
    "https://www.beyond-expertise.com/formation/sql-nosql",
    "https://www.beyond-expertise.com/formation/python-data-visualisation",
    "https://www.beyond-expertise.com/formation/etl-integration",
    "https://www.beyond-expertise.com/formation/deep-learning",
    "https://www.beyond-expertise.com/formation/machine-learning",
    "https://www.beyond-expertise.com/formation/cloud-azure",
    "https://www.beyond-expertise.com/formation/jira-atlassian",
    "https://www.beyond-expertise.com/formation/intelligence-artificielle",
    "https://www.beyond-expertise.com/formation/power-bi",
    "https://www.beyond-expertise.com/formation/data-analyst"
]

# === Scraping avec Playwright ===
with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()

    for url in urls:
        print(f"\nScraping : {url}")
        page.goto(url)
        page.wait_for_timeout(3000)  # Laisse Angular/JS charger
        page.wait_for_selector("app-formation")

        # --- Extraction des infos structurées ---
        titre = page.locator("h2").nth(0).inner_text()
        print("Titre :", titre)

        objectifs = page.locator("#section-objectives li span").all_inner_texts()
        prerequis = page.locator("#section-prerequisites li span").all_inner_texts()
        public = page.locator("#section-target li span").all_inner_texts()
        programme = page.locator("#section-program .module-header h3").all_inner_texts()

        # --- Métadonnées ---
        durée = page.locator(".highlight-item span").nth(0).inner_text()
        modalité = page.locator(".highlight-item span").nth(1).inner_text()
        tarif = page.locator(".highlight-item span").nth(2).inner_text()
        lieu = page.locator(".highlight-item span").nth(3).inner_text()
        prochaines_sessions = ""  # prévu pour le futur

        # --- Résumé HTML nettoyé ---
        html_content = page.content()
        soup = BeautifulSoup(html_content, 'html.parser')
        contenu_resume = soup.select_one('.content-wrapper')
        resume_html = clean_html(str(contenu_resume)) if contenu_resume else ""

        # --- Construction du dictionnaire formation ---
        formation_data = {
            "titre": titre,
            "objectifs": objectifs,
            "prerequis": prerequis,
            "public": public,
            "programme": programme,
            "durée": durée,
            "modalité": modalité,
            "tarif": tarif,
            "lieu": lieu,
            "prochaines_sessions": prochaines_sessions,
            "resume_html": resume_html
        }

        # --- Sauvegarde propre avec noms nettoyés ---
        titre_clean = clean_filename(titre)
        json_path = os.path.join(output_dir_json, f"{titre_clean}.json")
        csv_path = os.path.join(output_dir_csv, f"{titre_clean}.csv")

        with open(json_path, "w") as f:
            json.dump(formation_data, f, indent=2, ensure_ascii=False)
        print(f"JSON : {json_path}")

        pd.DataFrame([formation_data]).to_csv(csv_path, index=False)
        print(f"CSV : {csv_path}")

    browser.close()
