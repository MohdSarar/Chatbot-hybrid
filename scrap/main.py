from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup, Comment
import time
import json
import pandas as pd
import os
import re
import unicodedata

# Nettoyage du nom de fichier (accents, caractères spéciaux, espaces)
def clean_filename(text):
    text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode()
    text = re.sub(r'[^a-zA-Z0-9]+', '_', text)
    return text.strip('_')

# Nettoyage du HTML pour résumé (supprime scripts, styles, commentaires)
def clean_html(html):
    soup = BeautifulSoup(html, 'html.parser')

    # Supprimer les balises inutiles
    for tag in soup(['script', 'style']):
        tag.decompose()

    # Supprimer les commentaires
    for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
        comment.extract()

    # Retourner le HTML nettoyé
    return soup.decode_contents()

# URLs des pages de formations
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

# Préparation des dossiers de sortie
base_path = "/Users/michel/Documents/Data Analyst/STAGE/scrap/content/"
json_dir = os.path.join(base_path, "json/formations/")
csv_dir = os.path.join(base_path, "csv/formations/")
os.makedirs(json_dir, exist_ok=True)
os.makedirs(csv_dir, exist_ok=True)

# Lancement de Playwright
with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()

    # 🔁oucle principale sur les formations
    for url in urls:
        print(f"\nScraping {url}")
        page.goto(url)
        page.wait_for_timeout(3000)  # Attente chargement Angular
        page.wait_for_selector("app-formation")

        # --- Contenu structuré ---
        titre = page.locator("h2").nth(0).inner_text()
        print("📚 Titre :", titre)

        objectifs = page.locator("#section-objectives li span").all_inner_texts()
        prerequis = page.locator("#section-prerequisites li span").all_inner_texts()
        public = page.locator("#section-target li span").all_inner_texts()
        programme = page.locator("#section-program .module-header h3").all_inner_texts()

        # --- Métadonnées (durée, modalité, etc.) ---
        durée = page.locator(".highlight-item span").nth(0).inner_text()
        modalité = page.locator(".highlight-item span").nth(1).inner_text()
        tarif = page.locator(".highlight-item span").nth(2).inner_text()
        lieu = page.locator(".highlight-item span").nth(3).inner_text()

        # --- Préparation pour enrichissement futur ---
        prochaines_sessions = ""
        niveau = ""  # à compléter manuellement ou par LLM

        # --- Extraction et nettoyage du résumé HTML ---
        html_content = page.content()
        soup = BeautifulSoup(html_content, 'html.parser')
        contenu_resume = soup.select_one('.content-wrapper')
        resume_html = clean_html(contenu_resume.decode_contents()) if contenu_resume else ""

        # --- Structure de la formation ---
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
            "niveau": niveau,
            "resume_html": resume_html
        }

        # --- Sauvegarde JSON et CSV ---
        file_name = clean_filename(titre)
        json_path = os.path.join(json_dir, f"{file_name}.json")
        csv_path = os.path.join(csv_dir, f"{file_name}.csv")

        with open(json_path, "w") as f:
            json.dump(formation_data, f, indent=2, ensure_ascii=False)
        print(f"JSON sauvegardé : {json_path}")

        pd.DataFrame([formation_data]).to_csv(csv_path, index=False)
        print(f"CSV sauvegardé : {csv_path}")

    browser.close()
