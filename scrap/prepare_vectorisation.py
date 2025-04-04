import os
import json
import re
from pathlib import Path
from bs4 import BeautifulSoup
from uuid import uuid4

# Paramètres
INPUT_DIR = Path("content/json/formations")
OUTPUT_DIR = Path("content/chunks")
CHUNK_SIZE = 500  # nombre approximatif de tokens par chunk

# Création du dossier de sortie
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def clean_html(raw_html):
    """Nettoie le HTML en retirant les balises inutiles"""
    soup = BeautifulSoup(raw_html, 'html.parser')
    # Supprimer les balises de style ou script
    for tag in soup(["script", "style", "nav", "footer"]):
        tag.decompose()
    return soup.get_text(separator="\n").strip()

def split_text(text, max_length=CHUNK_SIZE):
    """Découpe le texte en morceaux de taille raisonnable"""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks, current = [], ""
    for sentence in sentences:
        if len(current) + len(sentence) <= max_length:
            current += " " + sentence
        else:
            chunks.append(current.strip())
            current = sentence
    if current:
        chunks.append(current.strip())
    return chunks

def extract_text(data):
    """Fusionne les champs utiles en un seul bloc de texte brut"""
    parts = []
    for key in ["objectifs", "prerequis", "public", "programme"]:
        if isinstance(data.get(key), list):
            parts.append("\n".join(data[key]))
    if "resume_html" in data:
        parts.append(clean_html(data["resume_html"]))
    return "\n\n".join(parts)

# Traitement de chaque fichier JSON
for file in INPUT_DIR.glob("*.json"):
    with open(file, "r") as f:
        data = json.load(f)
    
    titre = data.get("titre", "Sans titre")
    source = file.name
    texte_complet = extract_text(data)
    chunks = split_text(texte_complet)

    output = []
    for idx, chunk in enumerate(chunks):
        output.append({
            "chunk_id": str(uuid4()),
            "titre": titre,
            "content": chunk,
            "source": source
        })

    # Nom de fichier nettoyé
    file_stem = file.stem.replace(" ", "_")
    output_path = OUTPUT_DIR / f"{file_stem}.json"
    with open(output_path, "w") as out:
        json.dump(output, out, indent=2, ensure_ascii=False)

    print(f"✅ Chunks générés : {output_path}")
