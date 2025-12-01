# Chatbot Hybrid – TF-IDF & Intent Classification

## Objectif
Système de recommandation de formations basé sur :
- une recherche TF-IDF + similarité cosinus,
- un classifieur d’intentions pour comprendre la question utilisateur.

## Stack technique
- Python
- scikit-learn (TF-IDF, Similarity, Classifier)
- NLTK (prétraitement)
- FastAPI (API)

## Fonctionnalités
- Détection d’intention.
- Recherche des formations les plus proches.
- API pour interroger le moteur.

## Installation
```bash
git clone https://github.com/MohdSarar/Chatbot-hybrid.git
cd Chatbot-hybrid
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload
