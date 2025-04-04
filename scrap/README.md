# 📊 Projet de Scraping et Préparation de Données pour LLM

## ✨ Objectif

Ce projet a pour objectif de scraper, structurer et enrichir des données de formations issues d'un site web, en vue d'une exploitation par un LLM (Large Language Model) pour un système de recommandation de parcours personnalisé.

Il permet d'automatiser :

- L'extraction HTML via Playwright
- Le nettoyage + enrichissement des données (BeautifulSoup)
- La préparation à la vectorisation (chunking)
- La génération de documentation automatique

## 🧰 Stack Technique

- **Python 3.12**
- **Playwright** : scraping dynamique (Angular)
- **BeautifulSoup** : nettoyage HTML
- **pandas** : manipulation tabulaire
- **ChromaDB** : stockage des embeddings (optionnel)
- **SentenceTransformers** / **OpenAI** : vectorisation (au choix)

## 📚 Structure du pipeline

```
project/
├── main.py                      # Scraping initial des pages de formation
├── clean.py                     # Nettoyage, enrichissement, extraction du résumé HTML
├── prepare_vectorisation.py     # Chunking + structuration prête pour LLM
├── vectorize_chunks.py          # Vectorisation des textes (SentenceTransformers ou OpenAI)
├── README_generator.py          # Création automatique d'un récapitulatif des formations
├── run_pipeline.py              # Exécution interactive du pipeline complet
├── content/
│   ├── json/formations/        # Fichiers JSON individuels par formation
│   └── csv/formations/         # Fichiers CSV individuels par formation
└── README.md                   # Ce fichier
```

## 🔧 Utilisation

### 1. Scraping des formations

```bash
python main.py
```

> Crée les fichiers JSON et CSV à partir du site source

### 2. Nettoyage et enrichissement

```bash
python clean.py
```

> Nettoie le HTML, ajoute les métadonnées, un résumé HTML, et une colonne `niveau`

### 3. Préparation à la vectorisation

```bash
python prepare_vectorisation.py
```

> Divise le texte des formations en *chunks* prêts à être vectorisés (tokenisation, metadata, etc.)

### 4. Vectorisation

```bash
python vectorize_chunks.py
```

> Convertit les chunks en vecteurs, et les stocke dans **ChromaDB** (optionnel si vectorisation confiée au collaborateur)

### 5. Génération du README

```bash
python README_generator.py
```

> Génère automatiquement un tableau récapitulatif de toutes les formations

### 6. Exécution guidée du pipeline

```bash
python run_pipeline.py
```

> Permet de sélectionner les étapes à lancer via un menu interactif (scraping, cleaning, etc.)

## 📊 Exemple de résultat

Le fichier `content/README.md` liste toutes les formations disponibles sous forme de tableau :

| Titre               | Niveau        | Durée   | Modalité | Lieu       | Tarif   |
| ------------------- | ------------- | ------- | -------- | ---------- | ------- |
| Formation SQL/NoSQL | Intermédiaire | 3 jours | Hybride  | À distance | 2100 HT |
| ...                 | ...           | ...     | ...      | ...        | ...     |

## 🌐 Auteur du pipeline

**Michel**, Data Analyst passionné de structuration, UX, et pipeline intelligents.

## ✨ Prochaine étape

L'intégration du **LLM** sera prise en charge par **Mohammed**, en partant des données enrichies par ce pipeline.

> ✉️ Ce projet est en constante évolution. Chaque étape est modulable et documentée.

---

*"Vers l'infini et au-delà !"*

