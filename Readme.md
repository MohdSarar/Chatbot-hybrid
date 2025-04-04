# Documentation des Données - Projet Chatbot Formation

Ce document a pour objectif d'expliquer la structure des données disponibles dans le dossier `content/`, afin d'aider l'équipe LLM (notamment Mohammed) à exploiter les exports pour la vectorisation et le RAG sans avoir à revoir l'intégralité du pipeline de préparation.

---

## 📁 Structure du dossier `content/`

```bash
content/
├── json/                      # Formations au format JSON structuré (1 fichier par formation)
│   └── formations/
│       └── Formation_<titre>.json
├── csv/                       # Export CSV équivalent des JSON
│   └── formations/
│       └── Formation_<titre>.csv
├── chunks/                    # Textes découpés pour la vectorisation (1 fichier JSON par formation)
├── vectorized/                # Données vectorisées (FAISS/ChromaDB)
│   ├── chroma/                # Base ChromaDB contenant les embeddings et metadata
│   └── vector_data/          # Export brut des vecteurs (format JSON ou pickle)
└── README.md                  # Catalogue des formations généré automatiquement
```

---

## 📘 Description des dossiers & fichiers

### `json/formations/`
- Contient les fiches formations avec tous les champs utiles au LLM :
  - `titre`, `objectifs`, `prérequis`, `public`, `programme`
  - `niveau`, `durée`, `modalité`, `lieu`, `tarif`
  - `resume_html` : HTML complet de la page formation (utile pour le RAG ou résumé automatique)

### `csv/formations/`
- Même contenu que les fichiers JSON, dans un format tabulaire.
- Utilisable pour des analyses ou une conversion rapide.

### `chunks/`
- Contient le texte des formations découpé en "chunks" (morceaux de texte) pour faciliter la vectorisation.
- Chaque fichier correspond à une formation.
- Format : liste de dictionnaires avec au minimum :
  - `chunk_id`, `content`, `source`, `titre`

### `vectorized/`
- Ce répertoire contient les données prêtes à l'emploi pour une utilisation dans un système RAG.

#### `vectorized/chroma/`
- Structure interne gérée automatiquement par ChromaDB (persist directory)
- Inclut les vecteurs, les métadonnées et les documents originaux
- Peut être directement utilisé pour interroger une base Chroma avec un Retriever.

#### `vectorized/vector_data/`
- Format plus brut : JSON ou Pickle contenant :
  - les embeddings vectoriels
  - les textes associés
  - les métadonnées utiles pour filtrer/ranker
- Utile pour entraîner, tester ou migrer vers d'autres bases (FAISS, Pinecone, etc.)

---

## 🧠 Conseils pour Mohammed (LLM / RAG)

- Utiliser les chunks JSON pour construire une base Chroma ou FAISS si besoin.
- Les métadonnées (titre, source, etc.) sont conservées pour le filtrage.
- Le champ `resume_html` peut être utilisé pour générer un résumé contextuel.
- `vectorized/chroma` peut être utilisé directement avec LangChain / LlamaIndex.
- Si besoin de réentraînement, `chunks/` + modèle d’embedding suffisent.

---

## 🛠️ Outils compatibles

- LangChain (Chroma, FAISS, retrievers)
- LlamaIndex (Document loaders, vector store readers)
- Hugging Face Transformers (embeddings, RAG)
- FastAPI pour servir les réponses vectorielles

---

## ✅ Données prêtes pour exploitation

Aucune étape de nettoyage, découpage ou enrichissement supplémentaire n’est requise. Toutes les données sont prêtes pour l’intégration dans une architecture RAG.

👉 Il suffit de charger les chunks ou la base Chroma pour démarrer le développement LLM.

## Commandes

Lancer le serveur => ./start_api.sh