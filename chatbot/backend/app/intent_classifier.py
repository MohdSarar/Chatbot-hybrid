"""
intent_classifier.py
--------------------
Classificateur d'intentions léger et efficace (compatible SentenceTransformer)
"""
import joblib
import logging
from typing import Tuple, Optional

logger = logging.getLogger("intent_classifier")

class IntentClassifier:
    """Classificateur d'intentions basé sur ML et embeddings."""

    def __init__(self, model_path: str = r"app/intent_model.pkl"):
        """Charge le modèle pré-entraîné."""
        try:
            model_bundle = joblib.load(model_path)
            if model_bundle : 
                print("model intent classifier found and loaded successfully")
            self.model = model_bundle["classifier"]
            self.label_encoder = model_bundle["label_encoder"]
            # Nouveau : Charger le modèle d'embedding SentenceTransformer
            from sentence_transformers import SentenceTransformer
            self.embedder = SentenceTransformer(model_bundle["embedder_name"])
            logger.info("Intent classifier and embedder loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load intent classifier: {e}")
            self.model = None
            self.embedder = None
            self.label_encoder = None

    def predict(self, text: str) -> Tuple[str, float]:
        """
        Prédit l'intention d'un texte.
        Returns:
            (intent_tag, confidence_score)
        """
        if not self.model or not self.embedder:
            return "other", 0.0
        try:
            X = self.embedder.encode([text])
            intent_encoded = self.model.predict(X)[0]
            # DECODE THE INTENT NUMBER TO NAME!
            intent = self.label_encoder.inverse_transform([intent_encoded])[0]
            proba = self.model.predict_proba(X)[0]
            confidence = proba.max()
            if confidence < 0.3:
                return "other", confidence
            return intent, confidence
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return "other", 0.0

    def predict_top_k(self, text: str, k: int = 3) -> list:
        """
        Retourne les k intentions les plus probables.
        Returns:
            [(intent, confidence), ...]
        """
        if not self.model or not self.embedder:
            return [("other", 0.0)]
        try:
            X = self.embedder.encode([text])
            proba = self.model.predict_proba(X)[0]
            classes = self.model.classes_
            top_indices = proba.argsort()[-k:][::-1]
            results = []
            for idx in top_indices:
                if proba[idx] > 0.1:
                    # DECODE EACH CLASS TO GET THE INTENT NAME!
                    intent_name = self.label_encoder.inverse_transform([classes[idx]])[0]
                    results.append((intent_name, proba[idx]))
            return results if results else [("other", 0.0)]
        except Exception as e:
            logger.error(f"Top-k prediction error: {e}")
            return [("other", 0.0)]

    def extract_entities(self, text: str) -> dict:
        """
        Extrait des entités simples du texte.
        """
        entities = {}
        import re
        # Extraction d'âge (ex: "32 ans")
        age_match = re.search(r'\b(\d{1,2})\s*ans?\b', text)
        if age_match:
            entities['age'] = age_match.group(1)
        # Extraction de nombres 1-5 (pour sélection de formation)
        num_match = re.search(r'\b([1-5])\b', text)
        if num_match:
            entities['number'] = num_match.group(1)
        # Mots-clés de domaines (tech, marketing...)
        domains = {
            # Intelligence Artificielle et Data Science
            'intelligence artificielle': [
                'intelligence artificielle', 'ia', 'apprentissage automatique', 'machine learning', 
                'deep learning', 'apprentissage profond', 'réseaux de neurones', 'intelligence artificielle appliquée'
            ],
            'data science': [
                'data science', 'science des données', 'data scientist', 'scientifique de données', 'analyse prédictive',
                'modélisation de données', 'statistiques avancées', 'algorithmes de données'
            ],
            # Data/Analyse/BI
            'data analyst': [
                'data analyst', 'analyste de données', 'analyse de données', 'business analyst', 
                'analyse business', 'consultant data', 'traitement des données', 'manipulation de données'
            ],
            'data visualisation': [
                'data visualisation', 'visualisation de données', 'dataviz', 'tableau de bord', 'power bi', 
                'tableau', 'visualisation', 'matplotlib', 'seaborn', 'plotly'
            ],
            # Bases de données et ETL
            'bases de donnees': [
                'sql', 'no sql', 'nosql', 'base de données', 'bases de données', 'mongodb', 'cassandra',
                'administrateur base de données', 'requête sql', 'gestion base de données'
            ],
            'etl': [
                'etl', 'intégration de données', 'talend', 'pipeline de données', 'processus etl', 
                'extraction transformation chargement', 'flux de données', 'traitement des flux'
            ],
            # Cloud et DevOps
            'cloud': [
                'cloud', 'azure', 'cloud azure', 'microsoft azure', 'services cloud', 
                'ingénieur cloud', 'administrateur cloud', 'cloud computing', 'virtualisation', 'stockage cloud'
            ],
            # Programmation et Python
            'python': [
                'python', 'programmation python', 'développement python', 'script python', 'pandas', 'numpy'
            ],
            # Outils collaboratifs et gestion de projet
            'gestion de projet': [
                'jira', 'atlassian', 'gestion de projet', 'scrum', 'agile', 'chef de projet', 'scrum master',
                'projets agiles', 'kanban'
            ],
            # Autres thématiques (élargir si besoin)
            'business intelligence': [
                'business intelligence', 'bi', 'ingénieur en business intelligence', 'tableaux de bord bi'
            ],
            'big data': [
                'big data', 'traitement big data', 'données massives'
            ],
            'developpement': [
                'développement', 'développeur', 'développement backend', 'développement web', 'développeur backend'
            ]
        }

        text_lower = text.lower()
        for domain, keywords in domains.items():
            if any(kw in text_lower for kw in keywords):
                entities['domain'] = domain
                break
        # Utilisation de spaCy pour détecter un NOM de personne
        try:
            import spacy
            if not hasattr(self, 'nlp'):
                self.nlp = spacy.load("fr_core_news_sm")
            doc = self.nlp(text)
            for ent in doc.ents:
                if ent.label_ == "PER" and ent.text.strip():
                    entities['name'] = ent.text.strip()
                    break
        except Exception as e:
            pass
        # Détecter un âge donné sans "ans" (ex: "30")
        if 'age' not in entities:
            solo_num = re.fullmatch(r'\d{1,2}', text.strip())
            if solo_num:
                entities['age'] = solo_num.group(0)
        return entities