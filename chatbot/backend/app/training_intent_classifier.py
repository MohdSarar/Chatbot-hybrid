"""
intent_trainer_advanced.py
--------------------------
Entraîne un classificateur d'intentions à partir d'intents.json.
Supporte embeddings vectoriels (Sentence Transformers) OU fallback TF-IDF+SVM.
Augmentation, prétraitement, logging, rapport complet.
"""
import json, joblib, logging, re, os, time
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report

# Logging avancé
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("intent_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("intent_trainer")

# Import Sentence Transformers si possible
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
    logger.info("Sentence Transformers disponible.")
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("Sentence Transformers non disponible, fallback sur TF-IDF.")

# Import spaCy (optionnel, pour future extension)
try:
    import spacy
    SPACY_AVAILABLE = True
    logger.info("spaCy disponible.")
except ImportError:
    SPACY_AVAILABLE = False
    logger.warning("spaCy non disponible.")

EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
INTENTS_FILE = r"app\intents.json"
MODEL_FILE = r"app\intent_model.pkl"

class IntentTrainer:
    def __init__(self, intents_path=INTENTS_FILE, force_tfidf=False):
        self.intents_path = intents_path
        self.force_tfidf = force_tfidf
        self.use_embeddings = SENTENCE_TRANSFORMERS_AVAILABLE and not force_tfidf
        # Charger intents
        with open(intents_path, encoding="utf-8") as f:
            self.intents = json.load(f)["intents"]
        logger.info(f"{len(self.intents)} intentions chargées depuis {intents_path}")

        # Embedding model
        self.embedding_model = None
        if self.use_embeddings:
            self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    def _preprocess_text(self, text):
        """Prétraitement minimal : minuscules, accents, ponctuation."""
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text

    def _augment_data(self, patterns):
        """Augmentation simple des patterns."""
        augmented = list(patterns)
        for pattern in patterns:
            if pattern and not pattern.endswith('?'): augmented.append(pattern + '?')
            if pattern and not pattern.endswith('.'): augmented.append(pattern + '.')
            if 'é' in pattern: augmented.append(pattern.replace('é', 'e'))
            if 'è' in pattern: augmented.append(pattern.replace('è', 'e'))
            accents = {'é': 'e', 'è': 'e', 'ê': 'e', 'à': 'a', 'ù': 'u', 'î': 'i', 'ô': 'o'}
            pattern_no_accents = pattern
            for acc, repl in accents.items():
                pattern_no_accents = pattern_no_accents.replace(acc, repl)
            if pattern_no_accents != pattern:
                augmented.append(pattern_no_accents)
        return augmented

    def _prepare_training_data(self):
        """Génère X (textes) et y (labels) avec augmentation et prétraitement."""
        X, y = [], []
        for intent in self.intents:
            tag = intent["tag"]
            patterns = intent.get("patterns", [])
            augmented = self._augment_data(patterns)
            for pattern in augmented:
                proc = self._preprocess_text(pattern)
                if proc:
                    X.append(proc)
                    y.append(tag)
        return X, y

    def train(self, output_path=MODEL_FILE, test_size=0.2):
        logger.info("Préparation des données...")
        X, y = self._prepare_training_data()
        if not X:
            logger.error("Aucune donnée d'entraînement disponible.")
            return False

        logger.info(f"{len(X)} exemples générés ({len(set(y))} intentions). Split train/test...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=42
        )

        if self.use_embeddings:
            logger.info("Encodage avec Sentence Transformers...")
            embedder = self.embedding_model
            X_train_vec = embedder.encode(X_train, show_progress_bar=True)
            X_test_vec = embedder.encode(X_test, show_progress_bar=True)
            clf = SVC(kernel='linear', probability=True, class_weight='balanced')
            logger.info("Entraînement SVM...")
            clf.fit(X_train_vec, y_train)
            y_pred = clf.predict(X_test_vec)
            model = {
                "classifier": clf,
                "embedder_name": EMBEDDING_MODEL_NAME
            }
        else:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.pipeline import Pipeline
            logger.info("TF-IDF + SVM (pas d'embeddings)...")
            pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(
                    ngram_range=(1,2),
                    max_features=5000,
                    sublinear_tf=True,
                    strip_accents='unicode',
                    stop_words='french'
                )),
                ('clf', SVC(
                    kernel='linear',
                    probability=True,
                    class_weight='balanced'
                ))
            ])
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            model = {
                "pipeline": pipeline
            }

        logger.info("Classification report:\n" + classification_report(y_test, y_pred, digits=3))
        logger.info(f"Saving model to {output_path} ...")
        joblib.dump(model, output_path)
        logger.info("Entraînement terminé.")
        return True

if __name__ == "__main__":
    trainer = IntentTrainer()
    trainer.train()
