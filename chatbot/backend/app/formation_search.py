import os
import json
import nltk
import spacy
import joblib
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

class FormationSearch:
    def __init__(self, json_paths, model_cache=r"app\tfidf_model_all.joblib"):
        self.json_paths = json_paths
        self.cache_file = model_cache
        self.stop_words = set(stopwords.words("french"))
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = SnowballStemmer("french")
        self.nlp = spacy.load("fr_core_news_md")
        self.data = []

        if os.path.exists(self.cache_file):
            print("📦 Chargement du modèle TF-IDF depuis le cache...")
            self.vectorizer, self.tfidf_matrix, self.metadata = joblib.load(self.cache_file)
        else:
            print("⚙️  Traitement initial des données...")
            self.data = self.load_all_data()
            self.texts, self.metadata = self.preprocess_data()
            self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 6))
            self.tfidf_matrix = self.vectorizer.fit_transform(self.texts)
            joblib.dump((self.vectorizer, self.tfidf_matrix, self.metadata), self.cache_file)
            print("✅ Modèle sauvegardé dans :", self.cache_file)

    def load_all_data(self):
        all_data = []
        for path in self.json_paths:
            if not os.path.exists(path):
                print(f"⚠️ Fichier introuvable : {path}")
                continue
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                all_data.extend(data)
        return all_data

    def preprocess_text(self, text):
        # List of words to exclude
        exclude_words = set([
            "format", "programm", "exemple", "text", "data", "tutorial", "lecture", 
            "cours", "niveau", "objectif", "module", "distance", "lieu", "qui", "quoi", 
            "comment", "pourquoi", "où", "combien", "lequel", 
            "chaque", "tout", "aucun", "tous", "quel", "cela", "ça", 
            "celui", "autre", "même", "quelque", "ni", "sur"
        ])

        # Split the text into words and remove unwanted words
        words = text.lower().split()
        cleaned_words = [
            word for word in words if not any(exclude_word in word for exclude_word in exclude_words)
        ]

        # Re-create the cleaned text
        cleaned_text = " ".join(cleaned_words)

        # NLP processing
        doc = self.nlp(cleaned_text)
        filtered_tokens = []

        for token in doc:
            if token.is_alpha and token.lemma_ not in self.stop_words and token.pos_ != "VERB":
                lemma = self.lemmatizer.lemmatize(token.lemma_)
                stem = self.stemmer.stem(lemma)
                filtered_tokens.append(stem)

        print(f"\n\n\nprocessed text \n\n".join(filtered_tokens))
        return " ".join(filtered_tokens)



    def extract_searchable_text(self, fiche):
        parts = [
            fiche.get("titre", ""),
            fiche.get("TYPE_EMPLOI_ACCESSIBLES", ""),
            fiche.get("ACTIVITES_VISEES", ""),
            fiche.get("CAPACITES_ATTESTEES", "")
        ]
        return " ".join(parts)

    def preprocess_data(self):
        docs = []
        meta = []
        for fiche in self.data:
            full_text = self.extract_searchable_text(fiche)
            clean_text = self.preprocess_text(full_text)
            if not clean_text.strip():
                continue
            meta.append(fiche)
            docs.append(clean_text)
        return docs, meta

    def search(self, query, k=10):
        query_clean = self.preprocess_text(query)
        print(f"\n\n\Clean Query\n\n {query_clean}\n\n")
        query_vector = self.vectorizer.transform([query_clean])
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        top_indices = similarities.argsort()[::-1][:k]
        results = []
        for i in top_indices:
            if similarities[i] > 0:
                results.append((self.metadata[i], similarities[i]))
        print(f"🔍 {len(results)} résultats trouvés pour la requête '{query}'")
        return results

    def filter_formations(self, **criteria):
        """Filters formations based on dynamic criteria."""
        if not self.data:
            self.load_all_data()  # Ensure that the data is loaded before filtering.

        filtered = []
        for fiche in self.data:
            match = True
            for key, value in criteria.items():
                if fiche.get(key) != value:
                    match = False
                    break
            if match:
                filtered.append(fiche)
        return filtered



if __name__ == "__main__":
    engine = FormationSearch([
        "content/rncp/rncp.json",
        "content/formations_internes.json"
    ])

    print("🔍 Recherche de formations (internes + RNCP). Tapez 'exit' pour quitter.")

    while True:
        query = input("\nEntrez un mot-clé (ex: cloud, IA, gestion de projet) : ").strip()
        if query.lower() in ["exit", "quit", "q"]:
            break

        results = engine.search(query, k=4)
        internal = [r for r in results if r[0].get("_source") == "internal"]
        external = [r for r in results if r[0].get("_source") != "internal"]

        if internal:
            print(f"\n💼 {len(internal)} formations internes trouvées :")
            print("Voici des formations proposées par Beyond Expertise :")
            for meta, score in internal:
                print(f"\n▶ {meta['titre']}")
                print(f"  ID: {meta.get('ID', 'N/A')} — Score: {score:.3f}")
                print(f"  Lieu : {meta.get('lieu', '')}")
                print(f"  Tarif : {meta.get('tarif', '')}")
                print(f"  Durée : {meta.get('duree', '')}")
                print(f"  Modalité : {meta.get('modalite', '')}")
        elif external:
            print(f"\n📚 {len(external)} formations RNCP trouvées :")
            print("Voici des formations du RNCP que j'ai trouvé pour toi :")
            for meta, score in external:
                print(f"\n▶ {meta['titre']}")
                print(f"  ID: {meta.get('ID', 'N/A')} — Score: {score:.3f}")
                print(f"  Niveau : {meta.get('NOMENCLATURE_EUROPE_INTITULE', '')}")
                print(f"  Emplois : {meta.get('TYPE_EMPLOI_ACCESSIBLES', '')}")
               # print(f"  URL : {meta.get('LIEN_URL_DESCRIPTION', '')}")
        else:
            print("Aucune formation trouvée.")
