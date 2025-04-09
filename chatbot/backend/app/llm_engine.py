# llm_engine.py

from typing import List
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Dict, Any
import pandas as pd
from dotenv import load_dotenv
import os
from openai import OpenAI
#from app.utils import extract_keywords, partial_match_formations


load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
class LLMEngine:
    def __init__(self, df_formations: pd.DataFrame): 
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)
        self.vector_store = None
        self.qa_chain = None
        self.initialize_rag(df_formations)

    def _df_to_documents(self, df: pd.DataFrame) -> List[Document]:
        """Convertit le DataFrame des formations en documents LangChain"""
        docs = []
        for _, row in df.iterrows():
            content = f"""
            Formation: {row['titre']}
            Objectifs: {', '.join(row['objectifs'])}
            Prérequis: {', '.join(row['prerequis'])}
            Programme: {', '.join(row['programme'])}
            Public: {', '.join(row['public'])}
            Lien: {row['lien']}
            """
            docs.append(Document(
                page_content=content,
                metadata={
                    "source": "formations",
                    "titre": row["titre"],
                    "type": "formation"
                }
            ))
        return docs

    def initialize_rag(self, df_formations: pd.DataFrame):
        """Initialise le système RAG avec les données des formations"""
        # Conversion du DataFrame en documents
        documents = self._df_to_documents(df_formations)
        
        # Découpage des textes
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(documents)
        
        # Création du vector store
        self.vector_store = FAISS.from_documents(splits, self.embeddings)
        
        # Création de la chaîne de conversation
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            self.llm,
            self.vector_store.as_retriever(search_kwargs={"k": 15}),
            return_source_documents=True
        )

    # Modifier la méthode generate_response
    def generate_response(self, question: str, chat_history: List[tuple]) -> Dict[str, Any]:
        """Génère une réponse contextuelle avec 2 modes de recherche différents"""
        if not self.qa_chain:
            return {"answer": "Système non initialisé", "sources": []}

        # Détection des requêtes pour la liste complète
        list_keywords = ["liste complète", "toutes les formations", "10 formations", "liste des formations"]
        if any(kw in question.lower() for kw in list_keywords):
            # Mode spécial : récupérer TOUTES les formations
            all_docs = self.vector_store.similarity_search(question, k=len(self.vector_store.index_to_docstore_id))
            sources = list(set(doc.metadata["titre"] for doc in all_docs))
            return {
                "answer": "Voici la liste complète des formations disponibles :\n- " + "\n- ".join(sources),
                "sources": sources
            }
        
        # Mode normal avec contexte RAG
        result = self.qa_chain({
            "question": question,
            "chat_history": chat_history
        })
        
        sources = list(set(doc.metadata["titre"] for doc in result['source_documents']))
        return {"answer": result['answer'], "sources": sources}
    




#Exemples de requêtes potentiellement problématiques pour le système actuel :
list_keywords = [
    "liste complète",
    "toutes les formations", 
    "10 formations",
    "liste des formations",
    # Nouveaux cas à ajouter
    "formations pour débutants",          # Requiert un filtrage par niveau
    "formations avec certification",      # Nécessite un champ métadonnée spécifique
    "formations en présentiel",           # Dépend des modalités non stockées
    "moins de 20 heures",                 # Requiert la durée dans les données
    "formations gratuites",               # Besoin d'information sur les prix
    "spécialité data science",            # Filtrage thématique précis
    "nouvelles formations 2024",          # Nécessite une date de publication
    "sans prérequis techniques",          # Analyse des champs 'prerequis'
    "recommandation personnalisée",       # Nécessite un profiling utilisateur
    "comparaison entre Power BI et Tableau" # Logique comparative non implémentée
]

"""
Améliorations recommandées :

1.Ajouter dans les métadonnées :
metadata={
    "niveau": "débutant", 
    "modalite": "en ligne",
    "duree": "15h",
    "certifiante": True,
    "date_publication": "2024-03-01"
}

2.Étendre la logique de détection :
if any(kw in question.lower() for kw in list_keywords):
    # Nouveau traitement des filtres
    if "débutants" in question.lower():
        filtered = [d for d in all_docs if d.metadata.get("niveau") == "débutant"]
    elif "certification" in question.lower():
        filtered = [d for d in all_docs if d.metadata.get("certifiante")]
    # ... autres filtres

ex de Pattern à anticiper :
"Je veux [liste/filtrer] des formations [avec/sans/pour] [critère]"

"""