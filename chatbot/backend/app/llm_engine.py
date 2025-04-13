# llm_engine.py

from typing import List, Optional, Dict, Any
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from openai import OpenAI
from app.schemas import UserProfile
import pandas as pd
import os
import json
import re
import unicodedata
from difflib import get_close_matches


load_dotenv(dotenv_path="app/.env")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class LLMEngine:
    def __init__(self, df_formations: pd.DataFrame):
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)
        self.vector_store = None
        self.qa_chain = None
        self.initialize_rag(df_formations)

    def _df_to_documents(self, df: pd.DataFrame) -> List[Document]:
        docs = []
        for _, row in df.iterrows():
            content = f"""
            Formation: {row['titre']}
            Objectifs: {', '.join(row['objectifs'])}
            Prérequis: {', '.join(row['prerequis'])}
            Programme: {', '.join(row['programme'])}
            Public: {', '.join(row['public'])}
            Lien: {row['lien']}
            Durée: {row.get('durée', '')}
            Tarif: {row.get('tarif', '')}
            Modalité: {row.get('modalité', '')}
            """
            docs.append(Document(
                page_content=content,
                metadata={
                    "titre": row["titre"],
                    "type": "formation",
                    "niveau": row.get("niveau", ""),
                    "modalite": row.get("modalité", ""),
                    "duree": row.get("durée", ""),
                    "tarif": row.get("tarif", ""),
                    "objectifs": ", ".join(row.get("objectifs", [])),
                    "prerequis": ", ".join(row.get("prerequis", [])),
                    "programme": ", ".join(row.get("programme", [])),
                    "public": ", ".join(row.get("public", [])),
                }
            ))
        return docs

    def initialize_rag(self, df_formations: pd.DataFrame):
        documents = self._df_to_documents(df_formations)
        self.all_documents = documents
        self.formations_json = {}
        for file in os.listdir("./app/content"):
            if file.endswith(".json"):
                with open(os.path.join("./app/content", file), "r", encoding="utf-8") as f:
                    data = json.load(f)
                    titre = data.get("titre", "").lower()
                    self.formations_json[titre] = data

        splits = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(documents)
        self.vector_store = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            persist_directory="./chroma_db"
        )
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            self.llm,
            self.vector_store.as_retriever(search_kwargs={"k": 15}),
            return_source_documents=True
        )

    def detect_intent(self, question: str) -> str:
        prompt = f"""
            Tu es un classificateur d'intentions. Ton rôle est d'identifier l'intention de l'utilisateur à partir de sa question.

            Les intentions possibles sont :

            - liste_formations
            - recommandation
            - info_objectifs
            - info_prerequis
            - info_programme
            - info_public
            - info_tarif
            - info_duree
            - info_modalite
            - info_certification
            - info_lieu
            - info_prochaine_session
            - none

            Ne réponds qu'avec une seule de ces intentions, sans justification.

            Question : {question}
            """

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip()

    def normalize_text(self, text: str) -> str:
        return re.sub(r'\s+', ' ', unicodedata.normalize('NFD', text.lower()).encode('ascii', 'ignore').decode("utf-8")).strip()

    def generate_response(self, question: str, chat_history: List[tuple], profile: Optional[UserProfile] = None) -> Dict[str, Any]:
        if not self.qa_chain:
            return {"answer": "Système non initialisé", "sources": []}

        intent = self.detect_intent(question)
        print(f"####\nIntent détecté : {intent}\n####")

        if intent == "none":
            return {
                "answer": "Je suis un assistant spécialisé dans les formations. Vous pouvez me poser des questions sur les programmes, durées, tarifs, modalités, ou demander une recommandation personnalisée !",
                "sources": []
            }

        matched_title = profile.recommended_course.lower() if profile and profile.recommended_course else None
        self.current_title = getattr(self, "current_title", matched_title)

        titles = list(self.formations_json.keys())
        joined_titles = "\n".join(titles)
        detect_title_prompt = f"""
            Tu es un assistant intelligent.

            Ta tâche est de détecter, parmi la liste ci-dessous, le **titre exact** de formation mentionné ou sous-entendu dans la question de l'utilisateur.

            Voici la liste des titres disponibles :
            {joined_titles}

            Réponds uniquement par l'un de ces titres, exactement comme il apparaît dans la liste.
            - **N'ajoute rien** (pas de phrase, pas de guillemets, pas de ponctuation).
            - **Ne reformule pas** le titre.
            - Si aucun ne correspond, réponds simplement : aucun

            Question : {question}
            """


        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": detect_title_prompt}]
        )
        detected_title = response.choices[0].message.content.strip().lower()
        print(f"####\nFormation détectée par LLM : {detected_title}\n####")

        # Tenter de corriger la détection automatique
        if detected_title and detected_title != "aucun":
            # Filtrage simple pour éviter les phrases entières
            if any(titre in detected_title for titre in self.formations_json.keys()):
                close = get_close_matches(detected_title.lower(), self.formations_json.keys(), n=1, cutoff=0.7)
                if close:
                    self.current_title = close[0]
                    print(f"✅ Titre de formation mis à jour : {self.current_title}")
                else:
                    print(f"⚠️ Aucun titre proche trouvé pour : '{detected_title}'")
            else:
                print(f"❌ Titre détecté non conforme : '{detected_title}' (ignoré)")
        else:
            print("❌ Aucun titre détecté par le LLM.")

        print(f"#####\n####\n current title\n{self.current_title}\n#####\n####")
       

        formation_context = self.formations_json.get(self.current_title, {}) if self.current_title else {}
        titre_affiche = self.current_title.title() if self.current_title else "(inconnu)"

        rubriques_info = {
            "info_objectifs": {
                "key": "objectifs",
                "phrase": lambda val, titre: f"Les objectifs de la formation {titre} sont :\n- " + "\n- ".join(val) if isinstance(val, list) else f"Objectifs : {val}"
            },
            "info_prerequis": {
                "key": "prerequis",
                "phrase": lambda val, titre: f"Les prérequis pour la formation {titre} sont :\n- " + "\n- ".join(val) if isinstance(val, list) else f"Prérequis : {val}"
            },
            "info_programme": {
                "key": "programme",
                "phrase": lambda val, titre: f"Le programme de la formation {titre} contient :\n- " + "\n- ".join(val) if isinstance(val, list) else f"Programme : {val}"
            },
            "info_public": {
                "key": "public",
                "phrase": lambda val, titre: f"La formation {titre} s’adresse à :\n- " + "\n- ".join(val) if isinstance(val, list) else f"Public visé : {val}"
            },
            "info_tarif": {
                "key": "tarif",
                "phrase": lambda val, titre: f"Le tarif de la formation {titre} est de {val}."
            },
            "info_duree": {
                "key": "durée",
                "phrase": lambda val, titre: f"La durée de la formation {titre} est de {val}."
            },
            "info_modalite": {
                "key": "modalite",
                "phrase": lambda val, titre: f"La formation {titre} est proposée en modalité : {val}."
            },
            "info_certification": {
                "key": "certifiante",
                "phrase": lambda val, titre: f"La formation {titre} est {'certifiante' if val else 'non certifiante'}."
            },
            "info_lieu": {
                "key": "lieu",
                "phrase": lambda val, titre: f"Le lieu de la formation {titre} est {val}."
            },
            "info_prochaine_session": {
                "key": "prochaines_sessions",
                "phrase": lambda val, titre: f"Les prochaines sessions prévues sont :\n- " + "\n- ".join(val) if isinstance(val, list) else f"Sessions prévues : {val}"
            },
        }

        if intent in rubriques_info and self.current_title:
            rubrique_data = rubriques_info[intent]
            rubrique = rubrique_data["key"]
            print(f"Rubrique détectée : {rubrique} → Clé JSON : {rubrique}")
            valeur = formation_context.get(rubrique, "(non renseigné)")
            print(f"Valeur : {valeur}")
            reponse_directe = rubrique_data["phrase"](valeur, titre_affiche)
            return {"answer": reponse_directe, "sources": [titre_affiche]}

        elif intent == "liste_formations":
            titres = [doc.metadata["titre"] for doc in self.all_documents]
            return {
                "answer": "Voici la liste complète des formations disponibles :\n\n" + "\n- ".join(titres),
                "sources": titres
            }

        elif intent == "recommandation" and self.current_title:
            objectifs = formation_context.get("objectifs", [])
            public = formation_context.get("public", [])
            prerequis = formation_context.get("prerequis", [])

            objectifs_text = "- " + "\n- ".join(objectifs) if isinstance(objectifs, list) else objectifs
            public_text = "- " + "\n- ".join(public) if isinstance(public, list) else public
            prerequis_text = "- " + "\n- ".join(prerequis) if isinstance(prerequis, list) else prerequis

            context_parts = [f"Je vous recommande la formation **{titre_affiche}** pour les raisons suivantes :\n\n",
                            f"- Objectifs principaux :\n{objectifs_text}\n",
                            f"- Public visé :\n{public_text}\n",
                            f"- Prérequis recommandés :\n{prerequis_text}"]

        else:
            context_parts = [f"Profil : {profile.objective}, niveau {profile.level}, compétences : {profile.knowledge}"] if profile else []

        print(f"#####\n####\n current title\n{self.current_title}\n#####\n####")
        final_prompt = "\n\n".join(context_parts) + f"\n\nQuestion : {question}"
        result = self.qa_chain.invoke({"question": final_prompt, "chat_history": chat_history})
        sources = list(set(doc.metadata["titre"] for doc in result['source_documents']))
        return {"answer": result['answer'], "sources": sources}







"""
A faire : 
enrichir la logique pour détecter d’autres intentions comme 
« contact », « support »,"lien de formations" ou « aide générale » ?
"""






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


