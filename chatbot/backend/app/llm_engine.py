# llm_engine.py

from typing import List, Optional, Dict, Any
import os
import json
import re
import unicodedata
import logging
from difflib import get_close_matches

from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory, ConversationEntityMemory, CombinedMemory
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory, ConversationEntityMemory
from openai import OpenAI
from app.schemas import UserProfile, SessionState
# Chargement des variables d’environnement (clé API OpenAI, etc.)

load_dotenv(dotenv_path="app/.env")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialisation du client OpenAI pour les appels directs (classification d'intention, détection de titre)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class LLMEngine:
    def __init__(self, df_formations, session: Optional[SessionState] = None):
        """
        Initialise le moteur LLM avec embeddings, modèle LLM et base vectorielle.
        Charge les données de formations et prépare la chaîne RAG.
        """
        # Initialisation des embeddings OpenAI pour la vectorisation du texte
        self.embeddings = OpenAIEmbeddings()
        # Modèle ChatOpenAI pour la génération de réponses (ton légèrement aléatoire pour style commercial)
        self.llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)
        # Stockage vectoriel (sera initialisé plus loin)
        self.vector_store = None
        # Chaîne QA conversationnelle (non utilisée directement dans la nouvelle version, remplacée par logique manuelle)
        self.qa_chain = None
        self.session = session or SessionState(user_id="default")
        # Données et mémoire interne
        self.formations_json = {}       # Dictionnaire des formations (titre -> données JSON)
        self.all_documents = []         # Documents LangChain pour chaque formation
        self.titles_list = []           # Liste des titres de formation disponibles (en minuscules)
        self.titles_joined = ""         # Titres joints par sauts de ligne (pour prompt détection)
        #self.current_title = None       # Titre de formation courant (en minuscules) suivi durant la conversation

        # Initialisation de la base RAG (documents + vecteurs)
        self.initialize_rag(df_formations)

    def _df_to_documents(self, df) -> List[Document]:
        """
        Convertit le DataFrame des formations en une liste de Documents LangChain.
        Chaque document contient le contenu texte d'une formation (titre, objectifs, etc.),
        et les métadonnées correspondantes pour le filtrage.
        """
        docs = []
        for _, row in df.iterrows():
            # Construction du contenu textuel combinant les principales rubriques de la formation
            content = (
                f"Formation: {row['titre']}\n"
                f"Objectifs: {', '.join(row['objectifs'])}\n"
                f"Prérequis: {', '.join(row['prerequis'])}\n"
                f"Programme: {', '.join(row['programme'])}\n"
                f"Public: {', '.join(row['public'])}\n"
                f"Lien: {row['lien']}\n"
                f"Durée: {row.get('durée', '')}\n"
                f"Tarif: {row.get('tarif', '')}\n"
                f"Modalité: {row.get('modalité', '')}\n"
            )
            # Création du Document avec contenu et métadonnées
            docs.append(Document(
                page_content=content,
                metadata={
                    "titre": row["titre"],        # Titre exact de la formation
                    "type": "formation",
                    "niveau": row.get("niveau", ""),
                    "modalite": row.get("modalité", ""),
                    "duree": row.get("durée", ""),
                    "tarif": row.get("tarif", ""),
                    "objectifs": ", ".join(row.get("objectifs", [])),
                    "prerequis": ", ".join(row.get("prerequis", [])),
                    "programme": ", ".join(row.get("programme", [])),
                    "public": ", ".join(row.get("public", []))
                }
            ))
        return docs

    def initialize_rag(self, df_formations):
        """
        Initialise le système RAG:
        - Charge les documents de formation.
        - Construit la base vectorielle Chroma des documents.
        - Charge les données JSON complètes des formations pour accès direct.
        """
        # Conversion du DataFrame en Documents et stockage
        documents = self._df_to_documents(df_formations)
        self.all_documents = documents

        # Chargement des données JSON complètes de chaque formation (pour réponses directes)
        for file in os.listdir("./app/content"):
            if file.endswith(".json"):
                with open(os.path.join("./app/content", file), "r", encoding="utf-8") as f:
                    data = json.load(f)
                    titre = data.get("titre", "")
                    if titre:
                        # On stocke la formation avec un indice lowercase pour correspondance simplifiée
                        self.formations_json[titre.lower()] = data

        # Préparation des titres de formations disponibles (pour détection via LLM)
        self.titles_list = list(self.formations_json.keys())  # en minuscules
        self.titles_joined = "\n".join(self.titles_list)

        # Découpage des documents en chunks pour une meilleure vectorisation (limite 1000 caractères, chevauchement 200)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(documents)

        # Initialisation de la base vectorielle Chroma avec persistance locale
        self.vector_store = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            persist_directory="./chroma_db"
        )

        # (Optionnel) Initialisation d'une chaîne de QA conversationnelle standard 
        # Note: Dans cette refonte, on utilisera une approche manuelle pour plus de contrôle
        self.qa_chain = None  # ConversationalRetrievalChain.from_llm(self.llm, self.vector_store.as_retriever(search_kwargs={"k": 15}), return_source_documents=True)

        print("Base de connaissances vectorielle initialisée (%d documents).", len(splits))

    def _normalize_text(self, text: str) -> str:
        """
        Normalise une chaîne de caractères: suppression des accents, conversion en minuscules, et trim des espaces.
        Utile pour la correspondance de textes de façon robuste.
        """
        return re.sub(r'\s+', ' ', unicodedata.normalize('NFD', text.lower()).encode('ascii', 'ignore').decode("utf-8")).strip()

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

    def detect_formation_title(self, question: str) -> str:
        """
        Identifie le titre exact d’une formation mentionnée ou sous-entendue dans la question.
        Utilise un appel LLM en fournissant la liste des titres disponibles.
        Retourne le titre en minuscules si trouvé, ou 'aucun' si rien de correspondant.
        """
        detect_title_prompt = (
            "Tu es un assistant intelligent.\n\n"
            "Ta tâche est de détecter, parmi la liste de titres ci-dessous, le **titre exact** de formation auquel fait référence la question de l'utilisateur.\n"
            "Voici la liste des titres disponibles :\n"
            f"{self.titles_joined}\n\n"
            "Réponds uniquement par l'un de ces titres *exactement* comme il apparaît dans la liste (pas de phrase complète, pas de guillemets). "
            "Si aucun titre ne correspond, réponds simplement : aucun.\n\n"
            f"Question : {question}\n"
        )
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": detect_title_prompt}],
                temperature=0
            )
            detected = response.choices[0].message.content.strip()
            detected = detected.lower()
            print("Formation détectée par le LLM pour ", detected)
            return detected
        except Exception as e:
            print("Erreur lors de la détection du titre de formation:", e)
            return "aucun"

    def _resolve_pronouns(self, question: str) -> str:
        """
        Résout certains pronoms flous dans la question en utilisant le contexte courant (formation mentionnée précédemment).
        Par exemple, remplace "cette formation", "celle-ci", "celui-ci", "l'autre" par le nom de la formation correspondante si connu.
        """
        if not self.session.current_title:
            # Pas de formation courante connue pour donner du contexte
            return question

        resolved_question = question
        titre_display = self.formations_json.get(self.session.current_title, {}).get("titre", self.session.current_title.title())

        # Pronoms à remplacer s'ils sont présents dans la question
        pronoms_flous = ["cette formation", "cette formation-ci", "cette formation là", "celle-ci", "celui-ci", "celle la", "celle-là", "l'autre formation", "l'autre"]
        for pronom in pronoms_flous:
            if pronom.lower() in resolved_question.lower():
                # Remplacement par "la formation XYZ" pour plus de clarté
                resolved_question = re.sub(pronom, f"la formation {titre_display}", resolved_question, flags=re.IGNORECASE)
        if resolved_question != question:
            print("Question résolue (pronoms remplacés) :", resolved_question)
        return resolved_question

    def _is_vague_question(self, question: str) -> bool:
        """
        Détermine si la question utilisateur est vague/ambigüe.
        Une question est considérée comme vague si elle est très courte ou ne contient pas assez de contexte (ex: "et ensuite ?", "comment ?", etc.).
        """
        q = question.strip().lower()
        # Critères simples : longueur très courte OU phrases communes très générales
        if len(q) < 5:
            return True
        vague_phrases = ["et ensuite", "ensuite ?", "et après", "et apres", "comment ?", "pourquoi ?", "et puis"]
        for phrase in vague_phrases:
            if q == phrase or q.endswith(phrase):
                return True
        # Si la question demande une info (prérequis, durée, etc.) sans mentionner la formation et qu'on n'a pas de contexte courant
        if self.session.current_title is None:
            mots_clefs = ["objectif", "objectifs", "prérequis", "prerequis", "programme", "public", "durée", "duree", "tarif", "modalité", "modalite", "certification", "certifiante", "sessions"]
            for mot in mots_clefs:
                if mot in q:
                    # Mot clé présent, pas de formation spécifiée => potentiellement vague ("prérequis" de quoi ?)
                    return True
        return False

    def generate_response(self, question: str, chat_history: List[tuple], session: SessionState, profile: Optional[Any] = None) -> Dict[str, Any]:
        """
        Génère la réponse du chatbot pour une question donnée, en utilisant l'historique de conversation et le profil utilisateur.
        Ce moteur suit les étapes RAG : analyse d'intention, identification de la formation, recherche de contexte, génération de la réponse.
        Il gère aussi les questions vagues en fournissant une clarification ou une reformulation si nécessaire.
        """
        print(f"🎯 User session ID : {session.user_id}")
        print(f"🎯 Current title (avant traitement) : {self.session.current_title}")

        print(f"Question reçue : {question}")
        print(f"Profil utilisateur (recommended_course) : {profile.recommended_course if profile else 'Aucun'}")


        # Prétraitement de la question utilisateur
        question = question.strip()
        # On commence par résoudre les références floues (pronoms) en utilisant la formation courante connue
        question = self._resolve_pronouns(question)
        print(f"Question après résolution des pronoms : {question}")

        # Détection si la question est trop vague
        if self._is_vague_question(question):
            # Cas d'une question vague/ambiguë
            if ("ensuite" in question.lower() or "et après" in question.lower()) and self.session.current_title:
                # Si la question est du type "et ensuite ?" et qu'on a une formation contexte, on reformule automatiquement en question claire
                titre_display = self.formations_json.get(self.session.current_title, {}).get("titre", self.session.current_title.title())
                question = f"Que se passe-t-il après la formation {titre_display} ?"  # on reformule la question
                print("Reformulation automatique de la question vague en : ", question)
                # On continue le processus avec cette question reformulée
            else:
                # Sinon, on génère une demande de clarification sans aller plus loin
                if self.session.current_title:
                    # Si on connaît le contexte d'une formation, on demande une précision sur cette formation
                    titre_display = self.formations_json.get(self.session.current_title, {}).get("titre", self.session.current_title.title())
                    clarification = f"Pouvez-vous préciser ce que vous souhaitez savoir sur la formation {titre_display} ?"
                else:
                    # Pas de contexte, question trop vague de manière générale
                    clarification = "Pouvez-vous préciser ce que vous voulez savoir, s'il vous plaît ?"
                return {"answer": clarification}

        # Étape 1 : Détection de l'intention de l'utilisateur
        intent = self.detect_intent(question)
        print(f"Intention détectée : {intent}")
        known_intents = {
            "liste_formations", "recommandation",
            "info_objectifs", "info_prerequis", "info_programme",
            "info_public", "info_tarif", "info_duree",
            "info_modalite", "info_certification", "info_lieu", "info_prochaine_session"
        }

        # Intents pouvant nécessiter une détection de titre implicite même si ce n'est pas 'recommandation'
       # intents_requérant_titre = {"info_tarif", "info_duree", "info_lieu", "info_certification", "info_modalite", "info_public"}

        if intent in known_intents:
            detected_title = self.detect_formation_title(question)
            print(f"Titre détecté dans une question '{intent}' : {detected_title}")
            if detected_title and detected_title != "aucun":
                if detected_title in self.formations_json:
                    self.session.current_title = detected_title
                else:
                    close = get_close_matches(detected_title, self.titles_list, n=1, cutoff=0.7)
                    if close:
                        self.session.current_title = close[0]
                print("Titre mis à jour pour l’intention info_* :", self.session.current_title)

        
        formation_context = self.formations_json.get(self.session.current_title, {}) if self.session.current_title else {}
        titre_affiche = formation_context.get("titre", "(inconnu)")

        # if not self.session.current_title:
        #     print("❌ Aucun titre valide identifié, impossible de répondre précisément.")
        #     return {"answer": "Je n’ai pas identifié de formation précise. Pouvez-vous reformuler ou préciser le nom de la formation ?" }


        if intent not in known_intents:
            intent = "none"

        # Étape 2 : Identification du titre de formation mentionné ou implicite
        # Par défaut, on conserve la formation actuelle si aucune nouvelle formation n'est détectée
        if not self.session.current_title:
            matched_title = profile.recommended_course.lower() if profile and profile.recommended_course else None
            if matched_title:
                self.session.current_title = matched_title
                print("Initialisation de current_title depuis le profil recommandé :", self.session.current_title)
            else:
                print("Aucune formation recommandée dans le profil.")
        else:
            print("current_title déjà défini :", self.session.current_title)


        # detected_title = self.detect_formation_title(question)
        # print(f"####\\nFormation détectée par LLM : {detected_title}\\n####")

        formation_context = self.formations_json.get(self.session.current_title, {}) if self.session.current_title else {}
        titre_affiche = formation_context.get("titre", "(inconnu)")

        # Étape 3 : Gestion des intentions particulières avec réponse directe ou traitement spécialisé

        # 3.a. Intentions d'information directe sur une formation (objectifs, prérequis, etc.)
        print(f"Formation courante active : {self.session.current_title}")
        rubriques_info = {
            "info_objectifs": {
                "key": "objectifs",
                "format": lambda val, titre: "Les objectifs de la formation " + titre + " sont :\n- " + "\n- ".join(val) if isinstance(val, list) else f"Objectifs : {val}"
            },
            "info_prerequis": {
                "key": "prerequis",
                "format": lambda val, titre: "Les prérequis pour la formation " + titre + " sont :\n- " + "\n- ".join(val) if isinstance(val, list) else f"Prérequis : {val}"
            },
            "info_programme": {
                "key": "programme",
                "format": lambda val, titre: "Le programme de la formation " + titre + " contient :\n- " + "\n- ".join(val) if isinstance(val, list) else f"Programme : {val}"
            },
            "info_public": {
                "key": "public",
                "format": lambda val, titre: "La formation " + titre + " s’adresse à :\n- " + "\n- ".join(val) if isinstance(val, list) else f"Public visé : {val}"
            },
            "info_tarif": {
                "key": "tarif",
                "format": lambda val, titre: f"Le tarif de la formation {titre} est de {val}."
            },
            "info_duree": {
                "key": "durée",
                "format": lambda val, titre: f"La durée de la formation {titre} est de {val}."
            },
            "info_modalite": {
                "key": "modalité",
                "format": lambda val, titre: f"La formation {titre} est proposée en modalité : {val}."
            },
            "info_certification": {
                "key": "certifiant",
                "format": lambda val, titre: f"La formation {titre} est " + ("certifiante." if val=="Oui" else "non certifiante.")
            },
            "info_lieu": {
                "key": "lieu",
                "format": lambda val, titre: f"Le lieu de la formation {titre} est {val}."
            },
            "info_prochaine_session": {
                "key": "prochaines_sessions",
                "format": lambda val, titre: "Les prochaines sessions prévues sont :\n- " + "\n- ".join(val) if isinstance(val, list) else f"Prochaines sessions : {val}"
            }
        }

        
        # On s'assure que intent soit dans les valeurs attendues (sinon on le traitera comme 'none')
        if intent == "info_tarif" and self.session.current_title:
            rubrique = "tarif"
            valeur = formation_context.get(rubrique, "(non renseigné)")
            if any(mot in question.lower() for mot in ["cher", "coûteux", "prix élevé"]):
                justification = formation_context.get("objectifs", "")
                return {
                    "answer": f"Le tarif de la formation {titre_affiche} est de {valeur}. "
                            f"Ce tarif reflète les compétences acquises, notamment : "
                            f"{', '.join(justification) if isinstance(justification, list) else justification}"
                }
            else:
                return {"answer": f"Le tarif de la formation {titre_affiche} est de {valeur}."}

        if intent in rubriques_info and self.session.current_title:
            # Si l'utilisateur demande une information spécifique sur la formation courante
            rubrique = rubriques_info[intent]["key"]
            valeur = formation_context.get(rubrique, "(non renseigné)")
            reponse_directe = rubriques_info[intent]["format"](valeur, titre_affiche)
            return {"answer": reponse_directe}

        # 3.b. Intention de lister toutes les formations
        if intent == "liste_formations":
            titres = [data["titre"] for _, data in self.formations_json.items()]
            liste = "- " + "\n- ".join(titres) if titres else "(Aucune formation disponible)"
            return {"answer": "Voici la liste complète des formations disponibles :\n\n" + liste}

        # 3.c. Intention de recommandation (expliquer ou détailler la formation recommandée)
        if intent == "recommandation" and self.session.current_title:
            # Contexte basé sur la formation actuelle et le profil pour motiver la recommandation
            objectifs = formation_context.get("objectifs", [])
            public = formation_context.get("public", [])
            prerequis = formation_context.get("prerequis", [])
            objectifs_text = "- " + "\n- ".join(objectifs) if isinstance(objectifs, list) else str(objectifs)
            public_text = "- " + "\n- ".join(public) if isinstance(public, list) else str(public)
            prerequis_text = "- " + "\n- ".join(prerequis) if isinstance(prerequis, list) else str(prerequis)
            context_parts = [
                f"voici les informations générales de la formation **{titre_affiche}**:\n\n",
                f"- Objectifs principaux :\n{objectifs_text}\n",
                f"- Public visé :\n{public_text}\n",
                f"- Prérequis recommandés :\n{prerequis_text}"
            ]
            # On formule la question de l'utilisateur dans ce contexte (s'il y a une question précise, sinon question générale)
            #user_question = question if intent == "recommandation" and question.lower().startswith("pourquoi") else "Pourquoi cette formation est-elle recommandée ?"
            final_prompt = (
                f"Tu es un assistant virtuel spécialisé dans les formations professionnelles. "
                f"recommande cette formation **{titre_affiche}** et commence par le nom de la formation.\n\n"
                f"Contexte :\n"
                + "\n\n".join(context_parts)
                + f"\n\nQuestion de l'utilisateur : {question}"
            )

            # Appel du LLM sur ce prompt pour générer une réponse justifiant la recommandation
            try:
                result = self.llm.predict(final_prompt)
            except Exception as e:
                print("Erreur lors de la génération de réponse recommandation: ", e)
                return {"answer": "Désolé, je ne parviens pas à expliquer cette recommandation pour le moment."}
            return {"answer": result}


        if not self.session.current_title:
            print("❌ Aucun titre actif détecté, retour d'une réponse générique.")
            return {"answer": "Je n’ai pas identifié de formation précise. Pouvez-vous préciser le nom de la formation qui vous intéresse ?" }

        # Étape 4 : Cas général (intent 'none' ou question nécessitant recherche dans la base de connaissances)
        # On va utiliser le modèle RAG pour formuler une réponse en s'appuyant sur les documents.

        # a. Préparation de la mémoire conversationnelle (historique + entités)
        # On utilise ConversationBufferMemory et ConversationEntityMemory de LangChain pour reconstituer l'historique en texte et extraire les entités mentionnées.
        buffer_memory = ConversationBufferMemory(
            memory_key="history",
            human_prefix="Utilisateur",
            ai_prefix="Assistant",
            return_messages=False
        )
        entity_memory = ConversationEntityMemory(
            llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0),
            memory_key="entities",
            input_key="question"
        )
        # On alimente la mémoire avec l'historique de la conversation passée (paire utilisateur/assistant)
        for user_msg, assistant_msg in chat_history:
            buffer_memory.chat_memory.add_user_message(user_msg)
            buffer_memory.chat_memory.add_ai_message(assistant_msg)
            try:
                # Mise à jour de la mémoire d'entités à chaque échange
                entity_memory.save_context({"question": user_msg}, {"output": assistant_msg})
                history_text = buffer_memory.load_memory_variables({}).get("history", "")
                entities_text = entity_memory.load_memory_variables({"question": question}).get("entities", "")


            except Exception as e:
                print("Extraction entité échouée sur ", user_msg, e)
        # Récupération des contenus formatés de l'historique et des entités
        history_text = buffer_memory.load_memory_variables({}).get("history", "")
        entities_text = entity_memory.load_memory_variables({"question": question}).get("entities", "")
        # Si aucune entité identifiée, on met une valeur par défaut pour le prompt
        if not entities_text or entities_text.strip().lower() == "none":
            entities_text = "Aucune"

        # b. Recherche des documents pertinents via Chroma
        # On filtre par formation courante si connue, sinon on cherche globalement
        filter_criteria = {"titre": titre_affiche} if self.session.current_title and titre_affiche else None
        print(f"Filtre de recherche : {filter_criteria}")
        

        try:
            # Requête de recherche vectorielle (on utilise la question non modifiée pour la similarité)
            docs = self.vector_store.similarity_search(question, k=6, filter=filter_criteria)
            # Supprimer les doublons de contenu
            seen = set()
            unique_docs = []
            for doc in docs:
                title = doc.metadata.get("titre", "")
                if title not in seen:
                    seen.add(title)
                    unique_docs.append(doc)

        except Exception as e:
            print("Erreur lors de la recherche vectorielle : ", e)
            docs = []
        print(f"Documents trouvés : {len(docs)}")
        # Préparation du contexte textuel à partir des documents trouvés
        context_segments = []
        for doc in docs:
            titre_doc = doc.metadata.get("titre", "Formation inconnue")
            extrait = doc.page_content.strip()
            context_segments.append(f"Formation: {titre_doc}\n{extrait}")
        context_text = "\n\n".join(context_segments).strip()
        if not context_text:
            context_text = "(Aucun document pertinent trouvé)"

        # c. Construction du prompt de génération final avec contexte, historique et entités
        final_prompt_template = PromptTemplate(
            input_variables=["profil", "history", "entities", "context", "question"],
            template=(
                "Tu es un assistant virtuel spécialisé dans les formations professionnelles. "
                "Tu aides l'utilisateur en répondant de manière claire, utile et engageante à ses questions.\n\n"
                "{profil}\n"
                "Historique de la conversation :\n{history}\n"
                "Entités mentionnées dans le contexte :\n{entities}\n"
                "Contexte documentaire :\n{context}\n"
                "Question de l'utilisateur : {question}\n\n"
                "Consignes de réponse :\n"
                "- Appuie-toi sur le **contexte fourni** (documents et historique) pour formuler ta réponse.\n"
                "- **N'invente pas** d'informations qui ne figurent pas dans le contexte.\n"
                "- Si la question est ambiguë ou manque de contexte, propose une reformulation polie pour clarification.\n"
                "- Adopte un ton professionnel et accueillant (style commercial léger) pour mettre en valeur la formation.\n"
                "- Réponds en français de manière concise et compréhensible.\n\n"
                "Réponse :"
            )
        )
        # Intégration du profil utilisateur s'il est fourni (objectif, niveau, compétences)
        profil_text = ""
        if profile:
            profil_text = (f"Profil de l'utilisateur : "
                           f"Objectif={profile.objective}, Niveau={profile.level}, Compétences={profile.knowledge}")
        # Formatage final du prompt avec toutes les informations
        final_prompt = final_prompt_template.format(
            profil=profil_text,
            history=history_text or "(aucun)",
            entities=entities_text or "Aucune",
            context=context_text or "(vide)",
            question=question
        )
        print("Prompt final envoyé au LLM:\n", final_prompt)

        # d. Génération de la réponse finale à l'aide du modèle LLM (OpenAI GPT-3.5 Turbo)
        try:
            answer = self.llm.predict(final_prompt)
        except Exception as e:
            print("Erreur lors de la génération de la réponse finale: ", e)
            answer = "Désolé, je rencontre des difficultés pour répondre à votre question pour le moment."

        return {"answer": answer}
