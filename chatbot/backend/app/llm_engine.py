import os
import json
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from app.schemas import UserProfile
import re
from functools import lru_cache
import logging
import time

# Import centralisé des utilitaires
import app.globals as globs
from app.utils import search_and_format_courses

load_dotenv(dotenv_path="app/.env")
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))


class LLMEngine:
    def __init__(self,
                 content_dir: str = "app/content",
                 enable_rncp: bool = None,
                ):

        self.logger = logging.getLogger(__name__)
        self.logger.info("Initialisation de LLMEngine")

        self.content_dir = Path(content_dir)
        
        if enable_rncp is None:
            import app.globals as globs
            enable_rncp = globs.enable_rncp
        self.enable_rncp = enable_rncp
        self.logger.info(
            f"Accès aux formations RNCP : {'activé' if enable_rncp else 'désactivé'}"
        )
        

    
    
    # Modification de la méthode _build_prompt dans llm_engine.py

    # Il faut également s'assurer que le prompt du LLM Engine prend bien en compte le template de formations

    def _build_prompt(self, profile: dict, chat_history: list[dict]) -> list[dict]:
        """
        Construit la liste de messages pour l'API de complétion de chat, incluant :
        - Un message système avec les instructions pour le conseiller
        - Le modèle de formations en tant qu'indication système (le cas échéant)
        - Les 20 derniers messages de la conversation (10 tours)
        """
        messages: list[dict] = []

        # 1) Message système principal
        system_content = (
            "Vous êtes un conseiller professionnel expérimenté, orienté résultats et service client, "
            "bienveillant et proactif."
        )
        messages.append({"role": "system", "content": system_content})

        # 2) Modèle de formations facultatif comme indication système
        if profile and isinstance(profile, dict) and "formations_template" in profile:
            messages.append({
                "role": "system",
                "content": profile["formations_template"]
            })
            messages.append({
                "role": "system",
                "content": (
                    "À partir des résultats ci-dessus, sélectionnez la formation la plus pertinente pour l'utilisateur "
                    "et recomandez-la dans le champ `recommended_course` du JSON. "
                    "Incluez également dans `answer` une réponse naturelle qui présente cette formation et ses caractéristiques clés."
                )
            })

        # 3) Ajout des 20 derniers messages (10 tours) de l'historique
        if chat_history:
            last_msgs = [m for m in chat_history if 'role' in m and 'content' in m]
            for msg in last_msgs[-20:]:
                messages.append({"role": msg['role'], "content": msg['content']})

        return messages
        
    def detect_intent(self, question: str, chat_history=None) -> str:
        """Détecte l'intention de l'utilisateur à partir de la question avec assainissement des entrées."""
        
        # Traiter l'historique du chat pour le contexte
        context = ""
        if chat_history and len(chat_history) >= 2:
            last_messages = chat_history[-2:]
            for msg in last_messages:
                if isinstance(msg, dict) and 'content' in msg:
                    role = msg.get('role', 'unknown')
                    content = msg.get('content', '')
                    context += f"{role}: {content}\n"
        
        # Construire le prompt de détection d'intention
        prompt = f"""
            Tu es un classificateur d'intentions pour un chatbot de formation.
            Ta mission : renvoyer **exactement** le libellé d'intention (en minuscules)
            le plus pertinent parmi la liste ci-dessous.

            ────────────────────────────────────────────────────────────────
            CONTEXT : 3 DERNIERS MESSAGES UTILISATEUR/BOT (si dispo)
            {context if context.strip() else '[Pas de contexte précédent]'}
            ────────────────────────────────────────────────────────────────

            INTENTIONS AUTORISÉES
            ─────────────────────
            - liste_internes : l'utilisateur veut UNIQUEMENT **la liste des formations proposées par Beyond Expertise**  
            Ex. : "Quelles formations proposez-vous ?", "vos formations internes", "catalogue Beyond".
            
            - liste_externes : l'utilisateur veut **la liste des formations externes / RNCP**  
            Ex. : "Quelles sont les formations externes ?", "les formations RNCP ?", "formations que vous ne commercialisez pas".

            *si l'utilisateur demande une liste des formations choisis l'une des deux catégories ci-dessus et privilège liste_internes.)*

            - recommandation : l'utilisateur cherche **UNE** formation adaptée à un besoin/profil.

            - recherche_filtrée : l'utilisateur demande **une liste filtrée** contenant **Il faut au moins un critère de filtrage** (sur site, à distance, certifiante, etc.), ex : quels sont les formations certifiantes et à distance.

            - info_objectifs / info_prerequis / info_programme / info_public / info_tarif
            / info_financement / info_duree / info_modalite / info_lieu
            / info_certification / info_prochaine_session  
            → questions ciblant un champ précis d'une formation.

            - comparaison : l'utilisateur veut comparer deux formations (mots-clé « vs », « versus », « différence »…).

            - fallback : la question concerne la formation mais n'entre dans **aucune** catégorie ci-dessus.

            - none : hors-sujet (pas de lien avec la formation).

            ────────────────────────────────────────────────────────────────
            RÈGLES DE CHOIX
            1. S'il demande **toutes** les formations :
            • si la question contient « Beyond », « vos formations », etc. → liste_internes  
            • si elle dit « externes », « RNCP », « pas chez vous » → liste_externes  
            • sinon, s'il y a un filtre → recherche_filtrée  
            • sinon → liste_internes (par défaut privilégier l'interne).
            ne confonds pas la demandes des liste de métiers, prérequis, etc . avec une liste_internes ou liste_externes.
            pour l'intent recherche_filtrée, il faut avoir impérativement un filtre dans la question de l'utilisateur
            2. Pour une information précise sur une formation (tarif, durée, etc.)
            → choisir le préfixe **info_*** approprié.

            3. Ne renvoie **que** le nom d'intention, sans commentaire, sans ponctuation.
            ────────────────────────────────────────────────────────────────

            Question : {question}
            """
        
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            intent = response.choices[0].message.content.strip().lower()
            
            self.logger.info(f"Intention détectée: {intent}")
            return intent
        except Exception as e:
            self.logger.error(f"Erreur lors de la détection d'intention: {str(e)}")
            return "fallback"
            
  



    def generate_response(
        self,
        prompt: str,
        chat_history: list[dict] = None,
        profile: UserProfile | dict = None,
        session=None,
        *,
        course_source: str = "internal",
        is_ready_prompt: bool = True
    ) -> dict:
        """
        Génère une réponse structurée en JSON.
        - Si is_ready_prompt=False, intègre d'abord le template de formations.
        - Injecte toujours les 20 derniers messages pour le contexte.
        - Ajoute une mention Beyond Expertise, et un rappel RNCP si la source est externe.
        """
        # --- 1) System message #1 : rôle, ton et mention Beyond Expertise ---
        base_role = (
            "Vous êtes un conseiller professionnel expérimenté, orienté résultats et service client, "
            "bienveillant et proactif. "
            "Vous travaillez pour l'entreprise Beyond Expertise, Entreprise de formations et solutions Data et IA."
        )
        if course_source != "internal":
            # ajout d’un rappel pour les formations RNCP externes
            base_role += (
                " Les formations listées ci-dessous proviennent du RNCP et ne sont pas commercialisées par Beyond Expertise."
            )
        role_msg = {"role": "system", "content": base_role}

        # --- 2) System message #2 : contrat JSON strict ---
        json_schema_msg = {
            "role": "system",
            "content": (
                "Vous devez répondre STRICTEMENT au format JSON suivant :\n"
                "{\n"
                "  \"answer\": string,\n"
                "  \"recommended_course\": {\n"
                "      \"titre\": string,\n"
                "      \"objectifs\": [string],\n"
                "      \"public_cible\": [string],\n"
                "      \"lieu\" : [string],\n"
                "      \"duree\": string,\n"
                "      \"tarif\": string|null,\n"
                "      \"certifiant\": boolean,\n"
                "      \"_source\": string\n"
                "  } | null,\n"
                "  \"next_action\": string\n"
                "}\n"
                "Ne retournez **que** ces clés, sans autre texte."
            )
        }

        # --- 3) Préparation de l'historique (derniers 20 messages) ---
        history_msgs = []
        if chat_history:
            filtered = [m for m in chat_history if 'role' in m and 'content' in m]
            history_msgs = filtered[-20:]

        # --- 4) Assemblage des messages ---
        messages = [role_msg, json_schema_msg] + history_msgs

        if is_ready_prompt:
            # 5a) Ready prompt : on attend un prompt déjà complet
            messages.append({"role": "user", "content": prompt})
        else:
            # 5b) Build prompt : on génère le template de formations
            formatted_text, _ = search_and_format_courses(prompt, k=3)
            prof = profile.dict() if hasattr(profile, 'dict') else dict(profile or {})
            prof["formations_template"] = formatted_text
            messages.extend(self._build_prompt(prof, chat_history or []))
            messages.append({"role": "user", "content": prompt})

        # --- 6) Appel API OpenAI ---
        try:
            response = client.chat.completions.create(
                model='gpt-3.5-turbo-1106',
                messages=messages,
                temperature=0.7,
                response_format={"type": "json_object"},
                max_tokens=1000
            )
            raw = response.choices[0].message.content
            print("🧠 RÉPONSE BRUTE DE GPT:\n", raw)
            parsed = json.loads(raw)

            # --- 7) Validation minimale du JSON ---
            if not isinstance(parsed.get('answer'), str):
                parsed['answer'] = (
                    "Désolé, je n'ai pas pu générer une réponse structurée. "
                    "Pourriez-vous reformuler votre question ?"
                )
            parsed.setdefault('next_action', 'follow_up')
            parsed.setdefault('intent', 'fallback')

            # --- 8) Avertissement RNCP post‐processing (optionnel) ---
            #rec = parsed.get('recommended_course')
            # if course_source != "internal" and isinstance(rec, dict):
            #     parsed['answer'] = external_warning(rec.get('titre', 'Cette formation')) + "\n\n" + parsed['answer']

            return parsed

        except json.JSONDecodeError:
            self.logger.error("Erreur de parsing JSON de l'API OpenAI")
            return {
                "answer": (
                    "Je n'ai pas pu générer une réponse structurée. "
                    "Pourriez-vous reformuler votre question ?"
                ),
                "recommended_course": None,
                "next_action": "fallback",
                "intent": "fallback"
            }
        except Exception as e:
            self.logger.error("Erreur génération réponse: %s", str(e), exc_info=True)
            return {
                "answer": "Désolé, une erreur technique est survenue. Veuillez réessayer.",
                "recommended_course": None,
                "next_action": "error",
                "intent": "error"
            }
