"""
llm_driven_counselor.py
-----------------------
🎯 CONSEILLER UNIFIÉ PILOTÉ PAR LLM
Version où le LLM contrôle le flux de conversation
"""
import re
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import datetime
from pathlib import Path

# Imports des modules
from app.formation_search import FormationSearch
from app.mistral_client import MistralChat
from app.intent_classifier import IntentClassifier


import app.globals as globs

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("llm_driven_counselor")

@dataclass
class UserContext:
    """Contexte utilisateur simplifié."""
    # Profil (toujours pré-rempli)
    nom: str = ""
    age: str = "29"
    objectif: str = ""
    competences: List[str] = field(default_factory=lambda: ["Python", "DATA", "ETL", "SQL", "SCALA", "Excel"])
    
    # État conversation
    current_formation: Optional[Dict] = None
    search_results: List = field(default_factory=list)
    formations_vues: List[Dict] = field(default_factory=list)
    
    # Historique complet pour le LLM
    conversation_history: List[Dict] = field(default_factory=list)
    interactions: int = 0


class LLMDrivenCounselor:
    """
    Conseiller intelligent piloté par LLM avec enrichissement par intents.
    """

    def __init__(self, user_profile: Optional[Dict[str, Any]] = None):
        # Composants core

        #globs.formation_search = FormationSearch([r"content\formations_internes.json", r"content\rncp\rncp.json"])
        self.formations = globs.formation_search#FormationSearch([r"content\formations_internes.json", r"content\rncp\rncp.json"])
        self.llm = MistralChat()
        globs.intent_classifier = IntentClassifier()
        self.intent_classifier = globs.intent_classifier
        
        # Contexte utilisateur avec profil pré-rempli
        if user_profile:
            self.ctx = UserContext(
                nom=user_profile.get("nom", "Utilisateur"),
                age=user_profile.get("age", "30"),
                situation=user_profile.get("situation", "en recherche"),
                objectif=user_profile.get("objectif", "évoluer professionnellement"),
                competences=user_profile.get("competences", [])
            )
        else:
            # Profil par défaut
            self.ctx = UserContext()
        
        # Mapping intentions -> instructions pour le LLM
        self.intent_instructions = {
            "greeting": "L'utilisateur te salue. Sois chaleureux et propose ton aide.",
            
            "search_formation": "L'utilisateur cherche une formation. Utilise les résultats de recherche fournis pour l'aider.",
            
            "formation_select": "L'utilisateur veut sélectionner une formation. Guide-le dans son choix.",
            "formation_details_objectives": "L'utilisateur s'intéresse aux objectifs de la formation. Détaille-les.",
            "formation_details_public": "L'utilisateur veut savoir à qui s'adresse la formation. Explique le public cible.",
            "formation_details_duration": "L'utilisateur demande la durée. Donne cette information clairement.",
            "formation_details_price": "L'utilisateur s'intéresse au prix. Mentionne aussi les financements possibles.",
            "formation_details_location": "L'utilisateur demande où se passe la formation. Précise lieu et modalités.",
            "formation_details_inscription": "L'utilisateur veut s'inscrire. Guide-le dans les étapes.",
            "info_certif": "L'utilisateur s'intéresse à la certification. Explique la valeur du diplôme.",
            "info_prerequests": "L'utilisateur demande les prérequis. Rassure-le si possible.",
            
            "advice_reconversion": "L'utilisateur cherche des conseils pour sa reconversion. Sois encourageant et pratique.",
            "advice_cv": "L'utilisateur veut améliorer son CV. Donne des conseils concrets.",
            "advice_interview": "L'utilisateur prépare un entretien. Aide-le avec des tips pratiques.",
            "advice_motivation_letter": "L'utilisateur rédige une lettre de motivation. Guide-le efficacement.",
            "advice_job_search": "L'utilisateur cherche un emploi. Propose des stratégies.",
            "advice_skills_assessment": "L'utilisateur s'interroge sur ses compétences. Aide-le à les identifier.",
            "advice_financing": "L'utilisateur cherche à financer sa formation. Explique les options.",
            "advice_entrepreneurship": "L'utilisateur veut créer son entreprise. Donne les étapes clés.",
            
            "job_info": "L'utilisateur s'informe sur un métier. Donne des infos pertinentes.",
            "sector_info": "L'utilisateur explore un secteur. Présente les opportunités.",
            
            "help": "L'utilisateur a besoin d'aide. Clarifie ce que tu peux faire.",
            "unclear": "Le message n'est pas clair. Demande des précisions avec bienveillance.",
            "other": "Réponds de manière utile selon le contexte."
        }
        self._search_context = {
            "awaiting_confirmation": False,
            "pending_query": "",
            "show_results": False
        }

        # Initialiser l'historique avec le contexte utilisateur
        self._init_conversation_history()
        
        logger.info("🚀 LLM-Driven Counselor initialisé")
    
   
    def set_user_profile_from_pydantic(self, profile):
        """
        ✅ CLEAN: Just set profile data, no history management
        """
        # Store profile data directly
        self.ctx.nom = profile.name
        self.ctx.objectif = profile.objective
        self.ctx.age = "29"  
        
        # Handle knowledge/competences simply
        if profile.knowledge and profile.knowledge.strip():
            competences_raw = profile.knowledge.replace(',', ' ').replace(';', ' ').split()
            self.ctx.competences = [comp.strip() for comp in competences_raw if comp.strip()]
        else:
            self.ctx.competences = ["Motivation"]
        
        print(f"✅ PROFILE SET: {self.ctx.nom} | {self.ctx.objectif} | {self.ctx.competences}")
   

    def _init_conversation_history(self):
        """
        ✅ CLEAN: Simplified conversation history without situation
        """
        competences_text = ', '.join(self.ctx.competences) if self.ctx.competences else "motivation"
        
        self.ctx.conversation_history = [
            {
                "role": "system", 
                "content": (
                    "Tu es un conseiller en formation professionnelle expert et bienveillant de Beyond Expertise. "
                    "Tu aides les personnes dans leur orientation, reconversion et recherche de formation. "
                    "RÈGLE ABSOLUE : Réponds TOUJOURS en 50-90 mots MAXIMUM. Sois concis, direct et utile. "
                    "Utilise des emojis pour rendre la conversation plus chaleureuse."
                )
            },
            # ✅ SIMPLIFIED: Just essential profile info
            {"role": "assistant", "content": "Bonjour ! Je suis votre conseiller Beyond Expertise. Comment vous appelez-vous ?"},
            {"role": "user", "content": f"Je m'appelle {self.ctx.nom}"},
            {"role": "assistant", "content": "Parfait ! Quelles sont vos compétences principales ?"},
            {"role": "user", "content": f"J'ai des compétences en {competences_text}"},
            {"role": "assistant", "content": "Excellent ! Quel est votre objectif professionnel ?"},
            {"role": "user", "content": f"Je veux {self.ctx.objectif}"},
            {"role": "assistant", "content": f"Super projet ! Je vais vous aider pour {self.ctx.objectif}. Comment puis-je vous aider ?"}
        ]
    
    def _extract_search_query(self, text: str, entities: dict) -> str:
        """
        Extrait la query réelle de recherche utilisateur :
        - Filtrage amont des mots génériques (formation, fiche, cours…)
        - Nettoyage avec preprocess_text()
        - Fallback vers l’objectif utilisateur si la query est vide
        """
        # 1. Vérifie les entités extraites
        if entities.get("domain"):
            return entities["domain"].replace("_", " ").strip()

        # 2. Mots à ignorer dès le départ
        mots_vides = {"formation", "formations", "fiche", "fiches", "cours", "module", "modules", "programme", "programmes", "domaine", "intitulé", "trouver", "cherche", "cherche une", "adaptée", "bonjour", "salut", "aide", "recherche", "recherches", "recherche de", "recherches de", "recherche une", "recherches une", "recherche des", "recherches des", "besoin", "besoins", "besoin d'aide", "besoins d'aide", "aide à", "aide pour", "aide à trouver", "aide pour trouver", "aide à la recherche", "aide pour la recherche"}
        text_filtered = " ".join(
            word for word in text.lower().split() if word not in mots_vides
        )

        # 3. Nettoyage NLP via FormationSearch
        cleaned_query = self.formations.preprocess_text(text_filtered)

        # 4. Fallback si la query est vide ou inutile
        if not cleaned_query.strip() or len(cleaned_query.split()) <= 1:
            objectif = self.ctx.objectif.strip()
            logger.info(f"[Fallback] Utilisation de l’objectif utilisateur comme query : {objectif}")
            return objectif.lower()

        return cleaned_query



    
    def _handle_formation_search(self, text: str, entities: dict) -> str:
        """Effectue une recherche de formation avec relai intelligent."""
        query = self._extract_search_query(text, entities)
        
        if not query:
            return "L'utilisateur cherche une formation mais n'a pas précisé le domaine. Demande-lui de préciser."
        
        # Stocker dans le contexte
        self._search_context["pending_query"] = query
        self._search_context["awaiting_confirmation"] = True
        return f"Vous souhaitez rechercher des formations en **{query}** ?\n\n✅ Oui – Lancer la recherche\n❌ Non – Modifier"

        
    def _handle_formation_selection(self, text: str, entities: dict) -> str:
        num = entities.get('number')
        if not num:
            match = re.search(r'\b([1-5])\b', text)
            if match:
                num = match.group(1)

        if num and self.ctx.search_results:
            try:
                idx = int(num) - 1
                fiche, _ = self.ctx.search_results[idx]
                self.ctx.current_formation = fiche

                if fiche not in self.ctx.formations_vues:
                    self.ctx.formations_vues.append(fiche)

                titre = fiche.get('titre', 'Formation')
                duree = fiche.get('duree', 'Non spécifiée')
                modalite = fiche.get('modalite', 'Non spécifiée')
                tarif = fiche.get('tarif', 'Nous contacter')
                lieu = fiche.get('lieu', 'À définir')
                is_internal = fiche.get("_source") == "internal"

                type_formation = "Beyond Expertise" if is_internal else "RNCP externe"
                emoji = "🔒" if is_internal else "📚"

                return (f"{emoji} **{titre}** ({type_formation})\n\n"
                        f"⏰ Durée : {duree}\n"
                        f"💻 Modalité : {modalite}\n"
                        f"💰 Tarif : {tarif}\n"
                        f"📍 Lieu : {lieu}\n\n"
                        f"Que souhaitez-vous savoir ? Objectifs, prérequis, financement...")
            except Exception as e:
                logger.error(f"Erreur sélection: {e}")

        return "Merci de sélectionner une formation en tapant son numéro (1 à 5)."

    def _get_formation_details(self, aspect: str) -> str:
        """Retourne les VRAIS détails d'une formation selon l'aspect demandé."""
        if not self.ctx.current_formation:
            return f"Aucune formation sélectionnée. Propose à l'utilisateur de chercher ou sélectionner une formation d'abord."
        
        f = self.ctx.current_formation
        titre = f.get('titre', 'cette formation')
        is_internal = f.get("_source") == "internal"
        
        if aspect == "objectives":
            objectifs = None

            # Pour les formations internes, prends aussi ACTIVITES_VISEES et CAPACITES_ATTESTEES !
            if is_internal:
                objectifs = (
                    f.get('objectifs') or
                    f.get('objectifs_pedagogiques') or
                    f.get('CAPACITES_ATTESTEES') or
                    f.get('ACTIVITES_VISEES')
                )
            else:
                objectifs = f.get('ACTIVITES_VISEES') or f.get('CAPACITES_ATTESTEES')

            if objectifs and len(str(objectifs)) > 50:
                # Limiter proprement sans couper au milieu d'une phrase
                sentences = str(objectifs).replace('\n', '. ').split('.')
                limited_text = ""
                for sentence in sentences:
                    if len(limited_text) + len(sentence) < 400:
                        limited_text += sentence.strip() + ". "
                    else:
                        break
                return f"Les objectifs de {titre} sont : {limited_text.strip()}"
            elif objectifs:
                return f"Les objectifs de {titre} sont : {objectifs}"
            else:
                # Générer des objectifs génériques selon le type
                if 'data' in titre.lower() or 'ia' in titre.lower() or "intelligence artificielle" in titre.lower():
                    return (f"Les objectifs de {titre} incluent : maîtriser les fondamentaux, "
                            f"développer des solutions pratiques, et acquérir une expertise reconnue.")
                else:
                    return f"Cette formation {titre} vise à développer les compétences clés du domaine."

        
        elif aspect == "prerequisites":
            prerequisites = None

            if is_internal:
                # Corrige : on gère le cas où c'est une liste
                prerequisites = f.get('prerequis') or f.get('public_prerequis')
                if prerequisites:
                    if isinstance(prerequisites, list):
                        prerequisites = ", ".join(prerequisites)
                    return f"Prérequis pour {titre} : {prerequisites}"
                else:
                    return (f"{titre} (Beyond Expertise) est accessible aux débutants motivés. "
                            f"Aucun prérequis technique n'est exigé.")
            else:
                prerequisites = f.get('prerequis') or f.get('CONDITIONS_ACCES')
                niveau = f.get('NOMENCLATURE_EUROPE_INTITULE', '')

                if prerequisites:
                    if isinstance(prerequisites, list):
                        prerequisites = ", ".join(prerequisites)
                    return f"Prérequis pour {titre} : {prerequisites}"
                elif 'niveau 6' in niveau.lower() or 'niveau 7' in niveau.lower():
                    return (f"{titre} (RNCP {niveau}) nécessite généralement un Bac+2/3 "
                            f"ou une expérience professionnelle équivalente.")
                else:
                    return (f"Les prérequis pour {titre} (formation externe) varient. "
                            f"Contactez l'organisme certificateur pour plus d'infos.")
            
        elif aspect == "price":
            tarif = f.get('tarif')
            if not tarif and is_internal:
                tarif = "Selon profil et financement"
            elif not tarif:
                tarif = "Variable selon l'organisme"
            
            # Adapter les financements selon la situation
            situation = self.ctx.situation
            if situation == "recherche":
                financement = "AIF Pôle Emploi, CPF, aides régionales"
            elif situation == "salarié":
                financement = "CPF, plan de développement entreprise, Transition Pro"
            elif situation == "reconversion":
                financement = "CPF de transition, Transition Pro, aides reconversion"
            else:
                financement = "CPF, financements personnels, aides diverses"
            
            return (f"Tarif de {titre} : {tarif}.\n"
                    f"Financements possibles pour votre situation ({situation}) : {financement}")
        
        elif aspect == "duration":
            duree = f.get('duree')
            if not duree and is_internal:
                duree = "Variable selon le parcours"
            elif not duree:
                duree = "Selon l'organisme certificateur"
            
            modalite = f.get('modalite', 'Présentiel/Distanciel possible')
            
            return (f"Durée de {titre} : {duree}.\n"
                    f"Format : {modalite}.\n"
                    f"Rythme adapté à votre situation.")
        
        elif aspect == "location":
            lieu = f.get('lieu')
            modalite = f.get('modalite')
            
            if is_internal:
                if not lieu:
                    lieu = "Paris et autres villes"
                if not modalite:
                    modalite = "Présentiel, distanciel ou hybride"
                return (f"{titre} se déroule : {lieu}.\n"
                        f"Modalités flexibles : {modalite}.\n"
                        f"Adaptation possible selon vos contraintes.")
            else:
                return (f"Lieu et modalités pour {titre} : variables selon l'organisme.\n"
                        f"Formation disponible dans plusieurs régions.")
        
        elif aspect == "certification":
            certifiant = f.get('certifiant', True)  # Par défaut, on suppose que c'est certifiant
            niveau = f.get('NOMENCLATURE_EUROPE_INTITULE', '')
            
            if is_internal:
                return (f"✅ {titre} délivre une certification Beyond Expertise reconnue.\n"
                        f"Attestation de compétences valorisable sur le marché.\n"
                        f"Éligible CPF dans la plupart des cas.")
            else:
                niveau_text = f" (Niveau {niveau})" if niveau else ""
                return (f"✅ {titre} est une formation RNCP certifiante{niveau_text}.\n"
                        f"Diplôme reconnu par l'État.\n"
                        f"Inscription au répertoire national.")
        
        else:
            # Informations générales
            type_text = "Beyond Expertise" if is_internal else "RNCP externe"
            return (f"{titre} est une formation {type_text}.\n"
                    f"Pour plus d'infos, demandez un aspect spécifique : "
                    f"objectifs, prérequis, tarif, durée, lieu, certification.")



    def _handle_intent_search_formation(self, user_input: str, entities: dict) -> Optional[str]:
        """
        Gestion complète de la recherche avec distinction formations internes/externes.
        ✅ FIXED: This method now only returns responses, history saving is handled in respond()
        """
        
        # Étape 1 : confirmation d'une recherche en attente
        if self._search_context["awaiting_confirmation"]:
            if "non" in user_input.lower():
                self._search_context["awaiting_confirmation"] = False
                self._search_context["pending_query"] = ""
                return "Pas de souci. Précisez un autre domaine si vous avez une idée, ou dites-moi comment je peux vous aider."
            
            elif "oui" in user_input.lower():
                query = self._search_context["pending_query"]
                results = self.formations.search(query)
                self._search_context["awaiting_confirmation"] = False
                self._search_context["pending_query"] = ""

                if not results:
                    return f"Aucune formation trouvée pour '{query}'. Essayez un autre domaine ou reformulez."

                self.ctx.search_results = results[:5]
                self._search_context["show_results"] = True

                # Séparer formations internes et externes
                internal_formations = []
                external_formations = []
                
                for i, (fiche, score) in enumerate(self.ctx.search_results):
                    source = fiche.get("_source", "")
                    if source == "internal":
                        internal_formations.append((i+1, fiche))
                    else:
                        external_formations.append((i+1, fiche))
                
                # Construire l'affichage avec distinction
                response = "🎓 **Formations trouvées** :\n"
                
                if internal_formations:
                    response += "\n🔒 **Formations Beyond Expertise :**\n"
                    for idx, fiche in internal_formations:
                        titre = fiche.get('titre', 'Sans titre')
                        duree = fiche.get('duree', '')
                        response += f"{idx}. {titre}"
                        if duree:
                            response += f" - {duree}"
                        response += "\n"
                
                if external_formations:
                    if internal_formations:
                        response += "\n"
                    response += "📚 **Formations RNCP certifiantes** (externes) :\n"
                    for idx, fiche in external_formations:
                        titre = fiche.get('titre', 'Sans titre')
                        response += f"{idx}. {titre}\n"
                
                response += "\nTapez le numéro pour plus de détails."
                
                # Ajouter une recommandation si reconversion
                if hasattr(self.ctx, 'situation') and self.ctx.situation == "reconversion" and internal_formations:
                    response += "\n\n💡 *Les formations Beyond Expertise sont particulièrement adaptées aux reconversions !*"
                
                return response
            
            else:
                return "Souhaitez-vous lancer la recherche maintenant ? ✅ Oui / ❌ Non"
        
        # Étape 2 : sélection d'une formation dans les résultats
        if self._search_context["show_results"]:
            if entities.get("number"):
                try:
                    idx = int(entities["number"]) - 1
                    selected, _ = self.ctx.search_results[idx]
                    self.ctx.current_formation = selected
                    self.ctx.formations_vues.append(selected)
                    self._search_context["show_results"] = False
                    
                    titre = selected.get('titre', 'formation sélectionnée')
                    is_internal = selected.get("_source") == "internal"
                    emoji = "🔒" if is_internal else "📚"
                    type_formation = "Beyond Expertise" if is_internal else "RNCP externe"
                    
                    return (f"{emoji} Formation sélectionnée : **{titre}** ({type_formation})\n\n"
                            f"Souhaitez-vous connaître les objectifs, les prérequis, la durée ou le tarif ?")
                except Exception:
                    return "Numéro invalide. Tapez un numéro entre 1 et 5."
            else:
                return "Tapez le numéro d'une formation pour voir ses détails."
        
        # Étape 3 : nouvelle demande de recherche
        query = self._extract_search_query(user_input, entities)

        if not query and self.ctx.objectif:
            logger.info("[Fallback] Utilisation de l'objectif utilisateur comme query : %s", self.ctx.objectif)
            query = self.ctx.objectif

        if not query:
            return "Pouvez-vous préciser le domaine de formation que vous recherchez ?"

        self._search_context["pending_query"] = query
        self._search_context["awaiting_confirmation"] = True
        return f"Vous souhaitez rechercher des formations en **{query}** ? ✅ Oui / ❌ Non"



    def respond(self, user_input: str) -> str:
        """
        Point d'entrée principal - Analyse l'intent puis demande au LLM.
        ✅ FIXED: Save all interactions to conversation history
        """
        if not user_input.strip():
            return "Je vous écoute... 😊"

        self.ctx.interactions += 1

        # ✅ FIXED: Add user message to history FIRST
        self.ctx.conversation_history.append({"role": "user", "content": user_input})

        # 1. Classification de l'intention
        intent, confidence = self.intent_classifier.predict(user_input)
        entities = self.intent_classifier.extract_entities(user_input)

        logger.info(f"Intent: {intent} ({confidence:.2f}), Entities: {entities}")

        # 2. Construire l'instruction basée sur l'intent
        base_instruction = self.intent_instructions.get(intent, self.intent_instructions["other"])

        # 3. Enrichir l'instruction selon l'intent spécifique
        enriched_instruction = base_instruction

        # Cas spéciaux nécessitant des actions
        if intent == "search_formation":
            formation_response = self._handle_intent_search_formation(user_input, entities)
            if formation_response:
                # ✅ FIXED: Save assistant response before returning
                self.ctx.conversation_history.append({"role": "assistant", "content": formation_response})
                return formation_response

        elif intent == "formation_select":
            selection_response = self._handle_formation_selection(user_input, entities)
            # ✅ FIXED: Save assistant response before returning
            self.ctx.conversation_history.append({"role": "assistant", "content": selection_response})
            return selection_response

        elif intent == "formation_details_objectives":
            enriched_instruction += "\n" + self._get_formation_details("objectives")

        elif intent == "info_prerequests":
            enriched_instruction += "\n" + self._get_formation_details("prerequisites")

        elif intent == "formation_details_price":
            enriched_instruction += "\n" + self._get_formation_details("price")

        elif intent == "formation_details_duration":
            enriched_instruction += "\n" + self._get_formation_details("duration")

        elif intent == "formation_details_location":
            enriched_instruction += "\n" + self._get_formation_details("location")

        elif intent == "info_certif":
            if self.ctx.current_formation:
                titre = self.ctx.current_formation.get('titre', 'Cette formation')
                enriched_instruction += f"\n{titre} délivre une certification reconnue. Valorise cet aspect."
        print(f"[DEBUG] : Enriched Instruction : \n\n {enriched_instruction}\n\n")
        # 5. Créer le prompt system avec contexte utilisateur actuel
        system_prompt = (
            f"Tu es un conseiller professionnel de Beyond Expertise.\n\n"
            f"UTILISATEUR ACTUEL :\n"
            f"• Nom : {self.ctx.nom}\n"
            f"• Objectif : {self.ctx.objectif}\n" 
            f"• Compétences : {', '.join(self.ctx.competences)}\n\n"
            f"IMPORTANT : Adapte ta réponse à CE profil spécifique. Si son objectif ne correspond pas aux formations tech de Beyond Expertise, sois honnête et oriente-le ailleurs.\n\n"
            f"Formations Beyond Expertise disponibles :\n"
            f"Power BI, Cloud Azure, SQL/NoSQL, ETL, Deep Learning, Machine Learning, JIRA, Data Analyst, Python Visualisation, Intelligence Artificielle\n\n"
            f"Réponds en 50-80 mots maximum, sois concis et utile."
        )

        # 6. Construire les messages à envoyer AU LLM (temporaire, pas stockés !)
        llm_messages = self.ctx.conversation_history.copy()
        # Injecte enriched_instruction SANS polluer l'historique utilisateur
        if enriched_instruction and enriched_instruction != base_instruction:
            llm_messages.append({"role": "user", "content": enriched_instruction})
        llm_messages.append({"role": "system", "content": system_prompt})

        # 7. Gestion du relai recherche formation (avant LLM)
        if self._search_context["awaiting_confirmation"]:
            if intent == "confirmation":
                query = self._search_context["pending_query"]
                results = self.formations.search(query)
                if not results:
                    self._search_context["awaiting_confirmation"] = False
                    response = f"Aucune formation trouvée pour '{query}'. Essayez un autre domaine."
                    # ✅ FIXED: Save response before returning
                    self.ctx.conversation_history.append({"role": "assistant", "content": response})
                    return response
                self.ctx.search_results = results[:5]
                self._search_context["awaiting_confirmation"] = False
                self._search_context["show_results"] = True
                formation_list = []
                for i, (f, _) in enumerate(self.ctx.search_results):
                    is_internal = f.get("_source") == "internal"
                    emoji = "🔒" if is_internal else "📚"
                    type_label = "Beyond Expertise" if is_internal else "RNCP"
                    titre = f.get('titre', 'Sans titre')
                    duree = f.get('duree', '')
                    if duree:
                        formation_list.append(f"{emoji} {i+1}. {titre} ({type_label}) - {duree}")
                    else:
                        formation_list.append(f"{emoji} {i+1}. {titre} ({type_label})")
                response = f"🎓 Formations trouvées pour **{query}** :\n\n" + "\n".join(formation_list) + "\n\nTapez le numéro pour en savoir plus."
                # ✅ FIXED: Save response before returning
                self.ctx.conversation_history.append({"role": "assistant", "content": response})
                return response

        elif self._search_context["show_results"]:
            if intent == "formation_select":
                self._search_context["show_results"] = False
                response = self._handle_formation_selection(user_input, entities)
                # ✅ FIXED: Save response before returning
                self.ctx.conversation_history.append({"role": "assistant", "content": response})
                return response

        # 8. Appeler le LLM
        try:
            response = self.llm.send(
                prompt="",  # Prompt vide car tout est dans messages
                messages=llm_messages
            )
            # 9. Ajouter la réponse à l'historique propre
            self.ctx.conversation_history.append({"role": "assistant", "content": response})

            # 10. Limiter l'historique pour éviter de dépasser les limites
            if len(self.ctx.conversation_history) > 50:
                self.ctx.conversation_history = (
                    self.ctx.conversation_history[:6] +
                    self.ctx.conversation_history[-30:]
                )
            return response

        except Exception as e:
            logger.error(f"Erreur LLM: {e}")
            error_response = "Désolé, j'ai eu un problème technique. Pouvez-vous reformuler votre question ?"
            # ✅ FIXED: Save error response before returning
            self.ctx.conversation_history.append({"role": "assistant", "content": error_response})
            return error_response

def main():
    """Lanceur principal."""
    print("🎯 === CONSEILLER BEYOND EXPERTISE (LLM-Driven) ===")
    print("Version pilotée par LLM avec enrichissement par intents")
    print("Tapez 'quit' pour quitter\n")
    
    counselor = LLMDrivenCounselor()
    
    # Premier message du bot - DYNAMIQUE selon le contexte
    print(f"🤖 Bonjour {counselor.ctx.nom} ! Ravi de vous retrouver. "
          f"Comment puis-je vous aider aujourd'hui dans votre projet de devenir {counselor.ctx.objectif} ?\n")
    
    while True:
        try:
            user_input = input("💬 Vous: ").strip()
            if user_input.lower() in ["quit", "exit", "bye", "au revoir"]:
                print(f"🤖 Au revoir {counselor.ctx.nom} ! Bonne continuation dans votre projet ! 👋")
                break
            
            response = counselor.respond(user_input)
            print(f"🤖 {response}\n")
            
            # Commandes spéciales
            if user_input.lower() == "stats":
                stats = counselor.get_stats()
                print(f"📊 Statistiques: {json.dumps(stats, indent=2, ensure_ascii=False)}\n")
            
        except (EOFError, KeyboardInterrupt):
            print("\n🤖 Au revoir ! À bientôt ! 👋")
            break
        except Exception as e:
            logger.error(f"Erreur: {e}")
            print("🤖 Désolé, une erreur s'est produite. Réessayons.\n")

if __name__ == "__main__":
    # Vérifier que les modèles sont disponibles
    import os
    if not os.path.exists("intent_model.pkl"):
        print("⚠️ Modèle d'intentions non trouvé. Entraînement en cours...")
        from training_intent_classifier import IntentTrainer
        trainer = IntentTrainer()
        trainer.train("intent_model.pkl")
    
    main()