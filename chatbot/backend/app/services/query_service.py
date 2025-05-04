# app/services/query_service.py
"""
Service pour traiter les requêtes de l'utilisateur et interagir avec le moteur LLM.
"""
from fastapi import HTTPException
from pydantic import BaseModel, validator
import gc
import json
from pathlib import Path

from app.schemas import UserProfile, SessionState, QueryResponse
from app.logging_config import logger
from app.utils import sanitize_input, DataService
from app.langchain_rag_service import LangChainRAGService
import app.globals as globs

# Constantes
DATA_FOLDER = Path(__file__).resolve().parent.parent / "content"

class SanitizedQueryRequest(BaseModel):
    """Requête étendue avec validation des entrées."""
    profile: UserProfile
    history: list = []
    question: str
    
    @validator('question')
    def validate_question(cls, v):
        """Valider et assainir la question."""
        if not v or not isinstance(v, str):
            raise HTTPException(status_code=400, detail="La question ne peut pas être vide")
        
        sanitized = sanitize_input(v)
        if not sanitized:
            raise HTTPException(status_code=400, detail="Format de question invalide")
            
        return sanitized
    
    @validator('history')
    def validate_history(cls, v):
        """Valider et assainir l'historique du chat."""
        sanitized_history = []
        
        for msg in v:
            if isinstance(msg, dict) and 'role' in msg and 'content' in msg:
                sanitized_history.append({
                    'role': sanitize_input(msg['role']),
                    'content': sanitize_input(msg['content'])
                })
            elif hasattr(msg, 'dict'):
                # Pour les modèles Pydantic
                msg_dict = msg.dict()
                if 'role' in msg_dict and 'content' in msg_dict:
                    sanitized_history.append({
                        'role': sanitize_input(msg_dict['role']),
                        'content': sanitize_input(msg_dict['content'])
                    })
        
        return sanitized_history

# Services d'accès aux composants
def get_data_service():
    """Service de données unique, déjà créé dans main.lifespan"""
    return globs.data_service

def get_llm_engine():
    if globs.llm_engine is None:
        raise HTTPException(503, "Service en cours d'initialisation")
    return globs.llm_engine

def get_rag_service():
    # Initialisé paresseusement
    global _rag_service
    if '_rag_service' not in globals() or _rag_service is None:
        _rag_service = LangChainRAGService()
    return _rag_service

def process_llm_response(question: str, history: list, profile: UserProfile, session: SessionState) -> dict:
    """
    Génère une réponse basée sur l'historique du chat et le profil utilisateur.
    Utilise RAG amélioré avec LangChain pour une meilleure récupération.
    Détecte également les besoins en formations RNCP et notifie l'utilisateur si nécessaire.
    """
    logger.info(f"Traitement de la question: {question[:50]}...")
    
    try:
        # Obtenir le service RAG pour la récupération
        rag = get_rag_service()
        
        # Obtenir le moteur LLM
        engine = get_llm_engine()
        
        # Infos de débogage sur la longueur de l'historique
        logger.debug(f"Longueur de l'historique: {len(history)}")
        
        # Tronquer l'historique aux 10 derniers tours (20 messages)
        max_turns = 10
        max_messages = max_turns * 2
        truncated_history = history[-max_messages:] if len(history) > max_messages else history
        
        # Détecter l'intention - important pour le routage
        intent = engine.detect_intent(question, truncated_history)
        logger.info(f"Intention détectée: {intent}")
        
        
        # Cas spécial pour les questions sur la certification
        if (intent == "info_certification" or "certifi" in question.lower()) and intent != "recherche_filtrée":
            # code qui vérifie uniquement une formation spécifique
            logger.info("Traitement d'une demande d'information sur certification")
            
            # Essayer de récupérer la formation en cours de discussion
            current_course = None
            
            # Si une formation est recommandée dans la session, l'utiliser
            if session and session.recommended_course:
                current_course = session.recommended_course
            else:
                # Sinon, essayer de trouver la formation la plus pertinente
                try:
                    data_service = get_data_service()
                    # Récupérer d'abord les formations récemment mentionnées dans l'historique
                    last_messages = truncated_history[-6:] if truncated_history else []
                    course_titles = []
                    
                    for msg in last_messages:
                        if isinstance(msg, dict) and msg.get('role') == 'assistant' and msg.get('content'):
                            # Extraire les titres de formations des messages précédents
                            for course in data_service.get_all_courses():
                                if course.get('titre', '') in msg.get('content', ''):
                                    course_titles.append(course.get('titre'))
                                    break
                    
                    # Si des formations ont été trouvées dans l'historique, utiliser la première
                    if course_titles:
                        current_course = data_service.get_course_by_title(course_titles[0])
                        logger.info(f"Formation trouvée dans l'historique: {current_course.get('titre', '')}")
                    else:
                        # Sinon, essayer une recherche vectorielle
                        try:
                            relevant_courses = rag.retrieve_courses(question, top_k=1)
                            if relevant_courses:
                                current_course = relevant_courses[0]
                                logger.info(f"Formation trouvée par recherche vectorielle: {current_course.get('titre', '')}")
                        except Exception as e:
                            logger.error(f"Erreur lors de la recherche vectorielle: {str(e)}")
                except Exception as e:
                    logger.error(f"Erreur lors de la recherche de formation: {str(e)}")
            
            # Si on a trouvé une formation, vérifier son statut de certification
            if current_course:
                course_title = current_course.get('titre', 'Cette formation')
                # Important: vérifier explicitement le champ 'certifiant'
                is_certified = current_course.get('certifiant', False)
                
                logger.info(f"Vérification certification pour {course_title}: {is_certified}")
                certifiant_msg = "est certifiante"
                non_certifiant_msg = "n'est pas certifiante"
                
                response = {
                    "answer": f"La formation {course_title} {certifiant_msg if is_certified else non_certifiant_msg}",
                    "recommended_course": current_course,
                    "next_action": "follow_up",
                    "intent": "info_certification"
                }
                
                # Si la formation n'est pas certifiante et que l'utilisateur cherche une certification
                if not is_certified:
                    # Vérifier si on a besoin d'informer sur les formations RNCP
                    rncp_needed = engine.detect_rncp_need(question, intent)
                    if rncp_needed and not engine.enable_rncp:
                        response["answer"] += (
                            " Si vous recherchez spécifiquement une formation certifiante dans ce domaine, "
                            "je vous invite à contacter notre service formation qui pourra vous renseigner "
                            "sur les options de formations certifiantes RNCP disponibles."
                        )
                        response["next_action"] = "contact_for_rncp"
                
                return response
        
        # Vérifier si la question pourrait nécessiter des formations RNCP
        rncp_needed = engine.detect_rncp_need(question, intent)
        
        # Si la demande concerne des formations RNCP et que l'accès est désactivé,
        # informer l'utilisateur et lui proposer des alternatives
        if rncp_needed and not engine.enable_rncp:
            logger.info("Besoin RNCP détecté avec accès RNCP désactivé")
            
            # Message de base pour informer sur la limitation
            rncp_message = (
                "Je constate que vous recherchez une formation certifiante qui n'est pas disponible "
                "dans notre catalogue interne. Actuellement, je ne peux vous présenter que nos formations internes. "
                "Pour des informations sur les formations certifiantes RNCP, je vous invite à "
                "contacter notre service formation."
            )
            
            # Récupérer les formations internes les plus pertinentes pour proposer des alternatives
            try:
                # Utiliser directement la recherche RAG standard pour trouver des alternatives internes
                data_service = get_data_service()
                all_internal_courses = [c for c in data_service.get_all_courses() if c.get('_source') != 'rncp']
                
                # Prendre les 3 premières formations internes comme alternatives
                alternatives = all_internal_courses[:3] if all_internal_courses else []
                
                # Ajouter les alternatives au message
                if alternatives:
                    alt_message = "\n\nVoici quelques formations internes qui pourraient vous intéresser :\n"
                    for course in alternatives:
                        title = course.get('titre', 'Formation inconnue')
                        alt_message += f"- {title}\n"
                    alt_message += "\nSouhaitez-vous des informations sur l'une de ces formations ?"
                    
                    final_message = rncp_message + alt_message
                else:
                    final_message = rncp_message + "\n\nNous n'avons actuellement pas de formations internes qui correspondraient à votre demande."
                
                # Renvoyer la réponse adaptée
                return {
                    "answer": final_message,
                    "recommended_course": alternatives[0] if alternatives else None,
                    "next_action": "liste_formations",  # Rediriger vers la liste des formations
                    "intent": intent  # Garder l'intent original
                }
            except Exception as e:
                logger.error(f"Erreur lors de la préparation des alternatives: {str(e)}")
                # En cas d'erreur, retourner simplement le message de base
                return {
                    "answer": rncp_message,
                    "recommended_course": None,
                    "next_action": "contact_support",
                    "intent": intent  # Garder l'intent original
                }
        
        # Continuer avec le traitement normal si pas de besoin RNCP spécifique ou si RNCP est activé
        
        # Pour les requêtes de comparaison, utiliser directement le service RAG
        if any(kw in question.lower() for kw in ['compare', 'différence', 'similitude', 'versus', 'vs']):
            logger.info("Utilisation de la chaîne RAG pour une requête de comparaison")
            
            # Récupérer d'abord les cours pertinents
            try:
                relevant_courses = rag.retrieve_courses(question, top_k=3)
                
                # Interroger le service RAG
                rag_result = rag.query(question, truncated_history)
                
                # Convertir au format attendu
                return {
                    "answer": rag_result.get("answer", ""),
                    "recommended_course": rag_result.get("recommended_course") or 
                                         (relevant_courses[0] if relevant_courses else None),
                    "next_action": "follow_up",
                    "intent": intent  # Garder l'intent original
                }
            except Exception as e:
                logger.error(f"Erreur lors de la comparaison: {str(e)}")
                # Continuer vers les autres méthodes en cas d'erreur
        
        # Pour les requêtes d'informations spécifiques sur un cours, utiliser le service RAG
        if any(intent_kw in question.lower() for intent_kw in [
            'objectif', 'programme', 'prérequis', 'prerequis', 'public', 
            'tarif', 'prix', 'coût', 'durée', 'modalité'
        ]):
            logger.info("Utilisation de la chaîne RAG pour une requête d'information spécifique")
            
            try:
                # Récupérer d'abord les cours pertinents
                relevant_courses = rag.retrieve_courses(question, top_k=3)
                
                # Interroger le service RAG
                rag_result = rag.query(question, truncated_history)
                
                # Ne pas ajuster l'intent basé sur les mots-clés de la question
                # On conserve l'intent détectée par le détecteur d'intent
                
                return {
                    "answer": rag_result.get("answer", ""),
                    "recommended_course": rag_result.get("recommended_course") or 
                                         (relevant_courses[0] if relevant_courses else None),
                    "next_action": "follow_up",
                    "intent": intent  # Garder l'intent original
                }
            except Exception as e:
                logger.error(f"Erreur lors de la récupération d'infos spécifiques: {str(e)}")
                # Continuer vers les autres méthodes en cas d'erreur
        
        # Pour les requêtes de liste de formations, utiliser le service de données et formater la réponse
        if intent == "liste_formations":
            logger.info("Traitement d'une demande de liste de formations")
            
            try:
                data_service = get_data_service()
                courses = data_service.get_all_courses()
                
                # Formatage de la réponse
                course_list = "\n".join([f"- {course.get('titre', 'Titre inconnu')}" for course in courses[:10]])
                answer = f"Voici quelques-unes de nos formations disponibles:\n\n{course_list}\n\nSouhaitez-vous des informations spécifiques sur l'une de ces formations ?"
                
                return {
                    "answer": answer,
                    "recommended_course": None,
                    "next_action": "liste_formations",
                    "intent": "liste_formations"  # Intent déjà correcte ici
                }
            except Exception as e:
                logger.error(f"Erreur lors de la récupération de la liste des formations: {str(e)}")
                # Continuer vers les autres méthodes en cas d'erreur
        
        # Pour les recommandations et les recherches filtrées, utiliser le service RAG
        if intent == "recommandation":
            logger.info(f"Traitement d'une {intent}")
            
            try:
                # Récupérer les cours pertinents
                relevant_courses = rag.retrieve_courses(question, top_k=3)
                
                # Interroger le service RAG pour une réponse personnalisée
                rag_result = rag.query(question, truncated_history)
                
                return {
                    "answer": rag_result.get("answer", ""),
                    "recommended_course": rag_result.get("recommended_course") or 
                                         (relevant_courses[0] if relevant_courses else None),
                    "next_action": "follow_up",
                    "intent": intent  # Garder l'intent original
                }
            except Exception as e:
                logger.error(f"Erreur lors de la recommandation: {str(e)}")
                # Continuer vers les autres méthodes en cas d'erreur
        
        # Pour une recherche filtrée
        if intent == "recherche_filtrée":
            logger.info("Traitement d'une recherche filtrée")
            try:
                response_data = engine.generate_response_from_filtered_search(
                    question, history, profile, session
                )
                # S'assurer que l'intent est préservée
                response_data["intent"] = intent
                return response_data
            except Exception as e:
                logger.error(f"Erreur lors de la recherche filtrée: {str(e)}")
                # Continuer vers les autres méthodes en cas d'erreur

        # Pour les intents RNCP spécifiques, utiliser le service RAG avec filtrage
        if "rncp" in question.lower():
            logger.info("Traitement d'une demande spécifique RNCP")
            
            # Si RNCP est désactivé, informer l'utilisateur
            if not engine.enable_rncp:
                return {
                    "answer": "Je constate que vous recherchez des formations RNCP. Actuellement, nous présentons uniquement nos formations internes. Pour des informations sur les formations certifiantes RNCP, veuillez contacter notre service formation.",
                    "recommended_course": None,
                    "next_action": "contact_for_rncp",
                    "intent": intent  # Garder l'intent original
                }
            
            try:
                # Sinon, récupérer spécifiquement les cours RNCP
                rncp_result = rag.query(question + " RNCP", truncated_history)
                
                return {
                    "answer": rncp_result.get("answer", ""),
                    "recommended_course": rncp_result.get("recommended_course"),
                    "next_action": "follow_up",
                    "intent": intent  # Garder l'intent original
                }
            except Exception as e:
                logger.error(f"Erreur lors de la requête RNCP: {str(e)}")
                # Continuer vers les autres méthodes en cas d'erreur
        
        # Pour tout autre type de requête, utiliser le moteur LLM de base
        logger.info("Utilisation du moteur LLM standard pour la réponse")
        try:
            response = engine.generate_response(question, truncated_history, profile, session)
            
            # Ne pas modifier l'intent ici, conserver celle détectée par le détecteur d'intent
            # S'assurer que l'intent original est conservée dans la réponse
            response["intent"] = intent
            
            return response
        except Exception as e:
            logger.error(f"Erreur lors de la génération de réponse standard: {str(e)}")
            return {
                "answer": "Désolé, une erreur est survenue lors du traitement de votre question. Pourriez-vous reformuler ou essayer une autre question ?",
                "recommended_course": None,
                "next_action": "error",
                "intent": intent
            }
        
    except Exception as e:
        logger.error(f"Erreur globale dans process_llm_response: {str(e)}", exc_info=True)
        return {
            "answer": "Désolé, une erreur est survenue lors du traitement de votre question. Pourriez-vous reformuler ou essayer une autre question ?",
            "recommended_course": None,
            "next_action": "error",
            "intent": "fallback"
        }
    


def format_response(response_data: dict, session: SessionState) -> QueryResponse:
    """Formate les données de réponse en QueryResponse et met à jour la session."""
    
    if response_data is None:
        logger.warning("Response data is None, returning default error response")
        return QueryResponse(
            reply="Désolé, une erreur est survenue lors du traitement de votre requête.",
            intent="error",
            next_action="retry",
            recommended_course=None
        )
    # Convertir la course recommandée si c'est une chaîne
    if response_data and isinstance(response_data.get("recommended_course"), str):
        course_name = response_data["recommended_course"]
        data_service = get_data_service()

        course_dict = None
        for json_file in Path(DATA_FOLDER).glob('*.json'):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    course = json.load(f)
                    if course.get('titre') == course_name:
                        course_dict = course
                        course_dict['_source'] = 'internal'
                        break
            except Exception as e:
                logger.error(f"Erreur lors de la lecture du fichier {json_file}: {str(e)}")

        response_data["recommended_course"] = course_dict if course_dict else None

    # Mise à jour session
    if response_data.get("recommended_course"):
        session.recommended_course = response_data["recommended_course"]
    session.last_intent = response_data.get("intent", "fallback")

    if not isinstance(response_data.get("next_action"), str):
        # Si next_action n'est pas une chaîne, utilisez une valeur par défaut
        response_data["next_action"] = "fallback"

    return QueryResponse(
        reply=response_data.get("answer", ""),
        intent=response_data.get("intent", "fallback"),
        next_action=response_data.get("next_action", "follow_up"),
        recommended_course=response_data.get("recommended_course")
    )

def handle_query_exception(e: Exception) -> QueryResponse:
    """Gestion des exceptions pour l'endpoint query."""
    logger.error(f"Erreur non gérée dans query_endpoint: {str(e)}", exc_info=True)
    return QueryResponse(
        reply="Une erreur est survenue lors du traitement de votre demande. Notre équipe technique a été notifiée.",
        intent="error",
        next_action="contact_support",
        recommended_course=None
    )