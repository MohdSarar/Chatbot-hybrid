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
from app.utils import sanitize_input, DataService

# Import des composants LangChain avec les nouveaux packages
from langchain.docstore.document import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.docstore.document import Document
from pathlib import Path
import app.globals as globs

load_dotenv(dotenv_path="app/.env")
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))


class LLMEngine:
    def __init__(self,
                 content_dir: str = "app/content",
                 disable_auto_update: bool = True,
                 enable_rncp: bool = None,
                 data_service: "DataService | None" = None):

        self.logger = logging.getLogger(__name__)
        self.logger.info("Initialisation de LLMEngine")

        self.content_dir = Path(content_dir)
        self.disable_auto_update = disable_auto_update
        if enable_rncp is None:
            import app.globals as globs
            enable_rncp = globs.enable_rncp
        self.enable_rncp = enable_rncp
        self.logger.info(
            f"Accès aux formations RNCP : {'activé' if enable_rncp else 'désactivé'}"
        )

        #   Utilise le DataService passé par le caller, ne le recrée pas
        from app.utils import DataService          # import local pour éviter les boucles
        self.data_service = data_service or DataService(
            content_dir,
            disable_auto_update=disable_auto_update,
            enable_rncp=enable_rncp,
        )
        
        
        
        self.logger.info(f"Accès aux formations RNCP: {'activé' if enable_rncp else 'désactivé'}")
        
    
        # Composants LangChain pour améliorer le RAG
        self.embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
        # Stockage des chemins pour validation
        self.valid_course_ids = set()
        self._index_course_ids()
        
        # Stockage Chroma
        self._chroma_store = None
        
        # Pour l'assainissement des entrées
        self.sanitize_pattern = re.compile(r'[^\w\s.,;:!?()[\]{}\'"-]')
        
        # Attributs pour le caching
        self._cached_rncp_docs = None
        self._rncp_last_modified = 0
        self.rncp_data = None
    
    def _prepare_rncp_docs(self, force_reload=False):
        """
        Prépare les documents RNCP pour l'embedding avec LangChain.
        Utilise une approche de mise en cache pour éviter les rechargements inutiles.
        
        Args:
            force_reload: Force le rechargement des documents même si déjà en cache
            
        Returns:
            Liste de documents RNCP préparés pour l'embedding
        """
        # Si l'accès RNCP est désactivé, retourner une liste vide
        if not self.enable_rncp:
            self.logger.debug("Accès RNCP désactivé - aucun document RNCP chargé")
            return []
            
        # Si nous avons déjà traité ces documents et que nous ne forçons pas le rechargement, retourner le cache
        if self._cached_rncp_docs is not None and not force_reload:
            self.logger.debug(f"Utilisation de {len(self._cached_rncp_docs)} documents RNCP en cache")
            return self._cached_rncp_docs
            
        rncp_path = self.content_dir / 'rncp' / 'rncp.json'
        if not rncp_path.exists():
            self._cached_rncp_docs = []
            return []
            
        docs = []
        try:
            # Vérifier si le fichier a été modifié depuis le dernier chargement
            if self._rncp_last_modified > 0:
                last_mod = rncp_path.stat().st_mtime
                if last_mod <= self._rncp_last_modified and not force_reload:
                    self.logger.debug("Fichier RNCP non modifié, utilisation du cache")
                    if self._cached_rncp_docs is not None:
                        return self._cached_rncp_docs
            
            with open(rncp_path, 'r', encoding='utf-8') as f:
                rncp_data = json.load(f)
                self.rncp_data = rncp_data  # Stocker pour une utilisation ultérieure
                
                for course in rncp_data:
                    if not isinstance(course, dict) or 'titre' not in course:
                        continue
                        
                    # Combiner les champs pour une représentation riche
                    champs = ['titre', 'emplois_vises', 'objectifs', 'public', 'competences_visees']
                    textes = []
                    
                    for champ in champs:
                        if champ in course:
                            valeur = course[champ]
                            if isinstance(valeur, list):
                                champ_text = '. '.join(str(item) for item in valeur)
                            else:
                                champ_text = str(valeur)
                            textes.append(champ_text)
                    
                    text = '. '.join(textes)
                    
                    if len(text) > 10:  # Validation minimale
                        docs.append(Document(
                            page_content=text,
                            metadata={
                                "source": "rncp", 
                                "course_id": f"rncp_{course.get('id', '')}", 
                                "title": course.get("titre"),
                                "certifiant": True,
                                "modalite": course.get("modalité", ""),
                                "duree": course.get("durée", "")
                            }
                        ))
            
            # Stocker le temps de dernière modification
            self._rncp_last_modified = rncp_path.stat().st_mtime
            
            # Mettre en cache les documents pour éviter de retraiter à chaque fois
            self._cached_rncp_docs = docs
            
            self.logger.info(f"Préparation de {len(docs)} documents RNCP")
        except Exception as e:
            self.logger.error(f"Erreur lors de la préparation des documents RNCP: {str(e)}")
            # En cas d'erreur, conserver les documents mis en cache précédemment si disponibles
            if self._cached_rncp_docs is not None:
                return self._cached_rncp_docs
            else:
                self._cached_rncp_docs = []
        
        return docs
    
    def _get_chroma_store(self, force_init=False):
        """Obtient ou initialise le store Chroma avec meilleure gestion d'erreurs"""
        if self._chroma_store is not None and not force_init:
            return self._chroma_store
            
        persist_directory = "chroma_db"
        
        try:
            # Essayer de charger un store existant
            if Path(persist_directory).exists():
                self._chroma_store = Chroma(
                    persist_directory=persist_directory,
                    embedding_function=self.embedding_model
                )
                return self._chroma_store
        except Exception as e:
            self.logger.error(f"Erreur chargement Chroma: {str(e)}")

        # Créer un nouveau store
        try:
            # Charger les documents
            docs = []
            
            # 1. Charger les cours internes
            for course_id in self.valid_course_ids:
                if course_id.startswith("rncp_"):
                    continue
                    
                course = self.data_service.get_course_by_id(course_id)
                if course:
                    text = self._prepare_text(course)
                    docs.append(Document(
                        page_content=text,
                        metadata={
                            "source": "internal",
                            "course_id": course_id,
                            "title": course.get("titre", ""),
                            "certifiant": course.get("certifiant", False),
                            "modalite": course.get("modalité", ""),
                            "duree": course.get("durée", "")
                        }
                    ))
            
            # 2. Charger les documents RNCP si activé
            if self.enable_rncp:
                rncp_docs = self._prepare_rncp_docs()
                docs.extend(rncp_docs)

            if not docs:
                raise ValueError("Aucun document à indexer")
                
            # Créer par lots de 100 documents
            batch_size = 100
            batches = [docs[i:i + batch_size] for i in range(0, len(docs), batch_size)]
            
            # Premier lot pour créer la collection
            self._chroma_store = Chroma.from_documents(
                documents=batches[0],
                embedding=self.embedding_model,
                persist_directory=persist_directory
            )
            
            # Ajouter les lots restants
            for batch in batches[1:]:
                self._chroma_store.add_documents(batch)
                
            return self._chroma_store
            
        except Exception as e:
            self.logger.error(f"Erreur création Chroma: {str(e)}")
            # Retourner un store vide en dernier recours
            self._chroma_store = Chroma(embedding_function=self.embedding_model)
            return self._chroma_store
        
    def detect_rncp_need(self, question: str, intent: str = None) -> bool:
        """
        Détecte si la demande de l'utilisateur nécessite des formations RNCP,
        mais uniquement si l'utilisateur mentionne directement RNCP ou si
        la formation recherchée n'existe pas dans nos formations internes.
        
        Args:
            question: La question de l'utilisateur
            intent: L'intention détectée (optionnel)
            
        Returns:
            True si la demande nécessite des formations RNCP, False sinon
        """
        # Si l'accès RNCP est activé, retourner False (pas besoin de détection spéciale)
        if self.enable_rncp:
            return False
            
        # 1. Vérifier uniquement les mentions directes de RNCP (indicateur fort)
        question_lower = question.lower()
        if "rncp" in question_lower or "registre national" in question_lower:
            self.logger.info("Détection de besoin RNCP: mention directe")
            return True
        
        # 2. Si la question concerne des certifications, vérifier si les formations internes répondent au besoin
        if ("certifi" in question_lower or "diplôm" in question_lower or 
            "reconnu par l'état" in question_lower or "titre pro" in question_lower):
            
            # Rechercher des formations internes potentiellement pertinentes
            try:
                # Récupérer toutes les formations internes
                data_service = self.data_service
                internal_courses = [c for c in data_service.get_all_courses() if c.get('_source') != 'rncp']
                
                if not internal_courses:
                    self.logger.info("Aucune formation interne disponible")
                    return True
                
                # Vérifier si l'une des formations internes est certifiante
                has_certified_internal = any(course.get('certifiant', False) for course in internal_courses)
                
                # Si aucune formation interne n'est certifiante, et que la question porte sur la certification,
                # il y a un besoin RNCP
                if not has_certified_internal and intent == "info_certification":
                    self.logger.info("Détection de besoin RNCP: aucune formation interne certifiante + demande de certification")
                    return True
                
                # Rechercher le contexte de la question pour voir s'il s'agit d'une formation spécifique
                course_context = None
                
                # Extraire les mots significatifs (plus de 3 lettres, pas dans liste d'arrêt)
                stop_words = {"formation", "certifi", "certifiante", "diplôm", "titre", "cours", 
                            "apprendre", "étudier", "suivre", "cette", "votre", "être"}
                words = [w for w in question_lower.split() if len(w) > 3 and w not in stop_words]
                
                # Récupérer les noms de cours qui pourraient être mentionnés
                mentioned_courses = []
                for course in internal_courses:
                    course_title = course.get('titre', '').lower()
                    # Vérifier si le titre du cours est mentionné dans la question
                    if any(word in course_title or course_title in question_lower for word in words):
                        mentioned_courses.append(course)
                
                # Si des cours sont mentionnés, vérifier s'ils sont certifiants
                if mentioned_courses:
                    # Si l'un des cours mentionnés est certifiant, pas besoin de RNCP
                    if any(course.get('certifiant', False) for course in mentioned_courses):
                        return False
                    
                    # Si aucun n'est certifiant mais la question porte sur la certification,
                    # il pourrait y avoir un besoin RNCP
                    if intent == "info_certification" or "certifi" in question_lower:
                        self.logger.info(f"Détection de besoin RNCP: formations mentionnées non certifiantes + demande certification")
                        return True
                
                # Si aucun cours mentionné et la question insiste sur la certification,
                # il pourrait y avoir un besoin RNCP
                if intent == "info_certification" and not mentioned_courses:
                    self.logger.info("Détection de besoin RNCP: aucun cours mentionné + demande certification")
                    return True
                    
            except Exception as e:
                self.logger.error(f"Erreur lors de la détection de besoin RNCP: {str(e)}")
        
        # Par défaut, pas de besoin RNCP détecté
        return False    



    def _retrieve_content(self, question, top_k=3):
        """
        Récupère le contenu pertinent en utilisant la récupération de Chroma.
        Version robuste avec gestion d'erreurs améliorée pour éviter les problèmes "Ran out of input".
        """
        # Assainir l'entrée
        sanitized_question = sanitize_input(question)
        
        # Vérifier si la question pourrait nécessiter des formations RNCP
        rncp_needed = self.detect_rncp_need(sanitized_question)
        
        # Obtenir ou initialiser le store Chroma avec gestion d'erreurs robuste
        try:
            chroma_store = self._get_chroma_store()
        except Exception as e:
            self.logger.error(f"Erreur lors de l'initialisation du store Chroma: {str(e)}")
            return self._fallback_random_courses(top_k)
        
        # Récupérer les documents pertinents avec approche simplifiée
        try:
            # Utiliser la recherche de similarité directe au lieu du compresseur qui cause des erreurs
            # Cette approche est plus robuste mais un peu moins précise
            search_filters = {}
            if not self.enable_rncp and not rncp_needed:
                # Ne filtrer que si RNCP est désactivé et qu'il n'y a pas de besoin RNCP détecté
                search_filters = {"source": "internal"}
            
            # Exécuter la recherche avec les filtres appropriés
            docs = chroma_store.similarity_search(
                sanitized_question, 
                k=top_k * 2,  # Récupérer plus pour pouvoir filtrer ensuite
                filter=search_filters if search_filters else None
            )
            
            # Convertir les documents récupérés en objets de cours avec vérification
            best_courses = []
            seen_courses = set()
            
            for doc in docs:
                course_id = doc.metadata.get("course_id")
                
                # Vérifications de sécurité
                if not course_id:
                    continue
                    
                # Ignorer les cours RNCP si l'accès est désactivé et aucun besoin RNCP n'est détecté
                if course_id.startswith("rncp_") and not self.enable_rncp and not rncp_needed:
                    continue
                    
                # Éviter les doublons
                if course_id in seen_courses:
                    continue
                    
                # Récupérer le cours complet
                try:
                    course = self._get_course_content(course_id)
                    if course:
                        # Ajouter l'étiquette de source
                        course['_source'] = doc.metadata.get("source", "internal")
                        best_courses.append(course)
                        seen_courses.add(course_id)
                        
                        # Arrêter si nous avons assez de cours
                        if len(best_courses) >= top_k:
                            break
                except Exception as e:
                    self.logger.error(f"Erreur lors du chargement du cours {course_id}: {str(e)}")
                    continue
            
            # Si nous avons trouvé des cours, les retourner
            if best_courses:
                # Vérification explicite de certains champs critiques
                for course in best_courses:
                    # S'assurer que le champ certifiant est bien défini
                    if 'certifiant' not in course:
                        # Vérifier le contenu JSON source pour voir si le champ y est
                        try:
                            course_id = course.get('id', '')
                            if course_id:
                                course_path = self.content_dir / f"{course_id}.json"
                                if course_path.exists():
                                    with open(course_path, 'r', encoding='utf-8') as f:
                                        json_data = json.load(f)
                                        if 'certifiant' in json_data:
                                            course['certifiant'] = json_data['certifiant']
                                        else:
                                            course['certifiant'] = False
                                else:
                                    course['certifiant'] = False
                            else:
                                course['certifiant'] = False
                        except Exception as e:
                            self.logger.error(f"Erreur lors de la vérification du champ certifiant: {str(e)}")
                            course['certifiant'] = False
                
                return best_courses
            else:
                # Si aucun cours pertinent, utiliser la méthode de repli
                return self._fallback_random_courses(top_k)
                
        except Exception as e:
            self.logger.error(f"Erreur dans la récupération de contenu par similitude: {str(e)}")
            
            # En cas d'échec, essayer une approche encore plus simple
            try:
                # Chercher des cours par mots-clés dans les titres
                data_service = self.data_service
                all_courses = data_service.get_all_courses()
                
                # Filtrer par mots-clés
                words = sanitized_question.lower().split()
                significant_words = [w for w in words if len(w) > 3]
                
                matched_courses = []
                for course in all_courses:
                    title = course.get('titre', '').lower()
                    score = sum(1 for w in significant_words if w in title)
                    if score > 0:
                        matched_courses.append((course, score))
                
                # Trier par score et prendre les top_k
                if matched_courses:
                    matched_courses.sort(key=lambda x: x[1], reverse=True)
                    return [c[0] for c in matched_courses[:top_k]]
            except Exception as e:
                self.logger.error(f"Erreur dans la récupération par mots-clés: {str(e)}")
            
            # Si tout échoue, utiliser la méthode de repli
            return self._fallback_random_courses(top_k)

    def _fallback_random_courses(self, count=3):
        """Méthode de repli améliorée"""
        try:
            data_service = self.data_service
            all_courses = data_service.get_all_courses()
            
            # Filtrer selon les paramètres RNCP
            if not self.enable_rncp:
                all_courses = [c for c in all_courses if c.get('_source') != 'rncp']
                
            if not all_courses:
                return []
                
            # Trier par pertinence (certifiants d'abord, puis internes)
            all_courses.sort(
                key=lambda c: (
                    0 if c.get('certifiant', False) else 1,
                    0 if c.get('_source') == 'internal' else 1
                )
            )
            
            return all_courses[:min(count, len(all_courses))]
            
        except Exception as e:
            self.logger.error(f"Erreur méthode de repli: {str(e)}")
            return []                       

    def _index_course_ids(self):
        """Crée un index des IDs de cours valides."""
        # Utiliser directement les IDs du service de données
        self.valid_course_ids = self.data_service.course_ids.copy()
        self.logger.info(f"{len(self.valid_course_ids)} IDs de cours indexés")
    

    
    @lru_cache(maxsize=100)
    def _get_course_content(self, course_id):
        """Obtient le contenu d'un cours spécifique par ID avec mise en cache."""
        return self.data_service.get_course_by_id(course_id)
    
    def _prepare_rncp_docs(self, force_reload=False):
        """
        Prépare les documents RNCP pour l'embedding avec LangChain.
        Utilise une approche de mise en cache pour éviter les rechargements inutiles.
        
        Args:
            force_reload: Force le rechargement des documents même si déjà en cache
            
        Returns:
            Liste de documents RNCP préparés pour l'embedding
        """
        # Si nous avons déjà traité ces documents et que nous ne forçons pas le rechargement, retourner le cache
        if self._cached_rncp_docs is not None and not force_reload:
            self.logger.debug(f"Utilisation de {len(self._cached_rncp_docs)} documents RNCP en cache")
            return self._cached_rncp_docs
            
        rncp_path = self.content_dir / 'rncp' / 'rncp.json'
        if not rncp_path.exists():
            self._cached_rncp_docs = []
            return []
            
        docs = []
        try:
            # Vérifier si le fichier a été modifié depuis le dernier chargement
            if self._rncp_last_modified > 0:
                last_mod = rncp_path.stat().st_mtime
                if last_mod <= self._rncp_last_modified and not force_reload:
                    self.logger.debug("Fichier RNCP non modifié, utilisation du cache")
                    if self._cached_rncp_docs is not None:
                        return self._cached_rncp_docs
            
            with open(rncp_path, 'r', encoding='utf-8') as f:
                rncp_data = json.load(f)
                self.rncp_data = rncp_data  # Stocker pour une utilisation ultérieure
                
                for course in rncp_data:
                    if not isinstance(course, dict) or 'titre' not in course:
                        continue
                        
                    # Combiner les champs pour une représentation riche
                    champs = ['titre', 'emplois_vises', 'objectifs', 'public', 'competences_visees']
                    textes = []
                    
                    for champ in champs:
                        if champ in course:
                            valeur = course[champ]
                            if isinstance(valeur, list):
                                champ_text = '. '.join(str(item) for item in valeur)
                            else:
                                champ_text = str(valeur)
                            textes.append(champ_text)
                    
                    text = '. '.join(textes)
                    
                    if len(text) > 10:  # Validation minimale
                        docs.append(Document(
                            page_content=text,
                            metadata={
                                "source": "rncp", 
                                "course_id": f"rncp_{course.get('id', '')}", 
                                "title": course.get("titre"),
                                "certifiant": True,
                                "modalite": course.get("modalité", ""),
                                "duree": course.get("durée", "")
                            }
                        ))
            
            # Stocker le temps de dernière modification
            self._rncp_last_modified = rncp_path.stat().st_mtime
            
            # Mettre en cache les documents pour éviter de retraiter à chaque fois
            self._cached_rncp_docs = docs
            
            self.logger.info(f"Préparation de {len(docs)} documents RNCP")
        except Exception as e:
            self.logger.error(f"Erreur lors de la préparation des documents RNCP: {str(e)}")
            # En cas d'erreur, conserver les documents mis en cache précédemment si disponibles
            if self._cached_rncp_docs is not None:
                return self._cached_rncp_docs
            else:
                self._cached_rncp_docs = []
        
        return docs
    
    def _prepare_text(self, course):
        """Prépare la représentation textuelle d'un cours pour l'embedding."""
        parts = [course.get("titre", "")]
        for field in ['objectifs', 'programme', 'prerequis', 'public']:
            content = course.get(field, "")
            if isinstance(content, list):
                content = '; '.join(content)
            if content:
                parts.append(f"{field.capitalize()}: {content}")
        
        for field in ['durée', 'modalité', 'tarif', 'lieu', 'certifiant']:
            value = course.get(field, "")
            if value:
                parts.append(f"{field.capitalize()}: {value}")
                
        return '. '.join(parts)

            
    def _build_prompt(self, profile, relevant_courses, chat_history):
        """Construit le prompt avec des entrées assainies."""
        prompt = """Vous êtes un conseiller professionnel expérimenté, orienté résultats et service client, bienveillant et proactif. Répondez en français."""
        prompt += "\n\nProfil utilisateur:\n"
        
        # Assainir et traiter le profil
        if profile:
            profile_dict = profile
            if hasattr(profile, 'model_dump'):
                profile_dict = profile.model_dump()
            elif hasattr(profile, 'dict'):
                profile_dict = profile.dict()
            
            if isinstance(profile_dict, dict):
                for key, value in profile_dict.items():
                    sanitized_key = sanitize_input(key)
                    sanitized_value = sanitize_input(value)
                    prompt += f"- {sanitized_key.capitalize()}: {sanitized_value}\n"
            else:
                self.logger.warning(f"Le profil n'est pas un dictionnaire: {type(profile_dict)}")
        
        # Assainir l'historique de chat
        prompt += "\nHistorique de la conversation:\n"
        for message in chat_history:
            if isinstance(message, dict) and 'role' in message and 'content' in message:
                sanitized_role = sanitize_input(message['role'])
                sanitized_content = sanitize_input(message['content'])
                prompt += f"{sanitized_role.capitalize()}: {sanitized_content}\n"
            elif isinstance(message, tuple) and len(message) >= 2:
                sanitized_role = sanitize_input(str(message[0]))
                sanitized_content = sanitize_input(message[1])
                prompt += f"{sanitized_role.capitalize()}: {sanitized_content}\n"
            else:
                self.logger.warning(f"Format de message inconnu dans l'historique: {type(message)}")
        
        # Traiter les cours pertinents
        prompt += '\nFormations pertinentes:\n'
        if relevant_courses:
            for course in relevant_courses:
                if not isinstance(course, dict):
                    self.logger.warning(f"Entrée de cours invalide ignorée: {str(course)}")
                    continue
                
                # Déterminer la source
                source = course.get('_source', 'internal')
                title = course.get('titre') or course.get('title') or 'Titre inconnu'
                sanitized_title = sanitize_input(title)
                
                if source == 'rncp':
                    sanitized_title += ' (RNCP)'
                prompt += f"- {sanitized_title}\n"
                
                # Attributs de base
                for attr in ['durée', 'modalité', 'tarif', 'lieu', 'prochaines_sessions', 'certifiant']:
                    if attr in course and course[attr]:
                        sanitized_attr = sanitize_input(attr)
                        sanitized_value = sanitize_input(str(course[attr]))
                        prompt += f"    - {sanitized_attr.capitalize()}: {sanitized_value}\n"
                
                # Listes détaillées
                for list_attr in ['objectifs', 'prerequis', 'public', 'programme']:
                    if list_attr in course and course[list_attr]:
                        sanitized_attr = sanitize_input(list_attr)
                        
                        if isinstance(course[list_attr], list):
                            prompt += f"    - {sanitized_attr.capitalize()}:\n"
                            for item in course[list_attr]:
                                sanitized_item = sanitize_input(str(item))
                                prompt += f"      - {sanitized_item}\n"
                        else:
                            sanitized_value = sanitize_input(str(course[list_attr]))
                            prompt += f"    - {sanitized_attr.capitalize()}: {sanitized_value}\n"
                
                # Mention interne ou externe
                if source == 'internal':
                    prompt += "    - Formation interne proposée par Beyond Expertise\n"
                else:
                    prompt += "    - Formation externe (certifiée RNCP)\n"
                    lien = course.get('lien')
                    if lien:
                        sanitized_lien = sanitize_input(lien)
                        prompt += f"    - Lien: {sanitized_lien}\n"
        else:
            prompt += "- Aucune formation interne pertinente trouvée.\n"
        
        prompt += "\nRépondez au format JSON avec les clés suivantes: answer, recommended_course, next_action."
        self.logger.debug(f"Prompt construit: {prompt[:200]}...")
        return prompt
    
    def detect_intent(self, question: str, chat_history=None) -> str:
        """Détecte l'intention de l'utilisateur à partir de la question avec assainissement des entrées."""
        # Assainir la question
        sanitized_question = sanitize_input(question)
        
        # Traiter l'historique du chat pour le contexte
        context = ""
        if chat_history and len(chat_history) >= 2:
            last_messages = chat_history[-2:]
            for msg in last_messages:
                if isinstance(msg, dict) and 'content' in msg:
                    sanitized_role = sanitize_input(msg.get('role', 'unknown'))
                    sanitized_content = sanitize_input(msg.get('content', ''))
                    context += f"{sanitized_role}: {sanitized_content}\n"
        
        # Construire le prompt de détection d'intention
        prompt = f"""
            Tu es un classificateur d'intentions. Ton rôle est d'identifier l'intention exacte exprimée dans la question de l'utilisateur, afin de sélectionner une rubrique spécifique dans un fichier de formation structuré (.json).
            
            Historique récent de la conversation (pour le contexte):
            {context}
            
            Voici les intentions possibles :
            - liste_formations: L'utilisateur demande **uniquement la liste complète** des formations sans aucun critère de filtrage.
            Exemples: "Quelles sont les formations?", "Donne-moi la liste des formations", "Montre-moi toutes les formations proposées"
            
            - recommandation: L'utilisateur cherche une formation adaptée à son profil ou ses besoins.
            
            - info_objectifs: L'utilisateur demande des informations sur les objectifs d'une formation.
            
            - info_prerequis: L'utilisateur s'interroge sur les prérequis d'une formation.
            
            - info_programme: L'utilisateur veut connaître le contenu/programme d'une formation.
            
            - info_public: L'utilisateur demande quel est le public cible d'une formation.
            
            - info_tarif: L'utilisateur demande le **coût**, le **prix brut** ou les **réductions tarifaires** éventuelles.
            
            - info_financement: L'utilisateur s'intéresse aux **modes de financement** disponibles (CPF, France Travail, Pôle Emploi, OPCO, etc.).
            
            - info_duree: L'utilisateur veut connaître la durée d'une formation.
            
            - info_modalite: L'utilisateur demande le format des formations (présentiel/distanciel).
            Exemples: "Quelles formations sont à distance?", "Lesquelles sont en présentiel?"
            
            - info_certification: L'utilisateur demande spécifiquement quelles formations sont certifiantes ou diplômantes.
            Exemples: "Quelles formations sont certifiantes?", "Y a-t-il des formations diplômantes?"
            
            - info_prochaine_session: L'utilisateur veut connaître les dates des prochaines sessions.
            
            - recherche_filtrée: L'utilisateur demande **une liste filtrée** de formations selon des critères autres que ceux déjà définis.
            Exemples: "Quelles sont les formations pour débutants?", "Formations les moins chères?", "Formations en ligne?", "Formations certifiantes?"
            
            - info_lieu: L'utilisateur demande des informations sur le **lieu** de la formation.
            
            - fallback: Toute question qui a un lien avec les formations mais ne correspond à aucun des intents listés.
            
            - none: Question sans rapport avec le domaine de la formation.

            Question de l'utilisateur: {sanitized_question}
            
            Attention aux distinctions suivantes:
            - Si la question demande des informations sur les objectifs, prérequis, programme, public, tarif, financement, durée ou lieu d'une formation, choisir l'intention correspondante.
            - Si la question demande toutes les formations sans filtre, choisir "liste_formations"
            - Si la question demande toutes les formations mais avec filtre ( certifiés, à distance, .. etc ), choisir "recherche_filtrée"
            - Si la question demande des informations sur une formation SPÉCIFIQUE et sa certification, choisir "info_certification"
            Exemple: "Est-ce que la formation Python est certifiante?"

            - Si la question demande une LISTE de formations certifiantes, choisir "recherche_filtrée"
            Exemple: "Quelles sont les formations certifiantes?"
            
            Réponds uniquement par le nom exact de l'intention la plus adaptée.
            """
        
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            intent = response.choices[0].message.content.strip().lower()
            
            # Règles de correction pour les cas problématiques
            question_lower = sanitized_question.lower()
            
            # # Corrections spécifiques pour les cas problématiques identifiés
            # if "toutes les formations" in question_lower and "beyond expertise" in question_lower and "certifi" not in question_lower and "distance" not in question_lower and "présentiel" not in question_lower:
            #     intent = "liste_formations"
            # elif "certifi" in question_lower and "formation" in question_lower:
            #     intent = "info_certification"
            # elif any(mot in question_lower for mot in ["distance", "présentiel", "sur site", "en ligne"]):
            #     intent = "info_modalite"
            
            self.logger.info(f"Intention détectée: {intent}")
            return intent
        except Exception as e:
            self.logger.error(f"Erreur lors de la détection d'intention: {str(e)}")
            return "fallback"
            
    def generate_response(self, question, chat_history=None, profile=None, session=None):
        """
        Génère une réponse personnalisée en fonction de la question, de l'historique du chat,
        du profil utilisateur et de l'intention détectée.
        """
        self.logger.info(f"Traitement question: {question[:50]}...")
        
        # Assainir l'entrée
        sanitized_question = sanitize_input(question)
        
        # Initialisation sécurisée
        if chat_history is None:
            chat_history = []
        
        # Récupération du contexte pertinent
        relevant_courses = self._retrieve_content(sanitized_question, top_k=3)
        
        # Construction du prompt pour l'API
        prompt = self._build_prompt(profile, relevant_courses, chat_history)
        
        # Appel à l'API pour générer la réponse
        try:
            # Préparation des messages pour l'API
            system_msg = {"role": "system", "content": prompt}
            history_msgs = chat_history
            user_msg = {"role": "user", "content": sanitized_question}
            messages = [system_msg] + history_msgs + [user_msg]
            
            # Appel à l'API
            response = client.chat.completions.create(
                model='gpt-3.5-turbo-1106',
                messages=messages,
                temperature=0.7,
                response_format={"type": "json_object"},
                max_tokens=1000
            )
            
            response_content = response.choices[0].message.content
            
            # Parsing de la réponse
            try:
                parsed = json.loads(response_content)
                
                # Validation et nettoyage du format
                if 'answer' not in parsed or not isinstance(parsed['answer'], str):
                    parsed['answer'] = "Désolé, je n'ai pas pu générer une réponse structurée. Pourriez-vous reformuler votre question?"
                
                if 'next_action' not in parsed:
                    intent = self.detect_intent(sanitized_question, chat_history)
                    parsed['next_action'] = intent
                    
                # Ajouter l'intention à la réponse
                if 'intent' not in parsed:
                    intent = self.detect_intent(sanitized_question, chat_history)
                    parsed['intent'] = intent
                
                return parsed
                
            except json.JSONDecodeError as e:
                self.logger.error(f"Erreur parsing JSON: {str(e)}")
                intent = self.detect_intent(sanitized_question, chat_history)
                return {
                    "answer": "Je n'ai pas pu générer une réponse structurée. Pourriez-vous reformuler votre question?",
                    "recommended_course": None,
                    "next_action": intent,
                    "intent": intent
                }
        except Exception as e:
            self.logger.error(f"Erreur génération réponse: {str(e)}", exc_info=True)
            intent = self.detect_intent(sanitized_question, chat_history)
            return {
                "answer": "Désolé, une erreur technique est survenue. Veuillez réessayer.",
                "recommended_course": None,
                "next_action": "error",
                "intent": intent
            }
            

    def handle_filtered_search(self, question, intent=None):
        """
        Méthode qui extrait les filtres d'une question et récupère les formations correspondantes.
        
        Args:
            question: La question de l'utilisateur
            intent: L'intention détectée (optionnel)
            
        Returns:
            Dict avec les filtres et les formations correspondantes pour enrichir le contexte LLM
        """
        import json
        
        # Assainir l'entrée
        sanitized_question = sanitize_input(question)
        
        # 1. Extraire les filtres de la question avec LLM
        extract_prompt = f"""
        Tu es un assistant qui doit extraire les critères de filtrage d'une recherche de formations.
        
        Question de l'utilisateur: "{sanitized_question}"
        
        Les filtres possibles sont:
        - certifiante (formation avec certification/diplôme)
        - modalité (présentiel, distanciel, hybride, en ligne)
        - durée (courte, longue, nombre de jours/heures)
        - niveau (débutant, intermédiaire, avancé)
        - thématique (data, dev, IA, management, etc.)
        
        Réponds au format JSON avec uniquement les filtres trouvés:
        {{
            "filtres": {{"nom_du_filtre": "valeur", ...}}
        }}
        """
        
        # Utiliser l'API OpenAI pour extraire les filtres
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": extract_prompt}],
                response_format={"type": "json_object"},
                temperature=0.1
            )
            
            # Extraire les filtres
            extracted_filters = json.loads(response.choices[0].message.content)
            filters = extracted_filters.get("filtres", {})
        except Exception as e:
            self.logger.error(f"Erreur lors de l'extraction des filtres: {str(e)}")
            filters = {}
        
        # 2. Obtenir les formations correspondantes
        # Utiliser le service de données existant
        all_courses = self.data_service.get_all_courses()
        
        # Filtrer les formations
        filtered_courses = []
        
        for course in all_courses:
            # Vérifier si la formation correspond à tous les filtres
            match = True
            
            for filter_name, filter_value in filters.items():
                filter_name_lower = filter_name.lower()
                filter_value_lower = filter_value.lower()
                
                if filter_name_lower in ["certifiante", "certifiant"]:
                    # Cas spécial pour les formations certifiantes
                    is_certifying = course.get("certifiant", False) or course.get("_source") == "rncp"
                    if filter_value_lower in ["oui", "true", "1", "vrai", "yes"] and not is_certifying:
                        match = False
                        break
                    elif filter_value_lower in ["non", "false", "0", "faux", "no"] and is_certifying:
                        match = False
                        break
                
                elif filter_name_lower in ["modalité", "modalite"]:
                    # Vérifier la modalité
                    course_modality = course.get("modalité", "").lower()
                    if filter_value_lower not in course_modality:
                        match = False
                        break
                
                elif filter_name_lower in ["durée", "duree"]:
                    # Vérifier la durée (logique simplifiée)
                    course_duration = course.get("durée", "").lower()
                    if filter_value_lower not in course_duration:
                        match = False
                        break
                
                elif filter_name_lower in ["thématique", "thematique"]:
                    # Vérifier si la thématique est dans le titre ou les objectifs
                    theme = filter_value_lower
                    title = course.get("titre", "").lower()
                    objectives = str(course.get("objectifs", "")).lower()
                    
                    if theme not in title and theme not in objectives:
                        match = False
                        break
                
                elif filter_name_lower == "niveau":
                    # Vérifier le niveau dans les prérequis ou le titre
                    level = filter_value_lower
                    title = course.get("titre", "").lower()
                    prerequisites = str(course.get("prerequis", "")).lower()
                    public = str(course.get("public", "")).lower()
                    
                    if level not in title and level not in prerequisites and level not in public:
                        match = False
                        break
            
            if match:
                filtered_courses.append(course)
        
        # Cas particulier pour les formations certifiantes sans autres filtres
        if len(filters) == 1 and any(key.lower() in ["certifiante", "certifiant"] for key in filters.keys()):
            filter_key = next(key for key in filters.keys() if key.lower() in ["certifiante", "certifiant"])
            if filters[filter_key].lower() in ["oui", "true", "1", "vrai", "yes"]:
                # Vérifier à nouveau explicitement pour les formations certifiantes
                filtered_courses = [
                    course for course in all_courses 
                    if course.get("certifiant", False) == True or course.get("_source") == "rncp"
                ]
                # Limiter le nombre de cours pour éviter un dépassement de contexte
                self.logger.info(f"Trouvé {len(filtered_courses)} formations certifiantes, limitation à 100 pour l'affichage")
                if len(filtered_courses) > 100:
                    # Trier par source (mettre les formations internes en premier)
                    filtered_courses.sort(key=lambda c: 0 if c.get('_source') == 'internal' else 1)
                    # Garder seulement les 100 premières pour le traitement
                    filtered_courses = filtered_courses[:100]
                
        # 3. Construire un contexte enrichi pour le LLM
        context_for_llm = {
            "filtres_detectes": filters,
            "formations_filtrees": [
                {
                    "titre": course.get("titre", ""),
                    "source": "RNCP" if course.get("_source") == "rncp" else "interne",
                    "certifiant": course.get("certifiant", False) or course.get("_source") == "rncp",
                    "modalite": course.get("modalité", ""),
                    "duree": course.get("durée", ""),
                    "course_id": course.get("id", "")
                }
                for course in filtered_courses
            ],
            "nombre_formations": len(filtered_courses),
            "intent": intent or "recherche_filtrée"
        }
        
        self.logger.info(f"Recherche filtrée: {len(filtered_courses)} formations trouvées")
        return context_for_llm



    def generate_response_from_filtered_search(self, question, chat_history=None, profile=None, session=None):
        """
        Génère une réponse basée sur une recherche filtrée de formations.
        
        Args:
            question: La question de l'utilisateur
            chat_history: L'historique du chat (optionnel)
            profile: Le profil utilisateur (optionnel)
            session: La session utilisateur (optionnel)
        
        Returns:
            Dict avec la réponse générée
        """
        # Obtenir le contexte enrichi
        filtered_context = self.handle_filtered_search(question)
        
        # Ajouter une vérification pour les requêtes RNCP sans résultats
        if "rncp" in question.lower() and filtered_context["nombre_formations"] == 0:
            # Préparation du message spécifique
            rncp_message = (
                "Je n'ai pas trouvé de formations RNCP dans notre base concernant votre demande sur la chocolaterie. "
                "Cependant, il pourrait exister des formations certifiantes RNCP dans ce domaine auprès d'autres organismes. "
                "Je vous invite à contacter notre service formation pour plus d'informations à ce sujet."
            )
            
            # Retourner directement sans appeler l'API
            return {
                "answer": rncp_message,
                "recommended_course": None,
                "next_action": "contact_for_rncp",
                "intent": filtered_context['intent']
            }

        # Construire un prompt spécifique pour cette requête filtrée
        filtered_prompt = """Vous êtes un conseiller professionnel expérimenté, orienté résultats et service client, bienveillant et proactif. Répondez en français.
        
        En tant que conseiller formation, vous devez répondre à une requête de recherche filtrée.
        """
        
        if profile:
            filtered_prompt += "\n\nProfil utilisateur:\n"
            profile_dict = profile
            if hasattr(profile, 'model_dump'):
                profile_dict = profile.model_dump()
            elif hasattr(profile, 'dict'):
                profile_dict = profile.dict()
            
            if isinstance(profile_dict, dict):
                for key, value in profile_dict.items():
                    sanitized_key = sanitize_input(key)
                    sanitized_value = sanitize_input(value)
                    filtered_prompt += f"- {sanitized_key.capitalize()}: {sanitized_value}\n"
        
        # Ajouter l'historique du chat (limité)
        if chat_history:
            filtered_prompt += "\nHistorique de la conversation (derniers messages):\n"
            for message in chat_history[-3:]:  # Limiter aux 3 derniers messages
                if isinstance(message, dict) and 'role' in message and 'content' in message:
                    sanitized_role = sanitize_input(message['role'])
                    sanitized_content = sanitize_input(message['content'])
                    filtered_prompt += f"{sanitized_role.capitalize()}: {sanitized_content}\n"
        
        # Ajouter les filtres détectés
        filtered_prompt += "\nVoici les filtres détectés dans la question: \n"
        if filtered_context["filtres_detectes"]:
            for filtre, valeur in filtered_context["filtres_detectes"].items():
                filtered_prompt += f"- {filtre}: {valeur}\n"
        else:
            filtered_prompt += "Aucun filtre spécifique détecté.\n"
        
        # Ajouter les formations filtrées (LIMITÉES)
        total_formations = filtered_context['nombre_formations']
        filtered_prompt += f"\nJ'ai trouvé {total_formations} formations correspondantes.\n"
        
        # Limiter à 10 formations maximum pour éviter le dépassement de contexte
        max_formations_to_show = 10
        formations_to_show = filtered_context["formations_filtrees"][:max_formations_to_show]
        
        if formations_to_show:
            filtered_prompt += f"Voici les {len(formations_to_show)} premières formations (sur un total de {total_formations}):\n"
            for i, formation in enumerate(formations_to_show):
                filtered_prompt += f"{i+1}. {formation['titre']} - "
                filtered_prompt += f"Formation {formation['source']}, "
                filtered_prompt += f"{'certifiante' if formation['certifiant'] else 'non certifiante'}"
                # Limiter les détails supplémentaires pour économiser des tokens
                if formation['modalite']:
                    filtered_prompt += f", {formation['modalite']}"
                filtered_prompt += "\n"
        else:
            filtered_prompt += "Aucune formation ne correspond aux critères spécifiés.\n"
        
        filtered_prompt += f"""
        Veuillez répondre à l'utilisateur en présentant ces formations de manière conversationnelle.
        Si le nombre total est grand ({total_formations} formations), mentionnez qu'il s'agit d'une sélection parmi toutes les formations disponibles.
        Si aucune formation ne correspond, expliquez-le gentiment et proposez des alternatives.
        
        Répondez au format JSON avec les clés suivantes: answer, recommended_course, next_action.
        """
        
        # Assainir l'entrée
        sanitized_question = sanitize_input(question)
        
        # Initialisation sécurisée
        if chat_history is None:
            chat_history = []
        
        # Appel à l'API pour générer la réponse
        try:
            # Préparation des messages pour l'API
            system_msg = {"role": "system", "content": filtered_prompt}
            user_msg = {"role": "user", "content": sanitized_question}
            messages = [system_msg, user_msg]
            
            # Appel à l'API
            response = client.chat.completions.create(
                model='gpt-3.5-turbo-1106',
                messages=messages,
                temperature=0.7,
                response_format={"type": "json_object"},
                max_tokens=1000
            )
            
            response_content = response.choices[0].message.content
            
            # Parsing de la réponse
            try:
                parsed = json.loads(response_content)
                
                # Validation et nettoyage du format
                if 'answer' not in parsed or not isinstance(parsed['answer'], str):
                    parsed['answer'] = f"J'ai trouvé {total_formations} formations correspondant à votre recherche."
                
                # Ajouter l'intention
                parsed['intent'] = filtered_context['intent']
                
                if 'next_action' not in parsed:
                    parsed['next_action'] = "follow_up"
                
                # Si aucun cours recommandé n'est spécifié mais que des formations ont été trouvées
                if ('recommended_course' not in parsed or parsed['recommended_course'] is None) and len(formations_to_show) > 0:
                    # Récupérer la première formation filtrée comme recommandation
                    course_id = formations_to_show[0]['course_id']
                    parsed['recommended_course'] = self._get_course_content(course_id)
                
                return parsed
                
            except json.JSONDecodeError as e:
                self.logger.error(f"Erreur parsing JSON: {str(e)}")
                return {
                    "answer": f"J'ai trouvé {total_formations} formations correspondant à votre recherche. Voici les premières: " + 
                            ", ".join([f.get('titre', '') for f in formations_to_show[:5]]),
                    "recommended_course": None if len(formations_to_show) == 0 else self._get_course_content(formations_to_show[0]['course_id']),
                    "next_action": "follow_up",
                    "intent": filtered_context['intent']
                }
        except Exception as e:
            self.logger.error(f"Erreur génération réponse: {str(e)}", exc_info=True)
            return {
                "answer": "Désolé, une erreur technique est survenue lors de la recherche. De nombreuses formations correspondent à vos critères.",
                "recommended_course": None,
                "next_action": "error",
                "intent": filtered_context['intent']
            }