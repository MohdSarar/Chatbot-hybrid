import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import html
import re

# Imports des utilitaires
from app.utils import sanitize_input, check_file_modifications

# Imports LangChain
from langchain.docstore.document import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversation.memory import ConversationEntityMemory
from langchain.chains import ConversationalRetrievalChain
import app.globals as globs

# Updated LangChainRAGService class in langchain_rag_service.py

class LangChainRAGService:
    def __init__(self, disable_auto_update: bool = True, enable_rncp: bool = None):
        import app.globals as globs
        self.llm_engine = globs.llm_engine     
        self.data_service = globs.data_service
        if self.llm_engine is None:
            raise RuntimeError(
                "LLMEngine doit être initialisé via le lifespan manager (main.py)"
            )
        self.content_dir = self.llm_engine.content_dir
        if enable_rncp is None:
            import app.globals as globs
            enable_rncp = globs.enable_rncp
        
        self.enable_rncp = enable_rncp

        self.logger = logging.getLogger(__name__)
        self.logger.info("Initialisation du service RAG LangChain")
        
        
        self.api_key = os.getenv('OPENAI_API_KEY')
        self.disable_auto_update = disable_auto_update
        
        
        self.logger.info(f"Accès aux formations RNCP: {'activé' if enable_rncp else 'désactivé'}")
        

        
        # Initialiser les composants LangChain
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=self.api_key
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,  # Plus grand pour réduire le nombre de chunks
            chunk_overlap=100,  # Moins de chevauchement
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        self.llm = ChatOpenAI(
            temperature=0.1,
            model_name="gpt-3.5-turbo",
            openai_api_key=self.api_key
        )
        
        # Seront initialisés paresseusement
        self._vector_store = None
        self._chain = None
        self._conv_chain = None  # Chaîne conversationnelle
        self._initialized = False
        
        # Pour l'assainissement des entrées
        self.sanitize_pattern = re.compile(r'[^\w\s.,;:!?()[\]{}\'"-]')
        
        # Suivi des mises à jour de fichiers - initialisé une seule fois
        self._last_update_time = {"_last_check_time": 0}  # Initialisation avec une valeur par défaut
    
    def _load_rncp_documents(self, rncp_path):
        """Charger les documents RNCP à partir du fichier JSON RNCP."""
        docs = []
        
        # Si l'accès RNCP est désactivé, retourner une liste vide
        if not self.enable_rncp:
            self.logger.info("Chargement des documents RNCP ignoré - accès RNCP désactivé")
            return docs
            
        try:
            with open(rncp_path, 'r', encoding='utf-8') as f:
                rncp_data = json.load(f)
                
                for course in rncp_data:
                    if not isinstance(course, dict) or 'titre' not in course:
                        continue
                        
                    # Créer une représentation textuelle
                    text = self._create_document_text(course)
                    
                    # S'assurer que l'ID est présent et valide
                    course_id = str(course.get('ID') or abs(hash(course['titre'])))

                        
                    # Créer les métadonnées
                    metadata = {
                        "source": "rncp",
                        "course_id": f"rncp_{course_id}",
                        "title": course.get('titre', ''),
                        "certifiant": True,  # Les cours RNCP sont certifiants par définition
                        "modalite": course.get('modalité', ''),
                        "duree": course.get('durée', '')
                    }
                    
                    # Ajouter le document
                    docs.append(Document(
                        page_content=text,
                        metadata=metadata
                    ))
        except Exception as e:
            self.logger.error(f"Erreur lors du chargement du fichier RNCP: {str(e)}")
            
        return docs
    
    def _init_if_needed(self, force_init=False):
        """
        Initialiser le vector store et la chaîne de récupération si nécessaire.
        
        Args:
            force_init: Forcer l'initialisation même si déjà initialisé
        """
        # Si déjà initialisé et pas de force_init, retourner
        if self._initialized and not force_init:
            return
        
        # Essayer de charger un index existant depuis un répertoire persistant
        persist_directory = "chroma_db"
        try:
            # Vérifier si le répertoire existe et s'il contient des données
            if Path(persist_directory).exists():
                self._vector_store = Chroma(
                    persist_directory=persist_directory,
                    embedding_function=self.embeddings
                )
                
                # Vérifier s'il y a des données
                try:
                    collection_count = len(self._vector_store.get())
                    if collection_count > 0:
                        self.logger.info(f"Index Chroma chargé depuis {persist_directory} avec {collection_count} documents")
                        self._initialized = True
                        
                        # Créer les chaînes de récupération
                        self._setup_retrieval_chains()
                        return
                except Exception as e:
                    self.logger.error(f"Erreur lors de la vérification de l'index Chroma: {str(e)}")
        except Exception as e:
            self.logger.error(f"Erreur lors du chargement de l'index Chroma: {str(e)}")
        
        # Si nous sommes ici, nous devons créer un nouvel index
        self.logger.info("Création d'un nouvel index Chroma")
        
        # Charger progressivement et traiter les documents par lots
        total_docs = []
        
        # Charger les documents internes par lots
        self.logger.info("Chargement des documents internes...")
        for json_file in self.content_dir.glob('*.json'):
            doc = self._load_single_document(json_file)
            if doc:
                total_docs.append(doc)
        
        # Charger les documents RNCP par lots si activé
        if self.enable_rncp:
            rncp_path = self.content_dir / 'rncp' / 'rncp.json'
            if rncp_path.exists():
                self.logger.info("Chargement des documents RNCP...")
                rncp_docs = self._load_rncp_documents(rncp_path)
                total_docs.extend(rncp_docs)
        else:
            self.logger.info("Chargement des documents RNCP ignoré - accès RNCP désactivé")
        
        self.logger.info(f"Chargement de {len(total_docs)} documents terminé")
        
        # Diviser les documents en chunks
        chunks = self.text_splitter.split_documents(total_docs)
        self.logger.info(f"Création de {len(chunks)} chunks au total")
        
        # Créer l'index Chroma par lots
        try:
            # Créer avec le premier lot
            self._vector_store = Chroma.from_documents(
                documents=chunks[:500] if len(chunks) > 500 else chunks,
                embedding=self.embeddings,
                persist_directory=persist_directory
            )
            
            # Ajouter le reste des chunks par lots si nécessaire
            if len(chunks) > 500:
                batch_size = 500
                for i in range(500, len(chunks), batch_size):
                    end_idx = min(i + batch_size, len(chunks))
                    batch = chunks[i:end_idx]
                    self._vector_store.add_documents(documents=batch)
                    self.logger.info(f"Ajouté lot {i//batch_size + 1}/{(len(chunks) + batch_size - 1)//batch_size} à l'index Chroma")
            
            # Persister l'index
            # self._vector_store.persist()
            self.logger.info(f"Index Chroma créé et sauvegardé dans {persist_directory}")
            
            # Créer les chaînes de récupération
            self._setup_retrieval_chains()
            
            self._initialized = True
        except Exception as e:
            self.logger.error(f"Erreur lors de la création de l'index Chroma: {str(e)}")
            raise

    def retrieve_courses(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Récupérer les cours pertinents pour une requête."""
        # Assainir l'entrée
        sanitized_query = sanitize_input(query)
        
        # Assurer l'initialisation
        self._init_if_needed()
        
        try:
            # Interroger directement le vector store
            if self._vector_store:
                # Utiliser la fonction de similarité pour récupérer les documents pertinents
                docs_and_scores = self._vector_store.similarity_search_with_score(
                    sanitized_query,
                    k=top_k * 2  # Récupérer plus que nécessaire pour filtrer
                )
                
                # Traiter les résultats en objets de cours
                courses = []
                seen_ids = set()
                
                for doc, score in docs_and_scores:
                    course_id = doc.metadata.get('course_id')
                    
                    # Ignorer les doublons
                    if course_id in seen_ids:
                        continue
                    
                    seen_ids.add(course_id)
                    
                    # Ignorer les cours RNCP si accès désactivé
                    if course_id.startswith('rncp_') and not self.enable_rncp:
                        continue
                    
                    # Construire l'objet de cours à partir du document
                    if course_id.startswith('rncp_'):
                        # Cours RNCP (seulement si activé)
                        if self.enable_rncp:
                            course = self._load_rncp_course(course_id[5:])
                            if course:
                                course['_source'] = doc.metadata.get('source', 'internal')
                                courses.append(course)
                    else:
                        # Cours interne
                        course = self._load_internal_course(course_id)
                        if course:
                            course['_source'] = doc.metadata.get('source', 'internal')
                            courses.append(course)
                
                # Retourner les top k cours
                return courses[:top_k]
            
            return []
        except Exception as e:
            self.logger.error(f"Erreur lors de la récupération des cours: {str(e)}")
            return []
    
    def _load_rncp_course(self, rncp_id: str) -> Optional[Dict[str, Any]]:
        """Charger un cours RNCP par ID."""
        # Si l'accès RNCP est désactivé, retourner None
        if not self.enable_rncp:
            return None
            
        # Valider l'ID
        if not rncp_id or rncp_id.isspace():
            self.logger.warning(f"ID RNCP vide ou invalide")
            return None
            
        rncp_path = self.content_dir / 'rncp' / 'rncp.json'
        if not rncp_path.exists():
            return None
            
        try:
            with open(rncp_path, 'r', encoding='utf-8') as f:
                rncp_data = json.load(f)
                
                for course in rncp_data:
                    if course.get('id') == rncp_id:
                        course['_source'] = 'rncp'
                        return course
                
            return None
        except Exception as e:
            self.logger.error(f"Erreur lors du chargement du cours RNCP {rncp_id}: {str(e)}")
            return None
    

    
    def _check_for_updates(self, force_check=False):
        """
        Vérifier si des fichiers de contenu ont été modifiés.
        Désactivé par défaut pour éviter les vérifications excessives.
        
        Args:
            force_check: Forcer la vérification même si les vérifications automatiques sont désactivées
            
        Returns:
            True si des mises à jour ont été détectées, False sinon
        """
        # Si les vérifications automatiques sont désactivées et qu'on ne force pas, retourner False
        if self.disable_auto_update and not force_check:
            return False
            
        has_updates, self._last_update_time = check_file_modifications(
            self.content_dir, 
            self._last_update_time, 
            check_interval=3600,  # Une heure entre les vérifications
            force_check=force_check
        )
        
        # Si des mises à jour sont trouvées, réinitialiser le vector store
        if has_updates and self._initialized:
            self.logger.info("Fichiers de contenu mis à jour, réinitialisation du vector store")
            self._vector_store = None
            self._chain = None
            self._conv_chain = None
            self._initialized = False
        
        return has_updates

    def _load_single_document(self, json_file):
        """Charger un seul document à partir d'un fichier JSON."""
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                course = json.load(f)
                
                if not isinstance(course, dict) or 'titre' not in course:
                    return None
                    
                # Créer une représentation textuelle
                text = self._create_document_text(course)
                
                return Document(
                    page_content=text,
                    metadata={
                        "source": "internal",
                        "course_id": json_file.stem,
                        "title": course.get('titre', ''),
                        "certifiant": course.get('certifiant', False),
                        "modalite": course.get('modalité', ''),
                        "duree": course.get('durée', '')
                    }
                )
        except Exception as e:
            self.logger.error(f"Erreur lors du chargement du fichier {json_file}: {str(e)}")
            return None
            
    
    def _setup_retrieval_chains(self):
        """Configurer les chaînes de récupération après l'initialisation de l'index."""
        # Créer le retrieveur
        retriever = self._vector_store.as_retriever(
            search_kwargs={"k": 3}
        )
        
        # Créer le compresseur pour une meilleure qualité de contexte
        compressor = LLMChainExtractor.from_llm(self.llm)
        
        # Créer le récupérateur de compression
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=retriever
        )
        
        # Créer la chaîne QA
        self._chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=compression_retriever,
            return_source_documents=True
        )
    
    def _create_document_text(self, course: Dict[str, Any]) -> str:
        """Créer une représentation textuelle d'un cours."""
        parts = [f"Titre: {course.get('titre', '')}"]
        
        # Traiter les attributs de base
        for field in ['objectifs', 'programme', 'prerequis', 'public']:
            content = course.get(field, '')
            if content:
                if isinstance(content, list):
                    field_text = f"{field.capitalize()}: " + ". ".join(content)
                else:
                    field_text = f"{field.capitalize()}: {content}"
                parts.append(field_text)
        
        # Ajouter des métadonnées supplémentaires
        for field in ['durée', 'modalité', 'tarif', 'lieu', 'certifiant']:
            value = course.get(field, '')
            if value:
                parts.append(f"{field.capitalize()}: {value}")
        
        return "\n\n".join(parts)
    

    
    def _load_internal_course(self, course_id: str) -> Optional[Dict[str, Any]]:
        """Charger un cours interne par ID."""
        if not course_id:
            return None
            
        file_path = self.content_dir / f"{course_id}.json"
        if not file_path.exists():
            return None
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                course = json.load(f)
                course['_source'] = 'internal'
                return course
        except Exception as e:
            self.logger.error(f"Erreur lors du chargement du cours interne {course_id}: {str(e)}")
            return None
    

    
    def create_conversational_chain(self, memory: Optional[ConversationBufferMemory] = None) -> ConversationalRetrievalChain:
        """Créer une chaîne conversationnelle avec mémoire."""
        # Assurer l'initialisation
        self._init_if_needed()
        
        # Créer la mémoire si non fournie
        if memory is None:
            memory = ConversationBufferMemory(
                memory_key="chat_history",
                input_key="question",
                output_key="answer",
                return_messages=True,
            )
        
        # Créer le récupérateur
        retriever = self._vector_store.as_retriever(
            search_kwargs={"k": 3}
        )
        
        # Créer le prompt de condensation de question
        condense_prompt = PromptTemplate.from_template("""
        Vous êtes un assistant professionnel pour Beyond Expertise.
        Suivez ces instructions pour créer une question autonome :
        1. Utilisez l'historique de chat et la question actuelle
        2. Créez une nouvelle question qui capture l'intention complète de l'utilisateur
        3. La question doit être claire, précise et contenir tous les détails pertinents
        4. La question doit être autonome - quelqu'un sans accès à l'historique doit pouvoir la comprendre
        5. Répondez uniquement avec la question reformulée, sans commentaire ni préambule

        Chat History:
        {chat_history}

        Question actuelle: {question}

        Question autonome reformulée:
        """)
        
        # Créer le prompt QA
        qa_prompt = PromptTemplate.from_template("""
        Vous êtes un conseiller en formation professionnel et expert de Beyond Expertise.
        Utilisez les informations ci-dessous pour répondre à la question de l'utilisateur.
        Si vous ne trouvez pas la réponse dans le contexte, dites-le clairement et proposez des alternatives pertinentes.
        Répondez toujours en français et de manière professionnelle.

        Contexte:
        {context}

        Question: {question}

        Réponse:
        """)
        
        # Créer la chaîne conversationnelle
        return ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            memory=memory,
            condense_question_prompt=condense_prompt,
            combine_docs_chain_kwargs={"prompt": qa_prompt},
            return_source_documents=True,
            output_key="answer"
        )
    
    def query(self, question: str, chat_history=None, memory=None) -> Dict[str, Any]:
        """Interroger le système RAG avec une question (implémentation simplifiée)."""
        # Assainir l'entrée
        sanitized_question = sanitize_input(question)
        
        # Assurer l'initialisation
        self._init_if_needed()
        
        # Récupérer les formations par similarité directe
        courses = self.retrieve_courses(sanitized_question, top_k=5)
        
        # Recherche spécifique pour les chocolatiers
        if 'chocolat' in sanitized_question.lower() or 'patisserie' in sanitized_question.lower():
            self.logger.info("Détection de recherche chocolaterie/pâtisserie")
            
            # Récupérer toutes les formations
            all_courses = []
            
            # 1. Vérifier les formations internes
            for json_file in self.content_dir.glob('*.json'):
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        course = json.load(f)
                        course['_source'] = 'internal'
                        all_courses.append(course)
                except Exception as e:
                    self.logger.error(f"Erreur lors de la lecture: {str(e)}")
            
            # 2. Vérifier les formations RNCP
            rncp_path = self.content_dir / 'rncp' / 'rncp.json'
            if rncp_path.exists() and self.enable_rncp:
                try:
                    with open(rncp_path, 'r', encoding='utf-8') as f:
                        rncp_data = json.load(f)
                        self.logger.info(f"Fichier RNCP chargé: {len(rncp_data)} formations")
                        
                        for course in rncp_data:
                            course['_source'] = 'rncp'
                            all_courses.append(course)
                except Exception as e:
                    self.logger.error(f"Erreur lors du chargement RNCP: {str(e)}")
            
            # Rechercher les formations liées au chocolat
            chocolate_courses = []
            keywords = ['chocolat', 'pâtisserie', 'confiserie', 'boulangerie']
            
            for course in all_courses:
                title = str(course.get('titre', '')).lower()
                objectives = str(course.get('objectifs', '')).lower()
                
                if any(kw in title or kw in objectives for kw in keywords):
                    self.logger.info(f"Formation chocolat trouvée: {course.get('titre')}")
                    chocolate_courses.append(course)
            
            # Si des formations de chocolaterie sont trouvées, les utiliser
            if chocolate_courses:
                courses = chocolate_courses
                self.logger.info(f"{len(chocolate_courses)} formations de chocolaterie trouvées")
        
        # Générer une réponse avec les formations trouvées
        if courses:
            # Construire un prompt informatif avec les formations
            prompt = f"""
            En tant que conseiller en formation professionnel de Beyond Expertise, voici ma réponse à la question:
            "{sanitized_question}"
            
            D'après notre base de données, voici les formations qui pourraient vous intéresser:
            """
            
            for i, course in enumerate(courses):
                title = course.get('titre', 'Formation inconnue')
                source = "RNCP (certification reconnue)" if course.get('_source') == 'rncp' else "Formation interne"
                prompt += f"\n{i+1}. {title} - {source}"
                
                # Ajouter quelques détails
                if 'objectifs' in course:
                    objectives = course['objectifs']
                    if isinstance(objectives, list):
                        prompt += f"\n   Objectif principal: {objectives[0]}"
                    else:
                        prompt += f"\n   Objectif: {objectives}"
                
                if 'durée' in course:
                    prompt += f"\n   Durée: {course['durée']}"
                
                if 'modalité' in course:
                    prompt += f"\n   Modalité: {course['modalité']}"
                
                prompt += "\n"
            
            # Utilisez l'API LLM directement pour générer une réponse
            try:
                llm = self.llm
                response = llm.invoke(prompt)
                
                # Formater la réponse
                return {
                    "answer": response,
                    "recommended_courses": courses[:3],
                    "recommended_course": courses[0] if courses else None,
                    "source_documents": []
                }
            except Exception as e:
                self.logger.error(f"Erreur LLM: {str(e)}")
        
        # Réponse par défaut si aucune formation n'est trouvée
        return {
            "answer": "Je n'ai pas trouvé de formation spécifique correspondant à votre demande dans notre catalogue. Je vous invite à nous contacter directement pour discuter de vos besoins de formation.",
            "recommended_courses": [],
            "recommended_course": None,
            "source_documents": []
        }