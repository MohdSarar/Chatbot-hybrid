# app/utils.py
"""
Utilitaires centralisés pour l'application de chatbot de formation.
Contient des fonctions d'assainissement d'entrées, de gestion de fichiers
et un service de données partagé.
"""

import html
import json
import logging
import re
import time
import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import app.globals as globs
engine = globs.llm_engine

logger = logging.getLogger(__name__)

# --------------------------------------------------
# Fonctions d'assainissement des entrées
# --------------------------------------------------

def sanitize_input(text, max_length=1000):
    """
    Assainit l'entrée utilisateur pour prévenir les attaques par injection.
    
    Args:
        text: Texte à assainir
        max_length: Longueur maximale du texte à conserver
        
    Returns:
        Texte assaini
    """
    if not text or not isinstance(text, str):
        return ""
        
    # Échappement HTML
    text = html.escape(text)
    
    # Suppression des caractères de contrôle
    text = ''.join(c for c in text if c.isprintable() or c.isspace())
    
    # Ne garder que les alphanumériques et la ponctuation de base
    pattern = re.compile(r'[^\w\s.,;:!?()[\]{}\'"-]')
    text = pattern.sub('', text)
    
    # Tronquer si trop long
    if len(text) > max_length:
        text = text[:max_length]
        
    return text.strip()

# utils.py  (nouveau code)
def normalize_course(course: dict) -> dict:
    """
    Harmonise les clés et complète les valeurs manquantes d'un cours.
    - corrige les variantes d'orthographe ('duree' ➜ 'durée', etc.)
    - force la présence d'un champ 'id' et d'un champ 'certifiant'
    """
    if not isinstance(course, dict):
        return {}

    mapping = {
        "duree": "durée",
        "duration": "durée",
        "modalite": "modalité",
        "title": "titre"
    }
    for old, new in mapping.items():
        if old in course and new not in course:
            course[new] = course.pop(old)
    
    # Champs obligatoires avec valeurs par défaut
    course.setdefault("id", course.get("titre", "").lower().replace(" ", "_"))
    course.setdefault("certifiant", False)
    course.setdefault("durée", "Non spécifiée")
    course.setdefault("modalité", "Non spécifiée")
    course.setdefault("lieu", "Non spécifié")
    course.setdefault("prochaines_sessions", "Non spécifiées")

    return course



# --------------------------------------------------
# Fonctions de gestion des fichiers
# --------------------------------------------------

def check_file_modifications(
    content_dir: Union[Path, str], 
    last_update_times: Dict[str, float] = None, 
    check_interval: int = 3600,  # Augmenté à 1 heure (3600 secondes)
    force_check: bool = False
) -> Tuple[bool, Dict[str, float]]:
    """
    Vérifie si des fichiers ont été modifiés depuis la dernière vérification,
    mais seulement périodiquement pour éviter les vérifications excessives.
    
    Args:
        content_dir: Répertoire contenant les fichiers à surveiller
        last_update_times: Dictionnaire des derniers temps de mise à jour
        check_interval: Intervalle en secondes entre les vérifications (défaut: 1h)
        force_check: Forcer la vérification même si l'intervalle n'est pas atteint
        
    Returns:
        Tuple contenant (a changé, dictionnaire mis à jour)
    """
    # Conversion du chemin en Path si nécessaire
    if isinstance(content_dir, str):
        content_dir = Path(content_dir)
        
    if last_update_times is None:
        # Initialiser un dictionnaire vide plutôt que de charger depuis un fichier
        last_update_times = {}
    
    current_time = time.time()
    last_check = last_update_times.get('_last_check_time', 0)
    
    # Ne vérifier que périodiquement ou si forcé
    if not force_check and (current_time - last_check < check_interval):
        return False, last_update_times
    
    logger.debug(f"Vérification des fichiers modifiés après {current_time - last_check} secondes")
    
    has_updates = False
    last_update_times['_last_check_time'] = current_time
    
    # Vérifier les fichiers JSON normaux
    for json_file in content_dir.glob('**/*.json'):
        try:
            last_mod = json_file.stat().st_mtime
            prev_mod = last_update_times.get(str(json_file), 0)
            if last_mod > prev_mod:
                has_updates = True
                last_update_times[str(json_file)] = last_mod
                logger.info(f"Fichier modifié détecté: {json_file}")
        except Exception as e:
            logger.error(f"Erreur lors de la vérification du fichier {json_file}: {str(e)}")
    
    # Vérifier également si le dossier Chroma a été modifié (uniquement si un fichier a été modifié)
    chroma_dir = Path("chroma_db")
    if has_updates and chroma_dir.exists():
        try:
            # Uniquement vérifier le fichier metadata.json qui est un bon indicateur des modifications de l'index
            chroma_metadata = chroma_dir / "chroma.sqlite3"
            if chroma_metadata.exists():
                last_mod = chroma_metadata.stat().st_mtime
                prev_mod = last_update_times.get(str(chroma_metadata), 0)
                if last_mod > prev_mod:
                    last_update_times[str(chroma_metadata)] = last_mod
                    logger.info(f"Index Chroma modifié détecté: {chroma_metadata}")
        except Exception as e:
            logger.error(f"Erreur lors de la vérification de l'index Chroma: {str(e)}")
    
    return has_updates, last_update_times

def ensure_dir_exists(directory: Union[str, Path]) -> None:
    """
    S'assure qu'un répertoire existe, le crée si nécessaire.
    
    Args:
        directory: Chemin du répertoire à vérifier/créer
    """
    if isinstance(directory, str):
        directory = Path(directory)
    
    try:
        directory.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Répertoire vérifié/créé: {directory}")
    except Exception as e:
        logger.error(f"Erreur lors de la création du répertoire {directory}: {str(e)}")
        raise

# --------------------------------------------------
# Service de données partagé
# --------------------------------------------------

# Updated DataService class in utils.py



class DataService:
    """Service centralisé pour l'accès aux données et au cache."""
    
    def __init__(self, content_dir='app/content', disable_auto_update=True, enable_rncp=None):
        """
        Initialise le service de données.
        
        Args:
            content_dir: Chemin du répertoire contenant les fichiers de contenu
            disable_auto_update: Désactive les vérifications automatiques de mise à jour des fichiers
            enable_rncp: Active ou désactive l'accès aux formations RNCP
        """
        self.logger = logging.getLogger(__name__ + ".DataService")
        self.logger.info("Initialisation du service de données")
        
        # Conversion du chemin en Path si nécessaire
        self.content_dir = Path(content_dir) if isinstance(content_dir, str) else content_dir
        
        # Désactivation des vérifications automatiques
        self.disable_auto_update = disable_auto_update
        
        # Configuration d'accès aux formations RNCP
        if enable_rncp is None:
            import app.globals as globs
            enable_rncp = globs.enable_rncp
        self.enable_rncp = enable_rncp
        self.logger.info(f"Accès aux formations RNCP: {'activé' if enable_rncp else 'désactivé'}")
        
        # État interne
        self.file_timestamps = {}
        self.content_cache = {}
        self.course_ids = set()
        self.rncp_data = None
        
        # Chargement initial des données (sera effectué uniquement au démarrage)
        self._index_courses(force_load=True)
    
    def _index_courses(self, force_load=False):
        """
        Indexe tous les cours disponibles.
        
        Args:
            force_load: Force le chargement même si les vérifications automatiques sont désactivées
        """
        try:
            self.logger.info("Indexation des cours")
            logger.info(f"Indexation démarrée (force={force_load})")
            # Indexer les cours internes
            course_count = 0
            for json_file in self.content_dir.glob('*.json'):
                self.course_ids.add(json_file.stem)
                course_count += 1
            
            # Indexer les cours RNCP seulement si activé
            if self.enable_rncp:
                rncp_path = self.content_dir / 'rncp' / 'rncp.json'
                if rncp_path.exists():
                    try:
                        # Chargement différé - ne pas charger tout le contenu, juste prendre note du fichier
                        if force_load:
                            with open(rncp_path, 'r', encoding='utf-8') as f:
                                self.rncp_data = json.load(f)
                                if isinstance(self.rncp_data, list):
                                    for course in self.rncp_data:
                                        if isinstance(course, dict) and 'id' in course:
                                            self.course_ids.add(f"rncp_{course['id']}")
                                            course_count += 1
                        else:
                            # Juste noter que le fichier RNCP existe
                            self.rncp_data = None
                    except Exception as e:
                        self.logger.error(f"Erreur lors de l'indexation des cours RNCP: {str(e)}")
            else:
                self.logger.info("Indexation des cours RNCP désactivée")
                self.rncp_data = None
            logger.info(f"Indexation terminée - {course_count} cours")
            self.logger.info(f"{course_count} cours indexés au total")
        except Exception as e:
            self.logger.error(f"Erreur globale lors de l'indexation des cours: {str(e)}")
    def validate_rncp_course(self, course: dict) -> dict:
        """Valide et complète les données d'un cours RNCP"""
        if not isinstance(course, dict):
            return {}
            
        required_fields = {
            'titre': 'Titre non spécifié',
            'objectifs': ['Objectifs non spécifiés'],
            'public': ['Public non spécifié'],
            'durée': 'Durée non spécifiée',
            'modalité': 'Modalité non spécifiée'
        }
        
        for field, default in required_fields.items():
            if not course.get(field):
                course[field] = default
                
        # RNCP est toujours certifiant
        course['certifiant'] = True
        course['_source'] = 'rncp'
        
        return course
    
    def get_course_by_id(self, course_id) -> Optional[Dict[str, Any]]:
        """
        Récupère un cours par ID avec mise en cache.
        
        Args:
            course_id: Identifiant du cours à récupérer
            
        Returns:
            Dictionnaire du cours ou None si non trouvé
        """
        # Vérifier si l'ID du cours est valide (sans vérification des mises à jour)
        if not course_id:
            self.logger.warning(f"ID de cours vide")
            return None
        
        # Si c'est un cours RNCP et que l'accès RNCP est désactivé, retourner None
        if course_id.startswith("rncp_") and not self.enable_rncp:
            self.logger.info(f"Accès au cours RNCP {course_id} refusé - accès RNCP désactivé")
            return None
        
        # Rechercher dans le cache
        if course_id in self.content_cache:
            return self.content_cache[course_id]
        
        # Charger depuis les fichiers
        try:
            if course_id.startswith("rncp_"):
                # Si l'accès RNCP est désactivé, retourner None
                if not self.enable_rncp:
                    return None
                    
                # Charger depuis le fichier RNCP
                rncp_id = course_id[5:]
                if not rncp_id:
                    self.logger.warning("ID RNCP vide")
                    return None
                    
                # Si nous avons déjà chargé les données RNCP, les utiliser
                if self.rncp_data is not None:
                    for course in self.rncp_data:
                        if str(course.get('id')) == str(rncp_id):
                            # Marquer comme RNCP
                            course['_source'] = 'rncp'
                            # Ajouter au cache et retourner
                            self.content_cache[course_id] = course
                            return course
                else:
                    # Charger le fichier RNCP si nécessaire
                    rncp_path = self.content_dir / 'rncp' / 'rncp.json'
                    if rncp_path.exists():
                        with open(rncp_path, 'r', encoding='utf-8') as f:
                            self.rncp_data = json.load(f)
                            for course in self.rncp_data:
                                if str(course.get('id')) == str(rncp_id):
                                    # Marquer comme RNCP
                                    course['_source'] = 'rncp'
                                    # Ajouter au cache et retourner
                                    self.content_cache[course_id] = course
                                    return course
            else:
                # Charger depuis un fichier individuel
                course_path = self.content_dir / f"{course_id}.json"
                if course_path.exists():
                    with open(course_path, 'r', encoding='utf-8') as f:
                        course = json.load(f)
                        # Marquer comme interne
                        course['_source'] = 'internal'
                        # Ajouter au cache et retourner
                        self.content_cache[course_id] = course
                        return course
                else:
                    self.logger.warning(f"Fichier de cours non trouvé: {course_path}")
        except Exception as e:
            self.logger.error(f"Erreur lors du chargement du cours {course_id}: {str(e)}")
        
        return None
    
    def get_filtered_courses(self, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Récupère les cours correspondant aux filtres spécifiés.
        
        Args:
            filters: Dictionnaire de filtres (clé: valeur)
            
        Returns:
            Liste des cours correspondant aux filtres
        """
        all_courses = self.get_all_courses()
        
        if not filters:
            return all_courses
        
        filtered_courses = []
        
        for course in all_courses:
            match = True
            
            for key, value in filters.items():
                # Traiter les filtres booléens
                if key == 'certifiant':
                    if value is True and not course.get('certifiant', False) and course.get('_source') != 'rncp':
                        match = False
                        break
                
                # Traiter les filtres de texte
                elif key in ['modalité', 'durée', 'public']:
                    course_value = course.get(key, '')
                    
                    if isinstance(value, str) and isinstance(course_value, str):
                        if value.lower() not in course_value.lower():
                            match = False
                            break
                    elif isinstance(value, list) and isinstance(course_value, str):
                        if not any(v.lower() in course_value.lower() for v in value):
                            match = False
                            break
                
                # Filtre de source - si RNCP est désactivé, ignorer les cours RNCP
                elif key == 'source':
                    if value == 'rncp' and (not self.enable_rncp or course.get('_source') != 'rncp'):
                        match = False
                        break
                    elif value == 'internal' and course.get('_source') != 'internal':
                        match = False
                        break
            
            if match:
                filtered_courses.append(course)
        
        return filtered_courses

    def get_all_courses(self) -> List[Dict[str, Any]]:
        """
        Récupère tous les cours disponibles.
        
        Returns:
            Liste de tous les cours
        """
        courses = []
        
        # Récupérer les cours internes
        for json_file in self.content_dir.glob('*.json'):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    course = normalize_course(json.load(f))
                    course['_source'] = 'internal'
                    courses.append(course)
            except Exception as e:
                self.logger.error(f"Erreur lors de la lecture du fichier {json_file}: {str(e)}")
        
        # Récupérer les cours RNCP si activé
        if self.enable_rncp:
            if self.rncp_data is None:
                rncp_path = self.content_dir / 'rncp' / 'rncp.json'
                if rncp_path.exists():
                    try:
                        with open(rncp_path, 'r', encoding='utf-8') as f:
                            self.rncp_data = json.load(f)
                    except Exception as e:
                        self.logger.error(f"Erreur lors de la lecture du fichier RNCP: {str(e)}")
                        self.rncp_data = []
            
            if self.rncp_data:
                for course in self.rncp_data:
                    course = normalize_course(course) 
                    course['_source'] = 'rncp'
                    courses.append(course)
        
        return courses

    def check_updates(self, force_check=False) -> bool:
        """
        Vérifie les mises à jour et invalide le cache si nécessaire.
        
        Args:
            force_check: Force la vérification même si les vérifications automatiques sont désactivées
            
        Returns:
            True si des mises à jour ont été détectées, False sinon
        """
        # Si les vérifications automatiques sont désactivées et qu'on ne force pas, retourner False
        if self.disable_auto_update and not force_check:
            return False
        
        has_updates, self.file_timestamps = check_file_modifications(
            self.content_dir, self.file_timestamps, force_check=force_check
        )
        
        if has_updates:
            self.logger.info("Mises à jour détectées, invalidation du cache")
            self.content_cache = {}
            # Réindexer les cours pour prendre en compte les ajouts/suppressions
            self._index_courses()
        
        return has_updates
    
    
    def prepare_course_text(self, course: Dict[str, Any]) -> str:
        """
        Prépare une représentation textuelle standardisée d'un cours.
        
        Args:
            course: Dictionnaire contenant les données du cours
            
        Returns:
            Représentation textuelle du cours
        """
        if not course:
            return ""
            
        parts = [f"Titre: {course.get('titre', 'Sans titre')}"]
        
        # Traiter les attributs de base
        for field in ['objectifs', 'programme', 'prerequis', 'public']:
            content = course.get(field, '')
            if content:
                if isinstance(content, list):
                    field_text = f"{field.capitalize()}: " + ". ".join(str(item) for item in content)
                else:
                    field_text = f"{field.capitalize()}: {content}"
                parts.append(field_text)
        
        # Ajouter des métadonnées supplémentaires
        for field in ['durée', 'modalité', 'tarif', 'lieu', 'certifiant']:
            value = course.get(field, '')
            if value:
                parts.append(f"{field.capitalize()}: {value}")
        
        # Ajouter la source
        source = course.get('_source', 'internal')
        if source == 'rncp':
            parts.append("Source: RNCP (Formation certifiante)")
            
            # Ajouter le lien si disponible
            lien = course.get('lien')
            if lien:
                parts.append(f"Lien: {lien}")
        else:
            parts.append("Source: Beyond Expertise (Formation interne)")
        
        return "\n\n".join(parts)
    
    def get_course_by_title(self, title: str) -> Optional[Dict[str, Any]]:
        """
        Recherche un cours par son titre (recherche approximative).
        
        Args:
            title: Titre du cours à rechercher
            
        Returns:
            Dictionnaire du cours ou None si non trouvé
        """
        if not title:
            return None
            
        title_lower = title.lower().strip()
        
        # Recherche exacte dans les cours internes
        for json_file in self.content_dir.glob('*.json'):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    course = json.load(f)
                    if course.get('titre', '').lower() == title_lower:
                        course['_source'] = 'internal'
                        return course
            except Exception as e:
                self.logger.error(f"Erreur lors de la lecture du fichier {json_file}: {str(e)}")
        
        # Chargement des données RNCP si pas encore chargées
        if self.rncp_data is None:
            rncp_path = self.content_dir / 'rncp' / 'rncp.json'
            if rncp_path.exists():
                try:
                    with open(rncp_path, 'r', encoding='utf-8') as f:
                        self.rncp_data = json.load(f)
                except Exception as e:
                    self.logger.error(f"Erreur lors de la lecture du fichier RNCP: {str(e)}")
                    self.rncp_data = []
        
        # Recherche exacte dans les cours RNCP
        if self.rncp_data:
            for course in self.rncp_data:
                if course.get('titre', '').lower() == title_lower:
                    course['_source'] = 'rncp'
                    return course
        
        # Recherche partielle
        matches = []
        
        # Recherche partielle dans les cours internes
        for json_file in self.content_dir.glob('*.json'):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    course = json.load(f)
                    course_title = course.get('titre', '').lower()
                    if title_lower in course_title:
                        course['_source'] = 'internal'
                        # Calcul d'un score simple de pertinence
                        similarity = len(title_lower) / len(course_title) if course_title else 0
                        matches.append((course, similarity))
            except Exception as e:
                self.logger.error(f"Erreur lors de la lecture du fichier {json_file}: {str(e)}")
        
        # Recherche partielle dans les cours RNCP
        if self.rncp_data:
            for course in self.rncp_data:
                course_title = course.get('titre', '').lower()
                if title_lower in course_title:
                    course['_source'] = 'rncp'
                    # Calcul d'un score simple de pertinence
                    similarity = len(title_lower) / len(course_title) if course_title else 0
                    matches.append((course, similarity))
        
        # Trier par pertinence et retourner le plus pertinent
        if matches:
            matches.sort(key=lambda x: x[1], reverse=True)
            return matches[0][0]
        
        return None
    

    
# Fonction utilitaire pour charger tous les cours en une seule fois
def load_all_courses(content_dir='app/content'):
    """
    Charge tous les cours d'un coup (au démarrage de l'application)
    
    Args:
        content_dir: Répertoire contenant les fichiers de cours
    
    Returns:
        Dictionnaire avec tous les cours indexés par ID
    """
    all_courses = {}
    content_path = Path(content_dir)
    
    # Charger les cours internes
    for json_file in content_path.glob('*.json'):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                course = json.load(f)
                course['_source'] = 'internal'
                all_courses[json_file.stem] = course
                logger.debug(f"Fichier chargé : {json_file.name}")
        except Exception as e:
            logger.error(f"Erreur lors du chargement du fichier {json_file}: {str(e)}")
    
    # Charger les cours RNCP
    rncp_path = content_path / 'rncp' / 'rncp.json'
    if rncp_path.exists():
        try:
            with open(rncp_path, 'r', encoding='utf-8') as f:
                rncp_data = json.load(f)
                for course in rncp_data:
                    if isinstance(course, dict) and 'id' in course:
                        course['_source'] = 'rncp'
                        all_courses[f"rncp_{course['id']}"] = course
        except Exception as e:
            logger.error(f"Erreur lors du chargement du fichier RNCP: {str(e)}")
    
    logger.info(f"{len(all_courses)} formations chargées depuis {content_path}")
    
    return all_courses