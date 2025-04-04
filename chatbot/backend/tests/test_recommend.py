import requests

url = "http://localhost:8000/recommend"

# Exemple de profil utilisateur simulé
profil_utilisateur = {
    "nom": "Jean Dupont",
    "situation_actuelle": "en recherche d'emploi",
    "objectif": "se reconvertir dans l’analyse de données",
    "niveau_actuel": "débutant",
    "connaissances": ["Excel", "HTML"],
    "attentes": "apprendre Python, maîtriser Power BI"
}

response = requests.post(url, json=profil_utilisateur)

if response.status_code == 200:
    data = response.json()
    print("✅ Recommandation reçue :")
    print("Titre :", data.get("recommandation", "Aucune"))
    print("Détails :", data.get("details", "Aucun détail fourni."))
else:
    print("❌ Erreur :", response.status_code)
    print(response.text)
