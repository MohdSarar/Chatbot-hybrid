import requests

# URL de l'API FastAPI locale
url = "http://localhost:8000/query"

# Prompt de test
data = {
    "prompt": "Quels sont les objectifs de la formation Power BI ?"
}

# Envoi de la requête POST
response = requests.post(url, json=data)

# Affichage du résultat
if response.status_code == 200:
    print("✅ Réponse du LLM :")
    print(response.json()["response"])
else:
    print(f"❌ Erreur {response.status_code} : {response.text}")
