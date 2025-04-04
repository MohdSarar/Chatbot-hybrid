import json
from pathlib import Path

# Répertoire contenant les fichiers JSON
json_dir = Path("content/json/formations")

# Récupération de tous les fichiers JSON
json_files = list(json_dir.glob("*.json"))

# Initialisation de la liste des formations
formations = []

# Lecture de chaque fichier JSON
for file in json_files:
    with open(file, "r") as f:
        data = json.load(f)
        formations.append({
            "titre": data.get("titre", "Inconnu"),
            "niveau": data.get("niveau", "Non spécifié"),
            "durée": data.get("durée", "Inconnue"),
            "modalité": data.get("modalité", "Inconnue"),
            "lieu": data.get("lieu", "Inconnu"),
            "tarif": data.get("tarif", "Inconnu")
        })

# Tri des formations par titre
formations.sort(key=lambda x: x["titre"].lower())

# Construction du contenu du README
readme_lines = [
    "# Catalogue des Formations Beyond Expertise",
    "",
    f"Nombre total de formations : **{len(formations)}**",
    "",
    "## Aperçu synthétique",
    "",
    "| Titre | Niveau | Durée | Modalité | Lieu | Tarif |",
    "|-------|--------|--------|-----------|-------|--------|"
]

# Ajout des lignes de tableau
for f in formations:
    ligne = f"| {f['titre']} | {f['niveau']} | {f['durée']} | {f['modalité']} | {f['lieu']} | {f['tarif']} |"
    readme_lines.append(ligne)

# Ajout d’une description finale
readme_lines += [
    "",
    "## Structure des fichiers JSON",
    "",
    "Chaque fichier contient les champs suivants :",
    "- `titre`",
    "- `niveau`",
    "- `durée`",
    "- `modalité`",
    "- `lieu`",
    "- `tarif`",
    "- `objectifs`",
    "- `prerequis`",
    "- `public`",
    "- `programme`",
    "- `prochaines_sessions`",
    "- `resume_html`",
    "",
    "_Ce fichier est généré automatiquement à partir des données JSON extraites._"
]

# Création du contenu final
readme_content = "\n".join(readme_lines)

# Emplacement du README à générer
readme_path = json_dir.parent / "README.md"
readme_path.parent.mkdir(parents=True, exist_ok=True)

# Sauvegarde du README
with open(readme_path, "w") as f:
    f.write(readme_content)

print(f"README généré : {readme_path}")
