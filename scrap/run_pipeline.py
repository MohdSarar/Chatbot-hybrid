import subprocess
import sys

# Dictionnaire des scripts disponibles et leur description
scripts = {
    "1": ("Scraper les données du site web (main.py)", "main.py"),
    "2": ("Nettoyer et enrichir les fichiers JSON (clean.py)", "clean.py"),
    "3": ("Préparer les chunks de texte pour vectorisation (prepare_vectorisation.py)", "prepare_vectorisation.py"),
    "4": ("Vectoriser les textes et stocker dans une base (vectorize_chunks.py)", "vectorize_chunks.py"),
    "5": ("Générer automatiquement le fichier README (README_generator.py)", "README_generator.py"),
    "6": ("Tout exécuter (1 à 5 dans l'ordre)", None),
}

def display_menu():
    print("\nPipeline de traitement des formations - Choisissez une option :")
    for key, (desc, _) in scripts.items():
        print(f"{key}. {desc}")

def run_script(script_path):
    try:
        subprocess.run([sys.executable, script_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Erreur lors de l'exécution de {script_path} : {e}")

def run_all():
    for key in ["1", "2", "3", "4", "5"]:
        desc, script = scripts[key]
        print(f"\n--- {desc} ---")
        run_script(script)

if __name__ == "__main__":
    display_menu()
    choix = input("\nEntrez le numéro de l'étape à exécuter : ").strip()

    if choix in scripts:
        description, script = scripts[choix]
        if choix == "6":
            run_all()
        else:
            print(f"\n--- {description} ---")
            run_script(script)
    else:
        print("Option invalide. Veuillez choisir un numéro valide.")
