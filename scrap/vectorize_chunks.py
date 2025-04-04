
import os
import json
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import chromadb

from chromadb.config import Settings

# Dossier contenant les fichiers de chunks JSON
chunks_dir = Path("content/chunks/")
output_dir = Path("content/vectorized/chroma/")
output_dir.mkdir(parents=True, exist_ok=True)

# Fonction pour charger tous les chunks depuis le dossier
def load_chunks_from_directory(directory):
    all_chunks = []
    for file_path in Path(directory).glob("*.json"):
        with open(file_path, "r") as f:
            data = json.load(f)
            if isinstance(data, list):
                all_chunks.extend(data)
    return all_chunks

# Chargement des chunks
chunks = load_chunks_from_directory(chunks_dir)
print(f"Nombre total de chunks chargés : {len(chunks)}")

# Initialisation du modèle d'embedding
model = SentenceTransformer("all-MiniLM-L6-v2")

# Préparation des données
texts = [chunk["content"] for chunk in chunks]
ids = [chunk["chunk_id"] for chunk in chunks]
metadatas = [{"titre": chunk.get("titre", ""), "source": chunk.get("source", "")} for chunk in chunks]

# Encodage en embeddings
embeddings = model.encode(texts, show_progress_bar=True)

# Initialisation de ChromaDB
client = chromadb.PersistentClient(path=str(output_dir))
collection = client.get_or_create_collection("formations")

# Ajout des embeddings à la collection
if len(embeddings) > 0:
    collection.add(
        documents=texts,
        embeddings=embeddings,
        ids=ids,
        metadatas=metadatas
    )
    print("Embeddings ajoutés à la base Chroma.")
else:
    print("Aucun embedding à ajouter. Vérifiez que les chunks sont valides.")
