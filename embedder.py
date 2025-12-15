import json
from typing import List, Dict

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src import config


def load_chunks(path: str) -> List[Dict]:
    """
    Charge le fichier JSON contenant les chunks.
    Format attendu : [{"id": 0, "text": "..."}, ...]
    """
    print(f"[EMBED] Chargement des chunks depuis : {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"[EMBED] {len(data)} chunks chargés.")
    return data


def get_model(model_name: str = None) -> SentenceTransformer:
    """
    Charge le modèle d'embeddings SentenceTransformer.
    """
    if model_name is None:
        model_name = config.MODEL_NAME

    print(f"[EMBED] Chargement du modèle d'embeddings : {model_name}")
    model = SentenceTransformer(model_name)
    print("[EMBED] Modèle chargé.")
    return model


def compute_embeddings(chunks: List[Dict], model: SentenceTransformer) -> np.ndarray:
    """
    Calcule les embeddings pour la liste de chunks.
    Retourne un array NumPy de dimension (n_chunks, dim_embedding).
    """
    texts = [c["text"] for c in chunks]
    print(f"[EMBED] Calcul des embeddings pour {len(texts)} chunks...")

    # encode() peut gérer une grosse liste directement
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # pratique pour la similarité cosinus
    )

    print(f"[EMBED] Embeddings calculés. Shape = {embeddings.shape}")
    return embeddings


def save_embeddings(embeddings: np.ndarray, path: str) -> None:
    """
    Sauvegarde les embeddings dans un fichier .npy (format NumPy).
    """
    print(f"[EMBED] Sauvegarde des embeddings dans : {path}")
    np.save(path, embeddings)
    print("[EMBED] Sauvegarde terminée.")


def main():
    # 1) Charger les chunks
    chunks = load_chunks(config.CHUNKS_FILE)

    if not chunks:
        print("[EMBED] Aucun chunk trouvé. As-tu bien lancé src.ingest avant ?")
        return

    # 2) Charger le modèle d'embeddings
    model = get_model()

    # 3) Calculer les embeddings
    embeddings = compute_embeddings(chunks, model)

    # 4) Sauvegarder
    save_embeddings(embeddings, config.EMBEDDINGS_FILE)

    print("[EMBED] Pipeline terminé avec succès ✅")


if __name__ == "__main__":
    main()