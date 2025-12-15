import json
from typing import List, Dict

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from src import config


def load_chunks(path: str) -> List[Dict]:
    """
    Charge les chunks au format :
    [{"id": 0, "text": "..."}, ...]
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def load_embeddings(path: str) -> np.ndarray:
    """
    Charge le fichier .npy contenant les embeddings.
    """
    embeddings = np.load(path)
    return embeddings


def load_model() -> SentenceTransformer:
    """
    Charge le même modèle d'embeddings que pour l'index.
    """
    print(f"[RETRIEVER] Chargement du modèle : {config.MODEL_NAME}")
    model = SentenceTransformer(config.MODEL_NAME)
    print("[RETRIEVER] Modèle chargé.")
    return model


def load_index():
    """
    Charge les chunks, les embeddings et le modèle.
    Retourne (chunks, embeddings, model).
    """
    print("[RETRIEVER] Chargement de l'index...")
    chunks = load_chunks(config.CHUNKS_FILE)
    embeddings = load_embeddings(config.EMBEDDINGS_FILE)
    model = load_model()

    print(f"[RETRIEVER] Index chargé : {len(chunks)} chunks, embeddings shape = {embeddings.shape}")
    return chunks, embeddings, model


def search(
    query: str,
    chunks: List[Dict],
    embeddings: np.ndarray,
    model: SentenceTransformer,
    top_k: int = None,
) -> List[Dict]:
    """
    Recherche sémantique :
    - encode la requête
    - calcule la similarité cosinus avec tous les embeddings de documents
    - retourne les top_k meilleurs passages

    Retourne une liste de dicts :
    [{"id": ..., "text": "...", "score": 0.87}, ...]
    """
    if top_k is None:
        top_k = config.TOP_K

    print(f"[RETRIEVER] Recherche pour la requête : {query!r}")

    # 1) Embedding de la requête
    q_emb = model.encode([query], normalize_embeddings=True)

    # 2) Similarité cosinus avec tous les docs
    sims = cosine_similarity(q_emb, embeddings)[0]  # shape = (n_chunks,)

    # 3) Indices des top_k
    top_idx = np.argsort(-sims)[:top_k]

    results = []
    for i in top_idx:
        results.append(
            {
                "id": chunks[i]["id"],
                "text": chunks[i]["text"],
                "score": float(sims[i]),
            }
        )

    return results


if __name__ == "__main__":
    # Petit test rapide si tu exécutes:
    # python -m src.retriever
    chunks, embeddings, model = load_index()
    query = "De quoi parle ce livre ?"
    results = search(query, chunks, embeddings, model, top_k=3)
    print("\n=== Résultats de test ===")
    for r in results:
        print(f"[score={r['score']:.3f}] {r['text'][:200]}...\n")
