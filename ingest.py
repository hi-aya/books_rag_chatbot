import json
from typing import List

import fitz  # PyMuPDF
from tqdm import tqdm

from src import config


def load_pdf_text(pdf_path: str) -> str:
    """
    Charge un PDF et retourne tout le texte concaténé.
    Utilise PyMuPDF (fitz).
    """
    print(f"[INGEST] Chargement du PDF : {pdf_path}")
    doc = fitz.open(pdf_path)

    all_text = []

    for page_num in tqdm(range(len(doc)), desc="Extraction des pages"):
        page = doc[page_num]
        text = page.get_text("text")
        if text:
            all_text.append(text)

    doc.close()
    full_text = "\n".join(all_text)
    print(f"[INGEST] Extraction terminée. Longueur du texte : {len(full_text)} caractères.")
    return full_text


def clean_text(text: str) -> str:
    """
    Nettoyage léger du texte :
    - suppression des espaces en trop
    - normalisation des retours à la ligne
    """
    # Tu peux rajouter d'autres règles ici si besoin plus tard
    text = text.replace("\r", "\n")
    # Remplacer les multiples sauts de ligne par un seul
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")
    return text.strip()


def split_into_chunks(text: str, max_chars: int = 800) -> List[str]:
    """
    Découpe le texte en morceaux ("chunks") d'au plus max_chars caractères,
    en essayant de couper sur des fins de phrases (séparateur = '.').
    """
    print(f"[INGEST] Découpage du texte en chunks (max {max_chars} caractères)...")

    sentences = [s.strip() for s in text.split(".") if s.strip()]
    chunks: List[str] = []
    current = ""

    for sent in sentences:
        # On rajoute le point perdu par le split
        sentence = sent + ". "
        if len(current) + len(sentence) <= max_chars:
            current += sentence
        else:
            if current:
                chunks.append(current.strip())
            current = sentence

    if current:
        chunks.append(current.strip())

    print(f"[INGEST] Découpage terminé. Nombre de chunks : {len(chunks)}")
    return chunks


def save_chunks(chunks: List[str], path: str) -> None:
    """
    Sauvegarde la liste de chunks dans un fichier JSON.
    Format : [{"id": 0, "text": "..."}, ...]
    """
    print(f"[INGEST] Sauvegarde des chunks dans : {path}")

    data = [
        {"id": i, "text": chunk}
        for i, chunk in enumerate(chunks)
    ]

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"[INGEST] Sauvegarde terminée. {len(chunks)} chunks écrits.")


def main():
    # 1) Charger le texte du PDF
    full_text = load_pdf_text(config.PDF_FILE)

    # 2) Nettoyer légèrement
    cleaned = clean_text(full_text)

    # 3) Découper en chunks
    chunks = split_into_chunks(cleaned, max_chars=config.MAX_CHARS)

    # 4) Sauvegarder dans data/processed/chunks.json
    save_chunks(chunks, config.CHUNKS_FILE)

    print("[INGEST] Pipeline terminé avec succès ✅")


if __name__ == "__main__":
    main()
