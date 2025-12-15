import os

#
# =======================
# CONFIG GLOBALE DU PROJET
# =======================
#
# --- Modèle LLM OpenAI ---
LLM_MODEL = "gpt-4.1-mini"   # ou autre modèle OpenAI auquel tu as accès

# --- Dossiers principaux ---
BASE_DIR = os.path.dirname(os.path.dirname(__file__))        # dossier principal du projet
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

# --- Fichiers ---
PDF_FILE = os.path.join(RAW_DIR, "CAMUS.pdf")            # Remplace par ton nom de fichier PDF
CHUNKS_FILE = os.path.join(PROCESSED_DIR, "chunks.json")
EMBEDDINGS_FILE = os.path.join(PROCESSED_DIR, "embeddings.npy")

# --- Paramètres RAG ---
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"        # modèle d'embeddings léger et rapide
MAX_CHARS = 800                                              # taille max d'un chunk de texte
TOP_K = 3                                                    # nb de passages à retourner (retrieval)

# --- Vérification automatique des dossiers ---
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

print("[CONFIG] Chargée avec succès.")
