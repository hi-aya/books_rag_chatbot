import os
from typing import List

from dotenv import load_dotenv
from openai import OpenAI

from src import config

# Charger les variables d'environnement depuis .env
load_dotenv()


def get_client() -> OpenAI:
    """
    Initialise le client OpenAI avec la clé API.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY n'est pas défini dans le .env")

    client = OpenAI(api_key=api_key)
    return client


def build_rag_prompt(question: str, passages: List[str]) -> str:
    """
    Construit le prompt envoyé au LLM, avec :
    - instructions claires
    - le contexte (passages du livre)
    - la question de l'utilisateur
    """
    context = "\n\n---\n\n".join(passages)

    prompt = f"""
Tu es un assistant qui aide l'utilisateur à comprendre un livre.

RÈGLES :
- Tu réponds uniquement en utilisant le CONTEXTE ci-dessous.
- Tu réponds en français.
- Si l'information n'est pas clairement présente dans le contexte, tu dis honnêtement que tu ne peux pas répondre.

CONTEXTE :
{context}

QUESTION :
{question}

Donne une réponse claire, structurée et pédagogique.
"""
    return prompt.strip()


def rag_answer(question: str, passages: List[str]) -> str:
    """
    Prend une question + une liste de passages (texte du livre),
    construit un prompt RAG, appelle le LLM OpenAI et renvoie la réponse texte.
    """
    client = get_client()
    prompt = build_rag_prompt(question, passages)

    # Choix du modèle : depuis config si dispo, sinon valeur par défaut
    model_name = getattr(config, "LLM_MODEL", "gpt-4.1-mini")

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": (
                    "Tu es un assistant pédagogique qui répond uniquement à partir "
                    "du contexte fourni. Si tu ne sais pas, tu le dis."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0.2,
    )

    answer = response.choices[0].message.content.strip()
    return answer
