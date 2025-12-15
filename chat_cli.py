from textwrap import fill

from src import config
from src.retriever import load_index, search
from src.llm_client import rag_answer


def format_block(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60 + "\n")


def format_text(text: str, width: int = 100) -> str:
    return fill(text, width=width)


def chat_loop(show_passages: bool = False):
    format_block("🤖 Chat avec ton livre (RAG + OpenAI)")
    print("Tape 'quit' ou 'exit' pour sortir.\n")

    # Charger index (chunks + embeddings + modèle d'embeddings)
    chunks, embeddings, model = load_index()

    while True:
        question = input("👤 Toi : ").strip()
        if question.lower() in ("quit", "exit", "q"):
            print("👋 Au revoir !")
            break

        if not question:
            continue

        # 1) Récupérer les passages pertinents
        results = search(question, chunks, embeddings, model, top_k=config.TOP_K)

        if not results:
            print("🤖 Assistant : Je n'ai trouvé aucun passage pertinent dans le livre.\n")
            continue

        passages_text = [r["text"] for r in results]

        # (optionnel) afficher les passages utilisés
        if show_passages:
            format_block("📚 Passages utilisés")
            for i, r in enumerate(results, start=1):
                print(f"--- Passage {i} (score={r['score']:.3f}) ---")
                print(format_text(r["text"]))
                print()

        # 2) Appeler le LLM avec le contexte (RAG complet)
        print("🤖 Assistant : (je réfléchis avec le livre...)\n")
        try:
            answer = rag_answer(question, passages_text)
        except Exception as e:
            print("❌ Erreur lors de l'appel au LLM :", e)
            continue

        format_block("🤖 Réponse")
        print(format_text(answer))
        print()  # ligne vide


def main():
    # show_passages=False → ne montre que la réponse finale
    # mets True si tu veux voir aussi les passages récupérés
    chat_loop(show_passages=False)


if __name__ == "__main__":
    main()
