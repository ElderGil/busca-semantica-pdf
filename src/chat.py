from search import search_prompt


def main():
    answer = search_prompt()
    print("Faça sua pergunta sobre o PDF. Digite 'sair' para encerrar.")

    while True:
        try:
            question = input("PERGUNTA: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if question.lower() in {"sair", "exit", "quit"}:
            break
        if not question:
            continue

        try:
            print(f"RESPOSTA: {answer(question)}\n")
        except Exception as error:
            print(f"Erro ao responder: {error}\n")


if __name__ == "__main__":
    main()
