import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

PROMPT_TEMPLATE = """
CONTEXTO:
{contexto}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{pergunta}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""

def required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"Defina {name} no arquivo .env")
    return value


def search_prompt():
    required_env("GOOGLE_API_KEY")
    embeddings = GoogleGenerativeAIEmbeddings(
        model=required_env("GOOGLE_EMBEDDING_MODEL"),
        task_type="RETRIEVAL_QUERY",
    )
    store = PGVector(
        embeddings=embeddings,
        connection=required_env("DATABASE_URL"),
        collection_name=required_env("PG_VECTOR_COLLECTION_NAME"),
        use_jsonb=True,
        create_extension=False,
    )
    prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)
    model = ChatGoogleGenerativeAI(
        model=required_env("GOOGLE_CHAT_MODEL"),
        max_output_tokens=2048,
    )

    def answer(question: str) -> str:
        if not question.strip():
            raise ValueError("A pergunta não pode ser vazia")

        results = store.similarity_search_with_score(question, k=10)
        if not results:
            raise RuntimeError("Nenhum trecho encontrado. Execute a ingestão do PDF.")

        context = "\n\n".join(
            f"[Página {document.metadata['page'] + 1}]\n{document.page_content}"
            for document, _score in results
        )
        response = model.invoke(
            prompt.format(contexto=context, pergunta=question),
            timeout=60,
            max_retries=2,
        )
        return str(response.content).strip()

    return answer
