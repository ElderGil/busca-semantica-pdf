import os
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


class IndividualGoogleEmbeddings(GoogleGenerativeAIEmbeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # O endpoint em lote pode não estar disponível para a chave gratuita.
        vectors = []
        for index, text in enumerate(texts, start=1):
            vectors.append(self.embed_query(text, task_type="RETRIEVAL_DOCUMENT"))
            if index % 10 == 0 or index == len(texts):
                print(f"Embeddings gerados: {index}/{len(texts)}", flush=True)
        return vectors


def required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"Defina {name} no arquivo .env")
    return value


def ingest_pdf() -> None:
    pdf_path = Path(required_env("PDF_PATH"))
    if not pdf_path.is_absolute():
        pdf_path = PROJECT_ROOT / pdf_path
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF não encontrado: {pdf_path}")

    required_env("GOOGLE_API_KEY")
    database_url = required_env("DATABASE_URL")
    collection_name = required_env("PG_VECTOR_COLLECTION_NAME")
    embedding_model = required_env("GOOGLE_EMBEDDING_MODEL")

    pages = PyPDFLoader(str(pdf_path)).load()
    if not pages:
        raise ValueError("O PDF não contém páginas para ingestão")
    for page in pages:
        page.metadata["source"] = pdf_path.name

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        add_start_index=True,
    )
    chunks = splitter.split_documents(pages)
    if not chunks:
        raise ValueError("O PDF não contém texto para ingestão")

    ids = [
        str(uuid5(NAMESPACE_URL, f"{collection_name}:{pdf_path.name}:{index}"))
        for index in range(len(chunks))
    ]
    print(f"PDF: {pdf_path.name}; páginas: {len(pages)}; trechos: {len(chunks)}")

    embeddings = IndividualGoogleEmbeddings(
        model=embedding_model,
        task_type="RETRIEVAL_DOCUMENT",
    )
    store = PGVector(
        embeddings=embeddings,
        connection=database_url,
        collection_name=collection_name,
        use_jsonb=True,
        create_extension=False,
    )
    store.add_documents(chunks, ids=ids)
    print(f"Ingestão concluída na coleção {collection_name}.")


if __name__ == "__main__":
    ingest_pdf()
