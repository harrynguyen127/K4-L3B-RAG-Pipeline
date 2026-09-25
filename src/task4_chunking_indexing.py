"""Task 4 — Load, chunk, embed and index standardized Markdown."""

from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv

from .contracts import validate_document


load_dotenv()

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
CHUNKING_METHOD = "recursive"

EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "sentence_transformers")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EMBEDDING_DIM = 384

COLLECTION_NAME = "rag_documents"
UPSERT_BATCH_SIZE = 100

_MODEL_INSTANCE = None


def _get_embedding_model():
    global _MODEL_INSTANCE
    if _MODEL_INSTANCE is not None:
        return _MODEL_INSTANCE
    from sentence_transformers import SentenceTransformer
    # Dùng model gọn nhẹ, hiệu năng cao hoặc model cấu hình trong .env
    model_name = EMBEDDING_MODEL if EMBEDDING_MODEL and "bge-m3" not in EMBEDDING_MODEL else "all-MiniLM-L6-v2"
    _MODEL_INSTANCE = SentenceTransformer(model_name)
    return _MODEL_INSTANCE


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Tạo embeddings cho danh sách văn bản theo provider trong .env."""
    if not texts:
        return []

    provider = os.getenv("EMBEDDING_PROVIDER", EMBEDDING_PROVIDER).lower()

    if provider == "openai":
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.embeddings.create(
            input=texts,
            model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        )
        return [item.embedding for item in response.data]

    if provider == "gemini":
        from google import genai
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        result = client.models.embed_content(
            model=os.getenv("EMBEDDING_MODEL", "text-embedding-004"),
            contents=texts,
        )
        return [e.values for e in result.embeddings]

    # Mặc định: sentence_transformers
    model = _get_embedding_model()
    embeddings = model.encode(texts, show_progress_bar=False)
    return embeddings.tolist()


def get_collection():
    """Mở Chroma collection dùng cosine distance."""
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def load_documents() -> list[dict]:
    """Đọc Markdown và trả về danh sách Document theo contract."""
    documents: list[dict] = []
    if not STANDARDIZED_DIR.exists():
        return documents

    for path in sorted(STANDARDIZED_DIR.rglob("*.md")):
        raw_content = path.read_text(encoding="utf-8").strip()
        if not raw_content:
            continue

        doc_type = "legal" if "legal" in str(path.parent) else "news"
        title = path.stem.replace("-", " ").replace("_", " ").title()
        url = None

        # Trích xuất metadata từ phần header của file nếu có
        url_match = re.search(r"\*\*Source:\*\*\s*(https?://[^\s\n]+)", raw_content)
        if url_match:
            url = url_match.group(1).strip()

        title_match = re.search(r"^#\s+(.+)$", raw_content, re.MULTILINE)
        if title_match:
            title = title_match.group(1).strip()

        doc_id = path.relative_to(STANDARDIZED_DIR).as_posix().replace("/", "_").replace(".md", "")

        doc = {
            "id": doc_id,
            "content": raw_content,
            "metadata": {
                "source": path.name,
                "title": title,
                "doc_type": doc_type,
                "url": url,
            },
        }
        validate_document(doc, require_chunk=False)
        documents.append(doc)

    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """Chia Document thành chunks có id và chunk_index liên tục."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks: list[dict] = []

    for doc in documents:
        split_texts = splitter.split_text(doc["content"])
        if not split_texts:
            split_texts = [doc["content"]]

        for index, text in enumerate(split_texts):
            clean_text = text.strip()
            if not clean_text:
                continue

            chunk = {
                "id": f"{doc['id']}::chunk-{index}",
                "content": clean_text,
                "metadata": {
                    "source": doc["metadata"]["source"],
                    "title": doc["metadata"]["title"],
                    "doc_type": doc["metadata"]["doc_type"],
                    "url": doc["metadata"]["url"],
                    "chunk_index": index,
                },
            }
            validate_document(chunk, require_chunk=True)
            chunks.append(chunk)
    return chunks


@lru_cache(maxsize=1)
def _sentence_transformer_model():
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise RuntimeError(
            "sentence-transformers is required for local embeddings; "
            'install project dependencies with: python -m pip install -e ".[dev]"'
        ) from error
    return SentenceTransformer(EMBEDDING_MODEL)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed text with the configured provider; local BGE is the default."""
    if not texts:
        return []
    if any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError("embed_texts only accepts non-empty strings")

    if EMBEDDING_PROVIDER == "sentence_transformers":
        vectors = _sentence_transformer_model().encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vectors.tolist()

    if EMBEDDING_PROVIDER == "openai":
        from openai import OpenAI

        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("Set OPENAI_API_KEY to use OpenAI embeddings")
        client = OpenAI(api_key=api_key)
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
        return [item.embedding for item in response.data]

    if EMBEDDING_PROVIDER == "gemini":
        from google import genai

        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("Set GEMINI_API_KEY to use Gemini embeddings")
        client = genai.Client(api_key=api_key)
        vectors: list[list[float]] = []
        for text in texts:
            response = client.models.embed_content(model=EMBEDDING_MODEL, contents=text)
            if not response.embeddings or not response.embeddings[0].values:
                raise RuntimeError("Gemini returned an empty embedding")
            vectors.append(response.embeddings[0].values)
        return vectors

    raise ValueError(
        f"Unsupported EMBEDDING_PROVIDER={EMBEDDING_PROVIDER!r}; "
        "choose sentence_transformers, openai or gemini"
    )


def get_collection():
    """Open the persistent Chroma collection with cosine distance."""
    import chromadb

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def _chroma_metadata(metadata: dict) -> dict:
    """Chroma only accepts scalar metadata values; omit optional null fields."""
    return {key: value for key, value in metadata.items() if value is not None}


def index_to_vectorstore(chunks: list[dict]) -> None:
    """Upsert chunks in bounded batches and remove stale IDs from prior runs."""
    if not chunks:
        raise ValueError("No chunks to index; refusing to clear or alter the collection")
    for chunk in chunks:
        validate_document(chunk, require_chunk=True)
        embedding = chunk.get("embedding")
        if not isinstance(embedding, list) or not embedding:
            raise ValueError(f"Chunk {chunk['id']} has no embedding")

    collection = get_collection()
    ids = [chunk["id"] for chunk in chunks]
    existing_ids = collection.get(include=["metadatas"])["ids"]
    stale_ids = sorted(set(existing_ids) - set(ids))

    for start in range(0, len(chunks), UPSERT_BATCH_SIZE):
        batch = chunks[start : start + UPSERT_BATCH_SIZE]
        collection.upsert(
            ids=[chunk["id"] for chunk in batch],
            documents=[chunk["content"] for chunk in batch],
            embeddings=[chunk["embedding"] for chunk in batch],
            metadatas=[_chroma_metadata(chunk["metadata"]) for chunk in batch],
        )
    if stale_ids:
        collection.delete(ids=stale_ids)


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """Embed chunks in batches while preserving each chunk's identity."""
    embedded: list[dict] = []
    for start in range(0, len(chunks), UPSERT_BATCH_SIZE):
        batch = chunks[start : start + UPSERT_BATCH_SIZE]
        vectors = embed_texts([chunk["content"] for chunk in batch])
        if len(vectors) != len(batch):
            raise RuntimeError("Embedding provider returned a different number of vectors")
        for chunk, vector in zip(batch, vectors, strict=True):
            embedded.append({**chunk, "embedding": vector})
    return embedded


def run_pipeline() -> None:
    """Load, chunk, embed and index the current standardized corpus."""
    documents = load_documents()
    if not documents:
        raise ValueError(f"No non-empty Markdown documents found in {STANDARDIZED_DIR}")
    chunks = chunk_documents(documents)
    embedded_chunks = embed_chunks(chunks)
    index_to_vectorstore(embedded_chunks)
    print(f"Indexed {len(embedded_chunks)} chunks from {len(documents)} documents")


if __name__ == "__main__":
    run_pipeline()
