"""
Task 4 — Chunking, embedding và indexing.

Hướng dẫn:
    1. Đọc toàn bộ Markdown trong data/standardized/.
    2. Chia văn bản bằng strategy đã chọn.
    3. Embed chunks bằng một provider duy nhất.
    4. Upsert vào ChromaDB với cosine distance.

Mỗi document/chunk phải theo docs/MODULE_CONTRACTS.md. ID cần ổn định để
chạy lại pipeline không tạo dữ liệu trùng. Task 5 phải dùng chung embed_texts().
"""

import os
from pathlib import Path
import re

import chromadb
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from src.contracts import validate_document
except ImportError:
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


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """Thêm embedding vào từng chunk."""
    if not chunks:
        return []
    texts = [chunk["content"] for chunk in chunks]
    vectors = embed_texts(texts)
    for chunk, vector in zip(chunks, vectors):
        chunk["embedding"] = vector
    return chunks


def index_to_vectorstore(chunks: list[dict]) -> None:
    """Upsert chunks vào ChromaDB."""
    if not chunks:
        return

    collection = get_collection()
    ids = [chunk["id"] for chunk in chunks]
    documents = [chunk["content"] for chunk in chunks]
    embeddings = [chunk["embedding"] for chunk in chunks]

    # ChromaDB metadata chỉ chấp nhận str, int, float, bool
    metadatas = []
    for chunk in chunks:
        meta = dict(chunk["metadata"])
        if meta.get("url") is None:
            meta["url"] = ""
        metadatas.append(meta)

    # Upsert theo batch để đảm bảo an toàn với số lượng lớn
    batch_size = 100
    for i in range(0, len(ids), batch_size):
        collection.upsert(
            ids=ids[i : i + batch_size],
            documents=documents[i : i + batch_size],
            embeddings=embeddings[i : i + batch_size],
            metadatas=metadatas[i : i + batch_size],
        )


def run_pipeline() -> list[dict]:
    """Chạy toàn bộ pipeline load, chunk, embed và index."""
    documents = load_documents()
    chunks = chunk_documents(documents)
    embedded_chunks = embed_chunks(chunks)
    index_to_vectorstore(embedded_chunks)
    print(f"Indexed {len(embedded_chunks)} chunks from {len(documents)} documents.")
    return chunks


if __name__ == "__main__":
    run_pipeline()
