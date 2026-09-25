"""
Task 5 — Semantic search.

Embed query bằng chính hàm của Task 4, query ChromaDB và đổi cosine distance
thành similarity. Output phải theo SearchResult, sort giảm dần và không quá top_k.
"""

try:
    from src.task4_chunking_indexing import embed_texts, get_collection
except ImportError:
    from .task4_chunking_indexing import embed_texts, get_collection


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """Trả về dense SearchResult theo score giảm dần."""
    if not query.strip() or top_k <= 0:
        return []

    vectors = embed_texts([query])
    if not vectors:
        return []
    query_vector = vectors[0]

    collection = get_collection()
    response = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    ids = response.get("ids", [[]])[0]
    documents = response.get("documents", [[]])[0]
    metadatas = response.get("metadatas", [[]])[0]
    distances = response.get("distances", [[]])[0]

    results: list[dict] = []
    seen_ids: set[str] = set()

    for item_id, content, metadata, distance in zip(ids, documents, metadatas, distances):
        if item_id in seen_ids:
            continue
        seen_ids.add(item_id)

        meta = dict(metadata) if metadata else {}
        if meta.get("url") == "":
            meta["url"] = None

        # Đổi cosine distance thành cosine similarity score (thang 0.0 - 1.0)
        similarity = max(0.0, min(1.0, 1.0 - float(distance)))

        results.append({
            "id": item_id,
            "content": content,
            "score": similarity,
            "metadata": meta,
            "retrieval_method": "dense",
        })

    # Sắp xếp giảm dần theo điểm và giới hạn top_k
    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:top_k]


if __name__ == "__main__":
    for result in semantic_search("test query", top_k=3):
        print(result)
