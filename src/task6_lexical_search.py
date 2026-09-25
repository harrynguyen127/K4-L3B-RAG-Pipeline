"""
Task 6 — Lexical search bằng BM25.

Dùng cùng corpus chunks với Task 5. BM25 phù hợp với từ khóa chính xác, mã tài
liệu và tên riêng. Output phải theo SearchResult và sort score giảm dần.
"""


import math
from rank_bm25 import BM25Okapi


class RobustBM25Okapi(BM25Okapi):
    """BM25Okapi với công thức Robertson/Lucene IDF (+1) để đảm bảo điểm > 0 trên corpus nhỏ."""

    def _calc_idf(self, nd):
        idf_sum = 0
        for word, freq in nd.items():
            # Robertson-Sparck Jones formula with +1 floor: log((N - n + 0.5)/(n + 0.5) + 1)
            idf = math.log((self.corpus_size - freq + 0.5) / (freq + 0.5) + 1.0)
            self.idf[word] = idf
            idf_sum += idf
        self.average_idf = (idf_sum / len(self.idf)) if self.idf else 0.0


CORPUS: list[dict] = []
_CACHED_CORPUS_KEY: tuple[int, int] | None = None
_CACHED_BM25: RobustBM25Okapi | None = None


def build_bm25_index(corpus: list[dict]) -> RobustBM25Okapi | None:
    """Tạo BM25 index từ cùng corpus chunks của Task 4."""
    if not corpus:
        return None
    tokenized = [item["content"].lower().split() for item in corpus]
    return RobustBM25Okapi(tokenized)


def _get_bm25_index(corpus: list[dict]) -> BM25Okapi | None:
    """Lấy hoặc tạo cached BM25 index để tránh khởi tạo lại nhiều lần."""
    global _CACHED_CORPUS_KEY, _CACHED_BM25
    if not corpus:
        return None
    current_key = (id(corpus), len(corpus))
    if _CACHED_CORPUS_KEY == current_key and _CACHED_BM25 is not None:
        return _CACHED_BM25
    _CACHED_BM25 = build_bm25_index(corpus)
    _CACHED_CORPUS_KEY = current_key
    return _CACHED_BM25


def _ensure_corpus() -> None:
    """Tự động tải corpus từ Task 4 nếu chưa được nạp sẵn."""
    global CORPUS
    if not CORPUS:
        try:
            from src.task4_chunking_indexing import chunk_documents, load_documents
            CORPUS = chunk_documents(load_documents())
        except Exception:
            pass


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """Trả về BM25 SearchResult theo score giảm dần."""
    _ensure_corpus()
    if not CORPUS or top_k <= 0:
        return []

    tokens = query.strip().lower().split()
    if not tokens:
        return []

    bm25 = _get_bm25_index(CORPUS)
    if bm25 is None:
        return []

    scores = bm25.get_scores(tokens)
    scored_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True,
    )

    seen_ids: set[str] = set()
    results: list[dict] = []
    for index in scored_indices:
        if scores[index] <= 0:
            continue
        item = CORPUS[index]
        item_id = item["id"]
        if item_id in seen_ids:
            continue
        seen_ids.add(item_id)
        results.append({
            "id": item_id,
            "content": item["content"],
            "score": float(scores[index]),
            "metadata": item["metadata"],
            "retrieval_method": "bm25",
        })
        if len(results) >= top_k:
            break

    return results


if __name__ == "__main__":
    for result in lexical_search("test query", top_k=3):
        print(result)
