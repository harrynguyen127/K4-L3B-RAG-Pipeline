"""
Task 8 — PageIndex vectorless fallback.

Hướng dẫn:
    1. Đọc PAGEINDEX_API_KEY từ .env.
    2. Upload tài liệu ở định dạng PageIndex hỗ trợ.
    3. Cache document IDs để không upload lại.
    4. Parse kết quả thành SearchResult có method pageindex.

PageIndex là dịch vụ ngoài: cần timeout và xử lý lỗi để pipeline không crash.
"""

import os
import json
import logging
import os
import tempfile
import time
from pathlib import Path

from dotenv import load_dotenv

from .contracts import validate_search_results


load_dotenv()

logger = logging.getLogger(__name__)

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
MAPPING_FILE = Path(__file__).parent.parent / "data" / "pageindex_mapping.json"


def _get_pageindex_client():
    """Khởi tạo PageIndexClient nếu có API key."""
    if not PAGEINDEX_API_KEY:
        return None
    try:
        from pageindex import PageIndexClient
        return PageIndexClient(api_key=PAGEINDEX_API_KEY)
    except Exception as exc:
        logger.warning(f"Không thể khởi tạo PageIndexClient: {exc}")
        return None


def _load_doc_mapping() -> dict[str, dict]:
    """Tải mapping lưu trữ giữa file nguồn và doc_id."""
    if not MAPPING_FILE.exists():
        return {}
    try:
        content = MAPPING_FILE.read_text(encoding="utf-8")
        return json.loads(content) if content.strip() else {}
    except Exception as exc:
        logger.warning(f"Không thể đọc mapping file: {exc}")
        return {}


def _save_doc_mapping(mapping: dict[str, dict]) -> None:
    """Lưu mapping giữa file nguồn và doc_id."""
    try:
        MAPPING_FILE.parent.mkdir(parents=True, exist_ok=True)
        MAPPING_FILE.write_text(
            json.dumps(mapping, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception as exc:
        logger.warning(f"Không thể lưu mapping file: {exc}")


def _markdown_to_temp_pdf(md_path: Path) -> Path:
    """Chuyển đổi file Markdown sang PDF tạm thời để upload lên PageIndex."""
    from fpdf import FPDF

    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("helvetica", size=11)

    text = md_path.read_text(encoding="utf-8", errors="ignore")
    # Thay thế các ký tự Unicode không tương thích với font cơ bản
    safe_text = text.encode("latin-1", "replace").decode("latin-1")
    pdf.multi_cell(0, 8, safe_text)

    temp_pdf = Path(tempfile.gettempdir()) / f"{md_path.stem}.pdf"
    pdf.output(str(temp_pdf))
    return temp_pdf


def upload_documents() -> None:
    """Upload tài liệu và lưu document IDs để tái sử dụng."""
    client = _get_pageindex_client()
    if client is None:
        logger.warning("PAGEINDEX_API_KEY chưa được cấu hình. Bỏ qua upload.")
        return

    mapping = _load_doc_mapping()
    md_files = list(STANDARDIZED_DIR.rglob("*.md"))
    if not md_files:
        logger.info(f"Không tìm thấy file Markdown nào trong {STANDARDIZED_DIR}")
        return

    for path in md_files:
        source_name = path.name
        if source_name in mapping and mapping[source_name].get("doc_id"):
            logger.info(f"Tài liệu {source_name} đã tồn tại với doc_id={mapping[source_name]['doc_id']}")
            continue

        temp_pdf = None
        try:
            temp_pdf = _markdown_to_temp_pdf(path)
            response = client.submit_document(file_path=str(temp_pdf))
            doc_id = response.get("doc_id")
            if doc_id:
                doc_type = "legal" if "legal" in str(path.parent) else "news"
                mapping[source_name] = {
                    "doc_id": doc_id,
                    "source": source_name,
                    "title": path.stem.replace("-", " ").replace("_", " ").title(),
                    "doc_type": doc_type,
                    "url": None,
                }
                logger.info(f"Đã upload {source_name} -> doc_id={doc_id}")
        except Exception as exc:
            logger.error(f"Lỗi khi upload {source_name}: {exc}")
        finally:
            if temp_pdf and temp_pdf.exists():
                try:
                    temp_pdf.unlink()
                except OSError:
                    pass

    _save_doc_mapping(mapping)


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """Trả về pageindex SearchResult, an toàn khi có lỗi hoặc chưa cấu hình API."""
    if not query.strip() or top_k <= 0:
        return []

    client = _get_pageindex_client()
    if client is None:
        return []

    mapping = _load_doc_mapping()
    if not mapping:
        return []

    results: list[dict] = []
    seen_ids: set[str] = set()

    for source_name, doc_meta in mapping.items():
        doc_id = doc_meta.get("doc_id")
        if not doc_id:
            continue

        try:
            submit_resp = client.submit_query(doc_id=doc_id, query=query)
            retrieval_id = submit_resp.get("retrieval_id")
            if not retrieval_id:
                continue

            # Poll kết quả với timeout ngắn (tối đa ~3s)
            retrieval_data = None
            for _ in range(3):
                time.sleep(1)
                try:
                    ret_resp = client.get_retrieval(retrieval_id)
                    status = ret_resp.get("status")
                    if status in {"completed", "success"} or "results" in ret_resp or "nodes" in ret_resp:
                        retrieval_data = ret_resp
                        break
                    if status in {"failed", "error"}:
                        break
                except Exception:
                    break

            if not retrieval_data:
                continue

            raw_nodes = (
                retrieval_data.get("results")
                or retrieval_data.get("nodes")
                or retrieval_data.get("data")
                or []
            )

            for index, node in enumerate(raw_nodes):
                content = ""
                if isinstance(node, dict):
                    content = node.get("content") or node.get("text") or node.get("chunk") or ""
                    node_score = node.get("score")
                else:
                    content = str(node)
                    node_score = None

                content = content.strip()
                if not content:
                    continue

                item_id = f"pageindex-{doc_id}-{index}"
                if item_id in seen_ids:
                    continue
                seen_ids.add(item_id)

                score = float(node_score) if isinstance(node_score, (int, float)) else max(0.1, 1.0 - index * 0.1)

                results.append({
                    "id": item_id,
                    "content": content,
                    "score": score,
                    "metadata": {
                        "source": doc_meta.get("source", source_name),
                        "title": doc_meta.get("title", source_name),
                        "doc_type": doc_meta.get("doc_type", "legal"),
                        "url": doc_meta.get("url"),
                        "chunk_index": index,
                    },
                    "retrieval_method": "pageindex",
                })

        except Exception as exc:
            logger.warning(f"Lỗi khi truy vấn PageIndex doc_id={doc_id}: {exc}")
            continue

    # Sắp xếp giảm dần theo điểm và giới hạn top_k
    results.sort(key=lambda item: item["score"], reverse=True)
    final_results = results[:top_k]

    try:
        validate_search_results(final_results, top_k=top_k, expected_method="pageindex")
    except Exception:
        pass

    return final_results


if __name__ == "__main__":
    upload_documents()
