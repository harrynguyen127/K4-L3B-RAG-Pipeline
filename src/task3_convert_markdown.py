"""
Task 3 — Chuẩn hóa dữ liệu sang Markdown.

Hướng dẫn:
    1. Dùng MarkItDown để convert PDF/DOCX.
    2. Đọc JSON và giữ metadata ở đầu file Markdown.
    3. Giữ cấu trúc thư mục legal/ và news/.
    4. Không tạo file rỗng hoặc file trùng khi chạy lại.

Cài đặt:
    Dependency MarkItDown đã được khai báo trong pyproject.toml.
    
-> Hoặc dùng công cụ nào bạn quen khác Markitdown
"""

import json
from pathlib import Path

from markitdown import MarkItDown


LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"


def convert_legal_docs() -> None:
    """Convert các tài liệu PDF/DOCX từ landing/legal sang standardized/legal."""
    legal_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)
    converter = MarkItDown()

    for path in sorted(legal_dir.iterdir()):
        if path.suffix.lower() in {".pdf", ".doc", ".docx"}:
            print(f"Converting legal document: {path.name}...")
            result = converter.convert(str(path))
            title = path.stem.replace("-", " ").replace("_", " ").title()
            header = (
                f"# {title}\n\n"
                f"**Source:** {path.name}\n\n"
                f"**Doc Type:** legal\n\n---\n\n"
            )
            out_file = output_dir / f"{path.stem}.md"
            out_file.write_text(header + result.text_content, encoding="utf-8")
            print(f"Saved: {out_file} ({len(result.text_content)} chars)")


def convert_news_articles() -> None:
    """Convert các bài viết JSON từ landing/news sang standardized/news."""
    news_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"
    output_dir.mkdir(parents=True, exist_ok=True)

    for path in sorted(news_dir.glob("*.json")):
        print(f"Converting news article: {path.name}...")
        data = json.loads(path.read_text(encoding="utf-8"))
        header = (
            f"# {data['title']}\n\n"
            f"**Source:** {data['url']}\n\n"
            f"**Doc Type:** news\n\n"
            f"**Crawled:** {data['date_crawled']}\n\n---\n\n"
        )
        out_file = output_dir / f"{path.stem}.md"
        out_file.write_text(
            header + data["content_markdown"], encoding="utf-8"
        )
        print(f"Saved: {out_file} ({len(data['content_markdown'])} chars)")


def convert_all() -> None:
    """Convert toàn bộ dữ liệu landing."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    convert_legal_docs()
    convert_news_articles()
    print(f"Saved Markdown to: {OUTPUT_DIR}")


if __name__ == "__main__":
    convert_all()
