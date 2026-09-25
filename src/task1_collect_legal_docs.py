"""
Task 1 — Thu thập tài liệu chính sách/quy định.

Hướng dẫn:
    1. Chọn chủ đề của nhóm.
    2. Tìm tối thiểu 3 tài liệu PDF/DOCX từ nguồn công khai.
    3. Lưu file gốc vào data/landing/legal/.
    4. Đặt tên không dấu và thể hiện đúng nội dung.

Ví dụ tài liệu: học phí, học bổng, ký túc xá, quy trình đăng ký.
Nếu website chặn crawler, hãy chọn nguồn công khai khác; không vượt WAF.
"""

from pathlib import Path


import requests


DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"

LEGAL_SOURCES = {
    "ielts-writing-band-descriptors.pdf": "https://ielts.org/cdn/ielts-guides/ielts-writing-band-descriptors.pdf",
    "ielts-writing-key-assessment-criteria.pdf": "https://ielts.org/cdn/ielts-guides/ielts-writing-key-assessment-criteria.pdf",
    "ielts-writing-task1-descriptors.pdf": "https://www.ielts.org/-/media/pdfs/writing-band-descriptors-task-1.ashx",
    "ielts-writing-task2-descriptors.pdf": "https://www.ielts.org/-/media/pdfs/writing-band-descriptors-task-2.ashx",
}


def setup_directory() -> None:
    """Tạo thư mục lưu tài liệu gốc."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Ready: {DATA_DIR}")


def download_documents() -> None:
    """Tải các tài liệu chính thức từ IELTS.org về data/landing/legal/."""
    setup_directory()
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

    for filename, url in LEGAL_SOURCES.items():
        target_path = DATA_DIR / filename
        if target_path.exists() and target_path.stat().st_size > 1024:
            print(f"Already exists: {filename} ({target_path.stat().st_size} bytes)")
            continue

        print(f"Downloading {filename} from {url}...")
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        target_path.write_bytes(response.content)
        print(f"Saved: {target_path} ({len(response.content)} bytes)")


if __name__ == "__main__":
    download_documents()
