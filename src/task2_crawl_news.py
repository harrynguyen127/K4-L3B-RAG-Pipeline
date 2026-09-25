"""
Task 2 — Crawl bài viết/thông báo.

Hướng dẫn:
    1. Điền tối thiểu 5 URL công khai vào ARTICLE_URLS.
    2. Crawl từng URL bằng Crawl4AI.
    3. Lưu mỗi bài thành một JSON trong data/landing/news/.
    4. Giữ đủ url, title, date_crawled và content_markdown.

Cài browser trước khi chạy:
    python -m playwright install chromium
    
-> Dùng Firecrawl or bất cứ công cụ nào bạn quen    
"""

import asyncio
import json
from pathlib import Path


from datetime import datetime
import json
from pathlib import Path
import re

import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as md


DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"

ARTICLE_URLS = [
    "https://ielts.org/news-and-insights/ielts-writing-band-descriptors-and-key-assessment-criteria",
    "https://ielts.org/take-a-test/preparation-resources/writing-test-resources",
    "https://ielts.org/take-a-test/test-types/ielts-academic-test/ielts-academic-format-writing",
    "https://takeielts.britishcouncil.org/what-is-ielts/how-it-works/test-format/writing",
    "https://takeielts.britishcouncil.org/prepare/ielts-academic/writing",
]


async def crawl_article(url: str) -> dict:
    """Crawl bài viết từ URL và trích xuất markdown kèm metadata."""
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    
    # Thử crawl qua requests + BeautifulSoup + markdownify
    response = requests.get(url, headers=headers, timeout=20)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    # Trích xuất title
    title = ""
    if soup.find("h1"):
        title = soup.find("h1").get_text(strip=True)
    elif soup.find("title"):
        title = soup.find("title").get_text(strip=True)
    if not title:
        title = "IELTS Writing Resource"

    # Làm sạch các thẻ không cần thiết
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "aside"]):
        tag.decompose()

    # Tìm phần nội dung chính
    main_content = soup.find("main") or soup.find("article") or soup.find("div", class_=re.compile(r"content|body|main", re.I)) or soup.body

    html_str = str(main_content) if main_content else response.text
    content_markdown = md(html_str, heading_style="ATX", strip=["img"])
    
    # Loại bỏ các dòng trống liên tiếp
    content_markdown = re.sub(r"\n{3,}", "\n\n", content_markdown).strip()
    if not content_markdown:
        content_markdown = soup.get_text(separator="\n\n", strip=True)

    return {
        "url": url,
        "title": title,
        "date_crawled": datetime.now().isoformat(),
        "content_markdown": content_markdown,
    }


async def crawl_all() -> None:
    """Crawl và lưu từng bài thành một file JSON."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for index, url in enumerate(ARTICLE_URLS, 1):
        try:
            print(f"Crawling ({index}/{len(ARTICLE_URLS)}): {url}...")
            article = await crawl_article(url)
            output = DATA_DIR / f"article_{index:02d}.json"
            output.write_text(
                json.dumps(article, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"Saved: {output} (title: {article['title']})")
        except Exception as error:
            print(f"Failed: {url} — {error}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(crawl_all())
