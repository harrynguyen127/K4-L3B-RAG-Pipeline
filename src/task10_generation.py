"""
Task 10 — Generation có citation.

Hướng dẫn:
    1. Retrieve top-k chunks.
    2. Reorder để giảm lost-in-the-middle.
    3. Format context kèm title và source.
    4. Gọi provider được chọn trong .env.
    5. Trả answer, sources và retrieval_source.

Nếu context không đủ hoặc provider lỗi, trả safe refusal; không bịa thông tin.
"""

import os

from dotenv import load_dotenv

from .task9_retrieval_pipeline import retrieve


load_dotenv()

TOP_K = 5
TOP_P = 0.9
TEMPERATURE = 0.3

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
LLM_MODEL = os.getenv("LLM_MODEL", "")

SYSTEM_PROMPT = """Trả lời chỉ từ context được cung cấp.
Mỗi khẳng định phải có citation. Nếu thiếu evidence, hãy từ chối xác minh."""


def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """Đưa chunks quan trọng về đầu và cuối context (chống lost-in-the-middle), không mutate input."""
    if len(chunks) <= 2:
        return [dict(c) for c in chunks]
    # Lấy xen kẽ: phần đầu lấy chỉ số chẵn, phần sau lấy chỉ số lẻ đảo ngược
    front = [dict(c) for c in chunks[::2]]
    back = [dict(c) for c in chunks[1::2]]
    return front + back[::-1]


def format_context(chunks: list[dict]) -> str:
    """Tạo context có title và source label để LLM trích dẫn được."""
    parts = []
    for index, chunk in enumerate(chunks, 1):
        meta = chunk.get("metadata", {})
        title = meta.get("title", "Unknown")
        source = meta.get("source", "Unknown")
        parts.append(
            f"[Document {index} | Title: {title} | Source: {source}]\n{chunk['content']}"
        )
    return "\n\n---\n\n".join(parts)


def call_llm(system_prompt: str, user_message: str) -> str:
    """Gọi LLM (OpenAI, Gemini hoặc Anthropic) theo cấu hình trong .env."""
    load_dotenv(override=True)
    provider = os.getenv("LLM_PROVIDER", LLM_PROVIDER).lower()
    model_name = os.getenv("LLM_MODEL", "").strip()

    # Tự động chuyển sang gemini nếu có GEMINI_API_KEY mà không có OPENAI_API_KEY
    if (provider == "openai" or not provider) and not os.getenv("OPENAI_API_KEY") and os.getenv("GEMINI_API_KEY"):
        provider = "gemini"

    try:
        if provider == "gemini":
            api_key = os.getenv("GEMINI_API_KEY", "")
            if not api_key:
                return "Tôi không thể xác minh thông tin này do chưa cấu hình GEMINI_API_KEY."
            from google import genai
            client = genai.Client(api_key=api_key)
            prompt = f"{system_prompt}\n\n{user_message}"
            response = client.models.generate_content(
                model=model_name or "gemini-3.8-flash",
                contents=prompt,
            )
            return response.text or ""

        if provider == "anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY", "")
            if not api_key:
                return "Tôi không thể xác minh thông tin này do chưa cấu hình ANTHROPIC_API_KEY."
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            response = client.messages.create(
                model=model_name or "claude-3-5-sonnet-20241022",
                max_tokens=1024,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
                temperature=TEMPERATURE,
            )
            return response.content[0].text

        # Mặc định: openai
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            return "Tôi không thể xác minh thông tin này do chưa cấu hình OPENAI_API_KEY."
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model_name or "gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=TEMPERATURE,
        )
        return response.choices[0].message.content or ""

    except Exception as exc:
        return f"Tôi không thể xác minh thông tin này từ tài liệu hiện có (Lỗi provider: {exc})."


def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    """Trả về GenerationResult kèm trích dẫn nguồn hoặc từ chối an toàn."""
    chunks = retrieve(query, top_k=top_k)
    if not chunks:
        return {
            "answer": "Tôi không tìm thấy thông tin phù hợp trong bộ tài liệu IELTS Writing để trả lời câu hỏi này.",
            "sources": [],
            "retrieval_source": "none",
        }

    reordered = reorder_for_llm(chunks)
    context = format_context(reordered)
    user_message = f"Context:\n{context}\n\nQuestion: {query}"

    answer = call_llm(SYSTEM_PROMPT, user_message)

    retrieval_source = chunks[0]["retrieval_method"]
    if retrieval_source not in {"hybrid", "pageindex"}:
        retrieval_source = "hybrid"

    return {
        "answer": answer,
        "sources": chunks,
        "retrieval_source": retrieval_source,
    }


if __name__ == "__main__":
    print(generate_with_citation("test query"))
