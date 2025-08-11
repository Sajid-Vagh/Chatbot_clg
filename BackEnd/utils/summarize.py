# utils/summarizer.py
from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

def _get_text_from_response(resp) -> str:
    """
    Make response.text extraction robust (different client shapes).
    """
    if resp is None:
        return ""
    if hasattr(resp, "text") and resp.text:
        return resp.text
    try:
        # Common structure: resp.candidates[0].content -> list of parts with .text
        cands = getattr(resp, "candidates", None)
        if cands:
            first = cands[0]
            content = getattr(first, "content", None)
            if isinstance(content, (list, tuple)):
                parts = []
                for part in content:
                    t = getattr(part, "text", None)
                    if t:
                        parts.append(t)
                    elif isinstance(part, dict) and part.get("text"):
                        parts.append(part.get("text"))
                return "".join(parts)
            # fallback
            return str(content)
    except Exception:
        pass
    try:
        return str(resp)
    except Exception:
        return ""

def summarize_document_chunked(model, text: str, filename: str, chunk_size: int = 2000, chunk_overlap: int = 300, max_batch_size: int = 6) -> str:
    """
    Summarize long text safely by chunking. Returns a concise summary string.
    """
    if model is None:
        return "[Model not configured: cannot summarize]"

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = [Document(page_content=text)]
    chunks = splitter.split_documents(docs)
    if not chunks:
        return ""

    batch_summaries = []
    for i in range(0, len(chunks), max_batch_size):
        batch = chunks[i:i+max_batch_size]
        batch_text = "\n\n".join([c.page_content for c in batch])
        prompt = f"""
Summarize this section of document '{filename}' into:
- TL;DR: 1 sentence
- Key points: 3-5 concise bullets
Be factual and concise.

Text:
{batch_text}
"""
        resp = model.generate_content(prompt)
        batch_summaries.append(_get_text_from_response(resp).strip())

    combined = "\n\n".join(batch_summaries)
    final_prompt = f"""
You are an expert summarizer. Combine the following section summaries for document '{filename}' into:
## TL;DR (1 sentence)
## Key Points (3–6 bullets)

Summaries:
{combined}
"""
    final_resp = model.generate_content(final_prompt)
    return _get_text_from_response(final_resp).strip()


def summarize_all_documents(model, docs_with_metadata: List[Document]) -> Dict[str, str]:
    summaries = {}
    for doc in docs_with_metadata:
        src = doc.metadata.get("source", "unknown")
        try:
            summaries[src] = summarize_document_chunked(model, doc.page_content, src)
        except Exception as e:
            print(f"[summarizer] Summarization failed for {src}: {e}")
            summaries[src] = "[Summary unavailable due to error]"
    return summaries
