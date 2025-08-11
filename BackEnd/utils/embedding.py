# utils/embedding.py
import os
import math
from typing import List, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

# Lazy import of heavy libs so module import doesn't crash if env not yet available
try:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    from langchain_community.vectorstores import FAISS
    _HAS_LC_GOOGLE = True
except Exception:
    _HAS_LC_GOOGLE = False

# helper to create embedding model only when needed
def _get_embedding_model():
    if not _HAS_LC_GOOGLE:
        return None
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return None
    # pass api key explicitly so it doesn't try ADC
    return GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)


def create_vector_store(docs_with_metadata: List[Document]):
    """
    Splits documents into chunks and builds a FAISS vector store.
    Returns None if embeddings are not configured (so you can still upload docs).
    """
    if not docs_with_metadata:
        return None

    embedding_model = _get_embedding_model()
    if embedding_model is None:
        # Embeddings not available (no GOOGLE_API_KEY or missing package)
        print("[embedding] Embedding model not configured; skipping vector store creation.")
        return None

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs_with_metadata)
    if not chunks:
        return None

    vector_store = FAISS.from_documents(chunks, embedding_model)
    return vector_store


def _matches_source(doc: Document, source_filename: str) -> bool:
    md = doc.metadata or {}
    src = (md.get("source") or "").lower()
    basename = (md.get("basename") or "").lower()
    doc_id = str(md.get("doc_id", "")).lower()
    q = source_filename.lower()
    return q == src or q == basename or q == doc_id


def retrieve_relevant_chunks(vector_store, query: str, k: int = 5, sources: Optional[List[str]] = None) -> List[Document]:
    """
    Returns a list of langchain Document objects.
    - If vector_store is None -> returns [].
    - If sources is None -> top-k across all docs.
    - If sources provided -> tries to return results from each source (ensures results come from requested docs).
    """
    if not vector_store:
        return []

    if not sources:
        return vector_store.similarity_search(query, k=k)

    # We'll ask for many candidates and then filter by source
    candidate_k = max(k * 4, 50)
    candidates = vector_store.similarity_search(query, k=candidate_k)

    per_source_k = max(1, math.ceil(k / len(sources)))
    collected = []
    seen = set()

    for s in sources:
        matched = []
        for doc in candidates:
            if _matches_source(doc, s):
                key = (doc.metadata.get("source", ""), doc.page_content[:120])
                if key in seen:
                    continue
                matched.append(doc)
                seen.add(key)
            if len(matched) >= per_source_k:
                break
        collected.extend(matched)

    # If not enough, fill from other candidates that match any of the requested sources
    if len(collected) < k:
        for doc in candidates:
            key = (doc.metadata.get("source", ""), doc.page_content[:120])
            if key in seen:
                continue
            if any(_matches_source(doc, s) for s in sources):
                collected.append(doc)
                seen.add(key)
            if len(collected) >= k:
                break

    return collected[:k]
