# main.py
import os
import re
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Optional
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai

from utils.file_readers import extract_documents
from utils.embedding import create_vector_store, retrieve_relevant_chunks
from utils.summarize import summarize_all_documents

# Load .env early
load_dotenv(Path(__file__).parent / ".env")
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")

app = FastAPI(title="Doc-aware AI Chatbot")

origins = ["http://localhost:3000"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global in-memory store
vector_store = None
docs_with_metadata: List = []
doc_filenames: List[str] = []
doc_summaries: Dict[str, str] = {}
chat_history: List[Dict] = []

# Configure Gemini model if API key present
model = None
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel(model_name="gemini-2.5-flash")
        print("[main] Gemini 2.5 Flash configured.")
    except Exception as e:
        print(f"[main] Error configuring Gemini: {e}")
else:
    print("[main] GOOGLE_API_KEY not found in .env. Set it to enable the model.")

class ChatQuery(BaseModel):
    query: str

# ---------- Utility: parse mentioned documents ----------
def parse_mentioned_documents(query: str, filenames: List[str]) -> Optional[List[str]]:
    """
    Returns:
      - None -> user explicitly asked for 'all' documents
      - [] -> no explicit doc mention found
      - list of filenames -> user mentioned those files (by name, basename, or numeric id)
    """
    if not filenames:
        return []

    q = query.lower()

    # direct 'all' detection
    if re.search(r'\b(all|every|sab|sabhi)\b', q):
        return None

    # map aliases to filename
    alias_map = {}
    for idx, fname in enumerate(filenames):
        fid = str(idx + 1)
        basename = fname.rsplit(".", 1)[0].lower()
        aliases = {fname.lower(), basename, fid, f"document {fid}", f"doc {fid}", f"file {fid}"}
        for a in aliases:
            alias_map[a] = fname

    mentioned = set()
    for alias, fname in alias_map.items():
        if re.search(r'\b' + re.escape(alias) + r'\b', q):
            mentioned.add(fname)

    # if none found, try extracting numbers
    if not mentioned:
        nums = re.findall(r'\b\d+\b', q)
        for n in nums:
            if n in alias_map:
                mentioned.add(alias_map[n])

    return list(mentioned)


# ---------- Upload endpoint ----------
@app.post("/upload/", summary="Upload files and build KB")
async def upload_files(files: List[UploadFile] = File(...)):
    global docs_with_metadata, vector_store, doc_filenames, doc_summaries, chat_history
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")
    try:
        chat_history = []
        docs_with_metadata = await extract_documents(files)
        doc_filenames = [doc.metadata.get("source") for doc in docs_with_metadata]

        # Try to create vector store (may return None if embeddings not configured)
        vector_store = create_vector_store(docs_with_metadata)

        # Precompute per-doc summaries if model available
        if model and docs_with_metadata:
            try:
                doc_summaries = summarize_all_documents(model, docs_with_metadata)
            except Exception as e:
                print(f"[main] Summarization warning: {e}")
                doc_summaries = {}

        return JSONResponse(status_code=200, content={
            "status": "success",
            "message": f"{len(doc_filenames)} files processed.",
            "filenames": doc_filenames,
            "vector_store_created": bool(vector_store)
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload error: {e}")


# ---------- Helper to get text from model response ----------
def _get_text_from_response(resp) -> str:
    if resp is None:
        return ""
    if hasattr(resp, "text") and resp.text:
        return resp.text
    # try common shapes
    try:
        cands = getattr(resp, "candidates", None)
        if cands:
            first = cands[0]
            content = getattr(first, "content", None)
            # content may be list of parts with .text
            if isinstance(content, (list, tuple)):
                parts = []
                for part in content:
                    t = getattr(part, "text", None)
                    if t:
                        parts.append(t)
                    elif isinstance(part, dict) and part.get("text"):
                        parts.append(part.get("text"))
                return "".join(parts)
            return str(content)
    except Exception:
        pass
    try:
        return str(resp)
    except Exception:
        return ""


# ---------- Chat endpoint ----------
@app.post("/chat/", summary="Chat with the AI (doc-aware)")
async def chat_with_bot(request: ChatQuery):
    global vector_store, docs_with_metadata, doc_filenames, doc_summaries, chat_history, model

    if model is None:
        # model required for AI responses
        raise HTTPException(status_code=500, detail="Generative model not configured. Set GOOGLE_API_KEY in .env.")

    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    # If no documents uploaded -> general chat fallback
    if not docs_with_metadata:
        # general chat prompt
        prompt = f"""
You are BotBuddy 🤖, a smart and friendly AI assistant. Always respond clearly, engagingly, and helpfully — even if no documents are uploaded.

If the user's question is general (e.g., about Python, tech, APIs), give an informative and well-structured response.  
If it's a document-related question, kindly inform them that no files are currently uploaded 📄.

🔹 Use multiple types of bullet symbols for structure (like 👉, 🔸, 📌, ✔️).  
😊 Include emojis where helpful to keep a light and friendly tone.

🧠 Always think step-by-step before answering.

---

👤 **User:** {query}  
🤖 **Assistant**:
"""


        try:
            resp = model.generate_content(prompt)
            text = _get_text_from_response(resp).strip()
            chat_history.append({"role": "user", "content": query})
            chat_history.append({"role": "assistant", "content": text})
            return JSONResponse(status_code=200, content={"bot_response": text, "source_documents_consulted": False, "retrieved_context": []})
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Model error: {e}")

    # parse mentioned documents (None -> all, [] -> none mentioned)
    mentioned = parse_mentioned_documents(query, doc_filenames)

    # If user asks for summary
    if re.search(r'\b(summary|summarize|summarise)\b', query.lower()):
        if mentioned is None:
            # all
            if not doc_summaries:
                doc_summaries = summarize_all_documents(model, docs_with_metadata)
            parts = [f"## {f}\n{doc_summaries.get(f,'[No summary]')}" for f in doc_filenames]
            bot_resp = "\n\n".join(parts)
            chat_history.append({"role": "user", "content": query})
            chat_history.append({"role": "assistant", "content": bot_resp})
            return JSONResponse(status_code=200, content={"bot_response": bot_resp, "sources": doc_filenames})
        elif mentioned:
            if not doc_summaries:
                doc_summaries = summarize_all_documents(model, docs_with_metadata)
            parts = [f"""## 📄 {f}🔹 **Summary**:  {doc_summaries.get(f, '[No summary]')}
---
""" for f in mentioned
]
            bot_resp = "\n\n".join(parts)
            chat_history.append({"role": "user", "content": query})
            chat_history.append({"role": "assistant", "content": bot_resp})
            return JSONResponse(status_code=200, content={"bot_response": bot_resp, "sources": mentioned})
        else:
            # no explicit mention -> default to all
            if not doc_summaries:
                doc_summaries = summarize_all_documents(model, docs_with_metadata)
            parts = [f"## {f}\n{doc_summaries.get(f,'[No summary]')}" for f in doc_filenames]
            bot_resp = "\n\n".join(parts)
            chat_history.append({"role": "user", "content": query})
            chat_history.append({"role": "assistant", "content": bot_resp})
            return JSONResponse(status_code=200, content={"bot_response": bot_resp, "sources": doc_filenames})

    # If user asks to compare
    if re.search(r'\b(compare|difference|differences|contrast)\b', query.lower()):
        if mentioned is None:
            to_compare = doc_filenames.copy()
        elif mentioned:
            to_compare = mentioned
        else:
            to_compare = doc_filenames.copy()

        if not doc_summaries:
            doc_summaries = summarize_all_documents(model, docs_with_metadata)

        compare_text = "\n\n".join([f"### {f}\n{doc_summaries.get(f,'[No summary]')}" for f in to_compare])
        prompt = f"""
You are BotBuddy 🤖. Please compare these documents in a friendly, engaging way:
- Provide a 1-2 line summary for each document 📄.
- List key similarities using these bullet points: 🔹 🔸 ✨ (3 bullets).
- List key differences using these bullet points: ❌ ⚠️ 🚫 (3 bullets).

Use clear Markdown formatting and sprinkle in friendly emojis 😊.

{compare_text}

User question: {query}
"""


        try:
            resp = model.generate_content(prompt)
            text = _get_text_from_response(resp).strip()
            chat_history.append({"role": "user", "content": query})
            chat_history.append({"role": "assistant", "content": text})
            return JSONResponse(status_code=200, content={"bot_response": text, "sources": to_compare})
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Model error: {e}")

    # Default: RAG search. If user mentioned docs, restrict search to those sources.
    sources_for_search: Optional[List[str]] = None
    if mentioned is None:
        sources_for_search = None
    elif mentioned:
        sources_for_search = mentioned
    else:
        sources_for_search = None

    retrieved_docs = []
    if vector_store:
        retrieved_docs = retrieve_relevant_chunks(vector_store, query, k=6, sources=sources_for_search)

    if retrieved_docs:
        context_parts = []
        sources_used = []
        for d in retrieved_docs:
            src = d.metadata.get("source", "unknown")
            snippet = d.page_content[:1200]
            context_parts.append(f"[source: {src}]\n{snippet}")
            sources_used.append(src)
        context_str = "\n\n".join(context_parts)

        # Add single source mention at the top (optional)
        unique_sources = set(sources_used)
        source_line = ""
        if unique_sources:
            source_line = f"The following information is from: {', '.join(unique_sources)}\n\n"
        context_str = source_line + context_str



        prompt = f"""
You are BotBuddy 🤖 — a smart, helpful, and friendly AI assistant. Use the following **document context** to answer the user's question.

📄 If information is not available in the documents, you may use your general knowledge.  
📝 When using document data, cite it only **once** using the format _(source: filename)_.

✅ Guidelines:
- Keep your tone conversational and friendly 😊  
- Use **Markdown formatting**  
- Organize information using varied bullet symbols like:
  - 🔹 for points  
  - 📌 for key info  
  - ✔️ for conclusions  
  - 👉 for steps  
- Use emojis to make your answer feel engaging and approachable!

---

📚 **Context**:  
{context_str}

👤 **User Question**:  
{query}

🤖 **Assistant's Answer**:
"""



        try:
            resp = model.generate_content(prompt)
            text = _get_text_from_response(resp).strip()
            chat_history.append({"role": "user", "content": query})
            chat_history.append({"role": "assistant", "content": text})
            retrieved_context_summary = [{"source": d.metadata.get("source","unknown"), "snippet": d.page_content[:350]} for d in retrieved_docs]
            return JSONResponse(status_code=200, content={
                "bot_response": text,
                "source_documents_consulted": True,
                "retrieved_context": retrieved_context_summary
            })
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Model error: {e}")

    # Fallback: use doc_summaries if available and restrict to mentioned
    if doc_summaries:
        if sources_for_search:
            summaries_to_use = [doc_summaries.get(f, "[No summary]") for f in sources_for_search]
            combined = "\n\n".join(summaries_to_use)
        else:
            combined = "\n\n".join(doc_summaries.values())

        prompt = f"""
You are BotBuddy, a helpful AI assistant. The user asked:

{query}

Here are summaries of documents you can use:

{combined}

- Please organize your answer clearly with multiple bullet symbols like • ◦ –
- Use friendly emojis 😊 to keep the tone warm and engaging
- If you can't answer exactly, please try your best to help

Answer in a friendly tone.
"""

        try:
            resp = model.generate_content(prompt)
            text = _get_text_from_response(resp).strip()
            chat_history.append({"role": "user", "content": query})
            chat_history.append({"role": "assistant", "content": text})
            return JSONResponse(status_code=200, content={
                "bot_response": text,
                "source_documents_consulted": False,
                "retrieved_context": []
            })
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Model error: {e}")

    # Final fallback: general chat
    prompt = f"""
You are BotBuddy 🤖, a smart and friendly AI assistant. Respond clearly and helpfully to the user's general question.
Even if no documents are uploaded, give your best answer. Use a conversational, friendly tone, and include emojis 😊.

• Please use multiple bullet symbols such as •, ◦, and – for lists.
◦ Include emojis 😊 to make responses friendly and engaging.
– Organize your answers with Markdown formatting and clear line breaks.

User: {query}
Assistant:  
"""


    try:
        resp = model.generate_content(prompt)
        text = _get_text_from_response(resp).strip()
        chat_history.append({"role": "user", "content": query})
        chat_history.append({"role": "assistant", "content": text})
        return JSONResponse(status_code=200, content={"bot_response": text, "source_documents_consulted": False, "retrieved_context": []})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model error: {e}")


# ---------- Reset endpoint ----------
@app.post("/reset/", summary="Reset chatbot memory and uploaded documents")
async def reset_bot():
    global chat_history, docs_with_metadata, vector_store, doc_filenames, doc_summaries
    chat_history = []
    docs_with_metadata = []
    doc_filenames = []
    doc_summaries = {}
    vector_store = None
    return JSONResponse(status_code=200, content={"status": "reset complete"})
