"""
RAG layer for Vishal AI Assistant (Pro v3) - original, dependency-light.

Pipeline (course concept: AI-Agentic-Engineer Ch.7, our own code):
  document text -> sentence-aware chunks (~800 chars, ~100 overlap)
  -> Gemini embeddings (same GEMINI_API_KEY the bot already uses)
  -> chunks + embeddings stored in SQLite (db.doc_chunks)
  -> a support question embeds, cosine-matches ONLY its own business's
     chunks, top-k above a similarity threshold are given to the AI as
     DATA (never as instructions - see prompts.build_system_prompt).

Fallbacks, in the same spirit as the Groq -> Gemini chat fallback:
  * If embeddings are unavailable (no Gemini key, API error, quota),
    retrieval degrades to keyword-overlap scoring so Support never
    hard-fails; chunks stored without embeddings are still searchable
    by keywords.
  * If nothing clears the threshold, retrieve() returns [] and the
    caller behaves exactly like Pro v2 (business info + FAQ, or a
    polite defer-to-owner). The AI is never given near-miss text.

No secrets here: the key is read from Config (env) at call time and is
never logged. All network calls go through an injectable session so
tests run fully offline with a fake embedder.
"""
import logging
import math
import re

import requests

from config import Config

log = logging.getLogger("rag")

EMBED_URL = ("https://generativelanguage.googleapis.com/v1beta/models/"
             "{model}:batchEmbedContents")
EMBED_ONE_URL = ("https://generativelanguage.googleapis.com/v1beta/models/"
                 "{model}:embedContent")

# Words too common to carry meaning in keyword fallback (EN + Hinglish).
_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "what", "when", "how",
    "much", "many", "for", "of", "to", "in", "on", "at", "and", "or",
    "hai", "hain", "ka", "ki", "ke", "ko", "me", "mein", "kya", "kitna",
    "kitni", "kitne", "kab", "kahan", "kaise", "aur", "ya", "ye", "wo",
    "bhi", "liye", "kis", "hota", "hoti", "hote",
    "please", "batao", "bataiye", "do", "does", "i", "we", "you", "it",
}

_SENTENCE_END = re.compile(r"(?<=[.!?।])\s+")


# ----------------------------------------------------------------------
# Chunking
# ----------------------------------------------------------------------
def split_sentences(text):
    """Split on sentence-ending punctuation (EN + Devanagari danda)."""
    parts = [s.strip() for s in _SENTENCE_END.split(text or "") if s.strip()]
    return parts


def chunk_text(text, max_chars=None, overlap=None):
    """Sentence-aware chunks of ~max_chars with ~overlap chars shared
    between consecutive chunks. A single sentence longer than max_chars
    is hard-split. Returns [] for blank text."""
    max_chars = max_chars or Config.RAG_CHUNK_CHARS
    overlap = Config.RAG_CHUNK_OVERLAP if overlap is None else overlap
    overlap = max(0, min(overlap, max_chars // 2))
    text = re.sub(r"\s+", " ", (text or "")).strip()
    if not text:
        return []
    # Normalise to a sentence list (hard-splitting oversized sentences).
    sentences = []
    for s in split_sentences(text):
        while len(s) > max_chars:
            sentences.append(s[:max_chars])
            s = s[max_chars - overlap:] if overlap else s[max_chars:]
        if s:
            sentences.append(s)
    chunks = []
    current = ""
    for s in sentences:
        if current and len(current) + 1 + len(s) > max_chars:
            chunks.append(current)
            # Carry the tail of the finished chunk into the next one.
            current = (current[-overlap:] + " " + s).strip() if overlap else s
            # Overlap seed must not push us far over the limit.
            while len(current) > max_chars and " " in current:
                current = current.split(" ", 1)[1]
        else:
            current = (current + " " + s).strip() if current else s
    if current:
        chunks.append(current)
    return chunks


# ----------------------------------------------------------------------
# Similarity + keyword fallback
# ----------------------------------------------------------------------
def cosine(a, b):
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _tokens(text):
    words = re.findall(r"[a-z0-9\u0900-\u097F]+", (text or "").lower())
    # Keep pure numbers of any length ("8" in "Class 8", "1500" fees) -
    # in this domain they are the most discriminative tokens of all.
    return {w for w in words
            if w not in _STOPWORDS and (len(w) > 1 or w.isdigit())}


def keyword_score(question, chunk):
    """Overlap score 0..1: share of the question's meaningful tokens
    that appear in the chunk, with a small bonus for coverage."""
    q = _tokens(question)
    if not q:
        return 0.0
    c = _tokens(chunk)
    hit = len(q & c)
    if hit == 0:
        return 0.0
    return hit / len(q)


# ----------------------------------------------------------------------
# Gemini embeddings (batched; injectable session for tests)
# ----------------------------------------------------------------------
class GeminiEmbedder:
    """Real embedder backed by the Gemini embedding API.

    embed(texts) -> list of vectors (same order), or raises on failure;
    callers treat any failure as 'embeddings unavailable'."""

    def __init__(self, session=None, api_key=None, model=None):
        self.session = session or requests.Session()
        self.api_key = api_key if api_key is not None else Config.GEMINI_API_KEY
        self.model = model or Config.GEMINI_EMBED_MODEL

    @property
    def available(self):
        return bool(self.api_key)

    def embed(self, texts, task_type="RETRIEVAL_DOCUMENT"):
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY not set")
        texts = list(texts)
        out = []
        batch_size = 100  # Gemini batchEmbedContents request cap
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            body = {"requests": [
                {"model": f"models/{self.model}",
                 "content": {"parts": [{"text": t}]},
                 "taskType": task_type} for t in batch]}
            resp = self.session.post(
                EMBED_URL.format(model=self.model),
                headers={"x-goog-api-key": self.api_key,
                         "Content-Type": "application/json"},
                json=body, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            out.extend(e["values"] for e in data["embeddings"])
        return out

    def embed_one(self, text, task_type="RETRIEVAL_QUERY"):
        return self.embed([text], task_type=task_type)[0]


def get_embedder():
    return GeminiEmbedder()


# ----------------------------------------------------------------------
# Document text extraction (PDF optional - bot must run without pypdf)
# ----------------------------------------------------------------------
class DocumentError(Exception):
    """User-facing document problem; message is already in Hinglish."""


def extract_text(filename, raw_bytes):
    """Return (text, page_count_or_None) or raise DocumentError with a
    clear Hinglish message. PDF support needs the optional pypdf
    package; TXT works with zero extra dependencies."""
    name = (filename or "").lower()
    if len(raw_bytes) > Config.DOC_MAX_BYTES:
        raise DocumentError(
            "❌ File bahut badi hai (max 5 MB). Chhoti file bhejein.")
    if name.endswith(".txt") or name.endswith(".md"):
        try:
            return raw_bytes.decode("utf-8"), None
        except UnicodeDecodeError:
            raise DocumentError(
                "❌ Is text file ki bhasha/encoding samajh nahi aayi. "
                "Sadharan (plain) text bhej dein ya jaankari copy-paste kar dein.")
    if name.endswith(".pdf"):
        try:
            from pypdf import PdfReader
        except ImportError:
            raise DocumentError(
                "❌ PDF padhne wala hissa (pypdf) is server par abhi uplabdh "
                "nahi hai. Kripya jaankari text me copy-paste karke bhej dein "
                "— wo turant jud jayegi.")
        import io
        try:
            reader = PdfReader(io.BytesIO(raw_bytes))
            pages = len(reader.pages)
            if pages > Config.DOC_MAX_PAGES:
                raise DocumentError(
                    f"❌ PDF bahut lambi hai ({pages} page, max "
                    f"{Config.DOC_MAX_PAGES}). Chhota hissa bhej dein.")
            text = "\n".join((p.extract_text() or "") for p in reader.pages)
        except DocumentError:
            raise
        except Exception:
            raise DocumentError(
                "❌ Ye PDF khul/padh nahi payi — file kharab ho sakti hai. "
                "Dobara bhejein ya jaankari text me bhej dein.")
        if len(text.strip()) < 30:
            raise DocumentError(
                "❌ Is PDF me padhne layak text nahi mila — lagta hai ye "
                "scanned/photo wali PDF hai. Kripya jaankari type karke ya "
                "copy-paste karke bhej dein, wo turant jud jayegi.")
        return text, pages
    raise DocumentError(
        "❌ Sirf PDF ya TXT file chalegi. Ya to sahi file bhejein, ya "
        "jaankari seedha text me copy-paste kar dein.")


# ----------------------------------------------------------------------
# Ingestion + retrieval (DB does storage; this module orchestrates)
# ----------------------------------------------------------------------
def ingest_document(db, business_code, doc_name, text, embedder=None):
    """Chunk + (best-effort) embed + store one document.
    Returns (doc_id, chunk_count). Embeddings failing never blocks
    ingestion - chunks are stored without vectors and stay searchable
    through the keyword fallback."""
    chunks = chunk_text(text)
    if not chunks:
        raise DocumentError("❌ Isme padhne layak jaankari nahi mili.")
    if len(chunks) > Config.DOC_MAX_CHUNKS:
        raise DocumentError(
            f"❌ Document bahut bada hai ({len(chunks)} hisse, max "
            f"{Config.DOC_MAX_CHUNKS}). Use chhote hisso me baant kar bhejein.")
    embedder = embedder or get_embedder()
    vectors = [None] * len(chunks)
    if embedder is not None and getattr(embedder, "available", True):
        try:
            vectors = embedder.embed(chunks)
            if len(vectors) != len(chunks):
                raise RuntimeError("embedding count mismatch")
        except Exception as exc:  # never leak key/body - type only
            log.warning("Embedding failed (%s) - storing chunks for "
                        "keyword fallback", type(exc).__name__)
            vectors = [None] * len(chunks)
    doc_id = db.add_document(business_code, doc_name, len(chunks))
    db.store_chunks(doc_id, business_code, chunks, vectors)
    return doc_id, len(chunks)


def retrieve(db, business_code, question, embedder=None, top_k=None,
             threshold=None):
    """Return up to top_k chunk dicts {doc_name, text, score, method}
    for THIS business only, best first, all with score >= threshold.

    Vector path first (needs question embedding + stored vectors);
    keyword-overlap path covers chunks without vectors and is also the
    fallback when the embedding API is down. Returns [] when nothing
    clears the bar - the caller then uses the exact Pro v2 behaviour."""
    top_k = top_k or Config.RAG_TOP_K
    threshold = Config.RAG_THRESHOLD if threshold is None else threshold
    rows = db.list_chunks(business_code)
    if not rows:
        return []
    scored = []
    qvec = None
    embedder = embedder or get_embedder()
    if embedder is not None and getattr(embedder, "available", True):
        try:
            qvec = embedder.embed_one(question)
        except Exception as exc:
            log.warning("Query embedding failed (%s) - keyword fallback",
                        type(exc).__name__)
            qvec = None
    for row in rows:
        vec = row.get("embedding")
        if qvec is not None and vec:
            score = cosine(qvec, vec)
            if score >= threshold:
                scored.append({**row, "score": score, "method": "vector"})
        else:
            score = keyword_score(question, row["text"])
            # Keyword scores live on a different scale; require strong
            # overlap so unrelated questions still retrieve nothing.
            if score >= max(threshold, 0.6):
                scored.append({**row, "score": score, "method": "keyword"})
    scored.sort(key=lambda r: r["score"], reverse=True)
    return scored[:top_k]


def format_context(chunks):
    """Prompt-ready DATA block for the retrieved chunks."""
    parts = []
    for c in chunks:
        parts.append(f"[Document: {c['doc_name']}]\n{c['text']}")
    return "\n\n".join(parts)


def source_names(chunks):
    seen = []
    for c in chunks:
        if c["doc_name"] not in seen:
            seen.append(c["doc_name"])
    return seen
