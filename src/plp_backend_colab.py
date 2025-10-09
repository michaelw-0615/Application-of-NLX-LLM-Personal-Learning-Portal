# plp_backend_colab.py
# A reusable PLP backend: load JSONL corpus (id,title,url,text),
# chunk → TF-IDF index → cosine retrieval → synthesize answer with citations.

from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
from pathlib import Path
import json, re, textwrap
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------- Data Types ----------
@dataclass
class Chunk:
    doc_id: str
    chunk_id: str
    title: str
    url: str
    text: str


# ---------- Globals (built after init_backend) ----------
_chunks: List[Chunk] = []
_index = None

# ===== Reasoning Eval: config & logging =====
from datetime import datetime
from pathlib import Path

LOG_DIR = Path("/content/Drive/My Drive/NLX_LLM_Project_3/eval")
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = LOG_DIR / "reasoning_log.csv"

# Write header if log file doesn't exist
if not LOG_PATH.exists():
    LOG_PATH.write_text(
        "ts,query,mode,subqueries,answer,citation_count,groundedness,coverage,consistency,chunk_ids\n",
        encoding="utf-8"
    )



# ---------- Utilities ----------
def _simple_sentence_split(text: str) -> List[str]:
    sents = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s for s in sents if s]

def _build_chunks(docs: List[Dict[str, str]], max_sent_per_chunk: int = 2) -> List[Chunk]:
    chunks: List[Chunk] = []
    for d in docs:
        doc_id = d["id"]
        title = d.get("title", "")
        url = d.get("url", "")
        text = (d.get("text") or "").strip()
        if not text:
            continue
        sents = _simple_sentence_split(text)
        for i in range(0, len(sents), max_sent_per_chunk):
            piece = " ".join(sents[i:i + max_sent_per_chunk])
            chunks.append(Chunk(
                doc_id=doc_id,
                chunk_id=f"{doc_id}::ch{i//max_sent_per_chunk}",
                title=title,
                url=url,
                text=piece
            ))
    return chunks


class _TfidfIndex:
    def __init__(self, chunks: List[Chunk]):
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        self.matrix = self.vectorizer.fit_transform([c.text for c in chunks])

    def query(self, q: str, top_k: int = 5) -> List[Tuple[Chunk, float]]:
        qv = self.vectorizer.transform([q])
        sims = cosine_similarity(qv, self.matrix)[0]
        idx = np.argsort(-sims)[:top_k]
        return [(self.chunks[i], float(sims[i])) for i in idx]


def _cot_plan(query: str) -> List[str]:
    return [
        "1) Identify the finance concept(s) asked.",
        "2) Retrieve definitions/formulas and usage conditions.",
        "3) Cross-check top chunks for consistency and specificity.",
        "4) Compose a concise, grounded answer with citations."
    ]


def _synthesize_answer(query: str, retrieved, show_steps: bool = True) -> Dict[str, Any]:
    bullets = []
    for ch, score in retrieved:
        snippet = textwrap.shorten(ch.text, width=220, placeholder="…")
        bullets.append(f"- [{ch.doc_id}] {snippet} (score={score:.3f})")

    ql = query.lower()
    lines = []
    if "wacc" in ql or "cost of capital" in ql:
        lines.append("WACC = E/V·Re + D/V·Rd·(1−Tc). Use it when project risk is similar to the firm’s core assets.")
    if "npv" in ql or "irr" in ql:
        lines.append("NPV = Σ CFt/(1+r)^t − initial cost; accept if NPV>0. IRR sets NPV=0; best for conventional cash flows and comparable scales.")
    if not lines:
        lines.append(textwrap.shorten(" ".join([ch.text for ch, _ in retrieved]), width=500, placeholder="…"))

    citations = [{"doc_id": ch.doc_id, "chunk_id": ch.chunk_id, "url": ch.url} for ch, _ in retrieved]
    out = {"query": query, "answer": " ".join(lines), "citations": citations}
    if show_steps:
        out["reasoning_steps"] = _cot_plan(query) + bullets
    return out

# ===== Reasoning Eval: helpers =====
def _plan_subqueries(q: str) -> list[str]:
    """
    Lightweight subquery planner.
    Returns at least one subquery (the original query) if no good split found.
    """
    qn = q.strip()
    if not qn:
        return []
    # Split by common conjunctions/punctuation
    raw = re.split(r"\b(?:and|;|/|,| vs\.?| versus )\b|\?", qn, flags=re.I)
    subs = [s.strip() for s in raw if s and len(s.strip()) > 2]
    # If too many, keep only the first 3
    return subs or [qn]


def _metric_groundedness(citation_count: int, min_required: int = 1) -> float:
    """Groundedness: estimates if the answer is grounded on provided citations."""
    return 1.0 if citation_count >= min_required else 0.0


def _metric_coverage(subqueries: list[str], answered_texts: list[str]) -> float:
    """
    Coverage: estimates if the answer covers the subqueries.
    """
    if not subqueries:
        return 1.0
    import string
    trans = str.maketrans("", "", string.punctuation)
    hits = 0
    for sq in subqueries:
        tokens = [t.lower() for t in sq.translate(trans).split() if len(t) > 2][:3]
        hit = any(all(tok in ans.lower() for tok in tokens) for ans in answered_texts) if tokens else False
        hits += 1 if hit else 0
    return hits / max(1, len(subqueries))


def _metric_consistency(partials: list[str]) -> float:
    """
    Consistency: Consistency estimation.
    Only very-weak heuristic is provided, which can be replaced by a proper NLI model.
    Result: 1--consistent, 0--inconsistent.
    """
    text = " ".join(partials).lower()
    # Conflict clues
    conflicts = [
        ("always", "never"),
        ("increase", "decrease"),
        ("positive", "negative"),
        ("accept", "reject"),
    ]
    for a, b in conflicts:
        if a in text and b in text:
            return 0.0
    return 1.0


def _log_event(
    query: str,
    mode: str,
    subqueries: list[str],
    answer: str,
    citations: list[dict],
    partial_answers: list[str]
):
    ts = datetime.utcnow().isoformat()
    citation_count = len(citations)
    groundedness = _metric_groundedness(citation_count, min_required=1)
    coverage = _metric_coverage(subqueries, partial_answers if partial_answers else [answer])
    consistency = _metric_consistency(partial_answers if partial_answers else [answer])
    chunk_ids = ";".join([c.get("chunk_id","") for c in citations])

    line = (
        f"{ts},{json.dumps(query, ensure_ascii=False)},"
        f"{mode},{json.dumps(subqueries, ensure_ascii=False)},"
        f"{json.dumps(answer, ensure_ascii=False)},"
        f"{citation_count},{groundedness:.3f},{coverage:.3f},{consistency:.3f},"
        f"{json.dumps(chunk_ids, ensure_ascii=False)}\n"
    )
    with LOG_PATH.open("a", encoding="utf-8") as w:
        w.write(line)

    return {
        "groundedness": groundedness,
        "coverage": coverage,
        "consistency": consistency,
        "citation_count": citation_count
    }


# ---------- Public API ----------
def init_backend(corpus_jsonl_path: str, max_sent_per_chunk: int = 2) -> int:
    """
    Load JSONL corpus and build the TF-IDF index.
    Each line in corpus must be a JSON object: {id, title, url, text}
    Returns: number of chunks built.
    """
    global _chunks, _index
    docs: List[Dict[str, str]] = []
    p = Path(corpus_jsonl_path)
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            docs.append({
                "id": rec["id"],
                "title": rec.get("title", ""),
                "url": rec.get("url", ""),
                "text": rec.get("text", "")
            })
    _chunks = _build_chunks(docs, max_sent_per_chunk=max_sent_per_chunk)
    _index = _TfidfIndex(_chunks)
    return len(_chunks)

def answer_with_self_ask(query: str, k: int = 4, show_steps: bool = True, log: bool = True) -> Dict[str, Any]:
    """
    Self-Ask answer:
      1) Plan subqueries
      2) Query each subquestion, synthesize partial answers
      3) Summarize final answer
    """
    assert _index is not None, "Backend not initialized. Call init_backend(corpus_jsonl_path) first."

    subqs = _plan_subqueries(query)
    steps = [f"Plan subqueries: {subqs}"]
    partial_answers, all_cites = [], []

    for sq in subqs:
        retrieved = _index.query(sq, top_k=k)
        partial = _synthesize_answer(sq, retrieved, show_steps=False)
        partial_answers.append(partial["answer"])
        all_cites += partial["citations"]
        steps.append(f"SubQ: {sq} -> retrieved {len(retrieved)} chunks")

    # Summarize final answer
    final_answer = " ".join(partial_answers)
    out = {
        "query": query,
        "answer": final_answer,
        "citations": all_cites,
        "reasoning_steps": steps if show_steps else None,
        "mode": "self-ask"
    }

    if log:
        metrics = _log_event(
            query=query,
            mode="self-ask",
            subqueries=subqs,
            answer=final_answer,
            citations=all_cites,
            partial_answers=partial_answers
        )
        out["metrics"] = metrics

    return out


def answer(query: str, k: int = 4, show_steps: bool = True, log: bool = True) -> Dict[str, Any]:
    """
    Baseline answer: direct retrieval + synthesis.
    """
    assert _index is not None, "Backend not initialized. Call init_backend(corpus_jsonl_path) first."
    retrieved = _index.query(query, top_k=k)
    res = _synthesize_answer(query, retrieved, show_steps=show_steps)
    res["mode"] = "baseline"

    if log:
        metrics = _log_event(
            query=query,
            mode="baseline",
            subqueries=[query],              # no subqueries
            answer=res["answer"],
            citations=res["citations"],
            partial_answers=[res["answer"]]  # no partial answers
        )
        res["metrics"] = metrics

    return res

