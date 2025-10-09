import os, json, re, pathlib, time
from typing import List, Dict
import requests

# Optional：PDF/HTML dependency
# pip install pymupdf readability-lxml html2text pdfminer.six beautifulsoup4 lxml
# Import PyMuPDF (fitz)
# If importing wrong packages withoug API（比如没有 open） method），try pymupdf instead
try:
    import fitz  # try the common name
    # some unrelated packages also use the name `fitz`; ensure this is PyMuPDF by checking for `open`
    if not hasattr(fitz, "open"):
        raise ImportError("imported 'fitz' has no attribute 'open', trying 'pymupdf' instead")
except Exception:
    # fallback to the explicit pymupdf package
    import pymupdf as fitz
from readability import Document
from bs4 import BeautifulSoup
import html2text

BASE = pathlib.Path(__file__).resolve().parents[1]  # project/
DATA = BASE / "data"
RAW = DATA / "raw"
PROC = DATA / "processed"
RAW.mkdir(parents=True, exist_ok=True)
PROC.mkdir(parents=True, exist_ok=True)

def load_sources(path=DATA/"corpus_sources.json") -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def safe_filename(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", s)

def download(url: str, out_path: pathlib.Path):
    # Example only, can be extended with retries, proxies, etc.
    r = requests.get(url, timeout=30, headers={"User-Agent":"Mozilla/5.0"})
    r.raise_for_status()
    out_path.write_bytes(r.content)

def extract_pdf_text(pdf_path: pathlib.Path) -> str:
    doc = fitz.open(pdf_path)
    parts = []
    for page in doc:
        parts.append(page.get_text("text"))
    return "\n".join(parts)

def extract_html_text(html_path: pathlib.Path) -> str:
    raw = html_path.read_text("utf-8", errors="ignore")
    doc = Document(raw)
    article_html = doc.summary()  # Main content
    soup = BeautifulSoup(article_html, "lxml")
    # Delete unwanted tags
    for tag in soup(["script","style","nav","footer","header"]):
        tag.decompose()
    handler = html2text.HTML2Text()
    handler.body_width = 0
    handler.ignore_images = True
    handler.ignore_links = True
    text = handler.handle(str(soup))
    # Simple cleanup
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text

def ensure_processed_item(item: Dict):
    doc_id = item["id"]
    url = item.get("url","")
    fbase = safe_filename(doc_id)

    # Determine file extension
    ext = ""
    if url.endswith(".pdf"):
        ext = ".pdf"
    elif url.startswith("http"):
        ext = ".html"

    raw_path = RAW / f"{fbase}{ext if ext else '.txt'}"
    proc_path = PROC / f"{fbase}.txt"

    if not proc_path.exists():
        # If not exists, fetch or ask for manual input
        if not raw_path.exists():
            if url.startswith("http"):
                print(f"[fetch] {doc_id} <- {url}")
                try:
                    download(url, raw_path)
                    time.sleep(1.0)  
                except Exception as e:
                    print(f"  Download failed, use placeholder: {e}")
                    raw_path.write_text(f"[PLACEHOLDER for {doc_id}] Provide your manual excerpt here.", encoding="utf-8")
            else:
                # No URL, create a placeholder for manual input
                if not raw_path.exists():
                    raw_path.write_text(f"[MISSING SOURCE for {doc_id}] Put your excerpt in this file.", encoding="utf-8")

        # Extract text based on file type
        if raw_path.suffix.lower()==".pdf":
            text = extract_pdf_text(raw_path)
        elif raw_path.suffix.lower()==".html":
            text = extract_html_text(raw_path)
        else:
            text = raw_path.read_text("utf-8", errors="ignore")

        # Minimal cleanup
        text = re.sub(r"\s+\n", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        proc_path.write_text(text, encoding="utf-8")

    return proc_path

def build_corpus_docs_jsonl(sources: List[Dict], out_path=DATA/"corpus_docs.jsonl"):
    with open(out_path, "w", encoding="utf-8") as w:
        for item in sources:
            proc_path = ensure_processed_item(item)
            text = proc_path.read_text("utf-8", errors="ignore")
            record = {
                "id": item["id"],
                "title": item.get("title",""),
                "url": item.get("url",""),
                "text": text
            }
            w.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"[done] wrote {out_path}")

if __name__ == "__main__":
    sources = load_sources()
    build_corpus_docs_jsonl(sources)
