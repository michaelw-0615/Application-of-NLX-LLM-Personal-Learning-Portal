# Application-of-NLX-LLM-Personal-Learning-Portal
This is a repository for data, code and reports of the Personal Learning Portal project of CMU course 95820. It is a self-contained Retrieval-Augmented Generation (RAG) learning assistant for Corporate Finance with visible reasoning, reflection logging, and a reproducible evaluation pipeline. It is designed to run fully in Google Colab with no external LLM APIs.

## Project Structure
```php
Application-of-NLX-LLM-Personal-Learning-Portal/
├─ data/
│  ├─ /processed                   # Processed corpus
│  ├─ /raw                         # Raw corpus
│  ├─ corpus_sources.json          # Corpus sources
│  └─ corpus_docs.jsonl            # your curated corpus: {id,title,url,text} per line
├─ src/
│  ├─ build_corpus.py
│  └─ plp_backend_colab.py         # backend module: load → chunk → index → answer(+logging)
├─ notebooks/
│  ├─ PLP_Colab_Backend.ipynb                   
│  ├─ PLP_Colab_Frontend_UI.ipynb           
│  └─ PLP_RAG_Query_Response.ipynb # For testing and evaluation
├─ diagrams/
│  ├─ plp_system_diagram.png       # system overview diagram
│  └─ plp_flow_diagram.png         # pipeline tree diagram
├─ reports/                        # Experiment results, reports and appendices
└─ README.md
```
Paths above assume Google Drive mount at /content/drive/My Drive/Project_3/…. Adjust as needed if you run locally.

## Dependencies / Requirements
- Python: 3.10–3.12 (Colab OK)
- Core packages
```
scikit-learn>=1.3
pandas>=2.0
gradio>=4.40,<5
streamlit>=1.34
pyngrok==4.1.1            # only if using Streamlit public URL in Colab
Optional (future upgrades):
sentence-transformers>=2.2  # for MiniLM re-rank (not required for baseline)
rank-bm25                   # BM25 recall (optional)
requirements.txt (minimal)
scikit-learn>=1.3
pandas>=2.0
gradio>=4.40,<5
streamlit>=1.34
pyngrok==4.1.1
```
- Data Format: 
```json
// data/corpus_docs.jsonl (UTF-8, one JSON per line):
{"id":"doc_wacc_01","title":"Estimating WACC","url":"local://wacc","text":"The weighted average cost of capital (WACC) ..."}
{"id":"doc_npv_01","title":"NPV and IRR","url":"local://npv","text":"Net Present Value discounts expected cash flows ..."}
// Required fields: id, text.
// Optional: title, url.
// Texts are sentence-split and chunked (2 sentences/chunk by default).
```

## Quick Start (Colab)
- Mount Drive & open notebooks
- Upload/organize the repo under My Drive/Project_3/
- Open notebooks/PLP_Colab_Frontend_UI.ipynb, (or PLP_RAG_Query_Response.ipynb for testing and evaluation) and run each cell sequentially according to instructions. Change directories where necessary.

## License & Attribution
- Code: MIT
