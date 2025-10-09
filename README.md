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
Mount Drive & open notebooks
Upload/organize the repo under My Drive/Project_3/
Open frontends/PLP_Colab_Frontend_UI.ipynb
Install deps (Step 0 in the notebook)
!pip -q install scikit-learn gradio "streamlit>=1.34" pyngrok==4.1.1
Mount Drive (Step 1)
from google.colab import drive
drive.mount('/content/drive')
Backend init (Step 3)
from plp_backend_colab import init_backend, answer
CORPUS_JSONL = '/content/drive/My Drive/Project_3/data/corpus_docs.jsonl'
n = init_backend(CORPUS_JSONL, max_sent_per_chunk=2)
print("chunks:", n)
Run Gradio UI (Step 4)
Execute the UI cell and uncomment demo.launch().
Ask questions; toggle “Show Reasoning (CoT-style)” and choose Baseline / Self-Ask.
Use the Reflection panel to save confidence/usefulness/confusions → saved to eval/reflections.jsonl.
(Optional) Streamlit
The notebook writes frontends/plp_streamlit_app_colab.py.
# Optional: public URL in Colab
from pyngrok import ngrok
public_url = ngrok.connect(8501).public_url
print("Public URL:", public_url)
!streamlit run /content/drive/My\ Drive/Project_3/frontends/plp_streamlit_app_colab.py \
    --server.port 8501 --server.address 0.0.0.0
Running Evaluation
Open eval/eval.ipynb and run cells in order:
Imports & backend init
from google.colab import drive
drive.mount('/content/drive')

import sys
SRC="/content/drive/My Drive/Project_3/src"
if SRC not in sys.path: sys.path.append(SRC)

from plp_backend_colab import init_backend, answer, answer_with_self_ask
init_backend('/content/drive/My Drive/Project_3/data/corpus_docs.jsonl')
Define test queries (covering Cost of Capital, Capital Budgeting, Capital Structure, Payout, Governance).
Run suites (baseline & self-ask) — logs are auto-written to eval/reasoning_log.csv:
df_base = run_suite(eval_queries, mode="baseline", k=4)
df_self = run_suite(eval_queries, mode="self-ask", k=4)
df_all  = pd.concat([df_base, df_self], ignore_index=True)
df_all.to_csv("/content/drive/My Drive/Project_3/eval/run_records.csv", index=False)
Aggregate metrics → final_metrics.csv
Metrics collected: groundedness (≥1 citation), coverage (sub-Q hits), consistency (no contradictions), citation_count, answer_len.
(Optional) Merge reflections for module-level insights:
import json, pandas as pd, pathlib
ref = []
path = pathlib.Path("/content/drive/My Drive/Project_3/eval/reflections.jsonl")
if path.exists():
    with path.open() as f:
        for line in f: ref.append(json.loads(line))
    rdf = pd.DataFrame(ref)
    print(rdf.groupby("module")[["confidence","usefulness"]].mean())
Backend API
src/plp_backend_colab.py exposes:
init_backend(corpus_jsonl_path: str, max_sent_per_chunk: int = 2) -> int
answer(query: str, k: int = 4, show_steps: bool = True, log: bool = True) -> dict
answer_with_self_ask(query: str, k: int = 4, show_steps: bool = True, log: bool = True) -> dict
Return shape
{
  "query": "...",
  "mode": "baseline|self-ask",
  "answer": "...",
  "citations": [{"doc_id":"...", "chunk_id":"...", "url":"..."}],
  "reasoning_steps": ["..."],        // optional if show_steps=True
  "metrics": {                       // only if log=True
    "groundedness": 1.0,
    "coverage": 0.25,
    "consistency": 1.0,
    "citation_count": 4
  }
}
Automatic logging
eval/reasoning_log.csv (created if missing) with timestamp, mode, subquestions, metrics, and chunk IDs.
Reflection & Self-Assessment
Where: in both Gradio & Streamlit UIs.
What: module, confidence (1–5), usefulness (1–5), confusions, next_action, plus current query/answer/citations/mode.
Storage: appended to eval/reflections.jsonl.
Use: import in eval.ipynb to correlate self-ratings with automatic metrics (e.g., groundedness vs usefulness).
Diagrams
System overview: assets/plp_system_diagram.png
Pipeline tree: assets/plp_flow_diagram.png
You can embed them in your report/README:
![System Diagram](assets/plp_system_diagram.png)
![Pipeline Flow](assets/plp_flow_diagram.png)
Troubleshooting
ModuleNotFoundError: plp_backend_colab
Ensure src is on sys.path:
import sys
sys.path.append("/content/drive/My Drive/Project_3/src")
Gradio error: module 'gradio' has no attribute 'blocks'
Caused by old/broken install or local name shadowing. Fix:
!pip uninstall -y gradio
!pip install -U "gradio>=4.40,<5"
# Restart runtime after install
Make sure there is no local file/folder named gradio in your working directory.
Logs not created
Use backend with logging patch and call with log=True. Default log dir is /content/drive/My Drive/Project_3/eval.
Slow/low recall
Increase k, use overlapping chunks, or upgrade retriever (BM25 + MiniLM re-rank).
Roadmap (optional enhancements)
Retriever: BM25 recall → MiniLM re-rank (offline)
Coverage metric 2.0: embedding similarity between sub-Qs and answer sentences
Groundedness tightening: require ≥2 citations from distinct docs
Module-aware retrieval: prioritize sources tagged with the selected learning module
Governance exemplars: add short case snippets with mechanism→outcome pairs
License & Attribution
Code: add your preferred license (e.g., MIT).
Corpus: ensure you own or have rights to include the texts. Cite primary sources when sharing.
Citation (if you reference this project)
@project{plp_corporate_finance_2025,
  title   = {Corporate Finance PLP: Local RAG with Reasoning, Reflection, and Evaluation},
  year    = {2025},
  note    = {Google Colab-ready implementation}
}
Happy learning & evaluating!



思考时间


