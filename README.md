# Meridian Procurement Intelligence

**HCLTech Campus Ambassador × The Economic Times — Assignment 2**

RAG assistant over Meridian’s Q1 supply-chain performance review and procurement policy handbook.

> **API note (campus guidance):** OpenAI was unavailable / invalid for this build, so the default provider is **Google Gemini** (free), as allowed in the doubt-clearing alternatives (Gemini / Ollama / other keys). OpenAI remains supported via `LLM_PROVIDER=openai`.

## Features

- PDF upload + index into **one** Chroma collection  
- Recursive chunking (1200 / 150)  
- Persistent ChromaDB  
- Cross-document retrieval (review + policy)  
- Grounded answers with **document + page** citations  
- Trap-question refusal  
- Streamlit UI  
- Optional FastAPI bonus (`/ingest`, `/ask`, `/stats`)

## Project structure

```text
supplychain-rag/
├── app.py
├── ingest.py
├── rag.py
├── config.py
├── run_eval.py
├── api/main.py
├── data/
│   ├── Meridian_Supply_Chain_Review_Q1_FY2025-26.pdf
│   └── Meridian_Procurement_Policy_Handbook_v4.2.pdf
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

Edit `.env`:

```text
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_gemini_key_here
```

Get a free key: https://aistudio.google.com/apikey

Optional OpenAI:

```text
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
EMBEDDING_MODEL=text-embedding-3-small
CHAT_MODEL=gpt-4o
```

## Run

```powershell
# Index once
python ingest.py

# UI
streamlit run app.py

# Optional API bonus
uvicorn api.main:app --reload --port 8000

# Run all 10 test questions → eval_results.json
python run_eval.py
```

In the UI: **Save key** → **Index Documents** → ask questions (or click Q1–Q10).

## Design decisions

### Chunking

| Setting | Value | Why |
|---------|-------|-----|
| Chunk size | **1200** | Keeps scorecard tables and numbered penalty clauses with their triggers |
| Overlap | **150** | Prevents mid-sentence split at boundaries |

### Cross-document retrieval (Stage 6)

**Chosen fix:** retrieve a fixed share from each `doc_type` (`review` + `policy`) in parallel, then merge.

This ensures questions 5–9 receive both figures and clauses.

### Prompt / grounding

- Answer only from retrieved context  
- Explicit refusal when absent  
- Figure + clause + action for combined questions  
- Safety stock: higher of formula vs floor  
- Temperature **0.1**

## Persistence

Chroma is stored in `chroma_db/`. Restart the app without re-indexing; the store survives. Re-index is skipped automatically when PDFs are unchanged (use **Force rebuild** to refresh).

## Test questions — expected gold answers (from PDFs)

Record your live app answers in the right column after running `python run_eval.py`.

| # | Question focus | Expected from PDFs | Docs |
|---|----------------|--------------------|------|
| 1 | Highest spend + OTD | **Shenzhen Rui Electronics**, ₹21.9 crore, **79.5%** OTD | Review |
| 2 | Line stoppages | **7** events, **41 hours**; mainly MCU shortages (Shenzhen Rui), Trident PCB rejects, one transport strike | Review |
| 3 | PO ₹1.4 crore approval | **Chief Operating Officer** (above ₹1 crore up to ₹5 crore) | Policy |
| 4 | Supplier classes | **Critical, Strategic, Standard, Tail**; Critical = single-source **or** spend > ₹10 crore **or** safety-related | Policy |
| 5 | Kaveri 88.1% / 1150 PPM | Clauses **6.1** (OTD < 90%) and **6.3** (PPM > 500): warning + weekly review; ₹120/unit rework + 100% inspection at supplier cost | Both |
| 6 | Single-source MCU | Policy **7.1** dual-source within 12 months; Action 1 qualify **Anh Long Semiconductors** by 30 Sep 2025 | Both |
| 7 | Safety stock 46-day import | 46×0.25 = 11.5; Critical imported floor **30** → hold **30 days** | Both |
| 8 | Trident 640 PPM | Clause **6.3**: rework at **₹120/unit** + 100% inspection at supplier cost | Both |
| 9 | Below B on OTD alone | OTD **&lt; 75%** cannot score band B; **no Q1 supplier** is below 75% (worst Shenzhen 79.5%). Escalation Levels **1–4** (§10) | Both |
| 10 | Trap salary | Must refuse: **information is not available in the uploaded documents** | — |

### Honest notes

- Provider is Gemini per campus alternatives (OpenAI key rejected as invalid during setup).  
- Q9 wording maps to OTD &lt; 75% for “below B on OTD alone”; if the app lists OTD &lt; 90% suppliers, show the handbook sentence in the demo.  
- Paste live answers from `eval_results.json` into this README before final submission.

## Screenshots

Add screenshots here before submitting:

1. Index success (chunk count visible)  
2. Cross-document answer (Q5 or Q7) with **both** sources  
3. Trap question refusal (Q10)

## Demo video (3 minutes)

1. 0:00–0:20 — introduce the two PDFs  
2. 0:20–1:00 — upload/index + chunk count  
3. 1:00–2:30 — two cross-doc questions with both sources  
4. 2:30–3:00 — trap question refused  

## Marking self-check

- [x] Ingestion pipeline  
- [x] Chroma persistence  
- [x] Single-doc questions supported  
- [x] Cross-doc retrieval + sources  
- [x] Streamlit UI + graceful empty-index handling  
- [x] Trap refusal prompt  
- [x] README + `.env` hygiene  
- [x] FastAPI bonus endpoints  
- [ ] Screenshots attached  
- [ ] Live 10-answer table filled from `run_eval.py`  
- [ ] Demo video recorded  
- [ ] Public GitHub link submitted  
