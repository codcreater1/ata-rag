<div align="center">

# 🎓 ATA RAG

**A retrieval-augmented chatbot for [akademiata.pl](https://akademiata.pl)** — ask about tuition, admissions, programmes or student services and get a grounded answer with source links, in the language you asked in.

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![pgvector](https://img.shields.io/badge/pgvector-Neon-336791?logo=postgresql&logoColor=white)
![Gemini](https://img.shields.io/badge/Embeddings-Gemini-4285F4?logo=google&logoColor=white)
![React](https://img.shields.io/badge/React-61DAFB?logo=react&logoColor=black)

</div>

---

## Why it is built this way

Two findings from studying the site shaped the design:

1. **The public site is `akademiata.pl`, not `akademiata.edu.pl`** (the latter is the Moodle e-learning login). Easy to get wrong.
2. **Tuition figures are not in the page HTML.** Programme pages only link to a JavaScript "tuition calculator" that loads its numbers from a Google Apps Script endpoint. A crawl-and-embed pipeline alone therefore *cannot* answer "what is tuition for Computer Science?". We ingest that structured pricing data directly and turn each programme into a fact card, so tuition answers come back exact and sourced.

## Architecture

```
akademiata.pl sitemaps          Google Apps Script (tuition JSON)
        │                                │
   crawl + trafilatura            structured fee cards (PLN / EUR)
        │                                │
        └──────────── heading-aware chunking ───────────┘
                              │
                    Gemini embeddings (768-dim)
                              │
                    Postgres + pgvector (Neon)
   ───────────────────────────────────────────────────────
   question → embed → vector search → confidence gate → grounded LLM
                              │
                   answer + citations + confidence
```

**Deliberate choices**

- **Confidence gate** — if the best match is weak, the bot says it could not find the answer instead of guessing. For a university bot a wrong tuition figure is worse than "I don't know".
- **Cross-lingual retrieval** — the site is only partly translated, so a Polish page often answers an English question. One multilingual embedding space, no translation step; the model replies in the question's language.
- **Structured tuition ingestion** — the single most-asked question is answered from data, not scraped prose.

## Components

| Part | Path |
|---|---|
| Crawler (sitemap-driven, main-content extraction) | `backend/app/ingest/crawler.py` |
| Structured tuition ingestion (PLN + EUR) | `backend/app/ingest/prices.py` |
| Heading-aware chunker | `backend/app/ingest/chunker.py` |
| Embeddings (Gemini, 768-dim, normalised) | `backend/app/core/embeddings.py` |
| Vector store + search (pgvector) | `backend/app/core/db.py` |
| RAG answering (gate + grounding) | `backend/app/services/rag.py` |
| Chat API | `backend/app/routers/chat.py` |
| Dashboard analytics API | `backend/app/routers/dashboard.py` |
| Frontend (chat + dashboard) | `frontend/` |

## Getting started

```bash
cd backend
pip install -r requirements.txt
cp .env.example .env          # fill in GOOGLE_API_KEY, LLM_API_KEY, DATABASE_URL

python -m app.ingest.indexer  # crawl + tuition + embed into pgvector
uvicorn app.main:app --reload
```

Ask a question:

```bash
curl -X POST localhost:8000/chat/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"What is the tuition for Computer Science?"}'
```

## Observability

Every answer is traced in LangFuse (retrieval scores, latency, tokens) when `LANGFUSE_*` keys are set, and every question — answered or not — is logged to the `queries` table. The dashboard surfaces common questions, **unanswered ones** (where the knowledge base has gaps), retrieval quality and feedback.

## Nightly refresh

`python -m app.ingest.indexer` runs as a Coolify scheduled task. Pages whose content is byte-identical to the stored copy are skipped, so a routine run re-embeds only what changed.

## License

MIT
