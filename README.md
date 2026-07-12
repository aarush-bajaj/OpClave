# OpClave

**Four minds. One answer.**

OpClave sends your question simultaneously to GPT, Claude, Gemini, and Grok. A fifth model reads all four responses and synthesizes them into a single unified answer — surfacing where the models agree, flagging where they diverge, and producing a conclusion no single model could reach alone. That's **Voyance**, the core chat experience.

**Studio** is the companion workspace for heavier tasks:

- **Build** — generate a working single-file app or component, then iterate on it conversationally
- **Research** — a multi-model research explorer with live web sources, verified academic citations, and auto-generated charts
- **Review** — three specialized agents (security / quality / performance) audit a code snippet in parallel

Live at **[opclave.com](https://opclave.com)**

---

## How it works

1. You ask a question (or attach a file)
2. All four models answer independently and in parallel
3. A synthesis model reads every response and produces one unified answer with consensus, disagreement, and a final judgment

Two modes are available:

| Mode | Models | Use case |
|---|---|---|
| **General** | GPT-mini, Claude Sonnet, Gemini Flash, Grok Fast | Everyday questions, research, decisions |
| **Intensive** | GPT, Claude Opus, Gemini Pro | Coding, complex reasoning, technical tasks |

---

## Stack

- **Backend** — FastAPI + Mangum (deployed as a Vercel serverless function)
- **Frontend** — Vanilla HTML/CSS/JS with Tailwind
- **AI providers** — OpenAI, Anthropic, Google Gemini, xAI Grok, Perplexity (web search + research)
- **Auth & sync** — Supabase (email/password auth, Postgres with row-level security for threads/artifacts)
- **Rate limiting / logging** — Upstash Redis
- **Deployment** — Vercel

---

## Project structure

```
api/index.py            # FastAPI app — all routes and model calls
templates/
  index.html             # Main app UI (Voyance + Studio)
  landing.html           # Marketing landing page
  studio.html            # Studio mode picker
  preview_matrix.html    # Consensus matrix preview/demo page
  privacy.html           # Privacy policy
  assets/                # Fonts, logos, compiled Tailwind CSS
prompts/
  synthesis_text.txt     # Synthesis model system prompt (text)
  synthesis_image.txt    # Synthesis model system prompt (images)
public/                  # Static landing assets served directly
src/tailwind.css         # Tailwind source (compiled to templates/assets/tailwind.css)
supabase_schema.sql      # Postgres schema + RLS policies for threads/artifacts
logs/queries.jsonl       # Local query log fallback (gitignored)
```

---

## Running locally

**Prerequisites:** Python 3.9+, Node (only needed if you're editing Tailwind styles), and an API key for at least one model provider.

```bash
git clone https://github.com/aarush-bajaj/OpClave.git
cd OpClave
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env   # then fill in the values you have
```

See [`.env.example`](.env.example) for the full list of keys. Only the model provider keys (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `XAI_API_KEY`, `PERPLEXITY_API_KEY`) are required — Redis rate limiting, Supabase auth/sync, feedback email, and admin protection are all optional and degrade gracefully when unset.

If you want cloud-synced threads/artifacts and sign-in, run [`supabase_schema.sql`](supabase_schema.sql) in your Supabase project's SQL editor (Database → SQL Editor → New query) before setting `SUPABASE_URL` / `SUPABASE_ANON_KEY`.

```bash
uvicorn api.index:app --reload
```

Open [http://localhost:8000](http://localhost:8000).

---

## Deploying to Vercel

The repo is structured for zero-config Vercel deployment. `vercel.json` rewrites all routes to `api/index.py` via Mangum.

```bash
vercel deploy
```

Set the env vars from `.env.example` in your Vercel project settings. In production, always set `ADMIN_CODE` — without it, the `/logs` admin endpoints are open to anyone.

---

## Rate limits

| Bucket | Limit |
|---|---|
| General queries | 15 per IP (or account) per day |
| Intensive queries | 3 per IP (or account) per day |
| Feedback submissions | 3 per IP per hour |
| File uploads | 10 per IP per hour |

Limits are enforced via Upstash Redis. If Redis is unavailable, requests fail open.

---

## License

MIT — see [LICENSE](LICENSE).

---

## Built by

[Aarush Bajaj](https://github.com/p6oo) and [Aditya Mishra](https://www.linkedin.com/in/adityamishra-fintech/)
