# Voyance

**Four minds. One answer.**

Voyance sends your question simultaneously to GPT, Claude, Gemini, and Grok. A fifth model reads all four responses and synthesizes them into a single unified answer — surfacing where the models agree, flagging where they diverge, and producing a conclusion no single model could reach alone.

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
- **AI providers** — OpenAI, Anthropic, Google Gemini, xAI Grok, Perplexity (web search)
- **Rate limiting / logging** — Upstash Redis
- **Deployment** — Vercel

---

## Project structure

```
api/index.py          # FastAPI app — all routes and model calls
templates/
  index.html          # Main app UI
  landing.html        # Marketing landing page
prompts/
  synthesis_text.txt  # Synthesis model system prompt (text)
  synthesis_image.txt # Synthesis model system prompt (images)
public/               # Static assets served directly
logs/queries.jsonl    # Local query log fallback
```

---

## Running locally

**Prerequisites:** Python 3.9+, API keys for OpenAI, Anthropic, Google Gemini, and xAI.

```bash
# Clone and set up
git clone https://github.com/p6oo/OpClave
cd council
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Add your API keys
cp .env.example .env   # then fill in the values
```

Required `.env` keys:

```
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
GEMINI_API_KEY=
XAI_API_KEY=
PERPLEXITY_API_KEY=

# Optional — for rate limiting and logs (Upstash Redis)
UPSTASH_REDIS_REST_URL=
UPSTASH_REDIS_REST_TOKEN=
```

```bash
# Start the dev server
uvicorn api.index:app --reload
```

Open [http://localhost:8000](http://localhost:8000).

---

## Deploying to Vercel

The repo is structured for zero-config Vercel deployment. `vercel.json` rewrites all routes to `api/index.py` via Mangum.

```bash
vercel deploy
```

Set the env vars above in your Vercel project settings.

---

## Rate limits

| Bucket | Limit |
|---|---|
| General queries | 15 per IP per day |
| Intensive queries | 3 per IP per day |
| Feedback submissions | 3 per IP per hour |

Limits are enforced via Upstash Redis. If Redis is unavailable, requests fail open.

---

## Built by

[Aarush Bajaj](https://github.com/p6oo) and [Aditya Mishra](https://www.linkedin.com/in/adityamishra-fintech/)
