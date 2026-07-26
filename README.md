# TDSP1Q5
# Data-Analyst Telegram Bot — TDS Project 1

A Telegram bot backed by an LLM agent that answers data-analysis questions.
Message it a question and it replies with a single JSON object in exactly the
shape the question asks for, plus a public link to its full reasoning log.

**Bot:** `@Rituparna_tdsp1_bot` · **Log:** https://tdsp1q5.onrender.com/run.jsonl

## How it works

```
Telegram user ──▶ Telegram Bot API ──▶ webhook ──▶ FastAPI app (Render)
                                                        │
                                          agent loop (gpt-5-mini via AI Pipe)
                                          │        │
                                    run_python   fetch_url
                                    (pandas +    (browser-header
                                     helpers)     HTTP fetch)
                                                        │
                              one JSON reply ◀──────────┘
                              + JSONL log served at /run.jsonl
```

1. **Webhook receiver.** Telegram delivers each message to `/webhook`. The
   handler acknowledges instantly and hands the work to a background task —
   Telegram would otherwise time out and re-deliver, producing duplicate
   replies. Seen `update_id`s are also tracked and ignored as a second guard.
2. **Agent loop.** The question (plus per-chat history, for multi-turn tasks)
   goes to the LLM with two tools. The model calls tools, reads their output,
   and iterates until it has an answer, up to a round cap. If the cap is hit,
   a final forced call produces the best available answer instead of an error.
3. **Tools.**
   - `run_python` — executes model-written Python with a preloaded namespace:
     `pd`, `requests`, `StringIO`, `BeautifulSoup`, and two helpers —
     `read_tables(url)` (fetch a page with browser headers and return all its
     tables as DataFrames) and `get_text(url)`. Preloading these removes an
     entire class of failures (missing imports, user-agent blocks, pandas'
     StringIO requirement) observed during development.
   - `fetch_url` — plain HTTP fetch with browser-like headers.
4. **Reply shaping.** The final reply matches whatever JSON shape the question
   asks for. The `{"answer": ..., "log_url": ...}` envelope is applied only
   when the question mentions `log_url`; otherwise the bare shape is returned
   (e.g. `{"state": "..."}`).
5. **Run log.** Every question, tool call (with arguments and truncated
   results), and final answer is appended as one JSON object per line to
   `run.jsonl`, served publicly by the same app at `/run.jsonl` — the
   `log_url` in replies. The log resets on redeploys (ephemeral disk).

## Design decisions

- **System prompt discipline.** The prompt forbids fabricated or "sample"
  data, requires grounding answers in actually retrieved data, mandates
  `print()`-and-read verification, prescribes targeted sources over homepage
  fetches, and requires choosing the most recent period when tables span
  several survey years.
- **Multi-turn.** Per-chat message history is kept in memory, so a question
  can refer to data given in earlier messages; the agent answers the latest
  message in context. (History is per-process and clears on restart, which is
  acceptable for grading sessions.)
- **Failure containment.** LLM errors, tool crashes, and loop-cap exhaustion
  all degrade to a valid JSON reply rather than silence or prose.

## Stack

FastAPI + Uvicorn · httpx · pandas / lxml / html5lib · requests ·
BeautifulSoup · gpt-5-mini via [AI Pipe](https://aipipe.org) (OpenAI-compatible
endpoint) · deployed on Render (free tier) · kept warm by an UptimeRobot
monitor pinging `/` every 5 minutes.

## Deploying it yourself

1. Create a bot with @BotFather; note the token.
2. Deploy this repo as a Render (or similar) Python web service:
   - Build: `pip install -r requirements.txt`
   - Start: `uvicorn bot:app --host 0.0.0.0 --port $PORT`
   - Environment variables: `TELEGRAM_TOKEN` (BotFather), `LLM_API_KEY`
     (AI Pipe or another OpenAI-compatible provider), `PUBLIC_URL`
     (the deployed base URL, no trailing slash).
3. Point Telegram at the deployment (one-time):
   `https://api.telegram.org/bot<TOKEN>/setWebhook?url=<PUBLIC_URL>/webhook`
4. Message the bot. Check `<PUBLIC_URL>/run.jsonl` for the reasoning log.

No secrets live in this repository; all credentials are supplied via
environment variables at deploy time.

## Testing

Verified against the course's public grading pipeline
([tds-p1-t2-2026-telegram-bot](https://github.com/Jivraj-18/tds-p1-t2-2026-telegram-bot)):
the collector received exactly one correctly-shaped JSON reply within the
timeout, and the grader matched it against the local answer key (1/1).
Manual tests covered the worked-example envelope shape, bare shapes,
inline-data computation, and multi-turn context.
