"""
Data-Analyst Telegram Bot — v3.2.
New in v3.2:
  - temperature=0 for more deterministic answers across runs.
  - Recency rule made procedural: enumerate period columns, pick the
    single most recent, use only that one.
v3.1 recap: replies match the exact JSON shape each question asks for —
the {"answer": ..., "log_url": ...} envelope only when the question
mentions log_url; otherwise the bare shape.
v3 recap: preloaded data helpers (read_tables, get_text) inside
run_python; forced final answer when the round cap is hit.
v2 recap: instant webhook ack + background task (no duplicate replies),
duplicate update_id protection, strict anti-fabrication system prompt.

Environment variables (set on the host):
  TELEGRAM_TOKEN, LLM_API_KEY, PUBLIC_URL

Run locally:  python -m uvicorn bot:app --port 8000
Set webhook:  https://api.telegram.org/bot<TOKEN>/setWebhook?url=<PUBLIC_URL>/webhook
"""

import os, io, json, contextlib, traceback
from io import StringIO
import httpx
import requests as _requests
import pandas as _pd
from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import FileResponse

TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
LLM_API_KEY = os.environ["LLM_API_KEY"]
PUBLIC_URL = os.environ.get("PUBLIC_URL", "http://localhost:8000")
LLM_ENDPOINT = "https://aipipe.org/openai/v1/chat/completions"
MODEL = "gpt-5-mini"
LOG_PATH = "run.jsonl"
LOG_URL = f"{PUBLIC_URL}/run.jsonl"

BROWSER_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                                 "AppleWebKit/537.36 (KHTML, like Gecko) "
                                 "Chrome/126.0 Safari/537.36"}

app = FastAPI()
chat_histories: dict[int, list] = {}  # chat_id -> message history (multi-turn!)
processed_updates: set[int] = set()   # update_ids we've already handled


def log(event: dict):
    """Append one JSON object per line to the run log (this is your log_url)."""
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(event, default=str) + "\n")


@app.get("/run.jsonl")
def serve_log():
    """Public, wget-able log file — graders download this."""
    if not os.path.exists(LOG_PATH):
        log({"event": "log_created"})  # ensure the URL never 404s
    return FileResponse(LOG_PATH, media_type="application/jsonl")


@app.get("/")
def health():
    return {"status": "ok"}  # useful for uptime pingers


# ------------- Helpers preloaded into the agent's Python -------------

def _get_text(url: str, timeout: int = 60) -> str:
    """Fetch a URL with browser-like headers; return the response text."""
    r = _requests.get(url, headers=BROWSER_HEADERS, timeout=timeout,
                      allow_redirects=True)
    r.raise_for_status()
    return r.text


def _read_tables(url: str, timeout: int = 60):
    """Fetch a web page and return all its tables as a list of DataFrames.
    Handles user-agent headers and StringIO wrapping correctly."""
    return _pd.read_html(StringIO(_get_text(url, timeout)))


# ---------------------- The agent's tools ----------------------

def run_python(code: str) -> str:
    """Execute Python and return stdout (or the traceback).
    The namespace comes preloaded with helpers and common imports."""
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        BeautifulSoup = None
    namespace = {
        "__name__": "__main__",
        "pd": _pd, "pandas": _pd,
        "requests": _requests,
        "StringIO": StringIO,
        "BeautifulSoup": BeautifulSoup,
        "read_tables": _read_tables,
        "get_text": _get_text,
        "BROWSER_HEADERS": BROWSER_HEADERS,
    }
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            exec(code, namespace)
        return buf.getvalue()[:8000] or "(no output — use print())"
    except Exception:
        out = buf.getvalue()[:3000]
        return (out + "\nERROR:\n" if out else "ERROR:\n") \
            + traceback.format_exc()[:4000]


def fetch_url(url: str) -> str:
    """Download a page/dataset so the model can inspect it."""
    try:
        r = httpx.get(url, timeout=60, follow_redirects=True,
                      headers=BROWSER_HEADERS)
        return r.text[:8000]
    except Exception as e:
        return f"ERROR: {e}"


TOOLS = [
    {"type": "function", "function": {
        "name": "run_python",
        "description": ("Run Python code. PRELOADED in the namespace: "
                        "pd (pandas), requests, StringIO, BeautifulSoup, "
                        "read_tables(url) -> list of DataFrames from a web "
                        "page's tables (handles headers/parsing for you), "
                        "get_text(url) -> page text with browser headers. "
                        "Always print() what you want to see."),
        "parameters": {"type": "object", "properties": {
            "code": {"type": "string"}}, "required": ["code"]}}},
    {"type": "function", "function": {
        "name": "fetch_url",
        "description": "Fetch the text content of a public URL.",
        "parameters": {"type": "object", "properties": {
            "url": {"type": "string"}}, "required": ["url"]}}},
]

SYSTEM_PROMPT = """You are a careful data analyst agent answering via Telegram.

RULES — follow all of them:
1. NEVER answer from memory alone, and NEVER fabricate, simulate, or use
   "sample"/"example"/placeholder data — inventing a dataset and analyzing
   it is the worst possible failure. If you cannot obtain real data after
   genuine attempts, answer from your best knowledge as a last resort and
   never from an invented table.
2. In run_python these are ALREADY available (do not re-import or redefine):
   pd (pandas), requests, StringIO, BeautifulSoup, and two helpers:
     read_tables(url)  -> list of DataFrames from all tables on a web page
     get_text(url)     -> page text fetched with proper browser headers
   For tabular web data (e.g. Wikipedia lists), read_tables(url) is the
   fastest reliable route: call it, print the tables, pick the right one.
3. Finding data: homepages are useless (navigation HTML, not data). Use
   targeted pages — a specific Wikipedia list article, data.gov.in API
   (https://api.data.gov.in), or direct CSV/XLSX/JSON links. If a source
   fails, do NOT retry the same URL — change the URL or strategy.
4. Always print() what you want to see and READ the output before drawing
   conclusions. Work step by step: locate, load, compute, sanity-check.
   If entity names look generic ("State A") or values look fabricated,
   the data is WRONG — discard it and find the true source.
   When a table has multiple time-period columns, FIRST list all period
   columns, explicitly identify the single most recent one, and use ONLY
   that column for the answer (unless the question names a period).
5. Be efficient: you have a limited number of steps. Prefer one
   read_tables call over many exploratory fetches.
6. In multi-turn conversations, answer the LATEST message, using earlier
   messages as context.
7. Your FINAL reply must be EXACTLY one JSON object matching the shape the
   question asks for — no prose, no markdown fences, no explanations
   around it. Match the requested shape precisely: if the question shows
   {"state": "..."}, reply with exactly that shape; if it shows an
   {"answer": ..., "log_url": ...} envelope, use that.
8. Do not invent the log_url; the server fills it in."""


def _parse_final(raw: str, question: str = "") -> dict:
    """Turn the model's final text into the reply object, matching the
    shape the question asked for. Only add the answer/log_url envelope
    when the question mentions log_url."""
    raw = (raw or "").strip()
    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return {"answer": raw, "log_url": LOG_URL}   # unparseable fallback
    wants_envelope = "log_url" in question
    if wants_envelope:
        if not (isinstance(obj, dict) and "answer" in obj):
            obj = {"answer": obj}
        obj["log_url"] = LOG_URL                     # server owns this field
        return obj
    # Question wants a bare shape: unwrap if the model added the envelope.
    if isinstance(obj, dict) and "answer" in obj and \
            set(obj.keys()) <= {"answer", "log_url"}:
        return obj["answer"] if isinstance(obj["answer"], dict) else obj
    if isinstance(obj, dict):
        obj.pop("log_url", None)
    return obj


async def call_llm(messages: list) -> dict:
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(
            LLM_ENDPOINT,
            headers={"Authorization": f"Bearer {LLM_API_KEY}"},
            json={"model": MODEL, "messages": messages, "tools": TOOLS,
                  "temperature": 0},
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]


async def run_agent(chat_id: int, question: str) -> str:
    """The agent loop: LLM -> tool -> LLM -> ... -> final JSON answer."""
    history = chat_histories.setdefault(
        chat_id, [{"role": "system", "content": SYSTEM_PROMPT}])
    history.append({"role": "user", "content": question})
    log({"event": "question", "chat_id": chat_id, "text": question})

    for _ in range(15):  # safety cap on loop iterations
        try:
            msg = await call_llm(history)
        except Exception as e:
            log({"event": "llm_error", "error": str(e)})
            return json.dumps({"answer": f"LLM call failed: {e}",
                               "log_url": LOG_URL})
        history.append(msg)

        if msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                name = tc["function"]["name"]
                args = json.loads(tc["function"]["arguments"])
                result = run_python(args["code"]) if name == "run_python" \
                    else fetch_url(args["url"])
                log({"event": "tool", "name": name, "args": args,
                     "result": result[:2000]})
                history.append({"role": "tool",
                                "tool_call_id": tc["id"], "content": result})
            continue  # let the model see the tool results

        # No tool calls -> this is the final answer.
        obj = _parse_final(msg.get("content"), question)
        log({"event": "final_answer", "answer": obj})
        return json.dumps(obj)

    # Out of rounds — force a final answer from what we have.
    log({"event": "loop_limit_reached", "chat_id": chat_id})
    history.append({"role": "user", "content":
        "You are out of tool budget. Using everything learned so far, reply "
        "NOW with only the final JSON object in the required shape."})
    try:
        msg = await call_llm(history)
        obj = _parse_final(msg.get("content"), question)
        log({"event": "final_answer_forced", "answer": obj})
        return json.dumps(obj)
    except Exception as e:
        log({"event": "llm_error", "error": str(e)})
        return json.dumps({"answer": "unable to determine",
                           "log_url": LOG_URL})


# ---------------------- Telegram plumbing ----------------------

async def send_telegram(chat_id: int, text: str):
    async with httpx.AsyncClient(timeout=30) as client:
        await client.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json={"chat_id": chat_id, "text": text})


async def answer_and_send(chat_id: int, text: str):
    """Runs in the background AFTER the webhook has been acknowledged."""
    try:
        reply = await run_agent(chat_id, text)
    except Exception:
        log({"event": "agent_crash", "trace": traceback.format_exc()[:3000]})
        reply = json.dumps({"answer": "internal error", "log_url": LOG_URL})
    await send_telegram(chat_id, reply)


@app.post("/webhook")
async def webhook(request: Request, background_tasks: BackgroundTasks):
    update = await request.json()
    update_id = update.get("update_id")
    if update_id in processed_updates:
        return {"ok": True}                    # duplicate delivery — ignore
    if update_id is not None:
        processed_updates.add(update_id)

    message = update.get("message") or {}
    text, chat = message.get("text"), message.get("chat", {})
    if text and chat.get("id"):
        background_tasks.add_task(answer_and_send, chat["id"], text)
    return {"ok": True}                        # instant ack — no retries
