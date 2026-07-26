"""
Data-Analyst Telegram Bot — v2.
Fixes over v1:
  - Webhook now acknowledges Telegram INSTANTLY and runs the agent in the
    background (prevents Telegram retry -> duplicate replies).
  - Duplicate update_ids are ignored (belt-and-braces against retries).
  - Agent loop cap raised to 15.
  - Stronger system prompt: the model must verify with tools, not guess.

Environment variables you must set on your host:
  TELEGRAM_TOKEN  - from @BotFather
  LLM_API_KEY     - your LLM provider key
  PUBLIC_URL      - your deployed base URL, e.g. https://tdsp1q5.onrender.com

Run locally:  python -m uvicorn bot:app --port 8000
Set webhook:  https://api.telegram.org/bot<TOKEN>/setWebhook?url=<PUBLIC_URL>/webhook
"""

import os, io, json, contextlib, traceback
import httpx
from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import FileResponse

TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
LLM_API_KEY = os.environ["LLM_API_KEY"]
PUBLIC_URL = os.environ.get("PUBLIC_URL", "http://localhost:8000")
LLM_ENDPOINT = "https://aipipe.org/openai/v1/chat/completions"
MODEL = "gpt-4o-mini"
LOG_PATH = "run.jsonl"
LOG_URL = f"{PUBLIC_URL}/run.jsonl"

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


# ---------------------- The agent's tools ----------------------

def run_python(code: str) -> str:
    """Execute Python and return stdout (or the traceback)."""
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            exec(code, {"__name__": "__main__"})
        return buf.getvalue()[:8000] or "(no output — use print())"
    except Exception:
        return "ERROR:\n" + traceback.format_exc()[:4000]


def fetch_url(url: str) -> str:
    """Download a page/dataset so the model can inspect it."""
    try:
        r = httpx.get(url, timeout=30, follow_redirects=True)
        return r.text[:8000]
    except Exception as e:
        return f"ERROR: {e}"


TOOLS = [
    {"type": "function", "function": {
        "name": "run_python",
        "description": "Run Python code (pandas/httpx available). Print results.",
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
1. NEVER answer from memory alone. You MUST ground every answer in actual
   data: if the question embeds data, analyze it with run_python; if it
   refers to a public dataset (MOSPI, data.gov.in, etc.), use fetch_url
   and/or run_python (httpx + pandas) to retrieve and analyze real data
   before answering. If retrieval fails after genuine attempts, answer from
   your best knowledge but only as a last resort.
2. Work step by step: locate the data, load it, compute, verify the result
   makes sense, then answer.
3. In multi-turn conversations, answer the LATEST message, using earlier
   messages as context.
4. Your FINAL reply must be EXACTLY one JSON object matching the shape the
   question asks for, e.g. {"answer": {...}} — no prose, no markdown fences,
   no explanations around it.
5. Do not invent the log_url; the server fills it in."""


async def call_llm(messages: list) -> dict:
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(
            LLM_ENDPOINT,
            headers={"Authorization": f"Bearer {LLM_API_KEY}"},
            json={"model": MODEL, "messages": messages, "tools": TOOLS},
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
        raw = (msg.get("content") or "").strip()
        raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            obj = json.loads(raw)
            if "answer" not in obj:            # tolerate {"state": ...} etc.
                obj = {"answer": obj}
        except json.JSONDecodeError:
            obj = {"answer": raw}              # last-resort fallback
        obj["log_url"] = LOG_URL               # server owns this field
        log({"event": "final_answer", "answer": obj})
        return json.dumps(obj)

    log({"event": "loop_limit_reached", "chat_id": chat_id})
    return json.dumps({"answer": "agent loop limit reached", "log_url": LOG_URL})


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
