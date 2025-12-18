from __future__ import annotations

import os
import time
import traceback
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore

from src.helper import download_hugging_face_embeddings
from src.prompt import system_prompt


def required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is missing. Set it in your .env file or environment.")
    return value


# Load .env from this folder (End-to-end-Medical-Chatbot-Generative-AI)
load_dotenv()

# Serve the MedGuard compiled CSS from the main project ./dist folder.
# This makes templates/chatbot.html able to load /dist/output.css.
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # .../MEDGUARD
DIST_DIR = PROJECT_ROOT / "dist"

app = Flask(
    __name__,
    template_folder="templates",
    static_folder=str(DIST_DIR),
    static_url_path="/dist",
)

PINECONE_API_KEY = required_env("PINECONE_API_KEY")
OPENROUTER_API_KEY = required_env("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.environ.get(
    "OPENROUTER_MODEL",
    "meta-llama/llama-3.3-70b-instruct:free",
).strip() or "meta-llama/llama-3.3-70b-instruct:free"

embeddings = download_hugging_face_embeddings()


index_name = "medicalbot"

# Embed each chunk and upsert the embeddings into your Pinecone index.
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})


llm = ChatOpenAI(
    model=OPENROUTER_MODEL,
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    temperature=0.7,
    max_tokens=500,
    default_headers={
        "HTTP-Referer": "http://localhost",  # set to your site later if you deploy
        "X-Title": "MEDGUARD",
    },
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)


def docs_to_context(docs) -> str:
    if not docs:
        return ""
    parts = []
    for d in docs:
        try:
            content = getattr(d, "page_content", None) or ""
        except Exception:
            content = ""
        content = str(content).strip()
        if content:
            parts.append(content)
    return "\n\n---\n\n".join(parts)


def retrieve_docs(query: str, k: int = 3):
    # Compatible across LangChain versions.
    try:
        # Newer LC: retriever.invoke(query)
        return retriever.invoke(query)
    except Exception:
        # Older LC: retriever.get_relevant_documents(query)
        try:
            return retriever.get_relevant_documents(query)
        except Exception:
            return []


def _looks_like_transient_provider_error(err: Exception) -> bool:
    msg = str(err)
    return (
        "Internal Server Error" in msg
        or "'code': 500" in msg
        or "\"code\": 500" in msg
        or "HTTP 500" in msg
    )


def _looks_like_rate_limit_error(err: Exception) -> bool:
    # OpenRouter via OpenAI-compatible clients often raises RateLimitError (429)
    # but we keep this string-based to avoid tight coupling.
    msg = str(err)
    tname = type(err).__name__
    return (
        tname.lower().endswith("ratelimiterror")
        or "rate limit exceeded" in msg.lower()
        or "'code': 429" in msg
        or "\"code\": 429" in msg
        or "Error code: 429" in msg
    )


def _friendly_provider_message(err: Exception) -> str:
    if _looks_like_rate_limit_error(err):
        return (
            "I can’t answer , you have used up your free plan limit for the day. "
            "Please try again later,You've hit the free plan limit"
        )
    if _looks_like_transient_provider_error(err):
        return "I encountered an Issue. Please try again in a moment."
    return "Sorry — I ran into a problem while answering. Please try again in a moment."


def retrieval_only_fallback(query: str, *, meta: dict[str, Any] | None = None, history: Any = None) -> str:
    """Fallback when the LLM is unavailable (e.g., rate-limited).

    Returns a helpful, non-LLM response based on retrieved context + a small checklist.
    """
    user_ctx = build_user_context(meta)
    hist = normalize_history(history)
    hist_text = history_to_text(hist)

    docs = []
    try:
        docs = retrieve_docs(query)
    except Exception:
        docs = []

    snippets: list[str] = []
    for d in docs[:3]:
        content = str(getattr(d, "page_content", "") or "").strip()
        if not content:
            continue
        if len(content) > 420:
            content = content[:420].rstrip() + "…"
        snippets.append(content)

    lines: list[str] = []
    lines.append(
        "I’m temporarily unable to generate a full AI answer (provider rate limit). "
        "But I can still show relevant information from the knowledge base:"
    )
    if user_ctx:
        lines.append("\nUser context:\n" + user_ctx)
    if hist_text:
        # Keep history short here.
        short_hist = "\n".join(hist_text.splitlines()[-6:])
        lines.append("\nRecent conversation:\n" + short_hist)

    if snippets:
        lines.append("\nRelevant excerpts:\n- " + "\n- ".join(snippets))
    else:
        lines.append("\nI couldn't retrieve relevant excerpts for that question.")

    lines.append(
        "\nWhat you can do now:\n"
        "1) Share your main symptom(s), when they started, severity (1–10), and any fever/shortness of breath/chest pain.\n"
        "2) If symptoms are severe or worsening, seek urgent medical care."
    )

    return "\n".join(lines).strip()


def invoke_llm_with_retry(messages, attempts: int = 2, base_delay_s: float = 0.6):
    last_err: Exception | None = None
    for i in range(max(1, attempts)):
        try:
            return llm.invoke(messages)
        except Exception as e:
            last_err = e
            if i >= attempts - 1:
                break
            if not _looks_like_transient_provider_error(e):
                break
            time.sleep(base_delay_s * (i + 1))
    assert last_err is not None
    raise last_err


def answer_question(query: str) -> str:
    docs = retrieve_docs(query)
    context = docs_to_context(docs)
    messages = prompt.format_messages(input=query, context=context)
    resp = llm.invoke(messages)
    return str(getattr(resp, "content", "") or "").strip()


def build_user_context(meta: dict[str, Any] | None) -> str:
    if not meta:
        return ""

    parts: list[str] = []

    first_name = str(meta.get("first_name") or "").strip()
    if first_name:
        parts.append(f"User first name: {first_name}.")

    location = str(meta.get("location") or "").strip()
    if location:
        parts.append(f"User location/state: {location}.")

    age = str(meta.get("age") or "").strip()
    if age:
        parts.append(f"User age: {age}.")

    gender = str(meta.get("gender") or "").strip()
    if gender:
        parts.append(f"User gender: {gender}.")

    return "\n".join(parts).strip()


def normalize_history(history: Any, limit_turns: int = 10) -> list[dict[str, str]]:
    if not isinstance(history, list):
        return []

    normalized: list[dict[str, str]] = []
    for item in history:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "").strip().lower()
        content = str(item.get("content") or "").strip()
        if role not in ("user", "assistant"):
            continue
        if not content:
            continue
        # Basic size guard
        if len(content) > 2000:
            content = content[:2000]
        normalized.append({"role": role, "content": content})

    if len(normalized) > limit_turns * 2:
        normalized = normalized[-(limit_turns * 2):]
    return normalized


def history_to_text(history: list[dict[str, str]]) -> str:
    if not history:
        return ""
    lines: list[str] = []
    for turn in history:
        role = turn.get("role")
        content = (turn.get("content") or "").strip()
        if not content:
            continue
        if role == "user":
            lines.append(f"User: {content}")
        elif role == "assistant":
            lines.append(f"Assistant: {content}")
    return "\n".join(lines).strip()


def is_high_risk_message(text: str) -> bool:
    t = text.lower()
    # Minimal "seek urgent care" triggers; keep intentionally conservative.
    triggers = [
        "chest pain",
        "pressure in my chest",
        "can't breathe",
        "cannot breathe",
        "trouble breathing",
        "shortness of breath",
        "suicidal",
        "kill myself",
        "overdose",
        "seizure",
        "stroke",
        "face drooping",
        "slurred speech",
        "severe bleeding",
    ]
    return any(x in t for x in triggers)


def answer_question_with_context(query: str, *, meta: dict[str, Any] | None = None, history: Any = None) -> str:
    retrieved = ""
    try:
        docs = retrieve_docs(query)
        retrieved = docs_to_context(docs)
    except Exception:
        # If retrieval fails, still attempt to answer from conversation + user context.
        retrieved = ""

    user_ctx = build_user_context(meta)
    hist = normalize_history(history)
    hist_text = history_to_text(hist)

    context_parts: list[str] = []
    if user_ctx:
        context_parts.append("User context:\n" + user_ctx)
    if hist_text:
        context_parts.append("Conversation so far:\n" + hist_text)
    if retrieved:
        context_parts.append("Retrieved medical context:\n" + retrieved)

    context = "\n\n---\n\n".join(context_parts)
    messages = prompt.format_messages(input=query, context=context)
    resp = invoke_llm_with_retry(messages)
    return str(getattr(resp, "content", "") or "").strip()


@app.route("/")
def index():
    # Optional personalization forwarded from MedGuard/chatbot.html
    meta = {
        "first_name": (request.args.get("first_name") or "").strip(),
        "location": (request.args.get("location") or "").strip(),
        "age": (request.args.get("age") or "").strip(),
        "gender": (request.args.get("gender") or "").strip(),
    }

    display_name = meta.get("first_name") or "there"
    greeting = (
        f"Hi {display_name}! I’m your MedGuard AI Health Assistant. "
        "Describe your symptoms (when they started, severity, and anything that makes them better/worse)."
    )
    return render_template("chatbot.html", meta=meta, greeting=greeting)


@app.get("/favicon.ico")
def favicon():
    # Avoid noisy 404s in the browser console.
    return ("", 204)


@app.route("/get", methods=["GET", "POST"])
def chat():
    # Support both legacy form posts (msg=...) and JSON ({"msg": "..."}).
    msg = None
    meta: dict[str, Any] | None = None
    history: Any = None

    if request.method == "GET":
        msg = request.args.get("msg")
    else:
        if request.is_json:
            payload = request.get_json(silent=True) or {}
            msg = payload.get("msg")
            meta = payload.get("meta") if isinstance(payload, dict) else None
            history = payload.get("history") if isinstance(payload, dict) else None
        if msg is None:
            msg = request.form.get("msg")

    if not msg or not isinstance(msg, str):
        if request.is_json:
            return jsonify({"answer": "Please type a message so I can help."}), 400
        return "Missing msg", 400

    msg = msg.strip()
    if not msg:
        if request.is_json:
            return jsonify({"answer": "Please type a message so I can help."}), 400
        return "Missing msg", 400

    if is_high_risk_message(msg):
        urgent = (
            "I’m not able to help safely with that as a chatbot. "
            "If you think this could be an emergency, call your local emergency number now or go to the nearest emergency department. "
            "If you’re in the U.S., you can call or text 988 for immediate help."
        )
        if request.is_json:
            return jsonify({"answer": urgent})
        return urgent

    try:
        answer = answer_question_with_context(msg, meta=meta, history=history)
        if request.is_json:
            return jsonify({"answer": answer, "ok": True})
        return answer
    except Exception as e:
        # Keep error details server-side; return minimal info client-side.
        print("chat error:", repr(e))
        print(traceback.format_exc())
        if request.is_json:
            # Return 200 so the frontend doesn't log noisy 500s for expected provider failures.
            if _looks_like_rate_limit_error(e):
                answer = retrieval_only_fallback(msg, meta=meta, history=history)
                return jsonify({"answer": answer, "ok": False, "limited": True})

            friendly = _friendly_provider_message(e)
            return jsonify({"answer": friendly, "ok": False})
        return "Chatbot error", 500




if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
