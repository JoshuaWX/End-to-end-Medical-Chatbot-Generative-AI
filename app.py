from __future__ import annotations

import os
import re
import time
import traceback
import uuid
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request, session

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

# Session configuration for conversation memory
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "medguard-secret-key-change-in-production")
app.config["SESSION_TYPE"] = "filesystem"
app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(hours=24)

# Server-side conversation memory store (keyed by session_id)
# In production, replace with Redis or a database
conversation_memory: dict[str, list[dict[str, str]]] = defaultdict(list)
memory_timestamps: dict[str, datetime] = {}
MAX_MEMORY_TURNS = 50  # Maximum turns to store per session
MEMORY_EXPIRY_HOURS = 24  # Clear old sessions after this many hours

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


# ============================================================================
# INTENT RECOGNITION SYSTEM
# ============================================================================

# Define intents with keywords and patterns
INTENT_PATTERNS = {
    "symptom_report": {
        "keywords": [
            "symptom", "symptoms", "feeling", "feel", "pain", "ache", "hurt",
            "sore", "fever", "cough", "headache", "nausea", "dizzy", "tired",
            "fatigue", "swelling", "rash", "itching", "burning", "numbness",
            "vomiting", "diarrhea", "constipation", "bleeding", "weak"
        ],
        "patterns": [
            r"i have (a |an )?",
            r"i('m| am) (feeling|having|experiencing)",
            r"my .+ (hurt|ache|pain|sore)",
            r"(started|been) (feeling|having)",
        ],
        "response_template": "symptom_analysis"
    },
    "medication_inquiry": {
        "keywords": [
            "medication", "medicine", "drug", "pill", "tablet", "dose", "dosage",
            "prescription", "otc", "over the counter", "antibiotic", "painkiller",
            "side effect", "interaction", "take", "taking"
        ],
        "patterns": [
            r"what (medication|medicine|drug)",
            r"can i take",
            r"should i take",
            r"(is|are) .+ safe",
            r"side effects? of",
        ],
        "response_template": "medication_info"
    },
    "condition_inquiry": {
        "keywords": [
            "disease", "condition", "disorder", "syndrome", "diagnosis", "diagnose",
            "what is", "what are", "explain", "tell me about", "cause", "causes",
            "treatment", "cure", "chronic", "acute"
        ],
        "patterns": [
            r"what (is|are|causes)",
            r"tell me (about|more)",
            r"how (is|are) .+ (treated|diagnosed)",
            r"can .+ be cured",
        ],
        "response_template": "condition_info"
    },
    "emergency_check": {
        "keywords": [
            "emergency", "urgent", "hospital", "911", "ambulance", "er",
            "emergency room", "serious", "severe", "dangerous", "life threatening"
        ],
        "patterns": [
            r"should i (go to|call|visit)",
            r"is (this|it) (serious|an emergency|dangerous)",
            r"when (should|do) i (see|call|go)",
        ],
        "response_template": "emergency_guidance"
    },
    "lifestyle_advice": {
        "keywords": [
            "diet", "exercise", "sleep", "stress", "weight", "nutrition",
            "healthy", "lifestyle", "prevent", "prevention", "avoid", "reduce",
            "improve", "better", "tips", "advice", "recommend"
        ],
        "patterns": [
            r"how (can|do) i (prevent|improve|reduce)",
            r"what (should|can) i (eat|do|avoid)",
            r"(tips|advice) (for|on|about)",
        ],
        "response_template": "lifestyle_guidance"
    },
    "greeting": {
        "keywords": [
            "hello", "hi", "hey", "good morning", "good afternoon", "good evening",
            "howdy", "greetings"
        ],
        "patterns": [
            r"^(hi|hello|hey|howdy)[\s!.,?]*$",
        ],
        "response_template": "greeting"
    },
    "gratitude": {
        "keywords": [
            "thank", "thanks", "appreciate", "helpful", "great"
        ],
        "patterns": [
            r"^(thank|thanks)",
            r"(that|this) (was|is) helpful",
        ],
        "response_template": "gratitude"
    },
    "followup": {
        "keywords": [
            "more", "else", "also", "another", "what about", "how about",
            "and", "additionally", "furthermore"
        ],
        "patterns": [
            r"^(what|how) about",
            r"^and (what|how|if)",
            r"anything else",
            r"tell me more",
        ],
        "response_template": "followup"
    }
}

# Intent-specific system prompts for better responses
# Emojis: use sparingly (0-2 per response), only when contextually appropriate, never at the start
INTENT_PROMPTS = {
    "symptom_analysis": """You are analyzing symptoms. Ask clarifying questions if needed:
- When did symptoms start?
- Severity on a scale of 1-10?
- Any triggers or patterns?
- Other accompanying symptoms?
Provide possible explanations but always recommend professional consultation for diagnosis.""",
    
    "medication_info": """You are providing medication information. Include:
- General uses and how it works
- Common side effects
- Important interactions or warnings
- ALWAYS recommend consulting a doctor or pharmacist before starting/stopping medications.
You may use a ⚠️ for important warnings if appropriate.""",
    
    "condition_info": """You are explaining a medical condition. Cover:
- What the condition is
- Common causes and risk factors
- Typical symptoms
- General treatment approaches
- When to seek medical care""",
    
    "emergency_guidance": """You are helping assess urgency. Provide clear guidance on:
- Signs that require immediate emergency care
- When to see a doctor soon vs. wait
- What to do while waiting for care
- Local emergency numbers (remind them of 911 in US)
Use ⚠️ or 🚨 only for genuinely urgent warnings.""",
    
    "lifestyle_guidance": """You are providing health and lifestyle advice. Include:
- Evidence-based recommendations
- Practical, actionable tips
- Gradual changes over drastic measures
- Importance of consistency""",
    
    "greeting": """The user is greeting you. Respond warmly and invite them to share their health concerns. Keep it natural without emojis.""",
    
    "gratitude": """The user is expressing thanks. Acknowledge it briefly and offer further assistance. A simple 😊 at the end is fine.""",
    
    "followup": """The user wants more information on a previous topic. Reference the conversation history to provide relevant follow-up information.""",
    
    "general": """You are a helpful medical assistant. Answer the question accurately and concisely."""
}


def classify_intent(query: str, history: list[dict[str, str]] | None = None) -> tuple[str, float]:
    """
    Classify the intent of a user query using keyword matching and regex patterns.
    Returns (intent_name, confidence_score).
    """
    query_lower = query.lower().strip()
    scores: dict[str, float] = defaultdict(float)
    
    for intent_name, intent_data in INTENT_PATTERNS.items():
        keywords = intent_data.get("keywords", [])
        patterns = intent_data.get("patterns", [])
        
        # Keyword matching
        keyword_matches = sum(1 for kw in keywords if kw in query_lower)
        if keywords:
            keyword_score = keyword_matches / len(keywords)
            scores[intent_name] += keyword_score * 0.6  # Keywords worth 60%
        
        # Pattern matching
        pattern_matches = sum(1 for p in patterns if re.search(p, query_lower))
        if patterns:
            pattern_score = min(pattern_matches / len(patterns), 1.0)
            scores[intent_name] += pattern_score * 0.4  # Patterns worth 40%
    
    # Boost followup intent if history exists and query is short/contextual
    if history and len(query_lower.split()) < 5:
        scores["followup"] += 0.2
    
    if not scores:
        return "general", 0.0
    
    best_intent = max(scores, key=lambda k: scores[k])
    confidence = min(scores[best_intent], 1.0)
    
    # If confidence is too low, default to general
    if confidence < 0.15:
        return "general", confidence
    
    return best_intent, confidence


def get_intent_enhanced_prompt(intent: str, base_prompt: str) -> str:
    """Enhance the system prompt based on detected intent."""
    intent_addition = INTENT_PROMPTS.get(intent, INTENT_PROMPTS["general"])
    return f"{base_prompt}\n\nAdditional guidance for this query:\n{intent_addition}"


# ============================================================================
# CONVERSATION MEMORY MANAGEMENT
# ============================================================================

def get_or_create_session_id() -> str:
    """Get existing session ID or create a new one."""
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())
        session.permanent = True
    return session["session_id"]


def cleanup_old_sessions():
    """Remove expired sessions from memory (call periodically)."""
    now = datetime.now()
    expired = [
        sid for sid, ts in memory_timestamps.items()
        if now - ts > timedelta(hours=MEMORY_EXPIRY_HOURS)
    ]
    for sid in expired:
        conversation_memory.pop(sid, None)
        memory_timestamps.pop(sid, None)


def get_session_history(session_id: str) -> list[dict[str, str]]:
    """Retrieve conversation history for a session."""
    return conversation_memory.get(session_id, [])


def add_to_session_history(session_id: str, role: str, content: str):
    """Add a message to session history with proper management."""
    memory_timestamps[session_id] = datetime.now()
    
    history = conversation_memory[session_id]
    history.append({"role": role, "content": content, "timestamp": datetime.now().isoformat()})
    
    # Trim if exceeding max turns (each turn = user + assistant)
    if len(history) > MAX_MEMORY_TURNS * 2:
        conversation_memory[session_id] = history[-(MAX_MEMORY_TURNS * 2):]


def clear_session_history(session_id: str):
    """Clear conversation history for a session."""
    conversation_memory.pop(session_id, None)
    memory_timestamps.pop(session_id, None)


def format_history_for_llm(history: list[dict[str, str]], max_tokens_estimate: int = 2000) -> str:
    """
    Format conversation history for LLM context with token management.
    Prioritizes recent messages and summarizes older ones if needed.
    """
    if not history:
        return ""
    
    # Estimate ~4 chars per token
    char_limit = max_tokens_estimate * 4
    
    formatted_lines: list[str] = []
    total_chars = 0
    
    # Process from most recent to oldest
    for turn in reversed(history):
        role = turn.get("role", "unknown").capitalize()
        content = (turn.get("content") or "").strip()
        if not content:
            continue
        
        line = f"{role}: {content}"
        line_len = len(line)
        
        if total_chars + line_len > char_limit:
            # Add summary indicator and stop
            remaining = len(history) - len(formatted_lines)
            if remaining > 0:
                formatted_lines.append(f"[... {remaining} earlier messages omitted for context limit ...]")
            break
        
        formatted_lines.append(line)
        total_chars += line_len
    
    # Reverse back to chronological order
    formatted_lines.reverse()
    return "\n".join(formatted_lines)

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
            "Please try in 24 hours or upgrade your plan."
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


def answer_question_with_context(
    query: str,
    *,
    meta: dict[str, Any] | None = None,
    history: Any = None,
    session_id: str | None = None
) -> tuple[str, dict[str, Any]]:
    """
    Answer a question with full context including:
    - User metadata
    - Conversation history (from client + server session)
    - Retrieved documents
    - Intent-specific prompting
    
    Returns (answer_text, metadata_dict) where metadata includes intent info.
    """
    # Merge client history with server-side session history
    client_hist = normalize_history(history)
    server_hist = get_session_history(session_id) if session_id else []
    
    # Use server history if available and client history is empty/shorter
    if server_hist and len(server_hist) > len(client_hist):
        combined_hist = server_hist
    else:
        combined_hist = client_hist
    
    # Classify intent for better response routing
    intent, confidence = classify_intent(query, combined_hist)
    
    # Retrieve relevant documents
    retrieved = ""
    try:
        docs = retrieve_docs(query)
        retrieved = docs_to_context(docs)
    except Exception:
        retrieved = ""

    user_ctx = build_user_context(meta)
    hist_text = format_history_for_llm(combined_hist)

    # Build context parts
    context_parts: list[str] = []
    if user_ctx:
        context_parts.append("User context:\n" + user_ctx)
    if hist_text:
        context_parts.append("Conversation history:\n" + hist_text)
    if retrieved:
        context_parts.append("Retrieved medical knowledge:\n" + retrieved)

    context = "\n\n---\n\n".join(context_parts)
    
    # Get intent-enhanced system prompt
    enhanced_system_prompt = get_intent_enhanced_prompt(
        INTENT_PATTERNS.get(intent, {}).get("response_template", "general"),
        system_prompt
    )
    
    # Create prompt with enhanced system message
    intent_prompt = ChatPromptTemplate.from_messages([
        ("system", enhanced_system_prompt),
        ("human", "{input}"),
    ])
    
    messages = intent_prompt.format_messages(input=query, context=context)
    resp = invoke_llm_with_retry(messages)
    answer = str(getattr(resp, "content", "") or "").strip()
    
    # Return answer with metadata for logging/debugging
    response_meta = {
        "intent": intent,
        "intent_confidence": round(confidence, 3),
        "history_turns": len(combined_hist) // 2,
    }
    
    return answer, response_meta


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
            "If you’re in the U.S., you can call or text 112 for immediate help."
        )
        if request.is_json:
            return jsonify({"answer": urgent})
        return urgent

    # Get or create session for persistent memory
    session_id = get_or_create_session_id()
    
    # Periodically cleanup old sessions (simple approach)
    if len(conversation_memory) > 100:
        cleanup_old_sessions()

    try:
        # Store user message in server-side memory
        add_to_session_history(session_id, "user", msg)
        
        # Get answer with intent recognition
        answer, response_meta = answer_question_with_context(
            msg,
            meta=meta,
            history=history,
            session_id=session_id
        )
        
        # Store assistant response in server-side memory
        add_to_session_history(session_id, "assistant", answer)
        
        if request.is_json:
            return jsonify({
                "answer": answer,
                "ok": True,
                "intent": response_meta.get("intent"),
                "intent_confidence": response_meta.get("intent_confidence"),
                "session_id": session_id,
            })
        return answer
    except Exception as e:
        # Keep error details server-side; return minimal info client-side.
        print("chat error:", repr(e))
        print(traceback.format_exc())
        if request.is_json:
            # Return 200 so the frontend doesn't log noisy 500s for expected provider failures.
            if _looks_like_rate_limit_error(e):
                friendly = _friendly_provider_message(e)
                return jsonify({"answer": friendly, "ok": False, "limited": True})

            friendly = _friendly_provider_message(e)
            return jsonify({"answer": friendly, "ok": False})
        return "Chatbot error", 500


@app.route("/clear-history", methods=["POST"])
def clear_history():
    """Endpoint to clear conversation history for the current session."""
    session_id = session.get("session_id")
    if session_id:
        clear_session_history(session_id)
    return jsonify({"ok": True, "message": "Conversation history cleared."})


@app.route("/session-info", methods=["GET"])
def session_info():
    """Debug endpoint to see current session status."""
    session_id = session.get("session_id")
    history_count = len(get_session_history(session_id)) if session_id else 0
    return jsonify({
        "session_id": session_id,
        "history_messages": history_count,
        "active_sessions": len(conversation_memory),
    })


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
