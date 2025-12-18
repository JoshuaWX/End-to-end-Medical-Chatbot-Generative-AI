from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, render_template, request

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

embeddings = download_hugging_face_embeddings()


index_name = "medicalbot"

# Embed each chunk and upsert the embeddings into your Pinecone index.
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})


llm = ChatOpenAI(
    model="meta-llama/llama-3.3-70b-instruct:free",
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


def answer_question(query: str) -> str:
    docs = retrieve_docs(query)
    context = docs_to_context(docs)
    messages = prompt.format_messages(input=query, context=context)
    resp = llm.invoke(messages)
    return str(getattr(resp, "content", "") or "").strip()


@app.route("/")
def index():
    return render_template('chatbot.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    # Support both legacy form posts (msg=...) and JSON ({"msg": "..."}).
    msg = None

    if request.method == "GET":
        msg = request.args.get("msg")
    else:
        if request.is_json:
            payload = request.get_json(silent=True) or {}
            msg = payload.get("msg")
        if msg is None:
            msg = request.form.get("msg")

    if not msg or not isinstance(msg, str):
        return "Missing msg", 400

    msg = msg.strip()
    if not msg:
        return "Missing msg", 400

    try:
        return answer_question(msg)
    except Exception as e:
        # Keep error details server-side; return minimal info client-side.
        print("chat error:", repr(e))
        return "Chatbot error", 500




if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
