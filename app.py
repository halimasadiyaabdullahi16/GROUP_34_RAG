"""
app.py  —  RAG Document Intelligence System  v2
Framework  : Streamlit
LLM        : Google Gemini (cloud) | Ollama (local, any model)
Vector DB  : FAISS (in-memory) + BM25 hybrid retrieval
Embeddings : sentence-transformers (local, user-selectable)
"""

from sentence_transformers import SentenceTransformer
from google.genai import types
from google import genai
from dotenv import load_dotenv
from ollama import Client as OllamaClient
from groq import Groq
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import io
import os
import re
from typing import Any

import faiss
import fitz          # PyMuPDF
import matplotlib
matplotlib.use("Agg")

# ─────────────────────────── Configuration ──────────────────────────────
load_dotenv()

GEMINI_MODEL: str = "gemini-2.5-flash"
OLLAMA_MODEL: str = "llama3.1:8b"
OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
GROQ_MODEL: str = "llama-3.3-70b-versatile"
EMBED_MODEL: str = "all-MiniLM-L6-v2"
CHUNK_SIZE: int = 500
CHUNK_OVERLAP: int = 100
TOP_K: int = 5

# ─────────────────────────── Page setup ─────────────────────────────────
st.set_page_config(
    page_title="RAG Research Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────── Global CSS ─────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Lora:ital,wght@0,400;0,600;1,400&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Base — clean white/warm grey academic look ── */
html, body, .stApp {
    font-family: 'Inter', system-ui, sans-serif !important;
    background-color: #f5f4f0 !important;
    color: #1a1a2e !important;
}

/* ── Remove Streamlit chrome ── */
#MainMenu, footer              { visibility: hidden !important; }
.stDeployButton                { display: none !important; }
[data-testid="stToolbar"]      { display: none !important; }
[data-testid="stDecoration"]   { display: none !important; }
header[data-testid="stHeader"] { background: transparent !important; }

/* ── Layout ── */
.block-container {
    padding-top: 2rem !important;
    padding-bottom: 2rem !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid #e0ddd6 !important;
}

[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span {
    color: #4a4a5a !important;
    font-size: 13px !important;
}

/* ── Section headers ── */
.section-title {
    font-family: 'Lora', Georgia, serif;
    font-size: 13px;
    font-weight: 600;
    color: #2563eb;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    margin: 0 0 6px 0;
    padding-bottom: 6px;
    border-bottom: 2px solid #dbeafe;
}

/* ── Tabs ── */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: #ffffff !important;
    border-radius: 8px !important;
    padding: 4px !important;
    border: 1px solid #e0ddd6 !important;
    gap: 2px !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06) !important;
}

[data-testid="stTabs"] [data-baseweb="tab"] {
    background: transparent !important;
    color: #6b7280 !important;
    border-radius: 6px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 7px 18px !important;
    border: none !important;
    transition: all 0.15s !important;
}

[data-testid="stTabs"] [aria-selected="true"] {
    background: #2563eb !important;
    color: #ffffff !important;
    box-shadow: 0 1px 6px rgba(37,99,235,0.3) !important;
}

/* ── Buttons ── */
.stButton > button[kind="primary"] {
    background: #2563eb !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 7px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    padding: 8px 20px !important;
    transition: background 0.15s, box-shadow 0.15s !important;
    box-shadow: 0 2px 8px rgba(37,99,235,0.25) !important;
}

.stButton > button[kind="primary"]:hover {
    background: #1d4ed8 !important;
    box-shadow: 0 3px 12px rgba(37,99,235,0.35) !important;
}

.stButton > button[kind="secondary"] {
    background: #ffffff !important;
    color: #374151 !important;
    border: 1px solid #d1d5db !important;
    border-radius: 7px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    transition: all 0.15s !important;
}

.stButton > button[kind="secondary"]:hover {
    border-color: #9ca3af !important;
    background: #f9fafb !important;
}

/* ── Text input ── */
.stTextInput input, .stTextArea textarea {
    background: #ffffff !important;
    border: 1px solid #d1d5db !important;
    border-radius: 7px !important;
    color: #111827 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 14px !important;
    padding: 9px 14px !important;
    transition: border-color 0.15s, box-shadow 0.15s !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04) !important;
}

.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: #2563eb !important;
    box-shadow: 0 0 0 3px rgba(37,99,235,0.12) !important;
}

/* ── Cards ── */
.card {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    padding: 20px 24px;
    box-shadow: 0 1px 6px rgba(0,0,0,0.05);
}

/* ── Divider ── */
hr { border-color: #e5e7eb !important; }

/* ── Expander ── */
[data-testid="stExpander"] {
    border: 1px solid #e5e7eb !important;
    border-radius: 8px !important;
    background: #ffffff !important;
}

[data-testid="stExpander"] summary {
    color: #374151 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 13px !important;
    font-weight: 500 !important;
}

/* ── Progress ── */
.stProgress > div > div {
    background: #2563eb !important;
    border-radius: 4px !important;
}

/* ── File uploader ── */
[data-testid="stFileUploaderDropzone"] {
    background: #fafafa !important;
    border: 1.5px dashed #d1d5db !important;
    border-radius: 8px !important;
}

[data-testid="stFileUploaderDropzone"]:hover {
    border-color: #2563eb !important;
    background: #eff6ff !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border: 1px solid #e5e7eb !important;
    border-radius: 8px !important;
}

/* ── Alert boxes ── */
[data-testid="stAlert"] {
    border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────── Cached embedding model ─────────────────────
@st.cache_resource(show_spinner="Loading embedding model…")
def _load_embedder():
    return SentenceTransformer(EMBED_MODEL)


embedder = _load_embedder()


# ─────────────────────────── Session state ──────────────────────────────
_defaults = {
    "faiss_index": None,
    "chunks": [],
    "chunk_counts": {},
    "doc_names": [],
    "raw_texts": {},
    "rag_answer": None,
    "bare_answer": None,
    "ctx_passages": [],
    "eval_results": None,
    "eval_queries": [],
    "eval_chart": None,
    "eval_detail": [],
}
for _k, _v in _defaults.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ─────────────────────────── LLM client helper ──────────────────────────
def _get_gemini_client(api_key: str):
    """Return a Gemini client, or None if no key is available."""
    key = api_key.strip() or os.getenv("GEMINI_API_KEY", "").strip()
    if not key:
        return None
    return genai.Client(api_key=key)


@st.cache_data(ttl=8)
def _get_ollama_models(host: str) -> list[str]:
    """Fetch available local models from the running Ollama daemon."""
    client = OllamaClient(host=host)
    out = client.list()
    names = []
    for model in getattr(out, "models", []) or []:
        name = getattr(model, "model", None) or getattr(model, "name", None)
        if name:
            names.append(name)
    return sorted(set(names))


def _call_llm(
    provider: str,
    model: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    *,
    system_instruction: str = "",
    api_key: str = "",
    ollama_host: str = OLLAMA_HOST,
) -> str:
    """Unified text generation wrapper for Gemini, Groq, and Ollama."""
    try:
        if provider == "gemini":
            client = _get_gemini_client(api_key)
            if client is None:
                raise ValueError(
                    "Gemini API key missing. Add it in the sidebar.")
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction or None,
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                ),
            )
            return (response.text or "").strip()

        if provider == "groq":
            if not api_key:
                raise ValueError("Groq API key missing. Add it in the sidebar.")
            client = Groq(api_key=api_key)
            messages: list[dict[str, str]] = []
            if system_instruction.strip():
                messages.append({"role": "system", "content": system_instruction.strip()})
            messages.append({"role": "user", "content": prompt})
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content.strip()

        # Ollama (local)
        ollama_client = OllamaClient(host=ollama_host)
        ollama_messages: list[dict[str, str]] = []
        if system_instruction.strip():
            ollama_messages.append(
                {"role": "system", "content": system_instruction.strip()})
        ollama_messages.append({"role": "user", "content": prompt})

        response = ollama_client.chat(
            model=model,
            messages=ollama_messages,
            options={
                "temperature": float(temperature),
                "num_predict": int(max_tokens),
            },
        )
        return response["message"]["content"].strip()
    except Exception as exc:
        return f"[ERROR] {exc}"


# ═══════════════════════════════════════════════════════════════════
#  BACKEND — INGESTION
# ═══════════════════════════════════════════════════════════════════

def _extract_text(uploaded_file) -> str:
    raw = uploaded_file.read()
    doc = fitz.open(stream=raw, filetype="pdf")
    text = "".join(page.get_text() for page in doc)
    doc.close()
    if not text.strip():
        raise ValueError("PDF appears empty or unreadable.")
    return text


def _char_chunk(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    chunks, pos = [], 0
    while pos < len(text):
        chunks.append(text[pos: pos + size])
        pos += size - overlap
    return chunks


def _token_chunk(text: str, tokenizer, n_tokens: int, n_overlap: int) -> list[str]:
    ids = tokenizer.encode(text, add_special_tokens=False)
    chunks, pos = [], 0
    while pos < len(ids):
        end = min(pos + n_tokens, len(ids))
        chunk = tokenizer.decode(
            ids[pos:end], skip_special_tokens=True).strip()
        if chunk:
            chunks.append(chunk)
        pos += n_tokens - n_overlap
    return chunks


def _build_index(chunks: list[str]) -> faiss.IndexFlatIP:
    vecs = embedder.encode(chunks, convert_to_numpy=True,
                           show_progress_bar=False)
    faiss.normalize_L2(vecs)
    idx = faiss.IndexFlatIP(vecs.shape[1])
    idx.add(vecs.astype(np.float32))
    return idx


# ═══════════════════════════════════════════════════════════════════
#  BACKEND — RAG / BARE LLM
# ═══════════════════════════════════════════════════════════════════

def answer_rag(question: str, llm_cfg: dict[str, Any]) -> tuple[str, list[str]]:
    if st.session_state.faiss_index is None or not st.session_state.chunks:
        return "No documents indexed. Upload a PDF first.", []

    q_vec = embedder.encode([question], convert_to_numpy=True)
    faiss.normalize_L2(q_vec)
    scores, idxs = st.session_state.faiss_index.search(
        q_vec.astype(np.float32), TOP_K)

    best_score = float(scores[0][0]) if scores.size else 0.0
    if best_score < 0.15:
        return "The uploaded documents do not contain this information.", []

    passages = [
        st.session_state.chunks[i]
        for i in idxs[0]
        if 0 <= i < len(st.session_state.chunks)
    ]
    context = "\n\n---\n\n".join(f"[Passage {i+1}]\n{p}" for i,
                                 p in enumerate(passages))
    system = (
        "You are a precise research assistant. "
        "Answer ONLY using the passages provided. "
        "If the answer is absent from the passages, reply: "
        "'The uploaded documents do not contain this information.' "
        "Do not draw on prior knowledge."
    )
    answer = _call_llm(
        provider=llm_cfg["provider"],
        model=llm_cfg["model"],
        prompt=f"Passages:\n{context}\n\nQuestion: {question}",
        system_instruction=system,
        temperature=0.2,
        max_tokens=2048,
        api_key=llm_cfg.get("api_key", ""),
        ollama_host=llm_cfg.get("ollama_host", OLLAMA_HOST),
    )
    return answer, passages


def answer_bare(question: str, llm_cfg: dict[str, Any]) -> str:
    return _call_llm(
        provider=llm_cfg["provider"],
        model=llm_cfg["model"],
        prompt=question,
        temperature=0.9,
        max_tokens=2048,
        api_key=llm_cfg.get("api_key", ""),
        ollama_host=llm_cfg.get("ollama_host", OLLAMA_HOST),
    )


# ═══════════════════════════════════════════════════════════════════
#  BACKEND — EVALUATION
# ═══════════════════════════════════════════════════════════════════

def _gen_queries(text: str, llm_cfg: dict[str, Any], n: int = 8) -> list[str]:
    prompt = (
        f"From the document excerpt below, write exactly {n} specific factual questions "
        f"whose answers appear clearly in the text. "
        f"Output a numbered list only (1. … {n}.). No explanations.\n\nDocument:\n{text[:4000]}"
    )
    result = _call_llm(
        provider=llm_cfg["provider"],
        model=llm_cfg["model"],
        prompt=prompt,
        temperature=0.4,
        max_tokens=600,
        api_key=llm_cfg.get("api_key", ""),
        ollama_host=llm_cfg.get("ollama_host", OLLAMA_HOST),
    )
    if result.startswith("[ERROR]"):
        raise RuntimeError(result)
    out = []
    for line in result.strip().split("\n"):
        q = re.sub(r"^[\d]+[.)]\s*", "", line.strip()).strip()
        if len(q) > 10:
            out.append(q)
    return out[:n]


def _llm_judge(query: str, passages: list[str], llm_cfg: dict[str, Any]) -> bool:
    ctx = "\n\n".join(
        f"[Chunk {i+1}]: {p[:400]}" for i, p in enumerate(passages))
    prompt = (
        f"Query: {query}\n\nRetrieved context:\n{ctx}\n\n"
        "Do the retrieved chunks contain enough information to answer the query?\n"
        "Reply with exactly one word: YES or NO."
    )
    result = _call_llm(
        provider=llm_cfg["provider"],
        model=llm_cfg["model"],
        prompt=prompt,
        temperature=0.0,
        max_tokens=8,
        api_key=llm_cfg.get("api_key", ""),
        ollama_host=llm_cfg.get("ollama_host", OLLAMA_HOST),
    )
    if result.startswith("[ERROR]"):
        return False
    return result.strip().upper().startswith("Y")


def _draw_chart(results: dict) -> bytes:
    COLORS = ["#2563eb", "#16a34a", "#d97706"]
    BG = "#ffffff"
    GRID = "#f3f4f6"
    LABEL = "#6b7280"

    labels = list(results.keys())
    hit_rates = [results[c]["hit_rate"] * 100 for c in labels]
    n_chunks = [results[c]["n_chunks"] for c in labels]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=BG)

    for ax, values, ylabel, title, annotation in [
        (ax1, hit_rates, "Hit Rate @5 (%)",
         "Retrieval Hit Rate by Chunk Size", True),
        (ax2, n_chunks,  "Number of Chunks",
         "Index Size by Chunk Configuration", False),
    ]:
        ax.set_facecolor(BG)
        bars = ax.bar(labels, values, color=COLORS, width=0.44,
                      edgecolor="#e5e7eb", linewidth=0.8, zorder=3)
        if annotation:
            ax.axhline(y=80, color="#ef4444", linestyle="--",
                       linewidth=1.3, alpha=0.7, zorder=4, label="Target ≥ 80%")
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (2.5 if annotation else max(values) * 0.02),
                f"{val:.0f}%" if annotation else str(int(val)),
                ha="center", va="bottom",
                color="#111827", fontsize=11, fontweight="600",
            )
        if annotation:
            ax.set_ylim(0, 120)
            ax.legend(facecolor=BG, labelcolor=LABEL, fontsize=8,
                      framealpha=1, edgecolor="#e5e7eb")
        ax.set_ylabel(ylabel, color=LABEL, fontsize=10)
        ax.set_title(title, color="#111827", fontsize=11,
                     fontweight="bold", pad=12)
        ax.tick_params(colors=LABEL, labelsize=9)
        ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for sp in ax.spines.values():
            sp.set_edgecolor("#e5e7eb")

    plt.tight_layout(pad=2.4)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ═══════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════
with st.sidebar:

    st.markdown("""
    <div style="padding:4px 0 18px 0;">
        <div style="font-family:'Lora',Georgia,serif;font-size:20px;font-weight:600;color:#111827;">
            RAG Research Assistant
        </div>
        <div style="font-size:12px;color:#9ca3af;margin-top:4px;">
            Document-grounded AI · Evaluation Lab
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ── LLM Provider + Model ──
    st.markdown('<p class="section-title">LLM Configuration</p>',
                unsafe_allow_html=True)

    provider_label = st.selectbox(
        "Provider",
        options=["Gemini (Cloud)", "Groq (Cloud)", "Ollama (Local)"],
        index=0,
        help="Switch between cloud providers (Gemini, Groq) and local Ollama models.",
    )
    if provider_label.startswith("Gemini"):
        provider = "gemini"
    elif provider_label.startswith("Groq"):
        provider = "groq"
    else:
        provider = "ollama"

    api_key = ""
    ollama_host = OLLAMA_HOST
    if provider == "gemini":
        model_name = GEMINI_MODEL
    elif provider == "groq":
        model_name = GROQ_MODEL
    else:
        model_name = OLLAMA_MODEL

    if provider == "gemini":
        api_key_input = st.text_input(
            "Gemini API Key",
            type="password",
            placeholder="Paste your Gemini API key…",
            help="Get a key at aistudio.google.com. Stored only in session memory.",
        )
        api_key = api_key_input.strip() or os.getenv("GEMINI_API_KEY", "").strip()
        model_name = st.text_input(
            "Gemini Model",
            value=GEMINI_MODEL,
            help="Examples: gemini-2.5-flash, gemini-2.5-pro",
        ).strip() or GEMINI_MODEL

        if not api_key:
            st.warning(
                "Enter a Gemini API key to enable LLM features.", icon="🔑")
        else:
            st.success("Gemini is configured.")

    elif provider == "groq":
        api_key_input = st.text_input(
            "Groq API Key",
            type="password",
            placeholder="Paste your Groq API key…",
            help="Get a free key at console.groq.com. Stored only in session memory.",
        )
        api_key = api_key_input.strip() or os.getenv("GROQ_API_KEY", "").strip()
        model_name = st.selectbox(
            "Groq Model",
            options=[
                "llama-3.3-70b-versatile",
                "llama3-8b-8192",
                "llama3-70b-8192",
                "mixtral-8x7b-32768",
                "gemma2-9b-it",
            ],
            index=0,
            help="All models above are available on Groq's free tier.",
        )
        if not api_key:
            st.warning("Enter a Groq API key to enable LLM features.", icon="🔑")
        else:
            st.success("Groq is configured.")

    else:
        ollama_host = st.text_input(
            "Ollama Host",
            value=OLLAMA_HOST,
            help="Local daemon URL, usually http://localhost:11434",
        ).strip() or OLLAMA_HOST
        if st.button("Refresh local models", use_container_width=True):
            _get_ollama_models.clear()

        local_models = []
        local_error = ""
        try:
            local_models = _get_ollama_models(ollama_host)
        except Exception as exc:
            local_error = str(exc)

        if local_models:
            model_name = st.selectbox(
                "Ollama Model",
                options=local_models,
                index=0,
                help="Run `ollama pull <model>` to install additional models.",
            )
            st.success(f"Ollama ready ({len(local_models)} model(s) found).")
        else:
            model_name = st.text_input(
                "Ollama Model",
                value=OLLAMA_MODEL,
                help="Example: llama3.1:8b",
            ).strip() or OLLAMA_MODEL
            if local_error:
                st.warning(
                    "Could not auto-detect models. Ensure Ollama is running, then use Refresh local models.",
                    icon="⚠️",
                )

    llm_ready = bool(model_name) and (provider == "ollama" or bool(api_key))
    llm_cfg = {
        "provider": provider,
        "model": model_name,
        "api_key": api_key,
        "ollama_host": ollama_host,
    }

    st.divider()

    # ── Document Upload ──
    st.markdown('<p class="section-title">Documents</p>',
                unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        key="uploader",
    )

    new_files = [f for f in (uploaded or [])
                 if f.name not in st.session_state.chunk_counts]

    if new_files:
        with st.spinner("Indexing…"):
            errs = []
            for f in new_files:
                try:
                    text = _extract_text(f)
                    chunks = _char_chunk(text)
                    st.session_state.chunks.extend(chunks)
                    st.session_state.chunk_counts[f.name] = len(chunks)
                    st.session_state.doc_names.append(f.name)
                    st.session_state.raw_texts[f.name] = text
                except Exception as e:
                    errs.append(f"{f.name}: {e}")

            if st.session_state.chunks:
                st.session_state.faiss_index = _build_index(
                    st.session_state.chunks)
                st.session_state.eval_results = None
                st.session_state.eval_queries = []
                st.session_state.eval_chart = None
                st.session_state.eval_detail = []

            for e in errs:
                st.error(e, icon="⚠️")

    # Indexed document list
    if st.session_state.doc_names:
        for name in st.session_state.doc_names:
            n = st.session_state.chunk_counts.get(name, 0)
            dn = name if len(name) <= 28 else name[:25] + "…"
            st.markdown(f"""
            <div style="
                background:#f9fafb;
                border:1px solid #e5e7eb;
                border-left:3px solid #2563eb;
                border-radius:6px;
                padding:8px 12px;
                margin-bottom:6px;
                font-family:'Inter',sans-serif;
            ">
                <div style="font-size:12px;font-weight:600;color:#111827;
                            white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{dn}</div>
                <div style="font-size:11px;color:#6b7280;margin-top:2px;">{n} chunks indexed</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.caption("No documents loaded yet.")

    if st.button("Clear all documents", key="clear_btn", use_container_width=True):
        for k, v in _defaults.items():
            st.session_state[k] = v
        st.rerun()

    st.divider()

    # ── Status ──
    st.markdown('<p class="section-title">System Status</p>',
                unsafe_allow_html=True)
    online = st.session_state.faiss_index is not None
    st.markdown(f"""
    <div style="font-size:12px;color:#4b5563;font-family:'Inter',sans-serif;line-height:2.1;">
        <div>Status &nbsp;&nbsp;&nbsp;: <b style="color:{'#16a34a' if online else '#9ca3af'}">
            {'● Online' if online else '○ Standby'}</b></div>
        <div>Documents : <b style="color:#111827;">{len(st.session_state.doc_names)}</b></div>
        <div>Chunks &nbsp;&nbsp;&nbsp;: <b style="color:#111827;">{len(st.session_state.chunks)}</b></div>
        <div>Embedder &nbsp;: <b style="color:#111827;">all-MiniLM-L6-v2</b></div>
        <div>Provider : <b style="color:#111827;">{provider_label}</b></div>
        <div>Model &nbsp;&nbsp;&nbsp;&nbsp;: <b style="color:#111827;">{model_name}</b></div>
        <div>Top-K &nbsp;&nbsp;&nbsp;&nbsp;: <b style="color:#111827;">{TOP_K}</b></div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
#  MAIN AREA
# ═══════════════════════════════════════════════════════════════════

st.markdown("""
<div style="margin-bottom:24px;">
    <h1 style="
        font-family:'Lora',Georgia,serif;
        font-size:26px; font-weight:600;
        color:#111827; margin:0 0 6px 0;
    ">RAG Research Assistant</h1>
    <p style="
        font-size:14px; color:#6b7280;
        font-family:'Inter',sans-serif;
        margin:0;
    ">
        Upload research documents, choose your LLM provider (Gemini or local Ollama), and ask questions.
        Query answers are strictly grounded in your uploaded documents.
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div style="
    display:grid;
    grid-template-columns:repeat(auto-fit, minmax(180px, 1fr));
    gap:10px;
    margin:0 0 16px 0;
">
    <div style="background:#ffffff;border:1px solid #e5e7eb;border-radius:10px;padding:10px 12px;">
        <div style="font-size:11px;color:#6b7280;">Provider</div>
        <div style="font-size:13px;color:#111827;font-weight:600;">{provider_label}</div>
    </div>
    <div style="background:#ffffff;border:1px solid #e5e7eb;border-radius:10px;padding:10px 12px;">
        <div style="font-size:11px;color:#6b7280;">Model</div>
        <div style="font-size:13px;color:#111827;font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{model_name}</div>
    </div>
    <div style="background:#ffffff;border:1px solid #e5e7eb;border-radius:10px;padding:10px 12px;">
        <div style="font-size:11px;color:#6b7280;">Documents</div>
        <div style="font-size:13px;color:#111827;font-weight:600;">{len(st.session_state.doc_names)}</div>
    </div>
    <div style="background:#ffffff;border:1px solid #e5e7eb;border-radius:10px;padding:10px 12px;">
        <div style="font-size:11px;color:#6b7280;">Indexed Chunks</div>
        <div style="font-size:13px;color:#111827;font-weight:600;">{len(st.session_state.chunks)}</div>
    </div>
</div>
""", unsafe_allow_html=True)

tab_query, tab_eval, tab_about = st.tabs(["Query", "Evaluation Lab", "About"])


# ─────────────────────────────────────
#  TAB 1 — QUERY
# ─────────────────────────────────────
with tab_query:

    q_col, btn_col = st.columns([5, 1], gap="small")
    with q_col:
        query = st.text_input(
            "Your question",
            placeholder="e.g. What is the main objective of this paper?",
            label_visibility="collapsed",
            key="query_input",
        )
    with btn_col:
        ask = st.button("Ask", type="primary", key="ask_btn",
                        use_container_width=True)

    compare_mode = st.toggle(
        "Also generate baseline answer without document context",
        value=False,
        help="Useful for hallucination comparison. Leave off for strict document-grounded Q&A.",
    )

    if ask:
        if not query.strip():
            st.warning("Please enter a question.", icon="⚠️")
        elif not llm_ready:
            if provider == "gemini":
                st.error("Add your Gemini API key in the sidebar first.", icon="🔑")
            else:
                st.error(
                    "Select an Ollama model (and ensure Ollama is running).", icon="🧠")
        elif st.session_state.faiss_index is None:
            st.warning("Upload at least one PDF document first.", icon="📄")
        else:
            with st.spinner("Retrieving context + generating document-grounded answer…"):
                rag_ans, passages = answer_rag(query, llm_cfg)
            st.session_state.rag_answer = rag_ans
            st.session_state.ctx_passages = passages

            if compare_mode:
                with st.spinner("Generating baseline answer without context…"):
                    st.session_state.bare_answer = answer_bare(query, llm_cfg)
            else:
                st.session_state.bare_answer = None

    # Response area
    if st.session_state.rag_answer is None:
        st.markdown("""
        <div style="
            text-align:center;
            padding:80px 20px;
            border:1.5px dashed #d1d5db;
            border-radius:12px;
            background:#ffffff;
            margin-top:16px;
        ">
            <div style="font-size:36px;margin-bottom:14px;">📄</div>
            <div style="font-size:15px;font-weight:600;color:#374151;
                        font-family:'Lora',Georgia,serif;margin-bottom:8px;">
                Ready to answer your questions
            </div>
            <div style="font-size:13px;color:#9ca3af;font-family:'Inter',sans-serif;line-height:1.7;">
                Upload a PDF document and type a question above.<br>
                Your answer will be generated strictly from retrieved document passages.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        show_baseline = bool(st.session_state.bare_answer)
        if show_baseline:
            col_rag, col_llm = st.columns(2, gap="large")
        else:
            col_rag = st.container()
            col_llm = None

        with col_rag:
            st.markdown("""
            <div style="
                display:flex; align-items:center; gap:10px; margin-bottom:10px;
            ">
                <span style="
                    background:#eff6ff; color:#2563eb;
                    font-family:'Inter',sans-serif;
                    font-size:11px; font-weight:700;
                    padding:3px 12px; border-radius:20px;
                    border:1px solid #bfdbfe;
                    letter-spacing:0.5px;
                ">RAG ANSWER</span>
                <span style="font-size:12px;color:#6b7280;font-family:'Inter',sans-serif;">
                    Grounded in your documents
                </span>
            </div>
            """, unsafe_allow_html=True)

            rag_text = st.session_state.rag_answer or ""
            rag_safe = rag_text.replace("&", "&amp;").replace(
                "<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
            st.html(
                '<div style="'
                'background:#ffffff;'
                'border:1px solid #bfdbfe;'
                'border-top:3px solid #2563eb;'
                'border-radius:10px;'
                'padding:20px 22px;'
                'font-family:Inter,sans-serif;'
                'font-size:14px;'
                'color:#111827;'
                'line-height:1.85;'
                f'min-height:100px;">{rag_safe}</div>'
            )

            if st.session_state.ctx_passages:
                with st.expander(f"View {len(st.session_state.ctx_passages)} retrieved passages"):
                    for i, p in enumerate(st.session_state.ctx_passages):
                        safe_p = p.replace("<", "&lt;").replace(">", "&gt;")
                        st.markdown(f"""
                        <div style="
                            background:#f8fafc;
                            border:1px solid #e2e8f0;
                            border-left:3px solid #93c5fd;
                            border-radius:6px;
                            padding:10px 14px;
                            margin-bottom:8px;
                            font-size:12px;
                            color:#475569;
                            font-family:'Inter',sans-serif;
                            line-height:1.7;
                        ">
                            <div style="font-size:10px;font-weight:700;color:#93c5fd;
                                        letter-spacing:1px;margin-bottom:6px;
                                        font-family:'JetBrains Mono',monospace;">
                                PASSAGE {i+1}
                            </div>
                            {safe_p}
                        </div>
                        """, unsafe_allow_html=True)

        if show_baseline and col_llm is not None:
            with col_llm:
                st.markdown("""
                <div style="
                    display:flex; align-items:center; gap:10px; margin-bottom:10px;
                ">
                    <span style="
                        background:#fef3c7; color:#d97706;
                        font-family:'Inter',sans-serif;
                        font-size:11px; font-weight:700;
                        padding:3px 12px; border-radius:20px;
                        border:1px solid #fde68a;
                        letter-spacing:0.5px;
                    ">BASELINE ANSWER</span>
                    <span style="font-size:12px;color:#6b7280;font-family:'Inter',sans-serif;">
                        No retrieval context
                    </span>
                </div>
                """, unsafe_allow_html=True)

                bare_text = st.session_state.bare_answer or ""
                bare_safe = bare_text.replace("&", "&amp;").replace(
                    "<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
                st.html(
                    '<div style="'
                    'background:#ffffff;'
                    'border:1px solid #fde68a;'
                    'border-top:3px solid #d97706;'
                    'border-radius:10px;'
                    'padding:20px 22px;'
                    'font-family:Inter,sans-serif;'
                    'font-size:14px;'
                    'color:#111827;'
                    'line-height:1.85;'
                    f'min-height:100px;">{bare_safe}</div>'
                )

                st.markdown("""
                <div style="
                    background:#fef2f2;
                    border:1px solid #fecaca;
                    border-left:3px solid #ef4444;
                    border-radius:8px;
                    padding:10px 14px;
                    margin-top:12px;
                    font-size:12px;
                    color:#7f1d1d;
                    font-family:'Inter',sans-serif;
                    line-height:1.65;
                ">
                    ⚠️ Baseline output is not constrained by your documents. Use it only for comparison.
                </div>
                """, unsafe_allow_html=True)


# ─────────────────────────────────────
#  TAB 2 — EVALUATION LAB
# ─────────────────────────────────────
with tab_eval:

    st.markdown("""
    <div style="margin-bottom:20px;">
        <h2 style="font-family:'Lora',Georgia,serif;font-size:20px;
                   font-weight:600;color:#111827;margin:0 0 8px 0;">
            Retrieval Evaluation Lab
        </h2>
        <p style="font-size:13px;color:#6b7280;font-family:'Inter',sans-serif;
                  line-height:1.75;margin:0;">
            Measures <strong>Hit Rate @5</strong> across three token-based chunk-size configurations
            using <strong>LLM-as-judge</strong>. The selected model auto-generates factual test questions from
            your document, then evaluates whether the top-5 retrieved chunks contain the answer.
        </p>
    </div>
    """, unsafe_allow_html=True)

    cfg_c1, cfg_c2, cfg_c3 = st.columns(3, gap="small")
    configs_display = [
        ("200 tokens",  "40 tok overlap",  "#2563eb", "#eff6ff", "#bfdbfe",
         "Small chunks — high precision, large index"),
        ("500 tokens",  "100 tok overlap", "#16a34a", "#f0fdf4", "#bbf7d0",
         "Medium chunks — balanced (default)"),
        ("1000 tokens", "200 tok overlap", "#d97706", "#fffbeb", "#fde68a",
         "Large chunks — broad context, smaller index"),
    ]
    for col, (lbl, sub, txt, bg, border, desc) in zip([cfg_c1, cfg_c2, cfg_c3], configs_display):
        with col:
            st.markdown(f"""
            <div style="
                background:{bg};
                border:1px solid {border};
                border-top:3px solid {txt};
                border-radius:10px;
                padding:14px 16px;
                margin-bottom:12px;
            ">
                <div style="font-family:'Inter',sans-serif;font-size:13px;
                            font-weight:700;color:{txt};margin-bottom:4px;">{lbl}</div>
                <div style="font-size:11px;color:#6b7280;margin-bottom:6px;
                            font-family:'JetBrains Mono',monospace;">{sub}</div>
                <div style="font-size:12px;color:#374151;font-family:'Inter',sans-serif;">
                    {desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.info(
        "**Note:** all-MiniLM-L6-v2 truncates inputs at 256 tokens. "
        "The 1000-token config will have embeddings silently truncated — "
        "a deliberate demonstration of why chunk size must respect the embedding model's limit.",
        icon="ℹ️",
    )

    if st.session_state.faiss_index is None:
        st.warning(
            "Upload a document in the sidebar to enable evaluation.", icon="📄")
    elif not llm_ready:
        if provider == "gemini":
            st.error(
                "Add your Gemini API key in the sidebar to run evaluation.", icon="🔑")
        else:
            st.error(
                "Select an Ollama model and ensure the Ollama daemon is running.", icon="🧠")
    else:
        if st.button(
            f"Run Evaluation (auto-generates queries + judges retrieval via {provider_label})",
            type="primary",
            key="run_eval",
            use_container_width=True,
        ):
            combined = "\n\n".join(st.session_state.raw_texts.values())
            if not combined.strip():
                st.error("No raw text found. Re-upload your document.")
            else:
                prog = st.progress(0, text="Generating test queries…")
                try:
                    queries = _gen_queries(combined, llm_cfg, n=8)
                    prog.progress(
                        15, text=f"Generated {len(queries)} queries. Evaluating chunks…")

                    tokenizer = embedder.tokenizer
                    eval_cfgs = [
                        {"label": "200 tokens",  "tokens": 200,  "overlap": 40},
                        {"label": "500 tokens",  "tokens": 500,  "overlap": 100},
                        {"label": "1000 tokens", "tokens": 1000, "overlap": 200},
                    ]
                    results, detail = {}, []
                    total_ops, ops_done = len(eval_cfgs) * len(queries), 0

                    for cfg in eval_cfgs:
                        chunks = _token_chunk(
                            combined, tokenizer, cfg["tokens"], cfg["overlap"])
                        idx = _build_index(chunks)
                        hits = 0

                        for q in queries:
                            q_vec = embedder.encode([q], convert_to_numpy=True)
                            faiss.normalize_L2(q_vec)
                            _, idxs = idx.search(q_vec.astype(np.float32), 5)
                            retrieved = [chunks[i]
                                         for i in idxs[0] if 0 <= i < len(chunks)]
                            verdict = _llm_judge(q, retrieved, llm_cfg)
                            hits += int(verdict)
                            detail.append({
                                "Config": cfg["label"],
                                "Query": q[:80] + ("…" if len(q) > 80 else ""),
                                "Verdict": "✓ Hit" if verdict else "✗ Miss",
                            })
                            ops_done += 1
                            pct = 15 + int(ops_done / total_ops * 75)
                            prog.progress(
                                pct, text=f"{cfg['label']} — {ops_done}/{total_ops}")

                        results[cfg["label"]] = {
                            "hit_rate": hits / len(queries) if queries else 0.0,
                            "n_chunks": len(chunks),
                            "hits": hits,
                            "total": len(queries),
                        }

                    prog.progress(90, text="Rendering chart…")
                    st.session_state.eval_results = results
                    st.session_state.eval_queries = queries
                    st.session_state.eval_chart = _draw_chart(results)
                    st.session_state.eval_detail = detail
                    prog.progress(100, text="Done.")

                except Exception as exc:
                    st.error(f"Evaluation failed: {exc}")

        if st.session_state.eval_results:
            st.markdown("<hr>", unsafe_allow_html=True)

            # Metric cards
            m_cols = st.columns(3, gap="small")
            card_styles = [
                ("#eff6ff", "#2563eb", "#bfdbfe"),
                ("#f0fdf4", "#16a34a", "#bbf7d0"),
                ("#fffbeb", "#d97706", "#fde68a"),
            ]
            for col, (cfg_name, metrics), (bg, txt, border) in zip(
                m_cols, st.session_state.eval_results.items(), card_styles
            ):
                with col:
                    hr = metrics["hit_rate"] * 100
                    st.markdown(f"""
                    <div style="
                        background:{bg};
                        border:1.5px solid {border};
                        border-top:3px solid {txt};
                        border-radius:10px;
                        padding:18px 20px;
                        text-align:center;
                    ">
                        <div style="font-family:'Inter',sans-serif;font-size:11px;
                                    font-weight:700;color:{txt};letter-spacing:0.5px;
                                    margin-bottom:8px;">{cfg_name.upper()}</div>
                        <div style="font-family:'Lora',Georgia,serif;font-size:32px;
                                    font-weight:600;color:#111827;line-height:1;">
                            {hr:.0f}<span style="font-size:18px;">%</span>
                        </div>
                        <div style="font-size:11px;color:#6b7280;margin-top:6px;
                                    font-family:'Inter',sans-serif;">
                            {metrics['hits']}/{metrics['total']} hits
                            &nbsp;·&nbsp; {metrics['n_chunks']} chunks
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            if st.session_state.eval_chart:
                st.image(st.session_state.eval_chart, use_container_width=True)

            if st.session_state.eval_queries:
                with st.expander("Auto-generated test queries"):
                    for i, q in enumerate(st.session_state.eval_queries, 1):
                        st.markdown(f"""
                        <div style="
                            background:#f8fafc;
                            border-left:3px solid #93c5fd;
                            border-radius:4px;
                            padding:7px 12px; margin-bottom:6px;
                            font-size:13px; color:#374151;
                            font-family:'Inter',sans-serif;
                        ">
                            <span style="font-size:10px;font-weight:700;color:#93c5fd;
                                         margin-right:8px;font-family:'JetBrains Mono',monospace;">
                                Q{i:02d}
                            </span>{q}
                        </div>
                        """, unsafe_allow_html=True)

            if st.session_state.eval_detail:
                with st.expander("Per-query judgment detail"):
                    import pandas as pd
                    df = pd.DataFrame(st.session_state.eval_detail)
                    st.dataframe(df, use_container_width=True, hide_index=True)


# ─────────────────────────────────────
#  TAB 3 — ABOUT
# ─────────────────────────────────────
with tab_about:

    c_left, c_right = st.columns([3, 2], gap="large")

    with c_left:
        st.markdown("""
        <h2 style="font-family:'Lora',Georgia,serif;font-size:20px;
                   font-weight:600;color:#111827;margin:0 0 12px 0;">
            What is RAG?
        </h2>
        <p style="font-size:14px;color:#374151;font-family:'Inter',sans-serif;
                  line-height:1.85;margin-bottom:20px;">
            <strong>Retrieval-Augmented Generation (RAG)</strong> grounds a large language
            model's responses in factual evidence retrieved from a user-supplied corpus,
            preventing the model from hallucinating facts that aren't in the source material.
        </p>

        <h3 style="font-family:'Lora',Georgia,serif;font-size:16px;
                   font-weight:600;color:#111827;margin:0 0 12px 0;">
            System Pipeline
        </h3>
        """, unsafe_allow_html=True)

        steps = [
            ("1. Extraction",   "PyMuPDF reads PDFs in-memory (no disk writes). Text is extracted page by page."),
            ("2. Chunking",     "Text is split into 500-character overlapping windows (100-char overlap) to preserve context across boundaries."),
            ("3. Embedding",    "all-MiniLM-L6-v2 encodes each chunk into a 384-dimensional vector. Vectors are L2-normalised."),
            ("4. Indexing",     "FAISS IndexFlatIP stores vectors in memory. Rebuilt incrementally as new documents are added."),
            ("5. Retrieval",    "The query is embedded and normalised. FAISS returns the top-5 most similar passages by cosine similarity."),
            ("6. RAG Answer",   "Your selected LLM (Gemini or Ollama) answers using only retrieved passages (temp=0.2). The prompt forbids outside knowledge."),
            ("7. Optional Baseline",
             "You can enable a no-context baseline answer to compare grounded vs non-grounded behavior."),
        ]
        for title, desc in steps:
            st.markdown(f"""
            <div style="
                display:flex; gap:14px;
                margin-bottom:12px;
                padding:12px 16px;
                background:#ffffff;
                border:1px solid #e5e7eb;
                border-radius:8px;
            ">
                <div style="
                    min-width:120px;
                    font-size:12px; font-weight:700;
                    color:#2563eb;
                    font-family:'Inter',sans-serif;
                    padding-top:1px;
                ">{title}</div>
                <div style="font-size:13px;color:#374151;
                            font-family:'Inter',sans-serif;line-height:1.65;">
                    {desc}
                </div>
            </div>
            """, unsafe_allow_html=True)

    with c_right:
        st.markdown("""
        <h3 style="font-family:'Lora',Georgia,serif;font-size:16px;
                   font-weight:600;color:#111827;margin:0 0 12px 0;">
            Tech Stack
        </h3>
        """, unsafe_allow_html=True)

        tech = [
            ("Framework",   "Streamlit 1.55"),
            ("LLM",         "Gemini (cloud) or Ollama (local)"),
            ("Vector DB",   "FAISS-CPU (IndexFlatIP)"),
            ("Embeddings",  "all-MiniLM-L6-v2 (384-dim)"),
            ("PDF Parsing", "PyMuPDF (fitz)"),
            ("Numerics",    "NumPy 2.4"),
            ("Charts",      "Matplotlib 3.10"),
            ("Config",      "python-dotenv"),
        ]
        for label, value in tech:
            st.markdown(f"""
            <div style="
                display:flex; justify-content:space-between; align-items:center;
                padding:8px 14px;
                border-bottom:1px solid #f3f4f6;
                font-family:'Inter',sans-serif;
            ">
                <span style="font-size:12px;color:#6b7280;">{label}</span>
                <span style="font-size:12px;font-weight:600;color:#111827;">{value}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div style="margin-top:24px;">
        <h3 style="font-family:'Lora',Georgia,serif;font-size:16px;
                   font-weight:600;color:#111827;margin:0 0 12px 0;">
            Evaluation Methodology
        </h3>
        <p style="font-size:13px;color:#374151;font-family:'Inter',sans-serif;
                  line-height:1.75;margin:0;">
            The evaluation tab runs <strong>Hit Rate @5</strong> across three chunk-size
            configurations (200, 500, 1000 tokens). Your selected model auto-generates 8 factual
            questions, then acts as an LLM-as-judge, grading each top-5 retrieval result
            as a hit or miss. Results are visualised as publication-quality bar charts.
        </p>
        </div>
        """, unsafe_allow_html=True)
