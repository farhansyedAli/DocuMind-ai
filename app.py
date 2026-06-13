import os
import re
import time
import tempfile
import streamlit as st

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import (
    PyPDFLoader, Docx2txtLoader, TextLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

# ── page config ────────────────────────────────────────────────
st.set_page_config(
    page_title="DocuMind AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── session state defaults ─────────────────────────────────────
DEFAULTS = {
    "chat_history":   [],
    "messages":       [],
    "vectorstore":    None,
    "embeddings":     None,        # cached embeddings model (loaded once)
    "dark_mode":      True,
    "show_settings":  False,
    "api_key":        "",
    "model_name":     "gemini-2.0-flash-lite",
    "max_chunks":     3,
    "files_processed": False,
    "file_names":     [],
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── theme ──────────────────────────────────────────────────────
dark = st.session_state.dark_mode
theme = {
    "bg":        "#0a0a0f" if dark else "#f4f3ee",
    "surface":   "#13131a" if dark else "#ffffff",
    "surface2":  "#1c1c27" if dark else "#f0efe9",
    "border":    "#2a2a3d" if dark else "#e0ddd4",
    "accent":    "#7c6aff",
    "accent2":   "#a78bfa",
    "text":      "#e8e6ff" if dark else "#1a1a2e",
    "text2":     "#8b89a8" if dark else "#6b6880",
    "ai_bubble": "#13131a" if dark else "#ffffff",
    "danger":    "#ff6b6b",
    "success":   "#6bffb8",
    "warning":   "#ffd166",
}

# ── global CSS ─────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

html, body, .stApp {{
    background: {theme['bg']} !important;
    font-family: 'DM Sans', sans-serif;
    color: {theme['text']};
}}

#MainMenu, footer, header, .stDeployButton {{ display: none !important; }}
.block-container {{ padding-top: 1.5rem !important; max-width: 1200px !important; padding-bottom: 7rem !important; }}

/* Style the sidebar to match the dark theme */
section[data-testid="stSidebar"] {{
    background: {theme['surface']} !important;
    border-right: 1px solid {theme['border']} !important;
}}
section[data-testid="stSidebar"] > div {{
    background: {theme['surface']} !important;
    padding-top: 1rem !important;
}}
/* Hide the default Streamlit sidebar collapse arrow — the Settings
   button in the top bar is the only way to toggle the panel. */
[data-testid="stSidebarCollapseButton"] {{
    display: none !important;
}}

/* Style st.chat_input so it docks at the bottom looking clean */
[data-testid="stChatInput"] {{
    background: {theme['bg']} !important;
    border-top: 1px solid {theme['border']} !important;
}}
[data-testid="stBottomBlockContainer"] {{
    background: {theme['bg']} !important;
}}

::-webkit-scrollbar {{ width: 6px; }}
::-webkit-scrollbar-track {{ background: transparent; }}
::-webkit-scrollbar-thumb {{ background: {theme['border']}; border-radius: 4px; }}

.brand-title {{
    font-family: 'Syne', sans-serif;
    font-size: 22px; font-weight: 700;
    color: {theme['text']};
    letter-spacing: -0.3px;
    display: flex; align-items: center; gap: 12px;
}}
.brand-title .logo {{
    width: 36px; height: 36px; border-radius: 10px;
    background: linear-gradient(135deg, {theme['accent']}, {theme['accent2']});
    display: flex; align-items: center; justify-content: center;
    font-size: 18px;
    box-shadow: 0 4px 18px {theme['accent']}44;
}}
.brand-title .label-purple {{ color: {theme['accent2']}; }}

.welcome-wrap {{
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    min-height: 50vh;
    text-align: center; padding: 40px 20px;
}}
.welcome-icon {{
    width: 72px; height: 72px; border-radius: 20px;
    background: linear-gradient(135deg, {theme['accent']}, {theme['accent2']});
    display: flex; align-items: center; justify-content: center;
    font-size: 32px; margin-bottom: 20px;
    box-shadow: 0 8px 32px {theme['accent']}55;
}}
.welcome-title {{
    font-family: 'Syne', sans-serif;
    font-size: 28px; font-weight: 700;
    color: {theme['text']}; margin-bottom: 10px;
    letter-spacing: -0.5px;
}}
.welcome-sub {{
    font-size: 14px; color: {theme['text2']};
    line-height: 1.6; max-width: 420px;
    margin-bottom: 28px;
}}
.suggestion-grid {{
    display: grid; grid-template-columns: 1fr 1fr;
    gap: 10px; max-width: 520px; width: 100%;
    margin: 0 auto;
}}
.suggestion-card {{
    background: {theme['surface']};
    border: 1px solid {theme['border']};
    border-radius: 12px; padding: 14px 16px;
    font-size: 13px; color: {theme['text2']};
    text-align: left; line-height: 1.45;
    transition: all 0.2s;
}}
.suggestion-card:hover {{
    border-color: {theme['accent']};
    color: {theme['text']};
    background: {theme['accent']}11;
}}

[data-testid="stChatMessage"] {{
    background: transparent !important;
    border: none !important;
    padding: 8px 0 !important;
}}
[data-testid="stChatMessageContent"] {{
    background: {theme['surface']} !important;
    border: 1px solid {theme['border']} !important;
    border-radius: 14px !important;
    padding: 14px 18px !important;
    color: {theme['text']} !important;
}}

.file-chip {{
    display: flex; align-items: center; gap: 8px;
    background: {theme['surface2']};
    border: 1px solid {theme['border']};
    border-radius: 8px; padding: 8px 12px;
    margin-bottom: 6px; font-size: 12px;
    color: {theme['text']};
}}
.status-ready {{
    display: flex; align-items: center; gap: 8px;
    background: {'#6bffb822' if dark else '#d1fae5'};
    border: 1px solid {theme['success']}44;
    border-radius: 8px; padding: 10px 14px;
    font-size: 13px; color: {theme['success']};
    margin-top: 8px;
}}

.stChatInput textarea,
.stTextInput > div > div > input {{
    background: {theme['surface']} !important;
    border: 1px solid {theme['border']} !important;
    border-radius: 14px !important;
    color: {theme['text']} !important;
    font-family: 'DM Sans', sans-serif !important;
}}
.stChatInput textarea:focus,
.stTextInput > div > div > input:focus {{
    border-color: {theme['accent']} !important;
    box-shadow: 0 0 0 3px {theme['accent']}22 !important;
}}

.stButton > button {{
    background: {theme['surface2']} !important;
    border: 1px solid {theme['border']} !important;
    color: {theme['text']} !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 13px !important;
}}
.stButton > button:hover {{
    background: {theme['accent']}22 !important;
    border-color: {theme['accent']} !important;
}}
div[data-testid="stButton"] button[kind="primary"] {{
    background: linear-gradient(135deg, {theme['accent']}, {theme['accent2']}) !important;
    border: none !important;
    color: white !important;
    font-weight: 500 !important;
}}

.stSelectbox > div > div {{
    background: {theme['surface2']} !important;
    border: 1px solid {theme['border']} !important;
    border-radius: 10px !important;
    color: {theme['text']} !important;
}}
.stFileUploader {{
    background: {theme['surface2']} !important;
    border: 1px dashed {theme['border']} !important;
    border-radius: 12px !important;
}}
.stProgress > div > div {{ background: {theme['accent']} !important; }}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
#  Dynamic sidebar visibility — controlled by the Settings button.
#  Streamlit doesn't give us a Python API to programmatically open
#  or close the sidebar, so we inject one of two CSS rules on every
#  rerun depending on show_settings.
# ──────────────────────────────────────────────────────────────
if st.session_state.show_settings:
    st.markdown("""
        <style>
            section[data-testid="stSidebar"] {
                transform: translateX(0) !important;
                visibility: visible !important;
                min-width: 320px !important;
                max-width: 360px !important;
            }
        </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
        <style>
            section[data-testid="stSidebar"] {
                transform: translateX(-100%) !important;
                visibility: hidden !important;
                min-width: 0 !important;
                max-width: 0 !important;
            }
        </style>
    """, unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
#  HELPERS
# ──────────────────────────────────────────────────────────────

def get_file_icon(name):
    ext = os.path.splitext(name)[-1].lower().lstrip(".")
    return {"pdf": "📕", "docx": "📘", "txt": "📄"}.get(ext, "📎")


def load_file(path, name):
    ext = os.path.splitext(name)[-1].lower()
    if ext == ".pdf":
        loader = PyPDFLoader(path)
    elif ext == ".docx":
        loader = Docx2txtLoader(path)
    elif ext == ".txt":
        loader = TextLoader(path, encoding="utf-8")
    else:
        return []
    return loader.load()


def friendly_error(e):
    """Map raw exceptions to user-friendly text + the actual error for debugging."""
    raw = str(e)
    msg = raw.lower()
    # Always include the raw error tail so the user can see WHICH quota tripped
    # (per-minute vs per-day, free-tier vs paid). Without this, "rate limited"
    # is a black box and you can't tell if waiting will help.
    tail = f"\n\n<details><summary>Technical details</summary>\n\n```\n{raw[:600]}\n```\n</details>"

    if "429" in msg or "quota" in msg or "resource_exhausted" in msg or "rate limit" in msg:
        retry = re.search(r"retry.{0,15}?(\d+)\s*s", msg) or re.search(r"retry in (\d+)", msg)
        wait = retry.group(1) if retry else None
        wait_line = f" Suggested wait: **{wait} seconds**." if wait else ""
        return (
            f"⏳ **Gemini's free tier rejected this request.**{wait_line}\n\n"
            "Most common causes:\n"
            "1. Daily request quota exhausted (free tier is ~200/day per project)\n"
            "2. Per-minute limit hit (free tier is ~15 RPM on most Gemini models)\n"
            "3. The project is in 'ghost 429' lockout — try a different model in Settings, "
            "or create a brand-new Google Cloud project (not just a new key)."
            + tail
        )
    if "401" in msg or "403" in msg or "api key not valid" in msg or "invalid api key" in msg or "permission denied" in msg:
        return "🔑 Invalid Gemini API key. Please check it in Settings." + tail
    if "404" in msg or "model not found" in msg or "not supported" in msg:
        return "⚠️ Model not found. Try a different Gemini model in Settings." + tail
    if "network" in msg or "connection" in msg or "timeout" in msg:
        return "🌐 Network issue. Please check your internet connection." + tail
    if "500" in msg or "internal" in msg or "unavailable" in msg or "503" in msg:
        return "⚠️ Google's servers had a hiccup. Please try again in a moment." + tail
    return f"⚠️ Something went wrong." + tail


@st.cache_resource(show_spinner=False)
def load_embeddings_model():
    """
    Load the local embeddings model once, cached across reruns.

    Uses FastEmbed (ONNX Runtime under the hood) — no PyTorch dependency,
    no DLL conflicts on Windows, ~20 MB model download instead of ~90 MB,
    and faster on CPU than sentence-transformers.

    The model name is FastEmbed's identifier for the same all-MiniLM-L6-v2
    model — same 384-dim embeddings, same quality, different runtime.
    """
    return FastEmbedEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        max_length=512,
    )


def call_gemini_with_retry(chain_input, chain, max_attempts=2):
    """
    Stream a Gemini chain with retry ONLY on transient server errors.

    Important: we do NOT retry on 429 / quota errors. Retrying a quota
    failure burns more requests against the same daily allowance and
    deepens the per-project rate-limit lockout. If we hit a 429, we
    surface it immediately and let the user see it.
    """
    last_err = None
    for attempt in range(max_attempts):
        try:
            yield from chain.stream(chain_input)
            return
        except Exception as e:
            last_err = e
            msg = str(e).lower()
            # Never retry on rate limit, auth, or model-not-found errors.
            is_quota = ("429" in msg or "quota" in msg or "resource_exhausted" in msg
                        or "rate limit" in msg)
            is_auth  = ("401" in msg or "403" in msg or "api key" in msg
                        or "permission" in msg)
            is_404   = "404" in msg or "not found" in msg
            # Only retry on transient server/network errors.
            retryable = (not is_quota and not is_auth and not is_404
                         and any(t in msg for t in ["500", "internal", "503",
                                                     "unavailable", "timeout",
                                                     "deadline"]))
            if not retryable or attempt == max_attempts - 1:
                raise
            time.sleep(2 ** attempt)   # 1s, then 2s
    if last_err:
        raise last_err
    if last_err:
        raise last_err


# ──────────────────────────────────────────────────────────────
#  HEADER
# ──────────────────────────────────────────────────────────────

col_nav1, _, col_nav3 = st.columns([4, 1, 7])

with col_nav1:
    st.markdown(
        f"""
        <div class="brand-title">
          <div class="logo">🧠</div>
          <span>Docu<span class="label-purple">Mind</span> AI</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_nav3:
    nav1, nav2, nav3 = st.columns(3)
    with nav1:
        if st.button("⚙️ Settings", key="settings_toggle", use_container_width=True,
                     help="Open the settings panel (API key, model, upload documents)."):
            st.session_state.show_settings = not st.session_state.show_settings
            st.rerun()
    with nav2:
        if st.button("🧹 Clear chat", key="clear_chat", use_container_width=True,
                     disabled=not st.session_state.messages,
                     help="Clear the chat history but keep your loaded documents."):
            st.session_state.messages = []
            st.session_state.chat_history = []
            st.rerun()
    with nav3:
        if st.button("🔄 Reset all", key="reset_all", use_container_width=True,
                     disabled=not st.session_state.files_processed,
                     help="Remove all loaded documents and start fresh."):
            st.session_state.messages = []
            st.session_state.chat_history = []
            st.session_state.vectorstore = None
            st.session_state.files_processed = False
            st.session_state.file_names = []
            st.rerun()

st.markdown(
    f"<div style='height:1px;background:{theme['border']};margin:14px 0 18px;'></div>",
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────
#  SETTINGS — lives in the real Streamlit sidebar so the chat
#  input can dock to the bottom of the page (Streamlit only
#  docks st.chat_input when it's a top-level element, not
#  inside a column).
# ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        f"<p style='font-family:Syne,sans-serif;font-size:13px;font-weight:700;"
        f"color:{theme['text']};letter-spacing:0.5px;margin:8px 0 18px'>⚙️ Settings</p>",
        unsafe_allow_html=True,
    )

    # API Key
    st.markdown(
        f"<p style='font-size:11px;color:{theme['text2']};text-transform:uppercase;"
        f"letter-spacing:0.8px;font-weight:500;margin-bottom:4px'>Gemini API Key</p>",
        unsafe_allow_html=True,
    )
    api_input = st.text_input(
        "api", value=st.session_state.api_key,
        placeholder="AIza...", type="password",
        label_visibility="collapsed",
        help="Used only for answering questions, not for embedding documents.",
    )
    if api_input != st.session_state.api_key:
        st.session_state.api_key = api_input

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    # Model selection
    st.markdown(
        f"<p style='font-size:11px;color:{theme['text2']};text-transform:uppercase;"
        f"letter-spacing:0.8px;font-weight:500;margin-bottom:4px'>Gemini Model</p>",
        unsafe_allow_html=True,
    )
    model_options = [
        "gemini-2.0-flash-lite",   # most generous free-tier quota
        "gemini-2.0-flash",        # standard, slightly stricter quota
        "gemini-2.5-flash-lite",   # newest lite
        "gemini-2.5-flash",        # newest, strictest free-tier quota
    ]
    current_idx = (
        model_options.index(st.session_state.model_name)
        if st.session_state.model_name in model_options else 0
    )
    st.session_state.model_name = st.selectbox(
        "model", model_options, index=current_idx, label_visibility="collapsed",
    )

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    # Retrieval depth
    st.markdown(
        f"<p style='font-size:11px;color:{theme['text2']};text-transform:uppercase;"
        f"letter-spacing:0.8px;font-weight:500;margin-bottom:4px'>Search depth (chunks)</p>",
        unsafe_allow_html=True,
    )
    st.session_state.max_chunks = st.slider(
        "chunks", 1, 8, st.session_state.max_chunks, label_visibility="collapsed",
    )

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    dark_toggle = st.toggle("Dark mode", value=st.session_state.dark_mode)
    if dark_toggle != st.session_state.dark_mode:
        st.session_state.dark_mode = dark_toggle
        st.rerun()

    st.markdown(
        f"<div style='height:1px;background:{theme['border']};margin:18px 0'></div>",
        unsafe_allow_html=True,
    )

    # Upload
    st.markdown(
        f"<p style='font-size:11px;color:{theme['text2']};text-transform:uppercase;"
        f"letter-spacing:0.8px;font-weight:500;margin-bottom:8px'>Documents (max 5)</p>",
        unsafe_allow_html=True,
    )
    uploaded = st.file_uploader(
        "docs", type=["pdf", "docx", "txt"],
        accept_multiple_files=True, label_visibility="collapsed",
    )

    if uploaded and len(uploaded) > 5:
        st.markdown(
            f"<p style='color:{theme['danger']};font-size:12px'>❌ Max 5 files allowed.</p>",
            unsafe_allow_html=True,
        )
        uploaded = None

    if uploaded and not st.session_state.api_key:
        st.markdown(
            f"<p style='color:{theme['warning']};font-size:12px'>⚠️ Enter your API key first.</p>",
            unsafe_allow_html=True,
        )

    if uploaded and st.session_state.api_key:
        if st.button("⚡ Process Documents", type="primary", use_container_width=True):
            try:
                # 1. Load + chunk
                prog = st.progress(0, text="Reading documents...")
                all_chunks = []
                for i, f in enumerate(uploaded):
                    ext = os.path.splitext(f.name)[-1].lower()
                    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                        tmp.write(f.read())
                        tmp_path = tmp.name
                    try:
                        pages = load_file(tmp_path, f.name)
                        splitter = RecursiveCharacterTextSplitter(
                            chunk_size=1000, chunk_overlap=200,
                        )
                        all_chunks.extend(splitter.split_documents(pages))
                    finally:
                        try: os.unlink(tmp_path)
                        except OSError: pass
                    prog.progress((i + 1) / (len(uploaded) + 1),
                                  text=f"Reading {f.name}…")

                if not all_chunks:
                    prog.empty()
                    st.error("Could not extract any text from the uploaded documents.")
                else:
                    # 2. Load embeddings model (once, cached)
                    prog.progress(0.95, text="Loading embeddings model (first run may take ~30s)…")
                    if st.session_state.embeddings is None:
                        st.session_state.embeddings = load_embeddings_model()

                    # 3. Build vector store — LOCAL, no API calls
                    prog.progress(1.0, text=f"Embedding {len(all_chunks)} chunks locally…")
                    st.session_state.vectorstore = FAISS.from_documents(
                        all_chunks, st.session_state.embeddings,
                    )
                    st.session_state.files_processed = True
                    st.session_state.file_names = [f.name for f in uploaded]
                    st.session_state.chat_history = []
                    st.session_state.messages = []
                    prog.empty()
                    st.success(
                        f"✓ {len(uploaded)} document(s) indexed into "
                        f"{len(all_chunks)} chunks — locally, no API calls used."
                    )
                    st.rerun()
            except Exception as e:
                try: prog.empty()
                except Exception: pass
                st.error(friendly_error(e))

    # Show processed files
    if st.session_state.files_processed and st.session_state.file_names:
        st.markdown(
            f"<div style='height:1px;background:{theme['border']};margin:18px 0 14px'></div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<p style='font-size:11px;color:{theme['text2']};text-transform:uppercase;"
            f"letter-spacing:0.8px;font-weight:500;margin-bottom:8px'>Loaded files</p>",
            unsafe_allow_html=True,
        )
        for fname in st.session_state.file_names:
            icon = get_file_icon(fname)
            short = fname[:24] + "…" if len(fname) > 27 else fname
            st.markdown(
                f"""<div class="file-chip"><span>{icon}</span>
                <span style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap">{short}</span></div>""",
                unsafe_allow_html=True,
            )
        st.markdown(
            f"""<div class="status-ready"><span>✓ Ready to chat</span></div>""",
            unsafe_allow_html=True,
        )

# ──────────────────────────────────────────────────────────────
#  CHAT AREA
# ──────────────────────────────────────────────────────────────

# Read & clear pending suggested-question (set when user clicks a card)
pending_q = st.session_state.pop("_pending_question", None)

# Chat area renders at the top level so st.chat_input docks to bottom.
# ──────────────────────────────────────────────────────────
#  Layout order matters: history container goes FIRST,
#  then footer text, then chat_input is rendered LAST in
#  this column. That way:
#    - Old messages live in `history_area` at the top
#    - New messages get appended into `history_area`
#    - The input bar stays anchored at the bottom
#  This is the same pattern ChatGPT / Claude / Gemini use.
# ──────────────────────────────────────────────────────────
history_area = st.container()

# The chat_input MUST be the last interactive element so it stays
# at the bottom and the messages above. Streamlit renders elements
# in document order, so order here matters: history → input → footer.
disabled = not st.session_state.files_processed
placeholder_text = (
    "Ask anything about your documents..."
    if not disabled
    else "Upload documents in ⚙️ Settings to get started..."
)
typed_q = st.chat_input(placeholder_text, disabled=disabled)
question = pending_q or typed_q

# Footer caption renders BELOW the chat input
st.markdown(
    f"<p style='text-align:center;font-size:11px;color:{theme['text2']};"
    f"opacity:0.6;margin-top:8px'>DocuMind AI · Embeddings run locally · Answers powered by Gemini</p>",
    unsafe_allow_html=True,
)

# ── Render history (or welcome) INTO history_area ──────────
with history_area:
    if not st.session_state.messages and not question:
        # Welcome / suggestions screen
        st.markdown(
            f"""
            <div class="welcome-wrap">
              <div class="welcome-icon">🧠</div>
              <p class="welcome-title">What would you like to know?</p>
              <p class="welcome-sub">
                {'Ask anything about your documents.' if st.session_state.files_processed
                 else 'Upload documents in ⚙️ Settings to get started, then ask anything about them.'}
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.session_state.files_processed:
            SUGGESTIONS = [
                "📊 Summarize the key points of this document",
                "🔍 What are the main topics covered?",
                "💡 Explain the most complex section simply",
                "📋 List all important facts mentioned",
            ]
            sg1, sg2 = st.columns(2, gap="small")
            for i, suggestion in enumerate(SUGGESTIONS):
                target_col = sg1 if i % 2 == 0 else sg2
                with target_col:
                    if st.button(
                        suggestion,
                        key=f"sug_{i}",
                        use_container_width=True,
                    ):
                        st.session_state._pending_question = suggestion
                        st.rerun()
    else:
        # Replay full history (this includes the just-submitted
        # message because we append to messages before rendering)
        for msg in st.session_state.messages:
            with st.chat_message(
                msg["role"],
                avatar="👤" if msg["role"] == "user" else "🧠",
            ):
                if msg.get("warning"):
                    st.markdown(
                        f"<div style='background:{theme['warning']}22;"
                        f"border:1px solid {theme['warning']}66;"
                        f"border-radius:10px;padding:14px 18px;color:{theme['text']};'>"
                        f"{msg['content']}</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(msg["content"])

# ── handle submission ───────────────────────────────────────
if question and st.session_state.files_processed and st.session_state.api_key:
    # Append the user message FIRST so the history render above
    # picks it up correctly on the next rerun.
    st.session_state.messages.append({"role": "user", "content": question})

    # Render the user message + assistant streaming directly into
    # history_area so they land ABOVE the chat input, not below it.
    with history_area:
        with st.chat_message("user", avatar="👤"):
            st.markdown(question)

        with st.chat_message("assistant", avatar="🧠"):
            placeholder = st.empty()
            placeholder.markdown("_Searching your documents…_")

        try:
            # Build LLM and chains
            os.environ["GOOGLE_API_KEY"] = st.session_state.api_key
            model_lc = ChatGoogleGenerativeAI(
                model=st.session_state.model_name,
                temperature=0.2,
            )

            retriever = st.session_state.vectorstore.as_retriever(
                search_kwargs={"k": st.session_state.max_chunks},
            )

            contextualize_prompt = ChatPromptTemplate.from_messages([
                ("system",
                 "Given the chat history and the latest user question, "
                 "rewrite it as a standalone question. Do NOT answer it. "
                 "If it already makes sense alone, return it as is."),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ])
            rephrase_chain = contextualize_prompt | model_lc | StrOutputParser()

            answer_prompt = ChatPromptTemplate.from_messages([
                ("system",
                 "Answer the question based ONLY on the context below. "
                 "If the answer is not in the context, say 'I don't know based on "
                 "the provided documents.' Be clear, concise, and helpful.\n\n"
                 "Context:\n{context}"),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ])
            answer_chain = answer_prompt | model_lc

            # Step 1: rephrase question with history
            if st.session_state.chat_history:
                placeholder.markdown("_Understanding your question…_")
                standalone = rephrase_chain.invoke({
                    "input": question,
                    "chat_history": st.session_state.chat_history,
                })
            else:
                standalone = question

            # Step 2: retrieve (LOCAL, no API)
            placeholder.markdown("_Searching your documents…_")
            docs = retriever.invoke(standalone)
            context = "\n\n".join(d.page_content for d in docs)

            # Guard against runaway context: trim if it would blow past a
            # reasonable budget. Gemini Flash handles ~1M tokens but we
            # cap pragmatically to keep responses fast & cheap.
            MAX_CONTEXT_CHARS = 30_000
            if len(context) > MAX_CONTEXT_CHARS:
                context = context[:MAX_CONTEXT_CHARS] + "\n\n…[context trimmed]"

            if not context.strip():
                placeholder.warning(
                    "I couldn't find anything relevant in the documents to answer that."
                )
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "I couldn't find anything relevant in the documents to answer that.",
                    "warning": True,
                })
            else:
                # Step 3: stream the answer, with retry on transient failures
                placeholder.markdown("_Thinking…_")
                full_answer = ""
                chain_input = {
                    "input": question,
                    "chat_history": st.session_state.chat_history,
                    "context": context,
                }
                for chunk in call_gemini_with_retry(chain_input, answer_chain):
                    content = chunk.content if hasattr(chunk, "content") else str(chunk)
                    full_answer += content
                    placeholder.markdown(full_answer + " ▌")

                placeholder.markdown(full_answer)

                st.session_state.chat_history.append(HumanMessage(content=question))
                st.session_state.chat_history.append(AIMessage(content=full_answer))
                st.session_state.messages.append({
                    "role": "assistant", "content": full_answer,
                })

        except Exception as e:
            err = friendly_error(e)
            # Use markdown (with HTML allowed) so the <details> dropdown
            # with the raw error renders properly. st.warning would not.
            placeholder.markdown(
                f"<div style='background:{theme['warning']}22;border:1px solid {theme['warning']}66;"
                f"border-radius:10px;padding:14px 18px;color:{theme['text']};'>"
                f"{err}</div>",
                unsafe_allow_html=True,
            )
            st.session_state.messages.append({
                "role": "assistant", "content": err, "warning": True,
            })