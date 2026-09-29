import os
import html
import tempfile
from pathlib import Path

import streamlit as st

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from qdrant_client import QdrantClient

from agent import build_agent, run_agent


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="PatentLens",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
<style>

    /* Main page */
    .stApp {
        background: #0B0B0C;
        color: #F2EFE9;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #17181A;
    }

    section[data-testid="stSidebar"] > div {
        padding-top: 1.5rem;
    }

    /* Hide Streamlit decoration */
    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    /* Hero */
    .hero-box {
        border: 1px solid #6f5b22;
        border-radius: 18px;
        background: #1d1a0e;
        padding: 28px 36px;
        margin-bottom: 24px;
    }

    .hero-title {
        color: #E8C97A;
        font-size: 38px;
        font-weight: 800;
        line-height: 1.1;
    }

    .hero-subtitle {
        color: #D8D4CC;
        font-size: 16px;
        margin-top: 12px;
    }

    /* Metrics */
    .metric-box {
        border: 1px solid #514723;
        border-radius: 15px;
        background: #17181A;
        padding: 20px 10px;
        text-align: center;
        min-height: 90px;
    }

    .metric-number {
        color: #E8C97A;
        font-size: 30px;
        font-weight: 800;
    }

    .metric-label {
        color: #AAA7A0;
        font-size: 12px;
        letter-spacing: 0.5px;
        margin-top: 4px;
    }

    /* Query */
    .query-box {
        border: 1px solid #333438;
        border-radius: 15px;
        background: #17181A;
        padding: 18px 22px;
        margin-top: 18px;
        margin-bottom: 16px;
        color: #F2EFE9;
        font-size: 17px;
    }

    /* Result container */
    .result-box {
        border: 1px solid #333438;
        border-radius: 15px;
        background: #17181A;
        padding: 24px;
        margin-bottom: 18px;
    }

    /* Section headings */
    .section-title {
        color: #F2EFE9;
        font-size: 25px;
        font-weight: 750;
        margin-top: 18px;
        margin-bottom: 12px;
    }

    /* Source chips */
    .source-chip {
        display: inline-block;
        border: 1px solid #6f5b22;
        border-radius: 18px;
        background: #19180f;
        color: #E8C97A;
        padding: 7px 12px;
        margin: 3px 5px 3px 0;
        font-size: 13px;
    }

    /* Agent cards */
    .agent-card {
        border: 1px solid #343538;
        border-radius: 14px;
        background: #151617;
        padding: 16px;
        margin-bottom: 10px;
    }

    .agent-title {
        color: #E8C97A;
        font-size: 17px;
        font-weight: 700;
        margin-bottom: 8px;
    }

    .agent-text {
        color: #D7D4CE;
        font-size: 14px;
        white-space: pre-wrap;
        line-height: 1.55;
    }

    /* Status */
    .ready {
        color: #9DE6A7;
        font-size: 14px;
    }

    .not-ready {
        color: #D9D5CE;
        font-size: 14px;
    }

    /* Sidebar divider */
    .side-divider {
        border-top: 1px solid #3A3A3A;
        margin: 22px 0;
    }

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

DEFAULTS = {
    "processed": False,
    "vectorstore": None,
    "retriever": None,
    "agent": None,
    "file_count": 0,
    "page_count": 0,
    "chunk_count": 0,
    "file_names": [],
    "file_sizes": {},
    "last_question": "",
    "last_result": None,
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# HELPERS
# ============================================================

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


def get_llm(api_key):
    return ChatGroq(
        api_key=api_key,
        model="openai/gpt-oss-20b",
        temperature=0,
    )


def load_and_process_pdfs(uploaded_files):
    all_documents = []
    total_pages = 0

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    )

    temp_dir = Path(
        tempfile.mkdtemp(prefix="patentlens_")
    )

    for uploaded_file in uploaded_files:

        file_path = temp_dir / uploaded_file.name

        file_path.write_bytes(
            uploaded_file.getbuffer()
        )

        loader = PyPDFLoader(
            str(file_path)
        )

        pages = loader.load()

        total_pages += len(pages)

        patent_id = Path(
            uploaded_file.name
        ).stem

        for page in pages:
            page.metadata["patent_id"] = patent_id
            page.metadata["source_file"] = uploaded_file.name

        chunks = splitter.split_documents(
            pages
        )

        for chunk in chunks:
            chunk.metadata["patent_id"] = patent_id
            chunk.metadata["source_file"] = uploaded_file.name

        all_documents.extend(chunks)

    return all_documents, total_pages


def create_vectorstore(documents):
    embeddings = get_embeddings()

    vectorstore = QdrantVectorStore.from_documents(
        documents,
        embedding=embeddings,
        location=":memory:",
        collection_name="patentlens",
    )

    return vectorstore


def safe_text(value):
    return html.escape(str(value))


def render_metric(number, label):
    st.markdown(
        f"""
        <div class="metric-box">
            <div class="metric-number">{safe_text(number)}</div>
            <div class="metric-label">{safe_text(label)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sources(sources):
    if not sources:
        st.info("No sources available.")
        return

    chips = ""

    for source in sources:
        patent_id = safe_text(
            source.get("patent_id", "Unknown")
        )
        page = safe_text(
            source.get("page", "Unknown")
        )

        chips += (
            f'<span class="source-chip">'
            f'📄 {patent_id} · p.{page}'
            f'</span>'
        )

    st.markdown(
        chips,
        unsafe_allow_html=True,
    )


def render_agent_workflow(result):
    st.markdown(
        '<div class="section-title">🧠 Multi-Agent Workflow</div>',
        unsafe_allow_html=True,
    )

    outputs = result.get(
        "agent_outputs",
        {},
    )

    order = [
        "Supervisor Agent",
        "Search Agent",
        "Query Rewriter Agent",
        "Analysis Agent",
        "Research Analysis Agent",
        "Similarity Agent",
        "Comparison Agent",
        "Evidence Verification Agent",
        "Critic Agent",
        "Synthesis Agent",
    ]

    shown = set()

    for agent_name in order:

        if agent_name not in outputs:
            continue

        shown.add(agent_name)

        with st.expander(
            f"🤖 {agent_name}",
            expanded=True,
        ):
            st.write(
                outputs[agent_name]
            )

    # Display any future agents automatically.
    for agent_name, output in outputs.items():

        if agent_name in shown:
            continue

        with st.expander(
            f"🤖 {agent_name}",
            expanded=True,
        ):
            st.write(output)


def render_evidence(result):
    st.markdown(
        '<div class="section-title">📚 Retrieved Evidence</div>',
        unsafe_allow_html=True,
    )

    evidence = result.get(
        "retrieved_evidence",
        [],
    )

    if not evidence:
        st.info(
            "No retrieved evidence."
        )
        return

    for item in evidence:

        patent_id = item.get(
            "patent_id",
            "Unknown",
        )

        page = item.get(
            "page",
            "Unknown",
        )

        content = item.get(
            "content",
            "",
        )

        with st.expander(
            f"📄 {patent_id} · Page {page}"
        ):
            st.write(content)


def render_steps(result):
    st.markdown(
        '<div class="section-title">🔄 Agent Execution Steps</div>',
        unsafe_allow_html=True,
    )

    steps = result.get(
        "steps",
        [],
    )

    if not steps:
        st.info("No execution steps.")
        return

    for step in steps:
        st.write(step)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "## ⚙️ Setup"
    )

    st.caption(
        "Supervisor → Search → Verify → Specialist → Synthesis"
    )

    st.markdown(
        "**Groq API Key**"
    )

    api_key = st.text_input(
        "Groq API Key",
        type="password",
        value=os.getenv(
            "GROQ_API_KEY",
            "",
        ),
        label_visibility="collapsed",
        help="Enter your Groq API key.",
    )

    st.markdown(
        "### Upload patent PDFs"
    )

    uploaded_files = st.file_uploader(
        "Upload patent PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    process_button = st.button(
        "🚀 Process documents",
        use_container_width=True,
        type="primary",
    )

    st.markdown(
        '<div class="side-divider"></div>',
        unsafe_allow_html=True,
    )

    if st.session_state.processed:

        st.markdown(
            '<div class="ready">🟢 Agent ready. Ask a question.</div>',
            unsafe_allow_html=True,
        )

    else:

        st.markdown(
            '<div class="not-ready">⚪ Upload and process documents first.</div>',
            unsafe_allow_html=True,
        )


# ============================================================
# PROCESS DOCUMENTS
# ============================================================

if process_button:

    if not api_key:
        st.error(
            "Please enter your Groq API key."
        )
        st.stop()

    if not uploaded_files:
        st.error(
            "Please upload at least one patent PDF."
        )
        st.stop()

    with st.spinner(
        "Processing patent documents..."
    ):

        try:

            documents, page_count = (
                load_and_process_pdfs(
                    uploaded_files
                )
            )

            if not documents:
                st.error(
                    "No text could be extracted from the uploaded PDFs."
                )
                st.stop()

            vectorstore = create_vectorstore(
                documents
            )

            retriever = vectorstore.as_retriever(
                search_kwargs={
                    "k": 6
                }
            )

            llm = get_llm(
                api_key
            )

            agent = build_agent(
                retriever,
                llm,
            )

            st.session_state.processed = True
            st.session_state.vectorstore = vectorstore
            st.session_state.retriever = retriever
            st.session_state.agent = agent

            st.session_state.file_count = len(
                uploaded_files
            )

            st.session_state.page_count = page_count

            st.session_state.chunk_count = len(
                documents
            )

            st.session_state.file_names = [
                file.name
                for file in uploaded_files
            ]

            st.session_state.file_sizes = {
                file.name: file.size
                for file in uploaded_files
            }

            st.session_state.last_question = ""
            st.session_state.last_result = None

            st.success(
                "Patent documents processed successfully."
            )

        except Exception as exc:

            st.error(
                f"Processing failed: {exc}"
            )

            st.exception(exc)


# ============================================================
# QUESTION INPUT
# ============================================================

question = st.chat_input(
    "Ask PatentLens about your uploaded patents..."
)


# ============================================================
# RUN AGENT
# ============================================================

if question:

    if not st.session_state.processed:

        st.warning(
            "Please upload and process patent PDFs first."
        )

        st.stop()

    st.session_state.last_question = question

    with st.spinner(
        "PatentLens agents are working..."
    ):

        try:

            result = run_agent(
                st.session_state.agent,
                question,
            )

            st.session_state.last_result = result

        except Exception as exc:

            st.error(
                f"Agent execution failed: {exc}"
            )

            st.exception(exc)

            st.stop()


# ============================================================
# MAIN HERO
# ============================================================

st.markdown(
    """
<div class="hero-box">
<div class="hero-title">🔬 PatentLens</div>
<div class="hero-subtitle">Explainable Multi-Agent RAG — specialized agents search, verify, analyze and synthesize patent evidence.</div>
</div>
""",
    unsafe_allow_html=True,
)


# ============================================================
# METRICS
# ============================================================

metric1, metric2, metric3 = st.columns(3)

with metric1:
    render_metric(
        st.session_state.file_count,
        "FILES INDEXED",
    )

with metric2:
    render_metric(
        st.session_state.page_count,
        "PAGES",
    )

with metric3:
    render_metric(
        st.session_state.chunk_count,
        "CHUNKS",
    )


# ============================================================
# DISPLAY RESULT
# ============================================================

result = st.session_state.last_result

if result:

    # --------------------------------------------------------
    # USER QUESTION
    # --------------------------------------------------------

    st.markdown(
        f"""
        <div class="query-box">
            👤 {safe_text(st.session_state.last_question)}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # RESULT METRICS
    # --------------------------------------------------------

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            "**Intent**"
        )

        st.markdown(
            f"""
            <div style="
                font-size:30px;
                color:#F2EFE9;
                margin-top:3px;
            ">
                {safe_text(result.get("intent", "SEARCH"))}
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            "**Search attempts**"
        )

        st.markdown(
            f"""
            <div style="
                font-size:30px;
                color:#F2EFE9;
                margin-top:3px;
            ">
                {safe_text(result.get("attempts", 0))}
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            "**Sources**"
        )

        st.markdown(
            f"""
            <div style="
                font-size:30px;
                color:#F2EFE9;
                margin-top:3px;
            ">
                {safe_text(len(result.get("sources", [])))}
            </div>
            """,
            unsafe_allow_html=True,
        )

    # --------------------------------------------------------
    # FINAL ANSWER
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">✨ Final Answer</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="result-box">',
        unsafe_allow_html=True,
    )

    st.markdown(
        result.get(
            "answer",
            "No answer generated.",
        )
    )

    st.markdown(
        "</div>",
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # SOURCES
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">📚 Sources</div>',
        unsafe_allow_html=True,
    )

    render_sources(
        result.get(
            "sources",
            [],
        )
    )

    # --------------------------------------------------------
    # MULTI-AGENT WORKFLOW
    # --------------------------------------------------------

    render_agent_workflow(
        result
    )

    # --------------------------------------------------------
    # EXECUTION STEPS
    # --------------------------------------------------------

    render_steps(
        result
    )

    # --------------------------------------------------------
    # RETRIEVED EVIDENCE
    # --------------------------------------------------------

    render_evidence(
        result
    )

    # --------------------------------------------------------
    # VERIFICATION
    # --------------------------------------------------------

    verification = result.get(
        "verification",
        {},
    )

    st.markdown(
        '<div class="section-title">🔍 Evidence Verification</div>',
        unsafe_allow_html=True,
    )

    if verification.get(
        "sufficient",
        False,
    ):

        st.success(
            "Evidence is sufficient for the answer."
        )

    else:

        st.warning(
            "Evidence was not fully sufficient."
        )

    reason = verification.get(
        "reason",
        "",
    )

    if reason:
        st.write(reason)

