import os
import re
from pathlib import Path

import numpy as np
import streamlit as st
import fitz
from docx import Document
from dotenv import load_dotenv
from google import genai
from google.genai import errors


# ============================================================
# DOCULENS AI
# INTELLIGENT DOCUMENT INVESTIGATOR
# ============================================================

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    st.error(
        "Gemini API key not found. "
        "Please check your .env file."
    )
    st.stop()

client = genai.Client(
    api_key=API_KEY
)


# ============================================================
# GEMINI MODELS
# ============================================================

GENERATION_MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-3.6-flash",
    "gemini-3.7-flash",
    "gemini-3.8-flash"
]

EMBEDDING_MODEL = "gemini-embedding-2"


# ============================================================
# DIRECTORIES
# ============================================================

UPLOAD_DIR = Path("data") / "uploads"

if UPLOAD_DIR.exists() and not UPLOAD_DIR.is_dir():

    st.error(
        "The path 'data/uploads' exists as a file "
        "instead of a folder."
    )

    st.stop()

UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
TOP_K = 5


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="DocuLens AI",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM UI
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 0px;
    }

    .subtitle {
        font-size: 17px;
        color: #777;
        margin-bottom: 25px;
    }

    .hero-box {
        padding: 25px;
        border-radius: 16px;
        border: 1px solid #dddddd;
        margin-bottom: 25px;
    }

    .answer-box {
        padding: 22px;
        border-radius: 14px;
        border: 1px solid #dddddd;
        margin-top: 10px;
        margin-bottom: 20px;
    }

    .feature-card {
        padding: 18px;
        border-radius: 12px;
        border: 1px solid #dddddd;
        min-height: 120px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SESSION STATE
# ============================================================

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "documents" not in st.session_state:
    st.session_state.documents = []

if "history" not in st.session_state:
    st.session_state.history = []

if "contradictions" not in st.session_state:
    st.session_state.contradictions = []


# ============================================================
# PDF EXTRACTION
# ============================================================

def extract_pdf(
    file_bytes,
    filename
):

    pages = []

    pdf = fitz.open(
        stream=file_bytes,
        filetype="pdf"
    )

    for page_number, page in enumerate(
        pdf,
        start=1
    ):

        text = page.get_text(
            "text"
        ).strip()

        if text:

            pages.append(
                {
                    "text": text,
                    "source": filename,
                    "page": page_number,
                    "type": "PDF"
                }
            )

    pdf.close()

    return pages


# ============================================================
# DOCX EXTRACTION
# ============================================================

def extract_docx(
    file_bytes,
    filename
):

    temp_path = UPLOAD_DIR / filename

    with open(
        temp_path,
        "wb"
    ) as file:

        file.write(
            file_bytes
        )

    document = Document(
        temp_path
    )

    text = "\n".join(
        paragraph.text
        for paragraph in document.paragraphs
        if paragraph.text.strip()
    )

    try:
        temp_path.unlink()
    except Exception:
        pass

    if not text.strip():
        return []

    return [
        {
            "text": text,
            "source": filename,
            "page": None,
            "type": "DOCX"
        }
    ]


# ============================================================
# TXT EXTRACTION
# ============================================================

def extract_txt(
    file_bytes,
    filename
):

    text = file_bytes.decode(
        "utf-8",
        errors="ignore"
    )

    if not text.strip():
        return []

    return [
        {
            "text": text,
            "source": filename,
            "page": None,
            "type": "TXT"
        }
    ]


# ============================================================
# TEXT CHUNKING
# ============================================================

def split_text(
    text,
    chunk_size=CHUNK_SIZE,
    overlap=CHUNK_OVERLAP
):

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    if len(text) <= chunk_size:
        return [text]

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[
            start:end
        ].strip()

        if chunk:
            chunks.append(
                chunk
            )

        start = end - overlap

    return chunks


# ============================================================
# GEMINI EMBEDDING
# ============================================================

def get_embedding(
    text,
    task_type="document"
):

    if task_type == "query":

        content = (
            "task: question answering | "
            f"query: {text}"
        )

    else:

        content = (
            "title: document section | "
            f"text: {text}"
        )

    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=content,
        config={
            "output_dimensionality": 768
        }
    )

    return np.array(
        result.embeddings[0].values,
        dtype=np.float32
    )


# ============================================================
# COSINE SIMILARITY
# ============================================================

def cosine_similarity(
    a,
    b
):

    denominator = (
        np.linalg.norm(a)
        *
        np.linalg.norm(b)
    )

    if denominator == 0:
        return 0.0

    return float(
        np.dot(a, b)
        /
        denominator
    )


# ============================================================
# PROCESS DOCUMENT
# ============================================================

def process_uploaded_file(
    uploaded_file
):

    file_bytes = uploaded_file.getvalue()

    filename = uploaded_file.name

    extension = (
        Path(filename)
        .suffix
        .lower()
    )

    if extension == ".pdf":

        pages = extract_pdf(
            file_bytes,
            filename
        )

    elif extension == ".docx":

        pages = extract_docx(
            file_bytes,
            filename
        )

    elif extension == ".txt":

        pages = extract_txt(
            file_bytes,
            filename
        )

    else:

        return []

    all_chunks = []

    for page in pages:

        chunks = split_text(
            page["text"]
        )

        for index, chunk in enumerate(
            chunks
        ):

            all_chunks.append(
                {
                    "text": chunk,
                    "source": page["source"],
                    "page": page["page"],
                    "type": page["type"],
                    "chunk_id": index
                }
            )

    return all_chunks


# ============================================================
# BUILD INDEX
# ============================================================

def build_index(
    chunks
):

    indexed_chunks = []

    progress = st.progress(
        0
    )

    total = len(chunks)

    for index, chunk in enumerate(
        chunks
    ):

        try:

            embedding = get_embedding(
                chunk["text"],
                task_type="document"
            )

            item = chunk.copy()

            item["embedding"] = embedding

            indexed_chunks.append(
                item
            )

        except Exception as error:

            st.warning(
                "Could not process one "
                f"document section: {error}"
            )

        progress.progress(
            (index + 1) / total
        )

    progress.empty()

    return indexed_chunks


# ============================================================
# RETRIEVE RELEVANT SECTIONS
# ============================================================

def retrieve_chunks(
    question,
    chunks,
    top_k=TOP_K
):

    if not chunks:
        return []

    query_embedding = get_embedding(
        question,
        task_type="query"
    )

    scored_chunks = []

    for chunk in chunks:

        score = cosine_similarity(
            query_embedding,
            chunk["embedding"]
        )

        scored_chunks.append(
            (
                score,
                chunk
            )
        )

    scored_chunks.sort(
        key=lambda item: item[0],
        reverse=True
    )

    return scored_chunks[:top_k]


# ============================================================
# GEMINI REQUEST WITH FALLBACK
# ============================================================

def ask_gemini(
    prompt
):

    last_error = None

    for model_name in GENERATION_MODELS:

        try:

            response = client.models.generate_content(
                model=model_name,
                contents=prompt
            )

            return response.text

        except errors.ServerError as error:

            last_error = error

            continue

        except Exception as error:

            last_error = error

            continue

    return (
        "⚠️ Gemini is temporarily unavailable.\n\n"
        "Please try again in a moment.\n\n"
        f"Technical detail: {last_error}"
    )


# ============================================================
# GENERATE ANSWER
# ============================================================

def generate_answer(
    question,
    retrieved_chunks
):

    if not retrieved_chunks:

        return (
            "The uploaded documents do not contain "
            "enough evidence to answer this question."
        )

    evidence = []

    for rank, (
        score,
        chunk
    ) in enumerate(
        retrieved_chunks,
        start=1
    ):

        page = (
            str(chunk["page"])
            if chunk["page"]
            else "N/A"
        )

        evidence.append(
            f"""
SOURCE {rank}

Document: {chunk["source"]}
Page: {page}
Relevance: {score:.2f}

CONTENT:
{chunk["text"]}
"""
        )

    evidence_text = "\n".join(
        evidence
    )

    prompt = f"""
You are DocuLens AI,
an intelligent document investigation assistant.

Answer the user's question ONLY using
the document evidence supplied below.

RULES:

1. Do not invent facts.
2. Do not use outside knowledge.
3. If evidence is insufficient, say:
"The uploaded documents do not contain enough
evidence to answer this question."
4. Give a clear and concise answer.
5. Mention supporting documents.
6. Mention page numbers when available.
7. If documents conflict, identify the conflict.
8. Preserve dates, numbers and names accurately.

USER QUESTION:

{question}

DOCUMENT EVIDENCE:

{evidence_text}

Return:

ANSWER:
<answer>

SOURCES:
- <document name and page>
"""

    return ask_gemini(
        prompt
    )


# ============================================================
# CONTRADICTION DETECTION
# ============================================================

def detect_contradictions(
    chunks
):

    if len(chunks) < 2:

        return (
            "Not enough documents to perform "
            "a contradiction analysis."
        )

    selected_chunks = chunks[:20]

    evidence = []

    for index, chunk in enumerate(
        selected_chunks,
        start=1
    ):

        page = (
            str(chunk["page"])
            if chunk["page"]
            else "N/A"
        )

        evidence.append(
            f"""
DOCUMENT SECTION {index}

Document: {chunk["source"]}
Page: {page}

CONTENT:
{chunk["text"]}
"""
        )

    evidence_text = "\n".join(
        evidence
    )

    prompt = f"""
You are the contradiction-analysis engine
of DocuLens AI.

Analyze the uploaded document evidence below.

Identify meaningful factual conflicts between
different documents.

Look especially for conflicts involving:

- dates
- deadlines
- names
- numbers
- policies
- requirements
- quantities
- locations
- instructions

IMPORTANT:

Do NOT call something a contradiction merely
because two documents discuss different topics.

Only report a contradiction when two sources
make genuinely incompatible claims about
the same subject.

If no meaningful contradiction exists, say:

NO CONTRADICTIONS DETECTED

If a contradiction exists, use:

⚠️ POTENTIAL CONTRADICTION

Topic:
<topic>

Source A:
<document and page>

Claim A:
<claim>

Source B:
<document and page>

Claim B:
<claim>

Why this matters:
<short explanation>

Do not invent information.

DOCUMENT EVIDENCE:

{evidence_text}
"""

    return ask_gemini(
        prompt
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">'
    '🔎 DocuLens AI'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Intelligent Document Investigation Platform'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "📂 Document Collection"
    )

    uploaded_files = st.file_uploader(
        "Upload documents",
        type=[
            "pdf",
            "docx",
            "txt"
        ],
        accept_multiple_files=True
    )

    if uploaded_files:

        st.write(
            f"**{len(uploaded_files)} "
            "document(s) selected**"
        )

        if st.button(
            "🚀 Process Documents",
            use_container_width=True
        ):

            with st.spinner(
                "Extracting and indexing documents..."
            ):

                all_chunks = []

                for file in uploaded_files:

                    chunks = (
                        process_uploaded_file(
                            file
                        )
                    )

                    all_chunks.extend(
                        chunks
                    )

                if not all_chunks:

                    st.error(
                        "No readable text was found "
                        "in the uploaded documents."
                    )

                else:

                    st.session_state.chunks = (
                        build_index(
                            all_chunks
                        )
                    )

                    st.session_state.documents = [
                        file.name
                        for file in uploaded_files
                    ]

                    st.session_state.contradictions = []

                    st.success(
                        f"Indexed "
                        f"{len(st.session_state.chunks)} "
                        "document sections."
                    )

    st.divider()

    st.subheader(
        "📊 Collection"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Documents",
            len(
                st.session_state.documents
            )
        )

    with col2:

        st.metric(
            "Sections",
            len(
                st.session_state.chunks
            )
        )

    if st.session_state.documents:

        st.divider()

        st.subheader(
            "📚 Documents"
        )

        for document in (
            st.session_state.documents
        ):

            st.write(
                f"📄 {document}"
            )

    st.divider()

    if st.button(
        "🗑️ Clear Collection",
        use_container_width=True
    ):

        st.session_state.chunks = []

        st.session_state.documents = []

        st.session_state.history = []

        st.session_state.contradictions = []

        st.rerun()


# ============================================================
# MAIN APPLICATION
# ============================================================

if not st.session_state.chunks:

    st.markdown(
        """
        <div class="hero-box">

        ### 🕵️ Investigate Your Documents

        Upload multiple documents and ask questions
        using natural language.

        DocuLens AI retrieves relevant evidence,
        generates grounded answers and identifies
        potential conflicts between documents.

        </div>
        """,
        unsafe_allow_html=True
    )

    st.info(
        "👈 Upload documents from the sidebar "
        "to begin an investigation."
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown(
            """
            <div class="feature-card">

            ### 📄 Upload

            Upload PDF, DOCX or TXT documents.

            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:

        st.markdown(
            """
            <div class="feature-card">

            ### 🔎 Investigate

            Ask natural-language questions
            across your document collection.

            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:

        st.markdown(
            """
            <div class="feature-card">

            ### ⚠️ Detect Conflicts

            Identify potential contradictions
            between documents.

            </div>
            """,
            unsafe_allow_html=True
        )


else:

    tab1, tab2, tab3 = st.tabs(
        [
            "🔎 Investigation",
            "⚠️ Contradictions",
            "🕘 History"
        ]
    )


    # ========================================================
    # INVESTIGATION TAB
    # ========================================================

    with tab1:

        st.subheader(
            "💬 Ask Your Documents"
        )

        # IMPORTANT:
        # Using a Streamlit form makes ENTER submit
        # the question instead of requiring a button click.

        with st.form(
            key="investigation_form",
            clear_on_submit=False
        ):

            question = st.text_input(
                "What would you like to investigate?",
                placeholder=(
                    "Type your question and press Enter..."
                )
            )

            submitted = st.form_submit_button(
                "🔍 Investigate",
                type="primary",
                use_container_width=True
            )

        if submitted:

            if not question.strip():

                st.warning(
                    "Please enter a question."
                )

            else:

                with st.spinner(
                    "🕵️ Investigating your documents..."
                ):

                    try:

                        retrieved = retrieve_chunks(
                            question,
                            st.session_state.chunks
                        )

                        answer = generate_answer(
                            question,
                            retrieved
                        )

                    except Exception as error:

                        st.error(
                            "Something went wrong "
                            "during the investigation."
                        )

                        st.code(
                            str(error)
                        )

                        st.stop()

                st.markdown(
                    "### 🤖 Investigation Result"
                )

                st.markdown(
                    f"""
                    <div class="answer-box">
                    {answer}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.markdown(
                    "### 📌 Supporting Evidence"
                )

                for (
                    score,
                    chunk
                ) in retrieved:

                    if score >= 0.70:
                        confidence = "🟢 Strong"

                    elif score >= 0.50:
                        confidence = "🟡 Moderate"

                    else:
                        confidence = "🔴 Weak"

                    page_text = (
                        f"Page {chunk['page']}"
                        if chunk["page"]
                        else "Page N/A"
                    )

                    with st.expander(
                        f"📄 {chunk['source']} — "
                        f"{page_text} — "
                        f"{confidence}"
                    ):

                        st.write(
                            chunk["text"]
                        )

                        st.caption(
                            f"Relevance score: {score:.2f}"
                        )

                st.session_state.history.append(
                    {
                        "question": question,
                        "answer": answer
                    }
                )


    # ========================================================
    # CONTRADICTION TAB
    # ========================================================

    with tab2:

        st.subheader(
            "⚠️ Contradiction Detection"
        )

        st.write(
            "Analyze uploaded documents for potential "
            "conflicts in dates, numbers, requirements, "
            "policies and other facts."
        )

        if st.button(
            "🕵️ Analyze Documents",
            type="primary",
            use_container_width=True
        ):

            with st.spinner(
                "Comparing document evidence..."
            ):

                try:

                    contradiction_result = (
                        detect_contradictions(
                            st.session_state.chunks
                        )
                    )

                    st.session_state.contradictions = (
                        contradiction_result
                    )

                except Exception as error:

                    st.error(
                        "Contradiction analysis failed."
                    )

                    st.code(
                        str(error)
                    )

        if st.session_state.contradictions:

            result = (
                st.session_state.contradictions
            )

            if (
                "NO CONTRADICTIONS DETECTED"
                in result.upper()
            ):

                st.success(
                    "✅ No meaningful contradictions "
                    "were detected."
                )

            else:

                st.warning(
                    "⚠️ Potential contradiction(s) detected."
                )

            st.markdown(
                f"""
                <div class="answer-box">
                {result}
                </div>
                """,
                unsafe_allow_html=True
            )


    # ========================================================
    # HISTORY TAB
    # ========================================================

    with tab3:

        st.subheader(
            "🕘 Investigation History"
        )

        if not st.session_state.history:

            st.info(
                "No investigations have been performed yet."
            )

        else:

            for item in reversed(
                st.session_state.history
            ):

                with st.expander(
                    f"Q: {item['question']}"
                ):

                    st.write(
                        item["answer"]
                    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "DocuLens AI • Evidence-backed intelligent "
    "document investigation"
)