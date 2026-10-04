import os
import re
import time
import random
from pathlib import Path

import numpy as np
import streamlit as st
import pymupdf as fitz
from docx import Document
from dotenv import load_dotenv
from google import genai
from google.genai import types


# ============================================================
# DOCULENS AI
# Intelligent Document Investigator
# ============================================================

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

GENERATION_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.7-flash"
)

FALLBACK_MODEL = os.getenv(
    "GEMINI_FALLBACK_MODEL",
    ""
).strip()

EMBEDDING_MODEL = os.getenv(
    "GEMINI_EMBEDDING_MODEL",
    "gemini-embedding-2"
)

MAX_GENERATION_RETRIES = 4
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
TOP_K = 6
EMBEDDING_DIMENSION = 768

UPLOAD_DIR = Path("data") / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


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
# API CHECK
# ============================================================

if not API_KEY:
    st.error(
        "❌ GEMINI_API_KEY is missing.\n\n"
        "Please add GEMINI_API_KEY to Streamlit Secrets."
    )
    st.stop()

client = genai.Client(api_key=API_KEY)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
"""
<style>
.title {
    font-size: 42px;
    font-weight: 800;
    margin-bottom: 0;
}
.subtitle {
    color: #777;
    font-size: 17px;
    margin-bottom: 22px;
}
.hero {
    padding: 28px;
    border: 1px solid rgba(128,128,128,.25);
    border-radius: 18px;
    margin-bottom: 20px;
}
.result-card {
    padding: 22px;
    border: 1px solid rgba(128,128,128,.25);
    border-radius: 16px;
    margin-top: 15px;
}
.conflict-card {
    padding: 22px;
    border: 1px solid rgba(128,128,128,.25);
    border-radius: 16px;
    margin-top: 15px;
}
.metric {
    padding: 18px;
    border: 1px solid rgba(128,128,128,.25);
    border-radius: 14px;
    text-align: center;
    min-height: 105px;
}
.metric-number {
    font-size: 28px;
    font-weight: 800;
}
.metric-label {
    color: #777;
    font-size: 13px;
}
.high {
    border-left: 5px solid #2e8b57;
    padding: 13px 16px;
    border-radius: 8px;
    border-top: 1px solid rgba(128,128,128,.2);
    border-right: 1px solid rgba(128,128,128,.2);
    border-bottom: 1px solid rgba(128,128,128,.2);
}
.moderate {
    border-left: 5px solid #d99a00;
    padding: 13px 16px;
    border-radius: 8px;
    border-top: 1px solid rgba(128,128,128,.2);
    border-right: 1px solid rgba(128,128,128,.2);
    border-bottom: 1px solid rgba(128,128,128,.2);
}
.low {
    border-left: 5px solid #c0392b;
    padding: 13px 16px;
    border-radius: 8px;
    border-top: 1px solid rgba(128,128,128,.2);
    border-right: 1px solid rgba(128,128,128,.2);
    border-bottom: 1px solid rgba(128,128,128,.2);
}
.small {
    color: #777;
    font-size: 13px;
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
    st.session_state.contradictions = ""

if "last_answer" not in st.session_state:
    st.session_state.last_answer = ""

if "last_question" not in st.session_state:
    st.session_state.last_question = ""

if "last_retrieved" not in st.session_state:
    st.session_state.last_retrieved = []

if "processed" not in st.session_state:
    st.session_state.processed = False


# ============================================================
# HELPERS
# ============================================================

def clean_text(text):
    return re.sub(r"\s+", " ", text or "").strip()


def page_label(page):
    if page is None:
        return "Page N/A"
    return f"Page {page}"


# ============================================================
# GEMINI RELIABILITY
# ============================================================

def is_transient_error(error):
    message = str(error).lower()

    transient_markers = (
        "429",
        "500",
        "502",
        "503",
        "504",
        "unavailable",
        "high demand",
        "overloaded",
        "temporarily unavailable",
        "resource exhausted",
        "deadline exceeded"
    )

    return any(marker in message for marker in transient_markers)


def call_gemini_once(
    model,
    prompt,
    media_bytes=None,
    mime_type=None
):
    if media_bytes is None:
        response = client.models.generate_content(
            model=model,
            contents=prompt
        )
    else:
        media_part = types.Part.from_bytes(
            data=media_bytes,
            mime_type=mime_type
        )

        response = client.models.generate_content(
            model=model,
            contents=[prompt, media_part]
        )

    result = (response.text or "").strip()

    if not result:
        raise RuntimeError("Gemini returned an empty response.")

    return result


def generate_text(
    prompt,
    media_bytes=None,
    mime_type=None
):
    models = [GENERATION_MODEL]

    if FALLBACK_MODEL and FALLBACK_MODEL != GENERATION_MODEL:
        models.append(FALLBACK_MODEL)

    last_error = None

    for model_index, model in enumerate(models):
        for attempt in range(MAX_GENERATION_RETRIES):
            try:
                return call_gemini_once(
                    model=model,
                    prompt=prompt,
                    media_bytes=media_bytes,
                    mime_type=mime_type
                )

            except Exception as error:
                last_error = error

                if not is_transient_error(error):
                    raise RuntimeError(str(error)) from error

                if attempt < MAX_GENERATION_RETRIES - 1:
                    delay = (
                        1.5 * (2 ** attempt)
                        + random.uniform(0, 0.75)
                    )
                    time.sleep(delay)

        if model_index < len(models) - 1:
            continue

    raise RuntimeError(
        "Gemini is temporarily unavailable because "
        "the model is busy or experiencing high demand.\n\n"
        "DocuLens successfully retrieved your document "
        "evidence, but the final AI answer could not "
        "be generated.\n\n"
        "Please click Investigate again in a few seconds."
    ) from last_error


# ============================================================
# IMAGE OCR
# ============================================================

def ocr_image(
    image_bytes,
    mime_type,
    source_name,
    page=None
):
    prompt = """
You are the OCR engine for DocuLens AI.

Read the supplied image carefully.

Extract all meaningful visible text.

Rules:

1. Preserve names exactly.
2. Preserve dates exactly.
3. Preserve numbers exactly.
4. Preserve amounts exactly.
5. Preserve deadlines exactly.
6. Preserve headings and labels.
7. Preserve important instructions.
8. Maintain logical reading order.
9. Do not invent missing text.
10. If something is unreadable, write [UNCLEAR].

Return ONLY the extracted document text.
"""

    extracted = generate_text(
        prompt,
        media_bytes=image_bytes,
        mime_type=mime_type
    )

    if not extracted:
        return []

    return [{
        "text": extracted,
        "source": source_name,
        "page": page,
        "type": "IMAGE/OCR" if page is None else "PDF/OCR"
    }]


# ============================================================
# PDF EXTRACTION
# ============================================================

def extract_pdf(file_bytes, filename):
    try:
        pdf = fitz.open(
            stream=file_bytes,
            filetype="pdf"
        )
    except Exception as error:
        raise RuntimeError(
            f"Invalid or unreadable PDF: {error}"
        )

    pages = []

    try:
        for page_index in range(len(pdf)):
            page = pdf[page_index]
            page_number = page_index + 1

            native_text = clean_text(
                page.get_text("text")
            )

            if len(native_text) >= 30:
                pages.append({
                    "text": native_text,
                    "source": filename,
                    "page": page_number,
                    "type": "PDF"
                })
                continue

            try:
                matrix = fitz.Matrix(2.0, 2.0)

                pixmap = page.get_pixmap(
                    matrix=matrix,
                    alpha=False
                )

                image_bytes = pixmap.tobytes("png")

                ocr_result = ocr_image(
                    image_bytes=image_bytes,
                    mime_type="image/png",
                    source_name=filename,
                    page=page_number
                )

                pages.extend(ocr_result)

            except Exception:
                pages.append({
                    "text": "[This PDF page could not be OCR processed.]",
                    "source": filename,
                    "page": page_number,
                    "type": "PDF"
                })

    finally:
        pdf.close()

    return pages


# ============================================================
# DOCX EXTRACTION
# ============================================================

def extract_docx(file_bytes, filename):
    temp_path = UPLOAD_DIR / filename

    try:
        temp_path.write_bytes(file_bytes)

        document = Document(temp_path)
        content = []

        for paragraph in document.paragraphs:
            text = clean_text(paragraph.text)

            if text:
                content.append(text)

        for table_index, table in enumerate(
            document.tables,
            start=1
        ):
            content.append(f"Table {table_index}:")

            for row in table.rows:
                cells = []

                for cell in row.cells:
                    text = clean_text(cell.text)

                    if text:
                        cells.append(text)

                if cells:
                    content.append(" | ".join(cells))

        full_text = "\n".join(content).strip()

        if not full_text:
            return []

        return [{
            "text": full_text,
            "source": filename,
            "page": None,
            "type": "DOCX"
        }]

    except Exception as error:
        raise RuntimeError(
            f"Could not read DOCX: {error}"
        )

    finally:
        try:
            temp_path.unlink()
        except Exception:
            pass


# ============================================================
# TXT EXTRACTION
# ============================================================

def extract_txt(file_bytes, filename):
    text = file_bytes.decode(
        "utf-8",
        errors="ignore"
    )

    if not text.strip():
        return []

    return [{
        "text": text,
        "source": filename,
        "page": None,
        "type": "TXT"
    }]


# ============================================================
# FILE ROUTER
# ============================================================

def extract_file(uploaded_file):
    filename = uploaded_file.name

    extension = (
        Path(filename)
        .suffix
        .lower()
    )

    file_bytes = uploaded_file.getvalue()

    if extension == ".pdf":
        return extract_pdf(file_bytes, filename)

    if extension == ".docx":
        return extract_docx(file_bytes, filename)

    if extension == ".txt":
        return extract_txt(file_bytes, filename)

    if extension in {
        ".png",
        ".jpg",
        ".jpeg",
        ".webp"
    }:
        mime_types = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp"
        }

        return ocr_image(
            image_bytes=file_bytes,
            mime_type=mime_types[extension],
            source_name=filename
        )

    return []


# ============================================================
# CHUNKING
# ============================================================

def split_text(text):
    text = clean_text(text)

    if not text:
        return []

    if len(text) <= CHUNK_SIZE:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = start + CHUNK_SIZE
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = end - CHUNK_OVERLAP

    return chunks


def make_chunks(pages):
    chunks = []

    for page in pages:
        page_chunks = split_text(page["text"])

        for index, text in enumerate(page_chunks):
            chunks.append({
                "text": text,
                "source": page["source"],
                "page": page["page"],
                "type": page["type"],
                "chunk_id": index
            })

    return chunks


# ============================================================
# EMBEDDINGS
# ============================================================

def get_embedding(text):
    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
        config={
            "output_dimensionality": EMBEDDING_DIMENSION
        }
    )

    if not result.embeddings:
        raise RuntimeError(
            "Gemini returned no embedding."
        )

    return np.array(
        result.embeddings[0].values,
        dtype=np.float32
    )


def cosine_similarity(a, b):
    denominator = (
        np.linalg.norm(a)
        * np.linalg.norm(b)
    )

    if denominator == 0:
        return 0.0

    return float(
        np.dot(a, b) / denominator
    )


def build_index(chunks):
    indexed = []
    progress = st.progress(0)

    for index, chunk in enumerate(chunks):
        try:
            embedding = get_embedding(
                chunk["text"]
            )

            item = dict(chunk)
            item["embedding"] = embedding
            indexed.append(item)

        except Exception as error:
            st.warning(
                f"Could not index "
                f"{chunk['source']} "
                f"{page_label(chunk['page'])}: "
                f"{error}"
            )

        progress.progress(
            (index + 1) / max(1, len(chunks))
        )

    progress.empty()
    return indexed


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve(question, top_k=TOP_K):
    if not st.session_state.chunks:
        return []

    query_embedding = get_embedding(question)
    scored = []

    for chunk in st.session_state.chunks:
        score = cosine_similarity(
            query_embedding,
            chunk["embedding"]
        )

        scored.append((score, chunk))

    scored.sort(
        key=lambda item: item[0],
        reverse=True
    )

    return scored[:top_k]


# ============================================================
# CONFIDENCE
# ============================================================

def calculate_confidence(retrieved):
    if not retrieved:
        return (
            0.0,
            "Low",
            "No supporting evidence was retrieved."
        )

    scores = [
        max(0.0, min(1.0, score))
        for score, _ in retrieved
    ]

    top_score = scores[0]

    average_score = (
        sum(scores) / len(scores)
    )

    confidence = (
        0.65 * top_score
        + 0.35 * average_score
    )

    if confidence >= 0.75:
        return (
            confidence,
            "High",
            "Strong supporting evidence was retrieved."
        )

    if confidence >= 0.55:
        return (
            confidence,
            "Moderate",
            "Relevant evidence was found, but verify the answer."
        )

    return (
        confidence,
        "Low",
        "Evidence is weak. Verification is recommended."
    )


def show_confidence(retrieved):
    confidence, level, explanation = (
        calculate_confidence(retrieved)
    )

    percentage = int(
        round(confidence * 100)
    )

    if level == "High":
        css = "high"
        icon = "🟢"
    elif level == "Moderate":
        css = "moderate"
        icon = "🟡"
    else:
        css = "low"
        icon = "🔴"

    st.markdown(
        f"""
<div class="{css}">
    <b>
        {icon}
        Evidence Confidence:
        {level} — {percentage}%
    </b>
    <br>
    <span class="small">
        {explanation}
    </span>
</div>
""",
        unsafe_allow_html=True
    )

    return confidence, level


# ============================================================
# EVIDENCE
# ============================================================

def build_evidence(retrieved):
    blocks = []

    for rank, (score, chunk) in enumerate(
        retrieved,
        start=1
    ):
        blocks.append(
            f"""
SOURCE {rank}

Document:
{chunk["source"]}

Page:
{page_label(chunk["page"])}

Type:
{chunk["type"]}

Relevance:
{score:.3f}

CONTENT:
{chunk["text"]}
"""
        )

    return "\n".join(blocks)


# ============================================================
# ANSWER GENERATION
# ============================================================

def answer_question(question, retrieved):
    if not retrieved:
        return (
            "The uploaded documents do not contain "
            "enough evidence to answer this question."
        )

    confidence, level, _ = (
        calculate_confidence(retrieved)
    )

    evidence = build_evidence(retrieved)

    prompt = f"""
You are DocuLens AI,
an intelligent document investigator.

Answer the user's question using ONLY
the supplied document evidence.

STRICT RULES:

1. Never use outside knowledge.
2. Never invent facts.
3. If information is missing, say it is not found.
4. Preserve names exactly.
5. Preserve dates exactly.
6. Preserve numbers exactly.
7. Preserve deadlines exactly.
8. Preserve amounts exactly.
9. Cite document names.
10. Cite page numbers whenever available.
11. If documents conflict, explicitly explain the conflict.
12. Never silently choose one conflicting source.
13. Communicate uncertainty.
14. The certainty level MUST NOT exceed retrieval confidence.
15. If confidence is Moderate, certainty cannot be High.
16. If confidence is Low, certainty must be Low.
17. Be concise but useful.

RETRIEVAL CONFIDENCE:
{confidence:.2f} ({level})

USER QUESTION:
{question}

DOCUMENT EVIDENCE:
{evidence}

Return:

ANSWER:
<answer>

SOURCES:
- <document> — <page>
- <document> — <page>

CERTAINTY:
<High / Moderate / Low>

NOTE:
<uncertainty or conflict note>
If none, write:
None
"""

    return generate_text(prompt)


# ============================================================
# CONTRADICTION DETECTION
# ============================================================

def detect_contradictions():
    if len(st.session_state.chunks) < 2:
        return (
            "Not enough evidence sections are "
            "available for contradiction analysis."
        )

    selected = st.session_state.chunks[:40]
    evidence = []

    for index, chunk in enumerate(
        selected,
        start=1
    ):
        evidence.append(
            f"""
SECTION {index}

Document:
{chunk["source"]}

Page:
{page_label(chunk["page"])}

Type:
{chunk["type"]}

CONTENT:
{chunk["text"]}
"""
        )

    prompt = f"""
You are the contradiction-detection
engine of DocuLens AI.

Compare the supplied document sections.

Find genuine conflicts where two or more
sources make incompatible claims about
the same topic.

Check especially:

- deadlines
- dates
- names
- numbers
- amounts
- requirements
- policies
- quantities
- locations
- instructions
- project information

Do NOT report a contradiction if:

- documents discuss different topics
- values are compatible
- one source simply contains additional information
- the difference is not meaningful

If there are no genuine contradictions,
return exactly:

NO CONTRADICTIONS DETECTED

Otherwise use:

POTENTIAL CONTRADICTION

TOPIC:
<topic>

SOURCE A:
<document and page>

CLAIM A:
<claim>

SOURCE B:
<document and page>

CLAIM B:
<claim>

WHY THIS MATTERS:
<short explanation>

RECOMMENDED ACTION:
<short verification recommendation>

CONFIDENCE:
<High / Moderate / Low>

Never invent a claim.

DOCUMENT EVIDENCE:
{"".join(evidence)}
"""

    return generate_text(prompt)


def count_conflicts(text):
    if not text:
        return 0

    if "NO CONTRADICTIONS DETECTED" in text.upper():
        return 0

    return text.upper().count(
        "POTENTIAL CONTRADICTION"
    )


# ============================================================
# METRICS
# ============================================================

def show_metrics():
    documents = len(st.session_state.documents)
    sections = len(st.session_state.chunks)
    conflicts = count_conflicts(
        st.session_state.contradictions
    )
    investigations = len(st.session_state.history)

    columns = st.columns(4)

    metrics = [
        (documents, "📄 Documents"),
        (sections, "🧩 Evidence Sections"),
        (conflicts, "⚠️ Conflicts"),
        (investigations, "🔎 Investigations")
    ]

    for column, (number, label) in zip(
        columns,
        metrics
    ):
        with column:
            st.markdown(
                f"""
<div class="metric">
    <div class="metric-number">
        {number}
    </div>
    <div class="metric-label">
        {label}
    </div>
</div>
""",
                unsafe_allow_html=True
            )


# ============================================================
# HEADER
# ============================================================

st.markdown(
"""
<div class="title">
    🔎 DocuLens AI
</div>
""",
unsafe_allow_html=True
)

st.markdown(
"""
<div class="subtitle">
    Intelligent Document Investigation —
    Evidence • Sources • Conflicts • Uncertainty
</div>
""",
unsafe_allow_html=True
)

if st.session_state.documents:
    show_metrics()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("📂 Document Collection")

    uploaded_files = st.file_uploader(
        "Upload documents",
        type=[
            "pdf",
            "docx",
            "txt",
            "png",
            "jpg",
            "jpeg",
            "webp"
        ],
        accept_multiple_files=True,
        help="PDF, DOCX, TXT and image files are supported."
    )

    if uploaded_files:
        st.write(
            f"**{len(uploaded_files)} file(s) selected**"
        )

        if st.button(
            "🚀 Process Documents",
            type="primary",
            use_container_width=True
        ):
            all_pages = []
            status = st.empty()

            for index, uploaded_file in enumerate(
                uploaded_files,
                start=1
            ):
                status.info(
                    f"Reading "
                    f"{index}/{len(uploaded_files)}: "
                    f"{uploaded_file.name}"
                )

                try:
                    extracted = extract_file(
                        uploaded_file
                    )

                    if extracted:
                        all_pages.extend(extracted)
                    else:
                        st.warning(
                            f"No readable content found in "
                            f"{uploaded_file.name}"
                        )

                except Exception as error:
                    st.error(
                        f"Could not process "
                        f"{uploaded_file.name}: "
                        f"{error}"
                    )

            status.empty()

            chunks = make_chunks(all_pages)

            if not chunks:
                st.error(
                    "No readable content was extracted. "
                    "Please check your files."
                )

            else:
                with st.spinner(
                    "🧠 Building semantic evidence index..."
                ):
                    indexed = build_index(chunks)

                st.session_state.chunks = indexed

                st.session_state.documents = [
                    file.name
                    for file in uploaded_files
                ]

                st.session_state.contradictions = ""
                st.session_state.last_answer = ""
                st.session_state.last_question = ""
                st.session_state.last_retrieved = []
                st.session_state.history = []
                st.session_state.processed = True

                st.success(
                    f"Processed "
                    f"{len(uploaded_files)} document(s) "
                    f"and indexed "
                    f"{len(indexed)} evidence section(s)."
                )

    st.divider()

    st.subheader("📚 Current Collection")

    if st.session_state.documents:
        for document in st.session_state.documents:
            st.write(f"📄 {document}")

        st.caption(
            f"{len(st.session_state.chunks)} "
            f"evidence sections indexed"
        )

    else:
        st.caption("No documents processed yet.")

    st.divider()

    if st.button(
        "🗑️ Clear Collection",
        use_container_width=True
    ):
        st.session_state.chunks = []
        st.session_state.documents = []
        st.session_state.history = []
        st.session_state.contradictions = ""
        st.session_state.last_answer = ""
        st.session_state.last_question = ""
        st.session_state.last_retrieved = []
        st.session_state.processed = False

        st.rerun()


# ============================================================
# EMPTY STATE
# ============================================================

if not st.session_state.chunks:
    st.markdown(
"""
<div class="hero">

<h2>
    🕵️ Investigate Your Documents
</h2>

<p>
    Upload multiple documents, ask
    natural-language questions, inspect
    supporting evidence and detect
    conflicting claims.
</p>

<p>
    Supports
    <b>PDF, DOCX, TXT and Images</b>.
    Scanned PDF pages and images are
    OCR-processed so their information
    can also be retrieved.
</p>

</div>
""",
        unsafe_allow_html=True
    )

    st.info(
        "👈 Upload documents from the sidebar "
        "and click Process Documents."
    )

    columns = st.columns(4)

    cards = [
        ("📄", "Multi-format", "PDF • DOCX • TXT • Images"),
        ("🔎", "Smart Q&A", "Natural-language investigation"),
        ("📌", "Evidence", "Sources and page references"),
        ("⚠️", "Conflicts", "Cross-document contradiction detection")
    ]

    for column, (icon, title, description) in zip(
        columns,
        cards
    ):
        with column:
            st.markdown(
                f"""
<div class="metric">
<h3>
    {icon} {title}
</h3>
<div class="small">
    {description}
</div>
</div>
""",
                unsafe_allow_html=True
            )


# ============================================================
# MAIN APPLICATION
# ============================================================

else:
    (
        tab_investigate,
        tab_conflicts,
        tab_overview,
        tab_history
    ) = st.tabs(
        [
            "🔎 Investigation",
            "⚠️ Contradictions",
            "📊 Investigation Overview",
            "🕘 History"
        ]
    )

    # ========================================================
    # INVESTIGATION
    # ========================================================

    with tab_investigate:
        st.subheader("💬 Ask Your Documents")

        with st.form(
            "question_form",
            clear_on_submit=False
        ):
            question = st.text_input(
                "What would you like to investigate?",
                placeholder=(
                    "Example: "
                    "What is the project report submission date?"
                )
            )

            submitted = st.form_submit_button(
                "🔍 Investigate",
                type="primary",
                use_container_width=True
            )

        if submitted:
            if not question.strip():
                st.warning("Please enter a question.")
            else:
                try:
                    with st.spinner(
                        "🕵️ Searching evidence across all documents..."
                    ):
                        retrieved = retrieve(question)

                        answer = answer_question(
                            question,
                            retrieved
                        )

                    st.session_state.last_question = question
                    st.session_state.last_answer = answer
                    st.session_state.last_retrieved = retrieved

                    confidence, level, _ = (
                        calculate_confidence(retrieved)
                    )

                    st.session_state.history.append(
                        {
                            "question": question,
                            "answer": answer,
                            "confidence": confidence,
                            "level": level
                        }
                    )

                except Exception as error:
                    st.error(
                        "❌ Investigation could not be completed."
                    )

                    st.warning(str(error))

                    st.info(
                        "If Gemini is temporarily busy, "
                        "wait a few seconds and click "
                        "Investigate again."
                    )

        if st.session_state.last_answer:
            st.markdown("### 🤖 Investigation Result")

            show_confidence(
                st.session_state.last_retrieved
            )

            st.markdown(
                '<div class="result-card">',
                unsafe_allow_html=True
            )

            st.markdown(
                st.session_state.last_answer
            )

            st.markdown(
                '</div>',
                unsafe_allow_html=True
            )

            st.markdown("### 📌 Supporting Evidence")

            for rank, (score, chunk) in enumerate(
                st.session_state.last_retrieved,
                start=1
            ):
                if score >= 0.70:
                    strength = "🟢 Strong"
                elif score >= 0.50:
                    strength = "🟡 Moderate"
                else:
                    strength = "🔴 Weak"

                title = (
                    f"{rank}. "
                    f"{chunk['source']} — "
                    f"{page_label(chunk['page'])} — "
                    f"{strength}"
                )

                with st.expander(title):
                    st.write(chunk["text"])

                    st.caption(
                        f"Relevance: {score:.3f} | "
                        f"Type: {chunk['type']}"
                    )

            confidence, level, _ = (
                calculate_confidence(
                    st.session_state.last_retrieved
                )
            )

            if level == "High":
                st.success(
                    "The retrieved evidence strongly "
                    "supports this answer. Verify "
                    "critical decisions against the "
                    "original documents."
                )
            elif level == "Moderate":
                st.warning(
                    "Relevant evidence was found, "
                    "but this answer should be verified "
                    "against the cited sources."
                )
            else:
                st.error(
                    "Evidence is weak. Treat this answer "
                    "as uncertain and verify the original "
                    "documents."
                )

    # ========================================================
    # CONTRADICTIONS
    # ========================================================

    with tab_conflicts:
        st.subheader(
            "⚠️ Cross-Document Contradiction Detection"
        )

        st.write(
            "DocuLens compares claims across your "
            "uploaded documents instead of silently "
            "choosing one."
        )

        if st.button(
            "🕵️ Analyze Documents for Conflicts",
            type="primary",
            use_container_width=True
        ):
            try:
                with st.spinner(
                    "Comparing claims across documents..."
                ):
                    st.session_state.contradictions = (
                        detect_contradictions()
                    )

            except Exception as error:
                st.error(
                    "❌ Contradiction analysis "
                    "could not be completed."
                )

                st.warning(str(error))

                st.info(
                    "Retry in a few seconds if "
                    "Gemini is temporarily busy."
                )

        result = st.session_state.contradictions

        if not result:
            st.info(
                "Click the button above to analyze "
                "your documents for conflicts."
            )

        elif (
            "NO CONTRADICTIONS DETECTED"
            in result.upper()
        ):
            st.success(
                "✅ No meaningful contradictions were detected."
            )

        else:
            conflicts = count_conflicts(result)

            st.warning(
                f"⚠️ {conflicts} "
                f"potential contradiction(s) detected."
            )

            st.markdown(
                '<div class="conflict-card">',
                unsafe_allow_html=True
            )

            st.markdown(result)

            st.markdown(
                '</div>',
                unsafe_allow_html=True
            )

            st.info(
                "Recommended action: verify conflicting "
                "claims against the latest official source."
            )

    # ========================================================
    # OVERVIEW
    # ========================================================

    with tab_overview:
        st.subheader("📊 Investigation Overview")

        show_metrics()

        st.divider()

        st.markdown("### 📚 Document Evidence Map")

        for document in st.session_state.documents:
            count = sum(
                1
                for chunk in st.session_state.chunks
                if chunk["source"] == document
            )

            st.write(
                f"📄 **{document}** — "
                f"{count} evidence section(s)"
            )

        st.divider()

        st.markdown("### 🧠 DocuLens Capabilities")

        capabilities = [
            "📄 PDF extraction",
            "🔍 Scanned PDF OCR",
            "📝 DOCX paragraph extraction",
            "📊 DOCX table extraction",
            "📃 TXT extraction",
            "🖼️ Image OCR",
            "🔎 Semantic evidence retrieval",
            "🤖 Grounded natural-language Q&A",
            "📌 Source and page references",
            "⚠️ Cross-document contradiction detection",
            "🟢 High / Moderate / Low confidence",
            "🕘 Investigation history",
            "🛡️ Gemini temporary-error retry protection"
        ]

        for capability in capabilities:
            st.write(f"✅ {capability}")

        st.divider()

        st.markdown("### 🎯 Investigation Summary")

        if st.session_state.history:
            average = (
                sum(
                    item["confidence"]
                    for item in st.session_state.history
                )
                /
                len(st.session_state.history)
            )

            st.write(
                "Questions investigated: "
                f"**{len(st.session_state.history)}**"
            )

            st.write(
                "Average evidence confidence: "
                f"**{average:.0%}**"
            )

        else:
            st.info(
                "Ask a question to start building "
                "your investigation history."
            )

    # ========================================================
    # HISTORY
    # ========================================================

    with tab_history:
        st.subheader("🕘 Investigation History")

        if not st.session_state.history:
            st.info(
                "No investigations have been performed yet."
            )
        else:
            for index, item in enumerate(
                reversed(st.session_state.history),
                start=1
            ):
                with st.expander(
                    f"Investigation {index}: "
                    f"{item['question']}"
                ):
                    st.write(item["answer"])

                    st.caption(
                        f"Evidence confidence: "
                        f"{item['confidence']:.0%} "
                        f"({item['level']})"
                    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "DocuLens AI • Intelligent Document Investigation • "
    "Grounded Answers • Evidence • Conflict Detection • "
    "Uncertainty Handling"
)
