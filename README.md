# 🔎 DocuLens AI

## Intelligent Document Investigator

> **Upload → Investigate → See Evidence → Detect Conflicts → Understand Uncertainty**

DocuLens AI is an AI-powered document investigation platform that helps users extract, search, compare, and understand information scattered across multiple documents.

Instead of manually reading PDFs, Word documents, text files, notices, calendars, and images, users can upload their documents and ask questions in natural language.

DocuLens retrieves relevant evidence, provides source references, detects conflicting information, and communicates the confidence and uncertainty of its answers.

---

# 🚨 Problem Statement

Important information is often scattered across different document formats:

- PDF reports
- Academic calendars
- Official notices
- Word documents
- Text files
- Images
- Scanned documents

Finding a specific answer manually can be slow and error-prone.

The problem becomes even more difficult when different documents contain different versions of the same information.

For example:

```text
Academic Calendar
Project Report Submission → 15 October 2026

Updated Official Notice
Project Report Submission → 20 October 2026
```

A traditional question-answering system might simply return one answer.

An intelligent document investigator should instead recognize the conflict and tell the user that the documents disagree.

---

# 💡 Our Solution

**DocuLens AI** transforms a collection of documents into an intelligent investigation workspace.

The user can:

1. Upload multiple documents
2. Extract information automatically
3. Ask questions using natural language
4. Search relevant information semantically
5. Retrieve supporting evidence
6. View document and page references
7. Detect conflicting claims
8. Understand evidence confidence
9. Identify when information is missing
10. Investigate information contained inside images and scanned PDFs

The core idea is:

> **Don't just give an answer. Show the evidence behind the answer.**

---

# ✨ Key Features

## 📄 1. Multi-Format Document Processing

DocuLens AI supports multiple document formats:

- PDF
- Scanned PDF
- DOCX
- TXT
- PNG
- JPG
- JPEG
- WEBP

This allows users to investigate an entire document collection instead of working with only one format.

---

# 🔎 2. Natural-Language Document Investigation

Users can ask questions naturally.

Example:

```text
What is the project report submission date?
```

or:

```text
What are all the project-related deadlines?
```

or:

```text
What does this document say about project review?
```

DocuLens searches across the uploaded document collection and retrieves the most relevant evidence.

---

# 📌 3. Evidence-Based Answers

DocuLens does not treat an AI-generated answer as sufficient by itself.

The system retrieves supporting evidence and displays:

- Document name
- Page number when available
- Evidence content
- Relevance score
- Evidence confidence

This makes the investigation traceable.

Users can inspect the original retrieved evidence behind an answer.

---

# ⚠️ 4. Cross-Document Contradiction Detection

One of the key innovations of DocuLens AI is contradiction detection.

The system compares claims across uploaded documents and identifies potentially conflicting information.

### Example

**Academic Calendar**

```text
Project Report Submission:
15 October 2026
```

**Updated Notice**

```text
Project Report Submission:
20 October 2026
```

DocuLens does not silently choose one date.

Instead, it identifies:

```text
Potential Contradiction

Topic:
Project Report Submission Deadline

Source A:
academic_calendar_2026_2027.pdf — Page 1

Claim A:
Project reports must be submitted on or before
15 October 2026.

Source B:
updated_college_notice.txt

Claim B:
The final project report submission deadline
has been extended to 20 October 2026.

Why this matters:
Users relying on different documents may receive
different deadlines.

Recommended action:
Verify the latest official notice.
```

This makes DocuLens more useful than a basic document chatbot.

---

# 🛡️ 5. Uncertainty Handling

AI systems can sometimes produce answers that sound confident even when evidence is weak.

DocuLens explicitly communicates evidence confidence.

The application uses three levels:

### 🟢 High

Strong supporting evidence was retrieved.

### 🟡 Moderate

Relevant evidence was found, but verification is recommended.

### 🔴 Low

Evidence is weak or insufficient and the original documents should be checked.

The system is designed so that answer certainty does not exceed the strength of the retrieved evidence.

---

# 🖼️ 6. Image Understanding

Important information can also exist inside images.

Examples include:

- Posters
- Notices
- Screenshots
- Promotional material
- Scanned pages
- Photographed documents

DocuLens processes image content using AI vision/OCR so that text contained inside images can also become searchable evidence.

Example question:

```text
What does this poster say?
```

or:

```text
What does DocuLens AI do according to this image?
```

---

# 📑 7. Scanned PDF Support

Not every PDF contains selectable text.

Some PDFs are scanned images.

DocuLens handles this by using a two-stage approach:

```text
PDF Page
   │
   ├── Text available?
   │       │
   │       └── Yes → Extract text directly
   │
   └── No / very little text
           │
           ▼
        Render page
           │
           ▼
        OCR / Vision
           │
           ▼
      Extracted text
```

This allows scanned PDF pages to participate in document investigation.

---

# 📝 8. DOCX Processing

DOCX documents are processed for both normal text and tables.

The system extracts:

- Paragraph content
- Document text
- Table content

This allows structured information inside Word documents to become searchable evidence.

---

# 🧠 9. Semantic Retrieval

DocuLens uses semantic embeddings to find evidence based on meaning rather than relying only on exact keyword matches.

The process is:

```text
Documents
    ↓
Text Extraction
    ↓
Text Chunking
    ↓
Gemini Embeddings
    ↓
Semantic Index
    ↓
User Question
    ↓
Question Embedding
    ↓
Similarity Search
    ↓
Relevant Evidence
```

This allows questions to retrieve related information even when the wording is different.

---

# 🤖 10. Grounded AI Answers

After retrieving relevant evidence, Gemini generates the investigation result using the retrieved document content.

The system instructs the AI to:

- Use only supplied evidence
- Avoid inventing information
- Preserve important dates
- Preserve names
- Preserve numbers
- Preserve deadlines
- Cite document sources
- Mention conflicts
- Communicate uncertainty

If the information cannot be found in the uploaded documents, the system can state that sufficient evidence was not found instead of intentionally inventing an answer.

---

# 🕘 11. Investigation History

DocuLens maintains investigation history during the current application session.

Users can review:

- Previous questions
- Generated answers
- Evidence confidence

This allows multiple investigations to be performed during a document analysis session.

---

# 🔄 12. Gemini Reliability Handling

AI APIs can occasionally experience temporary service overload.

DocuLens includes retry handling for transient Gemini errors.

The application can retry temporary failures before displaying a user-friendly message.

Instead of exposing a raw API error, the application informs the user that the AI service may be temporarily unavailable and that the investigation can be retried.

---

# 🏗️ System Architecture

```text
                         USER
                           │
                           ▼
                ┌────────────────────┐
                │  Streamlit UI      │
                │                    │
                │ Upload / Ask /     │
                │ Investigate        │
                └─────────┬──────────┘
                          │
                          ▼
                ┌────────────────────┐
                │ Document Processing│
                │                    │
                │ PDF                │
                │ DOCX               │
                │ TXT                │
                │ Images             │
                └─────────┬──────────┘
                          │
                          ▼
                ┌────────────────────┐
                │ OCR / Extraction   │
                └─────────┬──────────┘
                          │
                          ▼
                ┌────────────────────┐
                │ Text Chunking      │
                └─────────┬──────────┘
                          │
                          ▼
                ┌────────────────────┐
                │ Gemini Embeddings  │
                └─────────┬──────────┘
                          │
                          ▼
                ┌────────────────────┐
                │ Semantic Retrieval │
                └─────────┬──────────┘
                          │
              ┌───────────┴───────────┐
              │                       │
              ▼                       ▼
     ┌─────────────────┐     ┌──────────────────┐
     │ Evidence        │     │ Contradiction    │
     │ Retrieval       │     │ Detection        │
     └────────┬────────┘     └────────┬─────────┘
              │                       │
              └───────────┬───────────┘
                          │
                          ▼
                ┌────────────────────┐
                │ Gemini Investigation│
                │ Engine             │
                └─────────┬──────────┘
                          │
                          ▼
                ┌────────────────────┐
                │ Final Investigation│
                │ Result             │
                │                    │
                │ Answer             │
                │ Sources            │
                │ Evidence           │
                │ Confidence         │
                │ Uncertainty        │
                └────────────────────┘
```

---

# 🔬 Investigation Pipeline

The complete investigation workflow is:

```text
1. Upload Documents
        ↓
2. Extract Text / OCR
        ↓
3. Split Documents into Evidence Chunks
        ↓
4. Generate Semantic Embeddings
        ↓
5. Build Searchable Evidence Index
        ↓
6. User Asks a Question
        ↓
7. Generate Question Embedding
        ↓
8. Retrieve Relevant Evidence
        ↓
9. Calculate Evidence Confidence
        ↓
10. Generate Grounded AI Answer
        ↓
11. Display Sources and Evidence
        ↓
12. Detect / Explain Conflicts
```

---

# 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core application logic |
| Streamlit | Interactive web interface |
| Google Gemini API | AI reasoning and vision processing |
| Gemini Embeddings | Semantic document retrieval |
| PyMuPDF | PDF extraction and scanned page rendering |
| python-docx | DOCX processing |
| NumPy | Vector similarity calculations |
| python-dotenv | Environment variable management |

---

# 📂 Supported Inputs

| File Type | Processing Method |
|---|---|
| PDF | Native text extraction |
| Scanned PDF | OCR / Vision |
| DOCX | Paragraph + table extraction |
| TXT | Text extraction |
| PNG | Vision / OCR |
| JPG | Vision / OCR |
| JPEG | Vision / OCR |
| WEBP | Vision / OCR |

---

# 🎯 Innovation

DocuLens AI is designed around a key idea:

> **Document investigation should be evidence-driven, conflict-aware, and uncertainty-aware.**

Traditional document Q&A focuses primarily on:

```text
Question → Answer
```

DocuLens extends this to:

```text
Question
   ↓
Search Multiple Documents
   ↓
Retrieve Evidence
   ↓
Compare Claims
   ↓
Detect Conflicts
   ↓
Measure Confidence
   ↓
Generate Grounded Answer
   ↓
Show Sources + Evidence + Uncertainty
```

This makes the system more suitable for real-world situations where information changes over time and different documents may disagree.

---

# 🌍 Potential Applications

DocuLens AI can be useful in many domains.

### 🎓 Education

- Academic calendars
- Exam schedules
- Project deadlines
- College notices
- Student guidelines

### 🏢 Organizations

- Internal policies
- Project documentation
- Meeting documents
- Company notices
- Operational guidelines

### ⚖️ Compliance

- Policy documents
- Rules and regulations
- Version comparison
- Requirement verification

### 🏥 Healthcare Documentation

- Document retrieval
- Policy investigation
- Information comparison

### 🔬 Research

- Multi-document literature investigation
- Evidence retrieval
- Comparing statements across documents

### 🏛️ Government / Public Information

- Notices
- Circulars
- Public announcements
- Policy updates

---

# 🧪 Example Use Case

Suppose a student uploads:

```text
academic_calendar_2026_2027.pdf
college_notice.txt
updated_college_notice.txt
project_guidelines.txt
```

The student asks:

```text
What is the current project report submission deadline?
```

DocuLens searches the entire collection.

It may find:

```text
Academic Calendar:
15 October 2026
```

and:

```text
Updated Notice:
20 October 2026
```

Instead of returning only one date, DocuLens identifies the contradiction and explains that the latest official notice should be verified.

This demonstrates the core **document investigation** capability.

---

# 📊 Hackathon Alignment

DocuLens AI is designed to address the major evaluation areas of the Intelligent Document Investigator problem.

| Evaluation Area | DocuLens AI |
|---|---|
| Innovation | Multi-document investigation + contradiction detection |
| Problem Understanding | Solves scattered information retrieval |
| Functionality | Upload → Process → Ask → Retrieve → Answer |
| Evidence | Source and page references |
| Conflict Detection | Cross-document claim comparison |
| Uncertainty | High / Moderate / Low confidence |
| UI/UX | Streamlit investigation dashboard |
| Impact | Faster and more transparent document analysis |
| Presentation | Visual evidence and contradiction workflow |

---

# 🧩 Project Structure

```text
DocuLens-AI/
│
├── app.py
│
├── requirements.txt
│
├── README.md
│
├── .gitignore
│
└── data/
    └── uploads/
```

Local uploaded documents and environment secrets are excluded from version control.

---

# 🚀 Installation

## 1. Clone the repository

```bash
git clone https://github.com/Indraja-A/DocuLens-AI.git
```

## 2. Enter the project

```bash
cd DocuLens-AI
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

# 🔑 Environment Configuration

Create a `.env` file in the project root.

```env
GEMINI_API_KEY=YOUR_API_KEY
GEMINI_MODEL=gemini-3.8-flash
GEMINI_EMBEDDING_MODEL=gemini-embedding-2
```

Replace:

```text
YOUR_API_KEY
```

with your Gemini API key.

### ⚠️ Security

Never commit `.env` to GitHub.

The project `.gitignore` excludes environment files containing API credentials.

---

# ▶️ Run the Application

Start Streamlit:

```bash
streamlit run app.py
```

The application will open in your browser.

---

# 🧪 Example Questions

After uploading documents, users can ask:

```text
What is the project report submission date?
```

```text
What are all the project-related deadlines?
```

```text
When is Project Review I?
```

```text
When is Project Review II?
```

```text
What does this document say about project submission?
```

```text
What does this image say?
```

```text
What is the current deadline according to all documents?
```

```text
What information is available about the project?
```

For information not contained in the uploaded documents, DocuLens is designed to communicate that sufficient evidence was not found.

---

# 🔐 Security and Privacy

DocuLens uses environment variables for API credentials.

The repository excludes:

```text
.env
.env.*
data/
data/uploads/
.streamlit/secrets.toml
```

API keys should never be hard-coded into the application or committed to a public repository.

---

# 📈 Future Improvements

Possible future improvements include:

- Persistent investigation history
- Larger-scale vector databases
- More advanced document version tracking
- Automatic source prioritization
- Timeline-based document comparison
- Better table understanding
- Multilingual document investigation
- Authentication and user accounts
- Cloud deployment
- Advanced citation highlighting
- Document-level change tracking
- Explainable contradiction graphs

---

# 🏆 Hackathon Vision

DocuLens AI aims to make document investigation:

### Faster
Reduce the time spent manually searching through large document collections.

### Transparent
Show where the answer came from.

### Evidence-Driven
Ground responses in retrieved document content.

### Conflict-Aware
Identify when different documents disagree.

### Uncertainty-Aware
Communicate when evidence is weak or incomplete.

---

# 👩‍💻 Developer

## Indraja A

GitHub:

https://github.com/Indraja-A

---

# ⭐ Project Summary

DocuLens AI is more than a document chatbot.

It is an:

> **Intelligent Document Investigator**

that combines:

```text
Multi-Format Processing
        +
Semantic Retrieval
        +
Evidence-Based Q&A
        +
Source References
        +
Contradiction Detection
        +
Confidence Scoring
        +
Uncertainty Handling
```

The goal is simple:

> **Help users find the right information, understand the evidence behind it, and know when the documents disagree.**

---

# 🔎 DocuLens AI

### Intelligent Document Investigation

**Upload → Investigate → See Evidence → Detect Conflicts → Understand Uncertainty**

---