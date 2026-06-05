## How to run the application (step-by-step)

### 1) Go to the project folder

```bash
cd rag_dsa_chatbot
```

### 2) Create and activate a virtual environment

Important (macOS):
- This project requires **PyTorch** (via `sentence-transformers`), and PyTorch wheels are often **not available for Python 3.13** yet.
- Use **Python 3.11 or 3.12** for the smoothest installation.

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

If `python3` on your machine points to Python 3.13 (common with some Conda installs), install a supported Python and create the venv with it. For example:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python --version
```

Windows (PowerShell):

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3) Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4) Configure environment variables (LLM)

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

Then edit `.env` and set:

- **`GEMINI_API_KEY`**: your Gemini API key
- **`GEMINI_MODEL`** (optional): defaults to `gemini-1.5-flash`

Optional fallback:
- **`OPENAI_API_KEY`** and **`OPENAI_MODEL`** (if you want to use OpenAI instead)

Important:
- If you leave **both** `GEMINI_API_KEY` and `OPENAI_API_KEY` empty, the app will still build embeddings + FAISS and show retrieval chunks, but it will **refuse to answer** (strict anti-hallucination mode).

### 5) Start Streamlit

```bash
streamlit run app.py
```

### 6) Use the app

- In the **sidebar**, upload one or more files (PDF/DOCX/TXT/MD).
- Click **Build / Load Vector DB**.
  - This will:
    - extract text
    - chunk (`chunk_size=500`, `chunk_overlap=100`)
    - embed (MiniLM)
    - persist FAISS to `vectorstore/`
    - include built-in DSA knowledge from `data/dsa_builtin_kb.md`
- Ask questions in the chat.
- See:
  - retrieved/reranked chunks in the **Retrieval viewer**
  - file/page/chunk citations per source

### Troubleshooting

- **“Not built yet”**: upload and click **Build / Load Vector DB**.
- **Model downloads are slow the first time**: SentenceTransformers and CrossEncoder models download on first run and are cached locally.
- **PDF text looks empty**: scanned PDFs need OCR. Convert to text PDFs or upload a text-based source.

### Production tips

- Put large documents in smaller sets to speed up indexing.
- Persisted FAISS index lives in `vectorstore/`. Delete it if you want to rebuild from scratch.

