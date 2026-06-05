# RAG-based DSA Chatbot (Streamlit + FAISS + Reranking)

This project is a production-ready starter for a **DSA Tutor + Interview Assistant** chatbot using:

- **Streamlit** UI
- **LangChain** document abstractions + FAISS vector store
- **Sentence Transformers** embeddings: `sentence-transformers/all-MiniLM-L6-v2`
- **Cross-Encoder reranking**: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Optional **OpenAI** LLM (recommended). If no API key is provided, the app stays strict and refuses to answer (no hallucinations), while still showing retrieval results.

## Key behavior (strict)

- Answers are generated **only** from:
  - user-uploaded documents
  - built-in DSA knowledge (`data/dsa_builtin_kb.md`)
- If the answer isn’t supported, the assistant replies:
  - `The answer is not available in the uploaded DSA material.`

## Project structure

```
rag_dsa_chatbot/
├── app.py
├── requirements.txt
├── .env.example
├── README.md
├── RUN_APPLICATION.md
├── data/
│   └── dsa_builtin_kb.md
├── uploads/
├── vectorstore/
├── rag/
│   ├── __init__.py
│   ├── loader.py
│   ├── chunker.py
│   ├── embedder.py
│   ├── vectorstore.py
│   ├── retriever.py
│   ├── reranker.py
│   ├── llm.py
│   ├── prompts.py
│   └── pipeline.py
└── utils/
    └── helpers.py
```

## Next

Follow the dedicated run guide: `RUN_APPLICATION.md`.

