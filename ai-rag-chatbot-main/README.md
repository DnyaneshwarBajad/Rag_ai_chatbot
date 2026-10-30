# 🧠 RAG-Based DSA Chatbot — AI Tutor & Interview Assistant

A Retrieval-Augmented Generation (RAG) powered chatbot designed to help developers learn **Data Structures and Algorithms (DSA)**, understand programming concepts, and prepare for technical interviews.

The application uses **Streamlit, LangChain, FAISS, Sentence Transformers, and Cross-Encoder Reranking** to retrieve relevant information from uploaded study materials and a built-in DSA knowledge base.

Unlike general-purpose AI chatbots, this application follows a strict, context-grounded answering approach to minimize hallucinations.

---

## 🚀 Key Features

- **Document-Based Question Answering:** Upload DSA notes and learning materials to ask questions based on their content.
- **Built-in DSA Knowledge Base:** Includes predefined DSA concepts and explanations through `data/dsa_builtin_kb.md`.
- **Semantic Search:** Uses Sentence Transformers to generate embeddings and retrieve semantically relevant document chunks.
- **FAISS Vector Search:** Enables efficient similarity search across indexed documents.
- **Cross-Encoder Reranking:** Reorders retrieved passages based on query-document relevance for improved retrieval quality.
- **Context-Grounded Responses:** Generates answers using retrieved content rather than unsupported external knowledge.
- **Strict Fallback Behavior:** If the available material does not support an answer, the chatbot responds with:

  > The answer is not available in the uploaded DSA material.

- **Optional OpenAI Integration:** Uses an OpenAI LLM for answer generation when an API key is configured.
- **Retrieval-Only Mode:** Without an API key, the application can still display relevant retrieved passages without generating unsupported answers.

---

## 🏗️ System Architecture

The application follows a modular RAG pipeline:

```text
             User Question
                   |
                   v
            Streamlit Interface
                   |
                   v
             Query Processing
                   |
                   v
          Embedding Generation
                   |
                   v
             FAISS Search
                   |
                   v
           Relevant Documents
                   |
                   v
          Cross-Encoder Reranker
                   |
                   v
          Top Relevant Context
                   |
                   v
          Context Validation
                   |
                   v
           OpenAI LLM (Optional)
                   |
                   v
          Grounded Answer / Fallback
```

### RAG Pipeline

1. **Document Ingestion:** Load uploaded documents and built-in DSA knowledge.
2. **Text Chunking:** Split documents into smaller chunks for efficient retrieval.
3. **Embedding Generation:** Convert text chunks into vector representations.
4. **Vector Storage:** Store embeddings in a FAISS vector index.
5. **Similarity Retrieval:** Retrieve relevant chunks based on the user's question.
6. **Reranking:** Use a Cross-Encoder to improve the relevance ordering of retrieved chunks.
7. **Context Validation:** Check whether the retrieved material sufficiently supports an answer.
8. **Answer Generation:** Generate a response using the retrieved context, or return the predefined fallback message.

---

## 🛠️ Tech Stack

| Category | Technologies |
|---|---|
| Frontend | Streamlit |
| Language | Python |
| RAG Framework | LangChain |
| Vector Database | FAISS |
| Embedding Model | `sentence-transformers/all-MiniLM-L6-v2` |
| Reranking Model | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| LLM | OpenAI API (Optional) |
| Environment Management | Python Virtual Environment, dotenv |

---

## 📁 Project Structure

```text
rag_dsa_chatbot/
│
├── app.py
├── requirements.txt
├── .env.example
├── README.md
├── RUN_APPLICATION.md
│
├── data/
│   └── dsa_builtin_kb.md
│
├── uploads/
│
├── vectorstore/
│
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
│
└── utils/
    └── helpers.py
```

### Module Responsibilities

| Module | Responsibility |
|---|---|
| `app.py` | Streamlit application and user interface |
| `loader.py` | Loads uploaded documents and built-in knowledge |
| `chunker.py` | Splits documents into manageable chunks |
| `embedder.py` | Generates text embeddings |
| `vectorstore.py` | Creates and manages the FAISS index |
| `retriever.py` | Retrieves relevant document chunks |
| `reranker.py` | Reranks retrieved results using a Cross-Encoder |
| `llm.py` | Handles optional LLM integration |
| `prompts.py` | Defines context-grounded prompt templates |
| `pipeline.py` | Coordinates the complete RAG workflow |
| `helpers.py` | Contains reusable utility functions |

---

## ⚙️ Installation and Setup

### Prerequisites

- Python 3.10+
- pip
- Git
- OpenAI API key (optional)

### 1. Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd rag_dsa_chatbot
```

### 2. Create a Virtual Environment

**Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

**Linux / macOS:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file using the example configuration:

```bash
cp .env.example .env
```

For Windows:

```bash
copy .env.example .env
```

Add your API configuration if you want to enable OpenAI-based answer generation:

```env
OPENAI_API_KEY=your_openai_api_key
```

The API key is optional. Without it, the application operates in retrieval-only mode.

### 5. Run the Application

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal, usually:

```text
http://localhost:8501
```

For detailed instructions, refer to [`RUN_APPLICATION.md`](RUN_APPLICATION.md).

---

## 🔒 Hallucination Control and Answer Grounding

The chatbot is designed around a strict retrieval-based answering policy.

- Answers must be supported by uploaded documents or the built-in DSA knowledge base.
- Retrieved context is passed to the answer-generation component.
- Unsupported questions receive a predefined fallback response.
- Without an LLM API key, the chatbot displays retrieval results instead of generating speculative answers.

**Important:** Retrieval-based grounding reduces unsupported responses, but it does not mathematically guarantee that every generated answer is correct. The validation and prompt logic should be tested against relevant and irrelevant queries.

---

## 💡 Example Use Cases

You can use the chatbot to explore topics such as:

- Arrays and Strings
- Linked Lists
- Stacks and Queues
- Binary Trees and Binary Search Trees
- Graphs and Graph Algorithms
- Dynamic Programming
- Greedy Algorithms
- Backtracking
- Sorting and Searching
- Time and Space Complexity
- Common Coding Interview Questions

The actual coverage depends on the uploaded documents and the contents of the built-in knowledge base.

---

## 🔮 Future Improvements

- [ ] Support additional document formats such as DOCX and PDF.
- [ ] Display source citations for generated answers.
- [ ] Add conversation history and chat memory.
- [ ] Add query rewriting for improved retrieval.
- [ ] Implement hybrid search using keyword and semantic retrieval.
- [ ] Add configurable chunk size and overlap.
- [ ] Add retrieval evaluation using Precision@K, Recall@K, and MRR.
- [ ] Add automated unit and integration tests.
- [ ] Deploy the application using Streamlit Community Cloud or Docker.

---

## 🤝 Contributing

Contributions, suggestions, and bug reports are welcome.

1. Fork the repository.
2. Create a feature branch.
3. Commit your changes.
4. Push your branch.
5. Open a Pull Request.

---

## 👨‍💻 Author

**Dnyaneshwar Bajad**

- GitHub: [DnyaneshwarBajad](https://github.com/DnyaneshwarBajad)
- LinkedIn: [Dnyaneshwar Bajad](https://www.linkedin.com/in/dnyaneshwar-bajad-0b4ba024a/)

---

## ⭐ Support

If you find this project useful, consider giving it a ⭐ on GitHub!

**Built to make DSA learning more interactive, searchable, and context-aware.**
