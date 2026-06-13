<div align="center">

<img src="https://img.shields.io/badge/DocuMind-AI-7c6aff?style=for-the-badge&logo=googlegemini&logoColor=white" />

# 🧠 DocuMind AI
### Chat with your documents

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=flat-square&logo=langchain&logoColor=white)](https://langchain.com)
[![Gemini](https://img.shields.io/badge/Gemini_API-4285F4?style=flat-square&logo=google&logoColor=white)](https://ai.google.dev)
[![License: MIT](https://img.shields.io/badge/License-MIT-a78bfa?style=flat-square)](LICENSE)
[![Live Demo](https://img.shields.io/badge/Live_Demo-Hugging_Face-orange?style=flat-square&logo=huggingface&logoColor=white)](https://huggingface.co/spaces/syedfarhanali99/DocuMind-ai)

<br/>

DocuMind AI Demo <img width="1880" height="932" alt="Rag-ChatBot Thumbnail" src="https://github.com/user-attachments/assets/3e504b80-a0af-464b-a9ff-3c90f99b0f41" />


</div>

---

## 📌 What is DocuMind AI?

DocuMind AI lets you upload any document like PDF, DOCX, or TXT simultaniously and have a real multi-turn conversation with it. Ask questions, request summaries, dig into specific sections, and get answers grounded entirely in your document's content.

Built with a proper RAG (Retrieval Augmented Generation) pipeline:
- **Embeddings run 100% locally** via FastEmbed — no API calls, no rate limits, no quota issues
- **Gemini is used only for answering** — one API call per question, nothing more
- **FAISS handles vector search** — fast similarity retrieval entirely on your machine

> 💡 This architecture came from a real lesson: using Google's embedding API burned the entire daily free-tier quota just uploading a single PDF. Switching embeddings to local fixed everything.

---

## ✨ Features

| Feature | Details |
|---|---|
| 📁 Multi-format upload | PDF, DOCX, TXT — up to 5 files at once |
| 🧠 Local embeddings | FastEmbed (ONNX) — no PyTorch, no API calls |
| ⚡ Vector search | FAISS similarity search across all uploaded documents |
| 💬 Conversational memory | Full multi-turn chat with context across questions |
| 🌊 Streaming responses | Token-by-token generation like ChatGPT |
| 🌙 Dark / Light mode | Toggle in Settings panel |
| ⚙️ Settings panel | API key, model selector, search depth slider |
| 🛡️ Clean error handling | No raw tracebacks — all errors shown as friendly messages |
| 🚫 No data storage | Documents never leave your session |

---

## 🏗️ Architecture

```
Your Document (PDF / DOCX / TXT)
         │
         ▼
   Text Extraction
   (PyPDF / Docx2txt / TextLoader)
         │
         ▼
   Chunking (1000 chars, 200 overlap)
   RecursiveCharacterTextSplitter
         │
         ▼
   Local Embeddings ── FastEmbed ONNX
   (no API call)           │
         │                 │
         ▼                 ▼
   FAISS Vector Store ◄────┘
         │
    [User asks a question]
         │
         ▼
   Similarity Search (top-k chunks)
         │
         ▼
   LangChain RAG Chain
   + Chat History (HumanMessage / AIMessage)
         │
         ▼
   Gemini API (answer generation only)
         │
         ▼
   Streamed answer → UI
```

---

## 🧰 Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| RAG Framework | LangChain |
| LLM | Google Gemini (2.0 Flash / 2.5 Flash) |
| Embeddings | FastEmbed — `BAAI/bge-small-en-v1.5` |
| Vector Store | FAISS (CPU) |
| PDF Loader | PyPDF |
| DOCX Loader | Docx2txt |
| Language | Python 3.10+ |

---

## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/DocuMind-ai.git
cd documind-ai
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Get a Gemini API key

Go to [Google AI Studio](https://aistudio.google.com/app/apikey) → Create API key → Copy it.
No billing required — the free tier is enough.

### 4. Run the app

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

### 5. Use it

1. Click **⚙️ Settings** → paste your Gemini API key
2. Upload your documents (PDF, DOCX, or TXT)
3. Click **⚡ Process Documents**
4. Start chatting

---

## 📦 Requirements

```
streamlit
langchain
langchain-google-genai
langchain-community
langchain-text-splitters
fastembed
faiss-cpu
pypdf
docx2txt
```

Install all at once:

```bash
pip install -r requirements.txt
```

---

## 💡 Key Technical Decisions

**Why local embeddings?**
Using Google's embedding API on a free tier burns your entire daily quota just processing a single PDF (300+ chunks = 300+ API calls in seconds). FastEmbed runs the embedding model locally via ONNX Runtime — no network calls, no rate limits, ~20MB model, faster on CPU than sentence-transformers.

**Why FAISS over a cloud vector DB?**
For a single-session app with up to 5 documents, FAISS in memory is instant and needs zero infrastructure. No Pinecone account, no API key, no latency.

**Why LangChain LCEL over RetrievalQA?**
`RetrievalQA` is deprecated in newer LangChain versions. The LCEL pipe syntax (`chain = prompt | model | parser`) is the current standard and gives full control over the retrieval and generation steps.

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you'd like to change.

---

## 👤 Author

**Farhan Syed Ali**
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=flat-square&logo=linkedin&logoColor=white)](www.linkedin.com/in/syed-farhan-ali-shah-ab2309287)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/farhansyedAli)
[![Hugging Face](https://img.shields.io/badge/Hugging_Face-FF9D00?style=flat-square&logo=huggingface&logoColor=white)](https://huggingface.co/syedfarhanali99)

---

## 📝 License

This project is licensed under the [MIT License](LICENSE).

---

<div align="center">
<sub>Built with 🧠 by Farhan Syed Ali · Embeddings run locally · Answers powered by Gemini</sub>
</div>
