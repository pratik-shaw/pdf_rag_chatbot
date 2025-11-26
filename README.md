# 📚 PDF RAG Chatbot - Level 1

A production-ready Retrieval-Augmented Generation (RAG) chatbot that enables intelligent question-answering on PDF documents using semantic search + OpenAI models.

Upload PDFs → System processes + indexes them → Ask questions → Get context-aware answers with sources.

---

# 🚀 Features

## 🔍 Core Capabilities
- Multi-PDF processing  
- Semantic similarity search (SentenceTransformers)
- Intelligent sentence-aware chunking  
- Persistent FAISS vector storage  
- GPT-powered answer generation  
- Source attribution with similarity scores  
- Confidence scoring  

## 💡 User Experience
- Clean Streamlit web UI  
- Real-time progress and logs  
- Chat history with context  
- Session persistence  
- System status dashboard  

---

# 🏗️ System Architecture

```
PDF RAG Chatbot
├── RAG Core
│   ├── PDF Processing
│   ├── Text Chunking
│   ├── Embeddings
│   ├── Vector Store (FAISS)
│   └── Retrieval + Answering
│
└── Streamlit App
    ├── Upload UI
    ├── Processing Dashboard
    └── Chat Interface
```

---

# 🛠️ Tech Stack

| Component | Technology |
|----------|------------|
| Language | Python 3.8+ |
| UI | Streamlit |
| Embeddings | SentenceTransformers (MiniLM-L6-v2) |
| Vector DB | FAISS IndexFlatIP |
| PDF Parsing | PyPDF2 |
| LLM | OpenAI GPT Models |
| Env Config | python-dotenv |

---

# 📐 Technical Specifications

### Embeddings
- Model: `all-MiniLM-L6-v2`
- Dim: 384  
- Similarity: Cosine (via Inner Product)

### Chunking
- Chunk size: `1000 chars`
- Overlap: `200 chars`
- Sentence-aware splitting

### Retrieval
- Top-K: 5 chunks
- Store: FAISS index + metadata `.pkl`

### Answer Generation
- Model: `gpt-3.5-turbo` (configurable)
- Temperature: `0.3`
- Max tokens: `500`

---

# 📁 Project Structure

```
pdf-rag-chatbot/
├── app.py
├── rag_system.py
├── utils/
│   ├── pdf_processor.py
│   ├── text_splitter.py
│   └── embeddings.py
├── vector_store/
│   ├── faiss_index.index
│   └── chunks_metadata.pkl
├── .env
└── requirements.txt
```

---

# ⚙️ Installation & Setup

## 1️⃣ Install Dependencies
```bash
pip install streamlit openai sentence-transformers faiss-cpu PyPDF2 python-dotenv numpy httpx
```

## 2️⃣ Create `.env`
```
OPENAI_API_KEY=your_api_key_here
CHAT_MODEL=gpt-3.5-turbo
```

## 3️⃣ Run App
```bash
streamlit run app.py
```

---

# 📖 Usage Guide

## 1. Upload PDFs
- Click "Choose PDF files"
- Upload single or multiple PDFs
- Click **Process PDFs**

## 2. System Workflow
1. Extract text  
2. Split into chunks  
3. Generate embeddings  
4. Build FAISS index  
5. Save metadata  

## 3. Ask Questions
- Type your question  
- Click **Ask Question**  
- Get:  
  ✔ Answer  
  ✔ Sources (PDF + Page)  
  ✔ Similarity score  
  ✔ Confidence  

## 4. Manage System
- Load existing vector store  
- Clear data  
- Upload new documents  

---

# 🔧 Configuration

### In `RAGSystem`:
```python
chunk_size=1000
chunk_overlap=200
top_k=5
```

### Embedding Model:
```python
model_name="all-MiniLM-L6-v2"
```

### OpenAI Model:
```
CHAT_MODEL=gpt-3.5-turbo
```

### Generation:
```python
temperature=0.3
max_tokens=500
```

---

# 🛡️ Error Handling

- Invalid PDF → gracefully skipped  
- API failure → retry mechanism  
- No text → shown to user  
- Corrupt FAISS → auto rebuild option  

---

# 🎯 Use Cases

### 👩‍🎓 Academia  
Search research papers  
Extract citations  

### ⚖ Legal  
Search clauses  
Extract definitions  

### 👨‍💻 Technical Docs  
Search APIs  
Extract code references  

### 📊 Business  
Analyze reports  
Extract data points  

---

# 🔮 Future Roadmap

### Level 2  
- Image + text multimodal  
- Structured table extraction  

### Level 3  
- Multi-tenant  
- Authentication  
- Batch ops  

### Level 4  
- Fine-tuned embeddings  
- Conversational memory  

---

# 📄 License
Open for educational and commercial use.

---

# 🤝 Contact & Support
- Check troubleshooting section  
- Verify `.env`  
- Check logs in Streamlit terminal  

---

**Version:** 1.0.0  
**Status:** Production Ready  
