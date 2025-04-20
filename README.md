# 🚉 Welcome to Depot-D.NLR!

## Hello There!

## 📘 Depot-D.NLR

*A local RAG system for Indian Railways – powered by embeddings, driven by LLaMA.*

---

## 📌 What is Depot-D.NLR?

**Depot-D.NLR** is a lightweight, locally running Retrieval-Augmented Generation (RAG) system designed to answer **natural language queries** from **Indian Railways PDFs**. 
It uses **semantic search** and **locally hosted language models** to provide answers with contextual metadata – even offline!

---

## 💡 Why the name “Depot-D.NLR”?

- **Depot**: A nod to Indian Railways’ infrastructure – think of it as a knowledge depot for documents.
- **D.NLR**:
  - **D** stands for **Document** (or **Depot**, double pun intended).
  - **NLR** stands for **Natural Language for Railways** – the true essence of this project.

> 📢 *“Depot-D.NLR is where documents meet locomotives and language.”*

---

## 🧠 Capabilities

✅ Ingests multiple PDFs from a folder  
✅ Splits text into metadata-rich chunks (includes PDF name + page number)  
✅ Embeds text using `all-MiniLM-L6-v2` via HuggingFace  
✅ Persists vectors in ChromaDB  
✅ Retrieves using MMR (Maximal Marginal Relevance) for diversity & precision  
✅ Generates answers with **LLaMA 3.1** via **Ollama** (locally)  
✅ Displays source info alongside each answer  
✅ Works 100% offline – no APIs, no cloud, no nonsense

---

## ⚙️ Pipeline Overview

    A[PDF Folder] --> B[Text Chunking w/ Metadata];
    B --> C[HuggingFace Embeddings];
    C --> D[ChromaDB (Vector Store)];
    D --> E[MMR Retriever];
    E --> F[LLaMA 3.1 via Ollama];
    F --> G[Answer + Source Info];
