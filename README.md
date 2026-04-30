<div align="center">

# 🔮 PRISM 
**Palm Reasoning Interpretation System with Models**

An advanced Retrieval-Augmented Generation (RAG) system dedicated to the art and science of palmistry, powered by Google's Gemini Models and ChromaDB vector search.

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Gemini](https://img.shields.io/badge/Google%20Gemini-8E75B2?style=for-the-badge&logo=google&logoColor=white)](https://deepmind.google/technologies/gemini/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-FF6B6B?style=for-the-badge&logo=database&logoColor=white)](https://trychroma.com/)

</div>

---

> ✨ **PRISM** blends ancient wisdom with cutting-edge artificial intelligence, offering deep insights through a tailored RAG pipeline.

## 🌟 Key Features

- **🧠 Advanced RAG Pipeline**: Combines context-aware document retrieval (via ChromaDB) with the reasoning capabilities of Gemini models (`gemini-2.0-flash-lite`).
- **⚡ Optimized Performance**: Global caching and efficient vector storage ensure low-latency responses.
- **💬 Interactive UI**: A sleek frontend featuring consistent markdown formatting and an intuitive chat interface.
- **📜 Persistent History**: Manage and review your past palmistry readings seamlessly through the chat history sidebar.
- **🧪 Modular Design**: Available as both an experimental Jupyter Notebook and a fully-fledged Web Application.

---

## 📂 Project Structure

PRISM is split into two main environments to cater to both exploration and production use cases:

### 1. `Web App Implementation/`
The complete, production-ready full-stack application.
- **Frontend**: Interactive user interface with chat persistence.
- **Backend**: API routing, RAG pipeline execution, and model management.

### 2. `Jupyter Notebook Implementation/`
A clean, step-by-step interactive environment.
- Perfect for learning, debugging, and understanding the RAG process (data loading, chunking, embedding, vector storage, and inference).

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- A Google Gemini API Key

### 🌐 Running the Web App

1. Navigate to the Web App directory:
   ```bash
   cd "Web App Implementation"
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv env
   # On Windows:
   .\env\Scripts\activate
   # On macOS/Linux:
   source env/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Set up your environment variables by editing the `.env` file and adding your API keys.
5. Start the backend and frontend servers as defined in the respective directories.

### 📓 Running the Notebook

1. Navigate to the Jupyter Notebook directory:
   ```bash
   cd "Jupyter Notebook Implementation"
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up your `.env` file with your Gemini API key.
4. Launch Jupyter Notebook or Jupyter Lab and open the notebook files to run the RAG pipeline step-by-step.

---

<div align="center">
  <br>
  <i>Crafted with precision to unlock the stories held in our hands.</i>
</div>
