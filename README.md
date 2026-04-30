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

- **🧠 Advanced RAG Pipeline** — Combines context-aware document retrieval (via ChromaDB) with the reasoning capabilities of Gemini models (`gemini-2.0-flash-lite`).
- **⚡ Optimized Performance** — Global caching and efficient vector storage ensure low-latency responses.
- **💬 Interactive UI** — A sleek frontend featuring consistent markdown formatting and an intuitive chat interface.
- **📜 Persistent History** — Manage and review your past palmistry readings seamlessly through the chat history sidebar.
- **🧪 Modular Design** — Available as both an experimental Jupyter Notebook and a fully-fledged Web Application.

---

## 📂 Project Structure

```
PRISM-Palm-Reasoning-Interpretation-System-with-Models/
│
├── 📄 README.md                          # You are here!
├── 📄 .gitignore                         # Git ignore rules
│
├── 📓 Jupyter Notebook Implementation/   # Interactive exploration environment
│   ├── 📄 .env                           # API keys (not tracked)
│   ├── 📄 requirements.txt               # Python dependencies
│   ├── 📄 LICENSE                        # Project license
│   │
│   ├── 📁 data/                          # Palmistry reference PDFs
│   │   ├── 1_A Little Bit of Palmistry.pdf
│   │   ├── 2_The Art and Science of Hand Reading.pdf
│   │   └── 3_Palmistry for All (Cheiro).pdf
│   │
│   └── 📁 research/                      # Experiments & prototyping
│       ├── 📄 trials.ipynb               # Step-by-step RAG notebook
│       ├── 📄 trials.py                  # Script version of the notebook
│       ├── 📁 chroma_store/              # Local vector database (not tracked)
│       └── 📁 test data/                 # Sample palm images for testing
│           ├── 1.jpeg
│           ├── 2.jpeg
│           └── 3.jpg
│
└── 🌐 Web App Implementation/           # Production-ready full-stack app
    ├── 📄 .env                           # API keys (not tracked)
    ├── 📄 requirements.txt               # Python dependencies
    │
    ├── 📁 data/                          # Palmistry reference PDFs
    │   ├── 1_A Little Bit of Palmistry.pdf
    │   ├── 2_The Art and Science of Hand Reading.pdf
    │   └── 3_Palmistry for All (Cheiro).pdf
    │
    ├── 📁 backend/                       # FastAPI server & RAG engine
    │   ├── 📄 main.py                    # API routes & server entry point
    │   ├── 📄 engine.py                  # Core RAG pipeline (Gemini + ChromaDB)
    │   ├── 📄 database.py                # Database helpers
    │   ├── 📄 start_backend.bat          # Windows launcher (CMD)
    │   ├── 📄 start_backend.ps1          # Windows launcher (PowerShell)
    │   ├── 📁 chroma_store/              # Local vector database (not tracked)
    │   └── 📁 uploads/                   # User-uploaded palm images (not tracked)
    │
    └── 📁 frontend/                      # Client-side UI
        ├── 📄 index.html                 # Main HTML page
        ├── 📄 script.js                  # Chat logic & API integration
        ├── 📄 style.css                  # Styling & animations
        └── 📁 assets/                    # Static images & icons
            └── palmistry-guessing-hand-lines-hand-77397685.jpg
```

> **Note:** Files marked *(not tracked)* are excluded from the repository via `.gitignore`. PDFs, images, `.env` files, and ChromaDB stores are kept local only. Folder structure is preserved using `.gitkeep` files.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- A [Google Gemini API Key](https://aistudio.google.com/app/apikey)

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
4. Create a `.env` file and add your API key:
   ```
   GOOGLE_API_KEY=your_api_key_here
   ```
5. Place the palmistry reference PDFs in the `data/` folder.
6. Start the backend server:
   ```bash
   cd backend
   uvicorn main:app --reload
   ```
7. Open `frontend/index.html` in your browser.

### 📓 Running the Notebook

1. Navigate to the Jupyter Notebook directory:
   ```bash
   cd "Jupyter Notebook Implementation"
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file with your Gemini API key.
4. Place the palmistry reference PDFs in the `data/` folder.
5. Launch Jupyter and open the notebook:
   ```bash
   jupyter notebook research/trials.ipynb
   ```

---

## ⚙️ How It Works

```
┌──────────────┐     ┌───────────────┐     ┌──────────────┐     ┌───────────────┐
│  Upload Palm │────▶│ Gemini Vision │────▶│   ChromaDB   │────▶│  Gemini LLM   │
│    Image     │     │   Analysis    │     │   Retrieval  │     │  Final Answer  │
└──────────────┘     └───────────────┘     └──────────────┘     └───────────────┘
       📷                  🔍                   📚                    🔮
   User uploads       Extracts line &       Finds matching       Generates a rich,
   a right-hand       mount features        book passages via    evidence-based
   palm photo         as structured JSON    vector similarity    palmistry reading
```

1. **Image Analysis** — Gemini Vision extracts palm features (lines, mounts, hand shape) into structured JSON.
2. **Contextual Retrieval** — The features are used to query ChromaDB, finding the most relevant passages from palmistry reference books.
3. **Augmented Generation** — Gemini combines the visual analysis with retrieved book knowledge to produce a detailed, evidence-based reading.

---

<div align="center">
  <br>
  <i>Crafted with precision to unlock the stories held in our hands.</i>
</div>
