import json
import os
import warnings
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from chromadb import PersistentClient
from chromadb.config import Settings
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader as PDFReader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import google.generativeai as genai

try:
    import posthog

    def _noop_posthog_capture(*args, **kwargs):
        return None

    posthog.capture = _noop_posthog_capture
except ImportError:
    pass

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*LangChainDeprecationWarning.*")

DATA_DIR = Path("../data")
DEFAULT_CHROMA_PATH = Path("./chroma_store")
FALLBACK_CHROMA_PATH = Path("./chroma_store_rebuilt_1777132406")
CHROMA_COLLECTION_NAME = "palm_books_chunks"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
GOOGLE_GEMINI_MODEL = "models/gemini-flash-latest"
GOOGLE_GEMINI_FALLBACK_MODEL = "models/gemini-2.5-flash-lite"

BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"
if not ENV_PATH.exists():
    ENV_PATH = BASE_DIR.parent / ".env"
if not ENV_PATH.exists():
    ENV_PATH = BASE_DIR.parent.parent / ".env"
if ENV_PATH.exists():
    load_dotenv(ENV_PATH)
else:
    load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
genai.configure(api_key=GOOGLE_API_KEY)

GEMINI_PALM_PROMPT = """A user uploads a right-hand palm photo. You are a technical vision assistant for a palmistry system.
Analyze the image carefully and return ONLY valid JSON in the exact schema below. Do not add any extra text before or after the JSON.

Schema:
{
  "life_line": {"length": "short|medium|long|unknown", "depth": "deep|faint|broken|unknown", "notes": "..."},
  "head_line": {"length": "short|medium|long|unknown", "quality": "straight|curved|broken|unknown", "notes": "..."},
  "heart_line": {"length": "short|medium|long|unknown", "depth": "deep|faint|broken|unknown", "notes": "..."},
  "fate_line": {"shape": "straight|wavy|broken|missing|unknown", "strength": "strong|weak|faint|unknown", "notes": "..."},
  "mounts": {"venus": "raised|flat|unknown", "moon": "raised|flat|unknown", "jupiter": "raised|flat|unknown", "saturn": "raised|flat|unknown", "mercury": "raised|flat|unknown", "apollo": "raised|flat|unknown", "mars": "raised|flat|unknown"},
  "hand_shape": "square|oval|triangular|spatulate|unknown",
  "summary": "A short, factual description of visible line and mount patterns."
}

If the image is not available to you, return {"error": "IMAGE_NOT_AVAILABLE"}.
Focus only on observable palm features, not on fortune telling."""

# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------

def show_header(title: str) -> None:
    separator = "=" * 72
    print(f"\n{separator}\n{title}\n{separator}")


def show_kv(key: str, value: Any) -> None:
    print(f"{key:<28}: {value}")


def text_len(text: str) -> int:
    return len((text or "").strip())

# -----------------------------------------------------------------------------
# PDF loading and chunking
# -----------------------------------------------------------------------------

def load_pdf_documents(source_dir: Path) -> tuple[List[Any], List[tuple[str, int, int, int, str]], list[tuple[str, str]]]:
    """Load PDFs and collect page metadata."""
    pdf_paths = sorted(source_dir.glob("*.pdf")) if source_dir.exists() else []
    documents: List[Any] = []
    file_stats: List[tuple[str, int, int, int, str]] = []
    failed_files: list[tuple[str, str]] = []

    for pdf_path in pdf_paths:
        try:
            docs = PDFReader(str(pdf_path)).load()
        except Exception as err:
            failed_files.append((pdf_path.name, str(err)))
            continue

        for doc in docs:
            doc.metadata["source_file"] = pdf_path.name

        pages = len(docs)
        non_empty_pages = sum(1 for d in docs if text_len(d.page_content) > 0)
        char_count = sum(text_len(d.page_content) for d in docs)
        status = "TEXT_OK" if char_count > 10 else "SCAN_OR_IMAGE_PDF"

        documents.extend(docs)
        file_stats.append((pdf_path.name, pages, non_empty_pages, char_count, status))

    return documents, file_stats, failed_files


def split_documents_into_chunks(documents: List[Any], chunk_size: int, chunk_overlap: int) -> List[Any]:
    """Split documents into overlapping text chunks for retrieval."""
    docs_to_split = [doc for doc in documents if text_len(doc.page_content) > 0]
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunked_documents = splitter.split_documents(docs_to_split)
    for idx, chunk in enumerate(chunked_documents):
        chunk.metadata.setdefault("source_file", chunk.metadata.get("source", "unknown"))
        chunk.metadata["chunk_index"] = idx

    return chunked_documents


def embed_documents(chunks: List[Any], model_name: str) -> List[list[float]]:
    """Compute embeddings for each chunk using HuggingFace."""
    embedding_model = HuggingFaceEmbeddings(model_name=model_name)
    texts = [chunk.page_content for chunk in chunks]
    embeddings = embedding_model.embed_documents(texts)
    return [list(vec) for vec in embeddings]

# -----------------------------------------------------------------------------
# Chroma store management
# -----------------------------------------------------------------------------

def create_chroma_collection(store_path: Path, collection_name: str):
    client = PersistentClient(
        path=str(store_path),
        settings=Settings(anonymized_telemetry=False),
    )
    return client.get_or_create_collection(name=collection_name)


def upsert_chunks_to_chroma(
    store_path: Path,
    collection_name: str,
    chunked_documents: List[Any],
    chunk_embeddings: List[list[float]],
):
    """Store chunk texts and embeddings in ChromaDB."""
    chunk_count = len(chunked_documents)
    ids = [f"chunk-{i}" for i in range(chunk_count)]
    documents = [doc.page_content for doc in chunked_documents]
    metadatas = [
        {
            "source_file": str(doc.metadata.get("source_file", "unknown")),
            "page": doc.metadata.get("page", -1),
            "chunk_index": doc.metadata.get("chunk_index", -1),
        }
        for doc in chunked_documents
    ]

    if not (len(ids) == len(documents) == len(metadatas) == len(chunk_embeddings)):
        raise ValueError("Chunk data lengths do not match before upsert.")

    collection = create_chroma_collection(store_path, collection_name)
    try:
        collection.upsert(ids=ids, documents=documents, metadatas=metadatas, embeddings=chunk_embeddings)
    except Exception as exc:
        if store_path.exists():
            fallback_path = Path(f"./chroma_store_rebuilt_{int(datetime.now().timestamp())}")
            show_header("Cell 5 - ChromaDB Recovery")
            print("Detected a corrupted ChromaDB store. Using a fresh store at:")
            print(f"  {fallback_path}")
            collection = create_chroma_collection(fallback_path, collection_name)
            collection.upsert(ids=ids, documents=documents, metadatas=metadatas, embeddings=chunk_embeddings)
            return collection, fallback_path
        raise

    return collection, store_path

# -----------------------------------------------------------------------------
# Gemini request helpers
# -----------------------------------------------------------------------------

def _generate_with_gemini_model(
    model_name: str,
    prompt: str,
    temperature: float,
    max_output_tokens: int,
    image_path: str | None = None,
) -> str:
    """Call the Gemini model and return the generated text."""
    model = genai.GenerativeModel(model_name)
    contents = [prompt]

    if image_path and Path(image_path).exists():
        try:
            from PIL import Image

            image = Image.open(image_path).convert("RGB")
            image_blob = genai.types.content_types.image_to_blob(image)
            contents.append(image_blob)
        except Exception as exc:
            print("Failed to attach the palm image to Gemini request:", type(exc).__name__, exc)

    response = model.generate_content(
        genai.types.content_types.to_contents(contents),
        generation_config=genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        ),
    )

    if hasattr(response, "text") and response.text:
        return response.text.strip()

    if hasattr(response, "content") and response.content:
        first_chunk = response.content[0]
        if hasattr(first_chunk, "text") and first_chunk.text:
            return first_chunk.text.strip()

    return ""


def send_gemini_stable_request(
    prompt: str,
    temperature: float = 0.7,
    max_output_tokens: int = 350,
    image_path: str | None = None,
) -> str:
    """Send a prompt and optional image to Gemini with fallback handling."""
    fallback_response = (
        "The palm image shows a deep Life Line, a faint and slightly curved Head Line, "
        "a short Heart Line, a weak Fate Line, and a raised mount under the index finger."
    )

    if not GOOGLE_API_KEY:
        print("GOOGLE_API_KEY is not configured. Using a sample palm observation for validation.")
        return fallback_response

    try:
        result = _generate_with_gemini_model(
            GOOGLE_GEMINI_MODEL,
            prompt,
            temperature,
            max_output_tokens,
            image_path=image_path,
        )
        if result:
            return result
        raise RuntimeError("Gemini returned an empty response.")
    except Exception as exc:
        print("Primary Gemini model failed:", type(exc).__name__, exc)
        if GOOGLE_GEMINI_FALLBACK_MODEL:
            try:
                print(f"Retrying with fallback Gemini model: {GOOGLE_GEMINI_FALLBACK_MODEL}")
                result = _generate_with_gemini_model(
                    GOOGLE_GEMINI_FALLBACK_MODEL,
                    prompt,
                    temperature,
                    max_output_tokens,
                    image_path=image_path,
                )
                if result:
                    return result
            except Exception as exc2:
                print("Fallback Gemini model also failed:", type(exc2).__name__, exc2)

    print("Falling back to sample palm observation.")
    return fallback_response


def send_gemini_palm_description_request(image_path: str, prompt: str) -> str:
    """Generate a structured palm description using Gemini."""
    if not GOOGLE_API_KEY:
        return send_gemini_stable_request(prompt, temperature=0.7, max_output_tokens=350)

    if not Path(image_path).exists():
        print("Palm image path does not exist; generating palm description from prompt alone.")
        return send_gemini_stable_request(prompt, temperature=0.7, max_output_tokens=350)

    return send_gemini_stable_request(
        prompt,
        temperature=0.7,
        max_output_tokens=350,
        image_path=str(image_path),
    )

# -----------------------------------------------------------------------------
# Retrieval and RAG logic
# -----------------------------------------------------------------------------

def build_chroma_query(description: str) -> str:
    """Turn a structured palm description into a focused Chroma query."""
    if not description or not description.strip():
        return "palmistry career meaning life line head line heart line fate line hand shape mount of Venus"

    normalized = " ".join(description.split())
    query_tokens: list[str] = []

    try:
        json_start = normalized.index("{")
        json_text = normalized[json_start:]
        parsed = json.loads(json_text)
    except Exception:
        parsed = None

    if isinstance(parsed, dict):
        for key in ("life_line", "head_line", "heart_line", "fate_line", "hand_shape"):
            value = parsed.get(key)
            if isinstance(value, dict):
                query_tokens.extend(
                    str(item).strip()
                    for item in value.values()
                    if item and str(item).strip().lower() != "unknown"
                )
            elif value and str(value).strip().lower() != "unknown":
                query_tokens.append(str(value).strip())

        mounts = parsed.get("mounts")
        if isinstance(mounts, dict):
            for mount_name, mount_value in mounts.items():
                if mount_value and str(mount_value).strip().lower() != "unknown":
                    query_tokens.append(f"{mount_name} mount {mount_value}")

        summary = parsed.get("summary")
        if summary and str(summary).strip().lower() != "unknown":
            query_tokens.append(str(summary).strip())

    if not query_tokens:
        cleaned = normalized.replace("{", " ").replace("}", " ").replace('"', " ")
        cleaned = " ".join(cleaned.split())
        query_tokens = [token for token in cleaned.split() if len(token) > 2]

    query_text = " ".join(query_tokens).strip()
    if not query_text:
        query_text = "palmistry career meaning life line head line heart line fate line hand shape mount of Venus"
    elif "career" not in query_text.lower():
        query_text = f"{query_text} career meaning"

    return query_text


def query_chroma_by_description(description: str, n_results: int = 5) -> list[Dict[str, Any]]:
    """Search ChromaDB for the most relevant chunks matching the description."""
    query_text = build_chroma_query(description)
    client = PersistentClient(path=str(DEFAULT_CHROMA_PATH), settings=Settings(anonymized_telemetry=False))
    collection = client.get_or_create_collection(name=CHROMA_COLLECTION_NAME)

    try:
        query_result = collection.query(
            query_texts=[query_text],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )
    except Exception as exc:
        print("ChromaDB query failed:", exc)
        return []

    if isinstance(query_result, dict):
        documents = query_result.get("documents", [[]])[0]
        metadatas = query_result.get("metadatas", [[]])[0]
        distances = query_result.get("distances", [[]])[0]
    else:
        documents = query_result.documents[0]
        metadatas = query_result.metadatas[0]
        distances = query_result.distances[0]

    hits: list[Dict[str, Any]] = []
    for idx, doc in enumerate(documents):
        hits.append(
            {
                "rank": idx + 1,
                "document": doc,
                "metadata": metadatas[idx],
                "distance": distances[idx],
                "query_text": query_text,
            }
        )

    return hits


def build_final_system_prompt(description: str, question: str, hits: List[Dict[str, Any]]) -> str:
    """Construct the final prompt to send to Gemini using retrieved knowledge."""
    excerpt_lines = []
    for hit in hits:
        excerpt_lines.append(
            f"### Source {hit['rank']}: {hit['metadata'].get('source_file', 'unknown')}\n"
            f"Chunk {hit['metadata'].get('chunk_index')}\n"
            f"{hit['document']}\n"
        )

    return (
        "You are an expert Palmist. Using the visual analysis below and the retrieved book excerpts, provide a detailed, respectful, and evidence-based reading in markdown format. "
        "Answer the user's question directly, using the exact palm features described. "
        "Include a clear reading and a separate reasoning section that explains how each palm feature supports the conclusion.\n\n"
        "### User Question\n"
        f"{question}\n\n"
        "### Visual Analysis\n"
        f"{description}\n\n"
        "### Retrieved References\n"
        f"{chr(10).join(excerpt_lines)}\n\n"
        "Your final response must use markdown with headings, paragraphs, and bullet points. Always include:\n"
        "- A short introduction referencing the right-hand palm and the question.\n"
        "- Specific observations about named lines and mounts.\n"
        "- A clear interpretation of what those observations mean for the user.\n"
        "- A 'Reasoning behind this reading' section that maps palm features to conclusions.\n\n"
        "For marriage timing questions, mention the marriage line, heart line, fate line, and overall relationship stability.\n"
        "For financial stability questions, mention the fate line, sun line, head line, and relevant mounts such as Venus.\n"
        "If a feature is not visible or not clearly described, say so honestly. Do not invent palm features beyond what is provided."
    )


def call_final_llm(prompt: str) -> str:
    """Generate the final answer with Gemini Stable."""
    if not GOOGLE_API_KEY:
        print("GOOGLE_API_KEY is not configured. Final answer will use a fallback placeholder.")
        return (
            "Final answer generation requires GOOGLE_API_KEY. "
            "Set the environment variable in your .env file and rerun this cell."
        )

    return send_gemini_stable_request(prompt, temperature=0.7, max_output_tokens=750)


def run_gemini_rag_pipeline(image_path: str, user_question: str) -> Dict[str, Any]:
    """Execute the full RAG pipeline for a palm image and question."""
    technical_description = send_gemini_palm_description_request(image_path, GEMINI_PALM_PROMPT)
    search_hits = query_chroma_by_description(technical_description, n_results=5)
    final_prompt = build_final_system_prompt(technical_description, user_question, search_hits)
    final_answer = call_final_llm(final_prompt)

    return {
        "visual_description": technical_description,
        "chroma_query": build_chroma_query(technical_description),
        "search_hits": search_hits,
        "final_prompt": final_prompt,
        "final_answer": final_answer,
    }


def print_pipeline_results(result: Dict[str, Any], user_question: str) -> None:
    """Print the main pipeline outputs in a human-readable format."""
    print("\n" + "=" * 72)
    print("QUESTION")
    print(user_question)
    print("\n" + "=" * 72)
    print("VISUAL DESCRIPTION")
    print(result.get("visual_description", "No visual description returned."))
    print("\n" + "=" * 72)
    print("CHROMA QUERY")
    print(result.get("chroma_query", "No query generated."))
    print("\n" + "=" * 72)
    print("RETRIEVED CHUNKS")
    if result.get("search_hits"):
        for hit in result["search_hits"]:
            source_file = hit["metadata"].get("source_file", "unknown")
            chunk_index = hit["metadata"].get("chunk_index")
            distance = hit.get("distance")
            print(f"- [{hit['rank']}] {source_file} | chunk {chunk_index} | distance {distance:.4f}")
    else:
        print("No matching book chunks were found.")

    print("\n" + "=" * 72)
    print("FINAL ANSWER")
    print(result.get("final_answer", "No final answer was generated."))
    print("\n" + "=" * 72)


def run_demo(image_path: Path, user_question: str, note: str) -> None:
    """Run a single RAG demo and print the outputs."""
    if not image_path.exists():
        raise FileNotFoundError(
            f"The image path '{image_path}' does not exist. Please upload the right-hand palm image and try again."
        )

    print(f"Using image_path: {image_path}")
    print(f"Using user_question: {user_question}\n")

    result = run_gemini_rag_pipeline(str(image_path), user_question)
    print_pipeline_results(result, user_question)
    print(note)


if __name__ == "__main__":
    run_demo(
        Path(r"D:/Web Projects/PalmRAG/PRISM-Palm-Reasoning-Interpretation-System-with-Models/research/test data/2.jpeg"),
        "What do the lines on my palm suggest about my future career growth and opportunities?",
        "NOTE: This tool is designed to analyze the right-hand palm image and answer based on palmistry references. "
        "The question is focused on career growth and opportunities.",
    )

    run_demo(
        Path(r"D:/Web Projects/PalmRAG/PRISM-Palm-Reasoning-Interpretation-System-with-Models/research/test data/3.jpg"),
        "What does my palm say about my marriage—when is it likely to happen?",
        "NOTE: This tool is designed to analyze the right-hand palm image and answer based on palmistry references. "
        "The question is focused on marriage timing and related palm lines.",
    )
