import os
from pathlib import Path

from chromadb import PersistentClient
from chromadb.config import Settings

BASE_DIR = Path(__file__).resolve().parent
CHROMA_STORE_PATH = BASE_DIR / "chroma_store"
CHROMA_COLLECTION_NAME = "palm_books_chunks"


def get_chroma_client() -> PersistentClient:
    CHROMA_STORE_PATH.mkdir(parents=True, exist_ok=True)
    return PersistentClient(path=str(CHROMA_STORE_PATH), settings=Settings(anonymized_telemetry=False))


def get_chroma_collection():
    client = get_chroma_client()
    return client.get_or_create_collection(name=CHROMA_COLLECTION_NAME)


def ensure_chroma_store() -> None:
    get_chroma_collection()
