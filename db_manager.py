# db_manager.py
import chromadb
from chromadb.config import Settings
from config import CHROMA_PERSIST_DIR, COLLECTION_NAME

def setup_collection():
    chroma_client = chromadb.Client(
        Settings(
            persist_directory=CHROMA_PERSIST_DIR,
            anonymized_telemetry=False,
        )
    )
    existing_collections = [col.name for col in chroma_client.list_collections()]
    if COLLECTION_NAME in existing_collections:
        return chroma_client.get_collection(name=COLLECTION_NAME)
    else:
        return chroma_client.create_collection(name=COLLECTION_NAME)

def store_image_caption(collection, doc_id: str, caption: str):
    embedding = [hash(caption) % 1000]  # Placeholder embedding
    metadata = {"caption": caption}
    collection.add(
        documents=[caption],
        metadatas=[metadata],
        ids=[doc_id],
        embeddings=[embedding]
    )

def query_chroma(collection, query: str, top_k: int = 1):
    results = collection.query(query_texts=[query], n_results=top_k)
    return results