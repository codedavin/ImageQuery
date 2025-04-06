# config.py
import torch

# Global Device Setting
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ChromaDB settings
CHROMA_PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "image_captions"