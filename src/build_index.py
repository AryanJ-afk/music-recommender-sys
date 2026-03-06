# src/build_index.py
import json
from pathlib import Path

import numpy as np
import pandas as pd
import faiss
import torch
from sentence_transformers import SentenceTransformer


TRACKS_PATH = Path("data/tracks.parquet")
INDEX_PATH = Path("data/index.faiss")
IDMAP_PATH = Path("data/id_map.json")

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 256  # if memory issues, drop to 64 or 128


def pick_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main():
    if not TRACKS_PATH.exists():
        raise FileNotFoundError(f"Missing {TRACKS_PATH}. Run build_catalog.py first.")

    df = pd.read_parquet(TRACKS_PATH)
    if "text" not in df.columns or "track_id" not in df.columns:
        raise ValueError("tracks.parquet must contain columns: track_id, text")

    texts = df["text"].astype(str).tolist()
    track_ids = df["track_id"].astype(str).tolist()

    device = pick_device()
    print("Device:", device)
    print("Loading embedding model:", EMBED_MODEL_NAME)

    model = SentenceTransformer(EMBED_MODEL_NAME, device=device)

    # Encode -> numpy float32
    print(f"Encoding {len(texts)} tracks...")
    emb = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # cosine similarity via inner product
    ).astype(np.float32)

    dim = emb.shape[1]
    print("Embeddings shape:", emb.shape)

    # FAISS index for cosine similarity (use Inner Product on normalized vectors)
    index = faiss.IndexFlatIP(dim)
    index.add(emb)

    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))

    # Map FAISS row -> track_id (because FAISS returns indices 0..N-1)
    with open(IDMAP_PATH, "w") as f:
        json.dump(track_ids, f)

    print(f"Saved FAISS index: {INDEX_PATH} (ntotal={index.ntotal})")
    print(f"Saved id map:     {IDMAP_PATH} (len={len(track_ids)})")


if __name__ == "__main__":
    main()