# src/recommender.py
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


def pick_device():
    if torch.backends.mps.is_available(): return "mps"
    if torch.cuda.is_available(): return "cuda"
    return "cpu"


class MusicRecommender:
    def __init__(self):
        self.df = pd.read_parquet(TRACKS_PATH)
        self.index = faiss.read_index(str(INDEX_PATH))
        self.id_map = json.loads(IDMAP_PATH.read_text())

        device = pick_device()
        self.embedder = SentenceTransformer(EMBED_MODEL_NAME, device=device)

        # fast lookup: track_id -> row index
        self._row_by_id = {tid: i for i, tid in enumerate(self.df["track_id"].astype(str).tolist())}

    def recommend(self, query: str, k: int = 10):
        qvec = self.embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
        scores, idxs = self.index.search(qvec, k)
        idxs = idxs[0].tolist()
        scores = scores[0].tolist()

        recs = []
        for faiss_i, score in zip(idxs, scores):
            tid = self.id_map[faiss_i]
            row = self.df.iloc[self._row_by_id[tid]]
            recs.append({
                "track_id": tid,
                "title": row["title"],
                "artist": row["artist"],
                "tags": row["tags"],
                "score": float(score),
            })
        return recs


def main():
    engine = MusicRecommender()
    query = "calm instrumental ambient music for studying"
    recs = engine.recommend(query, k=10)

    print("Query:", query)
    for i, r in enumerate(recs, 1):
        print(f"{i:02d}. {r['title']} — {r['artist']} | score={r['score']:.3f} | tags={r['tags'][:6]}")


if __name__ == "__main__":
    main()