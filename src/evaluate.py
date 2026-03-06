# src/evaluate.py
import ast
import json
import random
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
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def normalize_tags(x):
    """Return tags as a Python list[str] no matter how it was loaded from parquet."""
    if x is None:
        return []

    # Already a python list
    if isinstance(x, list):
        return [str(t) for t in x if str(t).strip()]

    # tuple / numpy array
    if isinstance(x, (tuple, np.ndarray)):
        return [str(t) for t in list(x) if str(t).strip()]

    # pyarrow scalar often supports .as_py()
    try:
        if hasattr(x, "as_py"):
            v = x.as_py()
            if isinstance(v, list):
                return [str(t) for t in v if str(t).strip()]
            return []
    except Exception:
        pass

    # stringified list or comma-separated string
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return []
        # "['rock','indie']"
        if s.startswith("[") and s.endswith("]"):
            try:
                v = ast.literal_eval(s)
                if isinstance(v, list):
                    return [str(t) for t in v if str(t).strip()]
                return []
            except Exception:
                return []
        # "rock, indie, 90s"
        if "," in s:
            return [t.strip() for t in s.split(",") if t.strip()]
        return [s]

    # unknown type
    return []


def tag_overlap_jaccard(tags_a, tags_b) -> float:
    a = set([str(t).lower() for t in tags_a if str(t).strip()])
    b = set([str(t).lower() for t in tags_b if str(t).strip()])
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class OfflineEngine:
    def __init__(self):
        if not TRACKS_PATH.exists():
            raise FileNotFoundError(f"Missing {TRACKS_PATH}. Run build_catalog.py first.")
        if not INDEX_PATH.exists() or not IDMAP_PATH.exists():
            raise FileNotFoundError("Missing FAISS index/id_map. Run build_index.py first.")

        self.df = pd.read_parquet(TRACKS_PATH)
        self.df["track_id"] = self.df["track_id"].astype(str)

        # Normalize tags robustly
        self.df["tags_norm"] = self.df["tags"].apply(normalize_tags)

        self.index = faiss.read_index(str(INDEX_PATH))
        self.id_map = json.loads(IDMAP_PATH.read_text())

        # lookups
        self.row_by_id = {tid: i for i, tid in enumerate(self.df["track_id"].tolist())}
        self.faiss_row_by_tid = {tid: i for i, tid in enumerate(self.id_map)}

        device = pick_device()
        self.embedder = SentenceTransformer(EMBED_MODEL_NAME, device=device)

    def _get_vec_from_faiss(self, faiss_row: int) -> np.ndarray:
        v = np.zeros((self.index.d,), dtype=np.float32)
        self.index.reconstruct(faiss_row, v)
        return v

    def profile_vec(self, liked_ids):
        vecs = []
        for tid in liked_ids:
            tid = str(tid)
            faiss_row = self.faiss_row_by_tid.get(tid)
            if faiss_row is None:
                continue
            vecs.append(self._get_vec_from_faiss(faiss_row))
        if not vecs:
            return None
        prof = np.mean(np.vstack(vecs), axis=0).astype(np.float32)
        return prof / (np.linalg.norm(prof) + 1e-12)

    def recommend_profile_only(self, liked_ids, k=10, max_per_artist=2):
        liked_set = set([str(x) for x in liked_ids])
        qvec = self.profile_vec(liked_ids)
        if qvec is None:
            return []

        retrieve_n = max(200, k * 30)
        scores, idxs = self.index.search(qvec.reshape(1, -1), retrieve_n)
        idxs = idxs[0].tolist()
        scores = scores[0].tolist()

        recs = []
        artist_count = {}

        for faiss_i, score in zip(idxs, scores):
            tid = self.id_map[faiss_i]
            if tid in liked_set:
                continue

            row = self.df.iloc[self.row_by_id[tid]]
            artist = str(row["artist"])

            artist_count[artist] = artist_count.get(artist, 0)
            if artist_count[artist] >= max_per_artist:
                continue

            recs.append((tid, float(score)))
            artist_count[artist] += 1
            if len(recs) >= k:
                break

        return recs


def main():
    random.seed(42)
    np.random.seed(42)

    engine = OfflineEngine()
    df = engine.df

    # --- sanity prints ---
    nonempty = (df["tags_norm"].apply(len) >= 1).sum()
    print("Total tracks:", len(df))
    print("Tracks with >=1 tag:", int(nonempty))
    print("Sample tags_norm:", df["tags_norm"].iloc[0][:10])

    # --- eval config ---
    N_USERS = 50
    LIKE_N = 5
    K = 10
    OVERLAP_THRESHOLD = 0.10  # Jaccard overlap >= 0.10 => "relevant" (proxy)

    candidates = df[df["tags_norm"].apply(len) >= 1]
    candidate_ids = candidates["track_id"].tolist()

    if len(candidate_ids) < LIKE_N:
        raise RuntimeError(f"Not enough candidate tracks to sample likes: {len(candidate_ids)}")

    precisions = []
    avg_overlaps = []
    diversities = []

    for _ in range(N_USERS):
        liked = random.sample(candidate_ids, LIKE_N)

        # user "profile tags" = union of tags from liked tracks
        liked_tag_set = set()
        for tid in liked:
            liked_tag_set.update([t.lower() for t in df.iloc[engine.row_by_id[tid]]["tags_norm"]])

        recs = engine.recommend_profile_only(liked, k=K, max_per_artist=2)
        if len(recs) < K:
            continue

        rel = 0
        overlaps = []
        artists = []

        for tid, _score in recs:
            row = df.iloc[engine.row_by_id[tid]]
            artists.append(str(row["artist"]))

            ov = tag_overlap_jaccard(liked_tag_set, row["tags_norm"])
            overlaps.append(ov)
            if ov >= OVERLAP_THRESHOLD:
                rel += 1

        precisions.append(rel / K)
        avg_overlaps.append(float(np.mean(overlaps)))
        diversities.append(len(set(artists)) / K)

    print("\nUsers evaluated:", len(precisions))
    if not precisions:
        print("No users evaluated (too many skipped). Try lowering K or max_per_artist.")
        return

    print(f"Precision@{K} (tag-overlap proxy): {np.mean(precisions):.3f}")
    print(f"Avg tag overlap: {np.mean(avg_overlaps):.3f}")
    print(f"Artist diversity@{K}: {np.mean(diversities):.3f}")


if __name__ == "__main__":
    main()