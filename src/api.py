# src/api.py
import json
import time
import uuid
from pathlib import Path
from typing import List, Optional
import numpy as np
import pandas as pd
import faiss
import torch
from fastapi import FastAPI, Query, Request
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from datetime import datetime


TRACKS_PATH = Path("data/tracks.parquet")
INDEX_PATH = Path("data/index.faiss")
IDMAP_PATH = Path("data/id_map.json")
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "app.log"

app = FastAPI(title="Music Recommender API")


def pick_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ---------- in-memory metrics ----------
METRICS = {
    "requests_total": 0,
    "errors_total": 0,
    "search_requests": 0,
    "recommend_requests": 0,
    "latency_ms_sum": 0,
    "latency_ms_count": 0,
}

# ---------- logs helper ----------
def write_log(log_line: dict):
    log_line["timestamp"] = datetime.utcnow().isoformat() + "Z"
    line = json.dumps(log_line)
    print(line)  # still show in terminal
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

# ---------- load once at startup ----------
df = pd.read_parquet(TRACKS_PATH)
index = faiss.read_index(str(INDEX_PATH))
id_map = json.loads(IDMAP_PATH.read_text())
device = pick_device()
embedder = SentenceTransformer(EMBED_MODEL_NAME, device=device)

df["track_id"] = df["track_id"].astype(str)
row_by_id = {tid: i for i, tid in enumerate(df["track_id"].tolist())}
faiss_row_by_track_id = {tid: i for i, tid in enumerate(id_map)}


def get_track_vector_by_faiss_row(i: int) -> np.ndarray:
    v = np.zeros((index.d,), dtype=np.float32)
    index.reconstruct(i, v)
    return v


# ---------- request/response models ----------
class RecommendRequest(BaseModel):
    query: Optional[str] = ""
    liked_track_ids: Optional[List[str]] = []
    k: int = 10


class TrackOut(BaseModel):
    track_id: str
    title: str
    artist: str
    tags: list
    score: float


class RecommendResponse(BaseModel):
    query: str
    liked_track_ids: List[str]
    recommendations: List[TrackOut]


# ---------- logging middleware ----------
@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
    t0 = time.time()

    METRICS["requests_total"] += 1
    status_code = 500

    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    except Exception:
        METRICS["errors_total"] += 1
        raise
    finally:
        latency_ms = int((time.time() - t0) * 1000)
        METRICS["latency_ms_sum"] += latency_ms
        METRICS["latency_ms_count"] += 1

        log_line = {
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status_code": status_code,
            "latency_ms": latency_ms,
        }
        write_log(log_line)


# ---------- endpoints ----------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    avg_latency = (
        METRICS["latency_ms_sum"] / METRICS["latency_ms_count"]
        if METRICS["latency_ms_count"] > 0
        else 0
    )

    return {
        "requests_total": METRICS["requests_total"],
        "errors_total": METRICS["errors_total"],
        "search_requests": METRICS["search_requests"],
        "recommend_requests": METRICS["recommend_requests"],
        "avg_latency_ms": round(avg_latency, 2),
    }


@app.get("/search", response_model=List[TrackOut])
def search(q: str = Query(..., min_length=1), limit: int = 20):
    METRICS["search_requests"] += 1

    ql = q.lower().strip()
    mask = df["title"].str.lower().str.contains(ql, na=False) | df["artist"].str.lower().str.contains(ql, na=False)
    hits = df[mask].head(limit)

    out = []
    for _, row in hits.iterrows():
        out.append(
            TrackOut(
                track_id=row["track_id"],
                title=row["title"],
                artist=row["artist"],
                tags=row["tags"],
                score=0.0,
            )
        )
    return out


def build_user_profile_vector(liked_track_ids: List[str]) -> Optional[np.ndarray]:
    vecs = []
    for tid in liked_track_ids:
        tid = str(tid)
        if tid not in faiss_row_by_track_id:
            continue
        faiss_i = faiss_row_by_track_id[tid]
        vecs.append(get_track_vector_by_faiss_row(faiss_i))

    if not vecs:
        return None

    prof = np.mean(np.vstack(vecs), axis=0).astype(np.float32)
    norm = np.linalg.norm(prof) + 1e-12
    return (prof / norm).astype(np.float32)


@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    METRICS["recommend_requests"] += 1

    query = (req.query or "").strip()
    k = int(req.k)
    liked = [str(x) for x in (req.liked_track_ids or [])]
    liked_set = set(liked)

    prof = build_user_profile_vector(liked)

    if query:
        qvec = embedder.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype(np.float32)[0]

        if prof is not None:
            qvec = (0.75 * qvec + 0.25 * prof).astype(np.float32)
            qvec = qvec / (np.linalg.norm(qvec) + 1e-12)
    else:
        if prof is None:
            return RecommendResponse(query=query, liked_track_ids=liked, recommendations=[])
        qvec = prof

    retrieve_n = max(200, k * 30)
    scores, idxs = index.search(qvec.reshape(1, -1), retrieve_n)
    idxs = idxs[0].tolist()
    scores = scores[0].tolist()

    recs = []
    artist_count = {}
    max_per_artist = 2

    for faiss_i, score in zip(idxs, scores):
        tid = id_map[faiss_i]

        if tid in liked_set:
            continue

        row = df.iloc[row_by_id[tid]]
        artist = str(row["artist"])

        artist_count[artist] = artist_count.get(artist, 0)
        if artist_count[artist] >= max_per_artist:
            continue

        recs.append(
            TrackOut(
                track_id=tid,
                title=str(row["title"]),
                artist=artist,
                tags=row["tags"],
                score=float(score),
            )
        )
        artist_count[artist] += 1

        if len(recs) >= k:
            break

    log_line = {
        "event": "recommendation_generated",
        "query_length": len(query),
        "liked_tracks_count": len(liked),
        "k": k,
        "results_returned": len(recs),
    }
    write_log(log_line)

    return RecommendResponse(
        query=query,
        liked_track_ids=liked,
        recommendations=recs
    )