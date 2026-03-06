# Music Recommender (HuggingFace Embeddings + FAISS + FastAPI + Streamlit)

## What this project demonstrates

- **Recommender basics**: semantic retrieval, ranking, diversity constraints, personalization via user profile embeddings  
- **ML engineering**: offline indexing pipeline, reproducible artifacts (parquet + FAISS index)  
- **API + product demo**: FastAPI backend + Streamlit UI  
- **Evaluation**: reproducible offline proxy metrics

---

## Setup

### 1) Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # mac/linux
# venv\Scripts\activate  # windows
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
```

---

## Dataset

This project uses **Last.fm Dataset 2020**, provided as a SQLite database with two tables:
https://github.com/renesemela/lastfm-dataset-2020

Place the database at:

```
data/raw/lastfm_dataset_2020.db
```

---

## Build pipeline

### Step A — Build the catalog (SQLite → Parquet)
Creates `data/tracks.parquet` with:
- `track_id`, `title`, `artist`, `tags` and a combined `text` field for embeddings.

```bash
python src/build_catalog.py
```

### Step B — Build embeddings + FAISS index
Creates:
- `data/index.faiss` (vector index)
- `data/id_map.json` (FAISS row → track_id mapping)

```bash
python src/build_index.py
```

### Step C — Quick CLI sanity test (optional)
```bash
python src/recommender.py
```

---

## Run the backend (FastAPI)

Start the API:

```bash
uvicorn src.api:app --reload --port 8000
```

Endpoints:
- Health:
  - `GET http://127.0.0.1:8000/health`
- Search (for UI selection chips):
  - `GET http://127.0.0.1:8000/search?q=radiohead&limit=20`
- Recommend:
  - `POST http://127.0.0.1:8000/recommend`
- Metrics:
  - `GET http://127.0.0.1:8000/metrics`

Example request body:

```json
{
  "query": "calm instrumental ambient music for studying",
  "liked_track_ids": ["<track_id_1>", "<track_id_2>"],
  "k": 10
}
```

Notes:
- **Query only** works (`liked_track_ids` empty)
- **Liked only** works (`query` empty) — recommendations based on user profile embedding
- **Query + liked** blends both signals

---

## Run the UI (Streamlit)

Make sure the FastAPI backend is running, then:

```bash
streamlit run ui.py
```

UI flow:
1) Search tracks and add to “Liked songs”  
2) (Optional) Enter a natural-language query  
3) Click Recommend  

---

## Offline evaluation

Runs a reproducible offline evaluation using proxy metrics (because we do not have user relevance labels):
- **Precision@10 (tag-overlap proxy)**
- **Average tag overlap**
- **Artist diversity@10**

```bash
python src/evaluate.py
```

Example output (will vary):
- Precision@10 (tag-overlap proxy): 0.630  
- Avg tag overlap: 0.134  
- Artist diversity@10: 0.860  

---

## Logging and Monitoring

### Structured Logging

The API uses structured JSON logging for observability.  
Every request generates a log entry containing:
- request_id
- HTTP method
- endpoint path
- response status code
- request latency (ms)
- timestamp

Logs are written to:

```
logs/app.log
```

Each log line is JSON, which makes it easy to inspect locally or ship to external logging systems later.

Example log entry:

```json
{
  "request_id": "c0a8c5c0-9b1c-4e6e-a2a2-5e4c9f1f3c44",
  "method": "POST",
  "path": "/recommend",
  "status_code": 200,
  "latency_ms": 128,
  "timestamp": "2026-03-06T06:15:21.201Z"
}
```

### Metrics Endpoint

The API exposes a runtime metrics endpoint:

```text
GET /metrics
```

This endpoint returns basic service metrics including:
- total requests
- total errors
- number of search requests
- number of recommendation requests
- average request latency

Example response:

```json
{
  "requests_total": 42,
  "errors_total": 0,
  "search_requests": 15,
  "recommend_requests": 27,
  "avg_latency_ms": 92.4
}
```

These metrics can be used for local monitoring or integrated with an external monitoring system later.

---

## Limitations / next improvements

- Current recommender is **semantic retrieval + simple personalization**. Next extensions:
  - add collaborative signals (if you add listening histories)
  - add learning-to-rank re-ranker
  - add caching (frequent queries)
  - add richer constraints (explicit, vocals, language) if metadata is available
