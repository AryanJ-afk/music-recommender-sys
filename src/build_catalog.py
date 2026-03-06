# src/build_catalog.py
import sqlite3
from pathlib import Path
import pandas as pd

DB_PATH = Path("data/raw/lastfm_dataset_2020.db")
OUT_PATH = Path("data/tracks.parquet")


def table_columns(conn, table: str) -> list[str]:
    rows = conn.execute(f"PRAGMA table_info({table});").fetchall()
    return [r[1] for r in rows]  


def main():
    if not DB_PATH.exists():
        raise FileNotFoundError(f"DB not found: {DB_PATH.resolve()}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(DB_PATH))

    # sanity: tables exist
    tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()]
    if "metadata" not in tables or "tags" not in tables:
        raise RuntimeError(f"Expected tables metadata,tags. Found: {tables}")

    meta_cols = table_columns(conn, "metadata")
    tag_cols = table_columns(conn, "tags")
    print("metadata columns:", meta_cols)
    print("tags columns:", tag_cols[:12], f"... ({len(tag_cols)} cols)")

    metadata = pd.read_sql_query("SELECT * FROM metadata;", conn)
    tags = pd.read_sql_query("SELECT * FROM tags;", conn)
    conn.close()

    # Find join key automatically
    candidates = ["id_dataset", "track_id", "id", "dataset_id"]
    join_key = next((k for k in candidates if k in metadata.columns and k in tags.columns), None)
    if join_key is None:
        # fall back: intersection of columns
        common = sorted(set(metadata.columns).intersection(tags.columns))
        raise RuntimeError(
            f"Could not infer join key. Common columns: {common}\n"
            f"metadata cols: {list(metadata.columns)}\n"
            f"tags cols: {list(tags.columns)}"
        )

    df = metadata.merge(tags, on=join_key, how="inner")

    # Identify tag indicator columns = all columns in tags except join key
    tag_indicator_cols = [c for c in tags.columns if c != join_key]

    def collect_tags(row):
        active = []
        for c in tag_indicator_cols:
            v = row[c]
            try:
                if int(v) == 1:
                    active.append(c)
            except Exception:
                pass
        return active

    df["tags_list"] = df.apply(collect_tags, axis=1)

    # Guess title/artist columns (dataset-dependent)
    title_col = next((c for c in ["track", "track_name", "title", "name"] if c in df.columns), None)
    artist_col = next((c for c in ["artist", "artist_name"] if c in df.columns), None)

    # Build text field for embeddings
    def build_text(row):
        parts = []
        if title_col and pd.notna(row[title_col]):
            parts.append(f"Track: {row[title_col]}")
        if artist_col and pd.notna(row[artist_col]):
            parts.append(f"Artist: {row[artist_col]}")
        if row["tags_list"]:
            parts.append("Tags: " + ", ".join(row["tags_list"]))
        return " | ".join(parts)

    df["text"] = df.apply(build_text, axis=1)

    out = pd.DataFrame({
        "track_id": df[join_key].astype(str),
        "title": df[title_col].astype(str) if title_col else "",
        "artist": df[artist_col].astype(str) if artist_col else "",
        "tags": df["tags_list"],
        "text": df["text"],
    })

    # cleanup
    out = out[out["text"].str.len() > 0].drop_duplicates(subset=["track_id"]).reset_index(drop=True)

    out.to_parquet(OUT_PATH, index=False)
    print(f"\nSaved catalog: {OUT_PATH}  (rows={len(out)})")
    print(out.head(5).to_string(index=False))


if __name__ == "__main__":
    main()