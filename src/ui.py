# ui.py
import requests
import streamlit as st

API_BASE = "http://127.0.0.1:8000"


st.set_page_config(page_title="Music Recommender", layout="wide")
st.title("🎵 Music Recommender (Embeddings + FAISS)")

st.caption("1) Search songs and select what you like  •  2) Enter a text prompt  •  3) Get recommendations")


# ---------- session state ----------
if "liked" not in st.session_state:
    st.session_state.liked = []  # list of dicts: {track_id,title,artist}

if "search_results" not in st.session_state:
    st.session_state.search_results = []


# ---------- layout ----------
left, right = st.columns([1, 1])

with left:
    st.subheader("1) Pick songs you like")

    q = st.text_input("Search by title or artist", placeholder="e.g. Radiohead, Wonderwall, Beatles")
    colA, colB = st.columns([1, 1])

    if colA.button("Search", use_container_width=True):
        if q.strip():
            r = requests.get(f"{API_BASE}/search", params={"q": q.strip(), "limit": 25}, timeout=30)
            r.raise_for_status()
            st.session_state.search_results = r.json()
        else:
            st.warning("Type something to search.")

    if colB.button("Clear liked", use_container_width=True):
        st.session_state.liked = []

    # Show search results with “Add” buttons
    if st.session_state.search_results:
        st.write("Search results:")
        for item in st.session_state.search_results:
            label = f"{item['title']} — {item['artist']}"
            c1, c2 = st.columns([5, 1])
            c1.write(label)
            if c2.button("Add", key=f"add_{item['track_id']}"):
                # avoid duplicates
                if not any(x["track_id"] == item["track_id"] for x in st.session_state.liked):
                    st.session_state.liked.append({
                        "track_id": item["track_id"],
                        "title": item["title"],
                        "artist": item["artist"]
                    })

    # Liked songs “chips” via multiselect
    st.divider()
    liked_labels = [f"{x['title']} — {x['artist']}" for x in st.session_state.liked]
    liked_map = {f"{x['title']} — {x['artist']}": x["track_id"] for x in st.session_state.liked}

    selected = st.multiselect(
        "Liked songs (selected)",
        options=liked_labels,
        default=liked_labels,
        help="These are the songs the recommender will use for personalization."
    )

    # Keep only selected liked tracks
    selected_ids = set(liked_map[s] for s in selected)
    st.session_state.liked = [x for x in st.session_state.liked if x["track_id"] in selected_ids]

with right:
    st.subheader("2) Describe what you want")

    query = st.text_area(
        "Your request",
        placeholder="e.g. calm instrumental ambient music for studying",
        height=120
    )

    k = st.slider("How many recommendations?", min_value=5, max_value=25, value=10, step=1)

    if st.button("Recommend", type="primary", use_container_width=True):
        liked_ids = [x["track_id"] for x in st.session_state.liked]

        # Allow either query OR liked songs (or both)
        if (not query.strip()) and (len(liked_ids) == 0):
            st.warning("Enter a request OR select some liked songs.")
        else:
            payload = {"query": query.strip(), "liked_track_ids": liked_ids, "k": int(k)}

            with st.spinner("Getting recommendations..."):
                r = requests.post(f"{API_BASE}/recommend", json=payload, timeout=60)
                r.raise_for_status()
                data = r.json()

            st.success("Done!")
            st.write(f"Liked tracks used: **{len(liked_ids)}**")

            st.subheader("Recommendations")
            for i, rec in enumerate(data["recommendations"], start=1):
                st.markdown(
                    f"**{i}. {rec['title']} — {rec['artist']}**  \n"
                    f"Score: `{rec['score']:.3f}`  \n"
                    f"Tags: {', '.join(rec['tags'][:10])}"
                )
                st.divider()

st.caption("Backend: FastAPI • Retrieval: HuggingFace embeddings + FAISS • Personalization: query/profile blending")