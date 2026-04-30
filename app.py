"""Streamlit UI for ToS Semantic Similarity & Unfair Clause Detection."""
import os

import httpx
import pandas as pd
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

CATEGORY_LABELS = {
    "a": "Arbitration",
    "ch": "Unilateral Change",
    "cr": "Content Removal",
    "j": "Jurisdiction",
    "law": "Choice of Law",
    "ltd": "Limitation of Liability",
    "ter": "Unilateral Termination",
    "use": "Contract by Using",
}

st.set_page_config(
    page_title="ToS Clause Analyzer",
    page_icon="⚖️",
    layout="wide",
)

# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚖️ ToS Analyzer")
    st.markdown("Semantic similarity search and unfair clause detection for Terms of Service.")
    st.divider()

    api_url = st.text_input("API URL", value=API_URL)
    top_k = st.slider("Similar sentences (top-k)", min_value=1, max_value=30, value=10)
    st.divider()

    # Health check
    try:
        resp = httpx.get(f"{api_url}/health", timeout=3.0)
        info = resp.json()
        st.success(f"API online")
        st.caption(f"Embedding: `{info.get('embedding_method', '?')}`")
        st.caption(f"Corpus: {info.get('corpus_size', '?'):,} sentences")
    except Exception:
        st.error("API unreachable")

# ── Main ───────────────────────────────────────────────────────────────────────

st.title("ToS Clause Analyzer")
st.markdown(
    "Paste a Terms of Service sentence below to find similar clauses in the corpus "
    "and detect whether it may be **potentially unfair** under EU consumer law."
)

sentence = st.text_area(
    "Sentence to analyze",
    placeholder="e.g. We may terminate your account at any time for any reason.",
    height=120,
)

analyze_btn = st.button("Analyze", type="primary", use_container_width=True)

if analyze_btn:
    if not sentence.strip():
        st.warning("Please enter a sentence.")
    else:
        with st.spinner("Analyzing..."):
            try:
                resp = httpx.post(
                    f"{api_url}/analyze",
                    json={"sentence": sentence.strip(), "top_k": top_k},
                    timeout=30.0,
                )
                resp.raise_for_status()
                data = resp.json()
            except httpx.RequestError:
                st.error(f"Cannot reach API at {api_url}. Is the service running?")
                st.stop()
            except httpx.HTTPStatusError as e:
                st.error(f"API error {e.response.status_code}: {e.response.text}")
                st.stop()

        clf = data["classification"]
        similar = data["similar"]

        # ── Classification result ──────────────────────────────────────────────
        st.divider()
        col1, col2 = st.columns([1, 2])

        with col1:
            if clf["is_unfair"]:
                st.error("🔴 Potentially Unfair")
            else:
                st.success("🟢 No Unfair Clause Detected")

        with col2:
            if clf["categories"]:
                st.markdown("**Flagged categories:**")
                cats = [CATEGORY_LABELS.get(c, c) for c in clf["categories"]]
                st.markdown(" · ".join(f"`{c}`" for c in cats))
            else:
                st.markdown("No unfairness categories triggered.")

        # ── Category details ───────────────────────────────────────────────────
        with st.expander("All category details"):
            details = clf.get("details", {})
            rows = [
                {"Category": CATEGORY_LABELS.get(k, k), "Key": k, "Flagged": "✅" if v else "—"}
                for k, v in details.items()
            ]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # ── Similar sentences ──────────────────────────────────────────────────
        st.divider()
        st.subheader(f"Top {len(similar)} similar sentences from corpus")

        if similar:
            rows = []
            for r in similar:
                rows.append({
                    "Score": f"{r['score']:.3f}",
                    "Fair/Unfair": "🔴" if r["is_unfair"] else "🟢",
                    "Document": r["doc"],
                    "Sentence": r["sentence"],
                })
            st.dataframe(
                pd.DataFrame(rows),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Score": st.column_config.TextColumn(width="small"),
                    "Fair/Unfair": st.column_config.TextColumn(width="small"),
                    "Document": st.column_config.TextColumn(width="medium"),
                    "Sentence": st.column_config.TextColumn(width="large"),
                },
            )
        else:
            st.info("No similar sentences found.")
