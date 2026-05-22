"""Optional Streamlit dashboard: browse ranked candidates per campaign.

Run with:  streamlit run src/matscout/viz/dashboard.py
Requires the `extra` optional dependencies (streamlit, plotly).
"""

from __future__ import annotations

from pathlib import Path


def main() -> None:  # pragma: no cover - interactive
    import pandas as pd
    import streamlit as st

    st.set_page_config(page_title="MatScout-4090", layout="wide")
    st.title("MatScout-4090 — candidate explorer")

    out = Path("outputs")
    ranked_path = out / "candidates_ranked.parquet"
    if not ranked_path.exists():
        st.warning("No outputs/candidates_ranked.parquet yet. Run `matscout run-campaign`.")
        return

    df = pd.read_parquet(ranked_path)
    campaigns = ["(all)"] + sorted({t for tags in df["tags"] for t in tags
                                    if t in {"battery", "semiconductor", "catalyst", "solar", "co2_capture"}})
    choice = st.sidebar.selectbox("Campaign", campaigns)
    if choice != "(all)":
        df = df[df["tags"].apply(lambda ts: choice in ts)]

    min_score = st.sidebar.slider("min final_score", 0.0, 1.0, 0.3, 0.05)
    df = df[df["final_score"] >= min_score].sort_values("final_score", ascending=False)

    st.metric("candidates", len(df))
    cols = ["reduced_formula", "final_score", "novelty_score", "ml_e_above_hull",
            "predicted_bandgap_ev", "uncertainty_score", "source"]
    st.dataframe(df[[c for c in cols if c in df.columns]].head(300), use_container_width=True)

    try:
        import plotly.express as px

        fig = px.scatter(df.head(500), x="ml_e_above_hull", y="novelty_score",
                         size="final_score", color="predicted_bandgap_ev",
                         hover_name="reduced_formula")
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        st.info("Install plotly for interactive scatter (pip install -e '.[extra]').")


if __name__ == "__main__":
    main()
